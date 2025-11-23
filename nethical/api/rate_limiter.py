"""
In-process rate limiting for Nethical API.

Implements sliding window rate limiting with per-identity tracking.
Designed for single-instance deployments with graceful degradation.
"""

import time
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    
    requests_per_second: float = 5.0  # Burst rate
    requests_per_minute: int = 100  # Sustained rate
    cleanup_interval: int = 300  # Clean old entries every 5 minutes


class TokenBucketLimiter:
    """
    Token bucket rate limiter with sliding window.
    
    Enforces both burst (req/sec) and sustained (req/min) limits per identity.
    Thread-safe for asyncio usage.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter with configuration."""
        self.config = config or RateLimitConfig()
        
        # Per-identity buckets: identity -> deque of timestamps
        # Note: We don't set maxlen here as we manually manage the sliding window
        self._buckets: Dict[str, deque] = defaultdict(lambda: deque())
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # Last cleanup time
        self._last_cleanup = time.time()
        
        logger.info(
            f"Rate limiter initialized: {self.config.requests_per_second} req/s burst, "
            f"{self.config.requests_per_minute} req/min sustained"
        )
    
    async def is_allowed(self, identity: str) -> Tuple[bool, Optional[float], Dict[str, int]]:
        """
        Check if request is allowed for identity.
        
        Args:
            identity: User identity (API key or IP address)
            
        Returns:
            Tuple of (allowed, retry_after_seconds, rate_limit_info)
            rate_limit_info contains:
                - limit: requests_per_minute
                - remaining: requests remaining in window
                - reset: unix timestamp when window resets
        """
        async with self._locks[identity]:
            now = time.time()
            bucket = self._buckets[identity]
            
            # Remove timestamps outside 1-minute window
            cutoff_minute = now - 60
            while bucket and bucket[0] < cutoff_minute:
                bucket.popleft()
            
            # Check sustained rate (per-minute limit)
            minute_count = len(bucket)
            if minute_count >= self.config.requests_per_minute:
                # Calculate retry-after based on oldest request in window
                oldest = bucket[0]
                retry_after = max(0.0, 60 - (now - oldest))
                
                info = {
                    "limit": self.config.requests_per_minute,
                    "remaining": 0,
                    "reset": int(oldest + 60)
                }
                
                logger.warning(
                    f"Rate limit exceeded for {identity}: {minute_count}/{self.config.requests_per_minute} req/min"
                )
                return False, retry_after, info
            
            # Check burst rate (per-second limit)
            cutoff_second = now - 1
            second_count = sum(1 for ts in bucket if ts >= cutoff_second)
            
            if second_count >= self.config.requests_per_second:
                # Calculate retry-after based on oldest request in 1-second window
                burst_requests = [ts for ts in bucket if ts >= cutoff_second]
                oldest_burst = burst_requests[0] if burst_requests else now
                retry_after = max(0.0, 1 - (now - oldest_burst))
                
                info = {
                    "limit": self.config.requests_per_minute,
                    "remaining": max(0, self.config.requests_per_minute - minute_count),
                    "reset": int(bucket[0] + 60) if bucket else int(now + 60)
                }
                
                logger.warning(
                    f"Burst rate limit exceeded for {identity}: {second_count}/{self.config.requests_per_second} req/s"
                )
                return False, retry_after, info
            
            # Allow request and record timestamp
            bucket.append(now)
            
            info = {
                "limit": self.config.requests_per_minute,
                "remaining": max(0, self.config.requests_per_minute - len(bucket)),
                "reset": int(bucket[0] + 60) if bucket else int(now + 60)
            }
            
            # Periodic cleanup of old identities
            if now - self._last_cleanup > self.config.cleanup_interval:
                await self._cleanup_old_entries(now)
            
            return True, None, info
    
    async def _cleanup_old_entries(self, now: float) -> None:
        """Remove inactive identities to prevent memory bloat."""
        cutoff = now - 600  # Remove identities inactive for 10+ minutes
        to_remove = []
        
        for identity, bucket in self._buckets.items():
            if not bucket or bucket[-1] < cutoff:
                to_remove.append(identity)
        
        for identity in to_remove:
            del self._buckets[identity]
            if identity in self._locks:
                del self._locks[identity]
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} inactive rate limit entries")
        
        self._last_cleanup = now
    
    def get_stats(self) -> Dict[str, int]:
        """Get current rate limiter statistics."""
        return {
            "active_identities": len(self._buckets),
            "total_locks": len(self._locks)
        }
