"""
In-process rate limiting for Nethical API.

Implements sliding window + token bucket hybrid with per-identity tracking.
Optimized to minimize per-request overhead while remaining simple.

Environment (consumed indirectly via RateLimitConfig in api.py):
    NETHICAL_RATE_BURST       float requests/sec (burst capacity)
    NETHICAL_RATE_SUSTAINED   int   requests/minute (sustained)

Returned rate info fields:
    limit          sustained (per-minute) limit
    burst_limit    burst (per-second) limit
    remaining      remaining requests in current minute window
    reset          UNIX timestamp when current minute window resets

Performance considerations:
    - Uses a deque of timestamps per identity (simple & adequate for modest rates)
    - For very high identities / traffic, consider a fixed-size ring buffer or token bucket struct
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
    """Rate limit configuration parameters."""

    requests_per_second: float = 5.0  # Burst rate
    requests_per_minute: int = 100  # Sustained rate
    cleanup_interval: int = 300  # Seconds between GC for inactive identities
    idle_eviction_seconds: int = 600  # Remove identities idle for > this


class TokenBucketLimiter:
    """
    Hybrid token bucket / sliding window limiter.

    For each identity we keep a deque of request timestamps (seconds resolution).
    Checks:
        1. Trim timestamps older than minute window (60s)
        2. Enforce sustained minute limit
        3. Count requests in last 1 second for burst limit
    """

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, deque] = defaultdict(deque)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_cleanup = time.time()
        logger.info(
            "Rate limiter initialized: %.2f req/s burst, %d req/min sustained",
            self.config.requests_per_second,
            self.config.requests_per_minute,
        )

    async def is_allowed(
        self, identity: str
    ) -> Tuple[bool, Optional[float], Dict[str, int]]:
        """
        Determine if the identity is allowed to make a request now.

        Returns:
            (allowed, retry_after_seconds, rate_info_dict)
        """
        async with self._locks[identity]:
            now = time.time()
            bucket = self._buckets[identity]

            # Trim entries outside 60s window
            cutoff_minute = now - 60
            while bucket and bucket[0] < cutoff_minute:
                bucket.popleft()

            # Sustained limit check
            minute_count = len(bucket)
            if minute_count >= self.config.requests_per_minute:
                oldest = bucket[0]
                retry_after = max(0.0, 60 - (now - oldest))
                info = {
                    "limit": self.config.requests_per_minute,
                    "burst_limit": int(self.config.requests_per_second),
                    "remaining": 0,
                    "reset": int(oldest + 60),
                }
                logger.warning(
                    "Sustained rate exceeded identity=%s %d/%d req/min",
                    identity,
                    minute_count,
                    self.config.requests_per_minute,
                )
                return False, retry_after, info

            # Burst limit check (count requests in last second)
            cutoff_second = now - 1
            second_count = 0
            # Iterate from right (newest) until older than cutoff_second for efficiency
            for ts in reversed(bucket):
                if ts >= cutoff_second:
                    second_count += 1
                else:
                    break

            if second_count >= self.config.requests_per_second:
                # Determine earliest ts in the last second window for retry-after
                # Gather burst timestamps once (small subset)
                burst_ts = [ts for ts in bucket if ts >= cutoff_second]
                oldest_burst = burst_ts[0] if burst_ts else now
                retry_after = max(0.0, 1 - (now - oldest_burst))
                info = {
                    "limit": self.config.requests_per_minute,
                    "burst_limit": int(self.config.requests_per_second),
                    "remaining": max(0, self.config.requests_per_minute - minute_count),
                    "reset": int(bucket[0] + 60) if bucket else int(now + 60),
                }
                logger.warning(
                    "Burst rate exceeded identity=%s %d/%.2f req/s",
                    identity,
                    second_count,
                    self.config.requests_per_second,
                )
                return False, retry_after, info

            # Allow request
            bucket.append(now)
            info = {
                "limit": self.config.requests_per_minute,
                "burst_limit": int(self.config.requests_per_second),
                "remaining": max(0, self.config.requests_per_minute - len(bucket)),
                "reset": int(bucket[0] + 60) if bucket else int(now + 60),
            }

            # Periodic cleanup
            if now - self._last_cleanup > self.config.cleanup_interval:
                await self._cleanup(now)

            return True, None, info

    async def _cleanup(self, now: float) -> None:
        """Remove identities idle beyond idle_eviction_seconds."""
        cutoff = now - self.config.idle_eviction_seconds
        to_remove = []
        for identity, bucket in self._buckets.items():
            if not bucket or bucket[-1] < cutoff:
                to_remove.append(identity)

        for identity in to_remove:
            self._buckets.pop(identity, None)
            self._locks.pop(identity, None)

        if to_remove:
            logger.debug(
                "Rate limiter cleanup removed %d inactive identities", len(to_remove)
            )
        self._last_cleanup = now

    def get_stats(self) -> Dict[str, int]:
        """Return current limiter stats."""
        return {
            "active_identities": len(self._buckets),
            "total_locks": len(self._locks),
        }
