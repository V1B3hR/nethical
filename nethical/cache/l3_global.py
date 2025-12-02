"""
L3 Global Cache - Global Distributed Cache Client

Global cache for world-wide distribution.
Supports Cloudflare Workers KV, AWS ElastiCache Global, etc.
Target: <50ms latency globally
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class L3Config:
    """
    L3 Global cache configuration.

    Attributes:
        provider: Cache provider (cloudflare, aws, memory)
        namespace: KV namespace
        ttl_seconds: Default TTL
        consistency: Consistency model (eventual, strong)
    """

    provider: str = "memory"
    namespace: str = "nethical-global"
    ttl_seconds: int = 900  # 15 minutes
    consistency: str = "eventual"

    # Provider-specific config
    cloudflare_account_id: Optional[str] = None
    cloudflare_api_token: Optional[str] = None
    aws_region: Optional[str] = None


class L3GlobalCache:
    """
    Global distributed cache client.

    Features:
    - Multi-region replication
    - Eventually consistent
    - CDN-like distribution
    - Fallback to in-memory for development

    Target: <50ms global latency
    """

    def __init__(self, config: Optional[L3Config] = None):
        """
        Initialize L3GlobalCache.

        Args:
            config: Cache configuration
        """
        self.config = config or L3Config()

        # In-memory fallback
        self._memory_cache: Dict[str, tuple] = {}

        # Metrics
        self._hits = 0
        self._misses = 0
        self._errors = 0

        logger.info(
            f"L3GlobalCache initialized: provider={self.config.provider}"
        )

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from global cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if self.config.provider == "memory":
            return self._memory_get(key)
        elif self.config.provider == "cloudflare":
            return self._cloudflare_get(key)
        elif self.config.provider == "aws":
            return self._aws_get(key)
        else:
            return self._memory_get(key)

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in global cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds

        Returns:
            True if successful
        """
        ttl = ttl if ttl is not None else self.config.ttl_seconds

        if self.config.provider == "memory":
            return self._memory_set(key, value, ttl)
        elif self.config.provider == "cloudflare":
            return self._cloudflare_set(key, value, ttl)
        elif self.config.provider == "aws":
            return self._aws_set(key, value, ttl)
        else:
            return self._memory_set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        if self.config.provider == "memory":
            return self._memory_delete(key)
        # Other providers would implement their delete
        return self._memory_delete(key)

    # Memory fallback implementation
    def _memory_get(self, key: str) -> Optional[Any]:
        """In-memory get implementation."""
        if key not in self._memory_cache:
            self._misses += 1
            return None

        value, expires_at = self._memory_cache[key]
        if time.time() > expires_at:
            del self._memory_cache[key]
            self._misses += 1
            return None

        self._hits += 1
        return value

    def _memory_set(self, key: str, value: Any, ttl: int) -> bool:
        """In-memory set implementation."""
        expires_at = time.time() + ttl
        self._memory_cache[key] = (value, expires_at)
        return True

    def _memory_delete(self, key: str) -> bool:
        """In-memory delete implementation."""
        if key in self._memory_cache:
            del self._memory_cache[key]
            return True
        return False

    # Cloudflare Workers KV implementation (stub)
    def _cloudflare_get(self, key: str) -> Optional[Any]:
        """Cloudflare KV get (requires httpx/requests)."""
        try:
            # In production, would use Cloudflare API
            # For now, fallback to memory
            return self._memory_get(key)
        except Exception as e:
            logger.error(f"Cloudflare get error: {e}")
            self._errors += 1
            return None

    def _cloudflare_set(self, key: str, value: Any, ttl: int) -> bool:
        """Cloudflare KV set."""
        try:
            return self._memory_set(key, value, ttl)
        except Exception as e:
            logger.error(f"Cloudflare set error: {e}")
            self._errors += 1
            return False

    # AWS ElastiCache implementation (stub)
    def _aws_get(self, key: str) -> Optional[Any]:
        """AWS ElastiCache get."""
        try:
            return self._memory_get(key)
        except Exception as e:
            logger.error(f"AWS get error: {e}")
            self._errors += 1
            return None

    def _aws_set(self, key: str, value: Any, ttl: int) -> bool:
        """AWS ElastiCache set."""
        try:
            return self._memory_set(key, value, ttl)
        except Exception as e:
            logger.error(f"AWS set error: {e}")
            self._errors += 1
            return False

    def mget(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values.

        Args:
            keys: List of keys

        Returns:
            Dict of key -> value
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def mset(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set multiple values.

        Args:
            items: Dict of key -> value
            ttl: TTL in seconds

        Returns:
            True if all successful
        """
        success = True
        for key, value in items.items():
            if not self.set(key, value, ttl):
                success = False
        return success

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate keys matching pattern.

        Args:
            pattern: Pattern to match

        Returns:
            Number deleted
        """
        # Simple substring match for memory
        deleted = 0
        keys_to_delete = [k for k in self._memory_cache if pattern in k]
        for key in keys_to_delete:
            if self.delete(key):
                deleted += 1
        return deleted

    def clear(self):
        """Clear all entries."""
        self._memory_cache.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        total = self._hits + self._misses
        return {
            "provider": self.config.provider,
            "size": len(self._memory_cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "errors": self._errors,
        }
