"""
Cache Hierarchy - Unified Multi-Level Cache Interface

Provides a unified interface to the L1/L2/L3 cache hierarchy.
Target: 95%+ cumulative hit rate
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class HierarchyConfig:
    """
    Cache hierarchy configuration.

    Attributes:
        enable_l1: Enable L1 memory cache
        enable_l2: Enable L2 Redis cache
        enable_l3: Enable L3 global cache
        write_through: Write to all levels on set
        read_through: Populate lower levels on L3 hit
        l1_ttl_seconds: L1 TTL
        l2_ttl_seconds: L2 TTL
        l3_ttl_seconds: L3 TTL
    """

    enable_l1: bool = True
    enable_l2: bool = False  # Disabled by default (requires Redis)
    enable_l3: bool = True  # Memory fallback
    write_through: bool = True
    read_through: bool = True
    l1_ttl_seconds: int = 30
    l2_ttl_seconds: int = 300
    l3_ttl_seconds: int = 900


class CacheHierarchy:
    """
    Unified cache hierarchy interface.

    Provides seamless access to three cache levels:
    - L1: In-memory (fastest, smallest)
    - L2: Regional Redis (fast, medium)
    - L3: Global distributed (slower, largest)

    Features:
    - Read-through caching
    - Write-through/write-back
    - Automatic tier promotion
    - Cumulative hit rate tracking

    Target: 95%+ cumulative hit rate
    """

    def __init__(
        self,
        config: Optional[HierarchyConfig] = None,
        l1_cache: Optional["L1MemoryCache"] = None,
        l2_cache: Optional["L2RedisCache"] = None,
        l3_cache: Optional["L3GlobalCache"] = None,
    ):
        """
        Initialize CacheHierarchy.

        Args:
            config: Hierarchy configuration
            l1_cache: L1 cache instance
            l2_cache: L2 cache instance
            l3_cache: L3 cache instance
        """
        self.config = config or HierarchyConfig()

        # Import here to avoid circular imports
        from .l1_memory import L1MemoryCache
        from .l2_redis import L2RedisCache
        from .l3_global import L3GlobalCache

        # Initialize caches
        self.l1 = l1_cache if l1_cache else L1MemoryCache() if self.config.enable_l1 else None
        self.l2 = l2_cache if l2_cache else L2RedisCache() if self.config.enable_l2 else None
        self.l3 = l3_cache if l3_cache else L3GlobalCache() if self.config.enable_l3 else None

        # Metrics
        self._l1_hits = 0
        self._l2_hits = 0
        self._l3_hits = 0
        self._misses = 0

        logger.info(
            f"CacheHierarchy initialized: L1={self.config.enable_l1}, "
            f"L2={self.config.enable_l2}, L3={self.config.enable_l3}"
        )

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache hierarchy.

        Checks L1 -> L2 -> L3 in order.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        # Try L1
        if self.l1:
            value = self.l1.get(key)
            if value is not None:
                self._l1_hits += 1
                return value

        # Try L2
        if self.l2 and self.l2.is_connected:
            value = self.l2.get(key)
            if value is not None:
                self._l2_hits += 1
                # Promote to L1
                if self.config.read_through and self.l1:
                    self.l1.set(key, value, self.config.l1_ttl_seconds)
                return value

        # Try L3
        if self.l3:
            value = self.l3.get(key)
            if value is not None:
                self._l3_hits += 1
                # Promote to lower levels
                if self.config.read_through:
                    if self.l1:
                        self.l1.set(key, value, self.config.l1_ttl_seconds)
                    if self.l2 and self.l2.is_connected:
                        self.l2.set(key, value, self.config.l2_ttl_seconds)
                return value

        self._misses += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: Optional[str] = None,
    ):
        """
        Set value in cache hierarchy.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Override TTL (uses level defaults if None)
            level: Specific level to write to (l1, l2, l3, all)
        """
        if level == "l1" or level is None or level == "all":
            if self.l1:
                l1_ttl = ttl if ttl else self.config.l1_ttl_seconds
                self.l1.set(key, value, l1_ttl)

        if self.config.write_through or level in ("l2", "all"):
            if self.l2 and self.l2.is_connected:
                l2_ttl = ttl if ttl else self.config.l2_ttl_seconds
                self.l2.set(key, value, l2_ttl)

            if self.l3:
                l3_ttl = ttl if ttl else self.config.l3_ttl_seconds
                self.l3.set(key, value, l3_ttl)

    def delete(self, key: str):
        """
        Delete value from all cache levels.

        Args:
            key: Cache key
        """
        if self.l1:
            self.l1.delete(key)
        if self.l2 and self.l2.is_connected:
            self.l2.delete(key)
        if self.l3:
            self.l3.delete(key)

    def invalidate_pattern(self, pattern: str):
        """
        Invalidate keys matching pattern in all levels.

        Args:
            pattern: Pattern to match
        """
        if self.l1:
            self.l1.invalidate_pattern(pattern)
        if self.l2 and self.l2.is_connected:
            self.l2.invalidate_pattern(pattern)
        if self.l3:
            self.l3.invalidate_pattern(pattern)

    def clear(self):
        """Clear all cache levels."""
        if self.l1:
            self.l1.clear()
        if self.l3:
            self.l3.clear()

    def get_or_set(
        self,
        key: str,
        factory: callable,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get from cache or compute and cache.

        Args:
            key: Cache key
            factory: Function to compute value
            ttl: TTL in seconds

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl)
        return value

    def get_metrics(self) -> Dict[str, Any]:
        """Get hierarchy metrics."""
        total = self._l1_hits + self._l2_hits + self._l3_hits + self._misses

        return {
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "l3_hits": self._l3_hits,
            "misses": self._misses,
            "total_requests": total,
            "l1_hit_rate": self._l1_hits / total if total > 0 else 0.0,
            "l2_hit_rate": self._l2_hits / total if total > 0 else 0.0,
            "l3_hit_rate": self._l3_hits / total if total > 0 else 0.0,
            "cumulative_hit_rate": (
                (self._l1_hits + self._l2_hits + self._l3_hits) / total
                if total > 0
                else 0.0
            ),
            "l1_metrics": self.l1.get_metrics() if self.l1 else None,
            "l2_metrics": self.l2.get_metrics() if self.l2 else None,
            "l3_metrics": self.l3.get_metrics() if self.l3 else None,
        }


# Type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .l1_memory import L1MemoryCache
    from .l2_redis import L2RedisCache
    from .l3_global import L3GlobalCache
