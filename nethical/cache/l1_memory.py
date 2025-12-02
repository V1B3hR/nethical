"""
L1 Memory Cache - Ultra-Fast In-Memory LRU Cache

Process-local cache with no serialization overhead.
Target: <0.05ms get/set operations
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class L1CacheEntry:
    """
    L1 cache entry.

    Attributes:
        value: Cached value (direct reference)
        created_at: Entry creation time
        expires_at: Expiration timestamp
        hits: Number of cache hits
    """

    value: Any
    created_at: float
    expires_at: float
    hits: int = 0


class L1MemoryCache:
    """
    Ultra-fast in-memory LRU cache.

    Features:
    - No serialization (direct object references)
    - LRU eviction
    - TTL-based expiration
    - Thread-safe

    Target: <0.05ms get/set operations
    """

    def __init__(
        self,
        max_size_mb: int = 256,
        max_entries: int = 100000,
        ttl_seconds: int = 30,
    ):
        """
        Initialize L1MemoryCache.

        Args:
            max_size_mb: Maximum cache size in MB (approximate)
            max_entries: Maximum number of entries
            ttl_seconds: Default TTL in seconds
        """
        self.max_size_mb = max_size_mb
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

        # LRU cache storage
        self._cache: OrderedDict[str, L1CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0

        logger.info(
            f"L1MemoryCache initialized: max_entries={max_entries}, "
            f"ttl={ttl_seconds}s"
        )

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Target: <0.05ms

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._expired += 1
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._hits += 1

            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ):
        """
        Set value in cache.

        Target: <0.05ms

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (optional, uses default)
        """
        with self._lock:
            # Evict if needed
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)
                self._evictions += 1

            ttl = ttl if ttl is not None else self.ttl_seconds
            now = time.time()

            entry = L1CacheEntry(
                value=value,
                created_at=now,
                expires_at=now + ttl,
            )

            self._cache[key] = entry
            self._cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    def invalidate_pattern(self, pattern: str):
        """
        Invalidate entries matching pattern.

        Args:
            pattern: Pattern to match (substring)
        """
        with self._lock:
            keys_to_delete = [k for k in self._cache if pattern in k]
            for key in keys_to_delete:
                del self._cache[key]
                self._evictions += 1

    def prune_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, v in self._cache.items() if now > v.expires_at
            ]
            for key in expired_keys:
                del self._cache[key]
                self._expired += 1

            return len(expired_keys)

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
            factory: Function to compute value if not cached
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

    def size(self) -> int:
        """Return number of entries."""
        return len(self._cache)

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_entries": self.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "evictions": self._evictions,
                "expired": self._expired,
            }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            if time.time() > self._cache[key].expires_at:
                return False
            return True
