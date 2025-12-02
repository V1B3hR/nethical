"""
Policy Cache - In-Memory Policy Cache with LRU Eviction

Ultra-fast policy caching for edge deployment.
Target: <0.1ms cache lookup
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CachedPolicy:
    """
    Cached policy entry.

    Attributes:
        policy_id: Unique policy identifier
        rules: Policy rules
        risk_weights: Risk weights for scoring
        blocked_patterns: Patterns that should be blocked
        restricted_patterns: Patterns that should be restricted
        created_at: Timestamp when cached
        expires_at: Timestamp when cache entry expires
        version: Policy version
        metadata: Additional metadata
    """

    policy_id: str
    rules: List[Dict[str, Any]]
    risk_weights: Dict[str, float] = field(default_factory=dict)
    blocked_patterns: List[str] = field(default_factory=list)
    restricted_patterns: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if policy cache entry has expired."""
        if self.expires_at == 0.0:
            return False  # Never expires
        return time.time() > self.expires_at


class PolicyCache:
    """
    In-memory LRU policy cache for edge deployment.

    Features:
    - Thread-safe access
    - LRU eviction
    - TTL-based expiration
    - Memory limit enforcement

    Target: <0.1ms get/set operations
    """

    def __init__(
        self,
        max_size_mb: int = 256,
        ttl_seconds: int = 30,
        max_entries: int = 10000,
    ):
        """
        Initialize PolicyCache.

        Args:
            max_size_mb: Maximum cache size in MB
            ttl_seconds: Time-to-live for cache entries
            max_entries: Maximum number of entries
        """
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

        # Thread-safe LRU cache
        self._cache: OrderedDict[str, CachedPolicy] = OrderedDict()
        self._lock = threading.RLock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.info(
            f"PolicyCache initialized: max_size={max_size_mb}MB, "
            f"ttl={ttl_seconds}s, max_entries={max_entries}"
        )

    def get(self, key: str) -> Optional[CachedPolicy]:
        """
        Get policy from cache.

        Target: <0.05ms

        Args:
            key: Cache key

        Returns:
            CachedPolicy if found and not expired, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            policy = self._cache[key]

            # Check expiration
            if policy.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1

            return policy

    def set(
        self,
        key: str,
        policy: CachedPolicy,
        ttl_override: Optional[int] = None,
    ):
        """
        Set policy in cache.

        Target: <0.05ms

        Args:
            key: Cache key
            policy: Policy to cache
            ttl_override: Override default TTL
        """
        with self._lock:
            # Set expiration
            ttl = ttl_override if ttl_override is not None else self.ttl_seconds
            if ttl > 0:
                policy.expires_at = time.time() + ttl

            # Check if need to evict
            while len(self._cache) >= self.max_entries:
                # Evict oldest (least recently used)
                self._cache.popitem(last=False)
                self._evictions += 1

            # Add to cache
            self._cache[key] = policy
            self._cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """
        Delete policy from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()

    def invalidate_pattern(self, pattern: str):
        """
        Invalidate all entries matching pattern.

        Args:
            pattern: Pattern to match (substring match)
        """
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self._cache[key]
                self._evictions += 1

    def get_or_create(
        self,
        key: str,
        factory: callable,
        ttl_override: Optional[int] = None,
    ) -> CachedPolicy:
        """
        Get policy from cache or create using factory.

        Args:
            key: Cache key
            factory: Function to create policy if not cached
            ttl_override: Override default TTL

        Returns:
            CachedPolicy from cache or newly created
        """
        policy = self.get(key)
        if policy is not None:
            return policy

        # Create and cache
        policy = factory()
        self.set(key, policy, ttl_override)
        return policy

    def prune_expired(self):
        """Remove all expired entries from cache."""
        with self._lock:
            current_time = time.time()
            keys_to_delete = [
                k
                for k, v in self._cache.items()
                if v.expires_at > 0 and v.expires_at < current_time
            ]
            for key in keys_to_delete:
                del self._cache[key]
                self._evictions += 1

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
            }

    def __len__(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (without affecting LRU)."""
        with self._lock:
            if key not in self._cache:
                return False
            if self._cache[key].is_expired():
                return False
            return True
