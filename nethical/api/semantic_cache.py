"""
Safe semantic similarity cache.

Stores ONLY derived float similarity scores (no raw text) keyed by SHA256 over:
    (model_version | normalized_intent | normalized_action | sorted_config_kv)

Features:
    - LRU + TTL eviction (cachetools.TTLCache)
    - Single-flight (per-key asyncio.Lock) to prevent thundering herd
    - Fail-open: errors return None (caller recomputes)
    - Stats: hits, misses, errors, hit rate
    - Lock pruning for keys no longer present
"""

import os
import hashlib
import asyncio
import logging
from typing import Optional, Dict, Any, Callable, Awaitable
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(
        self,
        maxsize: Optional[int] = None,
        ttl: Optional[int] = None,
        model_version: str = "default"
    ) -> None:
        self.maxsize = maxsize or int(os.getenv("NETHICAL_CACHE_MAXSIZE", "20000"))
        self.ttl = ttl or int(os.getenv("NETHICAL_CACHE_TTL", "600"))
        self.model_version = model_version

        self._cache: TTLCache = TTLCache(maxsize=self.maxsize, ttl=self.ttl)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

        self._hits = 0
        self._misses = 0
        self._errors = 0

        logger.info(
            "Semantic cache initialized: maxsize=%d ttl=%ds model=%s",
            self.maxsize, self.ttl, self.model_version
        )

    # ------------------------------------------------------------------ #
    # Key computation
    # ------------------------------------------------------------------ #
    def _compute_key(
        self,
        intent: str,
        action: str,
        config_params: Optional[Dict[str, Any]] = None
    ) -> str:
        norm_intent = intent.strip().lower()
        norm_action = action.strip().lower()

        parts = [self.model_version, norm_intent, norm_action]
        if config_params:
            config_str = ",".join(f"{k}={v}" for k, v in sorted(config_params.items()))
            parts.append(config_str)
        digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        return digest

    # ------------------------------------------------------------------ #
    # Basic operations
    # ------------------------------------------------------------------ #
    async def get(
        self,
        intent: str,
        action: str,
        config_params: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        try:
            key = self._compute_key(intent, action, config_params)
            if key in self._cache:
                val = self._cache[key]
                self._hits += 1
                logger.debug("Cache HIT key=%s… val=%.4f", key[:12], val)
                return val
            self._misses += 1
            logger.debug("Cache MISS key=%s…", key[:12])
            return None
        except Exception as e:
            self._errors += 1
            logger.warning("Cache get error (fail-open): %s", e)
            return None

    async def set(
        self,
        intent: str,
        action: str,
        similarity: float,
        config_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        try:
            key = self._compute_key(intent, action, config_params)
            if not isinstance(similarity, (float, int)):
                logger.error("Similarity must be numeric; got %s", type(similarity))
                return False
            similarity = float(similarity)
            if similarity < 0.0 or similarity > 1.0:
                similarity = max(0.0, min(1.0, similarity))  # clamp

            self._cache[key] = similarity
            logger.debug("Cache SET key=%s… = %.4f", key[:12], similarity)
            return True
        except Exception as e:
            self._errors += 1
            logger.warning("Cache set error (fail-open): %s", e)
            return False

    # ------------------------------------------------------------------ #
    # Single-flight compute
    # ------------------------------------------------------------------ #
    async def get_or_compute(
        self,
        intent: str,
        action: str,
        compute_fn: Callable[[], Awaitable[float]],
        config_params: Optional[Dict[str, Any]] = None
    ) -> float:
        cached = await self.get(intent, action, config_params)
        if cached is not None:
            return cached

        key = self._compute_key(intent, action, config_params)

        async with self._locks_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            lock = self._locks[key]

        async with lock:
            # Re-check after acquiring lock
            cached = await self.get(intent, action, config_params)
            if cached is not None:
                return cached

            try:
                similarity = await compute_fn()
                await self.set(intent, action, similarity, config_params)
                return similarity
            except Exception as e:
                logger.error("Compute error for key=%s…: %s", key[:12], e)
                # Fail-open fallback: neutral similarity (None would force recompute every time)
                return 0.5

    # ------------------------------------------------------------------ #
    # Maintenance / stats
    # ------------------------------------------------------------------ #
    def prune_locks(self) -> int:
        """
        Remove locks for keys no longer present in cache to prevent memory growth.
        Returns number of pruned locks.
        """
        to_remove = [k for k in self._locks.keys() if k not in self._cache]
        for k in to_remove:
            self._locks.pop(k, None)
        if to_remove:
            logger.debug("Pruned %d stale cache locks", len(to_remove))
        return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100.0) if total > 0 else 0.0
        return {
            "maxsize": self.maxsize,
            "ttl": self.ttl,
            "current_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
            "hit_rate_percent": round(hit_rate, 2),
            "active_locks": len(self._locks),
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = self._misses = self._errors = 0
        logger.info("Semantic cache cleared")
