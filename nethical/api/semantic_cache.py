"""
Safe cache for semantic similarity results.

Implements in-process LRU+TTL cache that stores only derived values (floats),
not raw text. Designed for single-instance deployments with fail-open behavior.
"""

import os
import hashlib
import asyncio
import logging
from typing import Optional, Dict, Any
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Thread-safe cache for semantic similarity scores.
    
    Features:
    - LRU eviction with TTL
    - Stores only derived values (floats), never raw text
    - SHA256-based keys for privacy
    - Single-flight control to prevent stampedes
    - Fail-open behavior (never crashes on cache errors)
    - Configurable via environment variables
    """
    
    def __init__(
        self,
        maxsize: Optional[int] = None,
        ttl: Optional[int] = None,
        model_version: str = "default"
    ):
        """
        Initialize semantic cache.
        
        Args:
            maxsize: Maximum number of entries (default from env or 20000)
            ttl: Time-to-live in seconds (default from env or 600)
            model_version: Embedding model version for cache key namespacing
        """
        self.maxsize = maxsize or int(os.getenv("NETHICAL_CACHE_MAXSIZE", "20000"))
        self.ttl = ttl or int(os.getenv("NETHICAL_CACHE_TTL", "600"))
        self.model_version = model_version
        
        # Initialize TTL cache
        self._cache: TTLCache = TTLCache(maxsize=self.maxsize, ttl=self.ttl)
        
        # Per-key locks for single-flight control
        self._locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Lock for locks dict
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._errors = 0
        
        logger.info(
            f"Semantic cache initialized: maxsize={self.maxsize}, ttl={self.ttl}s, "
            f"model={model_version}"
        )
    
    def _compute_key(
        self,
        intent: str,
        action: str,
        config_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Compute cache key from inputs.
        
        Uses SHA256 of normalized inputs plus model version and config.
        IMPORTANT: Raw text is NOT stored, only the hash.
        
        Args:
            intent: Stated intent
            action: Actual action
            config_params: Optional configuration parameters (thresholds, etc.)
            
        Returns:
            SHA256 hex digest as cache key
        """
        # Normalize inputs
        norm_intent = intent.strip().lower()
        norm_action = action.strip().lower()
        
        # Build key material
        key_parts = [
            self.model_version,
            norm_intent,
            norm_action
        ]
        
        # Include relevant config in key
        if config_params:
            # Sort for consistent hashing
            config_str = ",".join(f"{k}={v}" for k, v in sorted(config_params.items()))
            key_parts.append(config_str)
        
        # Compute hash
        key_material = "|".join(key_parts).encode("utf-8")
        return hashlib.sha256(key_material).hexdigest()
    
    async def get(
        self,
        intent: str,
        action: str,
        config_params: Optional[Dict[str, Any]] = None
    ) -> Optional[float]:
        """
        Get cached similarity score.
        
        Args:
            intent: Stated intent
            action: Actual action
            config_params: Optional configuration parameters
            
        Returns:
            Cached similarity score (float) or None if not found/error
        """
        try:
            key = self._compute_key(intent, action, config_params)
            
            if key in self._cache:
                value = self._cache[key]
                self._hits += 1
                logger.debug(f"Cache HIT for key {key[:16]}...")
                return value
            
            self._misses += 1
            logger.debug(f"Cache MISS for key {key[:16]}...")
            return None
            
        except Exception as e:
            # Fail open - never crash on cache errors
            self._errors += 1
            logger.warning(f"Cache get error (fail-open): {e}")
            return None
    
    async def set(
        self,
        intent: str,
        action: str,
        similarity: float,
        config_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store similarity score in cache.
        
        IMPORTANT: Only stores the derived float value, NOT the raw text.
        
        Args:
            intent: Stated intent (used only for key generation)
            action: Actual action (used only for key generation)
            similarity: Computed similarity score (the ONLY value stored)
            config_params: Optional configuration parameters
            
        Returns:
            True if stored successfully, False on error
        """
        try:
            key = self._compute_key(intent, action, config_params)
            
            # Validate similarity is a float
            if not isinstance(similarity, (float, int)):
                logger.error(f"Invalid similarity type: {type(similarity)}")
                return False
            
            # Clamp to valid range
            similarity = max(0.0, min(1.0, float(similarity)))
            
            # Store only the float
            self._cache[key] = similarity
            logger.debug(f"Cache SET for key {key[:16]}... = {similarity:.4f}")
            return True
            
        except Exception as e:
            # Fail open - never crash on cache errors
            self._errors += 1
            logger.warning(f"Cache set error (fail-open): {e}")
            return False
    
    async def get_or_compute(
        self,
        intent: str,
        action: str,
        compute_fn,
        config_params: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Get cached value or compute with single-flight control.
        
        Ensures only one concurrent computation per key (prevents stampedes).
        
        Args:
            intent: Stated intent
            action: Actual action
            compute_fn: Async function to compute similarity if not cached
            config_params: Optional configuration parameters
            
        Returns:
            Similarity score (from cache or computed)
        """
        # Try cache first (fast path)
        cached = await self.get(intent, action, config_params)
        if cached is not None:
            return cached
        
        # Single-flight control: only one computation per key
        key = self._compute_key(intent, action, config_params)
        
        # Get or create lock for this key
        async with self._locks_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            lock = self._locks[key]
        
        # Acquire lock for this key
        async with lock:
            # Check cache again (another coroutine may have computed it)
            cached = await self.get(intent, action, config_params)
            if cached is not None:
                return cached
            
            # Compute
            try:
                logger.debug(f"Computing similarity for key {key[:16]}...")
                similarity = await compute_fn()
                
                # Store in cache
                await self.set(intent, action, similarity, config_params)
                
                return similarity
                
            except Exception as e:
                logger.error(f"Compute function failed for key {key[:16]}...: {e}")
                # Return a safe default on compute error
                return 0.0
        
        # Clean up lock after use (optional, periodic cleanup is better)
        # We keep locks around for hot keys to avoid recreation overhead
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache performance metrics
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "maxsize": self.maxsize,
            "ttl": self.ttl,
            "current_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
            "hit_rate_percent": round(hit_rate, 2),
            "active_locks": len(self._locks)
        }
    
    def clear(self) -> None:
        """Clear all cached entries (useful for testing)."""
        self._cache.clear()
        logger.info("Cache cleared")
