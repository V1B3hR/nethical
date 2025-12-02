"""
Three-Level Caching Architecture Module

Provides a hierarchical caching system for ultra-low latency:
- L1: In-memory cache (process-local, <0.1ms)
- L2: Regional Redis (<5ms)
- L3: Global distributed cache (<50ms)

Target: 95%+ cumulative cache hit rate
"""

from .l1_memory import L1MemoryCache
from .l2_redis import L2RedisCache
from .l3_global import L3GlobalCache
from .cache_key import CacheKey, generate_cache_key
from .cache_hierarchy import CacheHierarchy
from .invalidation import InvalidationManager, InvalidationEvent
from .event_propagation import EventPropagation, CacheEvent

__all__ = [
    # L1 Cache
    "L1MemoryCache",
    # L2 Cache
    "L2RedisCache",
    # L3 Cache
    "L3GlobalCache",
    # Keys
    "CacheKey",
    "generate_cache_key",
    # Hierarchy
    "CacheHierarchy",
    # Invalidation
    "InvalidationManager",
    "InvalidationEvent",
    # Events
    "EventPropagation",
    "CacheEvent",
]
