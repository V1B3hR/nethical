"""
Three-Level Caching Architecture Module (Extended with Satellite Tier)

Provides a hierarchical caching system for ultra-low latency:
- L1: In-memory cache (process-local, <0.1ms)
- L2: Regional Redis (<5ms)
- L3: Global distributed cache (<50ms)
- Satellite: Edge cache with offline support (high-latency tolerant)

Target: 95%+ cumulative cache hit rate

Performance targets:
- L1 cache hit: <1ms even on satellite
- L2 regional cache: <5ms same-region
- L3 global cache: <50ms cross-region
- Satellite failover: <10 seconds
"""

from .l1_memory import L1MemoryCache
from .l2_redis import L2RedisCache
from .l3_global import L3GlobalCache
from .cache_key import CacheKey, generate_cache_key
from .cache_hierarchy import CacheHierarchy, HierarchyConfig
from .invalidation import InvalidationManager, InvalidationEvent
from .event_propagation import EventPropagation, CacheEvent
from .satellite_cache import (
    SatelliteCache,
    SatelliteCacheConfig,
    CacheEntry,
    ConflictResolutionStrategy,
    SyncState,
)

__all__ = [
    # L1 Cache
    "L1MemoryCache",
    # L2 Cache
    "L2RedisCache",
    # L3 Cache
    "L3GlobalCache",
    # Satellite Cache
    "SatelliteCache",
    "SatelliteCacheConfig",
    "CacheEntry",
    "ConflictResolutionStrategy",
    "SyncState",
    # Keys
    "CacheKey",
    "generate_cache_key",
    # Hierarchy
    "CacheHierarchy",
    "HierarchyConfig",
    # Invalidation
    "InvalidationManager",
    "InvalidationEvent",
    # Events
    "EventPropagation",
    "CacheEvent",
]
