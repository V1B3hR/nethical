"""
Tests for Cache Invalidation

Tests cache invalidation across all levels.
"""

import time

import pytest

from nethical.cache.l1_memory import L1MemoryCache
from nethical.cache.l3_global import L3GlobalCache
from nethical.cache.cache_hierarchy import CacheHierarchy, HierarchyConfig
from nethical.cache.invalidation import (
    InvalidationManager,
    InvalidationEvent,
    InvalidationType,
)
from nethical.cache.event_propagation import EventPropagation, CacheEvent, EventType


class TestL1MemoryCache:
    """Tests for L1MemoryCache."""

    def test_init(self):
        """Test cache initialization."""
        cache = L1MemoryCache(max_entries=1000, ttl_seconds=60)
        assert cache.max_entries == 1000
        assert cache.ttl_seconds == 60

    def test_set_get(self):
        """Test basic set and get."""
        cache = L1MemoryCache()

        cache.set("key1", {"value": "test"})
        result = cache.get("key1")

        assert result is not None
        assert result["value"] == "test"

    def test_get_miss(self):
        """Test cache miss."""
        cache = L1MemoryCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = L1MemoryCache(ttl_seconds=1)

        cache.set("key1", "value1")
        assert cache.get("key1") is not None

        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_lru_eviction(self):
        """Test LRU eviction."""
        cache = L1MemoryCache(max_entries=3)

        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new entry, should evict key2
        cache.set("key4", "value4")

        assert cache.get("key1") is not None  # Recently used
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_delete(self):
        """Test key deletion."""
        cache = L1MemoryCache()

        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") is False

    def test_invalidate_pattern(self):
        """Test pattern-based invalidation."""
        cache = L1MemoryCache()

        cache.set("policy:1", "value1")
        cache.set("policy:2", "value2")
        cache.set("agent:1", "value3")

        cache.invalidate_pattern("policy:")

        assert cache.get("policy:1") is None
        assert cache.get("policy:2") is None
        assert cache.get("agent:1") is not None

    def test_prune_expired(self):
        """Test expired entry pruning."""
        cache = L1MemoryCache(ttl_seconds=1)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        time.sleep(1.1)

        pruned = cache.prune_expired()
        assert pruned == 2
        assert cache.size() == 0

    def test_metrics(self):
        """Test metrics collection."""
        cache = L1MemoryCache()

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        metrics = cache.get_metrics()
        assert metrics["hits"] == 1
        assert metrics["misses"] == 1
        assert metrics["hit_rate"] == 0.5

    def test_get_or_set(self):
        """Test get_or_set pattern."""
        cache = L1MemoryCache()

        # First call should compute
        computed = False

        def factory():
            nonlocal computed
            computed = True
            return "computed_value"

        result1 = cache.get_or_set("key1", factory)
        assert result1 == "computed_value"
        assert computed is True

        # Second call should use cache
        computed = False
        result2 = cache.get_or_set("key1", factory)
        assert result2 == "computed_value"
        assert computed is False


class TestCacheHierarchy:
    """Tests for CacheHierarchy."""

    def test_init(self):
        """Test hierarchy initialization."""
        config = HierarchyConfig(enable_l1=True, enable_l2=False, enable_l3=True)
        hierarchy = CacheHierarchy(config=config)

        assert hierarchy.l1 is not None
        assert hierarchy.l2 is None
        assert hierarchy.l3 is not None

    def test_get_set(self):
        """Test basic get and set."""
        hierarchy = CacheHierarchy()

        hierarchy.set("key1", {"data": "value"})
        result = hierarchy.get("key1")

        assert result is not None
        assert result["data"] == "value"

    def test_l1_hit(self):
        """Test L1 cache hit."""
        hierarchy = CacheHierarchy()

        hierarchy.set("key1", "value1")
        
        # Multiple gets to ensure L1 hit tracking
        result = hierarchy.get("key1")
        result = hierarchy.get("key1")  # Second get

        assert result == "value1"

        metrics = hierarchy.get_metrics()
        # First set goes to L1+L3, first get from L1 is a hit
        # Since L1 has the value from set, both gets should be L1 hits
        assert metrics["l1_hits"] >= 1

    def test_read_through(self):
        """Test read-through caching."""
        l3 = L3GlobalCache()
        l3.set("key1", "value1")

        config = HierarchyConfig(enable_l1=True, enable_l2=False, enable_l3=True)
        hierarchy = CacheHierarchy(config=config, l3_cache=l3)

        # First get from L3
        result1 = hierarchy.get("key1")
        assert result1 == "value1"

        # Should now be in L1
        l3.delete("key1")  # Remove from L3
        result2 = hierarchy.get("key1")
        assert result2 == "value1"  # Still available from L1

    def test_delete(self):
        """Test deletion from all levels."""
        hierarchy = CacheHierarchy()

        hierarchy.set("key1", "value1")
        hierarchy.delete("key1")

        assert hierarchy.get("key1") is None

    def test_invalidate_pattern(self):
        """Test pattern invalidation."""
        hierarchy = CacheHierarchy()

        hierarchy.set("policy:1", "value1")
        hierarchy.set("policy:2", "value2")
        hierarchy.set("agent:1", "value3")

        hierarchy.invalidate_pattern("policy:")

        assert hierarchy.get("policy:1") is None
        assert hierarchy.get("policy:2") is None
        assert hierarchy.get("agent:1") is not None

    def test_metrics(self):
        """Test metrics collection."""
        hierarchy = CacheHierarchy()

        hierarchy.set("key1", "value1")
        hierarchy.get("key1")
        hierarchy.get("key2")

        metrics = hierarchy.get_metrics()
        assert "l1_hits" in metrics
        assert "cumulative_hit_rate" in metrics


class TestInvalidationManager:
    """Tests for InvalidationManager."""

    def test_init(self):
        """Test manager initialization."""
        manager = InvalidationManager()
        assert manager.cache_hierarchy is None

    def test_invalidate_policy(self):
        """Test policy invalidation."""
        hierarchy = CacheHierarchy()
        hierarchy.set("policy:123:data", "value")

        manager = InvalidationManager(cache_hierarchy=hierarchy)
        manager.invalidate_policy("123")

        assert hierarchy.get("policy:123:data") is None

    def test_invalidate_agent(self):
        """Test agent invalidation."""
        hierarchy = CacheHierarchy()
        hierarchy.set("agent:456:state", "value")

        manager = InvalidationManager(cache_hierarchy=hierarchy)
        manager.invalidate_agent("456")

        assert hierarchy.get("agent:456:state") is None

    def test_security_flush(self):
        """Test security flush."""
        hierarchy = CacheHierarchy()
        hierarchy.set("key1", "value1")
        hierarchy.set("key2", "value2")

        manager = InvalidationManager(cache_hierarchy=hierarchy)
        manager.security_flush("test_reason")

        # All keys should be cleared
        assert hierarchy.get("key1") is None
        assert hierarchy.get("key2") is None

    def test_subscribe(self):
        """Test event subscription."""
        manager = InvalidationManager()

        events_received = []

        def callback(event):
            events_received.append(event)

        manager.subscribe(callback)

        event = InvalidationEvent(
            event_type=InvalidationType.POLICY_UPDATE,
            pattern="test:*",
        )
        manager.invalidate(event)

        assert len(events_received) == 1
        assert events_received[0].event_type == InvalidationType.POLICY_UPDATE

    def test_metrics(self):
        """Test metrics collection."""
        manager = InvalidationManager()

        event = InvalidationEvent(
            event_type=InvalidationType.POLICY_UPDATE,
            pattern="test",
        )
        manager.invalidate(event)

        metrics = manager.get_metrics()
        assert metrics["total_invalidations"] == 1


class TestEventPropagation:
    """Tests for EventPropagation."""

    def test_init(self):
        """Test propagation initialization."""
        prop = EventPropagation(region="us-east-1")
        assert prop.region == "us-east-1"

    def test_publish(self):
        """Test event publishing."""
        prop = EventPropagation()

        events_received = []

        def callback(event):
            events_received.append(event)

        prop.subscribe(EventType.INVALIDATION, callback)

        event = CacheEvent(
            event_type=EventType.INVALIDATION,
            key="test:key",
        )
        prop.publish(event)

        assert len(events_received) == 1
        assert events_received[0].key == "test:key"

    def test_subscribe_unsubscribe(self):
        """Test subscribe and unsubscribe."""
        prop = EventPropagation()

        events_received = []

        def callback(event):
            events_received.append(event)

        prop.subscribe(EventType.UPDATE, callback)

        event = CacheEvent(event_type=EventType.UPDATE, key="test")
        prop.publish(event)
        assert len(events_received) == 1

        prop.unsubscribe(EventType.UPDATE, callback)

        prop.publish(event)
        assert len(events_received) == 1  # No new events

    def test_metrics(self):
        """Test metrics collection."""
        prop = EventPropagation()

        event = CacheEvent(event_type=EventType.INVALIDATION)
        prop.publish(event)

        metrics = prop.get_metrics()
        assert metrics["events_published"] == 1
