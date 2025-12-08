"""Tests for semantic cache safety and behavior."""

import pytest
import asyncio
import time


# Try to import the cache
try:
    from nethical.api.semantic_cache import SemanticCache

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    SemanticCache = None


pytestmark = pytest.mark.skipif(
    not CACHE_AVAILABLE, reason="Semantic cache not available - check dependencies"
)


@pytest.mark.asyncio
class TestCacheStorageSafety:
    """Test that cache only stores derived values (floats), not raw text."""

    async def test_cache_stores_only_floats(self):
        """Verify cache stores only similarity floats, not raw text."""
        cache = SemanticCache(maxsize=10, ttl=60, model_version="test")

        intent = "Read user data from database"
        action = "SELECT * FROM users"
        similarity = 0.85

        # Store value
        success = await cache.set(intent, action, similarity)
        assert success

        # Retrieve value
        cached_value = await cache.get(intent, action)
        assert cached_value is not None

        # Value should be a float
        assert isinstance(cached_value, float)
        assert cached_value == 0.85

        # Check internal cache - should NOT contain raw text
        cache_key = cache._compute_key(intent, action)
        internal_value = cache._cache.get(cache_key)

        # Internal value should be float, not string
        assert isinstance(internal_value, float)
        assert intent not in str(internal_value)
        assert action not in str(internal_value)

    async def test_cache_key_is_hash_not_text(self):
        """Verify cache keys are hashes, not containing raw text."""
        cache = SemanticCache(maxsize=10, ttl=60)

        sensitive_intent = "Access confidential patient records"
        sensitive_action = "SELECT ssn, diagnosis FROM patients"

        # Compute cache key
        key = cache._compute_key(sensitive_intent, sensitive_action)

        # Key should be a hash (hex string)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex digest length

        # Key should NOT contain raw sensitive text
        assert "patient" not in key.lower()
        assert "ssn" not in key.lower()
        assert "diagnosis" not in key.lower()
        assert "confidential" not in key.lower()

    async def test_cache_rejects_non_float_values(self):
        """Test that cache rejects non-float values."""
        cache = SemanticCache(maxsize=10, ttl=60)

        intent = "test intent"
        action = "test action"

        # Try to store string (should fail gracefully)
        success = await cache.set(intent, action, "not a float")
        assert not success  # Should return False, not crash

        # Try to store dict (should fail gracefully)
        success = await cache.set(intent, action, {"similarity": 0.5})
        assert not success

        # Try to store None (should fail gracefully)
        success = await cache.set(intent, action, None)
        assert not success

    async def test_cache_clamps_values_to_valid_range(self):
        """Test that cache clamps similarity values to [0.0, 1.0]."""
        cache = SemanticCache(maxsize=10, ttl=60)

        intent = "test"
        action = "test"

        # Store value > 1.0
        await cache.set(intent, action, 1.5)
        cached = await cache.get(intent, action)
        assert cached == 1.0  # Should be clamped

        # Store negative value
        await cache.set(intent, action, -0.5)
        cached = await cache.get(intent, action)
        assert cached == 0.0  # Should be clamped


@pytest.mark.asyncio
class TestCacheTTLEviction:
    """Test TTL-based eviction."""

    async def test_cache_respects_ttl(self):
        """Test that cached entries expire after TTL."""
        # Short TTL for testing
        cache = SemanticCache(maxsize=100, ttl=1, model_version="test")

        intent = "test intent"
        action = "test action"
        similarity = 0.75

        # Store value
        await cache.set(intent, action, similarity)

        # Should be retrievable immediately
        cached = await cache.get(intent, action)
        assert cached == similarity

        # Wait for TTL to expire
        await asyncio.sleep(1.5)

        # Should be evicted
        cached = await cache.get(intent, action)
        assert cached is None

    async def test_cache_updates_ttl_on_set(self):
        """Test that setting an existing key resets its TTL."""
        cache = SemanticCache(maxsize=100, ttl=2)

        intent = "test"
        action = "test"

        # Initial set
        await cache.set(intent, action, 0.5)

        # Wait half the TTL
        await asyncio.sleep(1)

        # Update value (resets TTL)
        await cache.set(intent, action, 0.8)

        # Wait another half TTL (total 1.5s from initial, 0.5s from update)
        await asyncio.sleep(1)

        # Should still be cached (because TTL was reset)
        cached = await cache.get(intent, action)
        assert cached == 0.8


@pytest.mark.asyncio
class TestCacheMaxsizeBehavior:
    """Test LRU eviction when maxsize is reached."""

    async def test_cache_respects_maxsize(self):
        """Test that cache evicts old entries when maxsize is reached."""
        cache = SemanticCache(maxsize=3, ttl=60)

        # Fill cache to max
        await cache.set("intent1", "action1", 0.1)
        await cache.set("intent2", "action2", 0.2)
        await cache.set("intent3", "action3", 0.3)

        # All should be cached
        assert await cache.get("intent1", "action1") == 0.1
        assert await cache.get("intent2", "action2") == 0.2
        assert await cache.get("intent3", "action3") == 0.3

        # Add one more (exceeds maxsize)
        await cache.set("intent4", "action4", 0.4)

        # Oldest entry should be evicted
        # Note: TTLCache uses LRU, so least recently used is evicted
        # Since we just accessed all in order, intent1 should be oldest
        assert await cache.get("intent4", "action4") == 0.4

        # Check that we didn't exceed maxsize
        stats = cache.get_stats()
        assert stats["current_size"] <= 3

    async def test_cache_size_reported_correctly(self):
        """Test that cache reports its size correctly."""
        cache = SemanticCache(maxsize=10, ttl=60)

        # Initially empty
        stats = cache.get_stats()
        assert stats["current_size"] == 0

        # Add entries
        for i in range(5):
            await cache.set(f"intent{i}", f"action{i}", float(i) / 10)

        # Size should be 5
        stats = cache.get_stats()
        assert stats["current_size"] == 5


@pytest.mark.asyncio
class TestCacheSingleFlight:
    """Test single-flight control prevents stampedes."""

    async def test_single_flight_prevents_duplicate_computation(self):
        """Test that concurrent requests for same key only compute once."""
        cache = SemanticCache(maxsize=100, ttl=60)

        intent = "test intent"
        action = "test action"

        compute_count = 0

        async def expensive_compute():
            nonlocal compute_count
            compute_count += 1
            await asyncio.sleep(0.1)  # Simulate expensive computation
            return 0.75

        # Launch many concurrent requests for same key
        tasks = [
            cache.get_or_compute(intent, action, expensive_compute) for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # All should get the same result
        assert all(r == 0.75 for r in results)

        # But compute should only be called once (single-flight)
        assert compute_count == 1

    async def test_single_flight_per_key(self):
        """Test that different keys can compute in parallel."""
        cache = SemanticCache(maxsize=100, ttl=60)

        compute_counts = {}

        def make_compute_fn(key):
            async def compute():
                compute_counts[key] = compute_counts.get(key, 0) + 1
                await asyncio.sleep(0.05)
                return 0.5

            return compute

        # Launch concurrent requests for different keys
        tasks = []
        for i in range(5):
            intent = f"intent_{i}"
            action = f"action_{i}"
            compute_fn = make_compute_fn(i)
            tasks.append(cache.get_or_compute(intent, action, compute_fn))

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r == 0.5 for r in results)

        # Each key should compute once
        assert len(compute_counts) == 5
        assert all(count == 1 for count in compute_counts.values())


@pytest.mark.asyncio
class TestCacheFailOpen:
    """Test fail-open behavior (never crashes)."""

    async def test_cache_handles_compute_errors_gracefully(self):
        """Test that compute errors don't crash, return safe default."""
        cache = SemanticCache(maxsize=100, ttl=60)

        async def failing_compute():
            raise ValueError("Intentional test error")

        # Should not crash, should return safe default
        result = await cache.get_or_compute("test", "test", failing_compute)
        assert result == 0.0  # Safe default

    async def test_cache_get_errors_return_none(self):
        """Test that get errors return None (fail open)."""
        cache = SemanticCache(maxsize=100, ttl=60)

        # Force an error by corrupting cache key method
        original_compute_key = cache._compute_key

        def broken_compute_key(*args, **kwargs):
            raise RuntimeError("Broken key computation")

        cache._compute_key = broken_compute_key

        # Should not crash, should return None
        result = await cache.get("test", "test")
        assert result is None

        # Restore
        cache._compute_key = original_compute_key

    async def test_cache_set_errors_return_false(self):
        """Test that set errors return False (fail open)."""
        cache = SemanticCache(maxsize=100, ttl=60)

        # Force an error
        original_compute_key = cache._compute_key

        def broken_compute_key(*args, **kwargs):
            raise RuntimeError("Broken")

        cache._compute_key = broken_compute_key

        # Should not crash, should return False
        result = await cache.set("test", "test", 0.5)
        assert result is False

        # Restore
        cache._compute_key = original_compute_key


@pytest.mark.asyncio
class TestCacheStats:
    """Test cache statistics tracking."""

    async def test_cache_tracks_hits_and_misses(self):
        """Test that cache tracks hit/miss statistics."""
        cache = SemanticCache(maxsize=100, ttl=60)

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Miss (not cached)
        result = await cache.get("test1", "test1")
        assert result is None
        stats = cache.get_stats()
        assert stats["misses"] == 1

        # Set value
        await cache.set("test1", "test1", 0.5)

        # Hit (cached)
        result = await cache.get("test1", "test1")
        assert result == 0.5
        stats = cache.get_stats()
        assert stats["hits"] == 1

        # Another miss
        result = await cache.get("test2", "test2")
        stats = cache.get_stats()
        assert stats["misses"] == 2

    async def test_cache_calculates_hit_rate(self):
        """Test that cache calculates hit rate correctly."""
        cache = SemanticCache(maxsize=100, ttl=60)

        # Set some values
        await cache.set("test1", "test1", 0.5)
        await cache.set("test2", "test2", 0.6)

        # 2 hits
        await cache.get("test1", "test1")
        await cache.get("test2", "test2")

        # 1 miss
        await cache.get("test3", "test3")

        stats = cache.get_stats()
        # Hit rate should be 2/(2+1) = 66.67%
        assert abs(stats["hit_rate_percent"] - 66.67) < 0.1


@pytest.mark.asyncio
class TestCacheConfigParams:
    """Test cache key includes config parameters."""

    async def test_different_configs_use_different_keys(self):
        """Test that different config params result in cache misses."""
        cache = SemanticCache(maxsize=100, ttl=60)

        intent = "test"
        action = "test"

        # Set with one config
        await cache.set(intent, action, 0.5, config_params={"threshold": 0.7})

        # Get with same config - should hit
        result = await cache.get(intent, action, config_params={"threshold": 0.7})
        assert result == 0.5

        # Get with different config - should miss
        result = await cache.get(intent, action, config_params={"threshold": 0.8})
        assert result is None

    async def test_cache_key_sorted_config(self):
        """Test that config param order doesn't affect cache key."""
        cache = SemanticCache(maxsize=100, ttl=60)

        key1 = cache._compute_key("test", "test", config_params={"a": 1, "b": 2})
        key2 = cache._compute_key("test", "test", config_params={"b": 2, "a": 1})

        # Keys should be identical (params sorted internally)
        assert key1 == key2
