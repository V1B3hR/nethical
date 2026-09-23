# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Unit and integration tests for SatelliteCache, CacheKey, L2, and L3 global caching."""

import asyncio
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nethical.cache.cache_key import (
    CacheKey,
    generate_agent_key,
    generate_cache_key,
    generate_decision_key,
    generate_policy_key,
)
from nethical.cache.l2_redis import L2Config, L2RedisCache
from nethical.cache.l3_global import L3Config, L3GlobalCache
from nethical.cache.satellite_cache import (
    CacheEntry,
    ConflictResolutionStrategy,
    OfflineRequest,
    SatelliteCache,
    SatelliteCacheConfig,
    SyncState,
)


class TestCacheKeyGeneration:
    """Tests for consistent cache key generation."""

    def test_cache_key_deterministic(self) -> None:
        """Verifies that cache keys are generated deterministically regardless of argument order."""
        key1 = generate_cache_key("test_ns", a=1, b="two", c=[3, 4])
        key2 = generate_cache_key("test_ns", c=[3, 4], b="two", a=1)
        assert key1.key == key2.key
        assert key1.full_key() == key2.full_key()
        assert key1.full_key().startswith("test_ns:v1:")

    def test_generate_decision_key(self) -> None:
        """Verifies decision key generation contains stable context."""
        key = generate_decision_key(
            agent_id="sentinel_01",
            action="execute_emergency_stop",
            action_type="kinetic_control",
            context={"domain": "defense", "environment": "tactical_edge", "volatile_noise": 999},
        )
        assert "decision:v1:" in key
        # Second call with same stable context yields identical key
        key2 = generate_decision_key(
            agent_id="sentinel_01",
            action="execute_emergency_stop",
            action_type="kinetic_control",
            context={"environment": "tactical_edge", "domain": "defense"},
        )
        assert key == key2

    def test_generate_policy_and_agent_keys(self) -> None:
        """Verifies policy and agent keys formatting."""
        p_key = generate_policy_key("pol_life_preservation", version="2.0")
        assert "policy:2.0:" in p_key

        a_key = generate_agent_key("agent_drone_42", "telemetry_quota")
        assert "agent:v1:" in a_key


class TestSatelliteCache:
    """Comprehensive test suite for satellite edge cache layer."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path: Path) -> str:
        cache_dir = tmp_path / "satellite_cache_test"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir)

    def test_satellite_cache_initialisation(self, temp_cache_dir: str) -> None:
        """Verifies initialisation of satellite cache with custom config and temp directory."""
        cfg = SatelliteCacheConfig(
            persistence_enabled=True,
            persistence_path=temp_cache_dir,
            satellite_ttl_multiplier=2.5,
        )
        cache = SatelliteCache(config=cfg, region_id="orbit_leo_01")
        assert cache.region_id == "orbit_leo_01"
        assert cache.is_online is True
        assert Path(temp_cache_dir).exists()

    def test_satellite_cache_set_get_and_persistence(self, temp_cache_dir: str) -> None:
        """Verifies basic get, set, and file-based persistence."""
        cfg = SatelliteCacheConfig(persistence_enabled=True, persistence_path=temp_cache_dir)
        cache = SatelliteCache(config=cfg, region_id="ground_station_01")

        cache.set("mission_status", {"status": "GREEN", "fuel_level": 0.88})
        retrieved = cache.get("mission_status")
        assert retrieved is not None
        assert retrieved["status"] == "GREEN"
        assert retrieved["fuel_level"] == 0.88

        # Verify disk persistence file exists
        persisted_file = Path(temp_cache_dir) / "mission_status.cache"
        assert persisted_file.exists()

        # Delete entry
        cache.delete("mission_status")
        assert cache.get("mission_status") is None
        assert not persisted_file.exists()

    def test_satellite_cache_compression(self, temp_cache_dir: str) -> None:
        """Verifies compression kicks in for payloads larger than compression_threshold_bytes."""
        cfg = SatelliteCacheConfig(
            persistence_enabled=False,
            compression_enabled=True,
            compression_threshold_bytes=100,
        )
        cache = SatelliteCache(config=cfg, region_id="orbital_mesh")

        # Large repetitive telemetry string compressible by >20%
        large_payload = "TELEMETRY_SAMPLE_BLOCK_" * 200
        cache.set("large_telemetry", large_payload)

        retrieved = cache.get("large_telemetry")
        assert retrieved == large_payload
        assert cache._compression_savings_bytes > 0

    def test_satellite_cache_ttl_and_expiry(self) -> None:
        """Verifies TTL calculation and safe timezone-aware expiry handling."""
        cfg = SatelliteCacheConfig(
            persistence_enabled=False,
            default_ttl_seconds=1,
            satellite_ttl_multiplier=1.0,
        )
        cache = SatelliteCache(config=cfg, region_id="cubesat_9")

        cache.set("ephemeral_heartbeat", "ALIVE", ttl_seconds=1)
        assert cache.get("ephemeral_heartbeat") == "ALIVE"

        # Advance entry updated_at into the past
        entry = cache._cache["ephemeral_heartbeat"]
        entry.updated_at = datetime.now(timezone.utc) - timedelta(seconds=10)

        # Should now be expired and return None
        assert cache.get("ephemeral_heartbeat") is None

    def test_satellite_cache_offline_queue_eviction(self) -> None:
        """Verifies offline queue buffering and prioritized capacity eviction."""
        cfg = SatelliteCacheConfig(
            persistence_enabled=False,
            offline_queue_max_size=3,
        )
        cache = SatelliteCache(config=cfg, region_id="deep_space_probe")
        cache.is_online = False

        # Queue 3 requests
        cache.set("item1", "v1", write_through=True)
        cache.set("item2", "v2", write_through=True)
        cache.set("item3", "v3", write_through=True)
        assert len(cache._offline_queue) == 3

        # Adding a 4th request must evict the lowest priority oldest request without error
        cache.set("item4", "v4", write_through=True)
        assert len(cache._offline_queue) == 3
        # item4 should be present in the offline queue
        assert any(r.key == "item4" for r in cache._offline_queue)

    @pytest.mark.asyncio
    async def test_satellite_cache_sync_pending(self) -> None:
        """Verifies asynchronous syncing of pending write-through entries upon reconnection."""
        cfg = SatelliteCacheConfig(persistence_enabled=False)
        cache = SatelliteCache(config=cfg, region_id="leo_sat_4")
        cache.is_online = False

        cache.set("pending_key_1", "payload1", write_through=True)
        cache.set("pending_key_2", "payload2", write_through=True)
        assert len(cache._pending_sync) == 2

        # Syncing while offline should return 0
        synced = await cache.sync_pending()
        assert synced == 0

        # Coming online should sync entries
        cache._is_online = True
        synced = await cache.sync_pending()
        assert synced >= 2
        assert len(cache._pending_sync) == 0

    def test_satellite_cache_conflict_resolution(self) -> None:
        """Verifies conflict resolution strategies (LAST_WRITE_WINS, FIRST_WRITE_WINS, CALLBACK)."""
        cfg = SatelliteCacheConfig(persistence_enabled=False)
        cache = SatelliteCache(config=cfg, region_id="leo_sat_5")

        now = datetime.now(timezone.utc)
        older = now - timedelta(seconds=60)

        entry_local = CacheEntry(
            key="k",
            value="local_val",
            created_at=older - timedelta(seconds=10),
            updated_at=older,
            ttl_seconds=300,
            version=1,
        )
        entry_remote = CacheEntry(
            key="k",
            value="remote_val",
            created_at=older,
            updated_at=now,
            ttl_seconds=300,
            version=2,
        )

        # LAST_WRITE_WINS
        cache.config.conflict_resolution = ConflictResolutionStrategy.LAST_WRITE_WINS
        resolved = cache.resolve_conflict(entry_local, entry_remote)
        assert resolved.value == "remote_val"

        # FIRST_WRITE_WINS
        cache.config.conflict_resolution = ConflictResolutionStrategy.FIRST_WRITE_WINS
        resolved = cache.resolve_conflict(entry_local, entry_remote)
        assert resolved.value == "local_val"

        # CALLBACK
        cache.config.conflict_resolution = ConflictResolutionStrategy.CALLBACK
        cache.set_conflict_callback(lambda loc, rem: loc)
        resolved = cache.resolve_conflict(entry_local, entry_remote)
        assert resolved.value == "local_val"


class TestL3GlobalCache:
    """Tests for L3 distributed cache in-memory fallback."""

    def test_l3_memory_fallback_lifecycle(self) -> None:
        """Verifies in-memory fallback behavior and metrics."""
        cfg = L3Config(provider="memory", ttl_seconds=60)
        cache = L3GlobalCache(config=cfg)

        cache.set("global_quota", 1000)
        assert cache.get("global_quota") == 1000  # 1st hit
        assert cache.exists("global_quota") is True  # 2nd hit via exists()

        metrics = cache.get_metrics()
        assert metrics["hits"] == 2
        assert metrics["misses"] == 0

        cache.delete("global_quota")
        assert cache.get("global_quota") is None
        assert cache.get_metrics()["misses"] == 1

