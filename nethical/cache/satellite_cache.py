"""
Satellite-Optimized Cache Layer

Provides satellite-aware caching with:
- Local persistent cache with sync-on-reconnect
- Offline queue for requests during connectivity gaps
- Write-through with read-local strategy
- Conflict resolution for eventual consistency
- Longer TTLs for satellite edge nodes
- Compressed cache payloads for bandwidth-constrained links
"""

import asyncio
import gzip
import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies for eventual consistency."""

    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MERGE = "merge"
    CALLBACK = "callback"


class SyncState(Enum):
    """Synchronization state."""

    SYNCED = "synced"
    PENDING = "pending"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass
class CacheEntry:
    """Cache entry with metadata for satellite sync."""

    key: str
    value: Any
    created_at: datetime
    updated_at: datetime
    ttl_seconds: int
    version: int = 1
    sync_state: SyncState = SyncState.SYNCED
    checksum: str = ""
    compressed: bool = False
    size_bytes: int = 0
    origin_region: str = ""


@dataclass
class OfflineRequest:
    """Queued request for offline processing."""

    request_id: str
    operation: str  # get, set, delete
    key: str
    value: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class SatelliteCacheConfig:
    """Configuration for satellite cache."""

    # TTL settings (longer for satellite)
    default_ttl_seconds: int = 1800  # 30 minutes (vs 5 min for terrestrial)
    max_ttl_seconds: int = 7200  # 2 hours
    min_ttl_seconds: int = 60

    # TTL multiplier for satellite mode
    satellite_ttl_multiplier: float = 3.0

    # Compression settings
    compression_enabled: bool = True
    compression_threshold_bytes: int = 1024
    compression_level: int = 6

    # Persistence settings
    persistence_enabled: bool = True
    persistence_path: str = "/tmp/nethical_cache"
    persistence_sync_interval_seconds: float = 60.0

    # Offline queue settings
    offline_queue_max_size: int = 1000
    offline_queue_persist: bool = True

    # Sync settings
    sync_batch_size: int = 50
    sync_retry_delay_seconds: float = 5.0
    conflict_resolution: ConflictResolutionStrategy = (
        ConflictResolutionStrategy.LAST_WRITE_WINS
    )

    # Delta updates
    delta_updates_enabled: bool = True
    delta_threshold_percent: float = 20.0


class SatelliteCache:
    """
    Satellite-optimized cache layer.

    Designed for high-latency, bandwidth-constrained satellite links:
    - Local persistent storage for offline operation
    - Automatic sync on reconnection
    - Compressed payloads
    - Longer TTLs
    - Conflict resolution for eventual consistency
    """

    def __init__(
        self,
        config: Optional[SatelliteCacheConfig] = None,
        region_id: str = "satellite",
    ):
        """
        Initialize satellite cache.

        Args:
            config: Cache configuration
            region_id: Region identifier
        """
        self.config = config or SatelliteCacheConfig()
        self.region_id = region_id

        # In-memory cache
        self._cache: Dict[str, CacheEntry] = {}

        # Offline request queue
        self._offline_queue: List[OfflineRequest] = []

        # Sync tracking
        self._pending_sync: Set[str] = set()
        self._last_sync: Optional[datetime] = None
        self._is_online = True

        # Persistence
        self._persistence_path = Path(self.config.persistence_path)
        if self.config.persistence_enabled:
            self._persistence_path.mkdir(parents=True, exist_ok=True)

        # Metrics
        self._hits = 0
        self._misses = 0
        self._sync_operations = 0
        self._compression_savings_bytes = 0

        # Callbacks
        self._conflict_callback: Optional[Callable] = None

        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None

        logger.info(
            f"SatelliteCache initialized for region {region_id}, "
            f"TTL multiplier: {self.config.satellite_ttl_multiplier}x"
        )

    @property
    def is_online(self) -> bool:
        """Check if cache is online."""
        return self._is_online

    @is_online.setter
    def is_online(self, value: bool):
        """Set online status."""
        old_value = self._is_online
        self._is_online = value
        if not old_value and value:
            # Coming back online - trigger sync
            logger.info("Connection restored - initiating cache sync")
            asyncio.create_task(self.sync_pending())

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (read-local strategy).

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        entry = self._cache.get(key)

        if entry is None:
            # Check persistent storage
            entry = self._load_from_persistence(key)
            if entry:
                self._cache[key] = entry

        if entry is None:
            self._misses += 1
            return None

        # Check expiry
        if self._is_expired(entry):
            self._misses += 1
            del self._cache[key]
            return None

        self._hits += 1

        # Decompress if needed
        if entry.compressed:
            return self._decompress(entry.value)

        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        write_through: bool = True,
    ):
        """
        Set value in cache (write-through strategy).

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
            write_through: Whether to queue for sync
        """
        # Calculate TTL with satellite multiplier
        base_ttl = ttl or self.config.default_ttl_seconds
        if not self._is_online:
            # Longer TTL when offline
            base_ttl = int(base_ttl * self.config.satellite_ttl_multiplier)

        # Compress if beneficial
        compressed = False
        stored_value = value
        size_bytes = len(pickle.dumps(value))

        if (
            self.config.compression_enabled
            and size_bytes > self.config.compression_threshold_bytes
        ):
            compressed_value = self._compress(value)
            compressed_size = len(compressed_value)
            if compressed_size < size_bytes * 0.8:  # At least 20% savings
                stored_value = compressed_value
                compressed = True
                self._compression_savings_bytes += size_bytes - compressed_size

        # Create entry
        now = datetime.utcnow()
        existing = self._cache.get(key)
        version = (existing.version + 1) if existing else 1

        entry = CacheEntry(
            key=key,
            value=stored_value,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            ttl_seconds=base_ttl,
            version=version,
            sync_state=SyncState.PENDING if write_through else SyncState.SYNCED,
            checksum=self._calculate_checksum(stored_value),
            compressed=compressed,
            size_bytes=len(pickle.dumps(stored_value)),
            origin_region=self.region_id,
        )

        self._cache[key] = entry

        # Persist locally
        if self.config.persistence_enabled:
            self._save_to_persistence(entry)

        # Queue for sync
        if write_through:
            self._pending_sync.add(key)
            if not self._is_online:
                self._queue_offline_request("set", key, value)

    def delete(self, key: str, sync: bool = True):
        """
        Delete value from cache.

        Args:
            key: Cache key
            sync: Whether to sync deletion
        """
        if key in self._cache:
            del self._cache[key]

        # Remove from persistence
        if self.config.persistence_enabled:
            self._delete_from_persistence(key)

        # Queue for sync
        if sync:
            if not self._is_online:
                self._queue_offline_request("delete", key)
            else:
                self._pending_sync.add(key)

    def invalidate_pattern(self, pattern: str):
        """
        Invalidate keys matching pattern.

        Args:
            pattern: Pattern to match (simple prefix match)
        """
        keys_to_delete = [k for k in self._cache.keys() if k.startswith(pattern)]
        for key in keys_to_delete:
            self.delete(key, sync=True)
        logger.debug(f"Invalidated {len(keys_to_delete)} keys matching '{pattern}'")

    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._pending_sync.clear()
        if self.config.persistence_enabled:
            for file in self._persistence_path.glob("*.cache"):
                file.unlink()
        logger.info("Satellite cache cleared")

    async def sync_pending(self) -> int:
        """
        Sync all pending changes.

        Returns:
            Number of entries synced
        """
        if not self._is_online:
            logger.warning("Cannot sync: offline")
            return 0

        synced = 0
        keys_to_sync = list(self._pending_sync)

        for key in keys_to_sync:
            entry = self._cache.get(key)
            if entry:
                try:
                    # In a real implementation, this would sync to remote
                    entry.sync_state = SyncState.SYNCED
                    self._pending_sync.discard(key)
                    synced += 1
                except Exception as e:
                    logger.error(f"Sync failed for {key}: {e}")
                    entry.sync_state = SyncState.ERROR

        # Process offline queue
        synced += await self._process_offline_queue()

        self._last_sync = datetime.utcnow()
        self._sync_operations += 1

        logger.info(f"Synced {synced} cache entries")
        return synced

    async def _process_offline_queue(self) -> int:
        """Process queued offline requests."""
        if not self._offline_queue:
            return 0

        processed = 0
        remaining = []

        for request in self._offline_queue:
            try:
                if request.operation == "set":
                    # Re-apply the set operation
                    if request.key in self._cache:
                        self._cache[request.key].sync_state = SyncState.SYNCED
                        processed += 1
                elif request.operation == "delete":
                    # Confirm deletion
                    processed += 1
            except Exception as e:
                logger.error(f"Failed to process offline request: {e}")
                request.retry_count += 1
                if request.retry_count < request.max_retries:
                    remaining.append(request)

        self._offline_queue = remaining
        return processed

    def _queue_offline_request(
        self,
        operation: str,
        key: str,
        value: Optional[Any] = None,
        priority: int = 0,
    ):
        """Queue a request for offline processing."""
        if len(self._offline_queue) >= self.config.offline_queue_max_size:
            # Remove oldest low-priority request
            self._offline_queue.sort(key=lambda r: (r.priority, r.timestamp))
            self._offline_queue.pop(0)

        request = OfflineRequest(
            request_id=f"{key}_{int(time.time()*1000)}",
            operation=operation,
            key=key,
            value=value,
            priority=priority,
        )
        self._offline_queue.append(request)

        if self.config.offline_queue_persist:
            self._persist_offline_queue()

    def _persist_offline_queue(self):
        """Persist offline queue to disk."""
        if not self.config.persistence_enabled:
            return

        queue_path = self._persistence_path / "offline_queue.json"
        try:
            queue_data = [
                {
                    "request_id": r.request_id,
                    "operation": r.operation,
                    "key": r.key,
                    "timestamp": r.timestamp.isoformat(),
                    "priority": r.priority,
                }
                for r in self._offline_queue
            ]
            queue_path.write_text(json.dumps(queue_data))
        except Exception as e:
            logger.error(f"Failed to persist offline queue: {e}")

    def _load_offline_queue(self):
        """Load offline queue from disk."""
        if not self.config.persistence_enabled:
            return

        queue_path = self._persistence_path / "offline_queue.json"
        if not queue_path.exists():
            return

        try:
            queue_data = json.loads(queue_path.read_text())
            self._offline_queue = [
                OfflineRequest(
                    request_id=r["request_id"],
                    operation=r["operation"],
                    key=r["key"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    priority=r["priority"],
                )
                for r in queue_data
            ]
        except Exception as e:
            logger.error(f"Failed to load offline queue: {e}")

    def resolve_conflict(
        self,
        local_entry: CacheEntry,
        remote_entry: CacheEntry,
    ) -> CacheEntry:
        """
        Resolve conflict between local and remote entries.

        Args:
            local_entry: Local cache entry
            remote_entry: Remote cache entry

        Returns:
            Resolved entry
        """
        strategy = self.config.conflict_resolution

        if strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            return (
                local_entry
                if local_entry.updated_at > remote_entry.updated_at
                else remote_entry
            )

        elif strategy == ConflictResolutionStrategy.FIRST_WRITE_WINS:
            return (
                local_entry
                if local_entry.created_at < remote_entry.created_at
                else remote_entry
            )

        elif strategy == ConflictResolutionStrategy.CALLBACK:
            if self._conflict_callback:
                return self._conflict_callback(local_entry, remote_entry)
            # Fall back to last-write-wins
            return (
                local_entry
                if local_entry.updated_at > remote_entry.updated_at
                else remote_entry
            )

        # Default: last-write-wins
        return (
            local_entry
            if local_entry.updated_at > remote_entry.updated_at
            else remote_entry
        )

    def set_conflict_callback(self, callback: Callable):
        """Set callback for conflict resolution."""
        self._conflict_callback = callback

    def _compress(self, value: Any) -> bytes:
        """Compress value for storage."""
        data = pickle.dumps(value)
        return gzip.compress(data, compresslevel=self.config.compression_level)

    def _decompress(self, data: bytes) -> Any:
        """Decompress stored value."""
        return pickle.loads(gzip.decompress(data))

    def _calculate_checksum(self, value: Any) -> str:
        """Calculate checksum for value."""
        data = pickle.dumps(value)
        return hashlib.sha256(data).hexdigest()[:16]

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired."""
        expiry = entry.updated_at + timedelta(seconds=entry.ttl_seconds)
        return datetime.utcnow() > expiry

    def _save_to_persistence(self, entry: CacheEntry):
        """Save entry to persistent storage."""
        if not self.config.persistence_enabled:
            return

        try:
            file_path = self._persistence_path / f"{entry.key}.cache"
            with open(file_path, "wb") as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.error(f"Failed to persist cache entry {entry.key}: {e}")

    def _load_from_persistence(self, key: str) -> Optional[CacheEntry]:
        """Load entry from persistent storage."""
        if not self.config.persistence_enabled:
            return None

        try:
            file_path = self._persistence_path / f"{key}.cache"
            if not file_path.exists():
                return None

            with open(file_path, "rb") as f:
                entry = pickle.load(f)

            if self._is_expired(entry):
                file_path.unlink()
                return None

            return entry
        except Exception as e:
            logger.error(f"Failed to load cache entry {key}: {e}")
            return None

    def _delete_from_persistence(self, key: str):
        """Delete entry from persistent storage."""
        if not self.config.persistence_enabled:
            return

        try:
            file_path = self._persistence_path / f"{key}.cache"
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to delete persisted cache entry {key}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        total_size = sum(e.size_bytes for e in self._cache.values())
        pending_count = len(self._pending_sync)
        offline_queue_size = len(self._offline_queue)

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "entries": len(self._cache),
            "total_size_bytes": total_size,
            "pending_sync": pending_count,
            "offline_queue_size": offline_queue_size,
            "sync_operations": self._sync_operations,
            "compression_savings_bytes": self._compression_savings_bytes,
            "is_online": self._is_online,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "region": self.region_id,
        }

    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status."""
        synced = sum(
            1 for e in self._cache.values() if e.sync_state == SyncState.SYNCED
        )
        pending = sum(
            1 for e in self._cache.values() if e.sync_state == SyncState.PENDING
        )
        conflict = sum(
            1 for e in self._cache.values() if e.sync_state == SyncState.CONFLICT
        )
        error = sum(
            1 for e in self._cache.values() if e.sync_state == SyncState.ERROR
        )

        return {
            "synced": synced,
            "pending": pending,
            "conflict": conflict,
            "error": error,
            "offline_queue": len(self._offline_queue),
            "is_online": self._is_online,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
        }
