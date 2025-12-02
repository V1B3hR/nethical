"""
Sync Manager - Reconnection Sync Logic

Manages synchronization when connectivity is restored.
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SyncStatus(str, Enum):
    """Sync status states."""

    IDLE = "idle"
    SYNCING = "syncing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class SyncResult:
    """
    Result of a sync operation.

    Attributes:
        status: Sync status
        synced_count: Number of items synced
        failed_count: Number of items failed
        duration_seconds: Sync duration
        error: Error message if failed
    """

    status: SyncStatus
    synced_count: int = 0
    failed_count: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None


class SyncManager:
    """
    Reconnection sync manager.

    Handles synchronization when connectivity is restored:
    - Queue non-critical updates
    - Prioritize safety-critical sync
    - Delta updates only
    """

    def __init__(
        self,
        decision_queue: Optional["DecisionQueue"] = None,
        network_monitor: Optional["NetworkMonitor"] = None,
        batch_size: int = 100,
        max_retries: int = 3,
        sync_callback: Optional[Callable[[List[Any]], bool]] = None,
    ):
        """
        Initialize SyncManager.

        Args:
            decision_queue: Queue of offline decisions
            network_monitor: Network connectivity monitor
            batch_size: Number of items per sync batch
            max_retries: Maximum retry attempts
            sync_callback: Callback for syncing items
        """
        from .decision_queue import DecisionQueue
        from .network_monitor import NetworkMonitor

        self.decision_queue = decision_queue or DecisionQueue()
        self.network_monitor = network_monitor or NetworkMonitor()
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.sync_callback = sync_callback

        # State
        self._status = SyncStatus.IDLE
        self._lock = threading.RLock()
        self._last_sync: Optional[float] = None

        # Metrics
        self._total_syncs = 0
        self._total_synced_items = 0
        self._total_failed_items = 0

        logger.info("SyncManager initialized")

    @property
    def status(self) -> SyncStatus:
        """Get current sync status."""
        return self._status

    def sync_now(self, force: bool = False) -> SyncResult:
        """
        Perform synchronization now.

        Args:
            force: Force sync even if network is down

        Returns:
            SyncResult with outcome
        """
        start_time = time.time()

        with self._lock:
            if self._status == SyncStatus.SYNCING:
                return SyncResult(
                    status=SyncStatus.SYNCING, error="Sync already in progress"
                )

            self._status = SyncStatus.SYNCING

        try:
            # Check network
            if not force and not self.network_monitor.is_connected():
                with self._lock:
                    self._status = SyncStatus.FAILED
                return SyncResult(status=SyncStatus.FAILED, error="Network unavailable")

            # Get unsynced decisions
            unsynced = self.decision_queue.get_unsynced(limit=self.batch_size)

            if not unsynced:
                with self._lock:
                    self._status = SyncStatus.COMPLETE
                return SyncResult(status=SyncStatus.COMPLETE)

            # Sync in batches
            synced_count = 0
            failed_count = 0

            for i in range(0, len(unsynced), self.batch_size):
                batch = unsynced[i : i + self.batch_size]

                if self._sync_batch(batch):
                    synced_count += len(batch)
                    self.decision_queue.mark_synced(batch)
                else:
                    failed_count += len(batch)

            # Update metrics
            self._total_syncs += 1
            self._total_synced_items += synced_count
            self._total_failed_items += failed_count
            self._last_sync = time.time()

            # Cleanup synced
            if synced_count > 0:
                self.decision_queue.remove_synced()

            with self._lock:
                self._status = (
                    SyncStatus.COMPLETE if failed_count == 0 else SyncStatus.FAILED
                )

            return SyncResult(
                status=self._status,
                synced_count=synced_count,
                failed_count=failed_count,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Sync failed: {e}")
            with self._lock:
                self._status = SyncStatus.FAILED
            return SyncResult(
                status=SyncStatus.FAILED,
                duration_seconds=time.time() - start_time,
                error=str(e),
            )

    def _sync_batch(self, batch: List[Any]) -> bool:
        """
        Sync a batch of decisions.

        Args:
            batch: Batch of decisions to sync

        Returns:
            True if successful
        """
        if self.sync_callback:
            try:
                return self.sync_callback(batch)
            except Exception as e:
                logger.error(f"Sync callback failed: {e}")
                return False

        # Default: just mark as synced (no remote sync configured)
        return True

    def sync_with_retry(self) -> SyncResult:
        """
        Perform sync with retries.

        Returns:
            SyncResult with outcome
        """
        last_result = SyncResult(status=SyncStatus.IDLE)

        for attempt in range(self.max_retries):
            result = self.sync_now()

            if result.status == SyncStatus.COMPLETE:
                return result

            last_result = result

            if attempt < self.max_retries - 1:
                # Exponential backoff
                time.sleep(2**attempt)

        return last_result

    def get_pending_count(self) -> int:
        """Get count of pending sync items."""
        return len(self.decision_queue.get_unsynced())

    def get_metrics(self) -> Dict[str, Any]:
        """Get sync metrics."""
        return {
            "status": self._status.value,
            "last_sync": self._last_sync,
            "total_syncs": self._total_syncs,
            "total_synced_items": self._total_synced_items,
            "total_failed_items": self._total_failed_items,
            "pending_items": self.get_pending_count(),
        }


# Import dependencies for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .decision_queue import DecisionQueue
    from .network_monitor import NetworkMonitor
