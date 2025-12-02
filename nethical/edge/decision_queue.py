"""
Decision Queue - Offline Decision Logging

Queues decisions made while offline for later sync.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueuedDecision:
    """
    A decision queued for later sync.

    Attributes:
        agent_id: Agent that made the decision
        action: The action content
        decision: Decision type
        context: Action context
        timestamp: When decision was made
        offline_mode: Offline mode at time of decision
        synced: Whether decision has been synced
        sync_attempts: Number of sync attempts
    """

    agent_id: str
    action: str
    decision: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    offline_mode: str = "offline"
    synced: bool = False
    sync_attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "action": self.action,
            "decision": self.decision,
            "context": self.context,
            "timestamp": self.timestamp,
            "offline_mode": self.offline_mode,
            "synced": self.synced,
            "sync_attempts": self.sync_attempts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueuedDecision":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            action=data["action"],
            decision=data["decision"],
            context=data.get("context", {}),
            timestamp=data.get("timestamp", time.time()),
            offline_mode=data.get("offline_mode", "offline"),
            synced=data.get("synced", False),
            sync_attempts=data.get("sync_attempts", 0),
        )


class DecisionQueue:
    """
    Queue for offline decisions.

    Features:
    - FIFO queue for decisions
    - Persistence to disk
    - Automatic cleanup
    - Sync management
    """

    def __init__(
        self,
        max_size: int = 100000,
        persist_path: Optional[str] = None,
        persist_interval_seconds: int = 60,
    ):
        """
        Initialize DecisionQueue.

        Args:
            max_size: Maximum queue size
            persist_path: Path for persistence (optional)
            persist_interval_seconds: Interval for auto-persist
        """
        self.max_size = max_size
        self.persist_path = persist_path
        self.persist_interval_seconds = persist_interval_seconds

        # Queue storage
        self._queue: Deque[QueuedDecision] = deque(maxlen=max_size)
        self._lock = threading.RLock()

        # Metrics
        self._total_enqueued = 0
        self._total_synced = 0
        self._dropped = 0

        # Load persisted queue
        if persist_path and os.path.exists(persist_path):
            self._load_from_disk()

        logger.info(f"DecisionQueue initialized: max_size={max_size}")

    def enqueue(self, decision: QueuedDecision):
        """
        Add decision to queue.

        Args:
            decision: Decision to queue
        """
        with self._lock:
            if len(self._queue) >= self.max_size:
                # Drop oldest unsynced
                self._dropped += 1
                logger.warning("Decision queue full, dropping oldest")

            self._queue.append(decision)
            self._total_enqueued += 1

    def dequeue(self) -> Optional[QueuedDecision]:
        """
        Remove and return oldest decision.

        Returns:
            Oldest decision or None if empty
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def peek(self) -> Optional[QueuedDecision]:
        """
        View oldest decision without removing.

        Returns:
            Oldest decision or None if empty
        """
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    def get_unsynced(self, limit: int = 100) -> List[QueuedDecision]:
        """
        Get unsynced decisions.

        Args:
            limit: Maximum number to return

        Returns:
            List of unsynced decisions
        """
        with self._lock:
            unsynced = [d for d in self._queue if not d.synced]
            return unsynced[:limit]

    def mark_synced(self, decisions: List[QueuedDecision]):
        """
        Mark decisions as synced.

        Args:
            decisions: Decisions to mark
        """
        synced_timestamps = {d.timestamp for d in decisions}

        with self._lock:
            for d in self._queue:
                if d.timestamp in synced_timestamps:
                    d.synced = True
                    self._total_synced += 1

    def remove_synced(self):
        """Remove all synced decisions from queue."""
        with self._lock:
            self._queue = deque(
                [d for d in self._queue if not d.synced], maxlen=self.max_size
            )

    def clear(self):
        """Clear all decisions from queue."""
        with self._lock:
            self._queue.clear()

    def _persist_to_disk(self):
        """Persist queue to disk."""
        if not self.persist_path:
            return

        try:
            with self._lock:
                data = [d.to_dict() for d in self._queue]

            with open(self.persist_path, "w") as f:
                json.dump(data, f)

            logger.debug(f"Persisted {len(data)} decisions to disk")

        except Exception as e:
            logger.error(f"Failed to persist queue: {e}")

    def _load_from_disk(self):
        """Load queue from disk."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return

        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)

            with self._lock:
                for item in data:
                    decision = QueuedDecision.from_dict(item)
                    self._queue.append(decision)

            logger.info(f"Loaded {len(data)} decisions from disk")

        except Exception as e:
            logger.error(f"Failed to load queue: {e}")

    def persist(self):
        """Force persist to disk."""
        self._persist_to_disk()

    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics."""
        with self._lock:
            unsynced = sum(1 for d in self._queue if not d.synced)
            return {
                "size": len(self._queue),
                "max_size": self.max_size,
                "unsynced": unsynced,
                "synced": len(self._queue) - unsynced,
                "total_enqueued": self._total_enqueued,
                "total_synced": self._total_synced,
                "dropped": self._dropped,
            }

    def __len__(self) -> int:
        """Return queue size."""
        return len(self._queue)
