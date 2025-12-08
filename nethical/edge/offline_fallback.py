"""
Offline Fallback - Graceful Degradation When Disconnected

Manages offline mode and safe fallback behavior when network is unavailable.
Philosophy: "Safe by default when disconnected"
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OfflineMode(str, Enum):
    """Offline mode states."""

    ONLINE = "online"
    PARTIAL = "partial"  # Degraded connectivity
    OFFLINE = "offline"


@dataclass
class OfflineConfig:
    """
    Offline mode configuration.

    Attributes:
        max_offline_hours: Maximum hours to operate offline
        conservative_mode: Use conservative risk thresholds when offline
        allow_cached_only: Only allow cached decisions when offline
        sync_on_reconnect: Sync queued decisions on reconnect
    """

    max_offline_hours: float = 24.0
    conservative_mode: bool = True
    allow_cached_only: bool = False
    sync_on_reconnect: bool = True


class OfflineFallback:
    """
    Offline fallback system.

    Manages behavior when network connectivity is lost.

    Features:
    - Use last-known-good policies
    - Apply conservative risk thresholds
    - Log decisions for later sync
    - Graceful degradation

    Safety Guarantees:
    - Never allow blocked-by-policy actions offline
    - Default to RESTRICT for uncertain actions
    - TERMINATE always available locally
    """

    def __init__(
        self,
        config: Optional[OfflineConfig] = None,
        network_monitor: Optional["NetworkMonitor"] = None,
        decision_queue: Optional["DecisionQueue"] = None,
    ):
        """
        Initialize OfflineFallback.

        Args:
            config: Offline configuration
            network_monitor: Network connectivity monitor
            decision_queue: Queue for offline decisions
        """
        self.config = config or OfflineConfig()

        # Import here to avoid circular imports
        from .network_monitor import NetworkMonitor
        from .decision_queue import DecisionQueue

        self.network_monitor = network_monitor or NetworkMonitor()
        self.decision_queue = decision_queue or DecisionQueue()

        # State
        self._current_mode = OfflineMode.ONLINE
        self._offline_since: Optional[float] = None
        self._lock = threading.RLock()

        # Last known good state
        self._last_policy_hash: Optional[str] = None
        self._last_policy_timestamp: Optional[float] = None

        # Metrics
        self._offline_decisions = 0
        self._fallback_decisions = 0

        logger.info("OfflineFallback initialized")

    @property
    def current_mode(self) -> OfflineMode:
        """Get current offline mode."""
        return self._current_mode

    @property
    def is_online(self) -> bool:
        """Check if currently online."""
        return self._current_mode == OfflineMode.ONLINE

    @property
    def is_offline(self) -> bool:
        """Check if currently offline."""
        return self._current_mode == OfflineMode.OFFLINE

    def update_mode(self, force_check: bool = False) -> OfflineMode:
        """
        Update offline mode based on network status.

        Args:
            force_check: Force network check

        Returns:
            Current mode
        """
        with self._lock:
            status = self.network_monitor.get_status(force_check)

            if status.is_connected and status.latency_ms is not None:
                if status.latency_ms < 100:
                    new_mode = OfflineMode.ONLINE
                else:
                    new_mode = OfflineMode.PARTIAL
            else:
                new_mode = OfflineMode.OFFLINE

            # Track offline duration
            if (
                new_mode == OfflineMode.OFFLINE
                and self._current_mode != OfflineMode.OFFLINE
            ):
                self._offline_since = time.time()
                logger.warning("Entering offline mode")
            elif (
                new_mode == OfflineMode.ONLINE
                and self._current_mode == OfflineMode.OFFLINE
            ):
                if self._offline_since:
                    duration = time.time() - self._offline_since
                    logger.info(f"Returning online after {duration:.1f}s offline")
                self._offline_since = None

            self._current_mode = new_mode
            return new_mode

    def get_offline_duration(self) -> Optional[float]:
        """Get how long system has been offline in seconds."""
        if self._offline_since is None:
            return None
        return time.time() - self._offline_since

    def should_use_fallback(self) -> bool:
        """
        Check if fallback behavior should be used.

        Returns:
            True if should use fallback
        """
        self.update_mode()

        if self._current_mode == OfflineMode.OFFLINE:
            return True

        if self._current_mode == OfflineMode.PARTIAL and self.config.conservative_mode:
            return True

        return False

    def get_risk_threshold_adjustment(self) -> float:
        """
        Get risk threshold adjustment for offline mode.

        Returns conservative adjustments when offline.

        Returns:
            Multiplier for risk thresholds (1.0 = no change)
        """
        if self._current_mode == OfflineMode.ONLINE:
            return 1.0

        if not self.config.conservative_mode:
            return 1.0

        if self._current_mode == OfflineMode.PARTIAL:
            return 1.2  # 20% more conservative

        # Fully offline - progressively more conservative
        duration = self.get_offline_duration() or 0
        hours_offline = duration / 3600

        # Cap at 2x after 12 hours
        adjustment = 1.5 + min(0.5, hours_offline / 24)
        return adjustment

    def record_offline_decision(
        self,
        agent_id: str,
        action: str,
        decision: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a decision made while offline.

        Args:
            agent_id: Agent identifier
            action: The action content
            decision: Decision made
            context: Decision context
        """
        self._offline_decisions += 1

        from .decision_queue import QueuedDecision

        queued = QueuedDecision(
            agent_id=agent_id,
            action=action,
            decision=decision,
            context=context or {},
            timestamp=time.time(),
            offline_mode=self._current_mode.value,
        )

        self.decision_queue.enqueue(queued)

    def record_fallback_decision(self):
        """Record that a fallback decision was made."""
        self._fallback_decisions += 1

    def set_last_known_good_policy(self, policy_hash: str):
        """
        Set the last known good policy state.

        Args:
            policy_hash: Hash of the last valid policy
        """
        with self._lock:
            self._last_policy_hash = policy_hash
            self._last_policy_timestamp = time.time()

    def is_policy_stale(self, max_age_hours: float = 24.0) -> bool:
        """
        Check if last known policy is too old.

        Args:
            max_age_hours: Maximum acceptable age in hours

        Returns:
            True if policy is stale
        """
        if self._last_policy_timestamp is None:
            return True

        age_hours = (time.time() - self._last_policy_timestamp) / 3600
        return age_hours > max_age_hours

    def get_metrics(self) -> Dict[str, Any]:
        """Get offline fallback metrics."""
        return {
            "current_mode": self._current_mode.value,
            "offline_duration_seconds": self.get_offline_duration(),
            "offline_decisions": self._offline_decisions,
            "fallback_decisions": self._fallback_decisions,
            "queued_decisions": len(self.decision_queue),
            "last_policy_timestamp": self._last_policy_timestamp,
            "policy_stale": self.is_policy_stale(),
        }


# Import dependencies for type hints only
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .network_monitor import NetworkMonitor
    from .decision_queue import DecisionQueue, QueuedDecision
