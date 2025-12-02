"""
Cache Invalidation Manager

Manages cache invalidation across all levels.

Features:
- Targeted invalidation
- Emergency flush
- Event-driven invalidation
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class InvalidationType(str, Enum):
    """Types of invalidation events."""

    POLICY_UPDATE = "policy_update"
    AGENT_SUSPENSION = "agent_suspension"
    SECURITY_EVENT = "security_event"
    EMERGENCY_FLUSH = "emergency_flush"
    PATTERN_MATCH = "pattern_match"


@dataclass
class InvalidationEvent:
    """
    Cache invalidation event.

    Attributes:
        event_type: Type of invalidation
        pattern: Pattern to invalidate (optional)
        keys: Specific keys to invalidate (optional)
        timestamp: Event timestamp
        source: Source of invalidation
        propagate: Whether to propagate to other regions
    """

    event_type: InvalidationType
    pattern: Optional[str] = None
    keys: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    source: str = "local"
    propagate: bool = True


class InvalidationManager:
    """
    Cache invalidation manager.

    Handles:
    - Policy update → Invalidate all policy caches
    - Agent suspension → Targeted invalidation
    - Security event → Emergency flush

    Propagation:
    - L1: In-process event
    - L2: Redis pub/sub
    - L3: Global event stream
    """

    def __init__(
        self,
        cache_hierarchy: Optional["CacheHierarchy"] = None,
        event_propagation: Optional["EventPropagation"] = None,
    ):
        """
        Initialize InvalidationManager.

        Args:
            cache_hierarchy: Cache hierarchy to invalidate
            event_propagation: Event propagation system
        """
        self.cache_hierarchy = cache_hierarchy
        self.event_propagation = event_propagation

        # Subscribers
        self._subscribers: List[Callable[[InvalidationEvent], None]] = []
        self._lock = threading.RLock()

        # Metrics
        self._total_invalidations = 0
        self._emergency_flushes = 0

        logger.info("InvalidationManager initialized")

    def invalidate(self, event: InvalidationEvent):
        """
        Process an invalidation event.

        Args:
            event: Invalidation event
        """
        with self._lock:
            self._total_invalidations += 1

            if event.event_type == InvalidationType.EMERGENCY_FLUSH:
                self._emergency_flushes += 1
                self._emergency_flush()
            elif event.keys:
                self._invalidate_keys(event.keys)
            elif event.pattern:
                self._invalidate_pattern(event.pattern)

            # Notify subscribers
            for subscriber in self._subscribers:
                try:
                    subscriber(event)
                except Exception as e:
                    logger.error(f"Subscriber error: {e}")

            # Propagate if needed
            if event.propagate and self.event_propagation:
                self.event_propagation.publish(event)

    def invalidate_policy(self, policy_id: str):
        """
        Invalidate all caches for a policy.

        Args:
            policy_id: Policy to invalidate
        """
        event = InvalidationEvent(
            event_type=InvalidationType.POLICY_UPDATE,
            pattern=f"policy:{policy_id}",
        )
        self.invalidate(event)

    def invalidate_agent(self, agent_id: str):
        """
        Invalidate all caches for an agent.

        Args:
            agent_id: Agent to invalidate
        """
        event = InvalidationEvent(
            event_type=InvalidationType.AGENT_SUSPENSION,
            pattern=f"agent:{agent_id}",
        )
        self.invalidate(event)

    def security_flush(self, reason: str = "security_event"):
        """
        Emergency security flush of all caches.

        Args:
            reason: Reason for flush
        """
        logger.warning(f"Security flush triggered: {reason}")
        event = InvalidationEvent(
            event_type=InvalidationType.EMERGENCY_FLUSH,
            source=reason,
        )
        self.invalidate(event)

    def _invalidate_keys(self, keys: List[str]):
        """Invalidate specific keys."""
        if self.cache_hierarchy:
            for key in keys:
                self.cache_hierarchy.delete(key)

    def _invalidate_pattern(self, pattern: str):
        """Invalidate keys matching pattern."""
        if self.cache_hierarchy:
            self.cache_hierarchy.invalidate_pattern(pattern)

    def _emergency_flush(self):
        """Perform emergency cache flush."""
        logger.warning("Emergency cache flush in progress")
        if self.cache_hierarchy:
            self.cache_hierarchy.clear()

    def subscribe(self, callback: Callable[[InvalidationEvent], None]):
        """
        Subscribe to invalidation events.

        Args:
            callback: Function to call on event
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[InvalidationEvent], None]):
        """
        Unsubscribe from invalidation events.

        Args:
            callback: Function to remove
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def get_metrics(self) -> Dict[str, Any]:
        """Get invalidation metrics."""
        return {
            "total_invalidations": self._total_invalidations,
            "emergency_flushes": self._emergency_flushes,
            "subscribers": len(self._subscribers),
        }


# Type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache_hierarchy import CacheHierarchy
    from .event_propagation import EventPropagation
