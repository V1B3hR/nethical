"""
Policy Subscriber - Policy Update Listener

Listens for policy updates from event stream.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PolicyEventType(str, Enum):
    """Policy event types."""

    CREATED = "created"
    UPDATED = "updated"
    DEPRECATED = "deprecated"
    ACTIVATED = "activated"


@dataclass
class PolicyUpdate:
    """
    Policy update event.

    Attributes:
        policy_id: Policy identifier
        event_type: Type of update
        version: Policy version
        content: Policy content (for create/update)
        timestamp: Event timestamp
        source: Update source
    """

    policy_id: str
    event_type: PolicyEventType
    version: str = "1.0"
    content: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"


class PolicySubscriber:
    """
    Policy update subscriber.

    Listens to policy events:
    - nethical.policy.*.created
    - nethical.policy.*.updated
    - nethical.policy.*.deprecated
    - nethical.policy.*.activated

    Features:
    - Async event handling
    - Policy cache invalidation
    - Change notification
    """

    SUBJECTS = [
        "nethical.policy.*.created",
        "nethical.policy.*.updated",
        "nethical.policy.*.deprecated",
        "nethical.policy.*.activated",
    ]

    def __init__(
        self,
        nats_client: Optional["NATSClient"] = None,
        cache_hierarchy: Optional[Any] = None,
    ):
        """
        Initialize PolicySubscriber.

        Args:
            nats_client: NATS client for streaming
            cache_hierarchy: Cache hierarchy for invalidation
        """
        self.nats_client = nats_client
        self.cache_hierarchy = cache_hierarchy

        # Update handlers
        self._handlers: List[Callable[[PolicyUpdate], None]] = []
        self._lock = threading.RLock()

        # Update history
        self._update_history: List[PolicyUpdate] = []
        self._max_history = 1000

        # Metrics
        self._updates_received = 0

        logger.info("PolicySubscriber initialized")

    async def start(self):
        """Start listening for policy updates."""
        if not self.nats_client:
            logger.warning("No NATS client configured")
            return

        for subject in self.SUBJECTS:
            await self.nats_client.subscribe(
                subject,
                self._handle_message,
                durable=f"policy-subscriber-{subject.replace('.', '-')}",
            )

        logger.info("PolicySubscriber started")

    def _handle_message(self, message: Dict[str, Any]):
        """
        Handle incoming policy message.

        Args:
            message: Message payload
        """
        try:
            # Parse update
            update = PolicyUpdate(
                policy_id=message.get("policy_id", "unknown"),
                event_type=PolicyEventType(message.get("event_type", "updated")),
                version=message.get("version", "1.0"),
                content=message.get("content"),
                timestamp=message.get("timestamp", time.time()),
                source=message.get("source", "unknown"),
            )

            self._updates_received += 1

            # Store in history
            with self._lock:
                self._update_history.append(update)
                if len(self._update_history) > self._max_history:
                    self._update_history.pop(0)

            # Invalidate cache
            self._invalidate_cache(update)

            # Notify handlers
            self._notify_handlers(update)

        except Exception as e:
            logger.error(f"Error handling policy message: {e}")

    def _invalidate_cache(self, update: PolicyUpdate):
        """Invalidate cache for policy update."""
        if not self.cache_hierarchy:
            return

        try:
            self.cache_hierarchy.invalidate_pattern(f"policy:{update.policy_id}")
            logger.debug(f"Cache invalidated for policy: {update.policy_id}")
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")

    def _notify_handlers(self, update: PolicyUpdate):
        """Notify registered handlers."""
        with self._lock:
            for handler in self._handlers:
                try:
                    handler(update)
                except Exception as e:
                    logger.error(f"Handler error: {e}")

    def on_update(self, handler: Callable[[PolicyUpdate], None]):
        """
        Register update handler.

        Args:
            handler: Function to call on update
        """
        with self._lock:
            self._handlers.append(handler)

    def remove_handler(self, handler: Callable[[PolicyUpdate], None]):
        """
        Remove update handler.

        Args:
            handler: Handler to remove
        """
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)

    def get_recent_updates(
        self,
        policy_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[PolicyUpdate]:
        """
        Get recent policy updates.

        Args:
            policy_id: Filter by policy ID (optional)
            limit: Maximum updates to return

        Returns:
            List of recent updates
        """
        with self._lock:
            if policy_id:
                updates = [u for u in self._update_history if u.policy_id == policy_id]
            else:
                updates = self._update_history.copy()

            return updates[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get subscriber metrics."""
        return {
            "updates_received": self._updates_received,
            "handlers_registered": len(self._handlers),
            "history_size": len(self._update_history),
        }


# Type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nats_client import NATSClient
