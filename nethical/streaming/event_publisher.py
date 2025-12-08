"""
Event Publisher - Event Publishing

Publishes events to the streaming system.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Stream event types."""

    # Policy events
    POLICY_CREATED = "policy.created"
    POLICY_UPDATED = "policy.updated"
    POLICY_DEPRECATED = "policy.deprecated"

    # Agent events
    AGENT_SUSPENDED = "agent.suspended"
    AGENT_QUOTA_EXCEEDED = "agent.quota_exceeded"
    AGENT_DECISION = "agent.decision"

    # Security events
    SECURITY_BREACH = "security.breach"
    SECURITY_EMERGENCY_FLUSH = "security.emergency_flush"

    # System events
    SYSTEM_HEALTH = "system.health"
    SYSTEM_METRIC = "system.metric"


@dataclass
class StreamEvent:
    """
    Event for streaming.

    Attributes:
        event_type: Type of event
        subject: Stream subject
        payload: Event payload
        timestamp: Event timestamp
        source: Event source
        priority: Event priority (high, normal, low)
    """

    event_type: StreamEventType
    subject: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = "local"
    priority: str = "normal"


class EventPublisher:
    """
    Event publisher for streaming.

    Publishes events to:
    - Policy updates stream
    - Agent events stream
    - Security events stream
    """

    def __init__(
        self,
        nats_client: Optional["NATSClient"] = None,
        batch_size: int = 100,
        flush_interval: float = 1.0,
    ):
        """
        Initialize EventPublisher.

        Args:
            nats_client: NATS client for publishing
            batch_size: Maximum batch size
            flush_interval: Auto-flush interval in seconds
        """
        self.nats_client = nats_client
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Event buffer for batching
        self._buffer: List[StreamEvent] = []
        self._lock = threading.RLock()

        # Background flusher
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None

        # Metrics
        self._events_published = 0
        self._events_buffered = 0
        self._batches_flushed = 0

        logger.info("EventPublisher initialized")

    def start(self):
        """Start background event flushing."""
        if self._running:
            return

        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        logger.info("EventPublisher started")

    def stop(self):
        """Stop background event flushing."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None

        # Final flush
        self._flush_buffer()
        logger.info("EventPublisher stopped")

    def _flush_loop(self):
        """Background flush loop."""
        while self._running:
            time.sleep(self.flush_interval)
            self._flush_buffer()

    def publish(self, event: StreamEvent):
        """
        Publish an event.

        Args:
            event: Event to publish
        """
        with self._lock:
            self._buffer.append(event)
            self._events_buffered += 1

            if len(self._buffer) >= self.batch_size:
                self._flush_buffer()

    def publish_immediate(self, event: StreamEvent):
        """
        Publish an event immediately (no buffering).

        Args:
            event: Event to publish
        """
        self._publish_event(event)

    def _flush_buffer(self):
        """Flush event buffer."""
        with self._lock:
            if not self._buffer:
                return

            events = self._buffer
            self._buffer = []

        for event in events:
            self._publish_event(event)

        self._batches_flushed += 1

    def _publish_event(self, event: StreamEvent):
        """Publish a single event."""
        self._events_published += 1

        message = {
            "event_type": event.event_type.value,
            "payload": event.payload,
            "timestamp": event.timestamp,
            "source": event.source,
            "priority": event.priority,
        }

        if self.nats_client:
            # Use async in background
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self.nats_client.publish(event.subject, message))
            except RuntimeError:
                # No event loop, publish sync
                asyncio.run(self.nats_client.publish(event.subject, message))

    # Convenience methods for common events

    def publish_policy_update(
        self,
        policy_id: str,
        event_type: str = "updated",
        content: Optional[Dict] = None,
    ):
        """Publish a policy update event."""
        event = StreamEvent(
            event_type=StreamEventType.POLICY_UPDATED,
            subject=f"nethical.policy.{policy_id}.{event_type}",
            payload={
                "policy_id": policy_id,
                "event_type": event_type,
                "content": content,
            },
        )
        self.publish(event)

    def publish_agent_decision(
        self,
        agent_id: str,
        decision: str,
        risk_score: float,
        action_type: str,
    ):
        """Publish an agent decision event."""
        event = StreamEvent(
            event_type=StreamEventType.AGENT_DECISION,
            subject=f"nethical.agent.{agent_id}.decision",
            payload={
                "agent_id": agent_id,
                "decision": decision,
                "risk_score": risk_score,
                "action_type": action_type,
            },
        )
        self.publish(event)

    def publish_agent_suspended(self, agent_id: str, reason: str):
        """Publish an agent suspension event."""
        event = StreamEvent(
            event_type=StreamEventType.AGENT_SUSPENDED,
            subject=f"nethical.agent.{agent_id}.suspended",
            payload={
                "agent_id": agent_id,
                "reason": reason,
            },
            priority="high",
        )
        self.publish_immediate(event)

    def publish_security_breach(self, details: Dict[str, Any]):
        """Publish a security breach event."""
        event = StreamEvent(
            event_type=StreamEventType.SECURITY_BREACH,
            subject="nethical.security.breach",
            payload=details,
            priority="high",
        )
        self.publish_immediate(event)

    def get_metrics(self) -> Dict[str, Any]:
        """Get publisher metrics."""
        return {
            "events_published": self._events_published,
            "events_buffered": self._events_buffered,
            "batches_flushed": self._batches_flushed,
            "buffer_size": len(self._buffer),
        }


# Type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nats_client import NATSClient
