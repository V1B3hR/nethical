"""
Event Propagation - Cross-Cache Event System

Propagates cache events across cache levels and regions.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Cache event types."""

    INVALIDATION = "invalidation"
    UPDATE = "update"
    SYNC = "sync"


@dataclass
class CacheEvent:
    """
    Cache event for propagation.

    Attributes:
        event_type: Type of event
        key: Cache key affected
        value: New value (for updates)
        pattern: Pattern affected
        timestamp: Event timestamp
        source_region: Source region
        propagate_globally: Whether to propagate globally
    """

    event_type: EventType
    key: Optional[str] = None
    value: Any = None
    pattern: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    source_region: str = "local"
    propagate_globally: bool = True


class EventPropagation:
    """
    Cross-cache event propagation system.

    Handles event propagation:
    - L1: In-process event callbacks
    - L2: Redis pub/sub
    - L3: Global event stream (NATS/Kafka)

    Consistency:
    - Strong consistency for security events
    - Eventual consistency for policy updates (max 30s lag)
    """

    def __init__(
        self,
        l2_cache: Optional["L2RedisCache"] = None,
        nats_client: Optional[Any] = None,
        region: str = "local",
    ):
        """
        Initialize EventPropagation.

        Args:
            l2_cache: L2 Redis cache for pub/sub
            nats_client: NATS client for global events
            region: Current region identifier
        """
        self.l2_cache = l2_cache
        self.nats_client = nats_client
        self.region = region

        # Local subscribers
        self._subscribers: Dict[EventType, List[Callable]] = {
            et: [] for et in EventType
        }
        self._lock = threading.RLock()

        # Event buffer for batching
        self._event_buffer: List[CacheEvent] = []
        self._buffer_size = 100
        self._flush_interval = 1.0  # seconds

        # Metrics
        self._events_published = 0
        self._events_received = 0

        logger.info(f"EventPropagation initialized for region: {region}")

    def publish(self, event: CacheEvent):
        """
        Publish a cache event.

        Args:
            event: Event to publish
        """
        event.source_region = self.region
        self._events_published += 1

        # Local notification
        self._notify_local(event)

        # L2 pub/sub
        if self.l2_cache and self.l2_cache.is_connected:
            self._publish_l2(event)

        # Global propagation
        if event.propagate_globally and self.nats_client:
            self._publish_global(event)

    def _notify_local(self, event: CacheEvent):
        """Notify local subscribers."""
        with self._lock:
            for callback in self._subscribers.get(event.event_type, []):
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Local subscriber error: {e}")

    def _publish_l2(self, event: CacheEvent):
        """Publish to L2 Redis pub/sub."""
        try:
            channel = f"nethical:cache:{event.event_type.value}"
            message = {
                "event_type": event.event_type.value,
                "key": event.key,
                "pattern": event.pattern,
                "timestamp": event.timestamp,
                "source_region": event.source_region,
            }
            self.l2_cache.publish(channel, message)
        except Exception as e:
            logger.error(f"L2 publish error: {e}")

    def _publish_global(self, event: CacheEvent):
        """Publish to global event stream."""
        # Would use NATS JetStream or Kafka
        # For now, buffer for later processing
        with self._lock:
            self._event_buffer.append(event)
            if len(self._event_buffer) >= self._buffer_size:
                self._flush_buffer()

    def _flush_buffer(self):
        """Flush event buffer to global stream."""
        if not self._event_buffer:
            return

        events = self._event_buffer
        self._event_buffer = []

        # Would send to NATS/Kafka here
        logger.debug(f"Flushing {len(events)} events to global stream")

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[CacheEvent], None],
    ):
        """
        Subscribe to cache events.

        Args:
            event_type: Type of events to receive
            callback: Function to call on event
        """
        with self._lock:
            self._subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: EventType,
        callback: Callable[[CacheEvent], None],
    ):
        """
        Unsubscribe from cache events.

        Args:
            event_type: Type of events
            callback: Function to remove
        """
        with self._lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)

    def start_l2_listener(self):
        """Start listening to L2 pub/sub events."""
        if not self.l2_cache or not self.l2_cache.is_connected:
            return

        def handle_message(message):
            try:
                data = json.loads(message["data"])
                event = CacheEvent(
                    event_type=EventType(data["event_type"]),
                    key=data.get("key"),
                    pattern=data.get("pattern"),
                    timestamp=data.get("timestamp", time.time()),
                    source_region=data.get("source_region", "unknown"),
                    propagate_globally=False,  # Don't re-propagate
                )
                self._events_received += 1
                self._notify_local(event)
            except Exception as e:
                logger.error(f"L2 message handling error: {e}")

        for event_type in EventType:
            channel = f"nethical:cache:{event_type.value}"
            self.l2_cache.subscribe(channel, handle_message)

    def get_metrics(self) -> Dict[str, Any]:
        """Get propagation metrics."""
        return {
            "events_published": self._events_published,
            "events_received": self._events_received,
            "buffer_size": len(self._event_buffer),
            "subscribers": {
                et.value: len(self._subscribers[et]) for et in EventType
            },
            "region": self.region,
        }


# Type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .l2_redis import L2RedisCache
