"""
NATS Client - NATS JetStream Client

Real-time event streaming using NATS JetStream.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NATSConfig:
    """
    NATS configuration.

    Attributes:
        servers: List of NATS servers
        user: Username (optional)
        password: Password (optional)
        token: Token (optional)
        tls: Use TLS
        stream_prefix: Prefix for stream names
        consumer_prefix: Prefix for consumer names
    """

    servers: List[str] = field(default_factory=lambda: ["nats://localhost:4222"])
    user: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    tls: bool = False
    stream_prefix: str = "nethical"
    consumer_prefix: str = "nethical-consumer"


class NATSClient:
    """
    NATS JetStream client.

    Features:
    - Connection management
    - Stream creation
    - Message publishing
    - Consumer subscription

    Fallback: In-memory event queue when NATS unavailable
    """

    def __init__(self, config: Optional[NATSConfig] = None):
        """
        Initialize NATSClient.

        Args:
            config: NATS configuration
        """
        self.config = config or NATSConfig()
        self._nc = None  # NATS connection
        self._js = None  # JetStream context
        self._connected = False

        # Fallback in-memory queue
        self._memory_queue: Dict[str, List[Dict]] = {}
        self._subscribers: Dict[str, List[Callable]] = {}

        # Metrics
        self._messages_published = 0
        self._messages_received = 0

        logger.info("NATSClient initialized")

    async def connect(self) -> bool:
        """
        Connect to NATS server.

        Returns:
            True if connected successfully
        """
        try:
            import nats

            self._nc = await nats.connect(
                servers=self.config.servers,
                user=self.config.user,
                password=self.config.password,
                token=self.config.token,
            )
            self._js = self._nc.jetstream()
            self._connected = True
            logger.info("NATS connected")
            return True

        except ImportError:
            logger.warning("NATS package not installed, using in-memory fallback")
            return False
        except Exception as e:
            logger.error(f"NATS connection failed: {e}")
            return False

    async def close(self):
        """Close NATS connection."""
        if self._nc:
            await self._nc.close()
            self._connected = False
            logger.info("NATS connection closed")

    @property
    def is_connected(self) -> bool:
        """Check if connected to NATS."""
        return self._connected

    async def create_stream(
        self,
        name: str,
        subjects: List[str],
        retention_hours: int = 24,
    ) -> bool:
        """
        Create a JetStream stream.

        Args:
            name: Stream name
            subjects: List of subjects
            retention_hours: Retention period in hours

        Returns:
            True if created successfully
        """
        if not self._js:
            return False

        try:
            from nats.js.api import StreamConfig, RetentionPolicy

            stream_name = f"{self.config.stream_prefix}_{name}"
            config = StreamConfig(
                name=stream_name,
                subjects=subjects,
                retention=RetentionPolicy.LIMITS,
                max_age=retention_hours * 3600 * 1_000_000_000,  # nanoseconds
            )

            await self._js.add_stream(config)
            logger.info(f"Created stream: {stream_name}")
            return True

        except Exception as e:
            logger.error(f"Stream creation failed: {e}")
            return False

    async def publish(
        self,
        subject: str,
        message: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Publish message to subject.

        Args:
            subject: Subject to publish to
            message: Message payload
            headers: Optional headers

        Returns:
            True if published successfully
        """
        self._messages_published += 1

        if self._connected and self._js:
            try:
                data = json.dumps(message).encode()
                await self._js.publish(subject, data, headers=headers)
                return True
            except Exception as e:
                logger.error(f"Publish failed: {e}")
                # Fallthrough to memory queue

        # Memory fallback
        if subject not in self._memory_queue:
            self._memory_queue[subject] = []
        self._memory_queue[subject].append(message)

        # Notify subscribers
        await self._notify_subscribers(subject, message)
        return True

    async def subscribe(
        self,
        subject: str,
        callback: Callable[[Dict[str, Any]], None],
        durable: Optional[str] = None,
    ):
        """
        Subscribe to subject.

        Args:
            subject: Subject to subscribe to
            callback: Function to call on message
            durable: Durable consumer name (optional)
        """
        if subject not in self._subscribers:
            self._subscribers[subject] = []
        self._subscribers[subject].append(callback)

        if self._connected and self._js:
            try:
                consumer_name = durable or f"{self.config.consumer_prefix}_{subject.replace('.', '_')}"

                async def message_handler(msg):
                    try:
                        data = json.loads(msg.data.decode())
                        self._messages_received += 1
                        callback(data)
                        await msg.ack()
                    except Exception as e:
                        logger.error(f"Message handling error: {e}")

                await self._js.subscribe(
                    subject,
                    cb=message_handler,
                    durable=consumer_name,
                )
                logger.info(f"Subscribed to: {subject}")

            except Exception as e:
                logger.error(f"Subscribe failed: {e}")

    async def _notify_subscribers(self, subject: str, message: Dict[str, Any]):
        """Notify local subscribers (memory fallback)."""
        self._messages_received += 1
        for callback in self._subscribers.get(subject, []):
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")

        # Also check pattern subscribers
        for pattern, callbacks in self._subscribers.items():
            if pattern.endswith("*") and subject.startswith(pattern[:-1]):
                for callback in callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Subscriber error: {e}")

    def get_queued_messages(self, subject: str) -> List[Dict[str, Any]]:
        """
        Get queued messages for subject (memory fallback).

        Args:
            subject: Subject to get messages for

        Returns:
            List of queued messages
        """
        return self._memory_queue.get(subject, [])

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        return {
            "connected": self._connected,
            "messages_published": self._messages_published,
            "messages_received": self._messages_received,
            "memory_queue_subjects": len(self._memory_queue),
            "subscribers": len(self._subscribers),
        }
