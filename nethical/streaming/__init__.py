"""
Real-Time Event Streaming Module

Replace polling with push-based updates for:
- Policy updates
- Agent events
- Security events

Technology: NATS JetStream (primary) / Kafka (alternative)
"""

from .nats_client import NATSClient, NATSConfig
from .policy_subscriber import PolicySubscriber, PolicyUpdate
from .event_publisher import EventPublisher, StreamEvent

__all__ = [
    "NATSClient",
    "NATSConfig",
    "PolicySubscriber",
    "PolicyUpdate",
    "EventPublisher",
    "StreamEvent",
]
