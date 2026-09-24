# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Comprehensive unit and integration tests for nethical.streaming module."""

import asyncio
import concurrent.futures
import time
import pytest

from nethical.streaming import (
    NATSClient,
    NATSConfig,
    PolicySubscriber,
    PolicyUpdate,
    EventPublisher,
    StreamEvent,
    EventStreamManager,
    StreamBackend,
    BackpressureStrategy,
    TelemetryEvent,
    get_stream_manager,
)
from nethical.streaming.event_publisher import StreamEventType
from nethical.streaming.policy_subscriber import PolicyEventType


def test_telemetry_event_to_dict():
    """Verify TelemetryEvent serialization contains ISO timestamp and standard fields."""
    event = TelemetryEvent(
        topic="governance.decisions",
        payload={"decision": "ALLOW", "risk": 0.05},
        source="agent_interceptor",
        metadata={"tenant": "sovereign_eu"},
    )
    d = event.to_dict()
    assert d["event_id"].startswith("EVT-")
    assert d["topic"] == "governance.decisions"
    assert d["payload"]["decision"] == "ALLOW"
    assert d["source"] == "agent_interceptor"
    assert "timestamp_iso" in d
    assert "T" in d["timestamp_iso"]


def test_event_stream_manager_drop_newest():
    """Verify DROP_NEWEST drops incoming events once buffer capacity is reached."""
    manager = EventStreamManager(
        backend=StreamBackend.MEMORY,
        max_queue_size=3,
        backpressure_strategy=BackpressureStrategy.DROP_NEWEST,
    )

    ev1 = manager.publish_nowait("test.topic", {"n": 1})
    ev2 = manager.publish_nowait("test.topic", {"n": 2})
    ev3 = manager.publish_nowait("test.topic", {"n": 3})

    # Buffer is full (size 3)
    ev4 = manager.publish_nowait("test.topic", {"n": 4})

    stats = manager.get_stats()
    assert stats["current_queue_depth"] == 3
    assert stats["published_total"] == 3
    assert stats["dropped_total"] == 1

    events = manager.get_events()
    assert [e.payload["n"] for e in events] == [1, 2, 3]


def test_event_stream_manager_block_sync_and_async():
    """Verify BLOCK backpressure raises BufferError on sync publish and succeeds asynchronously when space clears."""
    manager = EventStreamManager(
        backend=StreamBackend.MEMORY,
        max_queue_size=2,
        backpressure_strategy=BackpressureStrategy.BLOCK,
    )

    manager.publish_nowait("test.block", {"item": 1})
    manager.publish_nowait("test.block", {"item": 2})

    # Sync publish when full with BLOCK must raise BufferError
    with pytest.raises(BufferError, match="BLOCK strategy requires"):
        manager.publish_nowait("test.block", {"item": 3})

    # Async publish with timeout should timeout if buffer remains full
    async def run_timeout():
        with pytest.raises(TimeoutError):
            await manager.publish("test.block", {"item": 3}, timeout=0.05)

    asyncio.run(run_timeout())


def test_event_stream_manager_thread_safety():
    """Verify thread-safety of EventStreamManager under heavy concurrent publishing."""
    manager = EventStreamManager(
        backend=StreamBackend.MEMORY,
        max_queue_size=500,
        backpressure_strategy=BackpressureStrategy.DROP_OLDEST,
    )

    received = []
    manager.subscribe("thread.test", lambda ev: received.append(ev.event_id))

    def worker(idx: int):
        for i in range(20):
            manager.publish_nowait("thread.test", {"worker": idx, "seq": i})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, w) for w in range(8)]
        for f in futures:
            f.result()

    stats = manager.get_stats()
    assert stats["published_total"] == 160
    assert stats["delivered_total"] == 160
    assert len(received) == 160


def test_nats_client_memory_fallback_and_patterns():
    """Verify NATSClient in-memory fallback queue and pattern prefix subscription matching."""
    client = NATSClient()
    assert client.is_connected is False

    received_all = []
    received_agent = []

    async def run_nats_test():
        await client.subscribe("nethical.*", lambda msg: received_all.append(msg))
        await client.subscribe("nethical.agent.action", lambda msg: received_agent.append(msg))

        await client.publish("nethical.agent.action", {"action": "invoke_tool"})
        await client.publish("nethical.system.ping", {"status": "ok"})

        metrics = client.get_metrics()
        assert metrics["messages_published"] == 2
        assert metrics["messages_received"] == 2
        assert len(received_all) == 2
        assert len(received_agent) == 1

        queued = client.get_queued_messages("nethical.agent.action")
        assert len(queued) == 1
        assert queued[0]["action"] == "invoke_tool"

        await client.close()

    asyncio.run(run_nats_test())


def test_event_publisher_batching_and_immediate():
    """Verify EventPublisher batching flush, immediate publishing, and metrics tracking."""
    client = NATSClient()
    publisher = EventPublisher(nats_client=client, batch_size=3, flush_interval=10.0)

    # Publish 2 events (buffered, not flushed yet)
    publisher.publish(StreamEvent(
        event_type=StreamEventType.POLICY_UPDATED,
        subject="nethical.policy.pol_1.updated",
        payload={"pol": 1},
    ))
    publisher.publish(StreamEvent(
        event_type=StreamEventType.AGENT_DECISION,
        subject="nethical.agent.agt_1.decision",
        payload={"decision": "ALLOW"},
    ))

    m1 = publisher.get_metrics()
    assert m1["events_buffered"] == 2
    assert m1["events_published"] == 0

    # Publish 3rd event: batch size reached, triggers flush
    publisher.publish(StreamEvent(
        event_type=StreamEventType.SYSTEM_HEALTH,
        subject="nethical.system.health",
        payload={"cpu": 12},
    ))

    m2 = publisher.get_metrics()
    assert m2["events_published"] == 3
    assert m2["batches_flushed"] == 1
    assert m2["buffer_size"] == 0

    # Publish immediate security breach
    publisher.publish_security_breach({"breach_type": "exfiltration_attempt"})
    m3 = publisher.get_metrics()
    assert m3["events_published"] == 4

    publisher.stop()


def test_policy_subscriber_history_and_handlers():
    """Verify PolicySubscriber handles updates, bounded deque history, and callbacks."""
    client = NATSClient()

    class MockCache:
        def __init__(self):
            self.invalidated = []

        def invalidate_pattern(self, pattern: str):
            self.invalidated.append(pattern)

    cache = MockCache()
    subscriber = PolicySubscriber(nats_client=client, cache_hierarchy=cache, max_history=3)

    received_updates = []
    subscriber.on_update(lambda upd: received_updates.append(upd.policy_id))

    # Send 4 updates through subscriber
    for i in range(1, 5):
        msg = {
            "policy_id": f"pol_{i}",
            "event_type": "updated",
            "version": f"1.{i}",
            "content": {"rule": "allow_all"},
            "timestamp": time.time(),
        }
        subscriber._handle_message(msg)

    assert len(received_updates) == 4
    assert len(cache.invalidated) == 4
    assert cache.invalidated[0] == "policy:pol_1"

    # Max history is 3, pol_1 should be evicted from recent updates
    recent = subscriber.get_recent_updates(limit=10)
    assert len(recent) == 3
    assert [u.policy_id for u in recent] == ["pol_2", "pol_3", "pol_4"]

    # Filter by specific policy_id
    filtered = subscriber.get_recent_updates(policy_id="pol_3")
    assert len(filtered) == 1
    assert filtered[0].policy_id == "pol_3"

    metrics = subscriber.get_metrics()
    assert metrics["updates_received"] == 4
    assert metrics["history_size"] == 3
    assert metrics["handlers_registered"] == 1
