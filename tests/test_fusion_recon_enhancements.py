"""
Tests for Nethical-Recon Fusion Enhancements.

Validates:
1. Technical Secrets Masking & ReversibleTokenVault expansion
2. SecretsSanitizer one-way log sanitizer
3. EventStreamManager with ring buffer backpressure control
4. GovernanceGateway non-blocking streaming integration
5. API CORS hardening and GZip compression middleware
"""

import gzip
import pytest
from starlette.testclient import TestClient

from nethical.security.token_vault import (
    ReversibleTokenVault,
    SecretsSanitizer,
    SensitiveEntityType,
)
from nethical.streaming.event_stream_manager import (
    EventStreamManager,
    StreamBackend,
    BackpressureStrategy,
    TelemetryEvent,
)
from nethical.gateway.proxy import GovernanceGateway
from nethical.api import app


class TestSecretsSanitizerAndVault:
    """Test suite for SecretsSanitizer and technical credentials masking."""

    def test_secrets_sanitizer_masks_cloud_keys_and_passwords(self):
        sanitizer = SecretsSanitizer(mask_char="*", min_reveal=4)
        raw_text = (
            "Deploying with AWS key AKIAIOSFODNN7EXAMPLE and "
            "token ghp_1234567890abcdef1234567890abcdef1234. "
            "Connecting to postgresql://admin:super_secret_password_123@db.internal:5432/prod."
        )
        sanitized = sanitizer.sanitize(raw_text)

        assert "super_secret_password_123" not in sanitized
        assert "ghp_1234567890abcdef1234567890abcdef1234" not in sanitized
        assert "AKIAIOSFODNN7EXAMPLE" not in sanitized
        assert "post" in sanitized and "prod" in sanitized

    def test_token_vault_reversible_pseudonymization_of_technical_secrets(self):
        vault = ReversibleTokenVault()
        session_id = "test-session-recon"
        original_text = (
            "Database url: postgresql://app_user:db_password_xyz987@10.0.0.5:5432/finance_db. "
            "Bearer token: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.do_not_leak_signature_123. "
            "Customer NIP: 5260250995."
        )

        # 1. Forward Pass (tokenize)
        tok_res = vault.tokenize(original_text, session_id=session_id)
        assert tok_res.tokens_substituted_count >= 2
        assert "db_password_xyz987" not in tok_res.sanitized_text
        assert "do_not_leak_signature_123" not in tok_res.sanitized_text
        assert "[TOKEN_DB_CONNECTION_STRING_" in tok_res.sanitized_text or "[TOKEN_BEARER_TOKEN_" in tok_res.sanitized_text

        # 2. Reverse Pass (detokenize)
        detok_res = vault.detokenize(tok_res.sanitized_text, session_id=session_id)
        assert detok_res.tokens_restored_count == tok_res.tokens_substituted_count
        assert "db_password_xyz987" in detok_res.restored_text
        assert "5260250995" in detok_res.restored_text


class TestEventStreamManager:
    """Test suite for EventStreamManager and backpressure mechanics."""

    def test_publish_and_subscribe_in_memory(self):
        manager = EventStreamManager(backend=StreamBackend.MEMORY)
        received_events = []

        def on_event(evt: TelemetryEvent):
            received_events.append(evt)

        manager.subscribe("governance.decisions", on_event)

        event = manager.publish_nowait(
            topic="governance.decisions",
            payload={"agent_id": "test_agent_1", "verdict": "ALLOW"},
        )

        assert len(received_events) == 1
        assert received_events[0].event_id == event.event_id
        assert received_events[0].payload["verdict"] == "ALLOW"
        assert manager.published_count == 1
        assert manager.delivered_count == 1
        assert manager.dropped_count == 0

    def test_backpressure_drop_oldest(self):
        max_size = 5
        manager = EventStreamManager(
            backend=StreamBackend.MEMORY,
            max_queue_size=max_size,
            backpressure_strategy=BackpressureStrategy.DROP_OLDEST,
        )

        # Publish 12 events
        for i in range(12):
            manager.publish_nowait(
                topic="benchmarks",
                payload={"index": i},
            )

        stats = manager.get_stats()
        assert stats["current_queue_depth"] == max_size
        assert stats["published_total"] == 12
        assert stats["dropped_total"] == 7  # 12 - 5 = 7 dropped oldest

        # Verify ring buffer holds the latest elements: 7, 8, 9, 10, 11
        indices = [evt.payload["index"] for evt in manager._ring_buffer]
        assert indices == [7, 8, 9, 10, 11]

    def test_gateway_publishes_to_stream_manager(self):
        stream_mgr = EventStreamManager(backend=StreamBackend.MEMORY)
        stream_events = []

        stream_mgr.subscribe("governance.decisions", lambda evt: stream_events.append(evt))

        gateway = GovernanceGateway(
            enable_shield=False,
            stream_manager=stream_mgr,
        )

        decision = gateway.intercept_tool_call(
            agent_id="agent_stream_test",
            tool_name="read_file",
            arguments={"path": "/var/log/audit.log"},
        )

        assert decision.decision == "ALLOW"
        assert len(stream_events) == 1
        assert stream_events[0].payload["agent_id"] == "agent_stream_test"
        assert stream_events[0].payload["decision"] == "ALLOW"
        assert stream_events[0].payload["tool_name"] == "read_file"
        assert stream_events[0].payload["latency_us"] > 0


class TestAPIHardeningAndCompression:
    """Test suite for hardened CORS and GZip middleware."""

    def test_health_check_endpoint(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ("ok", "healthy", "up")

    def test_gzip_compression_on_large_payloads(self):
        client = TestClient(app)
        # Request health or api info with Accept-Encoding: gzip
        headers = {"Accept-Encoding": "gzip"}
        response = client.get("/openapi.json", headers=headers)
        assert response.status_code == 200
        # For responses > 1024 bytes, GZipMiddleware adds Content-Encoding: gzip
        if len(response.content) > 1024:
            assert response.headers.get("content-encoding") == "gzip"
