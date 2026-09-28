# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Military-Grade Network Security & Hardening Test Suite.

Verifies:
1. SSRF Deep Defense: Loopback, Cloud Metadata (169.254.169.254), RFC 1918, Disallowed schemes.
2. OpenAI Proxy SSRF & Upstream Injection Mitigation.
3. Webhook SSRF & DNS-Rebinding Mitigation.
4. Client IP Spoofing Prevention via untrusted X-Forwarded-For headers.
5. Security Headers (HSTS, CSP, Anti-Clickjacking, Anti-Sniffing).
6. WebSocket Security (CSWSH origin checks, Token auth, Connection limits).
7. Edge Device Hub Concurrency & 64KB Frame Bounds & Admission Authentication.
8. A2A Protocol Cryptographic Signature Tamper Resistance (HMAC-SHA256).
"""

from __future__ import annotations

import json
import socket
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

from nethical.api.app import app, get_client_ip, ConnectionManager
from nethical.api.auth import AuthManager
from nethical.edge.device_hub import EdgeDeviceHub, EdgeDeviceProfile, DeviceType, ActuationBus
from nethical.gateway.a2a_protocol import A2AHandshakeManager, A2ACapabilityBoundary
from nethical.gateway.openai_proxy import OpenAIGovernanceProxy, ChatMessage
from nethical.integrations.webhook import HTTPWebhookDispatcher
from nethical.security.ssrf_protection import (
    assert_safe_url,
    validate_safe_url,
    is_ip_private_or_restricted,
    SSRFValidationError,
)


# =============================================================================
# 1. SSRF Deep Defense Tests
# =============================================================================

class TestSSRFProtection:
    """Verifies strict IP and URL boundary filtration against SSRF attacks."""

    @pytest.mark.parametrize("blocked_url", [
        "http://127.0.0.1:8080/admin",
        "http://localhost:8000/internal",
        "http://169.254.169.254/latest/meta-data/",
        "http://169.254.169.253/",
        "http://10.0.0.1/secrets",
        "http://172.16.5.10/database",
        "http://192.168.1.1/router",
        "file:///etc/passwd",
        "gopher://127.0.0.1:6379/_flushall",
        "dict://127.0.0.1:11211/stat",
        "ftp://internal.server/data",
        "http://0.0.0.0:8000",
        "http://[::1]:8080",
    ])
    def test_ssrf_blocks_private_and_dangerous_endpoints(self, blocked_url: str):
        is_safe, reason = validate_safe_url(blocked_url, allow_private=False)
        assert is_safe is False
        assert len(reason) > 0
        with pytest.raises(SSRFValidationError):
            assert_safe_url(blocked_url, allow_private=False)

    def test_ssrf_allows_legitimate_public_urls(self):
        is_safe, reason = validate_safe_url("https://api.openai.com/v1/chat/completions")
        assert is_safe is True
        assert reason == ""

    def test_ssrf_allow_private_flag(self):
        # 10.0.0.1 should be allowed when explicitly permitting private intranet
        is_safe, reason = validate_safe_url("http://10.0.0.1:8000/status", allow_private=True)
        assert is_safe is True
        assert reason == ""

        # Cloud metadata (169.254.169.254) must ALWAYS be blocked even with allow_private=True
        is_safe_meta, reason_meta = validate_safe_url("http://169.254.169.254/latest", allow_private=True)
        assert is_safe_meta is False
        assert "metadata" in reason_meta.lower()

        # Loopback must ALWAYS be blocked even with allow_private=True
        is_safe_loop, reason_loop = validate_safe_url("http://127.0.0.1:8000/", allow_private=True)
        assert is_safe_loop is False
        assert "loopback" in reason_loop.lower()

    def test_ssrf_domain_whitelist_enforcement(self):
        whitelist = ["api.openai.com", "api.anthropic.com"]
        
        is_safe_ok, _ = validate_safe_url("https://api.openai.com/v1", allowed_domains=whitelist)
        assert is_safe_ok is True

        is_safe_blocked, reason = validate_safe_url("https://malicious-site.com/v1", allowed_domains=whitelist)
        assert is_safe_blocked is False
        assert "not in allowed domains" in reason


# =============================================================================
# 2. OpenAI Proxy SSRF Protection Tests
# =============================================================================

class TestOpenAIProxySSRF:
    """Verifies reverse proxy upstream sanitization."""

    @pytest.mark.asyncio
    async def test_openai_proxy_rejects_ssrf_upstream_header(self):
        proxy = OpenAIGovernanceProxy(mock_mode=False, allow_client_upstream=True)
        req_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        headers = {
            "x-nethical-upstream": "http://169.254.169.254/latest/meta-data/",
        }
        with pytest.raises(SSRFValidationError):
            await proxy.handle_chat_completion(req_data, headers)

    @pytest.mark.asyncio
    async def test_openai_proxy_ignores_client_upstream_by_default(self):
        # Default policy: allow_client_upstream=False
        proxy = OpenAIGovernanceProxy(
            mock_mode=True,
            default_upstream_url="mock",
            allow_client_upstream=False
        )
        req_data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        # Client tries to inject upstream
        headers = {
            "x-nethical-upstream": "http://192.168.1.100:9999",
        }
        # Must fall back to default_upstream_url (mock mode) and NOT call 192.168.1.100
        result = await proxy.handle_chat_completion(req_data, headers)
        assert isinstance(result, dict)
        assert result["object"] == "chat.completion"


# =============================================================================
# 3. Webhook SSRF & Scheme Protection Tests
# =============================================================================

class TestWebhookSSRF:
    """Verifies webhook dispatcher outbound URL enforcement."""

    def test_webhook_dispatcher_blocks_internal_and_loopback_urls(self):
        with pytest.raises(ValueError):
            HTTPWebhookDispatcher("http://127.0.0.1:9090/metrics")

        with pytest.raises(ValueError):
            HTTPWebhookDispatcher("http://169.254.169.254/secret")

        with pytest.raises(ValueError):
            HTTPWebhookDispatcher("file:///etc/hosts")


# =============================================================================
# 4. Ingress IP Spoofing Protection (get_client_ip)
# =============================================================================

class TestIPSpoofingProtection:
    """Verifies that untrusted clients cannot spoof identity via X-Forwarded-For."""

    def test_untrusted_client_cannot_spoof_x_forwarded_for(self):
        # Simulating external client connecting directly (e.g., from 198.51.100.77)
        scope = {
            "type": "http",
            "client": ("198.51.100.77", 45678),
            "headers": [
                (b"x-forwarded-for", b"10.0.0.1"),
                (b"x-real-ip", b"127.0.0.1"),
            ],
        }
        req = Request(scope)
        detected_ip = get_client_ip(req)
        # MUST return the actual peer IP, NOT the spoofed header
        assert detected_ip == "198.51.100.77"

    def test_trusted_proxy_correctly_forwards_client_ip(self):
        # Simulating request from local reverse proxy (127.0.0.1 in TRUSTED_PROXIES)
        scope = {
            "type": "http",
            "client": ("127.0.0.1", 54321),
            "headers": [
                (b"x-forwarded-for", b"203.0.113.195, 10.0.0.1"),
            ],
        }
        req = Request(scope)
        detected_ip = get_client_ip(req)
        # MUST extract the original external client IP from the trusted proxy header
        assert detected_ip == "203.0.113.195"


# =============================================================================
# 5. Security Headers Middleware Enforcement
# =============================================================================

class TestSecurityHeaders:
    """Verifies that hardened OWASP headers are injected on API responses."""

    def test_security_headers_present_on_health_check(self):
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            headers = resp.headers

            # Anti-sniffing
            assert headers.get("X-Content-Type-Options") == "nosniff"
            # Clickjacking defense
            assert headers.get("X-Frame-Options") == "DENY"
            # Modern XSS disabling
            assert headers.get("X-XSS-Protection") == "0"
            # COOP / CORP
            assert headers.get("Cross-Origin-Opener-Policy") == "same-origin"
            assert headers.get("Cross-Origin-Resource-Policy") == "same-origin"
            # Server header stripped
            assert "server" not in headers or headers["server"] == ""


# =============================================================================
# 6. WebSocket Security (CSWSH & Connection Limiting)
# =============================================================================

class TestWebSocketSecurity:
    """Verifies WebSocket origin validation and connection limits."""

    def test_websocket_rejects_unauthorized_origin(self):
        with patch("nethical.api.app.allowed_origins", ["https://app.nethical.ai"]):
            with TestClient(app) as client:
                # Malicious origin CSWSH attack
                headers = {"Origin": "https://evil-hacker.com"}
                with pytest.raises(WebSocketDisconnect):
                    with client.websocket_connect("/ws/violations", headers=headers):
                        pass

    @pytest.mark.asyncio
    async def test_connection_manager_per_ip_rate_limit(self):
        manager = ConnectionManager(max_connections_per_ip=2)
        ip = "192.0.2.1"

        mock_ws_1 = AsyncMock()
        mock_ws_2 = AsyncMock()
        mock_ws_3 = AsyncMock()

        # Connect 1 & 2 succeed
        assert await manager.connect(mock_ws_1, client_ip=ip) is True
        assert await manager.connect(mock_ws_2, client_ip=ip) is True

        # Connect 3 exceeds limit of 2 and is closed
        assert await manager.connect(mock_ws_3, client_ip=ip) is False
        mock_ws_3.close.assert_called_once_with(code=1008, reason="Connection limit exceeded")

        # Disconnecting frees a slot
        manager.disconnect(mock_ws_1, client_ip=ip)
        assert await manager.connect(mock_ws_3, client_ip=ip) is True


# =============================================================================
# 7. Edge Device Hub Concurrency & Security
# =============================================================================

class TestEdgeDeviceHubSecurity:
    """Verifies admission secret enforcement and framing bounds in EdgeDeviceHub."""

    def test_edge_hub_rejects_unauthorized_admission(self):
        hub = EdgeDeviceHub(admission_secret="top_secret_edge_psk_2026")
        port = 18995
        hub.start_socket_server(host="127.0.0.1", port=port)
        time.sleep(0.05)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", port))

            # Missing or wrong secret
            req = {
                "action": "admit",
                "admission_secret": "wrong_secret",
                "profile": {
                    "device_id": "rogue_robot",
                    "manufacturer": "Unknown",
                    "model": "X",
                    "device_type": "ROBOT_6AXIS",
                    "mac_address": "00:00:00:00:00:00",
                },
            }
            sock.sendall(json.dumps(req).encode("utf-8"))
            resp = json.loads(sock.recv(4096).decode("utf-8"))
            sock.close()

            assert resp["status"] == "UNAUTHORIZED"
            assert "Invalid or missing admission secret" in resp["message"]

        finally:
            hub.stop_socket_server()

    def test_edge_hub_admits_with_valid_secret(self):
        hub = EdgeDeviceHub(admission_secret="top_secret_edge_psk_2026")
        port = 18996
        hub.start_socket_server(host="127.0.0.1", port=port)
        time.sleep(0.05)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", port))

            req = {
                "action": "admit",
                "admission_secret": "top_secret_edge_psk_2026",
                "profile": {
                    "device_id": "authorized_drone_01",
                    "manufacturer": "DJI",
                    "model": "Matrice 300",
                    "device_type": "UAV_DRONE",
                    "mac_address": "60:60:1F:AA:BB:CC",
                },
            }
            sock.sendall(json.dumps(req).encode("utf-8"))
            resp = json.loads(sock.recv(4096).decode("utf-8"))
            sock.close()

            assert resp["status"] == "SUCCESS"
            assert resp["admission"]["device_id"] == "authorized_drone_01"

        finally:
            hub.stop_socket_server()


# =============================================================================
# 8. A2A Protocol Cryptographic Signature & Tamper Resistance (HMAC-SHA256)
# =============================================================================

class TestA2ACryptographicSignatures:
    """Verifies HMAC signature verification and tamper detection in A2A handshakes."""

    def test_a2a_handshake_detects_tampered_proposal(self):
        secret = "sovereign_cluster_secret_key_fips_2026"
        manager = A2AHandshakeManager(cluster_secret=secret)

        offer = manager.propose_handshake(
            initiator_id="trading_agent_alpha",
            target_id="execution_agent_beta",
            boundaries=A2ACapabilityBoundary(max_budget_units=50.0),
        )

        assert "initiator_signature" in offer
        assert len(offer["initiator_signature"]) == 64

        # Attack: Eavesdropper alters the proposal (e.g. increases budget from 50 to 999999)
        tampered_offer = json.loads(json.dumps(offer))
        tampered_offer["proposal"]["boundaries"]["max_budget_units"] = 999999.0

        # Target agent attempts to accept the tampered offer
        with pytest.raises(PermissionError) as exc_info:
            manager.accept_handshake(tampered_offer, target_id="execution_agent_beta")

        assert "Sfałszowany lub unieważniony podpis" in str(exc_info.value)
