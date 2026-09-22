# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Unit and Hardening Test Suite for SecurityHeadersMiddleware.

Verifies:
- RFC 6797 compliance: HSTS only added on HTTPS or trusted X-Forwarded-Proto
- Protection against silent security header overrides by custom_headers
- Fail-fast validation in __init__ for hsts_max_age
- CRLF injection prevention in custom headers
- Modern OWASP headers (X-XSS-Protection: 0, COOP, COEP, CORP, object-src 'none')
- Server and X-Powered-By fingerprint stripping
- Accurate URL segment matching in _is_sensitive_endpoint
"""

import pytest
from starlette.requests import Request
from starlette.responses import Response
from nethical.middleware.security import (
    SecurityHeadersMiddleware,
    create_security_middleware,
)


class TestSecurityHeadersHardening:
    """Test suite covering audit findings and hardening."""

    @pytest.fixture
    def app_with_server_headers(self):
        """Mock app returning response with server fingerprint headers."""
        async def app(scope, receive, send):
            pass
        return app

    def test_fail_fast_hsts_validation(self, app_with_server_headers) -> None:
        """K-3: Ensure invalid hsts_max_age fails immediately during __init__."""
        with pytest.raises(ValueError, match="non-negative integer"):
            SecurityHeadersMiddleware(app_with_server_headers, hsts_max_age=-1)

        with pytest.raises(ValueError, match="non-negative integer"):
            SecurityHeadersMiddleware(app_with_server_headers, hsts_max_age="invalid")  # type: ignore

    def test_crlf_injection_prevention(self, app_with_server_headers) -> None:
        """Ś-2: Ensure CRLF characters in custom headers are rejected."""
        with pytest.raises(ValueError, match="CRLF"):
            SecurityHeadersMiddleware(
                app_with_server_headers,
                custom_headers={"X-Test\r\nInjected": "value"}
            )

        with pytest.raises(ValueError, match="CRLF"):
            SecurityHeadersMiddleware(
                app_with_server_headers,
                custom_headers={"X-Test": "value\r\nInjected: true"}
            )

    @pytest.mark.asyncio
    async def test_hsts_rfc6797_enforcement(self) -> None:
        """K-1: HSTS must NEVER be sent over non-HTTPS transport."""
        async def mock_call_next(request: Request) -> Response:
            return Response("OK")

        middleware = SecurityHeadersMiddleware(None, enable_hsts=True, trust_proxy=True)

        # 1. Plain HTTP request -> HSTS must NOT be present
        http_scope = {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "headers": [(b"host", b"testserver")],
        }
        http_req = Request(http_scope)
        resp = await middleware.dispatch(http_req, mock_call_next)
        assert "Strict-Transport-Security" not in resp.headers

        # 2. HTTPS request -> HSTS MUST be present
        https_scope = {
            "type": "http",
            "method": "GET",
            "scheme": "https",
            "path": "/",
            "headers": [(b"host", b"testserver")],
        }
        https_req = Request(https_scope)
        resp_ssl = await middleware.dispatch(https_req, mock_call_next)
        assert "Strict-Transport-Security" in resp_ssl.headers
        assert "max-age=31536000" in resp_ssl.headers["Strict-Transport-Security"]

        # 3. HTTP with trusted X-Forwarded-Proto -> HSTS MUST be present
        proxied_scope = {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "headers": [(b"host", b"testserver"), (b"x-forwarded-proto", b"https")],
        }
        proxied_req = Request(proxied_scope)
        resp_proxy = await middleware.dispatch(proxied_req, mock_call_next)
        assert "Strict-Transport-Security" in resp_proxy.headers

    @pytest.mark.asyncio
    async def test_custom_headers_cannot_weaken_security(self) -> None:
        """K-2: Custom headers must not silently override critical security headers."""
        async def mock_call_next(request: Request) -> Response:
            return Response("OK")

        # Attempt to disable clickjacking protection
        middleware = SecurityHeadersMiddleware(
            None,
            custom_headers={"X-Frame-Options": "ALLOWALL", "X-Custom-Safe": "ok"},
            allow_security_override=False,
        )

        scope = {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "headers": [(b"host", b"testserver")],
        }
        req = Request(scope)
        resp = await middleware.dispatch(req, mock_call_next)

        # X-Frame-Options must remain DENY
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-Custom-Safe"] == "ok"

    @pytest.mark.asyncio
    async def test_server_and_x_powered_by_stripping(self) -> None:
        """W-6: Ensure Server and X-Powered-By headers are stripped."""
        async def mock_call_next(request: Request) -> Response:
            resp = Response("OK")
            resp.headers["Server"] = "uvicorn"
            resp.headers["X-Powered-By"] = "FastAPI"
            return resp

        middleware = SecurityHeadersMiddleware(None)
        scope = {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "headers": [(b"host", b"testserver")],
        }
        req = Request(scope)
        resp = await middleware.dispatch(req, mock_call_next)

        assert "Server" not in resp.headers
        assert "X-Powered-By" not in resp.headers

    @pytest.mark.asyncio
    async def test_modern_owasp_headers_present(self) -> None:
        """W-3, W-4, W-5: Verify modern OWASP headers (XSS=0, COOP, COEP, CORP, object-src)."""
        async def mock_call_next(request: Request) -> Response:
            return Response("OK")

        middleware = SecurityHeadersMiddleware(None)
        scope = {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "path": "/",
            "headers": [(b"host", b"testserver")],
        }
        req = Request(scope)
        resp = await middleware.dispatch(req, mock_call_next)

        assert resp.headers["X-XSS-Protection"] == "0"
        assert resp.headers["Cross-Origin-Opener-Policy"] == "same-origin"
        assert resp.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
        assert resp.headers["Cross-Origin-Resource-Policy"] == "same-origin"
        assert "object-src 'none'" in resp.headers["Content-Security-Policy"]
        assert "frame-src 'none'" in resp.headers["Content-Security-Policy"]

    def test_sensitive_endpoint_segment_matching(self) -> None:
        """W-1: Verify sensitive endpoint detection does not trigger false substring matches."""
        middleware = SecurityHeadersMiddleware(None)

        # Real sensitive endpoints
        assert middleware._is_sensitive_endpoint("/evaluate") is True
        assert middleware._is_sensitive_endpoint("/evaluate/item") is True
        assert middleware._is_sensitive_endpoint("/api/v1/data") is True
        assert middleware._is_sensitive_endpoint("/health") is True
        assert middleware._is_sensitive_endpoint("/status") is True

        # Non-sensitive endpoints
        assert middleware._is_sensitive_endpoint("/public/assets") is False
        assert middleware._is_sensitive_endpoint("/static/styles.css") is False
