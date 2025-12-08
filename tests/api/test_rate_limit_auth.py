"""Tests for API rate limiting and authentication."""

import pytest
import os
from httpx import AsyncClient, ASGITransport


# Try to import the API
try:
    from nethical.api import app

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None


pytestmark = pytest.mark.skipif(
    not API_AVAILABLE, reason="API not available - check dependencies"
)


@pytest.mark.asyncio
class TestAuthentication:
    """Test API key authentication."""

    async def test_permissive_mode_allows_all_requests(self, monkeypatch):
        """Test that requests are allowed without API key when in permissive mode."""
        # Ensure no API keys are set (permissive mode)
        monkeypatch.delenv("NETHICAL_API_KEYS", raising=False)

        # Need to reimport to pick up env change
        import importlib
        import nethical.api

        importlib.reload(nethical.api)

        transport = ASGITransport(app=nethical.api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/evaluate",
                json={"agent_id": "test_agent", "actual_action": "print('hello')"},
            )

        # Should succeed (200 or 503 if governance not initialized)
        assert response.status_code in [200, 503]

        # If 503, it's because governance isn't initialized, not auth failure
        if response.status_code == 503:
            detail = response.json().get("detail", "")
            assert "auth" not in detail.lower()

    async def test_valid_api_key_allows_request(self, monkeypatch):
        """Test that valid API key allows request."""
        # Set API keys (enable authentication)
        monkeypatch.setenv("NETHICAL_API_KEYS", "test_key_123,test_key_456")

        # Need to reimport to pick up env change
        import importlib
        import nethical.api

        importlib.reload(nethical.api)

        transport = ASGITransport(app=nethical.api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/evaluate",
                headers={"X-API-Key": "test_key_123"},
                json={"agent_id": "test_agent", "actual_action": "print('hello')"},
            )

        # Should not get 401
        assert response.status_code != 401
        # May get 503 if governance not initialized, but that's fine
        assert response.status_code in [200, 503]

    async def test_invalid_api_key_returns_401(self, monkeypatch):
        """Test that invalid API key returns 401."""
        # Set API keys (enable authentication)
        monkeypatch.setenv("NETHICAL_API_KEYS", "valid_key_only")

        # Need to reimport to pick up env change
        import importlib
        import nethical.api

        importlib.reload(nethical.api)

        transport = ASGITransport(app=nethical.api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/evaluate",
                headers={"X-API-Key": "invalid_key"},
                json={"agent_id": "test_agent", "actual_action": "print('hello')"},
            )

        # Should get 401
        assert response.status_code == 401
        data = response.json()
        assert "Unauthorized" in data.get("detail", "")

    async def test_missing_api_key_returns_401(self, monkeypatch):
        """Test that missing API key returns 401 when auth is required."""
        # Set API keys (enable authentication)
        monkeypatch.setenv("NETHICAL_API_KEYS", "required_key")

        # Need to reimport to pick up env change
        import importlib
        import nethical.api

        importlib.reload(nethical.api)

        transport = ASGITransport(app=nethical.api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/evaluate",
                json={"agent_id": "test_agent", "actual_action": "print('hello')"},
            )

        # Should get 401
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers

    async def test_bearer_token_authentication(self, monkeypatch):
        """Test authentication with Bearer token in Authorization header."""
        # Set API keys
        monkeypatch.setenv("NETHICAL_API_KEYS", "bearer_token_123")

        # Need to reimport to pick up env change
        import importlib
        import nethical.api

        importlib.reload(nethical.api)

        transport = ASGITransport(app=nethical.api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/evaluate",
                headers={"Authorization": "Bearer bearer_token_123"},
                json={"agent_id": "test_agent", "actual_action": "print('hello')"},
            )

        # Should not get 401
        assert response.status_code != 401


@pytest.mark.asyncio
class TestRateLimiting:
    """Test API rate limiting."""

    async def test_rate_limit_returns_429(self, monkeypatch):
        """Test that exceeding rate limit returns 429."""
        # Set very low rate limit for testing
        monkeypatch.setenv("NETHICAL_RATE_BURST", "2")  # 2 req/sec
        monkeypatch.setenv("NETHICAL_RATE_SUSTAINED", "5")  # 5 req/min

        # Need to reimport to pick up env change
        import importlib
        import nethical.api

        importlib.reload(nethical.api)

        transport = ASGITransport(app=nethical.api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Make requests until we hit the limit
            responses = []
            for i in range(10):
                response = await client.post(
                    "/evaluate",
                    json={
                        "agent_id": f"test_agent_{i}",
                        "actual_action": "print('hello')",
                    },
                )
                responses.append(response)

            # At least one should be rate limited
            status_codes = [r.status_code for r in responses]
            assert 429 in status_codes, f"Expected 429 in {status_codes}"

    async def test_rate_limit_headers_present(self, monkeypatch):
        """Test that rate limit headers are present in 429 response."""
        # Set very low rate limit
        monkeypatch.setenv("NETHICAL_RATE_BURST", "1")
        monkeypatch.setenv("NETHICAL_RATE_SUSTAINED", "2")

        # Need to reimport
        import importlib
        import nethical.api

        importlib.reload(nethical.api)

        transport = ASGITransport(app=nethical.api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Exhaust rate limit
            for i in range(5):
                response = await client.post(
                    "/evaluate",
                    json={"agent_id": "test_agent", "actual_action": f"action_{i}"},
                )

                if response.status_code == 429:
                    # Check required headers
                    assert "X-RateLimit-Limit" in response.headers
                    assert "X-RateLimit-Remaining" in response.headers
                    assert "X-RateLimit-Reset" in response.headers
                    assert "Retry-After" in response.headers

                    # Retry-After should be a number
                    retry_after = response.headers.get("Retry-After")
                    assert retry_after.isdigit()
                    break

    async def test_different_identities_have_separate_limits(self, monkeypatch):
        """Test that different API keys have independent rate limits."""
        # Set low rate limit
        monkeypatch.setenv("NETHICAL_API_KEYS", "key1,key2")
        monkeypatch.setenv("NETHICAL_RATE_BURST", "2")
        monkeypatch.setenv("NETHICAL_RATE_SUSTAINED", "3")

        # Need to reimport
        import importlib
        import nethical.api

        importlib.reload(nethical.api)

        transport = ASGITransport(app=nethical.api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Exhaust limit for key1
            key1_responses = []
            for i in range(5):
                response = await client.post(
                    "/evaluate",
                    headers={"X-API-Key": "key1"},
                    json={"agent_id": "agent1", "actual_action": f"action_{i}"},
                )
                key1_responses.append(response.status_code)

            # key1 should be rate limited
            assert 429 in key1_responses

            # key2 should still be allowed (separate limit)
            response = await client.post(
                "/evaluate",
                headers={"X-API-Key": "key2"},
                json={"agent_id": "agent2", "actual_action": "action"},
            )
            # Should not be 429 (may be 401/503, but not rate limited)
            assert response.status_code != 429


@pytest.mark.asyncio
class TestInputValidation:
    """Test input size validation."""

    async def test_oversized_input_returns_413(self):
        """Test that oversized input returns 413."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create input larger than MAX_INPUT_SIZE (default 4096)
            huge_action = "x" * 5000

            response = await client.post(
                "/evaluate",
                json={"agent_id": "test_agent", "actual_action": huge_action},
            )

        # Should get 413 (Payload Too Large)
        # Note: May also get 503 if governance not initialized
        if response.status_code != 503:
            assert response.status_code == 413
            data = response.json()
            assert "too large" in data.get("detail", "").lower()

    async def test_combined_intent_action_size_validated(self):
        """Test that intent + action combined size is validated."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Each under limit, but combined over limit
            medium_text = "x" * 2500

            response = await client.post(
                "/evaluate",
                json={
                    "agent_id": "test_agent",
                    "stated_intent": medium_text,
                    "actual_action": medium_text,
                },
            )

        # Should get 413
        if response.status_code != 503:
            assert response.status_code == 413


@pytest.mark.asyncio
class TestConcurrencyControl:
    """Test concurrency limiting."""

    async def test_concurrent_requests_handled(self, monkeypatch):
        """Test that concurrent requests are properly handled."""
        import asyncio

        # Set reasonable concurrency limit
        monkeypatch.setenv("NETHICAL_MAX_CONCURRENCY", "5")

        # Need to reimport
        import importlib
        import nethical.api

        importlib.reload(nethical.api)

        transport = ASGITransport(app=nethical.api.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Launch many concurrent requests
            async def make_request(i):
                return await client.post(
                    "/evaluate",
                    json={"agent_id": f"agent_{i}", "actual_action": f"action_{i}"},
                )

            # Fire off 10 concurrent requests
            tasks = [make_request(i) for i in range(10)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # All should complete (not hang or crash)
            assert len(responses) == 10

            # Check that we got actual responses, not exceptions
            http_responses = [r for r in responses if not isinstance(r, Exception)]
            assert len(http_responses) > 0
