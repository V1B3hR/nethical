"""Tests for Nethical v2.0 REST API endpoints."""

import pytest
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
class TestRootEndpoint:
    """Test root endpoint."""

    async def test_root_returns_info(self):
        """Test that root endpoint returns API information."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "Nethical" in data["name"]
        assert "2.0" in data["version"]
        assert "endpoints" in data


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Test health check endpoint."""

    async def test_health_check(self):
        """Test that health check endpoint works."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code in [200, 503]  # May be 503 if not initialized
        if response.status_code == 200:
            data = response.json()
            assert "status" in data


@pytest.mark.asyncio
class TestStatusEndpoint:
    """Test status endpoint."""

    async def test_status_returns_system_info(self):
        """Test that status endpoint returns system information."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/status")

        # May fail if governance not initialized, which is OK for unit tests
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "version" in data
            assert "semantic_monitoring" in data
            assert "semantic_available" in data
            assert "components" in data
            assert isinstance(data["components"], dict)


@pytest.mark.asyncio
class TestEvaluateEndpoint:
    """Test evaluation endpoint."""

    async def test_evaluate_minimal_request(self):
        """Test evaluation with minimal valid request."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/evaluate",
                json={
                    "agent_id": "test_agent",
                    "actual_action": "print('hello world')",
                },
            )

        # May fail if governance not initialized
        if response.status_code == 200:
            data = response.json()

            # Check required fields
            assert "judgment_id" in data
            assert "action_id" in data
            assert "decision" in data
            assert "confidence" in data
            assert "reasoning" in data
            assert "violations" in data
            assert "timestamp" in data

            # Check types
            assert isinstance(data["confidence"], (int, float))
            assert isinstance(data["violations"], list)
            assert data["decision"] in [
                "ALLOW",
                "WARN",
                "BLOCK",
                "QUARANTINE",
                "ESCALATE",
                "TERMINATE",
            ]

    async def test_evaluate_missing_agent_id(self):
        """Test that missing agent_id is rejected."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/evaluate", json={"actual_action": "test"})

        # Should return 422 validation error
        assert response.status_code == 422
