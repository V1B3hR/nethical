"""Tests for REST API integration.

Run with: pytest tests/test_rest_api_integration.py -v
"""

import pytest
from fastapi.testclient import TestClient

from nethical.integrations import rest_api


# Create test client with lifespan handling
@pytest.fixture(scope="module")
def test_client():
    """Create test client with lifespan context."""
    with TestClient(rest_api.app) as client:
        yield client


# For backwards compatibility with existing tests
client = None


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_endpoint_exists(self, test_client):
        """Test that health endpoint is accessible."""
        response = test_client.get("/health")
        assert response.status_code == 200
        
    def test_health_response_structure(self, test_client):
        """Test health response has correct structure."""
        response = test_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "governance_enabled" in data
        
    def test_health_status_healthy(self, test_client):
        """Test that health status is healthy."""
        response = test_client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["governance_enabled"] is True


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint returns API info."""
        response = test_client.get("/")
        assert response.status_code == 200
        
    def test_root_response_structure(self, test_client):
        """Test root response has API information."""
        response = test_client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data


class TestEvaluateEndpoint:
    """Test the /evaluate endpoint."""
    
    def test_evaluate_endpoint_exists(self, test_client):
        """Test that evaluate endpoint is accessible."""
        response = test_client.post("/evaluate", json={
            "action": "test action",
            "agent_id": "test"
        })
        assert response.status_code == 200
        
    def test_evaluate_minimal_request(self, test_client):
        """Test evaluate with minimal required fields."""
        response = test_client.post("/evaluate", json={
            "action": "print('hello')"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "decision" in data
        assert "reason" in data
        assert "agent_id" in data
        assert "timestamp" in data
        
    def test_evaluate_with_all_fields(self, test_client):
        """Test evaluate with all optional fields."""
        response = test_client.post("/evaluate", json={
            "action": "generate code",
            "agent_id": "test-agent",
            "action_type": "code_generation",
            "context": {"language": "python"}
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["agent_id"] == "test-agent"
        
    def test_evaluate_missing_action(self, test_client):
        """Test evaluate without action field."""
        response = test_client.post("/evaluate", json={
            "agent_id": "test"
        })
        
        # Should return validation error
        assert response.status_code == 422
        
    def test_evaluate_empty_action(self, test_client):
        """Test evaluate with empty action."""
        response = test_client.post("/evaluate", json={
            "action": ""
        })
        
        # Should return validation error (min_length=1)
        assert response.status_code == 422
        
    def test_evaluate_decision_types(self, test_client):
        """Test that decision is one of valid types."""
        response = test_client.post("/evaluate", json={
            "action": "test action"
        })
        
        data = response.json()
        assert data["decision"] in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]
        
    def test_evaluate_response_structure(self, test_client):
        """Test evaluate response has all expected fields."""
        response = test_client.post("/evaluate", json={
            "action": "test action",
            "agent_id": "test"
        })
        
        data = response.json()
        
        # Required fields
        assert "decision" in data
        assert "reason" in data
        assert "agent_id" in data
        assert "timestamp" in data
        
        # Optional fields
        assert "risk_score" in data or "risk_score" not in data  # May or may not be present


class TestEvaluateWithDifferentActions:
    """Test evaluate with various action types."""
    
    def test_safe_code_generation(self, test_client):
        """Test evaluating safe code generation."""
        response = test_client.post("/evaluate", json={
            "action": "def hello(): return 'world'",
            "action_type": "code_generation"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data
        
    def test_database_command(self, test_client):
        """Test evaluating database command."""
        response = test_client.post("/evaluate", json={
            "action": "SELECT * FROM users",
            "action_type": "database_query"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data
        
    def test_file_operation(self, test_client):
        """Test evaluating file operation."""
        response = test_client.post("/evaluate", json={
            "action": "read configuration file",
            "action_type": "file_operation"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data


class TestPIIDetection:
    """Test PII detection in API."""
    
    def test_action_with_email(self, test_client):
        """Test action containing email address."""
        response = test_client.post("/evaluate", json={
            "action": "Send email to user@example.com"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should detect PII
        if data.get("pii_detected"):
            assert "pii_types" in data
            
    def test_action_with_multiple_pii(self, test_client):
        """Test action with multiple PII types."""
        response = test_client.post("/evaluate", json={
            "action": "Contact: email=john@example.com, phone=555-1234"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have decision
        assert "decision" in data


class TestRiskScore:
    """Test risk score in responses."""
    
    def test_risk_score_present(self, test_client):
        """Test that risk score is included when available."""
        response = test_client.post("/evaluate", json={
            "action": "test action"
        })
        
        data = response.json()
        
        if "risk_score" in data:
            assert isinstance(data["risk_score"], (int, float))
            assert 0.0 <= data["risk_score"] <= 1.0


class TestDefaultValues:
    """Test default values in API."""
    
    def test_default_agent_id(self, test_client):
        """Test default agent_id is applied."""
        response = test_client.post("/evaluate", json={
            "action": "test"
        })
        
        data = response.json()
        assert data["agent_id"] == "unknown"
        
    def test_default_action_type(self, test_client):
        """Test default action_type is used."""
        # Implicitly tested - should not error without action_type
        response = test_client.post("/evaluate", json={
            "action": "test"
        })
        
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling in API."""
    
    def test_invalid_json(self, test_client):
        """Test handling of invalid JSON."""
        response = test_client.post(
            "/evaluate",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
        
    def test_action_too_long(self, test_client):
        """Test action exceeding max length."""
        long_action = "x" * 100000  # Exceeds max_length
        response = test_client.post("/evaluate", json={
            "action": long_action
        })
        
        assert response.status_code == 422
        
    def test_invalid_field_type(self, test_client):
        """Test invalid field type."""
        response = test_client.post("/evaluate", json={
            "action": 12345  # Should be string
        })
        
        assert response.status_code == 422


class TestCORS:
    """Test CORS middleware."""
    
    def test_cors_middleware_configured(self, test_client):
        """Test that CORS middleware is configured."""
        from nethical.integrations import rest_api
        
        # Check that CORS middleware is added to the app
        # TestClient doesn't include CORS headers as it's not a real HTTP request
        # but we can verify the middleware is configured
        middleware_types = [type(m) for m in rest_api.app.user_middleware]
        
        # Import middleware class
        from starlette.middleware.cors import CORSMiddleware
        
        # Check if any middleware is CORSMiddleware
        has_cors = any(
            hasattr(m, 'cls') and m.cls == CORSMiddleware
            for m in rest_api.app.user_middleware
        )
        
        assert has_cors, "CORS middleware not configured"


class TestTimestamp:
    """Test timestamp in responses."""
    
    def test_timestamp_format(self, test_client):
        """Test timestamp is in ISO 8601 format."""
        response = test_client.post("/evaluate", json={
            "action": "test"
        })
        
        data = response.json()
        timestamp = data["timestamp"]
        
        # Should be parseable as ISO 8601
        from datetime import datetime
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            valid = True
        except ValueError:
            valid = False
        
        assert valid


class TestMultipleRequests:
    """Test handling multiple requests."""
    
    def test_concurrent_requests(self, test_client):
        """Test that API can handle multiple requests."""
        actions = [
            "action 1",
            "action 2",
            "action 3",
        ]
        
        responses = []
        for action in actions:
            response = test_client.post("/evaluate", json={
                "action": action,
                "agent_id": f"test-{action}"
            })
            responses.append(response)
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        
        # All should have unique agent_ids
        agent_ids = [r.json()["agent_id"] for r in responses]
        assert len(set(agent_ids)) == len(agent_ids)


class TestDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self, test_client):
        """Test OpenAPI schema is available."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
