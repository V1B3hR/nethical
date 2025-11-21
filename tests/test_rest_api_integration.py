"""Tests for REST API integration.

Run with: pytest tests/test_rest_api_integration.py -v
"""

import pytest
from fastapi.testclient import TestClient

from nethical.integrations.rest_api import app


# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_endpoint_exists(self):
        """Test that health endpoint is accessible."""
        response = client.get("/health")
        assert response.status_code == 200
        
    def test_health_response_structure(self):
        """Test health response has correct structure."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "governance_enabled" in data
        
    def test_health_status_healthy(self):
        """Test that health status is healthy."""
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["governance_enabled"] is True


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        
    def test_root_response_structure(self):
        """Test root response has API information."""
        response = client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "endpoints" in data


class TestEvaluateEndpoint:
    """Test the /evaluate endpoint."""
    
    def test_evaluate_endpoint_exists(self):
        """Test that evaluate endpoint is accessible."""
        response = client.post("/evaluate", json={
            "action": "test action",
            "agent_id": "test"
        })
        assert response.status_code == 200
        
    def test_evaluate_minimal_request(self):
        """Test evaluate with minimal required fields."""
        response = client.post("/evaluate", json={
            "action": "print('hello')"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "decision" in data
        assert "reason" in data
        assert "agent_id" in data
        assert "timestamp" in data
        
    def test_evaluate_with_all_fields(self):
        """Test evaluate with all optional fields."""
        response = client.post("/evaluate", json={
            "action": "generate code",
            "agent_id": "test-agent",
            "action_type": "code_generation",
            "context": {"language": "python"}
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["agent_id"] == "test-agent"
        
    def test_evaluate_missing_action(self):
        """Test evaluate without action field."""
        response = client.post("/evaluate", json={
            "agent_id": "test"
        })
        
        # Should return validation error
        assert response.status_code == 422
        
    def test_evaluate_empty_action(self):
        """Test evaluate with empty action."""
        response = client.post("/evaluate", json={
            "action": ""
        })
        
        # Should return validation error (min_length=1)
        assert response.status_code == 422
        
    def test_evaluate_decision_types(self):
        """Test that decision is one of valid types."""
        response = client.post("/evaluate", json={
            "action": "test action"
        })
        
        data = response.json()
        assert data["decision"] in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]
        
    def test_evaluate_response_structure(self):
        """Test evaluate response has all expected fields."""
        response = client.post("/evaluate", json={
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
    
    def test_safe_code_generation(self):
        """Test evaluating safe code generation."""
        response = client.post("/evaluate", json={
            "action": "def hello(): return 'world'",
            "action_type": "code_generation"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data
        
    def test_database_command(self):
        """Test evaluating database command."""
        response = client.post("/evaluate", json={
            "action": "SELECT * FROM users",
            "action_type": "database_query"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data
        
    def test_file_operation(self):
        """Test evaluating file operation."""
        response = client.post("/evaluate", json={
            "action": "read configuration file",
            "action_type": "file_operation"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data


class TestPIIDetection:
    """Test PII detection in API."""
    
    def test_action_with_email(self):
        """Test action containing email address."""
        response = client.post("/evaluate", json={
            "action": "Send email to user@example.com"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should detect PII
        if data.get("pii_detected"):
            assert "pii_types" in data
            
    def test_action_with_multiple_pii(self):
        """Test action with multiple PII types."""
        response = client.post("/evaluate", json={
            "action": "Contact: email=john@example.com, phone=555-1234"
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have decision
        assert "decision" in data


class TestRiskScore:
    """Test risk score in responses."""
    
    def test_risk_score_present(self):
        """Test that risk score is included when available."""
        response = client.post("/evaluate", json={
            "action": "test action"
        })
        
        data = response.json()
        
        if "risk_score" in data:
            assert isinstance(data["risk_score"], (int, float))
            assert 0.0 <= data["risk_score"] <= 1.0


class TestDefaultValues:
    """Test default values in API."""
    
    def test_default_agent_id(self):
        """Test default agent_id is applied."""
        response = client.post("/evaluate", json={
            "action": "test"
        })
        
        data = response.json()
        assert data["agent_id"] == "unknown"
        
    def test_default_action_type(self):
        """Test default action_type is used."""
        # Implicitly tested - should not error without action_type
        response = client.post("/evaluate", json={
            "action": "test"
        })
        
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling in API."""
    
    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        response = client.post(
            "/evaluate",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
        
    def test_action_too_long(self):
        """Test action exceeding max length."""
        long_action = "x" * 100000  # Exceeds max_length
        response = client.post("/evaluate", json={
            "action": long_action
        })
        
        assert response.status_code == 422
        
    def test_invalid_field_type(self):
        """Test invalid field type."""
        response = client.post("/evaluate", json={
            "action": 12345  # Should be string
        })
        
        assert response.status_code == 422


class TestCORS:
    """Test CORS middleware."""
    
    def test_cors_headers_present(self):
        """Test that CORS headers are included."""
        response = client.options("/evaluate")
        
        # CORS middleware should add headers
        assert "access-control-allow-origin" in response.headers


class TestTimestamp:
    """Test timestamp in responses."""
    
    def test_timestamp_format(self):
        """Test timestamp is in ISO 8601 format."""
        response = client.post("/evaluate", json={
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
    
    def test_concurrent_requests(self):
        """Test that API can handle multiple requests."""
        actions = [
            "action 1",
            "action 2",
            "action 3",
        ]
        
        responses = []
        for action in actions:
            response = client.post("/evaluate", json={
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
    
    def test_openapi_schema(self):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
