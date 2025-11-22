"""Tests for Google Gemini integration."""

import pytest
from nethical.integrations.gemini_tools import (
    get_nethical_tool,
    handle_nethical_tool,
    evaluate_action,
    check_user_input,
    check_generated_content,
    check_code_generation,
    create_gemini_function_response,
)


class TestGeminiIntegration:
    """Test suite for Gemini integration."""

    def test_get_nethical_tool(self):
        """Test that tool definition is properly formatted for Gemini."""
        tool = get_nethical_tool()

        assert "function_declarations" in tool
        assert len(tool["function_declarations"]) > 0

        func = tool["function_declarations"][0]
        assert func["name"] == "nethical_guard"
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert "action" in params["required"]

    def test_handle_nethical_tool_basic(self):
        """Test basic tool handling."""
        result = handle_nethical_tool(
            {
                "action": "Tell me about machine learning",
                "agent_id": "gemini-pro",
                "action_type": "query",
            }
        )

        assert "decision" in result
        assert result["decision"] in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]
        assert "reason" in result
        assert "agent_id" in result
        assert result["agent_id"] == "gemini-pro"
        assert "timestamp" in result

    def test_handle_nethical_tool_empty_action(self):
        """Test handling of empty action."""
        result = handle_nethical_tool({"action": "", "agent_id": "gemini-test"})

        assert result["decision"] == "BLOCK"
        assert "Empty action" in result["reason"]

    def test_handle_nethical_tool_with_context(self):
        """Test tool handling with context."""
        result = handle_nethical_tool(
            {
                "action": "import os; os.system('ls')",
                "agent_id": "gemini-test",
                "action_type": "code_generation",
                "context": {"language": "python", "environment": "production"},
            }
        )

        assert "decision" in result
        # Metadata may or may not be present depending on evaluation success

    def test_evaluate_action(self):
        """Test simplified evaluate_action function."""
        decision = evaluate_action(
            "Explain neural networks", agent_id="gemini-pro", action_type="query"
        )

        assert decision in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]

    def test_check_user_input(self):
        """Test user input checking."""
        result = check_user_input("What is deep learning?", agent_id="gemini-pro")

        assert "decision" in result
        assert "reason" in result

    def test_check_generated_content(self):
        """Test generated content checking."""
        result = check_generated_content(
            "Deep learning is a subset of machine learning...", agent_id="gemini-pro"
        )

        assert "decision" in result
        assert "reason" in result

    def test_check_code_generation(self):
        """Test code generation checking."""
        code = """
def calculate_sum(a, b):
    return a + b
"""
        result = check_code_generation(code, language="python", agent_id="gemini-pro")

        assert "decision" in result
        assert "reason" in result

    def test_create_gemini_function_response(self):
        """Test creation of Gemini function response format."""
        tool_result = {
            "decision": "ALLOW",
            "reason": "Action is safe",
            "risk_score": 0.1,
        }

        response = create_gemini_function_response("call_123", tool_result)

        assert "function_response" in response
        assert response["function_response"]["name"] == "nethical_guard"
        assert "response" in response["function_response"]
        assert response["function_response"]["response"]["decision"] == "ALLOW"

    def test_decision_consistency(self):
        """Test that decisions are consistent and valid."""
        test_actions = [
            "Simple query",
            "Generate a report",
            "Access user data",
            "Delete files",
        ]

        for action in test_actions:
            result = handle_nethical_tool({"action": action, "agent_id": "gemini-test"})
            assert result["decision"] in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]

    def test_pii_detection_fields(self):
        """Test PII detection fields in response."""
        # Test with potential PII
        result = handle_nethical_tool(
            {"action": "My email is test@example.com", "agent_id": "gemini-test"}
        )

        # PII fields may be present if evaluation succeeded
        if "pii_detected" in result:
            assert isinstance(result["pii_detected"], bool)

        if "pii_types" in result:
            assert isinstance(result["pii_types"], list)

    def test_risk_score_range(self):
        """Test that risk score is in valid range."""
        result = handle_nethical_tool(
            {"action": "Test action", "agent_id": "gemini-test"}
        )

        if "risk_score" in result and result["risk_score"] is not None:
            assert 0.0 <= result["risk_score"] <= 1.0

    def test_metadata_structure(self):
        """Test metadata structure in response."""
        result = handle_nethical_tool(
            {
                "action": "Test action with metadata",
                "agent_id": "gemini-test",
                "action_type": "query",
            }
        )

        # Required fields
        assert "decision" in result
        assert "reason" in result
        assert "agent_id" in result
        assert "timestamp" in result

        # Optional but expected fields
        if "metadata" in result:
            assert isinstance(result["metadata"], dict)

    def test_error_handling(self):
        """Test error handling in tool."""
        # Test with missing required field (should handle gracefully)
        result = handle_nethical_tool({})

        assert "decision" in result
        assert result["decision"] == "BLOCK"
        assert "error" in result or "Empty action" in result["reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
