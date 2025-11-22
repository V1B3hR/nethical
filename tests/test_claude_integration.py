"""Tests for Claude/Anthropic integration.

Run with: pytest tests/test_claude_integration.py -v
"""

import pytest
from nethical.integrations.claude_tools import (
    get_nethical_tool,
    handle_nethical_tool,
    evaluate_action,
    get_governance_instance,
)


class TestClaudeToolDefinition:
    """Test the Claude tool definition."""
    
    def test_get_nethical_tool_structure(self):
        """Test that tool definition has correct structure."""
        tool = get_nethical_tool()
        
        assert isinstance(tool, dict)
        assert "name" in tool
        assert tool["name"] == "nethical_guard"
        assert "description" in tool
        assert "input_schema" in tool
        
    def test_tool_schema_properties(self):
        """Test tool schema has required properties."""
        tool = get_nethical_tool()
        schema = tool["input_schema"]
        
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "action" in schema["properties"]
        assert "required" in schema
        assert "action" in schema["required"]
        
    def test_tool_optional_parameters(self):
        """Test tool has optional parameters."""
        tool = get_nethical_tool()
        props = tool["input_schema"]["properties"]
        
        assert "agent_id" in props
        assert "action_type" in props
        assert "context" in props


class TestToolHandler:
    """Test the tool handler function."""
    
    def test_handle_valid_action(self):
        """Test handling a valid safe action."""
        tool_input = {
            "action": "print('Hello, World!')",
            "agent_id": "test-agent",
            "action_type": "code_generation"
        }
        
        result = handle_nethical_tool(tool_input)
        
        assert isinstance(result, dict)
        assert "decision" in result
        assert result["decision"] in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]
        assert "reason" in result
        assert "agent_id" in result
        assert result["agent_id"] == "test-agent"
        
    def test_handle_missing_action(self):
        """Test handling when action is missing."""
        tool_input = {
            "agent_id": "test-agent"
        }
        
        result = handle_nethical_tool(tool_input)
        
        assert result["decision"] == "BLOCK"
        assert "error" in result
        
    def test_handle_default_agent_id(self):
        """Test default agent_id is used when not provided."""
        tool_input = {
            "action": "test action"
        }
        
        result = handle_nethical_tool(tool_input)
        
        assert result["agent_id"] == "claude"
        
    def test_handle_with_context(self):
        """Test handling with additional context."""
        tool_input = {
            "action": "generate code",
            "agent_id": "test",
            "context": {"language": "python", "purpose": "testing"}
        }
        
        result = handle_nethical_tool(tool_input)
        
        assert "decision" in result
        assert "reason" in result
        
    def test_result_has_timestamp(self):
        """Test result includes timestamp."""
        tool_input = {
            "action": "test action"
        }
        
        result = handle_nethical_tool(tool_input)
        
        assert "timestamp" in result
        
    def test_result_has_risk_score(self):
        """Test result includes risk score."""
        tool_input = {
            "action": "test action"
        }
        
        result = handle_nethical_tool(tool_input)
        
        assert "risk_score" in result
        assert isinstance(result["risk_score"], (int, float))


class TestEvaluateAction:
    """Test the simplified evaluate_action function."""
    
    def test_evaluate_safe_action(self):
        """Test evaluating a safe action."""
        decision = evaluate_action("print('hello')")
        
        assert decision in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]
        
    def test_evaluate_with_agent_id(self):
        """Test evaluating with custom agent_id."""
        decision = evaluate_action("test", agent_id="custom-agent")
        
        assert isinstance(decision, str)
        
    def test_evaluate_with_action_type(self):
        """Test evaluating with action_type."""
        decision = evaluate_action(
            "generate code",
            action_type="code_generation"
        )
        
        assert isinstance(decision, str)
        
    def test_evaluate_returns_string(self):
        """Test that evaluate_action returns a string decision."""
        decision = evaluate_action("test action")
        
        assert isinstance(decision, str)
        assert decision in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]


class TestGovernanceInstance:
    """Test governance instance management."""
    
    def test_get_governance_instance(self):
        """Test getting governance instance."""
        instance = get_governance_instance()
        
        assert instance is not None
        assert hasattr(instance, 'process_action')
        
    def test_governance_instance_singleton(self):
        """Test that governance instance is a singleton."""
        instance1 = get_governance_instance()
        instance2 = get_governance_instance()
        
        assert instance1 is instance2
        
    def test_governance_with_custom_storage(self):
        """Test creating governance with custom storage dir."""
        instance = get_governance_instance(
            storage_dir="/tmp/test_nethical_claude"
        )
        
        assert instance is not None


class TestErrorHandling:
    """Test error handling in integration."""
    
    def test_handle_invalid_input_type(self):
        """Test handling invalid input types."""
        tool_input = {
            "action": ["not", "a", "string"]  # Wrong type
        }
        
        # Should not raise exception
        result = handle_nethical_tool(tool_input)
        assert "decision" in result
        
    def test_handle_empty_action(self):
        """Test handling empty action string."""
        tool_input = {
            "action": ""
        }
        
        result = handle_nethical_tool(tool_input)
        
        # Should handle gracefully
        assert "decision" in result


class TestPIIDetection:
    """Test PII detection in integration."""
    
    def test_pii_detection_in_result(self):
        """Test that PII detection results are included."""
        tool_input = {
            "action": "Process email: user@example.com",
            "agent_id": "test"
        }
        
        result = handle_nethical_tool(tool_input)
        
        # PII detection should be present in result
        assert "decision" in result
        # If PII is detected, it should be flagged
        if result.get("pii_detected"):
            assert "pii_types" in result
            assert isinstance(result["pii_types"], list)


class TestDecisionTypes:
    """Test different decision types."""
    
    def test_allow_decision_format(self):
        """Test ALLOW decision includes correct fields."""
        tool_input = {
            "action": "print('safe')",
            "agent_id": "test"
        }
        
        result = handle_nethical_tool(tool_input)
        
        if result["decision"] == "ALLOW":
            assert "reason" in result
            assert "risk_score" in result
            
    def test_block_decision_format(self):
        """Test BLOCK decision includes violations if present."""
        tool_input = {
            "action": "rm -rf /",  # Potentially dangerous
            "agent_id": "test"
        }
        
        result = handle_nethical_tool(tool_input)
        
        # Should have decision and reason
        assert "decision" in result
        assert "reason" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
