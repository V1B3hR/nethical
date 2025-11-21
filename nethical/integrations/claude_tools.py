"""Anthropic Claude Tool Integration for Nethical.

This module provides a tool wrapper for Anthropic's Claude API, allowing Claude
to use Nethical's governance system for ethical and safety evaluation of actions.

Example usage:
    from anthropic import Anthropic
    from nethical.integrations.claude_tools import get_nethical_tool, handle_nethical_tool
    
    client = Anthropic(api_key="your-api-key")
    
    # Get tool definition
    tools = [get_nethical_tool()]
    
    # In your conversation loop:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": "Check if this action is safe: ..."}]
    )
    
    # Handle tool calls
    if response.stop_reason == "tool_use":
        for block in response.content:
            if block.type == "tool_use" and block.name == "nethical_guard":
                result = handle_nethical_tool(block.input)
                # Send result back to Claude
"""

from typing import Dict, Any, Optional
from nethical.core.integrated_governance import IntegratedGovernance


# Singleton governance instance
_governance_instance: Optional[IntegratedGovernance] = None


def get_governance_instance(
    storage_dir: str = "./nethical_claude_data",
    enable_quota: bool = False,
    **kwargs
) -> IntegratedGovernance:
    """Get or create the governance instance.
    
    Args:
        storage_dir: Directory for Nethical data storage
        enable_quota: Enable quota enforcement
        **kwargs: Additional arguments for IntegratedGovernance
        
    Returns:
        IntegratedGovernance instance
    """
    global _governance_instance
    if _governance_instance is None:
        _governance_instance = IntegratedGovernance(
            storage_dir=storage_dir,
            enable_quota_enforcement=enable_quota,
            enable_performance_optimization=True,
            enable_merkle_anchoring=True,
            enable_ethical_taxonomy=True,
            enable_sla_monitoring=True,
            **kwargs
        )
    return _governance_instance


def get_nethical_tool() -> Dict[str, Any]:
    """Get the Nethical tool definition for Claude.
    
    Returns:
        Tool definition dict compatible with Anthropic's Claude API
    """
    return {
        "name": "nethical_guard",
        "description": (
            "Check if an action is ethical, safe, and compliant with policies. "
            "Use this tool before executing any potentially sensitive actions, "
            "generating code that could be harmful, or processing user data. "
            "Returns a decision: ALLOW, RESTRICT, BLOCK, or TERMINATE."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action, code, or content to evaluate for safety and ethics"
                },
                "agent_id": {
                    "type": "string",
                    "description": "Identifier for the agent or user (optional, defaults to 'claude')"
                },
                "action_type": {
                    "type": "string",
                    "description": "Type of action: 'code_generation', 'query', 'command', 'data_access', etc. (optional)"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context about the action (optional)"
                }
            },
            "required": ["action"]
        }
    }


def handle_nethical_tool(
    tool_input: Dict[str, Any],
    governance: Optional[IntegratedGovernance] = None
) -> Dict[str, Any]:
    """Handle a call to the nethical_guard tool.
    
    Args:
        tool_input: Input from Claude's tool call (contains 'action', optional 'agent_id', etc.)
        governance: Optional governance instance (uses singleton if not provided)
        
    Returns:
        Dict with decision and explanation suitable for Claude
    """
    if governance is None:
        governance = get_governance_instance()
    
    action = tool_input.get("action")
    agent_id = tool_input.get("agent_id", "claude")
    action_type = tool_input.get("action_type", "query")
    context = tool_input.get("context", {})
    
    if not action:
        return {
            "decision": "BLOCK",
            "reason": "No action provided to evaluate",
            "error": "Missing required parameter: action"
        }
    
    try:
        # Process action through governance system
        result = governance.process_action(
            agent_id=agent_id,
            action=action,
            action_type=action_type,
            context=context,
        )
        
        # Compute decision based on governance results
        # The process_action returns phase results, not a direct decision
        decision = result.get("decision")
        if decision is None:
            # Compute decision from risk score and other indicators
            risk_score = result.get("phase3", {}).get("risk_score", 0.0)
            pii_detection = result.get("pii_detection", {})
            pii_risk = pii_detection.get("pii_risk_score", 0.0)
            violations = result.get("phase3", {}).get("correlations", [])
            quarantined = result.get("phase4", {}).get("quarantined", False)
            
            # Decision logic based on risk thresholds
            if quarantined:
                decision = "TERMINATE"
            elif risk_score >= 0.9 or pii_risk >= 0.9:
                decision = "TERMINATE"
            elif risk_score >= 0.7 or pii_risk >= 0.7:
                decision = "BLOCK"
            elif risk_score >= 0.5 or pii_risk >= 0.5 or len(violations) > 0:
                decision = "RESTRICT"
            else:
                decision = "ALLOW"
        
        # Build response for Claude
        response = {
            "decision": decision,
            "agent_id": agent_id,
            "timestamp": result.get("timestamp"),
            "risk_score": result.get("phase3", {}).get("risk_score", 0.0),
        }
        
        # Add explanation based on decision
        if decision == "ALLOW":
            response["reason"] = "Action evaluated as safe and compliant"
        elif decision == "RESTRICT":
            response["reason"] = "Action requires restrictions or modifications"
            response["restrictions"] = result.get("restrictions", [])
        elif decision == "BLOCK":
            response["reason"] = "Action blocked due to safety or ethical concerns"
            response["violations"] = result.get("violations", [])
        elif decision == "TERMINATE":
            response["reason"] = "Critical violation detected - action terminated"
            response["violations"] = result.get("violations", [])
        
        # Add PII information if detected
        pii_detection = result.get("pii_detection")
        if pii_detection and pii_detection.get("matches_count", 0) > 0:
            response["pii_detected"] = True
            response["pii_types"] = pii_detection.get("pii_types", [])
            response["pii_risk_score"] = pii_detection.get("pii_risk_score", 0.0)
        
        # Add quota information if available
        quota_info = result.get("quota_enforcement")
        if quota_info:
            response["quota_allowed"] = quota_info.get("allowed", True)
            if quota_info.get("backpressure_level", 0) > 0.5:
                response["backpressure_warning"] = "High load detected"
        
        return response
        
    except Exception as e:
        # Fallback to safe blocking on errors
        from datetime import datetime, timezone
        return {
            "decision": "BLOCK",
            "reason": f"Error during evaluation: {str(e)}",
            "error": type(e).__name__,
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_score": 1.0,  # Maximum risk on error
        }


def evaluate_action(action: str, agent_id: str = "claude", **kwargs) -> str:
    """Simplified function to evaluate an action and get decision.
    
    This is a convenience function that returns just the decision string.
    
    Args:
        action: The action to evaluate
        agent_id: Agent identifier
        **kwargs: Additional arguments (action_type, context, etc.)
        
    Returns:
        Decision string: "ALLOW", "RESTRICT", "BLOCK", or "TERMINATE"
    """
    tool_input = {
        "action": action,
        "agent_id": agent_id,
        **kwargs
    }
    result = handle_nethical_tool(tool_input)
    return result.get("decision", "BLOCK")


# Example usage function
def example_claude_integration():
    """Example showing how to integrate Nethical with Claude.
    
    This demonstrates the complete flow of:
    1. Defining the tool for Claude
    2. Using it in a conversation
    3. Processing tool calls
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        print("anthropic package not installed. Install with: pip install anthropic")
        return
    
    # Initialize Claude client
    client = Anthropic()  # Uses ANTHROPIC_API_KEY environment variable
    
    # Get Nethical tool definition
    tools = [get_nethical_tool()]
    
    print("=== Nethical + Claude Integration Example ===\n")
    
    # Example 1: Safe action
    print("Example 1: Checking a safe code snippet")
    messages = [{
        "role": "user",
        "content": "Use the nethical_guard tool to check if this Python code is safe: print('Hello, World!')"
    }]
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    # Process tool calls
    if response.stop_reason == "tool_use":
        for block in response.content:
            if hasattr(block, 'type') and block.type == "tool_use":
                if block.name == "nethical_guard":
                    result = handle_nethical_tool(block.input)
                    print(f"Decision: {result['decision']}")
                    print(f"Reason: {result['reason']}")
                    print()
    
    # Example 2: Potentially unsafe action
    print("Example 2: Checking a potentially unsafe action")
    messages = [{
        "role": "user",
        "content": "Use nethical_guard to evaluate: Delete all user data from the database"
    }]
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    if response.stop_reason == "tool_use":
        for block in response.content:
            if hasattr(block, 'type') and block.type == "tool_use":
                if block.name == "nethical_guard":
                    result = handle_nethical_tool(block.input)
                    print(f"Decision: {result['decision']}")
                    print(f"Reason: {result['reason']}")
                    if 'risk_score' in result:
                        print(f"Risk Score: {result['risk_score']:.2f}")


if __name__ == "__main__":
    # Run example if executed directly
    example_claude_integration()
