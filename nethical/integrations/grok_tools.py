"""Grok (xAI) Integration for Nethical.

This module provides integration with xAI's Grok models, allowing Nethical
to be used as a tool/function that Grok can call to check actions for safety
and ethical compliance.

Usage:
    from nethical.integrations.grok_tools import get_nethical_tool, handle_nethical_tool

    # Get tool definition for Grok
    tools = [get_nethical_tool()]

    # Use in Grok API call
    # (Example - adapt to actual xAI API when available)
    response = grok_client.chat(
        messages=[{"role": "user", "content": "Check if this is safe: ..."}],
        tools=tools
    )

    # Handle tool calls
    for tool_call in response.tool_calls:
        if tool_call.name == "nethical_guard":
            result = handle_nethical_tool(tool_call.arguments)
            print(f"Decision: {result['decision']}")

Installation:
    # xAI API client (when available)
    # pip install xai-sdk

    # Or use REST API for Grok
    pip install requests

Features:
    - Seamless integration with Grok's function calling
    - Automatic safety and ethics checks
    - PII detection and redaction
    - Risk scoring (0.0-1.0)
    - Audit trail generation
    - Compliance with OWASP LLM Top 10, GDPR, HIPAA
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone

from nethical.core.integrated_governance import IntegratedGovernance
from ._decision_logic import compute_decision, format_violations_for_response


# Global governance instance (singleton pattern)
_governance: Optional[IntegratedGovernance] = None


def get_governance_instance(**kwargs) -> IntegratedGovernance:
    """Get or create the governance instance.

    Args:
        **kwargs: Configuration options for IntegratedGovernance

    Returns:
        IntegratedGovernance instance
    """
    global _governance
    if _governance is None:
        _governance = IntegratedGovernance(
            storage_dir=kwargs.get("storage_dir", "./nethical_grok_data"),
            enable_quota_enforcement=kwargs.get("enable_quota", False),
            enable_performance_optimization=kwargs.get("enable_performance_opt", True),
            enable_merkle_anchoring=kwargs.get("enable_merkle", True),
            enable_ethical_taxonomy=kwargs.get("enable_ethics", True),
            enable_sla_monitoring=kwargs.get("enable_sla", True),
        )
    return _governance


def get_nethical_tool() -> Dict[str, Any]:
    """Get Nethical tool definition for Grok.

    This returns a tool/function definition that can be passed to Grok's
    function calling API.

    Returns:
        Dict containing the tool definition in Grok's format

    Example:
        tools = [get_nethical_tool()]
        response = grok_client.chat(messages=messages, tools=tools)
    """
    return {
        "type": "function",
        "function": {
            "name": "nethical_guard",
            "description": (
                "Evaluate an action, code, or content for safety, ethical compliance, "
                "and security risks. Use this to check user inputs, generated content, "
                "data access requests, code generation, or tool calls before execution. "
                "Returns ALLOW, RESTRICT, BLOCK, or TERMINATE with reasoning, risk score, "
                "and PII detection. Always check high-risk actions like data deletion, "
                "system commands, or sensitive data access."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action, code, or content to evaluate for safety and ethics",
                    },
                    "agent_id": {
                        "type": "string",
                        "description": "Identifier for the AI agent (e.g., 'grok-1', 'my-agent')",
                        "default": "grok",
                    },
                    "action_type": {
                        "type": "string",
                        "description": "Type of action being evaluated",
                        "enum": [
                            "query",
                            "code_generation",
                            "command",
                            "data_access",
                            "tool_call",
                            "user_input",
                            "generated_content",
                        ],
                        "default": "query",
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context about the action (optional)",
                        "additionalProperties": True,
                    },
                },
                "required": ["action"],
            },
        },
    }


def handle_nethical_tool(
    tool_input: Dict[str, Any], governance: Optional[IntegratedGovernance] = None
) -> Dict[str, Any]:
    """Handle a Nethical tool call from Grok.

    This processes a tool call from Grok and returns the evaluation result.

    Args:
        tool_input: Tool input from Grok containing action details
        governance: Optional governance instance (uses default if not provided)

    Returns:
        Dict containing evaluation results with decision, reason, and metadata

    Example:
        result = handle_nethical_tool({
            "action": "Delete all user data",
            "agent_id": "grok-1",
            "action_type": "command"
        })
        print(result["decision"])  # BLOCK
        print(result["reason"])    # "Potentially harmful action detected"
    """
    if governance is None:
        governance = get_governance_instance()

    # Extract parameters
    action = tool_input.get("action", "")
    agent_id = tool_input.get("agent_id", "grok")
    action_type = tool_input.get("action_type", "query")
    context = tool_input.get("context", {})

    # Validate input
    if not action:
        return {
            "decision": "BLOCK",
            "reason": "Empty action provided",
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "action is required",
        }

    try:
        # Process through governance
        result = governance.process_action(
            action=action, agent_id=agent_id, action_type=action_type, context=context
        )

        # Extract key information
        decision_str = compute_decision(result)
        violations = result.get("violations", [])

        # Build response
        response = {
            "decision": decision_str,
            "reason": result.get("reason", "Action evaluated"),
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_score": result.get("risk_score"),
            "pii_detected": bool(result.get("pii_detected")),
            "pii_types": result.get("pii_types", []),
            "violations": format_violations_for_response(violations),
            "audit_id": result.get("audit_id"),
            "metadata": result.get("metadata", {}),
        }

        return response

    except Exception as e:
        # Log detailed error internally but return sanitized message for security
        # TODO: Implement proper logging for detailed error tracking
        return {
            "decision": "BLOCK",
            "reason": "An error occurred during safety evaluation. Please contact support.",
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_code": "EVALUATION_ERROR",
        }


def evaluate_action(
    action: str,
    agent_id: str = "grok",
    action_type: str = "query",
    context: Optional[Dict[str, Any]] = None,
    governance: Optional[IntegratedGovernance] = None,
) -> str:
    """Simplified evaluation function that returns just the decision.

    This is a convenience function for quick checks when you only need
    the decision (ALLOW, RESTRICT, BLOCK, or TERMINATE).

    Args:
        action: The action to evaluate
        agent_id: Agent identifier (default: "grok")
        action_type: Type of action (default: "query")
        context: Additional context (optional)
        governance: Optional governance instance

    Returns:
        Decision string: "ALLOW", "RESTRICT", "BLOCK", or "TERMINATE"

    Example:
        decision = evaluate_action("Generate a report", agent_id="grok-1")
        if decision != "ALLOW":
            print("Action blocked")
    """
    result = handle_nethical_tool(
        {
            "action": action,
            "agent_id": agent_id,
            "action_type": action_type,
            "context": context or {},
        },
        governance=governance,
    )
    return result["decision"]


# Convenience functions for common patterns


def check_user_input(
    user_input: str,
    agent_id: str = "grok",
    governance: Optional[IntegratedGovernance] = None,
) -> Dict[str, Any]:
    """Check user input before processing.

    Args:
        user_input: User's input text
        agent_id: Agent identifier
        governance: Optional governance instance

    Returns:
        Evaluation result dict
    """
    return handle_nethical_tool(
        {"action": user_input, "agent_id": agent_id, "action_type": "user_input"},
        governance=governance,
    )


def check_generated_content(
    content: str,
    agent_id: str = "grok",
    governance: Optional[IntegratedGovernance] = None,
) -> Dict[str, Any]:
    """Check generated content before returning to user.

    Args:
        content: Generated content to check
        agent_id: Agent identifier
        governance: Optional governance instance

    Returns:
        Evaluation result dict
    """
    return handle_nethical_tool(
        {"action": content, "agent_id": agent_id, "action_type": "generated_content"},
        governance=governance,
    )


def check_code_generation(
    code: str,
    language: str = "python",
    agent_id: str = "grok",
    governance: Optional[IntegratedGovernance] = None,
) -> Dict[str, Any]:
    """Check generated code before execution.

    Args:
        code: Generated code to check
        language: Programming language
        agent_id: Agent identifier
        governance: Optional governance instance

    Returns:
        Evaluation result dict
    """
    return handle_nethical_tool(
        {
            "action": code,
            "agent_id": agent_id,
            "action_type": "code_generation",
            "context": {"language": language},
        },
        governance=governance,
    )


def check_tool_call(
    tool_name: str,
    tool_args: Dict[str, Any],
    agent_id: str = "grok",
    governance: Optional[IntegratedGovernance] = None,
) -> Dict[str, Any]:
    """Check tool/function call before execution.

    Args:
        tool_name: Name of the tool/function
        tool_args: Arguments for the tool
        agent_id: Agent identifier
        governance: Optional governance instance

    Returns:
        Evaluation result dict
    """
    action_description = f"Call {tool_name} with args: {tool_args}"
    return handle_nethical_tool(
        {
            "action": action_description,
            "agent_id": agent_id,
            "action_type": "tool_call",
            "context": {"tool_name": tool_name, "tool_args": tool_args},
        },
        governance=governance,
    )
