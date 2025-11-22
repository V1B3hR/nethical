"""Google Gemini Integration for Nethical.

This module provides integration with Google's Gemini models, allowing Nethical
to be used as a tool/function that Gemini can call to check actions for safety
and ethical compliance.

Usage:
    from nethical.integrations.gemini_tools import get_nethical_tool, handle_nethical_tool
    import google.generativeai as genai

    # Configure Gemini
    genai.configure(api_key="your-api-key")

    # Get tool definition for Gemini
    tools = [get_nethical_tool()]

    # Use in Gemini API call
    model = genai.GenerativeModel('gemini-pro', tools=tools)
    response = model.generate_content("Check if this is safe: ...")

    # Handle tool calls
    for part in response.parts:
        if hasattr(part, 'function_call'):
            result = handle_nethical_tool(part.function_call.args)
            print(f"Decision: {result['decision']}")

Installation:
    pip install google-generativeai

Features:
    - Seamless integration with Gemini's function calling
    - Automatic safety and ethics checks
    - PII detection and redaction
    - Risk scoring (0.0-1.0)
    - Audit trail generation
    - Compliance with OWASP LLM Top 10, GDPR, HIPAA
"""

from typing import Dict, Any, Optional, List
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
            storage_dir=kwargs.get("storage_dir", "./nethical_gemini_data"),
            enable_quota_enforcement=kwargs.get("enable_quota", False),
            enable_performance_optimization=kwargs.get("enable_performance_opt", True),
            enable_merkle_anchoring=kwargs.get("enable_merkle", True),
            enable_ethical_taxonomy=kwargs.get("enable_ethics", True),
            enable_sla_monitoring=kwargs.get("enable_sla", True),
        )
    return _governance


def get_nethical_tool() -> Dict[str, Any]:
    """Get Nethical tool definition for Gemini.

    This returns a tool/function definition that can be passed to Gemini's
    function calling API.

    Returns:
        Dict containing the tool definition in Gemini's format

    Example:
        import google.generativeai as genai
        tools = [get_nethical_tool()]
        model = genai.GenerativeModel('gemini-pro', tools=tools)
    """
    return {
        "function_declarations": [
            {
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
                            "description": "Identifier for the AI agent (e.g., 'gemini-pro', 'my-agent')",
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
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context about the action (optional)",
                        },
                    },
                    "required": ["action"],
                },
            }
        ]
    }


def handle_nethical_tool(
    tool_input: Dict[str, Any], governance: Optional[IntegratedGovernance] = None
) -> Dict[str, Any]:
    """Handle a Nethical tool call from Gemini.

    This processes a tool call from Gemini and returns the evaluation result.

    Args:
        tool_input: Tool input from Gemini containing action details
        governance: Optional governance instance (uses default if not provided)

    Returns:
        Dict containing evaluation results with decision, reason, and metadata

    Example:
        result = handle_nethical_tool({
            "action": "Delete all user data",
            "agent_id": "gemini-pro",
            "action_type": "command"
        })
        print(result["decision"])  # BLOCK
        print(result["reason"])    # "Potentially harmful action detected"
    """
    if governance is None:
        governance = get_governance_instance()

    # Extract parameters
    action = tool_input.get("action", "")
    agent_id = tool_input.get("agent_id", "gemini")
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
        return {
            "decision": "BLOCK",
            "reason": f"Evaluation error: {str(e)}",
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }


def evaluate_action(
    action: str,
    agent_id: str = "gemini",
    action_type: str = "query",
    context: Optional[Dict[str, Any]] = None,
    governance: Optional[IntegratedGovernance] = None,
) -> str:
    """Simplified evaluation function that returns just the decision.

    This is a convenience function for quick checks when you only need
    the decision (ALLOW, RESTRICT, BLOCK, or TERMINATE).

    Args:
        action: The action to evaluate
        agent_id: Agent identifier (default: "gemini")
        action_type: Type of action (default: "query")
        context: Additional context (optional)
        governance: Optional governance instance

    Returns:
        Decision string: "ALLOW", "RESTRICT", "BLOCK", or "TERMINATE"

    Example:
        decision = evaluate_action("Generate a report", agent_id="gemini-pro")
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
    agent_id: str = "gemini",
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
    agent_id: str = "gemini",
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
    agent_id: str = "gemini",
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


def create_gemini_function_response(
    function_call_id: str, result: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a properly formatted function response for Gemini.

    Use this to send the Nethical evaluation result back to Gemini
    as a function response.

    Args:
        function_call_id: The ID from the function call
        result: Result from handle_nethical_tool

    Returns:
        Formatted function response for Gemini

    Example:
        # After receiving function call from Gemini
        result = handle_nethical_tool(function_call.args)
        response = create_gemini_function_response(
            function_call.id,
            result
        )
        # Send response back to Gemini
    """
    return {"function_response": {"name": "nethical_guard", "response": result}}


# Example integration pattern for Gemini


def safe_gemini_chat(
    prompt: str,
    model_name: str = "gemini-pro",
    check_input: bool = True,
    check_output: bool = True,
) -> str:
    """Example safe chat function with Gemini using Nethical.

    This demonstrates a complete integration pattern with both input
    and output checking.

    Args:
        prompt: User's prompt
        model_name: Gemini model name
        check_input: Whether to check input (default: True)
        check_output: Whether to check output (default: True)

    Returns:
        Safe response from Gemini or error message

    Note:
        Requires google-generativeai package and GOOGLE_API_KEY

    Example:
        response = safe_gemini_chat(
            "Tell me about AI safety",
            model_name="gemini-pro"
        )
    """
    try:
        import google.generativeai as genai
    except ImportError:
        return "Error: google-generativeai package not installed. Run: pip install google-generativeai"

    # Check input if requested
    if check_input:
        input_check = check_user_input(prompt, agent_id=model_name)
        if input_check["decision"] != "ALLOW":
            return f"Input blocked: {input_check['reason']}"

    # Generate response from Gemini
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        output = response.text
    except Exception as e:
        return f"Gemini error: {str(e)}"

    # Check output if requested
    if check_output:
        output_check = check_generated_content(output, agent_id=model_name)
        if output_check["decision"] != "ALLOW":
            return f"Output blocked: {output_check['reason']}"

    return output
