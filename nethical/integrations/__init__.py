"""LLM Integrations for Nethical.

This package provides integration wrappers for various LLM platforms and external systems:

- Claude (Anthropic): Tool-based integration for Claude's function calling
- REST API: HTTP endpoint for any LLM that can make REST calls (OpenAI, Gemini, etc.)
- Logging connectors, webhooks, ML platforms, and LangChain (legacy)

Usage Examples:

1. Claude Integration:
    from nethical.integrations.claude_tools import get_nethical_tool, handle_nethical_tool
    
    tools = [get_nethical_tool()]
    # Use with Anthropic client...

2. REST API:
    # Start server
    python -m nethical.integrations.rest_api
    
    # Or import in your app
    from nethical.integrations.rest_api import app
    # Use with uvicorn...

3. Simple evaluation:
    from nethical.integrations.claude_tools import evaluate_action
    
    decision = evaluate_action("Write code to delete files")
    if decision != "ALLOW":
        # Block the action
        pass
"""

from typing import Dict, Any

# Legacy integrations
__all__ = ["logging_connectors", "webhook", "ml_platforms", "langchain_tools"]

# Import key functions for convenience
try:
    from .claude_tools import (
        get_nethical_tool,
        handle_nethical_tool,
        evaluate_action,
        get_governance_instance,
    )
    CLAUDE_AVAILABLE = True
    __all__.extend([
        "get_nethical_tool",
        "handle_nethical_tool",
        "evaluate_action",
        "get_governance_instance",
    ])
except ImportError:
    CLAUDE_AVAILABLE = False

try:
    from .rest_api import app as rest_api_app
    REST_API_AVAILABLE = True
    __all__.append("rest_api_app")
except ImportError:
    REST_API_AVAILABLE = False

__all__.extend(["CLAUDE_AVAILABLE", "REST_API_AVAILABLE", "get_integration_info"])


def get_integration_info() -> Dict[str, Any]:
    """Get information about available integrations.
    
    Returns:
        Dict with integration availability and setup instructions
    """
    return {
        "claude": {
            "available": CLAUDE_AVAILABLE,
            "setup": "pip install anthropic",
            "docs": "See nethical.integrations.claude_tools"
        },
        "rest_api": {
            "available": REST_API_AVAILABLE,
            "setup": "pip install fastapi uvicorn",
            "docs": "See nethical.integrations.rest_api"
        }
    }


if __name__ == "__main__":
    info = get_integration_info()
    print("Nethical LLM Integrations:")
    for name, details in info.items():
        status = "✓ Available" if details["available"] else "✗ Not Available"
        print(f"\n{name.upper()}: {status}")
        print(f"  Setup: {details['setup']}")
        print(f"  Docs: {details['docs']}")
