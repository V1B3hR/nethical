"""LLM Integrations for Nethical.

This package provides integration wrappers for various LLM platforms and external systems:

- Claude (Anthropic): Tool-based integration for Claude's function calling
- Grok (xAI): Function calling integration for Grok models
- Gemini (Google): Function calling integration for Gemini models
- REST API: HTTP endpoint for any LLM that can make REST calls (OpenAI, LLaMA, etc.)
- Logging connectors, webhooks, ML platforms, and LangChain (legacy)

Usage Examples:

1. Claude Integration:
    from nethical.integrations.claude_tools import get_nethical_tool, handle_nethical_tool
    
    tools = [get_nethical_tool()]
    # Use with Anthropic client...

2. Grok Integration:
    from nethical.integrations.grok_tools import get_nethical_tool, handle_nethical_tool
    
    tools = [get_nethical_tool()]
    # Use with xAI client...

3. Gemini Integration:
    from nethical.integrations.gemini_tools import get_nethical_tool, handle_nethical_tool
    
    tools = [get_nethical_tool()]
    # Use with Google Generative AI client...

4. REST API:
    # Start server
    python -m nethical.integrations.rest_api
    
    # Or import in your app
    from nethical.integrations.rest_api import app
    # Use with uvicorn...

5. Simple evaluation:
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

try:
    from .grok_tools import (
        get_nethical_tool as get_grok_tool,
        handle_nethical_tool as handle_grok_tool,
        evaluate_action as evaluate_grok_action,
    )
    GROK_AVAILABLE = True
    __all__.extend([
        "get_grok_tool",
        "handle_grok_tool",
        "evaluate_grok_action",
    ])
except ImportError:
    GROK_AVAILABLE = False

try:
    from .gemini_tools import (
        get_nethical_tool as get_gemini_tool,
        handle_nethical_tool as handle_gemini_tool,
        evaluate_action as evaluate_gemini_action,
    )
    GEMINI_AVAILABLE = True
    __all__.extend([
        "get_gemini_tool",
        "handle_gemini_tool",
        "evaluate_gemini_action",
    ])
except ImportError:
    GEMINI_AVAILABLE = False

__all__.extend(["CLAUDE_AVAILABLE", "REST_API_AVAILABLE", "GROK_AVAILABLE", "GEMINI_AVAILABLE", "get_integration_info"])


def get_integration_info() -> Dict[str, Any]:
    """Get information about available integrations.
    
    Returns:
        Dict with integration availability and setup instructions
    """
    return {
        "claude": {
            "available": CLAUDE_AVAILABLE,
            "setup": "pip install anthropic",
            "docs": "See nethical.integrations.claude_tools",
            "manifest": "ai-plugin.json"
        },
        "grok": {
            "available": GROK_AVAILABLE,
            "setup": "pip install xai-sdk (when available) or use REST API",
            "docs": "See nethical.integrations.grok_tools",
            "manifest": "grok-manifest.json"
        },
        "gemini": {
            "available": GEMINI_AVAILABLE,
            "setup": "pip install google-generativeai",
            "docs": "See nethical.integrations.gemini_tools",
            "manifest": "gemini-manifest.json"
        },
        "rest_api": {
            "available": REST_API_AVAILABLE,
            "setup": "pip install fastapi uvicorn",
            "docs": "See nethical.integrations.rest_api",
            "openapi": "openapi.yaml"
        },
        "openai": {
            "available": True,
            "setup": "Use REST API endpoint",
            "docs": "See nethical.integrations.rest_api",
            "manifest": "ai-plugin.json"
        },
        "llama": {
            "available": True,
            "setup": "Use REST API endpoint",
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
