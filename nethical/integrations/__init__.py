"""LLM Integrations for Nethical.

This package provides integration wrappers for various LLM platforms and external systems:

- Claude (Anthropic): Tool-based integration for Claude's function calling
- Grok (xAI): Function calling integration for Grok models
- Gemini (Google): Function calling integration for Gemini models
- REST API: HTTP endpoint for any LLM that can make REST calls (OpenAI, LLaMA, etc.)
- Logging connectors, webhooks, ML platforms, and LangChain (legacy)
- Vector Stores: Pinecone, Weaviate, Chroma, Qdrant with governance
- ML Platforms: MLflow, W&B, SageMaker, Azure ML, Ray Serve
- LLM Providers: Cohere, Mistral, Together, Fireworks, Groq, Replicate
- Agent Frameworks: LlamaIndex, CrewAI, DSPy, AutoGen

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

5. Vector Stores:
    from nethical.integrations.vector_stores import ChromaConnector
    
    connector = ChromaConnector(collection_name="my_collection")
    connector.upsert(vectors)
    results = connector.query([...], top_k=10)

6. ML Platforms:
    from nethical.integrations.mlflow_connector import MLflowConnector
    
    mlflow = MLflowConnector(tracking_uri="file:./mlruns")
    run_id = mlflow.start_run("my_experiment")
    mlflow.log_metrics(run_id, {"accuracy": 0.95})

7. Simple evaluation:
    from nethical.integrations.claude_tools import evaluate_action
    
    decision = evaluate_action("Write code to delete files")
    if decision != "ALLOW":
        # Block the action
        pass

8. LLM Providers (Cohere, Mistral, etc.):
    from nethical.integrations.llm_providers import CohereProvider
    
    provider = CohereProvider(api_key="your-key")
    response = provider.safe_generate("Tell me about AI safety")
    print(f"Response: {response.content}")
    print(f"Risk Score: {response.risk_score}")

9. Agent Frameworks (LlamaIndex, CrewAI, etc.):
    from nethical.integrations.agent_frameworks import NethicalLlamaIndexTool
    
    tool = NethicalLlamaIndexTool(block_threshold=0.7)
    result = tool("Check if this action is safe")
"""

from typing import Dict, Any

# Legacy integrations
__all__ = ["logging_connectors", "webhook", "ml_platforms", "langchain_tools", "mlflow_connector", "ray_serve_connector", "llm_providers", "agent_frameworks"]

# LLM Providers
try:
    from .llm_providers import (
        LLMProviderBase,
        LLMResponse,
        get_provider_info,
    )
    LLM_PROVIDERS_AVAILABLE = True
    __all__.extend([
        "LLMProviderBase",
        "LLMResponse",
        "get_provider_info",
    ])
except ImportError:
    LLM_PROVIDERS_AVAILABLE = False

# Agent Frameworks
try:
    from .agent_frameworks import (
        AgentFrameworkBase,
        AgentWrapper,
        GovernanceDecision,
        GovernanceResult,
        get_framework_info,
    )
    AGENT_FRAMEWORKS_AVAILABLE = True
    __all__.extend([
        "AgentFrameworkBase",
        "AgentWrapper",
        "GovernanceDecision",
        "GovernanceResult",
        "get_framework_info",
    ])
except ImportError:
    AGENT_FRAMEWORKS_AVAILABLE = False

# Vector stores
try:
    from .vector_stores import (
        VectorStoreProvider,
        VectorSearchResult,
        PINECONE_AVAILABLE,
        WEAVIATE_AVAILABLE,
        CHROMA_AVAILABLE,
        QDRANT_AVAILABLE,
    )
    VECTOR_STORES_AVAILABLE = True
    __all__.extend([
        "VectorStoreProvider",
        "VectorSearchResult",
        "PINECONE_AVAILABLE",
        "WEAVIATE_AVAILABLE",
        "CHROMA_AVAILABLE",
        "QDRANT_AVAILABLE",
    ])
except ImportError:
    VECTOR_STORES_AVAILABLE = False

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

__all__.extend(["CLAUDE_AVAILABLE", "REST_API_AVAILABLE", "GROK_AVAILABLE", "GEMINI_AVAILABLE", "VECTOR_STORES_AVAILABLE", "LLM_PROVIDERS_AVAILABLE", "AGENT_FRAMEWORKS_AVAILABLE", "get_integration_info"])


def get_integration_info() -> Dict[str, Any]:
    """Get information about available integrations.
    
    Returns:
        Dict with integration availability and setup instructions
    """
    info = {
        "claude": {
            "available": CLAUDE_AVAILABLE,
            "setup": "pip install anthropic",
            "docs": "See nethical.integrations.claude_tools",
            "manifest": "config/integrations/ai-plugin.json"
        },
        "grok": {
            "available": GROK_AVAILABLE,
            "setup": "pip install xai-sdk (when available) or use REST API",
            "docs": "See nethical.integrations.grok_tools",
            "manifest": "config/integrations/grok-manifest.json"
        },
        "gemini": {
            "available": GEMINI_AVAILABLE,
            "setup": "pip install google-generativeai",
            "docs": "See nethical.integrations.gemini_tools",
            "manifest": "config/integrations/gemini-manifest.json"
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
            "manifest": "config/integrations/ai-plugin.json"
        },
        "llama": {
            "available": True,
            "setup": "Use REST API endpoint",
            "docs": "See nethical.integrations.rest_api"
        },
        "vector_stores": {
            "available": VECTOR_STORES_AVAILABLE,
            "setup": "pip install pinecone-client weaviate-client chromadb qdrant-client",
            "docs": "See docs/VECTOR_STORE_INTEGRATION_GUIDE.md",
            "manifest": "config/integrations/vector-stores-mcp.yaml",
            "providers": {
                "pinecone": PINECONE_AVAILABLE if VECTOR_STORES_AVAILABLE else False,
                "weaviate": WEAVIATE_AVAILABLE if VECTOR_STORES_AVAILABLE else False,
                "chroma": CHROMA_AVAILABLE if VECTOR_STORES_AVAILABLE else False,
                "qdrant": QDRANT_AVAILABLE if VECTOR_STORES_AVAILABLE else False,
            }
        },
        "ml_platforms": {
            "available": True,
            "setup": "pip install mlflow wandb boto3 sagemaker azureml-core ray[serve]",
            "docs": "See nethical.integrations.ml_platforms, mlflow_connector",
            "manifest": "config/integrations/mlflow-integration.yaml"
        },
        "llm_providers": {
            "available": LLM_PROVIDERS_AVAILABLE,
            "setup": "pip install cohere mistralai together fireworks-ai groq replicate",
            "docs": "See docs/LLM_PROVIDERS_GUIDE.md",
            "manifest": "config/integrations/llm-providers-mcp.yaml",
            "providers": ["cohere", "mistral", "together", "fireworks", "groq", "replicate"]
        },
        "agent_frameworks": {
            "available": AGENT_FRAMEWORKS_AVAILABLE,
            "setup": "pip install llama-index crewai dspy-ai pyautogen",
            "docs": "See docs/AGENT_FRAMEWORKS_GUIDE.md",
            "frameworks": ["llamaindex", "crewai", "dspy", "autogen"]
        }
    }
    return info


if __name__ == "__main__":
    info = get_integration_info()
    print("Nethical LLM Integrations:")
    for name, details in info.items():
        status = "✓ Available" if details["available"] else "✗ Not Available"
        print(f"\n{name.upper()}: {status}")
        print(f"  Setup: {details['setup']}")
        print(f"  Docs: {details['docs']}")
