"""
LLM Provider integrations with Nethical governance.

This package provides governed wrappers for various LLM providers:

- Cohere: Chat generation and reranking
- Mistral: Mistral AI models
- Together: Together AI inference
- Fireworks: Fireworks AI fast inference
- Groq: Ultra-fast LPU inference
- Replicate: Various open-source models

All providers inherit from LLMProviderBase and include:
- Input/output governance checks
- Risk scoring and blocking
- Tool definitions for function calling
- Configurable thresholds

Example:
    from nethical.integrations.llm_providers import CohereProvider
    
    provider = CohereProvider(
        api_key="your-api-key",
        model="command-r-plus",
        check_input=True,
        check_output=True
    )
    
    response = provider.safe_generate("Tell me about AI safety")
    print(f"Response: {response.content}")
    print(f"Risk Score: {response.risk_score}")
"""

from .base import LLMProviderBase, LLMResponse

# Import providers (graceful failures if dependencies missing)
try:
    from .cohere_tools import CohereProvider
    from .cohere_tools import get_nethical_tool as get_cohere_tool
    from .cohere_tools import handle_nethical_tool as handle_cohere_tool
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    CohereProvider = None

try:
    from .mistral_tools import MistralProvider
    from .mistral_tools import get_nethical_tool as get_mistral_tool
    from .mistral_tools import handle_nethical_tool as handle_mistral_tool
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    MistralProvider = None

try:
    from .together_tools import TogetherProvider
    from .together_tools import get_nethical_tool as get_together_tool
    from .together_tools import handle_nethical_tool as handle_together_tool
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    TogetherProvider = None

try:
    from .fireworks_tools import FireworksProvider
    from .fireworks_tools import get_nethical_tool as get_fireworks_tool
    from .fireworks_tools import handle_nethical_tool as handle_fireworks_tool
    FIREWORKS_AVAILABLE = True
except ImportError:
    FIREWORKS_AVAILABLE = False
    FireworksProvider = None

try:
    from .groq_tools import GroqProvider
    from .groq_tools import get_nethical_tool as get_groq_tool
    from .groq_tools import handle_nethical_tool as handle_groq_tool
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    GroqProvider = None

try:
    from .replicate_tools import ReplicateProvider
    from .replicate_tools import get_nethical_tool as get_replicate_tool
    from .replicate_tools import handle_nethical_tool as handle_replicate_tool
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    ReplicateProvider = None


__all__ = [
    # Base classes
    "LLMProviderBase",
    "LLMResponse",
    # Availability flags
    "COHERE_AVAILABLE",
    "MISTRAL_AVAILABLE",
    "TOGETHER_AVAILABLE",
    "FIREWORKS_AVAILABLE",
    "GROQ_AVAILABLE",
    "REPLICATE_AVAILABLE",
    # Providers (may be None if not available)
    "CohereProvider",
    "MistralProvider",
    "TogetherProvider",
    "FireworksProvider",
    "GroqProvider",
    "ReplicateProvider",
]


def get_provider_info():
    """Get information about available LLM providers.
    
    Returns:
        Dict with provider availability and setup instructions
    """
    return {
        "cohere": {
            "available": COHERE_AVAILABLE,
            "setup": "pip install cohere",
            "class": "CohereProvider",
            "features": ["chat", "rerank", "function_calling"]
        },
        "mistral": {
            "available": MISTRAL_AVAILABLE,
            "setup": "pip install mistralai",
            "class": "MistralProvider",
            "features": ["chat", "function_calling"]
        },
        "together": {
            "available": TOGETHER_AVAILABLE,
            "setup": "pip install together",
            "class": "TogetherProvider",
            "features": ["chat", "function_calling"]
        },
        "fireworks": {
            "available": FIREWORKS_AVAILABLE,
            "setup": "pip install fireworks-ai",
            "class": "FireworksProvider",
            "features": ["chat", "function_calling"]
        },
        "groq": {
            "available": GROQ_AVAILABLE,
            "setup": "pip install groq",
            "class": "GroqProvider",
            "features": ["chat", "function_calling", "ultra_fast"]
        },
        "replicate": {
            "available": REPLICATE_AVAILABLE,
            "setup": "pip install replicate",
            "class": "ReplicateProvider",
            "features": ["chat", "various_models"]
        }
    }
