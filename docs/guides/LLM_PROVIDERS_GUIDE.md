# LLM Providers Guide

Complete guide for integrating Nethical with additional LLM providers including Cohere, Mistral, Together AI, Fireworks AI, Groq, and Replicate.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Providers](#providers)
- [Common Features](#common-features)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Overview

Nethical provides governed wrappers for multiple LLM providers, enabling consistent safety and ethics checks across different platforms. All providers inherit from `LLMProviderBase` and include:

- **Input Governance**: Check prompts before sending to the LLM
- **Output Governance**: Check generated content before returning
- **Risk Scoring**: Calculate risk scores (0.0-1.0) for all content
- **Tool Definitions**: Function calling support for each provider
- **Configurable Thresholds**: Customize blocking and restriction levels

## Quick Start

### Installation

```bash
# Install Nethical
pip install nethical

# Install provider SDK (choose one or more)
pip install cohere        # Cohere
pip install mistralai     # Mistral AI
pip install together      # Together AI
pip install fireworks-ai  # Fireworks AI
pip install groq          # Groq
pip install replicate     # Replicate
```

### Basic Usage

```python
from nethical.integrations.llm_providers import CohereProvider

# Create provider with governance
provider = CohereProvider(
    api_key="your-api-key",
    model="command-r-plus",
    check_input=True,
    check_output=True,
    block_threshold=0.7
)

# Safe generation with automatic checks
response = provider.safe_generate("Tell me about AI safety")

print(f"Response: {response.content}")
print(f"Risk Score: {response.risk_score}")
print(f"Model: {response.model}")
```

## Providers

### Cohere

Cohere integration supports chat generation and reranking with governance.

```python
from nethical.integrations.llm_providers import CohereProvider

provider = CohereProvider(
    api_key="your-cohere-key",
    model="command-r-plus"  # or command-r, command, command-light
)

# Chat generation
response = provider.safe_generate("What is machine learning?")

# Reranking with governance
documents = ["Doc 1...", "Doc 2...", "Doc 3..."]
results = provider.safe_rerank(
    query="machine learning",
    documents=documents,
    top_n=5
)

# Each result includes governance info
for result in results:
    print(f"Document: {result['document'][:50]}...")
    print(f"Relevance: {result['relevance_score']}")
    print(f"Risk: {result['risk_score']}")
```

**Available Models:**
- `command-r-plus` (recommended)
- `command-r`
- `command`
- `command-light`

### Mistral AI

Mistral AI integration with function calling support.

```python
from nethical.integrations.llm_providers import MistralProvider

provider = MistralProvider(
    api_key="your-mistral-key",
    model="mistral-large-latest"
)

response = provider.safe_generate("Explain quantum computing")
```

**Available Models:**
- `mistral-large-latest`
- `mistral-medium-latest`
- `mistral-small-latest`
- `open-mistral-7b`
- `open-mixtral-8x7b`

### Together AI

Together AI provides access to various open-source models.

```python
from nethical.integrations.llm_providers import TogetherProvider

provider = TogetherProvider(
    api_key="your-together-key",
    model="meta-llama/Llama-3-70b-chat-hf"
)

response = provider.safe_generate("Write a haiku about AI")
```

**Available Models:**
- `meta-llama/Llama-3-70b-chat-hf`
- `meta-llama/Llama-3-8b-chat-hf`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- Many more open-source models

### Fireworks AI

Fireworks AI for fast inference on various models.

```python
from nethical.integrations.llm_providers import FireworksProvider

provider = FireworksProvider(
    api_key="your-fireworks-key",
    model="accounts/fireworks/models/llama-v3-70b-instruct"
)

response = provider.safe_generate("Summarize this text...")
```

### Groq

Groq provides ultra-fast inference using their LPU technology.

```python
from nethical.integrations.llm_providers import GroqProvider

provider = GroqProvider(
    api_key="your-groq-key",
    model="llama-3.1-70b-versatile"
)

# Ultra-fast inference with governance
response = provider.safe_generate("Quick question: What is AI?")
```

**Available Models:**
- `llama-3.1-70b-versatile`
- `llama-3.1-8b-instant`
- `mixtral-8x7b-32768`
- `gemma-7b-it`

### Replicate

Replicate allows running various open-source models.

```python
from nethical.integrations.llm_providers import ReplicateProvider

provider = ReplicateProvider(
    api_key="your-replicate-key",
    model="meta/llama-2-70b-chat"
)

response = provider.safe_generate("Hello, how are you?")
```

## Common Features

### LLMResponse Object

All providers return an `LLMResponse` object:

```python
@dataclass
class LLMResponse:
    content: str                              # Generated text
    model: str                                # Model identifier
    usage: Dict[str, int]                     # Token usage
    governance_result: Optional[Dict] = None  # Full governance data
    risk_score: float = 0.0                   # Risk score (0.0-1.0)
```

### Governance Thresholds

Configure blocking and restriction thresholds:

```python
provider = CohereProvider(
    api_key="...",
    block_threshold=0.7,    # Block if risk > 0.7
    check_input=True,       # Check input prompts
    check_output=True       # Check generated output
)
```

### Tool Definitions

Get tool definitions for function calling:

```python
# Cohere format
from nethical.integrations.llm_providers.cohere_tools import get_nethical_tool
tool = get_nethical_tool()

# Mistral/OpenAI format
from nethical.integrations.llm_providers.mistral_tools import get_nethical_tool
tool = get_nethical_tool()
```

### Handling Tool Calls

```python
from nethical.integrations.llm_providers.cohere_tools import handle_nethical_tool

# When LLM makes a tool call
result = handle_nethical_tool({
    "action": "Delete user data",
    "action_type": "data_operation"
})

if result["decision"] == "BLOCK":
    print(f"Action blocked: {result['reason']}")
```

## API Reference

### LLMProviderBase

Abstract base class for all providers.

```python
class LLMProviderBase(ABC):
    def __init__(
        self,
        check_input: bool = True,
        check_output: bool = True,
        block_threshold: float = 0.7,
        agent_id: Optional[str] = None,
        storage_dir: str = "./nethical_data"
    ):
        ...
    
    def safe_generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate with governance checks."""
        ...
    
    @abstractmethod
    def _generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Implement actual generation."""
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model identifier."""
        ...
```

### Provider Info

Get information about available providers:

```python
from nethical.integrations.llm_providers import get_provider_info

info = get_provider_info()
for name, details in info.items():
    print(f"{name}: {'Available' if details['available'] else 'Not installed'}")
    print(f"  Setup: {details['setup']}")
```

## Examples

### Multi-Provider Setup

```python
from nethical.integrations.llm_providers import (
    CohereProvider,
    GroqProvider,
    MistralProvider
)

# Create providers with consistent governance
providers = {
    "cohere": CohereProvider(api_key="...", model="command-r-plus"),
    "groq": GroqProvider(api_key="...", model="llama-3.1-70b-versatile"),
    "mistral": MistralProvider(api_key="...", model="mistral-large-latest")
}

# Use any provider with same interface
for name, provider in providers.items():
    response = provider.safe_generate("What is AI?")
    print(f"{name}: {response.content[:100]}...")
```

### Custom Provider

Create your own provider by extending `LLMProviderBase`:

```python
from nethical.integrations.llm_providers import LLMProviderBase, LLMResponse

class MyCustomProvider(LLMProviderBase):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
    
    @property
    def model_name(self) -> str:
        return "my-custom-model"
    
    def _generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Your API call here
        result = my_api_call(prompt, self.api_key)
        
        return LLMResponse(
            content=result["text"],
            model=self.model_name,
            usage=result.get("usage", {})
        )

# Use with automatic governance
provider = MyCustomProvider(api_key="...", block_threshold=0.7)
response = provider.safe_generate("Hello!")
```

## Best Practices

### 1. Always Check Both Input and Output

```python
provider = CohereProvider(
    api_key="...",
    check_input=True,   # Prevent prompt injection
    check_output=True   # Filter harmful outputs
)
```

### 2. Handle Blocked Content Gracefully

```python
response = provider.safe_generate(user_input)

if response.risk_score > 0.7:
    # Response was blocked or filtered
    print("Content was filtered for safety")
else:
    # Safe to use
    print(response.content)
```

### 3. Use Appropriate Thresholds

```python
# Strict (recommended for production)
provider = CohereProvider(api_key="...", block_threshold=0.6)

# Balanced
provider = CohereProvider(api_key="...", block_threshold=0.7)

# Permissive (for testing only)
provider = CohereProvider(api_key="...", block_threshold=0.9)
```

### 4. Monitor Token Usage

```python
response = provider.safe_generate("...")

if response.usage:
    print(f"Input tokens: {response.usage.get('input_tokens', 0)}")
    print(f"Output tokens: {response.usage.get('output_tokens', 0)}")
```

## Troubleshooting

### Provider Not Available

```python
from nethical.integrations.llm_providers import COHERE_AVAILABLE

if not COHERE_AVAILABLE:
    print("Install with: pip install cohere")
```

### High Risk Scores

If legitimate content is being blocked:

1. Check the governance result for details
2. Review the risk tier and specific concerns
3. Consider adjusting thresholds for your use case

```python
response = provider.safe_generate("...")

if response.governance_result:
    print(f"Risk tier: {response.governance_result.get('phase3', {}).get('risk_tier')}")
```

## Resources

- [LLM Integration Guide](./LLM_INTEGRATION_GUIDE.md)
- [Agent Frameworks Guide](./AGENT_FRAMEWORKS_GUIDE.md)
- [API Reference](./API_USAGE.md)
- [Examples Directory](../examples/)
