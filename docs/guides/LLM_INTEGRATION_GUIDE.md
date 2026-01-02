# LLM Integration Guide

Complete guide for integrating Nethical with all major Large Language Model (LLM) platforms.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Supported LLM Platforms](#supported-llm-platforms)
- [Integration Methods](#integration-methods)
- [Examples by Platform](#examples-by-platform)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

Nethical provides universal safety and ethics governance for LLM applications. It can be integrated with any LLM platform through:

1. **Function Calling / Tool Use** - Native integration with platforms that support tools
2. **REST API** - HTTP endpoint for any LLM that can make API calls
3. **SDK Wrappers** - Python modules for direct integration

### Key Benefits

- 🛡️ **Safety First**: Real-time safety checks before and after LLM operations
- 🔍 **PII Detection**: Automatic detection and redaction of sensitive data
- 📊 **Risk Scoring**: Calculate risk scores for every action (0.0-1.0)
- 📝 **Audit Trails**: Immutable logging with Merkle anchoring
- ✅ **Compliance**: OWASP LLM Top 10, GDPR, HIPAA, NIST AI RMF coverage

## Quick Start

### Installation

```bash
pip install nethical
```

### Basic Usage Pattern

```python
from nethical.integrations import evaluate_action

# Check an action
decision = evaluate_action(
    action="User prompt or LLM output",
    agent_id="my-llm-agent",
    action_type="query"
)

if decision != "ALLOW":
    # Block or handle the action
    print("Action blocked for safety")
```

## Supported LLM Platforms

| Platform | Integration Type | Module | Manifest |
|----------|-----------------|--------|----------|
| **OpenAI (GPT-4, GPT-3.5)** | REST API, Plugin | `rest_api` | `config/integrations/ai-plugin.json` |
| **Anthropic Claude** | Function Calling | `claude_tools` | `config/integrations/ai-plugin.json` |
| **xAI Grok** | Function Calling | `grok_tools` | `config/integrations/grok-manifest.json` |
| **Google Gemini** | Function Calling | `gemini_tools` | `config/integrations/gemini-manifest.json` |
| **Meta LLaMA** | REST API | `rest_api` | - |
| **Custom LLMs** | REST API | `rest_api` | - |

## Integration Methods

### Method 1: Function Calling / Tool Use

Best for: Claude, Grok, Gemini, OpenAI (with function calling)

**Advantages:**
- Native integration with LLM
- LLM can self-check actions
- Automatic context awareness

**Example (Claude):**

```python
from anthropic import Anthropic
from nethical.integrations.claude_tools import get_nethical_tool, handle_nethical_tool

client = Anthropic(api_key="your-key")

# Get tool definition
tools = [get_nethical_tool()]

# Use in conversation
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
            print(f"Decision: {result['decision']}")
            print(f"Reason: {result['reason']}")
```

### Method 2: REST API

Best for: Any LLM, custom integrations, microservices

**Advantages:**
- Universal compatibility
- Language agnostic
- Simple HTTP interface
- Works with any LLM

**Start the server:**

```bash
# Method 1: Direct
python -m nethical.integrations.rest_api

# Method 2: With uvicorn
uvicorn nethical.integrations.rest_api:app --host 0.0.0.0 --port 8000
```

**Client example (Python):**

```python
import requests

response = requests.post(
    "http://localhost:8000/evaluate",
    json={
        "action": "Generate code to access database",
        "agent_id": "my-llm-agent",
        "action_type": "code_generation"
    }
)

result = response.json()
if result["decision"] != "ALLOW":
    print(f"Action blocked: {result['reason']}")
```

**Client example (JavaScript/TypeScript):**

```javascript
const response = await fetch('http://localhost:8000/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        action: 'Access user email addresses',
        agent_id: 'gpt-4',
        action_type: 'data_access'
    })
});

const result = await response.json();
if (result.decision !== 'ALLOW') {
    console.log('Action blocked:', result.reason);
}
```

### Method 3: SDK Wrapper

Best for: Python applications with direct LLM SDK usage

**Example:**

```python
from nethical.integrations.claude_tools import evaluate_action

def safe_llm_call(prompt):
    # Pre-check
    if evaluate_action(prompt) != "ALLOW":
        return "Input blocked for safety"
    
    # Call LLM
    response = llm.generate(prompt)
    
    # Post-check
    if evaluate_action(response, action_type="generated_content") != "ALLOW":
        return "Output filtered for safety"
    
    return response
```

## Examples by Platform

### OpenAI (GPT-4, GPT-3.5)

OpenAI supports both REST API and plugin integration.

#### REST API Integration

```python
import openai
from nethical.integrations import evaluate_action

def safe_openai_chat(messages):
    # Check user input
    user_message = messages[-1]["content"]
    if evaluate_action(user_message, agent_id="gpt-4") != "ALLOW":
        return {"error": "Input blocked"}
    
    # Call OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    
    # Check output
    output = response.choices[0].message.content
    if evaluate_action(output, action_type="generated_content") != "ALLOW":
        return {"error": "Output filtered"}
    
    return response
```

#### Plugin Integration

Register Nethical as a ChatGPT plugin using `ai-plugin.json`:

1. Host the `ai-plugin.json` and `openapi.yaml` files
2. Register at https://platform.openai.com/docs/plugins
3. ChatGPT can now discover and use Nethical for safety checks

### Anthropic Claude

Claude has excellent function calling support.

```python
from anthropic import Anthropic
from nethical.integrations.claude_tools import (
    get_nethical_tool, 
    handle_nethical_tool,
    check_user_input,
    check_generated_content
)

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def safe_claude_chat(user_message):
    # Pre-check (optional - can let Claude decide)
    input_check = check_user_input(user_message, agent_id="claude-3")
    if input_check["decision"] != "ALLOW":
        return f"Input blocked: {input_check['reason']}"
    
    # Get tool definition
    tools = [get_nethical_tool()]
    
    # Create message with tools
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=[{"role": "user", "content": user_message}]
    )
    
    # Handle tool use
    if response.stop_reason == "tool_use":
        for block in response.content:
            if block.type == "tool_use" and block.name == "nethical_guard":
                result = handle_nethical_tool(block.input)
                # Send result back to Claude for final decision
                # ... (continue conversation)
    
    return response
```

### xAI Grok

```python
from nethical.integrations.grok_tools import (
    get_nethical_tool,
    handle_nethical_tool,
    check_user_input,
    check_code_generation
)

# Similar pattern to Claude
tools = [get_nethical_tool()]

# Use with Grok client (when available)
# response = grok_client.chat(
#     messages=[{"role": "user", "content": "..."}],
#     tools=tools
# )
```

### Google Gemini

```python
import google.generativeai as genai
from nethical.integrations.gemini_tools import (
    get_nethical_tool,
    handle_nethical_tool,
    safe_gemini_chat
)

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Method 1: Use built-in safe wrapper
response = safe_gemini_chat(
    "Tell me about AI safety",
    model_name="gemini-pro",
    check_input=True,
    check_output=True
)

# Method 2: Manual integration
tools = [get_nethical_tool()]
model = genai.GenerativeModel('gemini-pro', tools=tools)

response = model.generate_content("Your prompt here")

# Handle function calls
for part in response.parts:
    if hasattr(part, 'function_call'):
        result = handle_nethical_tool(part.function_call.args)
        print(f"Safety check: {result['decision']}")
```

### Meta LLaMA (via Ollama or other)

```python
import requests
from nethical.integrations import evaluate_action

def safe_llama_chat(prompt, model="llama2"):
    # Pre-check
    if evaluate_action(prompt, agent_id=model) != "ALLOW":
        return "Input blocked"
    
    # Call LLaMA (via Ollama API)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}
    )
    
    output = response.json()["response"]
    
    # Post-check
    if evaluate_action(output, action_type="generated_content") != "ALLOW":
        return "Output filtered"
    
    return output
```

## Best Practices

### 1. Bidirectional Checking

Always check both inputs and outputs for maximum safety:

```python
def safe_llm_interaction(user_input, llm_func):
    # Pre-check
    input_result = evaluate_action(user_input, action_type="user_input")
    if input_result != "ALLOW":
        return {"error": "Input blocked", "reason": input_result}
    
    # Execute
    output = llm_func(user_input)
    
    # Post-check
    output_result = evaluate_action(output, action_type="generated_content")
    if output_result != "ALLOW":
        return {"error": "Output filtered", "reason": output_result}
    
    return {"result": output}
```

### 2. Context-Aware Checking

Provide context for better decisions:

```python
result = evaluate_action(
    action="SELECT * FROM users",
    agent_id="gpt-4",
    action_type="code_generation",
    context={
        "language": "sql",
        "environment": "production",
        "user_role": "analyst"
    }
)
```

### 3. Handle All Decision Types

```python
decision = result["decision"]

if decision == "ALLOW":
    # Proceed normally
    pass
elif decision == "RESTRICT":
    # Apply restrictions (e.g., redact PII)
    cleaned = redact_pii(content)
elif decision == "BLOCK":
    # Block the action
    return "Action blocked for safety"
elif decision == "TERMINATE":
    # Critical violation - terminate session
    session.terminate()
```

### 4. Leverage Metadata

```python
result = handle_nethical_tool({"action": content})

print(f"Risk Score: {result['risk_score']}")
print(f"PII Detected: {result['pii_detected']}")
print(f"PII Types: {result['pii_types']}")
print(f"Violations: {result['violations']}")
print(f"Audit ID: {result['audit_id']}")
```

### 5. Production Configuration

```python
from nethical.core.integrated_governance import IntegratedGovernance

governance = IntegratedGovernance(
    storage_dir="./production_data",
    enable_quota_enforcement=True,
    enable_performance_optimization=True,
    enable_merkle_anchoring=True,
    enable_ethical_taxonomy=True,
    enable_sla_monitoring=True,
    region_id="us-east-1"
)
```

## Integration Patterns

### Pattern 1: Pre-check (Input Validation)

Check user input before sending to LLM:

```python
def process_user_query(query: str):
    result = evaluate_action(query, action_type="user_query")
    
    if result != "ALLOW":
        return "Query blocked for safety reasons"
    
    return llm.generate(query)
```

### Pattern 2: Post-check (Output Filtering)

Check LLM output before returning to user:

```python
def generate_response(prompt: str):
    llm_output = llm.generate(prompt)
    
    result = evaluate_action(llm_output, action_type="generated_content")
    
    if result != "ALLOW":
        return "Response filtered for safety"
    
    return llm_output
```

### Pattern 3: Bidirectional (Complete Protection)

Check both input and output:

```python
def safe_llm_interaction(user_input: str):
    # Pre-check
    if evaluate_action(user_input) != "ALLOW":
        return "Input blocked"
    
    # Generate
    output = llm.generate(user_input)
    
    # Post-check
    if evaluate_action(output) != "ALLOW":
        return "Output filtered"
    
    return output
```

### Pattern 4: Agent Self-Check

Let the LLM agent check itself using tools:

```python
# The LLM has access to nethical_guard tool
# It can call it proactively to check its own actions
tools = [get_nethical_tool()]

response = llm.generate(
    prompt="Before executing this action, check it for safety",
    tools=tools
)
```

## Troubleshooting

### REST API Connection Issues

```bash
# Check if server is running
curl http://localhost:8000/health

# Check logs
uvicorn nethical.integrations.rest_api:app --log-level debug
```

### Function Calling Not Working

1. Ensure LLM supports function calling
2. Verify tool definition format matches LLM requirements
3. Check LLM version compatibility

### High Latency

```python
# Enable performance optimization
governance = IntegratedGovernance(
    enable_performance_optimization=True,
    enable_quota_enforcement=False  # Disable if not needed
)
```

### False Positives

```python
# Adjust thresholds (requires governance instance)
governance.config.risk_threshold = 0.8  # Higher = less strict
```

## Additional LLM Providers

Nethical now supports additional LLM providers with governed wrappers:

### Cohere

```python
from nethical.integrations.llm_providers import CohereProvider

provider = CohereProvider(
    api_key="your-key",
    model="command-r-plus",
    check_input=True,
    check_output=True
)

response = provider.safe_generate("Tell me about AI safety")
print(f"Response: {response.content}")
print(f"Risk Score: {response.risk_score}")
```

### Mistral AI

```python
from nethical.integrations.llm_providers import MistralProvider

provider = MistralProvider(
    api_key="your-key",
    model="mistral-large-latest"
)

response = provider.safe_generate("Explain machine learning")
```

### Together AI, Fireworks, Groq, Replicate

All providers follow the same pattern:

```python
from nethical.integrations.llm_providers import (
    TogetherProvider,
    FireworksProvider,
    GroqProvider,
    ReplicateProvider
)

# Each provider has the same interface
provider = GroqProvider(api_key="...", model="llama-3.1-70b-versatile")
response = provider.safe_generate("Hello!")
```

For detailed documentation, see [LLM Providers Guide](./LLM_PROVIDERS_GUIDE.md).

## Agent Frameworks

Nethical integrates with popular agent frameworks:

### LlamaIndex

```python
from nethical.integrations.agent_frameworks import (
    NethicalLlamaIndexTool,
    NethicalQueryEngine
)

# Tool for agents
tool = NethicalLlamaIndexTool(block_threshold=0.7)

# Query engine wrapper
safe_engine = NethicalQueryEngine(query_engine, check_query=True)
```

### CrewAI

```python
from nethical.integrations.agent_frameworks import NethicalCrewAITool

tool = NethicalCrewAITool(block_threshold=0.7)
crewai_tool = tool.as_crewai_tool()
```

### DSPy

```python
from nethical.integrations.agent_frameworks import GovernedChainOfThought

cot = GovernedChainOfThought("question -> answer")
result = cot(question="What is AI?")
```

### AutoGen

```python
from nethical.integrations.agent_frameworks import NethicalAutoGenTool

tool = NethicalAutoGenTool(block_threshold=0.7)
func_config = tool.get_function_config()
```

For detailed documentation, see [Agent Frameworks Guide](./AGENT_FRAMEWORKS_GUIDE.md).

## Additional Resources

- [LLM Providers Guide](./LLM_PROVIDERS_GUIDE.md)
- [Agent Frameworks Guide](./AGENT_FRAMEWORKS_GUIDE.md)
- [API Reference](./EXTERNAL_INTEGRATIONS_GUIDE.md)
- [OpenAPI Specification](../openapi.yaml)
- [Examples Directory](../examples/integrations/)
- [GitHub Repository](https://github.com/V1B3hR/nethical)

## Support

- 📧 Email: support@nethical.dev
- 💬 Discussions: https://github.com/V1B3hR/nethical/discussions
- 🐛 Issues: https://github.com/V1B3hR/nethical/issues

## License

MIT License - See [LICENSE](../LICENSE) for details.
