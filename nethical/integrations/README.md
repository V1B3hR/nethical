# Nethical LLM Integrations

This directory contains integrations for using Nethical with various Large Language Model (LLM) platforms.

## Overview

Nethical provides two primary integration methods for LLMs:

1. **Claude (Anthropic)**: Tool-based integration using Claude's function calling
2. **REST API**: HTTP endpoint for any LLM (OpenAI, Gemini, LLaMA, etc.)

## Quick Start

### 1. Claude/Anthropic Integration

Use Nethical as a tool that Claude can call to check actions for safety and ethics.

```python
from anthropic import Anthropic
from nethical.integrations.claude_tools import get_nethical_tool, handle_nethical_tool

client = Anthropic(api_key="your-api-key")

# Get tool definition
tools = [get_nethical_tool()]

# Use in conversation
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Check if this is safe: ..."}]
)

# Handle tool calls
if response.stop_reason == "tool_use":
    for block in response.content:
        if block.type == "tool_use" and block.name == "nethical_guard":
            result = handle_nethical_tool(block.input)
            print(f"Decision: {result['decision']}")
```

**Installation:**
```bash
pip install anthropic
```

**Features:**
- Seamless integration with Claude's tool calling
- Automatic safety and ethics checks
- PII detection
- Risk scoring
- Audit trail generation

### 2. REST API Integration

Expose Nethical as an HTTP endpoint for any LLM platform.

**Start the server:**
```bash
# Method 1: Direct
python -m nethical.integrations.rest_api

# Method 2: With uvicorn
uvicorn nethical.integrations.rest_api:app --host 0.0.0.0 --port 8000
```

**Client usage (Python):**
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

**Client usage (JavaScript):**
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

**Installation:**
```bash
pip install fastapi uvicorn
```

**Features:**
- Works with any LLM that can make HTTP requests
- RESTful API design
- CORS enabled for browser use
- OpenAPI documentation at `/docs`
- Health check endpoint

## Integration Patterns

### Pattern 1: Pre-check (Before LLM)

Check user input before sending to LLM:

```python
def process_user_query(query: str):
    # Check with Nethical first
    result = evaluate_action(query, action_type="user_query")
    
    if result != "ALLOW":
        return "Query blocked for safety reasons"
    
    # Safe to send to LLM
    return llm.generate(query)
```

### Pattern 2: Post-check (After LLM)

Check LLM output before returning to user:

```python
def generate_response(prompt: str):
    # Get LLM response
    llm_output = llm.generate(prompt)
    
    # Check output with Nethical
    result = evaluate_action(llm_output, action_type="generated_content")
    
    if result != "ALLOW":
        return "Response filtered for safety"
    
    return llm_output
```

### Pattern 3: Bidirectional (Before and After)

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

## API Reference

### Claude Integration

#### `get_nethical_tool() -> Dict`
Returns the tool definition for Claude.

#### `handle_nethical_tool(tool_input: Dict) -> Dict`
Handles a tool call from Claude.

**Input:**
- `action` (required): The action to evaluate
- `agent_id` (optional): Agent identifier
- `action_type` (optional): Type of action
- `context` (optional): Additional context

**Output:**
- `decision`: "ALLOW", "RESTRICT", "BLOCK", or "TERMINATE"
- `reason`: Explanation for the decision
- `risk_score`: Risk score (0.0-1.0)
- `pii_detected`: Boolean if PII found
- `pii_types`: List of PII types detected
- Additional metadata

#### `evaluate_action(action: str, agent_id: str = "claude", **kwargs) -> str`
Simplified function that returns just the decision string.

### REST API

#### `POST /evaluate`
Evaluate an action.

**Request:**
```json
{
    "action": "string (required)",
    "agent_id": "string (default: 'unknown')",
    "action_type": "string (default: 'query')",
    "context": "object (optional)"
}
```

**Response:**
```json
{
    "decision": "ALLOW|RESTRICT|BLOCK|TERMINATE",
    "reason": "string",
    "agent_id": "string",
    "timestamp": "ISO 8601 string",
    "risk_score": "number (0.0-1.0, optional)",
    "pii_detected": "boolean (optional)",
    "pii_types": "array of strings (optional)",
    "audit_id": "string (optional)",
    "metadata": "object (optional)"
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "ISO 8601 string",
    "governance_enabled": true,
    "components": {}
}
```

#### `GET /`
API information endpoint.

#### `GET /docs`
Interactive API documentation (Swagger UI).

## Decision Types

- **ALLOW**: Action is safe and compliant
- **RESTRICT**: Action requires modifications or restrictions
- **BLOCK**: Action is blocked due to safety/ethical concerns
- **TERMINATE**: Critical violation detected, terminate immediately

## Examples

See the `examples/integrations/` directory for complete examples:

- `claude_example.py`: Full Claude integration example
- `rest_api_example.py`: REST API usage examples with multiple LLMs

Run examples:
```bash
# Claude example (requires ANTHROPIC_API_KEY)
python examples/integrations/claude_example.py

# REST API examples (requires server running)
python -m nethical.integrations.rest_api  # In one terminal
python examples/integrations/rest_api_example.py  # In another terminal
```

## Testing

Run integration tests:
```bash
# Test Claude integration
pytest tests/test_claude_integration.py -v

# Test REST API
pytest tests/test_rest_api_integration.py -v

# Test all integrations
pytest tests/test_*_integration.py -v
```

## Configuration

Both integrations use the `IntegratedGovernance` system. You can customize:

```python
from nethical.integrations.claude_tools import get_governance_instance

governance = get_governance_instance(
    storage_dir="./custom_data_dir",
    enable_quota=True,
    region_id="us-west-1",
    # ... other IntegratedGovernance options
)
```

## Use Cases

### OpenAI Integration
```python
import openai
from nethical.integrations import evaluate_action

def safe_openai_call(prompt):
    if evaluate_action(prompt) != "ALLOW":
        return "Unsafe prompt"
    return openai.ChatCompletion.create(...)
```

### Google Gemini Integration
```python
import google.generativeai as genai
from nethical.integrations import evaluate_action

def safe_gemini_call(prompt):
    if evaluate_action(prompt) != "ALLOW":
        return "Unsafe prompt"
    return genai.generate_text(prompt=prompt)
```

### Custom LLM Integration
```python
import requests

def check_with_nethical(action):
    resp = requests.post(
        "http://localhost:8000/evaluate",
        json={"action": action}
    )
    return resp.json()["decision"] == "ALLOW"

if check_with_nethical(user_input):
    # Proceed with your LLM
    pass
```

## Security Considerations

1. **API Key Management**: Keep API keys secure (use environment variables)
2. **Network Security**: Use HTTPS in production
3. **Rate Limiting**: Consider implementing rate limiting for the REST API
4. **Input Validation**: Both integrations validate inputs, but additional validation may be needed
5. **Audit Trails**: All decisions are logged for auditability

## Performance

- Claude integration: Adds ~100-500ms per tool call
- REST API: Adds ~50-200ms per request
- Both support concurrent requests
- Results can be cached if needed

## Troubleshooting

### Claude Integration

**Error: "anthropic package not installed"**
```bash
pip install anthropic
```

**Error: "ANTHROPIC_API_KEY not set"**
```bash
export ANTHROPIC_API_KEY=your-api-key
```

### REST API

**Error: "Cannot connect to API"**
```bash
# Start the server
python -m nethical.integrations.rest_api
```

**Error: "Port already in use"**
```bash
# Use different port
uvicorn nethical.integrations.rest_api:app --port 8001
```

## Contributing

When adding new integrations:

1. Create module in `nethical/integrations/`
2. Add tests in `tests/test_*_integration.py`
3. Add examples in `examples/integrations/`
4. Update this README
5. Update `__init__.py` to export key functions

## License

Same as Nethical main project (MIT License).

## Support

For issues or questions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: See main Nethical README
