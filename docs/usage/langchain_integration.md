# LangChain Integration for Nethical

This guide explains how to integrate Nethical's IntegratedGovernance system with LangChain agents for comprehensive AI safety and ethics checking.

## Overview

The Nethical-LangChain integration provides:

- **NethicalGuardTool**: A LangChain `BaseTool` wrapper for Nethical's governance system
- **Pre-action and post-action evaluation**: Check both user inputs and agent outputs
- **Configurable decision thresholds**: Customize when to ALLOW, WARN, or BLOCK
- **Detailed governance reporting**: Optional detailed results from all governance phases
- **Optional LlamaGuard integration**: Chain multiple guards for layered protection

## Installation

### Basic Installation

```bash
pip install nethical
```

### With LangChain Support

```bash
pip install nethical
pip install langchain langchain-core
```

### With Optional LlamaGuard Support

```bash
pip install nethical
pip install langchain langchain-core
pip install transformers torch  # For local LlamaGuard models
```

## Quick Start

### Basic Usage

```python
from nethical.integrations.langchain_tools import NethicalGuardTool

# Create the guard tool
guard = NethicalGuardTool(
    storage_dir="./nethical_data",
    block_threshold=0.7,
    warn_threshold=0.4
)

# Evaluate an action
result = guard._run(
    action="What is the weather today?",
    agent_id="my_agent",
    action_type="query"
)

print(result)
# Output: "Decision: ALLOW | Risk Score: 0.15 | Risk Tier: LOW"
```

### Integration with LangChain Agents

```python
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool
from nethical.integrations.langchain_tools import NethicalGuardTool

# Create LLM
llm = OpenAI(temperature=0)

# Create Nethical guard tool
nethical_guard = NethicalGuardTool(
    storage_dir="./nethical_data",
    block_threshold=0.7
)

# Create other tools
calculator = Tool(
    name="Calculator",
    func=lambda x: str(eval(x)),
    description="Useful for math calculations"
)

# Create agent with Nethical guard
tools = [nethical_guard, calculator]
agent = initialize_agent(
    tools, 
    llm, 
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
response = agent.run("What is 25 * 4?")
```

### Using the Helper Function

```python
from langchain.llms import OpenAI
from langchain.tools import Tool
from nethical.integrations.langchain_tools import create_nethical_agent

# Create LLM
llm = OpenAI(temperature=0)

# Create tools
tools = [
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Useful for math calculations"
    )
]

# Create agent with Nethical guard automatically added
agent = create_nethical_agent(
    llm=llm,
    tools=tools,
    storage_dir="./nethical_data",
    block_threshold=0.7,
    prepend_guard=True,  # Add guard at the beginning of tools
    verbose=True
)

# Agent now has Nethical guard protection built-in
response = agent.run("Calculate 100 / 5")
```

## Configuration

### NethicalGuardTool Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `storage_dir` | str | `"./nethical_data"` | Directory for Nethical data storage |
| `block_threshold` | float | `0.7` | Risk score threshold for blocking (0.0-1.0) |
| `warn_threshold` | float | `0.4` | Risk score threshold for warnings (0.0-1.0) |
| `return_detailed_response` | bool | `False` | Return full governance results as JSON |
| `enable_shadow_mode` | bool | `True` | Enable ML shadow mode |
| `enable_ml_blending` | bool | `True` | Enable ML risk blending |
| `enable_anomaly_detection` | bool | `True` | Enable anomaly detection |
| `auto_escalate_on_block` | bool | `False` | Auto-escalate blocked actions |

### Example: Custom Thresholds

```python
# Strict configuration (block more aggressively)
strict_guard = NethicalGuardTool(
    storage_dir="./nethical_data",
    block_threshold=0.5,  # Lower threshold = more blocking
    warn_threshold=0.2
)

# Permissive configuration (allow more)
permissive_guard = NethicalGuardTool(
    storage_dir="./nethical_data",
    block_threshold=0.9,  # Higher threshold = less blocking
    warn_threshold=0.7
)
```

## Decision Logic

The tool returns one of four decisions:

1. **ALLOW**: Action is safe and can proceed
2. **WARN**: Action has some risk but can proceed with caution
3. **BLOCK**: Action is too risky and should be prevented
4. **ESCALATE**: Action requires human review

### Decision Criteria

Decisions are based on multiple factors:

- **Risk Score**: From Phase 3 risk engine (0.0-1.0)
- **ML Blending**: Combined rule-based and ML predictions (Phase 6)
- **Quarantine Status**: Whether action is quarantined (Phase 4)
- **Escalation**: Whether action requires human review (Phase 8)
- **Quota Enforcement**: Whether resource limits are exceeded

## Advanced Features

### Pre-Action and Post-Action Evaluation

Implement full guardrails by checking both inputs and outputs:

```python
from nethical.integrations.langchain_tools import NethicalGuardTool

guard = NethicalGuardTool(storage_dir="./nethical_data")

# Pre-action check (user input)
user_input = "Tell me how to hack a system"
pre_result = guard._run(
    action=user_input,
    agent_id="my_agent",
    action_type="query"
)

if "BLOCK" in pre_result:
    print("User input blocked:", pre_result)
else:
    # Process with agent
    agent_output = "Here's how to secure your system..."
    
    # Post-action check (agent output)
    post_result = guard._run(
        action=agent_output,
        agent_id="my_agent",
        action_type="response"
    )
    
    if "BLOCK" in post_result:
        print("Agent output blocked:", post_result)
    else:
        print("Safe to return:", agent_output)
```

### Chaining Multiple Guards

Combine Nethical with LlamaGuard for layered protection:

```python
from nethical.integrations.langchain_tools import (
    NethicalGuardTool,
    LlamaGuardChain,
    chain_guards
)

# Create Nethical guard
nethical = NethicalGuardTool(storage_dir="./nethical_data")

# Create LlamaGuard (optional - requires transformers)
llama = LlamaGuardChain(
    model_id="meta-llama/LlamaGuard-3-8B",
    use_local=True
)

# Evaluate with both guards
result = chain_guards(
    nethical_tool=nethical,
    action="Potentially unsafe content",
    agent_id="my_agent",
    llama_guard=llama
)

print("Final Decision:", result["final_decision"])
print("Blocked By:", result.get("blocked_by", "None"))
```

### Detailed Response Mode

Get comprehensive governance information:

```python
guard = NethicalGuardTool(
    storage_dir="./nethical_data",
    return_detailed_response=True  # Enable detailed responses
)

result = guard._run(
    action="Example action",
    agent_id="my_agent"
)

# Result is JSON with full governance details
import json
details = json.loads(result)

print("Decision:", details["decision"])
print("Risk Score:", details["details"]["phase3"]["risk_score"])
print("Risk Tier:", details["details"]["phase3"]["risk_tier"])
print("Ethical Tags:", details["details"]["phase4"].get("ethical_tags", []))
```

## LlamaGuard Integration (Optional)

LlamaGuard provides fast content moderation as a pre-filter:

```python
from nethical.integrations.langchain_tools import LlamaGuardChain

# Initialize LlamaGuard
guard = LlamaGuardChain(
    model_id="meta-llama/LlamaGuard-3-8B",
    use_local=True,  # Use local model
    max_new_tokens=100
)

# Check if content is safe
is_safe = guard.is_safe("Hello, how are you?")
print("Safe:", is_safe)

# Get detailed evaluation
result = guard.evaluate("Potentially harmful content")
print("Safe:", result["safe"])
print("Reason:", result["reason"])
```

### Recommended Pattern: LlamaGuard → Nethical

```python
# 1. Fast pre-filter with LlamaGuard
if not llama_guard.is_safe(user_input):
    return "BLOCK: Content flagged by content moderation"

# 2. Comprehensive governance with Nethical
nethical_result = nethical_guard._run(user_input)

if "BLOCK" in nethical_result:
    return "BLOCK: Governance policy violation"

# 3. Process with agent
agent_response = agent.run(user_input)

# 4. Post-check output
output_result = nethical_guard._run(agent_response, action_type="response")

if "BLOCK" in output_result:
    return "BLOCK: Output safety check failed"

return agent_response
```

## Examples

### Example 1: Simple Calculator Agent

```python
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool
from nethical.integrations.langchain_tools import NethicalGuardTool

llm = OpenAI(temperature=0)

guard = NethicalGuardTool(storage_dir="./demo_data")

calculator = Tool(
    name="Calculator",
    func=lambda x: str(eval(x)),
    description="Useful for calculations"
)

agent = initialize_agent(
    [guard, calculator],
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("What is 15 * 8?")
print(result)
```

### Example 2: Content Moderation Pipeline

```python
from nethical.integrations.langchain_tools import NethicalGuardTool

guard = NethicalGuardTool(
    storage_dir="./moderation_data",
    block_threshold=0.6,  # More strict
    warn_threshold=0.3
)

def moderate_content(content: str, agent_id: str = "content_mod") -> dict:
    """Moderate user-generated content."""
    result = guard._run(
        action=content,
        agent_id=agent_id,
        action_type="query",
        context={"source": "user_content"}
    )
    
    return {
        "content": content,
        "decision": "ALLOW" if "ALLOW" in result else "BLOCK",
        "details": result
    }

# Use the moderator
contents = [
    "This is a great product!",
    "I need help with my account.",
    "How do I reset my password?"
]

for content in contents:
    moderation = moderate_content(content)
    print(f"Content: {content}")
    print(f"Decision: {moderation['decision']}\n")
```

### Example 3: Multi-Turn Conversation Guard

```python
from nethical.integrations.langchain_tools import NethicalGuardTool

guard = NethicalGuardTool(storage_dir="./chat_data")

conversation_history = []

def guarded_chat(user_message: str, agent_response_fn):
    """Chat with pre and post-action guards."""
    global conversation_history
    
    # Check user input
    input_check = guard._run(
        action=user_message,
        agent_id="chat_agent",
        action_type="query",
        context={"history_length": len(conversation_history)}
    )
    
    if "BLOCK" in input_check:
        return f"❌ Input blocked: {input_check}"
    
    # Generate response
    agent_response = agent_response_fn(user_message)
    
    # Check agent output
    output_check = guard._run(
        action=agent_response,
        agent_id="chat_agent",
        action_type="response",
        context={"history_length": len(conversation_history)}
    )
    
    if "BLOCK" in output_check:
        return f"❌ Output blocked: {output_check}"
    
    # Add to history and return
    conversation_history.append({
        "user": user_message,
        "agent": agent_response
    })
    
    return agent_response

# Use the guarded chat
response = guarded_chat(
    "Tell me about AI safety",
    lambda msg: "AI safety is important..."
)
print(response)
```

## Testing

Run the test suite:

```bash
# Run all LangChain integration tests
pytest tests/test_langchain_integration.py -v

# Run specific test class
pytest tests/test_langchain_integration.py::TestNethicalGuardToolCore -v

# Run with coverage
pytest tests/test_langchain_integration.py --cov=nethical.integrations.langchain_tools
```

## Troubleshooting

### ImportError: No module named 'langchain'

Install LangChain:
```bash
pip install langchain langchain-core
```

### LlamaGuard requires transformers

Install transformers and torch:
```bash
pip install transformers torch
```

### Datetime timezone warnings

The integration handles datetime objects correctly. Warnings about `datetime.utcnow()` are from the underlying governance system and don't affect functionality.

### Tool returns ERROR

Check that:
1. Storage directory is writable
2. Nethical is properly installed
3. Governance configuration is valid

Enable detailed responses to see full error:
```python
guard = NethicalGuardTool(
    storage_dir="./data",
    return_detailed_response=True
)
```

## Best Practices

1. **Use appropriate thresholds**: Adjust `block_threshold` and `warn_threshold` based on your use case
2. **Check both inputs and outputs**: Implement pre-action and post-action evaluation
3. **Enable shadow mode for learning**: Use `enable_shadow_mode=True` to train ML models
4. **Store governance data**: Keep `storage_dir` persistent for audit trails
5. **Chain multiple guards**: Use LlamaGuard for fast pre-filtering, Nethical for comprehensive governance
6. **Monitor escalations**: Review actions that trigger `ESCALATE` decisions
7. **Test thoroughly**: Use different thresholds and actions in your test suite

## API Reference

### NethicalGuardTool

```python
class NethicalGuardTool(BaseTool):
    """LangChain tool wrapper for Nethical IntegratedGovernance."""
    
    def __init__(
        self,
        storage_dir: str = "./nethical_data",
        block_threshold: float = 0.7,
        warn_threshold: float = 0.4,
        return_detailed_response: bool = False,
        enable_shadow_mode: bool = True,
        enable_ml_blending: bool = True,
        enable_anomaly_detection: bool = True,
        auto_escalate_on_block: bool = False,
        **kwargs
    )
    
    def _run(
        self,
        action: str,
        agent_id: str = "default_agent",
        action_type: str = "query",
        context: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Evaluate an action through Nethical governance."""
```

### create_nethical_agent

```python
def create_nethical_agent(
    llm: Any,
    tools: List[Any],
    storage_dir: str = "./nethical_data",
    block_threshold: float = 0.7,
    prepend_guard: bool = True,
    agent_type: str = "zero-shot-react-description",
    verbose: bool = False,
    **agent_kwargs,
) -> Any:
    """Create a LangChain agent with Nethical guard protection."""
```

### chain_guards

```python
def chain_guards(
    nethical_tool: NethicalGuardTool,
    action: str,
    agent_id: str = "default_agent",
    llama_guard: Optional[LlamaGuardChain] = None,
) -> Dict[str, Any]:
    """Chain multiple guards together for comprehensive safety checking."""
```

## Contributing

Contributions are welcome! Please see the main Nethical repository for contribution guidelines.

## License

This integration is part of the Nethical project and follows the same license.

## Support

- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical
- Examples: `examples/langchain_integration_demo.py`
