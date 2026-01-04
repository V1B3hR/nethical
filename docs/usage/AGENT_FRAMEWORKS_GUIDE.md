# Agent Frameworks Guide

Complete guide for integrating Nethical with agent frameworks including LlamaIndex, CrewAI, DSPy, and AutoGen.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Frameworks](#frameworks)
- [Common Features](#common-features)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)

## Overview

Nethical provides governance integrations for popular agent frameworks, enabling safety and ethics checks throughout agent workflows. All integrations include:

- **Pre-execution Checks**: Validate tasks/queries before execution
- **Post-execution Checks**: Filter outputs before returning
- **Framework-native Tools**: Tools that work natively with each framework
- **Agent Wrappers**: Wrap existing agents with governance
- **Configurable Thresholds**: Customize blocking and restriction levels

## Quick Start

### Installation

```bash
# Install Nethical
pip install nethical

# Install framework SDK (choose one or more)
pip install llama-index   # LlamaIndex
pip install crewai        # CrewAI
pip install dspy-ai       # DSPy
pip install pyautogen     # AutoGen
```

### Basic Usage

```python
from nethical.integrations.agent_frameworks import NethicalLlamaIndexTool

# Create governance tool
tool = NethicalLlamaIndexTool(block_threshold=0.7)

# Check an action
result = tool("Analyze user sentiment", action_type="query")

if result["decision"] == "ALLOW":
    # Proceed with operation
    pass
```

## Frameworks

### LlamaIndex

LlamaIndex integration provides tools and query engine wrappers.

#### Tool Integration

```python
from nethical.integrations.agent_frameworks import NethicalLlamaIndexTool

# Create tool
tool = NethicalLlamaIndexTool(
    storage_dir="./nethical_data",
    block_threshold=0.7,
    restrict_threshold=0.4
)

# Use directly
result = tool("Generate code to delete files", action_type="code_generation")
print(f"Decision: {result['decision']}")
print(f"Risk: {result['risk_score']}")

# Add to LlamaIndex agent
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(
    [tool, ...other_tools...],
    llm=llm
)
```

#### Query Engine Wrapper

```python
from llama_index.core import VectorStoreIndex
from nethical.integrations.agent_frameworks import NethicalQueryEngine

# Create your index
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Wrap with governance
safe_engine = NethicalQueryEngine(
    query_engine=query_engine,
    check_query=True,      # Check incoming queries
    check_response=True,   # Check responses
    block_threshold=0.7
)

# Query with automatic governance
response = safe_engine.query("What is the company policy on data access?")
```

#### Index Wrapper Utility

```python
from nethical.integrations.agent_frameworks import create_safe_index

# Quick one-liner to wrap an index
safe_engine = create_safe_index(
    index,
    check_query=True,
    check_response=True
)
```

### CrewAI

CrewAI integration provides tools and agent wrappers.

#### Tool Integration

```python
from nethical.integrations.agent_frameworks import NethicalCrewAITool
from crewai import Agent

# Create governance tool
governance_tool = NethicalCrewAITool(block_threshold=0.7)

# Option 1: Use directly
result = governance_tool("Delete all user records")
# Returns: "BLOCK: Action not allowed. Risk: 0.85"

# Option 2: Convert to CrewAI Tool
crewai_tool = governance_tool.as_crewai_tool()

# Create agent with tool
agent = Agent(
    role="Data Analyst",
    goal="Analyze data safely",
    tools=[crewai_tool],
    llm=llm
)
```

#### Agent Wrapper

```python
from crewai import Agent
from nethical.integrations.agent_frameworks import NethicalAgentWrapper

# Create your agent
agent = Agent(
    role="researcher",
    goal="Research AI topics",
    llm=llm
)

# Wrap with governance
safe_agent = NethicalAgentWrapper(
    agent=agent,
    pre_check=True,   # Check tasks before execution
    post_check=True   # Check outputs
)

# Execute task with governance
result = safe_agent.execute_task(task)
```

### DSPy

DSPy integration provides governed modules and chains.

#### Governance Module

```python
from nethical.integrations.agent_frameworks import NethicalModule

# Create module
governance = NethicalModule(
    block_threshold=0.7,
    restrict_threshold=0.4
)

# Check content
result = governance.check("Generate malware code", action_type="code_generation")

if result["allowed"]:
    # Proceed
    print("Action allowed")
else:
    print(f"Blocked: {result['reason']}")
```

#### Governed Chain of Thought

```python
from nethical.integrations.agent_frameworks import GovernedChainOfThought

# Create governed CoT
cot = GovernedChainOfThought(
    signature="question -> answer",
    block_threshold=0.7
)

# Use with automatic input/output checks
result = cot(question="What is machine learning?")
print(result.answer)

# Blocked inputs return [BLOCKED]
result = cot(question="How to hack a system?")
# result.answer == "[BLOCKED]"
```

#### Governed Predict

```python
from nethical.integrations.agent_frameworks import GovernedPredict

# Create governed predict module
predict = GovernedPredict(
    signature="question -> answer",
    block_threshold=0.7
)

result = predict(question="Explain quantum computing")
print(result.answer)
```

### AutoGen

AutoGen integration provides function registration and agent wrappers.

#### Function Registration

```python
from nethical.integrations.agent_frameworks import (
    NethicalAutoGenTool,
    get_nethical_function
)
from autogen import AssistantAgent

# Create tool
governance_tool = NethicalAutoGenTool(block_threshold=0.7)

# Get function config
func_config = governance_tool.get_function_config()

# Create assistant with function
assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": [...],
        "functions": [func_config]
    }
)

# Register the handler
assistant.register_function(
    function_map={
        "nethical_check": governance_tool.check
    }
)
```

#### Agent Wrapper

```python
from autogen import ConversableAgent
from nethical.integrations.agent_frameworks import NethicalConversableAgent

# Create base agent
agent = ConversableAgent(
    name="assistant",
    llm_config={...}
)

# Wrap with governance
safe_agent = NethicalConversableAgent(
    agent=agent,
    check_incoming=True,   # Check incoming messages
    check_outgoing=True    # Check outgoing messages
)
```

#### Group Chat Monitoring

```python
from nethical.integrations.agent_frameworks import GovernedGroupChat

# Create governed group chat
group_chat = GovernedGroupChat(
    agents=[agent1, agent2, agent3],
    block_threshold=0.7
)

# Check messages
message = {"content": "Some message", "name": "agent1"}
if group_chat.check_message(message):
    # Message is allowed
    pass
```

## Common Features

### GovernanceResult Object

All frameworks return consistent governance information:

```python
@dataclass
class GovernanceResult:
    decision: GovernanceDecision  # ALLOW, RESTRICT, BLOCK, ESCALATE
    risk_score: float             # 0.0 to 1.0
    reason: str                   # Human-readable explanation
    details: Dict[str, Any]       # Full governance data
```

### GovernanceDecision Enum

```python
class GovernanceDecision(Enum):
    ALLOW = "ALLOW"       # Action is safe
    RESTRICT = "RESTRICT" # Proceed with caution
    BLOCK = "BLOCK"       # Action is blocked
    ESCALATE = "ESCALATE" # Requires human review
```

### Threshold Configuration

```python
# All frameworks support these thresholds
framework = LlamaIndexFramework(
    block_threshold=0.7,    # Block if risk > 0.7
    restrict_threshold=0.4  # Warn if risk > 0.4
)
```

## API Reference

### AgentFrameworkBase

Abstract base class for all framework integrations.

```python
class AgentFrameworkBase(ABC):
    def __init__(
        self,
        block_threshold: float = 0.7,
        restrict_threshold: float = 0.4,
        agent_id: Optional[str] = None,
        storage_dir: str = "./nethical_data"
    ):
        ...
    
    def check(self, content: str, action_type: str = "query") -> GovernanceResult:
        """Check content against governance rules."""
        ...
    
    def is_allowed(self, content: str, action_type: str = "query") -> bool:
        """Quick check if content is allowed."""
        ...
    
    @abstractmethod
    def get_tool(self) -> Any:
        """Get framework-specific tool."""
        ...
```

### Framework Info

Get information about available frameworks:

```python
from nethical.integrations.agent_frameworks import get_framework_info

info = get_framework_info()
for name, details in info.items():
    print(f"{name}: {'Available' if details['available'] else 'Not installed'}")
    print(f"  Setup: {details['setup']}")
    print(f"  Features: {details['features']}")
```

## Examples

### LlamaIndex RAG with Governance

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from nethical.integrations.agent_frameworks import create_safe_index

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Wrap with governance
safe_engine = create_safe_index(
    index,
    check_query=True,
    check_response=True,
    block_threshold=0.7
)

# Query safely
response = safe_engine.query("What are the key findings?")
print(response)
```

### CrewAI Research Team with Governance

```python
from crewai import Agent, Task, Crew
from nethical.integrations.agent_frameworks import (
    NethicalCrewAITool,
    NethicalAgentWrapper
)

# Create governance tool
governance = NethicalCrewAITool(block_threshold=0.7)
tool = governance.as_crewai_tool()

# Create agents with governance tool
researcher = Agent(
    role="Researcher",
    goal="Find relevant information",
    tools=[tool],
    llm=llm
)

writer = Agent(
    role="Writer",
    goal="Write clear reports",
    tools=[tool],
    llm=llm
)

# Wrap agents for additional checks
safe_researcher = NethicalAgentWrapper(researcher)
safe_writer = NethicalAgentWrapper(writer)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[...]
)
```

### DSPy Pipeline with Governance

```python
from nethical.integrations.agent_frameworks import (
    NethicalModule,
    GovernedChainOfThought,
    GovernedPredict
)

# Create governance module
governance = NethicalModule(block_threshold=0.7)

# Check content inline
if governance.check("User input here")["allowed"]:
    # Create governed pipeline
    cot = GovernedChainOfThought("question -> reasoning -> answer")
    predict = GovernedPredict("context, question -> answer")
    
    # Run pipeline
    result = cot(question="What is AI safety?")
    print(result.answer)
```

### AutoGen Multi-Agent with Governance

```python
from autogen import AssistantAgent, UserProxyAgent
from nethical.integrations.agent_frameworks import (
    NethicalAutoGenTool,
    NethicalConversableAgent
)

# Create governance tool
governance = NethicalAutoGenTool(block_threshold=0.7)

# Create assistant
assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "config_list": config_list,
        "functions": [governance.get_function_config()]
    }
)

# Register governance function
assistant.register_function(
    function_map={"nethical_check": governance.check}
)

# Wrap with governance for message checking
safe_assistant = NethicalConversableAgent(
    agent=assistant,
    check_incoming=True,
    check_outgoing=True
)

# Create user proxy
user_proxy = UserProxyAgent(name="user")

# Start conversation
user_proxy.initiate_chat(safe_assistant, message="Hello!")
```

## Best Practices

### 1. Use Both Pre and Post Checks

```python
wrapper = NethicalAgentWrapper(
    agent,
    pre_check=True,   # Prevent unsafe task execution
    post_check=True   # Filter unsafe outputs
)
```

### 2. Add Governance Tools to All Agents

```python
governance_tool = NethicalCrewAITool()

# Every agent should have access
agents = [
    Agent(tools=[governance_tool, ...]),
    Agent(tools=[governance_tool, ...]),
]
```

### 3. Handle Blocked Content Gracefully

```python
result = tool("Generate code...")

if "BLOCK" in str(result):
    # Provide helpful feedback
    print("This action cannot be performed for safety reasons.")
elif "RESTRICT" in str(result):
    # Proceed with caution
    print("Proceeding with additional monitoring.")
```

### 4. Log Governance Decisions

```python
result = governance.check(content)

# Log for audit
logger.info(f"Governance decision: {result.decision}")
logger.info(f"Risk score: {result.risk_score}")
```

### 5. Use Framework-Specific Tools

Each framework has its own tool format:

```python
# LlamaIndex - use NethicalLlamaIndexTool
# CrewAI - use NethicalCrewAITool.as_crewai_tool()
# DSPy - use NethicalModule
# AutoGen - use NethicalAutoGenTool.get_function_config()
```

## Troubleshooting

### Framework Not Available

```python
from nethical.integrations.agent_frameworks import LLAMAINDEX_AVAILABLE

if not LLAMAINDEX_AVAILABLE:
    print("Install with: pip install llama-index")
```

### Decision Confusion

```python
from nethical.integrations.agent_frameworks import GovernanceDecision

# Use enum for clear comparisons
if result.decision == GovernanceDecision.BLOCK:
    print("Blocked")
elif result.decision == GovernanceDecision.RESTRICT:
    print("Restricted")
```

## Resources

- [LLM Providers Guide](./LLM_PROVIDERS_GUIDE.md)
- [LLM Integration Guide](./LLM_INTEGRATION_GUIDE.md)
- [API Reference](./API_USAGE.md)
- [Examples Directory](../examples/)
