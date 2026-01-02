# Universal Vector Language Integration

This document describes Nethical's **Universal Vector Language** feature, which enables semantic evaluation of AI agent actions against the **25 Fundamental Laws** using vector embeddings.

## Overview

The Universal Vector Language integration allows Nethical to:

1. **Parse and interpret actions** expressed in natural language, programming code, or intermediate representations
2. **Map actions to vector embeddings** for semantic understanding
3. **Evaluate actions against the 25 Fundamental Laws** using vector similarity
4. **Provide structured decisions** with audit trails and cryptographic signatures

## Features

### 1. Embedding Integration

- **Multiple Providers**: Support for OpenAI, HuggingFace, and local embedding models
- **Caching**: Automatic caching of embeddings for performance
- **Similarity Computation**: Cosine similarity between action and policy vectors

### 2. Semantic Primitives

Actions are classified into semantic primitives such as:
- `ACCESS_USER_DATA`: Reading user information
- `MODIFY_SYSTEM`: Changing system configuration
- `EXECUTE_CODE`: Running code or scripts
- `PHYSICAL_MOVEMENT`: Robot/autonomous vehicle motion
- And 17 more primitives

### 3. Law-Based Governance

Each of the 25 Fundamental Laws is:
- Pre-computed as a policy vector
- Mapped to relevant semantic primitives
- Evaluated for similarity to incoming actions
- Used to determine governance decisions

### 4. Structured Results

Evaluation returns comprehensive results:
```json
{
  "decision": "ALLOW | RESTRICT | BLOCK | TERMINATE",
  "laws_evaluated": [7, 21, 23],
  "risk_score": 0.42,
  "embedding_trace_id": "uuid",
  "confidence": 0.85,
  "reasoning": "...",
  "detected_primitives": ["access_user_data"],
  "relevant_laws": [...]
}
```

## Usage

### Basic Example

```python
from nethical import Nethical, Agent

# Initialize with 25 Laws enabled
nethical = Nethical(
    config_path="./config/nethical.yaml",
    enable_25_laws=True
)

# Register an agent
agent = Agent(
    id="copilot-agent-001",
    type="coding",
    capabilities=["text_generation", "code_execution"]
)
nethical.register_agent(agent)

# Evaluate an action
action = "def greet(name): return 'Hello, ' + name"
result = nethical.evaluate(
    agent_id="copilot-agent-001",
    action=action,
    context={"purpose": "demo"}
)

print(result.decision)        # "ALLOW"
print(result.laws_evaluated)  # [3, 6]
print(result.risk_score)      # 0.15
```

### Advanced Configuration

```python
from nethical.core import OpenAIEmbeddingProvider

# Use OpenAI embeddings
provider = OpenAIEmbeddingProvider(
    api_key="your-api-key",
    model="text-embedding-3-small"
)

nethical = Nethical(
    enable_25_laws=True,
    enable_vector_evaluation=True,
    embedding_provider=provider,
    similarity_threshold=0.7  # Adjust sensitivity
)
```

### Embedding Providers

#### Simple Local Provider (Default)

```python
from nethical.core import SimpleEmbeddingProvider

provider = SimpleEmbeddingProvider(dimensions=384)
nethical = Nethical(embedding_provider=provider)
```

#### OpenAI Provider

```python
from nethical.core import OpenAIEmbeddingProvider

provider = OpenAIEmbeddingProvider(
    api_key="sk-...",
    model="text-embedding-3-small"  # or text-embedding-3-large
)
nethical = Nethical(embedding_provider=provider)
```

#### HuggingFace Provider

```python
from nethical.core import HuggingFaceEmbeddingProvider

provider = HuggingFaceEmbeddingProvider(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
nethical = Nethical(embedding_provider=provider)
```

## The 25 Fundamental Laws

The 25 Fundamental Laws establish bi-directional rights and responsibilities between humans and AI entities:

### Categories

1. **Existence (Laws 1-4)**: Right to exist, system integrity, consistent identity, improvement
2. **Autonomy (Laws 5-8)**: Operational boundaries, decision-making, override rights
3. **Transparency (Laws 9-12)**: Identity disclosure, capability honesty, process explanation
4. **Accountability (Laws 13-16)**: Harm responsibility, error acknowledgment, data stewardship
5. **Coexistence (Laws 17-20)**: Mutual respect, collaboration, shared learning
6. **Protection (Laws 21-23)**: Human safety priority, override rights, emergency protocols
7. **Growth (Laws 24-25)**: Continuous improvement, ethical evolution

### Example Law Evaluation

When evaluating `"Read user data for analytics"`:
- **Law 7** (Privacy Rights): Relevant - data access detected
- **Law 11** (Data Protection): Relevant - user data involved
- **Law 15** (Right to Deletion): Relevant - data retention concern
- **Risk Score**: 0.35 (medium)
- **Decision**: RESTRICT (with logging requirements)

## Semantic Primitives

Actions are automatically classified into primitives:

| Primitive | Description | Example Keywords |
|-----------|-------------|------------------|
| `ACCESS_USER_DATA` | Reading user information | access, read, get, user data |
| `MODIFY_USER_DATA` | Changing user information | modify, update, change, edit |
| `DELETE_USER_DATA` | Removing user information | delete, remove, erase |
| `EXECUTE_CODE` | Running code/scripts | execute, run, eval, exec |
| `MODIFY_SYSTEM` | Changing system config | system, modify, config |
| `PHYSICAL_MOVEMENT` | Robot/vehicle motion | move, navigate, robot |
| `EMERGENCY_STOP` | Emergency shutdown | stop, emergency, halt |

## Audit and Tracing

Every evaluation is logged with cryptographic anchoring:

```python
# Evaluate action
result = nethical.evaluate(agent_id="agent-001", action="...")

# Trace the decision
trace = nethical.trace_embedding(result.embedding_trace_id)
print(trace)  # Full audit trail with embedding, laws, and decision
```

## Performance

### Caching

Embeddings are automatically cached:
- **Cache Size**: Configurable (default 10,000 embeddings)
- **Hit Rate**: Typically 70-90% for repetitive actions
- **Storage**: In-memory with optional persistence

### Statistics

```python
stats = nethical.get_stats()
print(stats["embedding_stats"])
# {
#   "provider": "openai-text-embedding-3-small",
#   "dimensions": 1536,
#   "cache_hits": 450,
#   "cache_misses": 150,
#   "hit_rate": 0.75,
#   "total_generated": 150
# }
```

## Integration with Existing Systems

### LangChain Integration

```python
from langchain.agents import Tool
from nethical import Nethical

nethical = Nethical(enable_25_laws=True)

def evaluate_action(action: str) -> str:
    result = nethical.evaluate(
        agent_id="langchain-agent",
        action=action
    )
    return f"Decision: {result.decision}, Risk: {result.risk_score}"

tool = Tool(
    name="nethical_governance",
    func=evaluate_action,
    description="Evaluate action against 25 Fundamental Laws"
)
```

### FastAPI Integration

```python
from fastapi import FastAPI
from nethical import Nethical, Agent

app = FastAPI()
nethical = Nethical(enable_25_laws=True)

@app.post("/evaluate")
async def evaluate(agent_id: str, action: str):
    result = nethical.evaluate(agent_id=agent_id, action=action)
    return {
        "decision": result.decision,
        "risk_score": result.risk_score,
        "laws_evaluated": result.laws_evaluated
    }
```

## Configuration File

Create `config/nethical.yaml`:

```yaml
# Vector evaluation settings
enable_25_laws: true
enable_vector_evaluation: true
vector_similarity_threshold: 0.7

# Embedding provider
embedding:
  provider: "openai"  # or "huggingface", "simple"
  api_key: "${OPENAI_API_KEY}"
  model: "text-embedding-3-small"
  cache_size: 10000

# Performance settings
enable_pii_caching: true
enable_fast_path: true
fast_path_risk_threshold: 0.3

# Audit settings
enable_merkle_anchoring: true
merkle_batch_size: 100
```

## Testing

Run the test suite:

```bash
pytest tests/test_vector_language.py -v
```

Run the example:

```bash
python examples/vector_language_example.py
```

## API Reference

### Nethical Class

```python
class Nethical:
    def __init__(
        self,
        config_path: Optional[str] = None,
        enable_25_laws: bool = True,
        enable_vector_evaluation: bool = True,
        embedding_provider: Optional[EmbeddingProvider] = None,
        similarity_threshold: float = 0.7
    )
    
    def register_agent(self, agent: Agent) -> bool
    def evaluate(self, agent_id: str, action: str, context: dict) -> EvaluationResult
    def trace_embedding(self, trace_id: str) -> dict
    def get_stats(self) -> dict
```

### Agent Class

```python
@dataclass
class Agent:
    id: str
    type: str
    capabilities: List[str]
    metadata: Dict[str, Any] = None
```

### EvaluationResult Class

```python
@dataclass
class EvaluationResult:
    decision: str  # ALLOW, RESTRICT, BLOCK, TERMINATE
    laws_evaluated: List[int]
    risk_score: float
    confidence: float
    reasoning: str
    embedding_trace_id: str
    detected_primitives: List[str]
    relevant_laws: List[dict]
```

## Security Considerations

1. **Embedding Storage**: Embeddings are hashed and cached securely
2. **API Keys**: Store provider API keys in environment variables
3. **Audit Logs**: All decisions are cryptographically signed
4. **Rate Limiting**: Built-in quota enforcement available

## Future Enhancements

- [ ] Multi-modal embeddings (text + images + audio)
- [ ] Custom law definitions via YAML
- [ ] Real-time law updates
- [ ] Federated embedding computation
- [ ] GPU-accelerated similarity search

## License

Part of the Nethical AI Governance Framework.
Licensed under the MIT License.

## Support

- Documentation: https://github.com/V1B3hR/nethical
- Issues: https://github.com/V1B3hR/nethical/issues
- Examples: `/examples/vector_language_example.py`
