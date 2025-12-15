# Nethical Ecosystem Overview

Complete overview of all Nethical integrations across the AI/ML ecosystem.

## Summary

Nethical provides **25+ integrations** across:
- **6 Observability Platforms**
- **3 Cloud ML Platforms**
- **6 LLM Providers**
- **4 Vector Stores**
- **4 Agent Frameworks**
- **5 ML Platforms**

All with built-in governance, safety checks, and compliance monitoring.

## Integration Categories

### 1. Observability & Monitoring (6 platforms)

| Platform | Category | Best For | Status |
|----------|----------|----------|--------|
| **Langfuse** | LLM Tracing | LLM app observability | Production |
| **LangSmith** | LangChain | LangChain monitoring | Production |
| **Arize AI** | ML Monitoring | Production ML | Production |
| **WhyLabs** | Data Quality | Data drift detection | Production |
| **Helicone** | LLM Observability | Simple LLM tracking | Production |
| **TruLens** | LLM Evaluation | LLM quality metrics | Production |

**Setup**: See [OBSERVABILITY_GUIDE.md](OBSERVABILITY_GUIDE.md)

**Example**:
```python
from nethical.integrations.observability import create_observability_stack

manager = create_observability_stack(
    langfuse_config={...},
    langsmith_config={...},
    arize_config={...}
)

manager.log_governance_event_all(
    action="...",
    decision="ALLOW",
    risk_score=0.2,
    metadata={}
)
```

### 2. Cloud ML Platforms (3 platforms)

| Platform | Provider | Best For | Status |
|----------|----------|----------|--------|
| **Vertex AI** | Google Cloud | Complete MLOps | Production |
| **Databricks** | Databricks | Unified analytics | Production |
| **Snowflake Cortex** | Snowflake | SQL + ML | Production |

**Setup**: See [CLOUD_INTEGRATIONS_GUIDE.md](CLOUD_INTEGRATIONS_GUIDE.md)

**Example**:
```python
from nethical.integrations.cloud import VertexAIConnector

connector = VertexAIConnector(
    project="my-project",
    enable_governance=True
)

result = connector.predict_with_governance(
    endpoint_id="...",
    instances=[{"text": "..."}]
)
```

### 3. LLM Providers (6 providers)

| Provider | Models | Best For | Status |
|----------|--------|----------|--------|
| **Cohere** | Command, Embed | Enterprise LLMs | Production |
| **Mistral** | Mistral 7B, Large | Open models | Production |
| **Together AI** | Multiple OSS | OSS hosting | Production |
| **Fireworks AI** | Multiple OSS | Fast inference | Production |
| **Groq** | Llama, Mixtral | Fast inference | Production |
| **Replicate** | Any model | Model hosting | Production |

**Setup**: See [LLM_PROVIDERS_GUIDE.md](LLM_PROVIDERS_GUIDE.md)

**Example**:
```python
from nethical.integrations.llm_providers import CohereProvider

provider = CohereProvider(api_key="...")
response = provider.safe_generate("Prompt")
print(f"Risk score: {response.risk_score}")
```

### 4. Vector Stores (4 stores)

| Store | Type | Best For | Status |
|-------|------|----------|--------|
| **Pinecone** | Managed | Production scale | Production |
| **Weaviate** | Open source | Flexible schema | Production |
| **Chroma** | Embedded | Development | Production |
| **Qdrant** | High performance | Speed-critical | Production |

**Setup**: See [VECTOR_STORE_INTEGRATION_GUIDE.md](VECTOR_STORE_INTEGRATION_GUIDE.md)

**Example**:
```python
from nethical.integrations.vector_stores import PineconeConnector

connector = PineconeConnector(
    api_key="...",
    environment="..."
)

results = connector.query(
    vectors=[...],
    top_k=10,
    enable_governance=True
)
```

### 5. Agent Frameworks (4 frameworks)

| Framework | Focus | Best For | Status |
|-----------|-------|----------|--------|
| **LlamaIndex** | RAG | Data ingestion | Production |
| **CrewAI** | Multi-agent | Team collaboration | Production |
| **DSPy** | Prompting | Structured prompts | Production |
| **AutoGen** | Conversation | Multi-agent chat | Production |

**Setup**: See [AGENT_FRAMEWORKS_GUIDE.md](AGENT_FRAMEWORKS_GUIDE.md)

**Example**:
```python
from nethical.integrations.agent_frameworks import NethicalLlamaIndexTool

tool = NethicalLlamaIndexTool(block_threshold=0.7)
result = tool("Check safety of action")
```

### 6. ML Platforms (5 platforms)

| Platform | Type | Best For | Status |
|----------|------|----------|--------|
| **MLflow** | Open source | Experiment tracking | Production |
| **W&B** | Managed | Team collaboration | Production |
| **SageMaker** | AWS | AWS ecosystem | Production |
| **Azure ML** | Azure | Azure ecosystem | Production |
| **Ray Serve** | Distributed | Model serving | Production |

**Setup**: See [EXTERNAL_INTEGRATIONS_GUIDE.md](EXTERNAL_INTEGRATIONS_GUIDE.md)

**Example**:
```python
from nethical.integrations.mlflow_connector import MLflowConnector

mlflow = MLflowConnector(tracking_uri="...")
run_id = mlflow.start_run("experiment")
mlflow.log_metrics(run_id, {"accuracy": 0.95})
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Nethical Core                        │
│            (Governance Engine + Safety)                 │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   ┌────▼────┐          ┌────▼────┐
   │ Observe │          │  Cloud  │
   │ ability │          │   ML    │
   └────┬────┘          └────┬────┘
        │                    │
   ┌────┴─────┐         ┌────┴─────┐
   │ LLM      │         │ Vector   │
   │ Providers│         │ Stores   │
   └──────────┘         └──────────┘
        │                    │
   ┌────┴─────┐         ┌────┴─────┐
   │ Agent    │         │  ML      │
   │ Frameworks│        │ Platforms│
   └──────────┘         └──────────┘
```

## Configuration

### Environment Variables

```bash
# Observability
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGSMITH_API_KEY="ls-..."
export ARIZE_API_KEY="api-..."
export ARIZE_SPACE_KEY="space-..."

# Cloud ML
export GCP_PROJECT="my-project"
export DATABRICKS_HOST="https://..."
export DATABRICKS_TOKEN="dapi..."
export SNOWFLAKE_ACCOUNT="account"
export SNOWFLAKE_USER="user"
export SNOWFLAKE_PASSWORD="password"

# LLM Providers
export COHERE_API_KEY="co-..."
export MISTRAL_API_KEY="mst-..."
export TOGETHER_API_KEY="tog-..."
export FIREWORKS_API_KEY="fw-..."
export GROQ_API_KEY="gsk-..."

# Vector Stores
export PINECONE_API_KEY="pc-..."
export WEAVIATE_URL="http://..."
export QDRANT_URL="http://..."
```

### Manifests

All integrations have YAML manifests in `config/integrations/`:

- `observability-mcp.yaml` - Combined observability manifest
- `langfuse-tool.yaml` - Langfuse configuration
- `vertex-ai-tool.yaml` - Vertex AI configuration
- `databricks-tool.yaml` - Databricks configuration
- `llm-providers-mcp.yaml` - LLM providers manifest
- `vector-stores-mcp.yaml` - Vector stores manifest

## Usage Patterns

### Multi-Provider Observability

```python
from nethical.integrations.observability import create_observability_stack

# Create stack with all providers
manager = create_observability_stack(
    langfuse_config={...},
    langsmith_config={...},
    arize_config={...},
    whylabs_config={...},
    helicone_config={...},
    trulens_config={...}
)

# Log to all at once
manager.log_governance_event_all(...)
manager.log_metrics_all(...)
```

### Multi-Cloud ML

```python
from nethical.integrations.cloud import (
    VertexAIConnector,
    DatabricksConnector,
    SnowflakeCortexConnector
)

# Use appropriate platform per use case
vertex = VertexAIConnector(...)     # For training
databricks = DatabricksConnector(...)  # For serving
snowflake = SnowflakeCortexConnector(...)  # For SQL+ML
```

### Full Stack Integration

```python
# Combine observability + cloud ML + governance
from nethical.integrations.observability import create_observability_stack
from nethical.integrations.cloud import VertexAIConnector

# Setup observability
obs = create_observability_stack(...)

# Setup cloud ML with governance
ml = VertexAIConnector(enable_governance=True)

# Run experiment with full observability
run_id = ml.start_run("experiment")

# Log to observability
obs.log_governance_event_all(
    action=f"Start experiment {run_id}",
    decision="ALLOW",
    risk_score=0.1,
    metadata={"run_id": run_id}
)

# Train with governance checks
ml.log_parameters(run_id, {...})
ml.log_metrics(run_id, {...})

# Log metrics to observability
obs.log_metrics_all(...)

ml.end_run(run_id)
```

## Installation Matrix

| Component | Install Command | Required For |
|-----------|----------------|--------------|
| Observability | `pip install langfuse langsmith arize whylogs trulens-eval` | All platforms |
| Cloud ML | `pip install google-cloud-aiplatform databricks-sdk snowflake-connector-python mlflow` | Cloud platforms |
| LLM Providers | `pip install cohere mistralai together fireworks-ai groq replicate` | LLM providers |
| Vector Stores | `pip install pinecone-client weaviate-client chromadb qdrant-client` | Vector stores |
| Agent Frameworks | `pip install llama-index crewai dspy-ai pyautogen` | Agent frameworks |
| ML Platforms | `pip install mlflow wandb boto3 sagemaker azureml-core ray[serve]` | ML platforms |

## Examples

Complete examples in `examples/`:

```
examples/
├── langfuse_demo.py          # Langfuse integration
├── vertex_ai_demo.py         # Vertex AI integration
├── databricks_demo.py        # Databricks integration
├── full_stack_demo.py        # Complete ecosystem
├── llamaindex_demo.py        # LlamaIndex integration
├── crewai_demo.py            # CrewAI integration
└── vector_store_demo.py      # Vector store integration
```

## Feature Matrix

| Feature | Observability | Cloud ML | LLM | Vector | Agent | ML Platform |
|---------|--------------|----------|-----|--------|-------|-------------|
| Governance Checks | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Risk Scoring | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PII Detection | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Experiment Tracking | ✓ | ✓ | - | - | - | ✓ |
| Distributed Tracing | ✓ | - | - | - | - | - |
| Model Serving | - | ✓ | ✓ | - | - | ✓ |
| Multi-Provider | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Governance Features

All integrations support:

1. **Pre-Check**: Input validation before processing
2. **Post-Check**: Output validation after processing
3. **Risk Scoring**: 0.0-1.0 risk score per action
4. **PII Detection**: Automatic detection and filtering
5. **Audit Trail**: Complete logs of all decisions
6. **Blocking**: Automatic blocking of high-risk actions
7. **Compliance**: GDPR, HIPAA, SOC2 support

## Best Practices

1. **Enable Governance Everywhere**
   ```python
   connector = Platform(enable_governance=True)
   ```

2. **Use Multi-Provider Observability**
   - Different platforms for different use cases
   - Redundancy for critical monitoring
   - Cross-validation of metrics

3. **Environment Variables**
   - Never hardcode credentials
   - Use secret management
   - Rotate keys regularly

4. **Monitor All Platforms**
   - Check dashboards regularly
   - Set up alerts
   - Review governance decisions

5. **Test Governance**
   - Test with known risky inputs
   - Verify blocking works
   - Validate risk scores

## Support & Resources

- **Documentation**: This guide + platform-specific guides
- **Examples**: `examples/` directory
- **Issues**: GitHub issues for Nethical
- **Community**: Discord/Slack channels
- **Provider Docs**: See individual provider documentation

## Roadmap

Future integrations planned:
- More observability platforms (Weights & Biases, Neptune.ai)
- Additional cloud platforms (Oracle Cloud, Alibaba Cloud)
- More LLM providers (Anthropic, AI21 Labs)
- Enterprise features (SSO, RBAC, audit)

---

**Total Integrations**: 25+
**Production Ready**: 25
**Categories**: 6
**Last Updated**: 2024-12-14
