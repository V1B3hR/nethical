# Observability Guide

Complete guide to using Nethical with ML observability platforms.

## Overview

Nethical integrates with 6 major observability platforms:

1. **Langfuse** - LLM tracing and monitoring
2. **LangSmith** - LangChain observability  
3. **Arize AI** - ML monitoring and explainability
4. **WhyLabs** - Data and ML monitoring
5. **Helicone** - LLM observability
6. **TruLens** - LLM evaluation and monitoring

## Installation

```bash
# Install all observability providers
pip install langfuse langsmith arize whylogs trulens-eval

# Or install individually
pip install langfuse        # For Langfuse
pip install langsmith       # For LangSmith
pip install arize          # For Arize AI
pip install whylogs        # For WhyLabs
pip install trulens-eval   # For TruLens
# Helicone requires no SDK - API only
```

## Quick Start

### Single Provider

```python
from nethical.integrations.observability import LangfuseConnector

# Initialize connector
connector = LangfuseConnector(
    public_key="pk-lf-...",
    secret_key="sk-lf-..."
)

# Log governance event
connector.log_governance_event(
    action="Generate code",
    decision="ALLOW",
    risk_score=0.2,
    metadata={"agent_id": "assistant-1"}
)
```

### Multi-Provider Stack

```python
from nethical.integrations.observability import create_observability_stack

# Create stack with multiple providers
manager = create_observability_stack(
    langfuse_config={
        "public_key": "pk-...",
        "secret_key": "sk-..."
    },
    langsmith_config={
        "api_key": "ls-...",
        "project_name": "my-project"
    },
    arize_config={
        "api_key": "api-...",
        "space_key": "space-...",
        "model_id": "governance-model"
    }
)

# Log to all providers at once
manager.log_governance_event_all(
    action="Process user data",
    decision="RESTRICT",
    risk_score=0.65,
    metadata={}
)
```

## Provider Details

### Langfuse

Best for: LLM application tracing and monitoring

```python
from nethical.integrations.observability import LangfuseConnector, TraceSpan
from datetime import datetime

connector = LangfuseConnector(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host="https://cloud.langfuse.com"
)

# Log trace with governance
span = TraceSpan(
    trace_id="trace-123",
    span_id="span-456",
    parent_span_id=None,
    name="code_generation",
    start_time=datetime.utcnow(),
    end_time=datetime.utcnow(),
    attributes={"language": "python"},
    governance_result={
        "decision": "ALLOW",
        "risk_score": 0.1
    }
)
connector.log_trace(span)
```

**Dashboard**: https://cloud.langfuse.com

### LangSmith

Best for: LangChain application monitoring

```python
from nethical.integrations.observability import LangSmithConnector

connector = LangSmithConnector(
    api_key=os.getenv("LANGSMITH_API_KEY"),
    project_name="governance-project"
)

connector.log_governance_event(
    action="Chain execution",
    decision="ALLOW",
    risk_score=0.15,
    metadata={"chain_id": "chain-1"}
)
```

**Dashboard**: https://smith.langchain.com

### Arize AI

Best for: Production ML monitoring and explainability

```python
from nethical.integrations.observability import ArizeConnector

connector = ArizeConnector(
    api_key=os.getenv("ARIZE_API_KEY"),
    space_key=os.getenv("ARIZE_SPACE_KEY"),
    model_id="governance-model",
    model_version="1.0"
)

connector.log_governance_event(
    action="Model prediction",
    decision="ALLOW",
    risk_score=0.25,
    metadata={"prediction_id": "pred-123"}
)
```

**Dashboard**: https://app.arize.com

### WhyLabs

Best for: Data quality and drift monitoring

```python
from nethical.integrations.observability import WhyLabsConnector

connector = WhyLabsConnector(
    api_key=os.getenv("WHYLABS_API_KEY"),
    org_id=os.getenv("WHYLABS_ORG_ID"),
    dataset_id="governance-dataset"
)

connector.log_governance_event(
    action="Data processing",
    decision="ALLOW",
    risk_score=0.1,
    metadata={}
)
```

**Dashboard**: https://hub.whylabsapp.com

### Helicone

Best for: Simple LLM observability (API-only, no SDK)

```python
from nethical.integrations.observability import HeliconeConnector

connector = HeliconeConnector(
    api_key=os.getenv("HELICONE_API_KEY")
)

connector.log_governance_event(
    action="LLM call",
    decision="ALLOW",
    risk_score=0.2,
    metadata={}
)
```

**Dashboard**: https://www.helicone.ai/dashboard

### TruLens

Best for: LLM evaluation and feedback

```python
from nethical.integrations.observability import TruLensConnector

connector = TruLensConnector(
    database_url="sqlite:///trulens.db",  # Optional
    app_id="governance-app"
)

connector.log_governance_event(
    action="LLM evaluation",
    decision="ALLOW",
    risk_score=0.15,
    metadata={}
)
```

**Dashboard**: Run `tru.run_dashboard()`

## Logging Metrics

All providers support aggregated metrics:

```python
from nethical.integrations.observability import GovernanceMetrics
from datetime import datetime

metrics = GovernanceMetrics(
    total_evaluations=1000,
    allowed_count=850,
    blocked_count=100,
    restricted_count=50,
    average_risk_score=0.25,
    pii_detections=15,
    latency_p50_ms=12.5,
    latency_p99_ms=45.2,
    timestamp=datetime.utcnow()
)

manager.log_metrics_all(metrics)
```

## Best Practices

1. **Use Environment Variables**
   ```bash
   export LANGFUSE_PUBLIC_KEY="pk-..."
   export LANGFUSE_SECRET_KEY="sk-..."
   export LANGSMITH_API_KEY="ls-..."
   ```

2. **Multi-Provider Strategy**
   - Use Langfuse for LLM tracing
   - Use Arize for production monitoring
   - Use WhyLabs for data quality

3. **Error Handling**
   - Observability failures don't block governance
   - Providers automatically handle errors gracefully

4. **Performance**
   - Logging is async when possible
   - Minimal overhead (<10ms per event)
   - Automatic batching where supported

## Environment Variables

| Provider | Variables | Required |
|----------|-----------|----------|
| Langfuse | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | Yes |
| LangSmith | `LANGSMITH_API_KEY` | Yes |
| Arize | `ARIZE_API_KEY`, `ARIZE_SPACE_KEY` | Yes |
| WhyLabs | `WHYLABS_API_KEY`, `WHYLABS_ORG_ID` | Yes |
| Helicone | `HELICONE_API_KEY` | Yes |
| TruLens | None (local DB) | No |

## Examples

See complete examples in `examples/`:
- `langfuse_demo.py` - Langfuse integration
- `full_stack_demo.py` - Multi-provider setup

## Troubleshooting

**Provider not available?**
```python
from nethical.integrations.observability import get_observability_info

info = get_observability_info()
for name, details in info.items():
    if not details["available"]:
        print(f"{name}: {details['setup']}")
```

**Events not appearing?**
- Check API keys are correct
- Verify network connectivity
- Call `flush()` to force send buffered events

**Performance issues?**
- Reduce number of providers
- Use sampling for high-volume events
- Enable async mode where available

## Support

- **Documentation**: See provider-specific docs
- **Issues**: GitHub issues for Nethical
- **Community**: Nethical Discord/Slack
