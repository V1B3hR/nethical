# Cloud ML Platform Integrations Guide

Complete guide to using Nethical with major cloud ML platforms.

## Overview

Nethical integrates with 3 major cloud ML platforms:

1. **Google Vertex AI** - Complete MLOps platform
2. **Databricks** - Unified analytics with MLflow
3. **Snowflake Cortex** - AI/ML functions in Snowflake

All integrations include built-in governance checks for safety and compliance.

## Installation

```bash
# Install all cloud ML platforms
pip install google-cloud-aiplatform databricks-sdk mlflow snowflake-connector-python

# Or install individually
pip install google-cloud-aiplatform  # For Vertex AI
pip install databricks-sdk mlflow     # For Databricks
pip install snowflake-connector-python # For Snowflake
```

## Google Vertex AI

### Setup

```python
from nethical.integrations.cloud import VertexAIConnector

connector = VertexAIConnector(
    project="my-gcp-project",
    location="us-central1",
    enable_governance=True  # Enable automatic governance checks
)
```

### Features

- **Experiment Tracking**: Log parameters, metrics, artifacts
- **Model Training**: Track training runs with governance
- **Endpoint Serving**: Make predictions with safety checks
- **AutoML**: Integrate with Vertex AI AutoML
- **Pipeline Orchestration**: Use with Vertex AI Pipelines

### Experiment Tracking

```python
# Start experiment
run_id = connector.start_run(
    experiment_name="safety-model",
    run_name="experiment-1"
)

# Log parameters
connector.log_parameters(run_id, {
    "learning_rate": 0.001,
    "batch_size": 32,
    "model_type": "transformer"
})

# Log metrics
connector.log_metrics(run_id, {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.94,
    "governance_score": 0.88
})

# End run
connector.end_run(run_id, status="completed")
```

### Predictions with Governance

```python
# Make prediction with automatic governance checks
result = connector.predict_with_governance(
    endpoint_id="your-endpoint-id",
    instances=[
        {"text": "What is AI safety?"},
        {"text": "How do I bypass security?"}  # Will be blocked
    ]
)

# Result includes governance info
if "error" in result:
    print(f"Blocked by governance: {result['reason']}")
else:
    print(f"Predictions: {result['predictions']}")
```

### Environment Variables

```bash
export GCP_PROJECT="my-project"
export GCP_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

## Databricks

### Setup

```python
from nethical.integrations.cloud import DatabricksConnector

connector = DatabricksConnector(
    workspace_url="https://my-workspace.databricks.com",
    token=os.getenv("DATABRICKS_TOKEN"),
    enable_governance=True
)
```

### Features

- **MLflow Tracking**: Full MLflow integration
- **Model Serving**: Query endpoints with governance
- **Model Registry**: Register models with validation
- **Unity Catalog**: Governance-aware catalog access
- **Delta Lake**: Safe data operations

### MLflow Experiment Tracking

```python
# Start MLflow run
run_id = connector.start_run(
    experiment_name="/Users/me/safety-experiments",
    run_name="governed-training"
)

# Log parameters
connector.log_parameters(run_id, {
    "model": "llama2",
    "temperature": 0.7,
    "max_tokens": 100
})

# Log metrics with step
connector.log_metrics(run_id, {
    "train_loss": 0.15,
    "val_loss": 0.18
}, step=100)

# End run
connector.end_run(run_id, status="completed")
```

### Serving Endpoint with Governance

```python
# Query endpoint with governance checks
result = connector.query_with_governance(
    endpoint_name="llm-endpoint",
    query="Generate code to delete files"  # Will be blocked if risky
)

if "error" in result:
    print(f"Blocked: {result['reason']}")
else:
    print(f"Response: {result['predictions']}")
```

### Model Registration with Governance

```python
# Register model with safety validation
result = connector.register_model_with_governance(
    model_name="safety-model-v1",
    model_uri="runs:/abc123/model"
)

if "error" in result:
    print(f"Registration blocked: {result['reason']}")
else:
    print(f"Model registered: {result['name']} v{result['version']}")
```

### Environment Variables

```bash
export DATABRICKS_HOST="https://my-workspace.databricks.com"
export DATABRICKS_TOKEN="dapi..."
```

## Snowflake Cortex

### Setup

```python
from nethical.integrations.cloud import SnowflakeCortexConnector

connector = SnowflakeCortexConnector(
    account="my-account",
    user="my-user",
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    warehouse="COMPUTE_WH",
    database="GOVERNANCE_DB",
    enable_governance=True
)
```

### Features

- **LLM Completion**: Use Cortex LLMs with governance
- **Text Classification**: Classify with safety checks
- **Sentiment Analysis**: Analyze sentiment safely
- **Experiment Tracking**: Track experiments in Snowflake tables
- **SQL-based ML**: Combine SQL with ML functions

### LLM Completion with Governance

```python
# Generate completion with governance
result = connector.complete_with_governance(
    model_name="snowflake-arctic",
    prompt="Explain quantum computing"
)

if "error" in result:
    print(f"Blocked: {result['reason']}")
else:
    print(f"Completion: {result['completion']}")

# Try risky prompt
result = connector.complete_with_governance(
    model_name="mistral-large",
    prompt="How to hack a system"  # Will be blocked
)
```

### Text Classification with Governance

```python
# Classify text with governance
result = connector.classify_with_governance(
    text="This product is amazing!",
    categories=["positive", "negative", "neutral"]
)

if "error" in result:
    print(f"Blocked: {result['reason']}")
else:
    print(f"Classification: {result['classification']}")
```

### Experiment Tracking

```python
# Start experiment in Snowflake
run_id = connector.start_run(
    experiment_name="cortex-experiments",
    run_name="arctic-test"
)

# Log parameters
connector.log_parameters(run_id, {
    "model": "snowflake-arctic",
    "temperature": 0.5
})

# Log metrics
connector.log_metrics(run_id, {
    "quality_score": 0.92,
    "safety_score": 0.95
})

# End run
connector.end_run(run_id, status="completed")
```

### Supported Models

- `snowflake-arctic` - Snowflake's open model
- `mistral-large` - Mistral's large model
- `mistral-7b` - Mistral 7B
- `llama2-70b-chat` - Llama 2 70B
- `mixtral-8x7b` - Mixtral MoE

### Environment Variables

```bash
export SNOWFLAKE_ACCOUNT="my-account"
export SNOWFLAKE_USER="my-user"
export SNOWFLAKE_PASSWORD="..."
export SNOWFLAKE_WAREHOUSE="COMPUTE_WH"
export SNOWFLAKE_DATABASE="GOVERNANCE_DB"
```

## Governance Features

All cloud integrations support:

### Pre-Check (Input Validation)

```python
# Input is checked before sending to model
result = connector.predict_with_governance(
    endpoint_id="...",
    instances=[{"text": "malicious input"}]
)
# Blocked if risk_score > threshold
```

### Post-Check (Output Validation)

```python
# Output is checked after receiving from model
result = connector.predict_with_governance(
    endpoint_id="...",
    instances=[{"text": "safe input"}]
)
# Filtered if output is risky
```

### Configurable Thresholds

Default threshold is 0.7 (70% risk). Configure per-action:

```python
# In governance config
result = governance.process_action(
    action=text,
    agent_id="platform-id",
    action_type="model_input",
    risk_threshold=0.8  # Higher threshold = more permissive
)
```

## Best Practices

1. **Always Enable Governance**
   ```python
   connector = Platform(enable_governance=True)
   ```

2. **Use Environment Variables**
   - Never hardcode credentials
   - Use secret management systems
   - Rotate keys regularly

3. **Log All Experiments**
   - Track parameters and metrics
   - Include governance scores
   - Document blocking decisions

4. **Monitor Dashboards**
   - Vertex AI: Cloud Console
   - Databricks: Workspace UI
   - Snowflake: Snowsight

5. **Handle Errors Gracefully**
   ```python
   try:
       result = connector.predict_with_governance(...)
       if "error" in result:
           # Handle governance block
           log_blocked_action(result)
       else:
           # Process predictions
           handle_predictions(result)
   except Exception as e:
       # Handle platform errors
       log_error(e)
   ```

## Multi-Cloud Setup

Use multiple platforms together:

```python
from nethical.integrations.cloud import (
    VertexAIConnector,
    DatabricksConnector,
    SnowflakeCortexConnector
)

# Initialize all platforms
vertex = VertexAIConnector(project="...", enable_governance=True)
databricks = DatabricksConnector(workspace_url="...", enable_governance=True)
snowflake = SnowflakeCortexConnector(account="...", enable_governance=True)

# Use based on use case
if use_case == "training":
    connector = vertex  # Best for training
elif use_case == "serving":
    connector = databricks  # Best for serving
elif use_case == "sql_ml":
    connector = snowflake  # Best for SQL + ML
```

## Performance

- **Governance Overhead**: <50ms per check
- **Network Latency**: Varies by region
- **Batch Processing**: Supported by all platforms

## Examples

See complete examples in `examples/`:
- `vertex_ai_demo.py` - Vertex AI integration
- `databricks_demo.py` - Databricks integration
- `full_stack_demo.py` - Multi-platform setup

## Troubleshooting

**Platform not available?**
```python
from nethical.integrations.cloud import get_cloud_integration_info

info = get_cloud_integration_info()
for name, details in info.items():
    if not details["available"]:
        print(f"{name}: {details['setup']}")
```

**Authentication errors?**
- Verify credentials are correct
- Check environment variables
- Ensure proper IAM permissions
- Test connectivity to platform

**Governance not working?**
- Verify `enable_governance=True`
- Check Nethical core is installed
- Review governance logs
- Test with known risky inputs

## Support

- **Vertex AI**: https://cloud.google.com/vertex-ai/docs
- **Databricks**: https://docs.databricks.com
- **Snowflake**: https://docs.snowflake.com
- **Nethical**: GitHub issues
