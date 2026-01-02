# MCP & AI Platform Integration Guide

Complete guide for integrating Nethical with Multimodal Control Platforms (MCPs), MLOps platforms, and AI orchestration infrastructures.

## Table of Contents

- [Overview](#overview)
- [Supported Platforms](#supported-platforms)
- [Integration Patterns](#integration-patterns)
- [Platform-Specific Guides](#platform-specific-guides)
- [Enterprise Integrations](#enterprise-integrations)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

Nethical provides comprehensive safety and ethics governance for AI platforms and MLOps infrastructures. It integrates seamlessly with:

- **Agent Frameworks**: LangChain, AutoGen, CrewAI
- **ML Platforms**: HuggingFace, MLflow, Ray Serve
- **Cloud ML Services**: AWS SageMaker, Azure ML, Google Vertex AI
- **Experiment Tracking**: Weights & Biases, MLflow
- **Data Platforms**: Databricks

### Key Capabilities

- 🔄 **Lifecycle Governance**: Safety checks throughout the ML lifecycle
- 🤖 **Multi-Agent Safety**: Monitor and govern agent interactions
- 📊 **MLOps Integration**: Seamless integration with existing workflows
- 🏢 **Enterprise Ready**: Support for major cloud platforms
- 📈 **Observability**: Integration with monitoring stacks

## Supported Platforms

| Platform | Type | Status | Manifest |
|----------|------|--------|----------|
| **LangChain** | Agent Framework | ✅ Stable | `langchain-tool.json` |
| **HuggingFace** | ML Platform | ✅ Stable | `huggingface-tool.yaml` |
| **AutoGen** | Multi-Agent | ✅ Stable | `autogen-manifest.json` |
| **Ray Serve** | Model Serving | 🔨 Beta | - |
| **MLflow** | MLOps | 🔨 Beta | `mlflow-integration.yaml` |
| **AWS SageMaker** | Cloud ML | 📋 Stub | `enterprise-mcp-integrations.yaml` |
| **Azure ML** | Cloud ML | 📋 Stub | `enterprise-mcp-integrations.yaml` |
| **Vertex AI** | Cloud ML | 📋 Stub | `enterprise-mcp-integrations.yaml` |
| **Weights & Biases** | Experiment Tracking | 📋 Stub | `enterprise-mcp-integrations.yaml` |
| **Databricks** | Data Platform | 📋 Stub | `enterprise-mcp-integrations.yaml` |

**Status Legend:**
- ✅ Stable: Production-ready
- 🔨 Beta: Working with limited testing
- 📋 Stub: Connector stub and manifest available

## Integration Patterns

### Pattern 1: Tool/Function Integration

Add Nethical as a tool in agent frameworks.

**Use Cases:**
- LangChain agents
- AutoGen assistants
- Custom agent systems

**Benefits:**
- Native integration
- Agents can self-check
- Context-aware decisions

### Pattern 2: Pipeline Wrapper

Wrap ML pipelines with safety checks.

**Use Cases:**
- HuggingFace pipelines
- Inference endpoints
- Model serving

**Benefits:**
- Transparent integration
- No code changes needed
- Automatic monitoring

### Pattern 3: Middleware Layer

Add as middleware in serving platforms.

**Use Cases:**
- Ray Serve deployments
- MLflow serving
- Custom APIs

**Benefits:**
- Centralized governance
- Pre/post processing
- Observability integration

### Pattern 4: Lifecycle Hooks

Integrate at key lifecycle points.

**Use Cases:**
- MLflow model registry
- SageMaker deployments
- Azure ML pipelines

**Benefits:**
- Validation gates
- Compliance checks
- Audit trails

## Platform-Specific Guides

### LangChain Integration

LangChain is a popular framework for building LLM applications.

#### Installation

```bash
pip install nethical langchain
```

#### Basic Tool Integration

```python
from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from nethical.integrations.langchain_tools import NethicalTool

# Create Nethical tool
nethical_tool = NethicalTool()

# Initialize agent with Nethical
tools = [nethical_tool, *other_tools]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Agent can now check actions for safety
result = agent.run("Execute this action...")
```

#### Chain Integration

```python
from langchain.chains import LLMChain
from nethical.integrations.langchain_tools import create_nethical_chain

# Create safe chain
safe_chain = create_nethical_chain(
    llm=llm,
    check_input=True,
    check_output=True
)

result = safe_chain.run("Your prompt here")
```

#### Callback Integration

```python
from nethical.integrations.langchain_tools import NethicalCallback

# Add callback for monitoring
callbacks = [NethicalCallback()]

chain = LLMChain(llm=llm, callbacks=callbacks)
result = chain.run("Your prompt")
```

#### Use Cases

- **Pre-execution Guards**: Check prompts before LLM calls
- **Output Filters**: Validate LLM responses
- **Agent Safety**: Let agents check their own actions
- **RAG Protection**: Validate retrieved documents and generations

### HuggingFace Integration

HuggingFace provides transformers and model hosting.

#### Installation

```bash
pip install nethical transformers
```

#### Pipeline Wrapper

```python
from transformers import pipeline
from nethical.integrations.ml_platforms import wrap_hf_pipeline

# Create standard pipeline
pipe = pipeline("text-generation", model="gpt2")

# Wrap with Nethical
safe_pipe = wrap_hf_pipeline(
    pipe,
    check_input=True,
    check_output=True
)

# Use normally - automatically protected
result = safe_pipe("Generate code to...")
```

#### Inference API Integration

```python
from huggingface_hub import InferenceClient
from nethical.integrations.ml_platforms import check_with_nethical

client = InferenceClient()

def safe_inference(prompt, model="gpt2"):
    # Pre-check
    if not check_with_nethical(prompt):
        return {"error": "Input blocked"}
    
    # Inference
    result = client.text_generation(prompt, model=model)
    
    # Post-check
    if not check_with_nethical(result):
        return {"error": "Output filtered"}
    
    return result
```

#### Gradio Spaces

```python
import gradio as gr
from nethical.integrations.ml_platforms import create_gradio_wrapper

def model_fn(text):
    # Your model logic
    return model.generate(text)

# Wrap with Nethical
safe_fn = create_gradio_wrapper(
    model_fn,
    enable_pii_detection=True,
    enable_risk_scoring=True
)

# Create Gradio interface
demo = gr.Interface(
    fn=safe_fn,
    inputs="text",
    outputs="text"
)

demo.launch()
```

### AutoGen Integration

Microsoft AutoGen enables multi-agent conversations.

#### Installation

```bash
pip install nethical pyautogen
```

#### Agent Wrapper

```python
from autogen import AssistantAgent, UserProxyAgent
from nethical.integrations.ml_platforms import wrap_autogen_agent

# Create agents
assistant = AssistantAgent("assistant")
user_proxy = UserProxyAgent("user")

# Wrap with Nethical
safe_assistant = wrap_autogen_agent(
    assistant,
    check_messages=True,
    check_function_calls=True
)

# Use normally - agent actions are monitored
user_proxy.initiate_chat(
    safe_assistant,
    message="Execute this task..."
)
```

#### Message Filter

```python
from nethical.integrations.ml_platforms import create_nethical_filter

# Create filter
filter_fn = create_nethical_filter(
    block_unsafe=True,
    redact_pii=True
)

# Register filter
user_proxy.register_reply(
    AssistantAgent,
    filter_fn
)
```

#### Function Call Guard

```python
from nethical.integrations.ml_platforms import guard_function_call

@guard_function_call
def sensitive_function(args):
    """This function is automatically checked before execution."""
    # Function implementation
    return result
```

#### Group Chat Monitoring

```python
from autogen import GroupChat, GroupChatManager
from nethical.integrations.ml_platforms import create_group_chat_monitor

# Create group chat
group_chat = GroupChat(agents=[agent1, agent2, agent3])

# Add monitor
monitor = create_group_chat_monitor(
    log_all_messages=True,
    block_unsafe=True
)

manager = GroupChatManager(
    groupchat=group_chat,
    monitors=[monitor]
)
```

### Ray Serve Integration

Ray Serve enables scalable model serving.

#### Installation

```bash
pip install nethical ray[serve]
```

#### Deployment Wrapper

```python
from ray import serve
from nethical.integrations.ray_serve_connector import NethicalRayServeMiddleware

@serve.deployment
class MyModel:
    def __call__(self, request):
        return self.model.predict(request)

# Wrap with Nethical
safe_model = NethicalRayServeMiddleware(
    MyModel,
    check_input=True,
    check_output=True
)

serve.run(safe_model.bind())
```

#### Decorator Style

```python
from nethical.integrations.ray_serve_connector import create_safe_deployment

@serve.deployment
@create_safe_deployment
def predict(request):
    return model.predict(request)

serve.run(predict.bind())
```

### MLflow Integration

MLflow provides ML lifecycle management.

#### Installation

```bash
pip install nethical mlflow
```

#### Model Wrapper

```python
import mlflow
from nethical.integrations.ml_platforms import wrap_mlflow_model

# Train model
model = train_model()

# Wrap with Nethical
safe_model = wrap_mlflow_model(
    model,
    check_input=True,
    check_output=True
)

# Log to MLflow
mlflow.sklearn.log_model(safe_model, "safe_model")
```

#### Custom Model with Nethical

```python
import mlflow.pyfunc
from nethical.integrations.ml_platforms import NethicalModelWrapper

class SafeModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.wrapper = NethicalModelWrapper()
    
    def predict(self, context, model_input):
        # Check input
        if not self.wrapper.check_input(model_input):
            return {"error": "Input blocked"}
        
        # Predict
        prediction = self.model.predict(model_input)
        
        # Check output
        if not self.wrapper.check_output(prediction):
            return {"error": "Output blocked"}
        
        return prediction

# Register model
mlflow.pyfunc.log_model("safe_model", python_model=SafeModel())
```

## Enterprise Integrations

### AWS SageMaker

```python
from nethical.integrations.ml_platforms import SageMakerConnector

connector = SageMakerConnector(
    endpoint_name="my-endpoint",
    enable_safety_checks=True,
    enable_pii_detection=True
)

# Predictions automatically checked
result = connector.predict(input_data)
```

**Features:**
- Real-time endpoint monitoring
- Batch transform validation
- Pipeline step guards
- CloudWatch integration

**Documentation:** See `enterprise-mcp-integrations.yaml`

### Azure Machine Learning

```python
from nethical.integrations.ml_platforms import AzureMLConnector

connector = AzureMLConnector(
    workspace_name="my-workspace",
    endpoint_name="my-endpoint",
    enable_governance=True
)

result = connector.predict(input_data)
```

**Features:**
- Online endpoint monitoring
- Pipeline component validation
- Application Insights integration
- Managed identity support

### Google Vertex AI

```python
from nethical.integrations.ml_platforms import VertexAIConnector

connector = VertexAIConnector(
    project="my-project",
    endpoint_id="my-endpoint",
    enable_governance=True
)

result = connector.predict(instances)
```

**Features:**
- Endpoint monitoring
- Batch prediction validation
- Cloud Logging integration
- VPC-SC support

### Weights & Biases

```python
import wandb
from nethical.integrations.ml_platforms import init_wandb_with_nethical

# Initialize with Nethical
run = init_wandb_with_nethical(
    project="my-project",
    enable_safety_logging=True
)

# Safety metrics automatically logged
wandb.log({"prediction": output})
```

**Features:**
- Safety metrics logging
- Risk score tracking
- PII detection metrics
- Artifact validation

### Databricks

```python
from nethical.integrations.ml_platforms import DatabricksConnector

connector = DatabricksConnector(
    workspace_url="https://my-workspace.databricks.com",
    model_name="my-model",
    enable_governance=True
)

result = connector.predict(input_data)
```

**Features:**
- Model serving monitoring
- Job safety validation
- Unity Catalog integration
- Delta Lake integration

## Best Practices

### 1. Choose the Right Integration Pattern

| Use Case | Pattern | Platform Examples |
|----------|---------|-------------------|
| Agent applications | Tool Integration | LangChain, AutoGen |
| Model serving | Middleware | Ray Serve, MLflow |
| Cloud deployments | Lifecycle Hooks | SageMaker, Azure ML |
| Experimentation | Logging Integration | W&B, MLflow |

### 2. Enable Appropriate Features

```python
# Development
connector = Platform(
    enable_safety_checks=True,
    enable_pii_detection=False,  # If not needed
    verbose_logging=True
)

# Production
connector = Platform(
    enable_safety_checks=True,
    enable_pii_detection=True,
    enable_audit_logging=True,
    enable_performance_optimization=True
)
```

### 3. Monitor Performance

```python
# Configure observability
connector = Platform(
    enable_metrics=True,
    enable_tracing=True,
    metrics_backend="prometheus",
    tracing_backend="jaeger"
)
```

### 4. Implement Gradual Rollout

```python
# Start in monitoring mode
connector = Platform(
    enforcement_mode="monitor",  # Log violations but don't block
)

# After validation, enable enforcement
connector.set_enforcement_mode("block")
```

### 5. Handle Failures Gracefully

```python
try:
    result = connector.predict(input_data)
except SafetyViolationError as e:
    # Log violation
    logger.error(f"Safety violation: {e}")
    # Return safe fallback
    return fallback_response
```

## Configuration Examples

### Development Configuration

```python
from nethical.core.integrated_governance import IntegratedGovernance

governance = IntegratedGovernance(
    storage_dir="./dev_data",
    enable_quota_enforcement=False,
    enable_performance_optimization=False,
    enable_merkle_anchoring=False,
    verbose=True
)
```

### Production Configuration

```python
governance = IntegratedGovernance(
    storage_dir="/var/lib/nethical/data",
    enable_quota_enforcement=True,
    enable_performance_optimization=True,
    enable_merkle_anchoring=True,
    enable_ethical_taxonomy=True,
    enable_sla_monitoring=True,
    region_id="us-east-1",
    compliance_mode="strict"
)
```

### Multi-Region Configuration

```python
governance = IntegratedGovernance(
    storage_dir=f"/data/{region}",
    region_id=region,
    enable_regional_compliance=True,
    data_residency_rules={
        "eu": ["GDPR"],
        "us": ["HIPAA", "CCPA"]
    }
)
```

## Troubleshooting

### Integration Not Working

1. Check platform compatibility
2. Verify installation: `pip list | grep nethical`
3. Check logs for errors
4. Review manifest files

### Performance Issues

```python
# Enable performance optimization
connector.enable_performance_optimization()

# Reduce logging verbosity
connector.set_log_level("WARNING")

# Use async where available
result = await connector.predict_async(input_data)
```

### Compliance Violations

```python
# Check compliance status
status = governance.get_compliance_status()
print(f"Violations: {status['violations']}")

# Review audit logs
logs = governance.get_audit_logs(limit=100)
```

## Additional Resources

- [LLM Integration Guide](./LLM_INTEGRATION_GUIDE.md)
- [API Reference](./EXTERNAL_INTEGRATIONS_GUIDE.md)
- [Manifests](../) - All manifest files
- [Examples](../examples/)
- [GitHub Repository](https://github.com/V1B3hR/nethical)

## Support

- 📧 Email: support@nethical.dev
- 💬 Discussions: https://github.com/V1B3hR/nethical/discussions
- 🐛 Issues: https://github.com/V1B3hR/nethical/issues
- 📚 Documentation: https://github.com/V1B3hR/nethical/tree/main/docs

## Contributing

We welcome contributions for new platform integrations! See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

MIT License - See [LICENSE](../LICENSE) for details.
