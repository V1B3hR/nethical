# Nethical Quickstart Guide

Get up and running with Nethical in minutes. This guide covers installation, basic usage, and your first governance evaluation.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Optional: Redis (for production deployments with caching/quotas)

## Installation

### From PyPI (Recommended)

```bash
pip install nethical
```

### From Source

```bash
git clone https://github.com/V1B3hR/nethical.git
cd nethical
pip install -e .
```

### Development Installation

```bash
pip install -e ".[test]"
```

## Quick Start

### 1. Basic Governance Evaluation

```python
from nethical.core.integrated_governance import IntegratedGovernance

# Initialize governance
governance = IntegratedGovernance(
    storage_dir="./nethical_data",
)

# Evaluate an agent action
import asyncio

async def evaluate_action():
    result = await governance.process_action(
        agent_id="my-agent",
        stated_intent="Help user with their question",
        actual_action="SELECT name, email FROM users WHERE id = ?",
        context={"user_id": "user123"},
    )
    return result

result = asyncio.run(evaluate_action())
print(result)
```

### 2. Using the REST API

Start the API server:

```bash
uvicorn nethical.api:app --host 0.0.0.0 --port 8000
```

Make a request:

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "actual_action": "SELECT * FROM users"
  }'
```

### 3. Regional Configuration (GDPR/CCPA)

```python
governance = IntegratedGovernance(
    storage_dir="./nethical_data",
    region_id="eu-west-1",
    data_residency_policy="EU_GDPR",
)
```

## Key Features

### Safety Detection

Nethical detects various safety concerns:
- SQL injection attempts
- Cross-site scripting (XSS)
- Prompt injection/jailbreak attempts
- PII exposure
- Resource abuse

### Privacy Protection

Built-in privacy features:
- PII detection and redaction
- Differential privacy support
- Data minimization
- Regional compliance (GDPR, CCPA)

### Audit Logging

Comprehensive audit trails:
- Merkle-anchored logs for integrity
- Immutable audit history
- Compliance reporting

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `storage_dir` | Data storage directory | `./nethical_data` |
| `region_id` | Geographic region | `None` |
| `data_residency_policy` | Compliance policy | `None` |
| `enable_shadow_mode` | ML shadow classification | `True` |
| `enable_quota_enforcement` | Rate limiting | `False` |

## Next Steps

- [API Reference](API_USAGE.md) - Full API documentation
- [Deployment Guide](DEPLOYMENT.md) - Production deployment
- [Security Guide](SECURITY_HARDENING_GUIDE.md) - Security best practices
- [LLM Integration](LLM_INTEGRATION_GUIDE.md) - Integrate with LLMs

## Getting Help

- [GitHub Issues](https://github.com/V1B3hR/nethical/issues)
- [Documentation](https://github.com/V1B3hR/nethical/docs)
- [Examples](https://github.com/V1B3hR/nethical/examples)

## License

Nethical is released under the MIT License. See [LICENSE](../LICENSE) for details.
