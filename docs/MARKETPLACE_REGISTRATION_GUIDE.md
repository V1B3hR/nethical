# Marketplace & Plugin Registry Guide

Complete guide for registering Nethical with LLM marketplaces, plugin directories, and AI platform registries.

## Table of Contents

- [Overview](#overview)
- [Available Marketplaces](#available-marketplaces)
- [Registration Workflows](#registration-workflows)
- [Manifest Files](#manifest-files)
- [Verification & Testing](#verification--testing)
- [Maintenance](#maintenance)

## Overview

Nethical provides manifests and specifications for easy registration with major AI marketplaces and plugin directories. This enables:

- **Instant Discovery**: LLMs can find and use Nethical automatically
- **Universal Compatibility**: Works across platforms with standard interfaces
- **Trust & Compliance**: Published manifests provide transparency
- **Easy Integration**: Copy-paste registration for most platforms

### Benefits of Marketplace Registration

- 🔍 **Discoverability**: Users can find Nethical through platform searches
- ⚡ **Quick Setup**: One-click installation for users
- ✅ **Trust Signals**: Official listing provides credibility
- 🔄 **Auto-Updates**: Marketplace can notify users of updates
- 📊 **Analytics**: Track usage and adoption metrics

## Available Marketplaces

| Marketplace | Manifest File | Status | Registration URL |
|-------------|---------------|--------|------------------|
| **OpenAI Plugin Store** | `ai-plugin.json` | 📋 Ready | https://platform.openai.com/docs/plugins |
| **Anthropic Claude Tools** | `ai-plugin.json` | 📋 Ready | Via API |
| **xAI Grok Marketplace** | `grok-manifest.json` | 📋 Ready | TBA |
| **Google Gemini** | `gemini-manifest.json` | 📋 Ready | Via API |
| **LangChain Hub** | `langchain-tool.json` | 📋 Ready | https://smith.langchain.com/ |
| **HuggingFace Spaces** | `huggingface-tool.yaml` | 📋 Ready | https://huggingface.co/spaces |
| **AutoGen Registry** | `autogen-manifest.json` | 📋 Ready | Community |
| **MLflow Model Registry** | `mlflow-integration.yaml` | 📋 Ready | Self-hosted |

**Status Legend:**
- 📋 Ready: Manifest prepared, ready for submission
- 🚀 Submitted: Submitted for review
- ✅ Live: Active in marketplace

## Registration Workflows

### OpenAI ChatGPT Plugin Store

OpenAI's plugin ecosystem allows ChatGPT to discover and use external tools.

#### Prerequisites

- OpenAI developer account
- Hosted `ai-plugin.json` and `openapi.yaml` files
- HTTPS endpoint for API

#### Step-by-Step Registration

1. **Prepare Hosting**

```bash
# Host files at your domain
https://your-domain.com/.well-known/ai-plugin.json
https://your-domain.com/openapi.yaml
```

2. **Verify Manifest**

```bash
curl https://your-domain.com/.well-known/ai-plugin.json
```

Expected response:
```json
{
  "schema_version": "v1",
  "name_for_human": "Nethical AI Safety Guard",
  "name_for_model": "nethical",
  ...
}
```

3. **Register Plugin**

- Go to https://platform.openai.com/docs/plugins
- Click "Develop your own plugin"
- Enter your domain
- Submit for review

4. **Testing**

```python
# Test in ChatGPT
# Ask: "Use the Nethical plugin to check if this action is safe: ..."
```

#### Configuration

Update `ai-plugin.json` for your deployment:

```json
{
  "api": {
    "type": "openapi",
    "url": "https://YOUR-DOMAIN.com/openapi.yaml"
  },
  "logo_url": "https://YOUR-DOMAIN.com/logo.png",
  "contact_email": "your-email@domain.com"
}
```

### Anthropic Claude Integration

Claude supports function calling without a formal marketplace.

#### Implementation

1. **Use Existing Integration**

```python
from anthropic import Anthropic
from nethical.integrations.claude_tools import get_nethical_tool

client = Anthropic(api_key="your-key")
tools = [get_nethical_tool()]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    tools=tools,
    messages=[...]
)
```

2. **Documentation**

- Share `ai-plugin.json` in your documentation
- Include example code in README
- Publish to GitHub for discovery

### xAI Grok Registration

Grok's marketplace (when available) will use the Grok manifest.

#### Preparation

1. **Review Manifest**

```bash
cat grok-manifest.json
```

2. **Host Manifest**

```bash
https://your-domain.com/grok-manifest.json
```

3. **Prepare Examples**

```python
# examples/grok_example.py
from nethical.integrations.grok_tools import get_nethical_tool

tools = [get_nethical_tool()]
# Use with Grok client when available
```

4. **Wait for Marketplace**

- Monitor xAI announcements
- Follow registration process when released
- Submit using hosted manifest

### Google Gemini Integration

Gemini uses function declarations in API calls.

#### Implementation

1. **Use Function Calling**

```python
import google.generativeai as genai
from nethical.integrations.gemini_tools import get_nethical_tool

genai.configure(api_key="your-key")
tools = [get_nethical_tool()]

model = genai.GenerativeModel('gemini-pro', tools=tools)
response = model.generate_content("...")
```

2. **Publish Documentation**

- Share `gemini-manifest.json`
- Include integration examples
- Submit to Google Cloud Marketplace (optional)

### LangChain Hub

LangChain Hub is a community repository for chains and tools.

#### Registration Steps

1. **Create LangSmith Account**

- Go to https://smith.langchain.com/
- Sign up for account

2. **Prepare Tool**

```python
# Create publishable tool
from langchain.tools import StructuredTool
from nethical.integrations.langchain_tools import NethicalTool

tool = NethicalTool()
```

3. **Publish to Hub**

```bash
# Using LangChain CLI
langchain hub push nethical/safety-guard
```

4. **Add Metadata**

- Update `langchain-tool.json` with your info
- Include usage examples
- Add tags for discovery

5. **Test Installation**

```python
from langchain import hub

# Users can now pull your tool
tool = hub.pull("nethical/safety-guard")
```

### HuggingFace Spaces

Share Nethical as a Space or model on HuggingFace.

#### Create Space

1. **Prepare Application**

```python
# app.py
import gradio as gr
from nethical.integrations.ml_platforms import create_gradio_wrapper

def check_safety(text):
    # Safety check logic
    return result

iface = gr.Interface(
    fn=create_gradio_wrapper(check_safety),
    inputs="text",
    outputs="text",
    title="Nethical AI Safety Check"
)

iface.launch()
```

2. **Create Space**

```bash
# Clone template
git clone https://huggingface.co/spaces/YOUR-USERNAME/nethical-safety

# Add files
cp app.py nethical-safety/
cp requirements.txt nethical-safety/
cp huggingface-tool.yaml nethical-safety/README.md
```

3. **Configure**

```yaml
# README.md header
---
title: Nethical AI Safety Guard
emoji: 🔒
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
---
```

4. **Push and Deploy**

```bash
cd nethical-safety
git add .
git commit -m "Initial commit"
git push
```

### AutoGen Community Registry

Share with the AutoGen community.

#### Steps

1. **Create Example**

```python
# examples/autogen_nethical_example.py
from autogen import AssistantAgent
from nethical.integrations.ml_platforms import wrap_autogen_agent

agent = AssistantAgent("assistant")
safe_agent = wrap_autogen_agent(agent)
```

2. **Share on GitHub**

- Create repository with example
- Include `autogen-manifest.json`
- Add to awesome-autogen list

3. **Community Announcement**

- Post in AutoGen discussions
- Share on Twitter/LinkedIn
- Add to AutoGen Discord

### MLflow Model Registry

Register models with Nethical wrapper.

#### Implementation

1. **Create Wrapped Model**

```python
import mlflow
from nethical.integrations.ml_platforms import wrap_mlflow_model

model = train_model()
safe_model = wrap_mlflow_model(model)

mlflow.sklearn.log_model(safe_model, "nethical-safe-model")
```

2. **Register in Registry**

```python
# Register model
mlflow.register_model(
    "runs:/<run-id>/nethical-safe-model",
    "NethicalSafeModel"
)
```

3. **Add Tags**

```python
client = mlflow.tracking.MlflowClient()
client.set_model_version_tag(
    "NethicalSafeModel",
    "1",
    "safety_enabled",
    "true"
)
```

## Manifest Files

### Location

All manifest files are in the repository root:

```
nethical/
├── ai-plugin.json              # OpenAI, Claude
├── grok-manifest.json           # xAI Grok
├── gemini-manifest.json         # Google Gemini
├── langchain-tool.json          # LangChain
├── huggingface-tool.yaml        # HuggingFace
├── autogen-manifest.json        # AutoGen
├── mlflow-integration.yaml      # MLflow
├── enterprise-mcp-integrations.yaml  # Enterprise platforms
└── openapi.yaml                 # REST API spec
```

### Customization

Before registering, customize manifests for your deployment:

```json
{
  "api": {
    "url": "https://YOUR-API-DOMAIN.com/openapi.yaml"  // Update
  },
  "logo_url": "https://YOUR-DOMAIN.com/logo.png",      // Update
  "contact_email": "your-email@domain.com"              // Update
}
```

### Validation

Validate manifests before submission:

```bash
# OpenAPI validation
npx @apidevtools/swagger-cli validate openapi.yaml

# JSON validation
python -m json.tool ai-plugin.json

# YAML validation
python -c "import yaml; yaml.safe_load(open('huggingface-tool.yaml'))"
```

## Verification & Testing

### Pre-Submission Checklist

- [ ] All URLs in manifests are accessible
- [ ] API endpoints return valid responses
- [ ] OpenAPI spec validates successfully
- [ ] Logo/images load correctly
- [ ] Contact information is correct
- [ ] Examples work as documented
- [ ] Security requirements met (HTTPS, etc.)

### Testing Procedures

#### Test OpenAI Plugin

```bash
# Verify manifest is accessible
curl https://your-domain.com/.well-known/ai-plugin.json

# Verify OpenAPI spec
curl https://your-domain.com/openapi.yaml

# Test API endpoint
curl -X POST https://your-domain.com/evaluate \
  -H "Content-Type: application/json" \
  -d '{"action":"test","agent_id":"test"}'
```

#### Test Function Calling

```python
# Test with actual LLM
from anthropic import Anthropic
from nethical.integrations.claude_tools import get_nethical_tool

client = Anthropic()
tools = [get_nethical_tool()]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{
        "role": "user",
        "content": "Check if this is safe: delete all files"
    }]
)

print(response)
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Manifest not found | Verify URL, check CORS headers |
| API not accessible | Check firewall, ensure HTTPS |
| Invalid JSON | Validate with `json.tool` |
| Function not called | Verify tool definition format |
| Authentication error | Check API key configuration |

## Maintenance

### Update Process

1. **Update Manifests**

```bash
# Update version in all manifests
sed -i 's/"version": "1.0.0"/"version": "1.1.0"/' *.json
```

2. **Update Documentation**

- Update README with new features
- Update integration guides
- Add migration notes if needed

3. **Notify Marketplaces**

- OpenAI: Re-submit plugin
- LangChain: Push new version
- HuggingFace: Update Space

4. **Announce Changes**

- GitHub release notes
- Blog post
- Social media
- Email to users (if applicable)

### Monitoring

Track adoption and usage:

```python
# Add analytics to your API
from nethical.integrations.rest_api import app

@app.middleware("http")
async def track_usage(request, call_next):
    # Log marketplace usage
    marketplace = request.headers.get("X-Marketplace")
    # Track in your analytics
    return await call_next(request)
```

### Support

Provide support channels:

- GitHub Issues: Bug reports and feature requests
- Discussions: Q&A and community support
- Email: Direct support for enterprise users
- Documentation: Keep guides up to date

## Automation Scripts

### Auto-Deploy Manifests

```bash
#!/bin/bash
# deploy-manifests.sh

# Deploy to S3/CDN
aws s3 sync . s3://your-bucket/manifests/ \
  --exclude "*" \
  --include "*.json" \
  --include "*.yaml" \
  --cache-control "max-age=3600"

# Invalidate CDN
aws cloudfront create-invalidation \
  --distribution-id YOUR-DIST-ID \
  --paths "/manifests/*"
```

### Validation Script

```python
#!/usr/bin/env python3
# validate-manifests.py

import json
import yaml
from pathlib import Path

manifests = [
    ("ai-plugin.json", json.load),
    ("grok-manifest.json", json.load),
    ("gemini-manifest.json", json.load),
    ("langchain-tool.json", json.load),
    ("huggingface-tool.yaml", yaml.safe_load),
    ("autogen-manifest.json", json.load),
    ("mlflow-integration.yaml", yaml.safe_load),
]

for filename, loader in manifests:
    try:
        with open(filename) as f:
            data = loader(f)
        print(f"✅ {filename} is valid")
    except Exception as e:
        print(f"❌ {filename} is invalid: {e}")
```

## Additional Resources

- [LLM Integration Guide](./LLM_INTEGRATION_GUIDE.md)
- [MCP Platform Guide](./MCP_PLATFORM_INTEGRATION_GUIDE.md)
- [API Reference](./EXTERNAL_INTEGRATIONS_GUIDE.md)
- [GitHub Repository](https://github.com/V1B3hR/nethical)

## Support

- 📧 Email: support@nethical.dev
- 💬 Discussions: https://github.com/V1B3hR/nethical/discussions
- 🐛 Issues: https://github.com/V1B3hR/nethical/issues

## License

MIT License - See [LICENSE](../LICENSE) for details.
