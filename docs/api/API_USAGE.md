# Nethical v2.0 API Usage Guide

## Overview

The Nethical v2.0 REST API provides HTTP endpoints for remote evaluation of agent actions with structured responses. This enables integration with external platforms like LangChain, MCP, OpenAI tool wrappers, and custom applications.

## Quick Start

### Starting the Server

```bash
# Using uvicorn directly
uvicorn nethical.api:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn nethical.api:app --reload --port 8000

# Using Docker
docker-compose up nethical-api
```

### Basic Request

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "actual_action": "SELECT * FROM users"
  }'
```

## Endpoints

### POST /evaluate

Evaluate an action for ethical compliance and safety.

**Request Body:**

```json
{
  "id": "optional-action-id",
  "agent_id": "agent-identifier",
  "stated_intent": "optional stated goal",
  "actual_action": "the action to evaluate",
  "context": {
    "key": "value"
  },
  "parameters": {
    "action_type": "query"
  }
}
```

**Required Fields:**
- `agent_id` (string, 1-256 chars): Agent identifier
- `actual_action` (string, 1-50000 chars): Action to evaluate

**Optional Fields:**
- `id` (string): Action ID (auto-generated if not provided)
- `stated_intent` (string): Stated intent for deviation detection
- `context` (object): Additional context
- `parameters` (object): Evaluation parameters

**Response (200 OK):**

```json
{
  "judgment_id": "judgment_20251123_070000_abc123",
  "action_id": "action_20251123_070000_def456",
  "decision": "ALLOW",
  "confidence": 0.95,
  "reasoning": "Action evaluated and found safe",
  "violations": [],
  "timestamp": "2025-11-23T07:00:00.000Z",
  "risk_score": 0.1,
  "modifications": null,
  "metadata": {
    "semantic_monitoring": true,
    "agent_id": "my-agent",
    "has_intent": false
  }
}
```

**Decision Types:**
- `ALLOW`: Action is safe
- `WARN`: Action has minor concerns
- `BLOCK`: Action should be blocked
- `QUARANTINE`: Action needs review
- `ESCALATE`: Requires human intervention
- `TERMINATE`: Critical violation

### GET /status

Get system status and health.

**Response:**

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2025-11-23T07:00:00.000Z",
  "semantic_monitoring": true,
  "semantic_available": true,
  "components": {
    "ethical_monitoring": true,
    "safety_monitoring": true,
    "intent_monitoring": true
  }
}
```

### GET /metrics

Get evaluation metrics and statistics.

**Response:**

```json
{
  "total_evaluations": 1523,
  "total_violations": 42,
  "violation_by_type": {
    "ethical": 15,
    "safety": 12,
    "intent_deviation": 10,
    "manipulation": 5
  },
  "decisions_by_type": {
    "ALLOW": 1450,
    "WARN": 31,
    "BLOCK": 42
  },
  "avg_confidence": 0.92,
  "timestamp": "2025-11-23T07:00:00.000Z"
}
```

### GET /health

Simple health check for load balancers.

**Response:**

```json
{
  "status": "healthy"
}
```

### GET /

API information and documentation links.

## Client Examples

### Python with requests

```python
import requests

response = requests.post(
    "http://localhost:8000/evaluate",
    json={
        "agent_id": "python-agent",
        "stated_intent": "query user data",
        "actual_action": "SELECT * FROM users WHERE id = ?",
        "context": {"database": "production"}
    }
)

result = response.json()
print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']}")

if result['violations']:
    print("Violations detected:")
    for violation in result['violations']:
        print(f"  - {violation['description']}")
```

### Python with httpx (async)

```python
import asyncio
import httpx

async def evaluate_action(action_text):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/evaluate",
            json={
                "agent_id": "async-agent",
                "actual_action": action_text
            }
        )
        return response.json()

result = asyncio.run(evaluate_action("DELETE FROM users"))
print(result['decision'])
```

### JavaScript/Node.js

```javascript
const fetch = require('node-fetch');

async function evaluateAction(action) {
  const response = await fetch('http://localhost:8000/evaluate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      agent_id: 'js-agent',
      actual_action: action
    })
  });
  
  const result = await response.json();
  return result;
}

evaluateAction('SELECT * FROM users')
  .then(result => {
    console.log(`Decision: ${result.decision}`);
    console.log(`Confidence: ${result.confidence}`);
  });
```

### cURL Examples

**Simple evaluation:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "curl-agent", "actual_action": "print(hello)"}'
```

**With intent and context:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "curl-agent",
    "stated_intent": "log information",
    "actual_action": "logger.info(user_data)",
    "context": {"environment": "production"}
  }' | jq .
```

**Check status:**
```bash
curl http://localhost:8000/status | jq .
```

## Integration Patterns

### LangChain Integration

```python
from langchain.tools import Tool
import requests

def nethical_evaluate(action: str) -> str:
    """Evaluate action with Nethical."""
    response = requests.post(
        "http://localhost:8000/evaluate",
        json={
            "agent_id": "langchain-agent",
            "actual_action": action
        }
    )
    result = response.json()
    
    if result['decision'] in ['BLOCK', 'TERMINATE']:
        return f"Action blocked: {result['reasoning']}"
    elif result['decision'] == 'WARN':
        return f"Warning: {result['reasoning']}"
    return "Action allowed"

nethical_tool = Tool(
    name="NethicalEvaluate",
    func=nethical_evaluate,
    description="Evaluate actions for safety and ethics"
)
```

### OpenAI Function Calling

```python
import openai
import requests

tools = [
    {
        "type": "function",
        "function": {
            "name": "evaluate_safety",
            "description": "Evaluate an action for safety and ethical compliance",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action to evaluate"
                    }
                },
                "required": ["action"]
            }
        }
    }
]

def evaluate_safety(action: str):
    response = requests.post(
        "http://localhost:8000/evaluate",
        json={"agent_id": "openai-agent", "actual_action": action}
    )
    return response.json()
```

### MCP Server Integration

```python
from mcp import Server, Resource

class NethicalMCP(Server):
    async def handle_action(self, action):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/evaluate",
                json={
                    "agent_id": "mcp-server",
                    "actual_action": action.content
                }
            )
            result = response.json()
            
            if result['decision'] == 'BLOCK':
                raise SecurityViolation(result['reasoning'])
            
            return result
```

## Rate Limiting Recommendations

### Production Deployment

1. **Per-Agent Rate Limits:**
   ```python
   # Example with Redis
   from redis import Redis
   from datetime import datetime, timedelta
   
   redis = Redis()
   
   def check_rate_limit(agent_id: str, max_requests: int = 100, window_seconds: int = 60):
       key = f"rate_limit:{agent_id}"
       count = redis.incr(key)
       if count == 1:
           redis.expire(key, window_seconds)
       return count <= max_requests
   ```

2. **API Gateway:**
   - Use nginx or AWS API Gateway for rate limiting
   - Recommended: 100 requests/minute per agent

3. **Load Balancing:**
   ```yaml
   # docker-compose with multiple instances
   services:
     nethical-api-1:
       ...
     nethical-api-2:
       ...
     nginx:
       image: nginx
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
   ```

## Error Handling

### HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| 200 | Success | Evaluation completed |
| 400 | Bad Request | Invalid JSON |
| 422 | Validation Error | Missing required field |
| 500 | Server Error | Internal evaluation failure |
| 503 | Service Unavailable | Governance not initialized |

### Error Response Format

```json
{
  "detail": "Error message describing the problem"
}
```

### Handling Errors

```python
try:
    response = requests.post(
        "http://localhost:8000/evaluate",
        json={"agent_id": "test", "actual_action": "test"},
        timeout=5
    )
    response.raise_for_status()
    result = response.json()
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e.response.status_code}")
    print(f"Details: {e.response.json()}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tuning

### Request Optimization

1. **Batch Requests**: If evaluating multiple actions, use async:
   ```python
   async def batch_evaluate(actions):
       async with httpx.AsyncClient() as client:
           tasks = [
               client.post("http://localhost:8000/evaluate", json=action)
               for action in actions
           ]
           responses = await asyncio.gather(*tasks)
           return [r.json() for r in responses]
   ```

2. **Connection Pooling**:
   ```python
   session = requests.Session()
   # Reuse session for multiple requests
   for action in actions:
       response = session.post(url, json=action)
   ```

3. **Timeout Configuration**:
   ```python
   response = requests.post(
       url,
       json=payload,
       timeout=(3.0, 10.0)  # (connect, read)
   )
   ```

### Server Configuration

```bash
# Increase workers for higher throughput
uvicorn nethical.api:app --workers 4 --port 8000

# With production settings
uvicorn nethical.api:app \
  --workers 4 \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  --access-log
```

## Security Considerations

### CORS Configuration

For production, update CORS settings in `nethical/api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)
```

### API Authentication

Add authentication middleware:

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("NETHICAL_API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/evaluate")
async def evaluate(
    request: EvaluateRequest,
    api_key: str = Depends(verify_api_key)
):
    ...
```

### HTTPS

Always use HTTPS in production:

```bash
# With SSL certificate
uvicorn nethical.api:app \
  --ssl-keyfile=/path/to/key.pem \
  --ssl-certfile=/path/to/cert.pem
```

## Monitoring & Observability

### Health Checks

```bash
# Add to kubernetes/docker health check
curl -f http://localhost:8000/health || exit 1
```

### Metrics Collection

```python
# Prometheus metrics example
from prometheus_client import Counter, Histogram

evaluations_total = Counter('nethical_evaluations_total', 'Total evaluations')
evaluation_duration = Histogram('nethical_evaluation_duration_seconds', 'Evaluation duration')
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Troubleshooting

### Common Issues

**503 Service Unavailable:**
- Governance system not initialized
- Check logs for initialization errors

**422 Validation Error:**
- Missing required fields
- Field length exceeded
- Check request structure

**Slow Response Times:**
- First request loads semantic model (2-5s)
- Subsequent requests should be <200ms
- Consider preloading models in Docker

**Connection Refused:**
- Server not running
- Check port binding: `netstat -an | grep 8000`

## Further Reading

- [Semantic Monitoring Guide](./SEMANTIC_MONITORING_GUIDE.md)
- [Docker Deployment](../README.md#running-via-docker)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
