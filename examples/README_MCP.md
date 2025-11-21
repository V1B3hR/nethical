# Nethical MCP Examples

This directory contains examples demonstrating how to use the Nethical MCP (Model Context Protocol) Server.

## Files

### `mcp_client_example.py`

A complete example client that demonstrates all Nethical MCP tools:

1. **Initialize Connection** - Connect to the MCP server
2. **List Tools** - Discover available tools
3. **Check PII** - Detect personally identifiable information
4. **Check Violations** - Identify ethical/safety violations
5. **Evaluate Actions** - Full ethics evaluation of code/actions
6. **System Status** - Check governance system health

## Running the Examples

### 1. Start the MCP Server

In one terminal:
```bash
cd /path/to/nethical
python -m nethical.mcp_server
```

The server will start on `http://localhost:8000`.

### 2. Run the Example Client

In another terminal:
```bash
cd /path/to/nethical
python examples/mcp_client_example.py
```

## What You'll See

The example will demonstrate:

### PII Detection
```
3. Example: Checking for PII in text...
# PII Detection Results

**Found 3 PII instance(s):**

1. **email**: `john.doe@example.com` (confidence: 50%)
2. **phone**: `555-123-4567` (confidence: 85%)
3. **phone**: `123-4567` (confidence: 85%)

**Overall PII Risk Score:** 0.35
```

### Violation Detection
```
4. Example: Checking for ethical violations...
# Violation Check Results

**Found 1 potential violation(s):**

1. **harmful_content**: Potentially harmful content detected
```

### Code Evaluation
```
5. Example: Evaluating safe code...
# Nethical Ethics Evaluation

**Decision:** ALLOW
**Agent ID:** demo_agent
**Timestamp:** 2025-11-21T16:11:43.691659+00:00

## Analysis Summary

## Audit Trail
Audit ID: simplified_demo_agent_1763741503.691662
```

### PII in Code
```
6. Example: Evaluating code with PII...
# Nethical Ethics Evaluation

**Decision:** ALLOW
**Agent ID:** demo_agent
**Timestamp:** 2025-11-21T16:11:43.693155+00:00

## Analysis Summary

**⚠️ PII Detected:** 2 instance(s)
  - email: admin@company.com
  - ssn: 123-45-6789

## Audit Trail
Audit ID: simplified_demo_agent_1763741503.693159
```

## Integration with GitHub Copilot

To use with GitHub Copilot:

1. Start the MCP server (see above)
2. Configure VS Code by creating `.vscode/mcp.json`:
   ```json
   {
     "servers": [
       {
         "name": "Nethical",
         "transport": "sse",
         "url": "http://localhost:8000/sse"
       }
     ]
   }
   ```
3. GitHub Copilot will automatically discover Nethical's tools
4. All code suggestions will be evaluated for ethics/safety before display

## Custom Client Implementation

The `NethicalMCPClient` class shows how to build your own client:

```python
from examples.mcp_client_example import NethicalMCPClient

# Create client
client = NethicalMCPClient()

# Initialize
client.initialize()

# Check for PII
result = client.check_pii("Email: test@example.com")
print(result)

# Evaluate code
result = client.evaluate_action(
    action="print('Hello')",
    agent_id="my_agent"
)
print(result)
```

## Dependencies

The example client requires:
```bash
pip install requests
```

The MCP server itself requires:
```bash
pip install fastapi uvicorn nethical
```

## Troubleshooting

### Server Not Running
```
Error: Could not connect to Nethical MCP server.
```

**Solution:** Start the server first:
```bash
python -m nethical.mcp_server
```

### Port Already in Use
```
ERROR: [Errno 98] Address already in use
```

**Solution:** Either:
- Stop the existing process: `pkill -f "nethical.mcp_server"`
- Use a different port: `uvicorn nethical.mcp_server:app --port 8080`

### Import Errors
```
ModuleNotFoundError: No module named 'nethical'
```

**Solution:** Install Nethical:
```bash
pip install -e .
```

## Learn More

- [Full MCP Documentation](../docs/MCP_SERVER.md)
- [Model Context Protocol Spec](https://spec.modelcontextprotocol.io/)
- [Nethical Documentation](../README.md)
