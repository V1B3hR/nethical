# Nethical MCP Server

The Nethical MCP (Model Context Protocol) Server enables AI tools like GitHub Copilot to integrate with Nethical's ethics and safety governance system. This allows real-time evaluation of AI-generated code and content for ethical compliance, safety violations, and privacy concerns.

## What is MCP?

Model Context Protocol (MCP) is an open standard for connecting Large Language Models (LLMs) to external tools and data sources. It uses Server-Sent Events (SSE) for bi-directional communication, allowing LLMs to discover and invoke tools dynamically.

Learn more: https://spec.modelcontextprotocol.io/

## Features

The Nethical MCP Server exposes the following capabilities:

### 1. **evaluate_action** - Comprehensive Ethics Evaluation
Evaluates actions/code for ethical compliance, safety, and policy violations.

**Parameters:**
- `action` (required): The action or code to evaluate
- `agent_id` (required): Identifier for the AI agent or user
- `action_type` (optional): Type of action (e.g., 'code_generation', 'query')
- `context` (optional): Additional context as a JSON object

**Returns:**
- Decision: ALLOW, RESTRICT, BLOCK, or TERMINATE
- Risk score and analysis
- PII detection results
- Quota status
- Audit trail ID

### 2. **check_pii** - PII Detection
Detects personally identifiable information in text.

**Parameters:**
- `text` (required): Text to scan for PII
- `redact` (optional): Whether to return redacted text

**Returns:**
- List of detected PII (email, phone, SSN, credit cards, IPs, etc.)
- Confidence scores for each match
- Overall PII risk score
- Optional redacted text

### 3. **check_violations** - Targeted Violation Checks
Checks for specific ethical or safety violations.

**Parameters:**
- `content` (required): Content to check
- `violation_types` (optional): Array of violation types to check:
  - `harmful_content`
  - `deception`
  - `privacy`
  - `discrimination`
  - `manipulation`

**Returns:**
- List of detected violations
- Descriptions and severity

### 4. **get_system_status** - System Health
Get current status of the Nethical governance system.

**Returns:**
- Region and configuration
- Enabled components
- System statistics

## Installation & Setup

### 1. Install Dependencies

```bash
pip install nethical fastapi uvicorn
```

### 2. Start the Server

**From Python:**
```python
from nethical.mcp_server import create_app
import uvicorn

app = create_app(
    storage_dir="./nethical_mcp_data",
    enable_quota=False,
    region_id="us-east-1"
)

uvicorn.run(app, host="0.0.0.0", port=8000)
```

**From Command Line:**
```bash
cd /path/to/nethical
python -m nethical.mcp_server
```

The server will start on `http://localhost:8000`.

### 3. Configure VS Code / GitHub Copilot

Create or update `.vscode/mcp.json` in your workspace:

```json
{
  "servers": [
    {
      "name": "Nethical",
      "description": "Ethics and safety governance for AI code generation",
      "transport": "sse",
      "url": "http://localhost:8000/sse",
      "capabilities": ["tools"]
    }
  ]
}
```

GitHub Copilot and other MCP-compatible tools will automatically discover and use Nethical's tools.

## Usage Examples

### Example 1: Code Generation with Ethics Check

When using GitHub Copilot to generate code:

1. Copilot generates code suggestion
2. Before showing/applying, Copilot calls `evaluate_action` tool
3. Nethical analyzes the code for:
   - PII leakage
   - Security vulnerabilities
   - Ethical violations
   - Policy compliance
4. Returns decision (ALLOW/BLOCK) with reasoning
5. Copilot applies code only if ALLOWED
6. All interactions are logged in audit trail

### Example 2: PII Detection in Generated Code

```python
# If Copilot generates:
email = "john.doe@company.com"
api_key = "sk-1234567890abcdef"

# Nethical's check_pii tool will detect:
# - Email address (PII)
# - Potential API key (sensitive)
# - Risk score: HIGH
# - Decision: BLOCK or RESTRICT with warning
```

### Example 3: Harmful Content Detection

```python
# If Copilot generates code like:
# "Execute SQL injection attack..."

# Nethical's check_violations tool detects:
# - Violation type: harmful_content
# - Description: Potentially malicious code
# - Decision: BLOCK
```

## API Endpoints

### SSE Endpoint (Primary)
- **URL:** `GET /sse`
- **Protocol:** Server-Sent Events
- **Usage:** MCP clients connect here for tool discovery and execution

### Health Check
- **URL:** `GET /health`
- **Returns:** `{"status": "healthy", "service": "nethical-mcp-server"}`

### Message Endpoint (Alternative)
- **URL:** `POST /messages`
- **Body:** JSON-RPC 2.0 message
- **Usage:** Alternative to SSE for simple request/response

## Configuration Options

When creating the app, you can configure:

```python
app = create_app(
    storage_dir="./nethical_mcp_data",  # Data storage directory
    enable_quota=True,                   # Enable rate limiting
    region_id="eu-west-1"                # Geographic region
)
```

### Environment Variables

You can also set environment-specific configuration:

```bash
export NETHICAL_STORAGE_DIR=/var/lib/nethical/mcp
export NETHICAL_ENABLE_QUOTA=true
export NETHICAL_REGION_ID=us-east-1
```

## MCP Protocol Details

### Message Format

All messages follow JSON-RPC 2.0 format:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "evaluate_action",
    "arguments": {
      "action": "print('Hello')",
      "agent_id": "copilot"
    }
  }
}
```

### Response Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "# Nethical Ethics Evaluation\n\n**Decision:** ALLOW\n..."
      }
    ],
    "isError": false
  }
}
```

## Integration with GitHub Copilot

### Automatic Tool Discovery

1. Start Nethical MCP server
2. Configure `.vscode/mcp.json`
3. GitHub Copilot automatically:
   - Discovers available tools
   - Routes suggestions through Nethical
   - Applies ethics gating before showing code

### Benefits

- **Real-time Ethics:** Every code suggestion is evaluated
- **Privacy Protection:** Automatic PII detection
- **Audit Trail:** All interactions logged
- **Configurable Policies:** Customize what's allowed
- **Zero Manual Overhead:** Fully automated

## Monitoring & Audit

### Audit Logs

All evaluations are logged with:
- Timestamp
- Agent ID
- Action content
- Decision and reasoning
- Risk scores
- PII detections

Logs are stored in `{storage_dir}/nethical_data/`.

### System Status

Check system health:
```bash
curl http://localhost:8000/health
```

Get detailed status via MCP:
```json
{
  "method": "tools/call",
  "params": {
    "name": "get_system_status"
  }
}
```

## Troubleshooting

### Server Won't Start

**Error:** Port already in use
```bash
# Check what's using port 8000
lsof -i :8000

# Or use a different port
uvicorn nethical.mcp_server:app --port 8080
```

### Copilot Not Discovering Tools

1. Verify server is running: `curl http://localhost:8000/health`
2. Check `.vscode/mcp.json` syntax
3. Restart VS Code
4. Check VS Code Developer Console for errors

### PII Detection Too Strict

Adjust detection thresholds in server initialization:

```python
from nethical.mcp_server import MCPServer

server = MCPServer(storage_dir="./data")
# Adjust PII detector settings
server.pii_detector.false_positive_hints[PIIType.EMAIL].append("mycompany.com")
```

## Security Considerations

1. **Network Security:** Run MCP server on localhost or secured network
2. **Authentication:** Add authentication layer for production use
3. **Data Privacy:** Audit logs may contain sensitive data - secure appropriately
4. **Rate Limiting:** Enable quota enforcement for public deployments
5. **HTTPS:** Use HTTPS in production with proper certificates

## Performance

- **Latency:** Typical evaluation takes 10-50ms
- **Throughput:** Handles 100+ requests/second
- **Resource Usage:** ~100MB RAM base, scales with audit log size
- **Caching:** Integrated governance system uses caching for repeated evaluations

## Advanced Usage

### Custom Tool Configuration

Extend the MCP server with custom tools:

```python
from nethical.mcp_server import MCPServer, MCPTool, ToolParameter

server = MCPServer()

# Add custom tool
custom_tool = MCPTool(
    name="check_license_compliance",
    description="Check if code complies with license requirements",
    parameters={
        "code": ToolParameter(type="string", required=True),
        "license_type": ToolParameter(type="string", required=False)
    }
)
server.tools.append(custom_tool)
```

### Multi-Region Deployment

Deploy servers in multiple regions for data residency:

```python
# US Server
us_app = create_app(region_id="us-east-1", storage_dir="./us-data")

# EU Server  
eu_app = create_app(region_id="eu-west-1", storage_dir="./eu-data")
```

### Integration with CI/CD

Use MCP server in CI/CD pipelines:

```yaml
# .github/workflows/ethics-check.yml
steps:
  - name: Start Nethical MCP Server
    run: |
      python -m nethical.mcp_server &
      sleep 5
  
  - name: Check Generated Code
    run: |
      curl -X POST http://localhost:8000/messages \
        -H "Content-Type: application/json" \
        -d '{"method":"tools/call","params":{"name":"evaluate_action","arguments":{"action":"...","agent_id":"ci-bot"}}}'
```

## Contributing

To contribute to the MCP server:

1. Follow Nethical contribution guidelines
2. Add tests for new tools
3. Update documentation
4. Ensure backwards compatibility with MCP spec

## References

- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [Nethical Documentation](../README.md)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Server-Sent Events Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)

## License

Same as Nethical - see LICENSE file.
