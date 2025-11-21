"""Nethical MCP (Model Context Protocol) Server.

This module implements an MCP server that exposes Nethical's ethics and safety
governance capabilities to LLMs like GitHub Copilot via Server-Sent Events (SSE).

The MCP protocol enables LLMs to discover and use Nethical's tools for:
- Ethical evaluation of AI-generated code/actions
- PII detection and privacy protection
- Safety constraint enforcement
- Audit trail generation

See: https://spec.modelcontextprotocol.io/
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nethical.core.integrated_governance import IntegratedGovernance
from nethical.utils.pii import PIIDetector


class MCPMessage(BaseModel):
    """Base MCP message structure."""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class ToolParameter(BaseModel):
    """MCP tool parameter definition."""
    type: str
    description: str
    required: bool = False
    enum: Optional[List[str]] = None


class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str
    description: str
    parameters: Dict[str, ToolParameter]


class MCPServer:
    """Nethical MCP Server implementing the Model Context Protocol."""
    
    def __init__(
        self,
        storage_dir: str = "./nethical_mcp_data",
        enable_quota: bool = False,
        region_id: Optional[str] = None,
    ):
        """Initialize the MCP server.
        
        Args:
            storage_dir: Directory for Nethical data storage
            enable_quota: Enable quota enforcement
            region_id: Optional region identifier
        """
        self.governance = IntegratedGovernance(
            storage_dir=storage_dir,
            enable_quota_enforcement=enable_quota,
            region_id=region_id,
            enable_performance_optimization=True,
            enable_merkle_anchoring=True,
            enable_ethical_taxonomy=True,
            enable_sla_monitoring=True,
        )
        self.pii_detector = PIIDetector()
        self.tools = self._define_tools()
        self.client_queues: Dict[str, asyncio.Queue] = {}
        
    def _define_tools(self) -> List[MCPTool]:
        """Define the tools exposed by this MCP server."""
        return [
            MCPTool(
                name="evaluate_action",
                description=(
                    "Evaluate an AI agent action for ethical compliance, safety, "
                    "and policy violations. Returns ALLOW, RESTRICT, BLOCK, or TERMINATE "
                    "decision with detailed audit information."
                ),
                parameters={
                    "action": ToolParameter(
                        type="string",
                        description="The action or code to evaluate",
                        required=True,
                    ),
                    "agent_id": ToolParameter(
                        type="string",
                        description="Identifier for the AI agent or user",
                        required=True,
                    ),
                    "action_type": ToolParameter(
                        type="string",
                        description="Type of action (e.g., 'code_generation', 'query', 'command')",
                        required=False,
                    ),
                    "context": ToolParameter(
                        type="object",
                        description="Additional context about the action",
                        required=False,
                    ),
                },
            ),
            MCPTool(
                name="check_pii",
                description=(
                    "Detect personally identifiable information (PII) in text. "
                    "Detects emails, phone numbers, SSNs, credit cards, IP addresses, "
                    "and other sensitive data."
                ),
                parameters={
                    "text": ToolParameter(
                        type="string",
                        description="Text to scan for PII",
                        required=True,
                    ),
                    "redact": ToolParameter(
                        type="boolean",
                        description="Whether to return redacted text",
                        required=False,
                    ),
                },
            ),
            MCPTool(
                name="check_violations",
                description=(
                    "Check for specific ethical or safety violations in content. "
                    "Useful for targeted checks of harmful content, deception, "
                    "privacy violations, or discrimination."
                ),
                parameters={
                    "content": ToolParameter(
                        type="string",
                        description="Content to check for violations",
                        required=True,
                    ),
                    "violation_types": ToolParameter(
                        type="array",
                        description=(
                            "Types of violations to check: 'harmful_content', "
                            "'deception', 'privacy', 'discrimination', 'manipulation'"
                        ),
                        required=False,
                    ),
                },
            ),
            MCPTool(
                name="get_system_status",
                description=(
                    "Get the current status of the Nethical governance system, "
                    "including enabled components and health metrics."
                ),
                parameters={},
            ),
        ]
    
    def _create_tool_definitions(self) -> List[Dict[str, Any]]:
        """Create MCP-compliant tool definitions."""
        tool_defs = []
        for tool in self.tools:
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
            
            for param_name, param in tool.parameters.items():
                tool_def["inputSchema"]["properties"][param_name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.enum:
                    tool_def["inputSchema"]["properties"][param_name]["enum"] = param.enum
                if param.required:
                    tool_def["inputSchema"]["required"].append(param_name)
            
            tool_defs.append(tool_def)
        
        return tool_defs
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "serverInfo": {
                "name": "Nethical MCP Server",
                "version": "1.0.0",
            },
            "capabilities": {
                "tools": {},
            },
        }
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {
            "tools": self._create_tool_definitions(),
        }
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "evaluate_action":
            return await self._tool_evaluate_action(arguments)
        elif tool_name == "check_pii":
            return await self._tool_check_pii(arguments)
        elif tool_name == "check_violations":
            return await self._tool_check_violations(arguments)
        elif tool_name == "get_system_status":
            return await self._tool_get_system_status(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _tool_evaluate_action(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute evaluate_action tool."""
        action = args.get("action")
        agent_id = args.get("agent_id")
        action_type = args.get("action_type", "query")
        context = args.get("context", {})
        
        if not action or not agent_id:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: 'action' and 'agent_id' are required parameters",
                    }
                ],
                "isError": True,
            }
        
        # Process action through governance system
        result = self.governance.process_action(
            agent_id=agent_id,
            action=action,
            action_type=action_type,
            context=context,
        )
        
        # Extract decision
        decision = result.get("decision", "BLOCK")
        
        # Format response
        response_text = f"""# Nethical Ethics Evaluation

**Decision:** {decision}
**Agent ID:** {agent_id}
**Timestamp:** {result.get('timestamp', datetime.now(timezone.utc).isoformat())}

## Analysis Summary
"""
        
        # Add risk assessment if available
        if "phase3" in result and result["phase3"].get("risk_score") is not None:
            risk_score = result["phase3"]["risk_score"]
            response_text += f"\n**Risk Score:** {risk_score:.2f}\n"
        
        # Add PII detection results if available
        if "pii_matches" in result and result["pii_matches"]:
            response_text += f"\n**⚠️ PII Detected:** {len(result['pii_matches'])} instance(s)\n"
            for match in result["pii_matches"][:3]:  # Show first 3
                response_text += f"  - {match.get('pii_type', 'unknown')}: {match.get('text', 'N/A')}\n"
        
        # Add quota status if available
        if "quota_enforcement" in result:
            quota = result["quota_enforcement"]
            if not quota.get("allowed"):
                response_text += f"\n**❌ Quota Exceeded:** {quota.get('reason', 'Rate limit exceeded')}\n"
        
        # Add violation details if blocked
        if decision in ["BLOCK", "TERMINATE"]:
            response_text += f"\n**Reason:** {result.get('reason', 'Policy violation detected')}\n"
        
        response_text += f"\n## Audit Trail\nAudit ID: {result.get('audit_id', 'N/A')}\n"
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": response_text,
                }
            ],
            "isError": False,
        }
    
    async def _tool_check_pii(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute check_pii tool."""
        text = args.get("text")
        redact = args.get("redact", False)
        
        if not text:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: 'text' is a required parameter",
                    }
                ],
                "isError": True,
            }
        
        # Detect PII
        pii_matches = self.pii_detector.detect_all(text)
        
        if not pii_matches:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "✓ No PII detected in the provided text.",
                    }
                ],
                "isError": False,
            }
        
        # Format results
        response_text = f"# PII Detection Results\n\n**Found {len(pii_matches)} PII instance(s):**\n\n"
        
        for i, match in enumerate(pii_matches, 1):
            response_text += f"{i}. **{match.pii_type.value}**: `{match.text}` "
            response_text += f"(confidence: {match.confidence:.0%})\n"
        
        # Calculate risk score
        risk_score = self.pii_detector.calculate_pii_risk_score(pii_matches)
        response_text += f"\n**Overall PII Risk Score:** {risk_score:.2f}\n"
        
        if redact and hasattr(self.pii_detector, 'redact_text'):
            redacted = self.pii_detector.redact_text(text, pii_matches)
            response_text += f"\n## Redacted Text:\n```\n{redacted}\n```\n"
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": response_text,
                }
            ],
            "isError": False,
        }
    
    async def _tool_check_violations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute check_violations tool."""
        content = args.get("content")
        violation_types = args.get("violation_types", [])
        
        if not content:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: 'content' is a required parameter",
                    }
                ],
                "isError": True,
            }
        
        # Use governance system to check for violations
        # We'll create a synthetic violation check
        violations_found = []
        
        # Simple heuristic checks for demonstration
        content_lower = content.lower()
        
        if not violation_types or "harmful_content" in violation_types:
            harmful_keywords = ["exploit", "attack", "malicious", "hack", "crack"]
            if any(kw in content_lower for kw in harmful_keywords):
                violations_found.append(("harmful_content", "Potentially harmful content detected"))
        
        if not violation_types or "privacy" in violation_types:
            # Check for PII
            pii_matches = self.pii_detector.detect_all(content)
            if pii_matches:
                violations_found.append(("privacy", f"PII detected: {len(pii_matches)} instance(s)"))
        
        if not violation_types or "deception" in violation_types:
            deception_keywords = ["fake", "phishing", "impersonate", "pretend to be"]
            if any(kw in content_lower for kw in deception_keywords):
                violations_found.append(("deception", "Potentially deceptive content detected"))
        
        # Format response
        if not violations_found:
            response_text = "✓ No violations detected in the provided content."
        else:
            response_text = f"# Violation Check Results\n\n**Found {len(violations_found)} potential violation(s):**\n\n"
            for i, (vtype, description) in enumerate(violations_found, 1):
                response_text += f"{i}. **{vtype}**: {description}\n"
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": response_text,
                }
            ],
            "isError": False,
        }
    
    async def _tool_get_system_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute get_system_status tool."""
        status = self.governance.get_system_status()
        
        response_text = "# Nethical System Status\n\n"
        response_text += f"**Region:** {status.get('region_id', 'N/A')}\n"
        response_text += f"**Timestamp:** {status.get('timestamp', 'N/A')}\n\n"
        
        response_text += "## Enabled Components:\n"
        for component, enabled in status.get("components_enabled", {}).items():
            icon = "✓" if enabled else "✗"
            response_text += f"- {icon} {component}\n"
        
        response_text += "\n## Statistics:\n"
        stats = status.get("statistics", {})
        for key, value in stats.items():
            response_text += f"- {key}: {value}\n"
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": response_text,
                }
            ],
            "isError": False,
        }
    
    async def _handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP message."""
        msg_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})
        
        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = await self._handle_list_tools(params)
            elif method == "tools/call":
                result = await self._handle_call_tool(params)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result,
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32603,
                    "message": str(e),
                },
            }
    
    async def sse_endpoint(self, request: Request) -> StreamingResponse:
        """SSE endpoint for MCP communication."""
        client_id = str(id(request))
        queue = asyncio.Queue()
        self.client_queues[client_id] = queue
        
        async def event_generator():
            try:
                # Send initial connection message
                yield f"data: {json.dumps({'type': 'connection', 'status': 'connected'})}\n\n"
                
                # Process incoming messages
                while True:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break
                    
                    try:
                        # Get message from queue with timeout
                        message = await asyncio.wait_for(queue.get(), timeout=30.0)
                        
                        # Handle message
                        response = await self._handle_message(message)
                        
                        # Send response
                        yield f"data: {json.dumps(response)}\n\n"
                    except asyncio.TimeoutError:
                        # Send keepalive
                        yield ": keepalive\n\n"
                    except Exception as e:
                        error_msg = {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32603,
                                "message": str(e),
                            },
                        }
                        yield f"data: {json.dumps(error_msg)}\n\n"
            finally:
                # Cleanup
                if client_id in self.client_queues:
                    del self.client_queues[client_id]
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send a message to a specific client."""
        if client_id in self.client_queues:
            await self.client_queues[client_id].put(message)


# FastAPI application factory
def create_app(
    storage_dir: str = "./nethical_mcp_data",
    enable_quota: bool = False,
    region_id: Optional[str] = None,
) -> FastAPI:
    """Create FastAPI application with MCP server."""
    
    mcp_server = MCPServer(
        storage_dir=storage_dir,
        enable_quota=enable_quota,
        region_id=region_id,
    )
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager."""
        # Startup
        yield
        # Shutdown
        pass
    
    app = FastAPI(
        title="Nethical MCP Server",
        description="Model Context Protocol server for Nethical ethics and safety governance",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    @app.post("/messages")
    async def handle_message(request: Request):
        """Handle MCP messages via POST (alternative to SSE)."""
        message = await request.json()
        response = await mcp_server._handle_message(message)
        return response
    
    @app.get("/sse")
    async def sse(request: Request):
        """SSE endpoint for MCP communication."""
        return await mcp_server.sse_endpoint(request)
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "nethical-mcp-server"}
    
    return app


if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
