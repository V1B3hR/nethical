"""
MCP Server for Nethical

FastAPI-based Model Context Protocol server providing:
- SSE endpoint at /sse for streaming MCP events
- Invocation endpoint /invoke for tool calls
- Health endpoint /health
"""

import asyncio
import json
from typing import Dict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from .models import (
    CallToolRequest,
    ListToolsRequest,
    MCPRequest,
    ToolListEvent,
    ToolResultEvent,
    ErrorEvent,
    AuditLogEvent,
    ToolResult,
)
from .audit import AuditLogger
from .tools import evaluate_code_tool, check_pii_tool


# Global state
subscribers: List[asyncio.Queue] = []
audit_logger = AuditLogger()

# Available tools registry
TOOLS = {
    "evaluate_code": evaluate_code_tool,
    "check_pii": check_pii_tool,
}


async def broadcast_event(event: dict):
    """Broadcast an event to all SSE subscribers."""
    dead_queues = []
    for queue in subscribers:
        try:
            await asyncio.wait_for(queue.put(event), timeout=1.0)
        except (asyncio.TimeoutError, asyncio.QueueFull):
            dead_queues.append(queue)
    
    # Remove dead queues
    for queue in dead_queues:
        if queue in subscribers:
            subscribers.remove(queue)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    # Startup
    print("MCP Server starting up...")
    yield
    # Shutdown
    print("MCP Server shutting down...")


app = FastAPI(
    title="Nethical MCP Server",
    description="Model Context Protocol server for Nethical ethics checking tools",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "tools_count": len(TOOLS),
        "subscribers_count": len(subscribers)
    }


@app.get("/sse")
async def sse_endpoint():
    """
    Server-Sent Events endpoint for streaming MCP events.
    
    Automatically sends tool_list event upon connection.
    """
    queue = asyncio.Queue(maxsize=100)
    subscribers.append(queue)
    
    async def event_generator():
        try:
            # Send tool_list event immediately upon connection
            tool_definitions = [tool["definition"] for tool in TOOLS.values()]
            tool_list_event = ToolListEvent(tools=tool_definitions)
            event_data = tool_list_event.model_dump_json()
            yield f"event: mcp\ndata: {event_data}\n\n"
            
            # Stream subsequent events
            while True:
                event = await queue.get()
                event_json = json.dumps(event)
                yield f"event: mcp\ndata: {event_json}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in subscribers:
                subscribers.remove(queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/invoke")
async def invoke(request: dict):
    """
    Invoke endpoint for MCP requests.
    
    Supports:
    - list_tools: Returns list of available tools
    - call_tool: Invokes a specific tool and returns results
    """
    try:
        request_type = request.get("type")
        
        if request_type == "list_tools":
            # Return list of tools
            tool_definitions = [tool["definition"].model_dump() for tool in TOOLS.values()]
            return {"tools": tool_definitions}
        
        elif request_type == "call_tool":
            # Parse call_tool request
            try:
                call_request = CallToolRequest(**request)
            except ValidationError as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            tool_name = call_request.tool
            arguments = call_request.arguments
            
            # Check if tool exists
            if tool_name not in TOOLS:
                error_msg = f"Unknown tool: {tool_name}"
                error_event = ErrorEvent(message=error_msg)
                await broadcast_event(error_event.model_dump())
                raise HTTPException(status_code=404, detail=error_msg)
            
            # Get the tool
            tool = TOOLS[tool_name]
            tool_function = tool["function"]
            
            # Extract the argument based on tool type
            if tool_name == "evaluate_code":
                if "code" not in arguments:
                    raise HTTPException(status_code=400, detail="Missing 'code' argument")
                arg_value = arguments["code"]
            elif tool_name == "check_pii":
                if "text" not in arguments:
                    raise HTTPException(status_code=400, detail="Missing 'text' argument")
                arg_value = arguments["text"]
            else:
                raise HTTPException(status_code=400, detail="Invalid tool arguments")
            
            # Execute the tool
            try:
                findings = tool_function(arg_value)
            except Exception as e:
                error_msg = f"Error executing tool: {str(e)}"
                error_event = ErrorEvent(message=error_msg, details=str(e))
                await broadcast_event(error_event.model_dump())
                raise HTTPException(status_code=500, detail=error_msg)
            
            # Determine status based on findings
            has_high_severity = any(f.severity == "HIGH" for f in findings)
            status = "BLOCK" if has_high_severity else "ALLOW"
            
            # Create summary
            if findings:
                severity_counts = {}
                for f in findings:
                    severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
                summary = f"Found {len(findings)} issue(s): " + ", ".join(
                    f"{count} {sev}" for sev, count in severity_counts.items()
                )
            else:
                summary = "No issues found"
            
            # Log to audit
            audit_id = audit_logger.log_tool_invocation(
                tool=tool_name,
                arguments=arguments,
                findings=findings,
                status=status,
                summary=summary
            )
            
            # Create result
            result = ToolResult(
                tool=tool_name,
                status=status,
                findings=findings,
                summary=summary,
                audit_id=audit_id
            )
            
            # Broadcast tool_result event
            tool_result_event = ToolResultEvent(result=result)
            await broadcast_event(tool_result_event.model_dump())
            
            # Broadcast audit_log event
            audit_log_event = AuditLogEvent(
                audit_id=audit_id,
                tool=tool_name,
                status=status,
                findings_count=len(findings),
                summary=summary
            )
            await broadcast_event(audit_log_event.model_dump())
            
            # Return result
            return result.model_dump()
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown request type: {request_type}")
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        error_event = ErrorEvent(message=error_msg, details=str(e))
        await broadcast_event(error_event.model_dump())
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
