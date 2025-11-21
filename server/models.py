"""
MCP Server Models

Pydantic models for Model Context Protocol (MCP) message structures.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""
    name: str
    type: str
    description: str
    required: bool = True


class ToolDefinition(BaseModel):
    """Definition of an MCP tool."""
    name: str
    description: str
    parameters: List[ToolParameter]


class Finding(BaseModel):
    """Represents a single finding from a tool."""
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    category: str
    message: str
    line: Optional[int] = None
    code_snippet: Optional[str] = None


class ToolResult(BaseModel):
    """Result from a tool invocation."""
    tool: str
    status: Literal["ALLOW", "BLOCK"]
    findings: List[Finding]
    summary: str
    audit_id: str


class MCPRequest(BaseModel):
    """Base MCP request."""
    type: Literal["list_tools", "call_tool"]


class ListToolsRequest(MCPRequest):
    """Request to list available tools."""
    type: Literal["list_tools"] = "list_tools"


class CallToolRequest(MCPRequest):
    """Request to call a specific tool."""
    type: Literal["call_tool"] = "call_tool"
    tool: str
    arguments: Dict[str, Any]


class MCPEvent(BaseModel):
    """Base class for MCP events sent over SSE."""
    event_type: Literal["tool_list", "tool_result", "error", "audit_log"]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ToolListEvent(MCPEvent):
    """Event containing list of available tools."""
    event_type: Literal["tool_list"] = "tool_list"
    tools: List[ToolDefinition]


class ToolResultEvent(MCPEvent):
    """Event containing result of tool invocation."""
    event_type: Literal["tool_result"] = "tool_result"
    result: ToolResult


class ErrorEvent(MCPEvent):
    """Event containing error information."""
    event_type: Literal["error"] = "error"
    message: str
    details: Optional[str] = None


class AuditLogEvent(MCPEvent):
    """Event containing audit log information."""
    event_type: Literal["audit_log"] = "audit_log"
    audit_id: str
    tool: str
    status: str
    findings_count: int
    summary: str
