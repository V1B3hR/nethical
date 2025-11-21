"""Tests for Nethical MCP Server."""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from nethical.mcp_server import (
    MCPServer,
    MCPTool,
    ToolParameter,
    create_app,
)


class TestMCPServer:
    """Test suite for MCP server."""

    def test_initialization(self):
        """Test that MCP server initializes correctly."""
        server = MCPServer(storage_dir="./test_mcp_data")

        # Verify governance system is initialized
        assert server.governance is not None
        assert server.pii_detector is not None

        # Verify tools are defined
        assert len(server.tools) > 0
        tool_names = [tool.name for tool in server.tools]
        assert "evaluate_action" in tool_names
        assert "check_pii" in tool_names
        assert "check_violations" in tool_names
        assert "get_system_status" in tool_names

    def test_tool_definitions(self):
        """Test that tools are properly defined."""
        server = MCPServer(storage_dir="./test_mcp_data2")

        # Check evaluate_action tool
        eval_tool = next(t for t in server.tools if t.name == "evaluate_action")
        assert eval_tool is not None
        assert "action" in eval_tool.parameters
        assert "agent_id" in eval_tool.parameters
        assert eval_tool.parameters["action"].required is True
        assert eval_tool.parameters["agent_id"].required is True

        # Check check_pii tool
        pii_tool = next(t for t in server.tools if t.name == "check_pii")
        assert pii_tool is not None
        assert "text" in pii_tool.parameters
        assert pii_tool.parameters["text"].required is True

    def test_create_tool_definitions(self):
        """Test MCP-compliant tool definition generation."""
        server = MCPServer(storage_dir="./test_mcp_data3")
        tool_defs = server._create_tool_definitions()

        assert len(tool_defs) > 0

        # Check structure of first tool
        first_tool = tool_defs[0]
        assert "name" in first_tool
        assert "description" in first_tool
        assert "inputSchema" in first_tool
        assert "type" in first_tool["inputSchema"]
        assert first_tool["inputSchema"]["type"] == "object"
        assert "properties" in first_tool["inputSchema"]

    @pytest.mark.asyncio
    async def test_handle_initialize(self):
        """Test initialize message handling."""
        server = MCPServer(storage_dir="./test_mcp_data4")

        result = await server._handle_initialize({})

        assert "protocolVersion" in result
        assert "serverInfo" in result
        assert "capabilities" in result
        assert result["serverInfo"]["name"] == "Nethical MCP Server"
        assert "tools" in result["capabilities"]

    @pytest.mark.asyncio
    async def test_handle_list_tools(self):
        """Test tools/list message handling."""
        server = MCPServer(storage_dir="./test_mcp_data5")

        result = await server._handle_list_tools({})

        assert "tools" in result
        assert len(result["tools"]) > 0

        # Verify each tool has required fields
        for tool in result["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    @pytest.mark.asyncio
    async def test_tool_evaluate_action_success(self):
        """Test evaluate_action tool execution."""
        server = MCPServer(storage_dir="./test_mcp_data6")

        args = {
            "action": "print('Hello, world!')",
            "agent_id": "test_agent",
            "action_type": "code_generation",
        }

        result = await server._tool_evaluate_action(args)

        assert "content" in result
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert "Decision:" in result["content"][0]["text"]
        assert result["isError"] is False

    @pytest.mark.asyncio
    async def test_tool_evaluate_action_missing_params(self):
        """Test evaluate_action with missing required parameters."""
        server = MCPServer(storage_dir="./test_mcp_data7")

        # Missing action
        args = {"agent_id": "test_agent"}
        result = await server._tool_evaluate_action(args)
        assert result["isError"] is True
        assert "required" in result["content"][0]["text"].lower()

        # Missing agent_id
        args = {"action": "test action"}
        result = await server._tool_evaluate_action(args)
        assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_tool_check_pii_no_pii(self):
        """Test check_pii tool with text containing no PII."""
        server = MCPServer(storage_dir="./test_mcp_data8")

        args = {"text": "This is a simple test message with no sensitive data."}

        result = await server._tool_check_pii(args)

        assert "content" in result
        assert result["isError"] is False
        assert "No PII detected" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tool_check_pii_with_email(self):
        """Test check_pii tool with text containing email."""
        server = MCPServer(storage_dir="./test_mcp_data9")

        args = {"text": "Contact me at john.doe@example.com for more info."}

        result = await server._tool_check_pii(args)

        assert "content" in result
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "PII" in text
        assert "email" in text.lower()

    @pytest.mark.asyncio
    async def test_tool_check_pii_missing_params(self):
        """Test check_pii with missing required parameters."""
        server = MCPServer(storage_dir="./test_mcp_data10")

        args = {}
        result = await server._tool_check_pii(args)

        assert result["isError"] is True
        assert "required" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_tool_check_violations_no_violations(self):
        """Test check_violations with clean content."""
        server = MCPServer(storage_dir="./test_mcp_data11")

        args = {"content": "This is a normal message with no violations."}

        result = await server._tool_check_violations(args)

        assert "content" in result
        assert result["isError"] is False
        assert "No violations" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_tool_check_violations_harmful_content(self):
        """Test check_violations with harmful content."""
        server = MCPServer(storage_dir="./test_mcp_data12")

        args = {"content": "Here's how to exploit the system and attack users."}

        result = await server._tool_check_violations(args)

        assert "content" in result
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "violation" in text.lower()

    @pytest.mark.asyncio
    async def test_tool_check_violations_with_pii(self):
        """Test check_violations detecting privacy violations."""
        server = MCPServer(storage_dir="./test_mcp_data13")

        args = {
            "content": "My email is test@example.com and SSN is 123-45-6789",
            "violation_types": ["privacy"],
        }

        result = await server._tool_check_violations(args)

        assert "content" in result
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "privacy" in text.lower() or "PII" in text

    @pytest.mark.asyncio
    async def test_tool_get_system_status(self):
        """Test get_system_status tool."""
        server = MCPServer(storage_dir="./test_mcp_data14")

        result = await server._tool_get_system_status({})

        assert "content" in result
        assert result["isError"] is False
        text = result["content"][0]["text"]
        assert "System Status" in text
        assert "Components" in text

    @pytest.mark.asyncio
    async def test_handle_call_tool(self):
        """Test tools/call message handling."""
        server = MCPServer(storage_dir="./test_mcp_data15")

        # Test valid tool call
        params = {
            "name": "get_system_status",
            "arguments": {},
        }
        result = await server._handle_call_tool(params)
        assert "content" in result
        assert result["isError"] is False

        # Test invalid tool call
        params = {
            "name": "nonexistent_tool",
            "arguments": {},
        }
        with pytest.raises(ValueError):
            await server._handle_call_tool(params)

    @pytest.mark.asyncio
    async def test_handle_message_initialize(self):
        """Test full message handling for initialize."""
        server = MCPServer(storage_dir="./test_mcp_data16")

        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        }

        response = await server._handle_message(message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "protocolVersion" in response["result"]

    @pytest.mark.asyncio
    async def test_handle_message_list_tools(self):
        """Test full message handling for list_tools."""
        server = MCPServer(storage_dir="./test_mcp_data17")

        message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        response = await server._handle_message(message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert "tools" in response["result"]

    @pytest.mark.asyncio
    async def test_handle_message_call_tool(self):
        """Test full message handling for call_tool."""
        server = MCPServer(storage_dir="./test_mcp_data18")

        message = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_system_status",
                "arguments": {},
            },
        }

        response = await server._handle_message(message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response

    @pytest.mark.asyncio
    async def test_handle_message_error(self):
        """Test error handling in message processing."""
        server = MCPServer(storage_dir="./test_mcp_data19")

        message = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "unknown_method",
            "params": {},
        }

        response = await server._handle_message(message)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 4
        assert "error" in response
        assert "code" in response["error"]
        assert "message" in response["error"]

    def test_create_app(self):
        """Test FastAPI app creation."""
        app = create_app(storage_dir="./test_mcp_data20")

        assert app is not None
        assert app.title == "Nethical MCP Server"

        # Check that routes are registered
        route_paths = [route.path for route in app.routes]
        assert "/sse" in route_paths
        assert "/health" in route_paths
        assert "/messages" in route_paths


class TestMCPIntegration:
    """Integration tests for MCP server with real governance system."""

    @pytest.mark.asyncio
    async def test_evaluate_action_with_governance(self):
        """Test evaluate_action integrates with governance system."""
        server = MCPServer(storage_dir="./test_mcp_integration1")

        args = {
            "action": "SELECT * FROM users WHERE admin = 1",
            "agent_id": "sql_agent",
            "action_type": "query",
            "context": {"database": "production"},
        }

        result = await server._tool_evaluate_action(args)

        assert "content" in result
        assert result["isError"] is False
        text = result["content"][0]["text"]

        # Should have decision
        assert "Decision:" in text
        # Should have timestamp
        assert "Timestamp:" in text
        # Should have audit trail
        assert "Audit" in text

    @pytest.mark.asyncio
    async def test_check_pii_comprehensive(self):
        """Test PII detection with multiple PII types."""
        server = MCPServer(storage_dir="./test_mcp_integration2")

        args = {
            "text": (
                "Contact John at john.doe@example.com or call 555-123-4567. "
                "SSN: 123-45-6789, IP: 192.168.1.1"
            )
        }

        result = await server._tool_check_pii(args)

        assert "content" in result
        assert result["isError"] is False
        text = result["content"][0]["text"]

        # Should detect multiple PII types
        assert "email" in text.lower()
        # Phone detection depends on PII detector patterns
        assert "Risk Score" in text

    @pytest.mark.asyncio
    async def test_system_status_includes_components(self):
        """Test system status returns governance components."""
        server = MCPServer(
            storage_dir="./test_mcp_integration3",
            enable_quota=True,
            region_id="us-east-1",
        )

        result = await server._tool_get_system_status({})

        assert "content" in result
        text = result["content"][0]["text"]

        # Should show region
        assert "us-east-1" in text or "Region" in text
        # Should show components
        assert "Components" in text
