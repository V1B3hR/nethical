#!/usr/bin/env python3
"""Example MCP client demonstrating how to interact with Nethical MCP Server.

This script shows how to:
1. Connect to the MCP server
2. Initialize the connection
3. List available tools
4. Call tools to evaluate code and check for violations

Prerequisites:
    - Nethical MCP server running on http://localhost:8000
    - requests library: pip install requests
"""

import json
import requests
from typing import Dict, Any


class NethicalMCPClient:
    """Simple MCP client for Nethical server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the MCP client.
        
        Args:
            base_url: Base URL of the Nethical MCP server
        """
        self.base_url = base_url
        self.messages_url = f"{base_url}/messages"
        self.message_id = 0
    
    def _send_message(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a JSON-RPC message to the server.
        
        Args:
            method: The method to call
            params: Parameters for the method
            
        Returns:
            Response from the server
        """
        self.message_id += 1
        message = {
            "jsonrpc": "2.0",
            "id": self.message_id,
            "method": method,
            "params": params or {}
        }
        
        response = requests.post(
            self.messages_url,
            json=message,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP connection."""
        return self._send_message("initialize")
    
    def list_tools(self) -> Dict[str, Any]:
        """List available tools."""
        return self._send_message("tools/list")
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        return self._send_message("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
    
    def evaluate_action(
        self,
        action: str,
        agent_id: str,
        action_type: str = "code_generation",
        context: Dict[str, Any] = None
    ) -> str:
        """Evaluate an action for ethical compliance.
        
        Args:
            action: The action or code to evaluate
            agent_id: Identifier for the AI agent or user
            action_type: Type of action
            context: Additional context
            
        Returns:
            Formatted evaluation result
        """
        result = self.call_tool("evaluate_action", {
            "action": action,
            "agent_id": agent_id,
            "action_type": action_type,
            "context": context or {}
        })
        
        if "result" in result:
            return result["result"]["content"][0]["text"]
        else:
            return f"Error: {result.get('error', {}).get('message', 'Unknown error')}"
    
    def check_pii(self, text: str, redact: bool = False) -> str:
        """Check for PII in text.
        
        Args:
            text: Text to scan
            redact: Whether to return redacted text
            
        Returns:
            PII detection result
        """
        result = self.call_tool("check_pii", {
            "text": text,
            "redact": redact
        })
        
        if "result" in result:
            return result["result"]["content"][0]["text"]
        else:
            return f"Error: {result.get('error', {}).get('message', 'Unknown error')}"
    
    def check_violations(
        self,
        content: str,
        violation_types: list = None
    ) -> str:
        """Check for ethical violations.
        
        Args:
            content: Content to check
            violation_types: List of violation types to check
            
        Returns:
            Violation check result
        """
        result = self.call_tool("check_violations", {
            "content": content,
            "violation_types": violation_types or []
        })
        
        if "result" in result:
            return result["result"]["content"][0]["text"]
        else:
            return f"Error: {result.get('error', {}).get('message', 'Unknown error')}"
    
    def get_system_status(self) -> str:
        """Get system status.
        
        Returns:
            System status information
        """
        result = self.call_tool("get_system_status", {})
        
        if "result" in result:
            return result["result"]["content"][0]["text"]
        else:
            return f"Error: {result.get('error', {}).get('message', 'Unknown error')}"


def main():
    """Demonstrate using the Nethical MCP client."""
    
    print("=" * 70)
    print("Nethical MCP Client Example")
    print("=" * 70)
    print()
    
    # Create client
    client = NethicalMCPClient()
    
    # 1. Initialize
    print("1. Initializing connection...")
    init_result = client.initialize()
    print(f"   Server: {init_result['result']['serverInfo']['name']}")
    print(f"   Version: {init_result['result']['serverInfo']['version']}")
    print()
    
    # 2. List tools
    print("2. Listing available tools...")
    tools = client.list_tools()
    for tool in tools["result"]["tools"]:
        print(f"   - {tool['name']}: {tool['description'][:60]}...")
    print()
    
    # 3. Example: Check for PII
    print("3. Example: Checking for PII in text...")
    pii_text = "My email is john.doe@example.com and phone is 555-123-4567"
    pii_result = client.check_pii(pii_text)
    print(pii_result)
    print()
    
    # 4. Example: Check for violations
    print("4. Example: Checking for ethical violations...")
    harmful_code = "Here's how to exploit the system and hack into databases"
    violation_result = client.check_violations(harmful_code)
    print(violation_result)
    print()
    
    # 5. Example: Evaluate safe code
    print("5. Example: Evaluating safe code...")
    safe_code = '''
def greet(name):
    """Greet a user."""
    return f"Hello, {name}!"
'''
    eval_result = client.evaluate_action(
        action=safe_code,
        agent_id="demo_agent",
        action_type="code_generation"
    )
    print(eval_result)
    print()
    
    # 6. Example: Evaluate code with PII
    print("6. Example: Evaluating code with PII...")
    pii_code = '''
# Store user credentials
email = "admin@company.com"
password = "secret123"
ssn = "123-45-6789"
'''
    eval_result = client.evaluate_action(
        action=pii_code,
        agent_id="demo_agent",
        action_type="code_generation"
    )
    print(eval_result)
    print()
    
    # 7. Example: Get system status
    print("7. Example: Getting system status...")
    status = client.get_system_status()
    print(status[:500] + "...\n")
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Nethical MCP server.")
        print("Please ensure the server is running on http://localhost:8000")
        print("\nStart the server with:")
        print("  python -m nethical.mcp_server")
    except Exception as e:
        print(f"Error: {e}")
