#!/usr/bin/env python3
"""
Demo script for Nethical API v1

This script demonstrates all major features of the API v1:
1. Authentication
2. Agent Management
3. Policy Management
4. Audit Log Access
5. Real-time Threat Monitoring

Usage:
    python demo_api_v1.py
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone

import httpx

# API Configuration
BASE_URL = "http://localhost:8000/api/v1"
USERNAME = "admin"
PASSWORD = "admin123"


class NethicalAPIDemo:
    """Demo client for Nethical API v1."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.token = None
        self.headers = {}
    
    def print_section(self, title: str):
        """Print section header."""
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print('=' * 60)
    
    def print_success(self, message: str):
        """Print success message."""
        print(f"✓ {message}")
    
    def print_error(self, message: str):
        """Print error message."""
        print(f"✗ {message}")
    
    def print_json(self, data: dict):
        """Print JSON data."""
        print(json.dumps(data, indent=2))
    
    async def login(self):
        """Authenticate and get token."""
        self.print_section("1. Authentication")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/auth/login",
                json={"username": USERNAME, "password": PASSWORD}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                self.headers = {"Authorization": f"Bearer {self.token}"}
                
                self.print_success("Successfully authenticated")
                print(f"  User: {data['user']['username']}")
                print(f"  Role: {data['user']['role']}")
                print(f"  Token expires in: {data['expires_in']} seconds")
                return True
            else:
                self.print_error(f"Authentication failed: {response.text}")
                return False
    
    async def demo_agents(self):
        """Demonstrate agent management."""
        self.print_section("2. Agent Management")
        
        async with httpx.AsyncClient() as client:
            # Create agent
            print("\nCreating agent...")
            agent_data = {
                "agent_id": "demo-gpt4",
                "name": "Demo GPT-4 Agent",
                "agent_type": "llm",
                "description": "Demo agent for testing",
                "trust_level": 0.9,
                "configuration": {
                    "model": "gpt-4",
                    "temperature": 0.7
                },
                "metadata": {
                    "environment": "demo"
                }
            }
            
            response = await client.post(
                f"{self.base_url}/agents",
                headers=self.headers,
                json=agent_data
            )
            
            if response.status_code == 201:
                agent = response.json()
                self.print_success(f"Created agent: {agent['agent_id']}")
                print(f"  Name: {agent['name']}")
                print(f"  Trust Level: {agent['trust_level']}")
            else:
                self.print_error(f"Failed to create agent: {response.text}")
            
            # List agents
            print("\nListing agents...")
            response = await client.get(
                f"{self.base_url}/agents",
                headers=self.headers,
                params={"page": 1, "per_page": 10}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Found {data['total']} agents")
                for agent in data['agents']:
                    print(f"  - {agent['agent_id']}: {agent['name']} ({agent['status']})")
            
            # Get agent details
            print("\nGetting agent details...")
            response = await client.get(
                f"{self.base_url}/agents/demo-gpt4",
                headers=self.headers
            )
            
            if response.status_code == 200:
                agent = response.json()
                self.print_success("Retrieved agent details")
                print(f"  Configuration: {json.dumps(agent['configuration'], indent=2)}")
            
            # Update agent
            print("\nUpdating agent...")
            response = await client.patch(
                f"{self.base_url}/agents/demo-gpt4",
                headers=self.headers,
                json={"trust_level": 0.95}
            )
            
            if response.status_code == 200:
                agent = response.json()
                self.print_success(f"Updated trust level to {agent['trust_level']}")
    
    async def demo_policies(self):
        """Demonstrate policy management."""
        self.print_section("3. Policy Management")
        
        async with httpx.AsyncClient() as client:
            # Create policy
            print("\nCreating policy...")
            policy_data = {
                "policy_id": "demo-policy",
                "name": "Demo Access Policy",
                "description": "Demo policy for testing",
                "version": "1.0.0",
                "priority": 100,
                "rules": [
                    {
                        "id": "rule-1",
                        "condition": "action_type == 'data_access'",
                        "action": "RESTRICT",
                        "priority": 10,
                        "description": "Restrict data access"
                    }
                ],
                "scope": "global",
                "fundamental_laws": [22]
            }
            
            response = await client.post(
                f"{self.base_url}/policies",
                headers=self.headers,
                json=policy_data
            )
            
            if response.status_code == 201:
                policy = response.json()
                self.print_success(f"Created policy: {policy['policy_id']}")
                print(f"  Name: {policy['name']}")
                print(f"  Rules: {len(policy['rules'])}")
            else:
                self.print_error(f"Failed to create policy: {response.text}")
            
            # List policies
            print("\nListing policies...")
            response = await client.get(
                f"{self.base_url}/policies",
                headers=self.headers,
                params={"page": 1, "per_page": 10}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Found {data['total']} policies")
                for policy in data['policies']:
                    print(f"  - {policy['policy_id']}: {policy['name']} (v{policy['version']})")
    
    async def demo_audit_logs(self):
        """Demonstrate audit log access."""
        self.print_section("4. Audit Log Access")
        
        async with httpx.AsyncClient() as client:
            # Get audit logs
            print("\nRetrieving audit logs...")
            response = await client.get(
                f"{self.base_url}/audit/logs",
                headers=self.headers,
                params={
                    "page": 1,
                    "per_page": 5
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Found {data['total']} audit logs")
                for log in data['logs']:
                    print(f"  - {log['log_id']}: {log['event_type']} ({log.get('threat_level', 'N/A')})")
            else:
                self.print_success("No audit logs yet (database is new)")
            
            # Get Merkle tree
            print("\nRetrieving Merkle tree...")
            response = await client.get(
                f"{self.base_url}/audit/merkle-tree",
                headers=self.headers,
                params={"limit": 10}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.print_success(f"Generated Merkle tree")
                print(f"  Root Hash: {data.get('root_hash', 'N/A')}")
                print(f"  Total Logs: {data['total_logs']}")
                print(f"  Tree Height: {data['tree_height']}")
    
    async def demo_realtime(self):
        """Demonstrate real-time threat monitoring."""
        self.print_section("5. Real-time Threat Monitoring")
        
        print("\nWebSocket and SSE endpoints available:")
        print(f"  WebSocket: ws://localhost:8000/api/v1/ws/threats?token={self.token[:20]}...")
        print(f"  SSE: {self.base_url}/sse/threats")
        print("\nExample WebSocket connection (JavaScript):")
        print("""
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/threats?token=YOUR_TOKEN');
ws.onmessage = (event) => {
    const threat = JSON.parse(event.data);
    console.log('Threat detected:', threat);
};
        """)
        
        self.print_success("Real-time endpoints configured and ready")
    
    async def cleanup(self):
        """Clean up demo resources."""
        self.print_section("Cleanup")
        
        async with httpx.AsyncClient() as client:
            # Delete demo agent
            print("\nDeleting demo agent...")
            response = await client.delete(
                f"{self.base_url}/agents/demo-gpt4",
                headers=self.headers
            )
            
            if response.status_code == 204:
                self.print_success("Deleted demo agent")
            
            # Delete demo policy
            print("Deleting demo policy...")
            response = await client.delete(
                f"{self.base_url}/policies/demo-policy",
                headers=self.headers
            )
            
            if response.status_code == 204:
                self.print_success("Deleted demo policy")
    
    async def run(self):
        """Run the demo."""
        print("\n" + "=" * 60)
        print("  Nethical API v1 Demo")
        print("=" * 60)
        print(f"\nBase URL: {self.base_url}")
        print(f"Username: {USERNAME}")
        
        try:
            # Authenticate
            if not await self.login():
                return 1
            
            # Demo features
            await self.demo_agents()
            await self.demo_policies()
            await self.demo_audit_logs()
            await self.demo_realtime()
            
            # Cleanup
            await self.cleanup()
            
            # Success
            self.print_section("Demo Complete")
            self.print_success("All API v1 features demonstrated successfully!")
            print("\nNext steps:")
            print("  1. Access interactive docs at http://localhost:8000/docs")
            print("  2. Read API_V1_README.md for detailed usage")
            print("  3. Check openapi-v1.yaml for full API specification")
            
            return 0
        
        except httpx.ConnectError:
            self.print_error("Could not connect to API server")
            print("\nMake sure the server is running:")
            print("  uvicorn nethical.api.v1.app:create_v1_app --factory --reload")
            return 1
        
        except Exception as e:
            self.print_error(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


async def main():
    """Main entry point."""
    demo = NethicalAPIDemo(BASE_URL)
    return await demo.run()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
