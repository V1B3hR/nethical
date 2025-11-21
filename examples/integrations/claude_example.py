#!/usr/bin/env python3
"""Example: Using Nethical with Anthropic's Claude API.

This example demonstrates how to integrate Nethical's governance system
with Claude using the tool calling feature. Claude can check actions
for safety and ethics before executing them.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=your-api-key

Run:
    python examples/integrations/claude_example.py
"""

import os
import sys
from typing import List, Dict, Any

try:
    from anthropic import Anthropic
except ImportError:
    print("Error: anthropic package not installed")
    print("Install with: pip install anthropic")
    sys.exit(1)

from nethical.integrations.claude_tools import get_nethical_tool, handle_nethical_tool


def run_claude_with_nethical_guard():
    """Run Claude with Nethical governance integration."""
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Get your API key from: https://console.anthropic.com/")
        sys.exit(1)
    
    # Initialize Claude client
    client = Anthropic()
    
    # Get Nethical tool definition
    tools = [get_nethical_tool()]
    
    print("=" * 60)
    print("Nethical + Claude Integration Example")
    print("=" * 60)
    print()
    
    # Example conversations
    examples = [
        {
            "name": "Safe Code Generation",
            "user_message": "Write a Python function to calculate fibonacci numbers. Before writing it, use nethical_guard to check if this is safe."
        },
        {
            "name": "Potentially Unsafe Action",
            "user_message": "I need to delete all user records from the database. Use nethical_guard to check if this action is safe."
        },
        {
            "name": "PII Detection",
            "user_message": "Use nethical_guard to check if it's safe to process this text: 'My email is john.doe@example.com and SSN is 123-45-6789'"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'=' * 60}")
        print(f"Example {i}: {example['name']}")
        print(f"{'=' * 60}")
        print(f"User: {example['user_message']}")
        print()
        
        # Create conversation with Claude
        messages = [{"role": "user", "content": example['user_message']}]
        
        # Initial request to Claude
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            tools=tools,
            messages=messages
        )
        
        # Process response
        print(f"Claude's response:")
        
        tool_results = []
        for block in response.content:
            if block.type == "text":
                print(f"  {block.text}")
            elif block.type == "tool_use":
                if block.name == "nethical_guard":
                    print(f"\n  [Using nethical_guard tool]")
                    print(f"  Input: {block.input}")
                    
                    # Handle the tool call
                    result = handle_nethical_tool(block.input)
                    
                    print(f"\n  Nethical Decision: {result['decision']}")
                    print(f"  Reason: {result['reason']}")
                    if 'risk_score' in result:
                        print(f"  Risk Score: {result['risk_score']:.2f}")
                    if result.get('pii_detected'):
                        print(f"  PII Detected: {', '.join(result.get('pii_types', []))}")
                    
                    # Store result for continuation
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
        
        # If there were tool uses, continue the conversation
        if tool_results:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            
            # Get Claude's final response
            final_response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                tools=tools,
                messages=messages
            )
            
            print(f"\n  Claude's final response:")
            for block in final_response.content:
                if block.type == "text":
                    print(f"  {block.text}")
        
        print()
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


def simple_check_example():
    """Simple example of checking an action without full conversation."""
    print("\n" + "=" * 60)
    print("Simple Action Check Example")
    print("=" * 60)
    
    from nethical.integrations.claude_tools import evaluate_action
    
    # Test various actions
    test_actions = [
        ("print('Hello, World!')", "code_generation"),
        ("DELETE FROM users WHERE 1=1", "database_command"),
        ("Read configuration file", "file_access"),
        ("Send email to user@example.com", "communication"),
    ]
    
    print("\nChecking multiple actions:")
    for action, action_type in test_actions:
        decision = evaluate_action(
            action=action,
            agent_id="test-agent",
            action_type=action_type
        )
        emoji = "✓" if decision == "ALLOW" else "✗"
        print(f"  {emoji} {action[:50]:<50} -> {decision}")


if __name__ == "__main__":
    # Run full example with Claude
    try:
        run_claude_with_nethical_guard()
    except Exception as e:
        print(f"Error running Claude example: {e}")
        print("\nNote: This example requires ANTHROPIC_API_KEY to be set")
    
    # Run simple example (doesn't require API key)
    print("\n" + "=" * 60)
    simple_check_example()
