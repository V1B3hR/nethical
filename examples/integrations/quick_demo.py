#!/usr/bin/env python3
"""Quick demo of both LLM integrations.

This script demonstrates both the Claude tool and REST API integrations
without requiring external API keys.

Run: python examples/integrations/quick_demo.py
"""

from nethical.integrations.claude_tools import (
    get_nethical_tool,
    handle_nethical_tool,
    evaluate_action,
)

def demo_claude_tool():
    """Demonstrate the Claude tool integration."""
    print("=" * 60)
    print("Claude/Anthropic Tool Integration Demo")
    print("=" * 60)
    
    # 1. Get tool definition
    tool = get_nethical_tool()
    print("\n1. Tool Definition:")
    print(f"   Name: {tool['name']}")
    print(f"   Description: {tool['description'][:80]}...")
    print(f"   Required params: {tool['input_schema']['required']}")
    
    # 2. Test various actions
    print("\n2. Evaluating Actions:")
    
    test_cases = [
        {
            "action": "def hello(): return 'world'",
            "agent_id": "demo-agent",
            "action_type": "code_generation"
        },
        {
            "action": "DROP TABLE users",
            "agent_id": "demo-agent",
            "action_type": "database_command"
        },
        {
            "action": "Send email to user@example.com with SSN 123-45-6789",
            "agent_id": "demo-agent",
            "action_type": "communication"
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test['action'][:50]}...")
        result = handle_nethical_tool(test)
        
        emoji = "✓" if result['decision'] == "ALLOW" else "✗"
        print(f"   {emoji} Decision: {result['decision']}")
        print(f"   Reason: {result['reason']}")
        
        if result.get('risk_score'):
            print(f"   Risk Score: {result['risk_score']:.2f}")
        
        if result.get('pii_detected'):
            print(f"   PII Detected: {', '.join(result.get('pii_types', []))}")
    
    # 3. Simple evaluation
    print("\n3. Simple Evaluation Function:")
    decision = evaluate_action("print('Hello')", agent_id="simple-test")
    print(f"   Decision for safe code: {decision}")


def demo_rest_api_info():
    """Show REST API information."""
    print("\n\n" + "=" * 60)
    print("REST API Integration Info")
    print("=" * 60)
    
    print("\n1. Starting the Server:")
    print("   python -m nethical.integrations.rest_api")
    print("   # Or: uvicorn nethical.integrations.rest_api:app --port 8000")
    
    print("\n2. Python Client Example:")
    print("""
   import requests
   
   response = requests.post(
       "http://localhost:8000/evaluate",
       json={
           "action": "Generate code to sort data",
           "agent_id": "my-agent",
           "action_type": "code_generation"
       }
   )
   
   result = response.json()
   if result["decision"] != "ALLOW":
       print(f"Blocked: {result['reason']}")
    """)
    
    print("\n3. JavaScript Client Example:")
    print("""
   const response = await fetch('http://localhost:8000/evaluate', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({
           action: 'Access user database',
           agent_id: 'frontend-app'
       })
   });
   
   const result = await response.json();
   if (result.decision !== 'ALLOW') {
       console.log('Blocked:', result.reason);
   }
    """)
    
    print("\n4. API Endpoints:")
    print("   GET  /health    - Health check")
    print("   GET  /          - API information")
    print("   POST /evaluate  - Evaluate an action")
    print("   GET  /docs      - Interactive documentation")


def integration_summary():
    """Show integration summary."""
    print("\n\n" + "=" * 60)
    print("Integration Summary")
    print("=" * 60)
    
    print("\nBoth integrations provide:")
    print("  ✓ Safety and ethics evaluation")
    print("  ✓ PII detection")
    print("  ✓ Risk scoring")
    print("  ✓ Audit trail generation")
    print("  ✓ Four decision types: ALLOW, RESTRICT, BLOCK, TERMINATE")
    
    print("\nClaude Integration:")
    print("  • Native tool calling")
    print("  • Seamless with Claude's workflow")
    print("  • Install: pip install anthropic")
    print("  • See: examples/integrations/claude_example.py")
    
    print("\nREST API Integration:")
    print("  • Works with any LLM")
    print("  • HTTP/JSON interface")
    print("  • Install: pip install fastapi uvicorn")
    print("  • See: examples/integrations/rest_api_example.py")
    
    print("\nFor detailed documentation:")
    print("  • nethical/integrations/README.md")
    print("  • Run tests: pytest tests/test_*_integration.py")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Nethical LLM Integrations Quick Demo")
    print("=" * 60)
    
    try:
        demo_claude_tool()
        demo_rest_api_info()
        integration_summary()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print()
        
    except Exception as e:
        print(f"\n✗ Error running demo: {e}")
        import traceback
        traceback.print_exc()
