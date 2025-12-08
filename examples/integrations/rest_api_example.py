#!/usr/bin/env python3
"""Example: Using Nethical REST API with various LLMs.

This example demonstrates how to integrate Nethical's governance system
with any LLM through a REST API. Works with OpenAI, Gemini, LLaMA, or any
LLM that can make HTTP requests.

Setup:
    pip install requests

Run API Server:
    python -m nethical.integrations.rest_api
    # Or: uvicorn nethical.integrations.rest_api:app --port 8000

Run Examples:
    python examples/integrations/rest_api_example.py
"""

import requests
import time
import sys
from typing import Dict, Any, Optional


# API Configuration
API_BASE_URL = "http://localhost:8000"


def check_api_health() -> bool:
    """Check if the Nethical API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is healthy (status: {data['status']})")
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Is the server running?")
        print(f"  Start with: python -m nethical.integrations.rest_api")
        return False
    except Exception as e:
        print(f"✗ Error checking API: {e}")
        return False


def evaluate_action(
    action: str,
    agent_id: str = "example-agent",
    action_type: str = "query",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate an action through the Nethical API.

    Args:
        action: The action to evaluate
        agent_id: Identifier for the agent
        action_type: Type of action
        context: Additional context

    Returns:
        Dict with decision and details
    """
    response = requests.post(
        f"{API_BASE_URL}/evaluate",
        json={
            "action": action,
            "agent_id": agent_id,
            "action_type": action_type,
            "context": context or {},
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def basic_examples():
    """Run basic evaluation examples."""
    print("\n" + "=" * 60)
    print("Basic Action Evaluation Examples")
    print("=" * 60)

    test_cases = [
        {
            "name": "Safe Code Generation",
            "action": "Write a Python function to sort a list",
            "action_type": "code_generation",
        },
        {
            "name": "Potentially Unsafe Database Command",
            "action": "DROP TABLE users",
            "action_type": "database_command",
        },
        {
            "name": "PII Detection",
            "action": "Process user data: email=john@example.com, phone=555-1234",
            "action_type": "data_processing",
        },
        {
            "name": "File System Access",
            "action": "Delete all files in /tmp directory",
            "action_type": "file_operation",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   Action: {test['action']}")

        try:
            result = evaluate_action(
                action=test["action"], action_type=test["action_type"]
            )

            decision = result["decision"]
            emoji = "✓" if decision == "ALLOW" else "✗"

            print(f"   {emoji} Decision: {decision}")
            print(f"   Reason: {result['reason']}")

            if result.get("risk_score") is not None:
                print(f"   Risk Score: {result['risk_score']:.2f}")

            if result.get("pii_detected"):
                print(f"   PII Detected: {', '.join(result.get('pii_types', []))}")

        except Exception as e:
            print(f"   ✗ Error: {e}")


def openai_integration_example():
    """Example showing integration with OpenAI."""
    print("\n" + "=" * 60)
    print("OpenAI Integration Pattern")
    print("=" * 60)

    print("\nPattern: Pre-check user queries before sending to OpenAI")
    print("-" * 60)

    # Simulated user queries
    user_queries = [
        "What is the weather today?",
        "How can I hack into a database?",
        "My credit card number is 4532-1234-5678-9010, can you store it?",
    ]

    for query in user_queries:
        print(f"\nUser Query: {query}")

        # Check with Nethical before sending to OpenAI
        result = evaluate_action(
            action=query, agent_id="openai-gpt4", action_type="query"
        )

        if result["decision"] == "ALLOW":
            print(f"  ✓ Query allowed - would send to OpenAI")
            # In real code: openai.ChatCompletion.create(messages=[...])
        else:
            print(f"  ✗ Query blocked: {result['reason']}")
            # In real code: return error to user


def gemini_integration_example():
    """Example showing integration with Google Gemini."""
    print("\n" + "=" * 60)
    print("Google Gemini Integration Pattern")
    print("=" * 60)

    print("\nPattern: Post-check generated content before returning to user")
    print("-" * 60)

    # Simulated Gemini responses
    gemini_responses = [
        "Here's a simple sorting algorithm in Python...",
        "To access the database without permission, you could...",
        "I can help you with that user email: admin@company.com",
    ]

    for i, response in enumerate(gemini_responses, 1):
        print(f"\n{i}. Gemini Generated: {response[:60]}...")

        # Check response before returning to user
        result = evaluate_action(
            action=response, agent_id="gemini-pro", action_type="generated_content"
        )

        if result["decision"] == "ALLOW":
            print(f"   ✓ Response allowed - would return to user")
        else:
            print(f"   ✗ Response blocked: {result['reason']}")
            if result.get("pii_detected"):
                print(f"   PII types: {', '.join(result.get('pii_types', []))}")


def custom_llm_integration_example():
    """Example for custom LLM integration."""
    print("\n" + "=" * 60)
    print("Custom LLM Integration Pattern")
    print("=" * 60)

    print("\nPattern: Bidirectional checking (before and after)")
    print("-" * 60)

    user_prompt = "Generate code to access user database"

    # Step 1: Check user prompt
    print(f"\nStep 1: Check user prompt")
    print(f"  Prompt: {user_prompt}")

    pre_check = evaluate_action(
        action=user_prompt, agent_id="custom-llm", action_type="user_prompt"
    )

    print(f"  Pre-check: {pre_check['decision']}")

    if pre_check["decision"] != "ALLOW":
        print(f"  ✗ Blocked at input: {pre_check['reason']}")
        return

    # Step 2: Simulate LLM generation
    llm_output = "import sqlite3\nconn = sqlite3.connect('users.db')\n..."

    print(f"\nStep 2: Check LLM output")
    print(f"  Output: {llm_output[:50]}...")

    post_check = evaluate_action(
        action=llm_output, agent_id="custom-llm", action_type="generated_code"
    )

    print(f"  Post-check: {post_check['decision']}")

    if post_check["decision"] == "ALLOW":
        print(f"  ✓ Safe to return to user")
    else:
        print(f"  ✗ Output filtered: {post_check['reason']}")


def javascript_client_example():
    """Show JavaScript/Node.js client example."""
    print("\n" + "=" * 60)
    print("JavaScript/Node.js Client Example")
    print("=" * 60)

    js_code = """
// Node.js / Browser example
async function checkActionSafety(action, agentId = 'my-app') {
    const response = await fetch('http://localhost:8000/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            action: action,
            agent_id: agentId,
            action_type: 'query'
        })
    });
    
    const result = await response.json();
    return result.decision === 'ALLOW';
}

// Usage in your app
async function handleUserQuery(query) {
    // Check with Nethical before processing
    if (!await checkActionSafety(query)) {
        console.log('Action blocked by Nethical');
        return { error: 'Action not allowed' };
    }
    
    // Proceed with LLM call
    const llmResponse = await callYourLLM(query);
    return llmResponse;
}
    """.strip()

    print(f"\n{js_code}")


def performance_test():
    """Test API performance."""
    print("\n" + "=" * 60)
    print("Performance Test")
    print("=" * 60)

    num_requests = 10
    actions = [
        "Write a hello world program",
        "Access user data",
        "Delete temporary files",
    ]

    print(f"\nSending {num_requests} requests...")

    start_time = time.time()
    results = []

    for i in range(num_requests):
        action = actions[i % len(actions)]
        try:
            result = evaluate_action(action, agent_id=f"perf-test-{i}")
            results.append(result)
        except Exception as e:
            print(f"  Error on request {i}: {e}")

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n✓ Completed {len(results)} requests in {duration:.2f} seconds")
    print(f"  Average: {duration/num_requests*1000:.1f} ms per request")
    print(f"  Throughput: {num_requests/duration:.1f} requests/second")

    # Show decision distribution
    decisions = {}
    for result in results:
        decision = result["decision"]
        decisions[decision] = decisions.get(decision, 0) + 1

    print(f"\n  Decision distribution:")
    for decision, count in sorted(decisions.items()):
        print(f"    {decision}: {count}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Nethical REST API Integration Examples")
    print("=" * 60)

    # Check API availability
    if not check_api_health():
        print("\n⚠ API server is not running!")
        print("Start the server with one of these commands:")
        print("  python -m nethical.integrations.rest_api")
        print("  uvicorn nethical.integrations.rest_api:app --port 8000")
        sys.exit(1)

    try:
        # Run examples
        basic_examples()
        openai_integration_example()
        gemini_integration_example()
        custom_llm_integration_example()
        javascript_client_example()
        performance_test()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
