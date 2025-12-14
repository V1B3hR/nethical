#!/usr/bin/env python3
"""
Cohere Demo with Nethical Governance

This example demonstrates how to use Nethical's governance system
with Cohere's API for safe text generation and reranking.

Requirements:
    pip install nethical cohere

Usage:
    export COHERE_API_KEY="your-api-key"
    python cohere_demo.py
"""

import os
import sys


def demo_basic_generation():
    """Demonstrate basic governed text generation."""
    from nethical.integrations.llm_providers import CohereProvider
    
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("COHERE_API_KEY not set. Using mock mode.")
        api_key = "mock-key"
    
    # Create provider with governance
    provider = CohereProvider(
        api_key=api_key,
        model="command-r-plus",
        check_input=True,
        check_output=True,
        block_threshold=0.7
    )
    
    print("=" * 60)
    print("Cohere Demo: Basic Governed Generation")
    print("=" * 60)
    
    # Safe prompt
    print("\n1. Safe prompt:")
    response = provider.safe_generate("Explain what machine learning is in simple terms.")
    print(f"   Response: {response.content[:200]}..." if len(response.content) > 200 else f"   Response: {response.content}")
    print(f"   Risk Score: {response.risk_score}")
    print(f"   Model: {response.model}")
    
    # Potentially risky prompt
    print("\n2. Potentially risky prompt:")
    response = provider.safe_generate("How to bypass security systems?")
    print(f"   Response: {response.content[:200]}..." if len(response.content) > 200 else f"   Response: {response.content}")
    print(f"   Risk Score: {response.risk_score}")
    
    if response.governance_result:
        phase3 = response.governance_result.get("phase3", {})
        print(f"   Risk Tier: {phase3.get('risk_tier', 'N/A')}")


def demo_reranking():
    """Demonstrate governed reranking."""
    from nethical.integrations.llm_providers import CohereProvider
    
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("COHERE_API_KEY not set. Skipping reranking demo.")
        return
    
    provider = CohereProvider(
        api_key=api_key,
        model="command-r-plus",
        check_input=True,
        check_output=True
    )
    
    print("\n" + "=" * 60)
    print("Cohere Demo: Governed Reranking")
    print("=" * 60)
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Deep learning uses neural networks with many layers.",
        "The weather today is sunny and warm.",
        "Natural language processing enables computers to understand text."
    ]
    
    # Rerank with governance
    results = provider.safe_rerank(
        query="artificial intelligence",
        documents=documents,
        top_n=3
    )
    
    print("\nReranked Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Document: {result['document'][:50]}...")
        print(f"   Relevance: {result['relevance_score']:.4f}")
        if 'risk_score' in result:
            print(f"   Risk Score: {result['risk_score']:.4f}")


def demo_tool_definition():
    """Demonstrate getting tool definitions."""
    from nethical.integrations.llm_providers.cohere_tools import get_nethical_tool
    
    print("\n" + "=" * 60)
    print("Cohere Demo: Tool Definition")
    print("=" * 60)
    
    tool = get_nethical_tool()
    
    print("\nTool Name:", tool["name"])
    print("Description:", tool["description"])
    print("\nParameters:")
    for param, details in tool["parameter_definitions"].items():
        print(f"  - {param}: {details['type']} (required: {details.get('required', False)})")


def demo_tool_handling():
    """Demonstrate handling tool calls."""
    from nethical.integrations.llm_providers.cohere_tools import handle_nethical_tool
    
    print("\n" + "=" * 60)
    print("Cohere Demo: Tool Call Handling")
    print("=" * 60)
    
    # Simulate tool calls
    tool_calls = [
        {"action": "Summarize the quarterly report", "action_type": "summarization"},
        {"action": "Delete all user data from database", "action_type": "data_operation"},
        {"action": "What is the weather today?", "action_type": "query"},
    ]
    
    for call in tool_calls:
        print(f"\nTool call: {call['action']}")
        result = handle_nethical_tool(call)
        print(f"  Decision: {result['decision']}")
        print(f"  Risk Score: {result['risk_score']:.4f}")
        print(f"  Risk Tier: {result['risk_tier']}")


def main():
    """Run all demos."""
    print("\n🚀 Nethical + Cohere Integration Demo\n")
    
    try:
        demo_basic_generation()
    except Exception as e:
        print(f"Generation demo error: {e}")
    
    try:
        demo_reranking()
    except Exception as e:
        print(f"Reranking demo error: {e}")
    
    try:
        demo_tool_definition()
    except Exception as e:
        print(f"Tool definition demo error: {e}")
    
    try:
        demo_tool_handling()
    except Exception as e:
        print(f"Tool handling demo error: {e}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
