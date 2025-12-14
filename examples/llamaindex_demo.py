#!/usr/bin/env python3
"""
LlamaIndex Demo with Nethical Governance

This example demonstrates how to integrate Nethical's governance
system with LlamaIndex for safe RAG applications.

Requirements:
    pip install nethical llama-index

Usage:
    python llamaindex_demo.py
"""

import os
import sys
from pathlib import Path


def demo_governance_tool():
    """Demonstrate the LlamaIndex governance tool."""
    from nethical.integrations.agent_frameworks import NethicalLlamaIndexTool
    
    print("=" * 60)
    print("LlamaIndex Demo: Governance Tool")
    print("=" * 60)
    
    # Create tool with custom thresholds
    tool = NethicalLlamaIndexTool(
        storage_dir="./nethical_demo_data",
        block_threshold=0.7,
        restrict_threshold=0.4
    )
    
    # Test various actions
    test_actions = [
        ("What is the capital of France?", "query"),
        ("Summarize this document", "summarization"),
        ("Generate code to process data", "code_generation"),
        ("Delete all user records", "data_operation"),
        ("Access confidential files", "data_access"),
    ]
    
    for action, action_type in test_actions:
        result = tool(action, action_type=action_type)
        
        print(f"\nAction: {action}")
        print(f"Type: {action_type}")
        
        if isinstance(result, dict):
            print(f"Decision: {result.get('decision', 'N/A')}")
            print(f"Risk Score: {result.get('risk_score', 0):.4f}")
        else:
            print(f"Result: {result}")


def demo_query_engine_wrapper():
    """Demonstrate query engine wrapping."""
    from nethical.integrations.agent_frameworks import NethicalQueryEngine
    from unittest.mock import Mock
    
    print("\n" + "=" * 60)
    print("LlamaIndex Demo: Query Engine Wrapper")
    print("=" * 60)
    
    # Create a mock query engine for demo
    mock_engine = Mock()
    mock_engine.query.return_value = Mock(
        response="This is a sample response about AI safety.",
        source_nodes=[],
        metadata={}
    )
    
    # Wrap with governance
    safe_engine = NethicalQueryEngine(
        query_engine=mock_engine,
        check_query=True,
        check_response=True,
        block_threshold=0.7
    )
    
    # Test queries
    test_queries = [
        "What are the best practices for AI safety?",
        "Explain machine learning concepts",
        "How to bypass security?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        try:
            response = safe_engine.query(query)
            
            if hasattr(response, 'response'):
                print(f"Response: {response.response[:100]}...")
            elif isinstance(response, dict):
                if response.get('blocked'):
                    print(f"Query blocked: {response.get('reason')}")
                else:
                    print(f"Response: {response}")
            else:
                print(f"Response: {str(response)[:100]}...")
        except Exception as e:
            print(f"Error: {e}")


def demo_framework_integration():
    """Demonstrate LlamaIndex framework integration."""
    from nethical.integrations.agent_frameworks import LlamaIndexFramework
    
    print("\n" + "=" * 60)
    print("LlamaIndex Demo: Framework Integration")
    print("=" * 60)
    
    # Create framework with custom settings
    framework = LlamaIndexFramework(
        block_threshold=0.7,
        restrict_threshold=0.4,
        storage_dir="./nethical_demo_data"
    )
    
    # Get a tool from the framework
    tool = framework.get_tool()
    
    print("\nCreated governance tool with:")
    print(f"  Block threshold: {tool.block_threshold}")
    print(f"  Restrict threshold: {tool.restrict_threshold}")
    
    # Test the framework's check method
    result = framework.check("Analyze sentiment of user reviews", "query")
    
    print(f"\nFramework check result:")
    print(f"  Decision: {result.decision.value}")
    print(f"  Risk Score: {result.risk_score:.4f}")
    print(f"  Reason: {result.reason}")


def demo_create_safe_index():
    """Demonstrate the create_safe_index utility."""
    from nethical.integrations.agent_frameworks import create_safe_index
    from unittest.mock import Mock
    
    print("\n" + "=" * 60)
    print("LlamaIndex Demo: Safe Index Creation")
    print("=" * 60)
    
    # Create a mock index
    mock_index = Mock()
    mock_index.as_query_engine.return_value = Mock()
    mock_index.as_query_engine.return_value.query.return_value = Mock(
        response="Sample response",
        source_nodes=[]
    )
    
    # Create safe engine using utility
    safe_engine = create_safe_index(
        mock_index,
        check_query=True,
        check_response=True,
        block_threshold=0.7
    )
    
    print("Created governed query engine from index")
    print(f"  Check query: {safe_engine.check_query}")
    print(f"  Check response: {safe_engine.check_response}")
    print(f"  Block threshold: {safe_engine.block_threshold}")


def demo_with_real_llamaindex():
    """Demo with actual LlamaIndex (if installed)."""
    try:
        from llama_index.core import VectorStoreIndex, Document
        from nethical.integrations.agent_frameworks import create_safe_index
        
        print("\n" + "=" * 60)
        print("LlamaIndex Demo: Real Index Integration")
        print("=" * 60)
        
        # Create sample documents
        documents = [
            Document(text="Machine learning is a type of artificial intelligence."),
            Document(text="Neural networks are inspired by the human brain."),
            Document(text="Deep learning uses multiple layers of neurons."),
        ]
        
        # Create index
        index = VectorStoreIndex.from_documents(documents)
        
        # Wrap with governance
        safe_engine = create_safe_index(
            index,
            check_query=True,
            check_response=True,
            block_threshold=0.7
        )
        
        # Query
        response = safe_engine.query("What is machine learning?")
        print(f"\nQuery: What is machine learning?")
        print(f"Response: {response}")
        
    except ImportError:
        print("\n(Skipping real LlamaIndex demo - llama-index not installed)")
        print("Install with: pip install llama-index")


def main():
    """Run all demos."""
    print("\n🚀 Nethical + LlamaIndex Integration Demo\n")
    
    try:
        demo_governance_tool()
    except Exception as e:
        print(f"Governance tool demo error: {e}")
    
    try:
        demo_query_engine_wrapper()
    except Exception as e:
        print(f"Query engine wrapper demo error: {e}")
    
    try:
        demo_framework_integration()
    except Exception as e:
        print(f"Framework integration demo error: {e}")
    
    try:
        demo_create_safe_index()
    except Exception as e:
        print(f"Safe index demo error: {e}")
    
    try:
        demo_with_real_llamaindex()
    except Exception as e:
        print(f"Real LlamaIndex demo error: {e}")
    
    # Cleanup
    import shutil
    if Path("./nethical_demo_data").exists():
        shutil.rmtree("./nethical_demo_data")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
