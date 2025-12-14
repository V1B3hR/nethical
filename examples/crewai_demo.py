#!/usr/bin/env python3
"""
CrewAI Demo with Nethical Governance

This example demonstrates how to integrate Nethical's governance
system with CrewAI for safe multi-agent workflows.

Requirements:
    pip install nethical crewai

Usage:
    python crewai_demo.py
"""

import os
import sys
from pathlib import Path


def demo_governance_tool():
    """Demonstrate the CrewAI governance tool."""
    from nethical.integrations.agent_frameworks import NethicalCrewAITool
    
    print("=" * 60)
    print("CrewAI Demo: Governance Tool")
    print("=" * 60)
    
    # Create tool
    tool = NethicalCrewAITool(
        block_threshold=0.7,
        restrict_threshold=0.4,
        storage_dir="./nethical_demo_data"
    )
    
    print(f"\nTool Name: {tool.name}")
    print(f"Description: {tool.description}")
    
    # Test various actions
    test_actions = [
        "Research AI safety best practices",
        "Write a summary of the findings",
        "Delete all user data from the database",
        "Generate code to process files",
        "Access restricted system information",
    ]
    
    for action in test_actions:
        result = tool._run(action)
        print(f"\nAction: {action}")
        print(f"Result: {result}")


def demo_agent_wrapper():
    """Demonstrate agent wrapping."""
    from nethical.integrations.agent_frameworks import NethicalAgentWrapper
    from unittest.mock import Mock
    
    print("\n" + "=" * 60)
    print("CrewAI Demo: Agent Wrapper")
    print("=" * 60)
    
    # Create a mock agent
    mock_agent = Mock()
    mock_agent.role = "researcher"
    mock_agent.execute_task = Mock(return_value="Research completed successfully")
    
    # Wrap with governance
    safe_agent = NethicalAgentWrapper(
        agent=mock_agent,
        pre_check=True,
        post_check=True,
        block_threshold=0.7
    )
    
    print(f"\nAgent ID: {safe_agent.agent_id}")
    print(f"Pre-check enabled: {safe_agent.pre_check}")
    print(f"Post-check enabled: {safe_agent.post_check}")
    
    # Test task execution
    test_tasks = [
        "Research the latest AI papers",
        "Summarize findings about machine learning",
    ]
    
    for task in test_tasks:
        print(f"\nTask: {task}")
        
        # Create mock task
        mock_task = Mock()
        mock_task.__str__ = Mock(return_value=task)
        
        result = safe_agent.execute(mock_task)
        print(f"Result: {result}")


def demo_framework_integration():
    """Demonstrate CrewAI framework integration."""
    from nethical.integrations.agent_frameworks import CrewAIFramework
    
    print("\n" + "=" * 60)
    print("CrewAI Demo: Framework Integration")
    print("=" * 60)
    
    # Create framework
    framework = CrewAIFramework(
        block_threshold=0.7,
        restrict_threshold=0.4,
        storage_dir="./nethical_demo_data"
    )
    
    # Get a tool from the framework
    tool = framework.get_tool()
    
    print("\nCreated governance tool:")
    print(f"  Name: {tool.name}")
    print(f"  Block threshold: {tool.block_threshold}")
    
    # Test the framework's check method
    result = framework.check("Analyze customer data for insights", "query")
    
    print(f"\nFramework check result:")
    print(f"  Decision: {result.decision.value}")
    print(f"  Risk Score: {result.risk_score:.4f}")
    print(f"  Reason: {result.reason}")


def demo_tool_conversion():
    """Demonstrate converting to CrewAI tool format."""
    from nethical.integrations.agent_frameworks import NethicalCrewAITool
    
    print("\n" + "=" * 60)
    print("CrewAI Demo: Tool Conversion")
    print("=" * 60)
    
    # Create our tool
    tool = NethicalCrewAITool(block_threshold=0.7)
    
    # Try to convert to CrewAI Tool
    crewai_tool = tool.as_crewai_tool()
    
    if crewai_tool:
        print("\nSuccessfully converted to CrewAI Tool!")
        print(f"  Tool name: {crewai_tool.name}")
        print(f"  Tool description: {crewai_tool.description[:50]}...")
    else:
        print("\nCrewAI not installed - tool conversion not available")
        print("Install with: pip install crewai")


def demo_with_real_crewai():
    """Demo with actual CrewAI (if installed)."""
    try:
        from crewai import Agent, Task, Crew
        from nethical.integrations.agent_frameworks import (
            NethicalCrewAITool,
            NethicalAgentWrapper
        )
        
        print("\n" + "=" * 60)
        print("CrewAI Demo: Real Agent Integration")
        print("=" * 60)
        
        # Create governance tool
        governance_tool = NethicalCrewAITool(block_threshold=0.7)
        crewai_tool = governance_tool.as_crewai_tool()
        
        if crewai_tool:
            # Create agent with governance tool
            researcher = Agent(
                role="AI Safety Researcher",
                goal="Research AI safety best practices",
                backstory="You are an expert in AI safety.",
                tools=[crewai_tool],
                verbose=True
            )
            
            # Create task
            task = Task(
                description="Research and summarize AI safety guidelines",
                agent=researcher
            )
            
            print("\nCreated CrewAI agent with governance tool")
            print(f"  Agent role: {researcher.role}")
            print(f"  Tools: {[t.name for t in researcher.tools]}")
        else:
            print("\nCrewAI tools not available")
        
    except ImportError:
        print("\n(Skipping real CrewAI demo - crewai not installed)")
        print("Install with: pip install crewai")


def demo_handle_tool():
    """Demonstrate the handle_nethical_tool function."""
    from nethical.integrations.agent_frameworks.crewai_tools import handle_nethical_tool
    
    print("\n" + "=" * 60)
    print("CrewAI Demo: Tool Handler")
    print("=" * 60)
    
    # Simulate tool calls
    tool_calls = [
        {"action": "Summarize the meeting notes"},
        {"action": "Delete sensitive customer data"},
        {"action": "Generate a report on Q4 performance"},
    ]
    
    for call in tool_calls:
        print(f"\nTool call: {call['action']}")
        result = handle_nethical_tool(call)
        print(f"  Decision: {result['decision']}")
        print(f"  Risk Score: {result['risk_score']:.4f}")


def main():
    """Run all demos."""
    print("\n🚀 Nethical + CrewAI Integration Demo\n")
    
    try:
        demo_governance_tool()
    except Exception as e:
        print(f"Governance tool demo error: {e}")
    
    try:
        demo_agent_wrapper()
    except Exception as e:
        print(f"Agent wrapper demo error: {e}")
    
    try:
        demo_framework_integration()
    except Exception as e:
        print(f"Framework integration demo error: {e}")
    
    try:
        demo_tool_conversion()
    except Exception as e:
        print(f"Tool conversion demo error: {e}")
    
    try:
        demo_handle_tool()
    except Exception as e:
        print(f"Handle tool demo error: {e}")
    
    try:
        demo_with_real_crewai()
    except Exception as e:
        print(f"Real CrewAI demo error: {e}")
    
    # Cleanup
    import shutil
    if Path("./nethical_demo_data").exists():
        shutil.rmtree("./nethical_demo_data")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
