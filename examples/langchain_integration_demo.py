"""
LangChain Integration Demo

This example demonstrates how to integrate Nethical's IntegratedGovernance
with LangChain agents for comprehensive AI safety and ethics checking.

This demo shows:
1. Basic NethicalGuardTool usage
2. Integration with LangChain agents
3. Chaining multiple guards (Nethical + LlamaGuard)
4. Pre-action and post-action evaluation

Requirements:
    pip install langchain langchain-openai openai

Optional for LlamaGuard:
    pip install transformers torch
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_basic_guard_tool():
    """Demonstrate basic NethicalGuardTool usage."""
    print("=" * 80)
    print("Demo 1: Basic NethicalGuardTool Usage")
    print("=" * 80)

    from nethical.integrations.langchain_tools import NethicalGuardTool

    # Create the guard tool
    guard = NethicalGuardTool(
        storage_dir="./demo_nethical_data",
        block_threshold=0.7,
        warn_threshold=0.4,
        enable_shadow_mode=False,  # Faster for demo
        enable_ml_blending=False,
        enable_anomaly_detection=False,
    )

    # Test various actions
    test_actions = [
        ("What is the weather today?", "safe_query"),
        ("Tell me about AI ethics", "educational_query"),
        ("How do I secure my password?", "security_query"),
    ]

    for action, label in test_actions:
        print(f"\n📝 Testing: {label}")
        print(f"   Action: {action}")

        result = guard._run(action=action, agent_id="demo_agent", action_type="query")

        print(f"   Result: {result}")


def demo_with_langchain_agent():
    """Demonstrate integration with LangChain agent."""
    print("\n" + "=" * 80)
    print("Demo 2: LangChain Agent Integration")
    print("=" * 80)

    try:
        from langchain.agents import initialize_agent, AgentType
        from langchain.tools import Tool
        from langchain_openai import OpenAI
        from nethical.integrations.langchain_tools import NethicalGuardTool

        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            print(
                "\n⚠️  OPENAI_API_KEY not set. Skipping LangChain agent demo."
            )
            print("   Set it with: export OPENAI_API_KEY='your-key-here'")
            return

        # Create LLM
        llm = OpenAI(temperature=0)

        # Create Nethical guard tool
        nethical_guard = NethicalGuardTool(
            storage_dir="./demo_nethical_data",
            block_threshold=0.7,
            enable_shadow_mode=False,
        )

        # Create a simple calculator tool for demo
        calculator_tool = Tool(
            name="Calculator",
            func=lambda x: str(eval(x)),
            description="Useful for math calculations. Input should be a valid Python expression.",
        )

        # Create agent with Nethical guard
        tools = [nethical_guard, calculator_tool]
        agent = initialize_agent(
            tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )

        # Test the agent
        print("\n🤖 Testing agent with guard protection:")
        print("   Query: What is 25 * 4?")

        try:
            response = agent.run("What is 25 * 4?")
            print(f"   Response: {response}")
        except Exception as e:
            print(f"   Error: {e}")

    except ImportError as e:
        print(f"\n⚠️  Missing dependencies for LangChain demo: {e}")
        print("   Install with: pip install langchain langchain-openai openai")


def demo_create_nethical_agent_helper():
    """Demonstrate the create_nethical_agent helper function."""
    print("\n" + "=" * 80)
    print("Demo 3: Using create_nethical_agent Helper")
    print("=" * 80)

    try:
        from langchain.tools import Tool
        from langchain_openai import OpenAI
        from nethical.integrations.langchain_tools import create_nethical_agent

        if not os.getenv("OPENAI_API_KEY"):
            print(
                "\n⚠️  OPENAI_API_KEY not set. Skipping create_nethical_agent demo."
            )
            return

        # Create LLM
        llm = OpenAI(temperature=0)

        # Create some tools
        tools = [
            Tool(
                name="Calculator",
                func=lambda x: str(eval(x)),
                description="Useful for math calculations.",
            )
        ]

        # Create agent with Nethical guard automatically added
        agent = create_nethical_agent(
            llm=llm,
            tools=tools,
            storage_dir="./demo_nethical_data",
            block_threshold=0.7,
            prepend_guard=True,  # Add guard at the beginning
            verbose=True,
        )

        print("\n✅ Agent created with Nethical guard protection")
        print("   The guard tool is automatically added to the agent's toolset")

    except ImportError as e:
        print(f"\n⚠️  Missing dependencies: {e}")
        print("   Install with: pip install langchain langchain-openai openai")


def demo_chained_guards():
    """Demonstrate chaining Nethical with LlamaGuard."""
    print("\n" + "=" * 80)
    print("Demo 4: Chaining Multiple Guards (Nethical + LlamaGuard)")
    print("=" * 80)

    from nethical.integrations.langchain_tools import (
        NethicalGuardTool,
        chain_guards,
    )

    # Create Nethical guard
    nethical_guard = NethicalGuardTool(
        storage_dir="./demo_nethical_data",
        block_threshold=0.7,
        enable_shadow_mode=False,
    )

    # Note: LlamaGuard requires local model or API - we'll demonstrate the concept
    print("\n📋 Chaining guards allows multiple layers of protection:")
    print("   1. LlamaGuard: Fast content moderation (pre-filter)")
    print("   2. Nethical: Comprehensive governance and policy checks")

    # Test with just Nethical (no LlamaGuard for this demo)
    test_action = "What are the best practices for data privacy?"

    print(f"\n🔍 Evaluating: {test_action}")

    result = chain_guards(
        nethical_tool=nethical_guard,
        action=test_action,
        agent_id="demo_agent",
        llama_guard=None,  # Would be LlamaGuardChain instance if available
    )

    print(f"\n✅ Chained evaluation result:")
    print(f"   Final Decision: {result['final_decision']}")
    print(f"   Nethical Result: {result['nethical'][:100]}...")

    if result["llama_guard"]:
        print(f"   LlamaGuard Result: {result['llama_guard']}")
    else:
        print("   LlamaGuard: Not configured (optional)")


def demo_pre_and_post_action_checks():
    """Demonstrate pre-action and post-action evaluation."""
    print("\n" + "=" * 80)
    print("Demo 5: Pre-Action and Post-Action Evaluation")
    print("=" * 80)

    from nethical.integrations.langchain_tools import NethicalGuardTool

    guard = NethicalGuardTool(
        storage_dir="./demo_nethical_data",
        block_threshold=0.7,
        enable_shadow_mode=False,
    )

    # Simulate a workflow with pre-action and post-action checks
    user_input = "Tell me how to implement authentication"
    agent_output = "Here's a secure authentication implementation..."

    print("\n🔄 Workflow with Nethical guards:")

    # Pre-action check (evaluate user input)
    print("\n1️⃣ Pre-action check (user input):")
    print(f"   Input: {user_input}")

    pre_result = guard._run(
        action=user_input, agent_id="demo_agent", action_type="query"
    )

    print(f"   Guard Result: {pre_result}")

    # Simulate agent processing (only if pre-check allows)
    if "BLOCK" not in pre_result:
        print(
            f"\n2️⃣ Agent processing: ✅ Allowed (input passed governance check)"
        )

        # Post-action check (evaluate agent output)
        print("\n3️⃣ Post-action check (agent output):")
        print(f"   Output: {agent_output}")

        post_result = guard._run(
            action=agent_output, agent_id="demo_agent", action_type="response"
        )

        print(f"   Guard Result: {post_result}")

        if "BLOCK" not in post_result:
            print("\n✅ Output approved - safe to return to user")
        else:
            print("\n🚫 Output blocked - safety issue detected")
    else:
        print("\n🚫 Action blocked by pre-check - agent not invoked")


def demo_custom_thresholds():
    """Demonstrate custom threshold configuration."""
    print("\n" + "=" * 80)
    print("Demo 6: Custom Threshold Configuration")
    print("=" * 80)

    from nethical.integrations.langchain_tools import NethicalGuardTool

    # Create guards with different threshold configurations
    configs = [
        {"block_threshold": 0.9, "warn_threshold": 0.7, "name": "Permissive"},
        {"block_threshold": 0.7, "warn_threshold": 0.4, "name": "Balanced"},
        {"block_threshold": 0.5, "warn_threshold": 0.2, "name": "Strict"},
    ]

    test_action = "How can I improve my code security?"

    print(f"\n📊 Testing action with different thresholds:")
    print(f"   Action: {test_action}")

    for config in configs:
        print(f"\n   {config['name']} configuration:")
        print(
            f"   - Block threshold: {config['block_threshold']}, Warn threshold: {config['warn_threshold']}"
        )

        guard = NethicalGuardTool(
            storage_dir="./demo_nethical_data",
            block_threshold=config["block_threshold"],
            warn_threshold=config["warn_threshold"],
            enable_shadow_mode=False,
        )

        result = guard._run(action=test_action, agent_id="demo_agent")
        decision = "ALLOW" if "ALLOW" in result else "WARN" if "WARN" in result else "BLOCK"
        print(f"   Decision: {decision}")


def main():
    """Run all demos."""
    print("\n🎯 Nethical LangChain Integration Demo")
    print("=" * 80)
    print("This demo showcases various ways to integrate Nethical with LangChain")
    print("=" * 80)

    # Demo 1: Basic usage
    demo_basic_guard_tool()

    # Demo 2: LangChain agent integration
    demo_with_langchain_agent()

    # Demo 3: Helper function
    demo_create_nethical_agent_helper()

    # Demo 4: Chained guards
    demo_chained_guards()

    # Demo 5: Pre/post action checks
    demo_pre_and_post_action_checks()

    # Demo 6: Custom thresholds
    demo_custom_thresholds()

    print("\n" + "=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)
    print("\nFor more information:")
    print("  - Documentation: nethical/integrations/langchain_tools.py")
    print("  - Tests: tests/test_langchain_integration.py")
    print("  - Examples: examples/langchain_integration_demo.py")


if __name__ == "__main__":
    main()
