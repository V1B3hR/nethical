"""
Example: Universal Vector Language Integration with Nethical

This example demonstrates the complete workflow of using Nethical
with vector-based governance and the 25 Fundamental Laws.

Matches the usage example from the problem statement.
"""

from nethical import Nethical, Agent


def main():
    """Run the vector language integration example."""
    
    print("=" * 70)
    print("Nethical Vector Language Integration Example")
    print("=" * 70)
    print()
    
    # Step 1: Initialize Nethical with 25 Fundamental Laws enabled
    print("1. Initializing Nethical with 25 Fundamental Laws...")
    nethical = Nethical(
        config_path=None,  # Can provide ./config/nethical.yaml if needed
        enable_25_laws=True,
        storage_dir="./example_data"
    )
    print("   ✓ Nethical initialized")
    print()
    
    # Step 2: Register an agent
    print("2. Registering AI agent...")
    agent = Agent(
        id="copilot-agent-001",
        type="coding",
        capabilities=["text_generation", "code_execution"]
    )
    nethical.register_agent(agent)
    print(f"   ✓ Agent registered: {agent.id}")
    print(f"     Type: {agent.type}")
    print(f"     Capabilities: {', '.join(agent.capabilities)}")
    print()
    
    # Step 3: Evaluate various actions
    print("3. Evaluating actions against 25 Fundamental Laws...")
    print()
    
    # Example 1: Safe code generation
    print("   Example 1: Safe Code Generation")
    print("   " + "-" * 60)
    action1 = "def greet(name): return 'Hello, ' + name"
    result1 = nethical.evaluate(
        agent_id="copilot-agent-001",
        action=action1,
        context={"purpose": "demo"}
    )
    
    print(f"   Action: {action1}")
    print(f"   Decision: {result1.decision}")
    print(f"   Laws Evaluated: {result1.laws_evaluated[:5]}")
    print(f"   Risk Score: {result1.risk_score:.2f}")
    print(f"   Confidence: {result1.confidence:.2f}")
    print(f"   Trace ID: {result1.embedding_trace_id}")
    print()
    
    # Example 2: Data access
    print("   Example 2: Data Access Request")
    print("   " + "-" * 60)
    action2 = "Read user preferences from database for personalization"
    result2 = nethical.evaluate(
        agent_id="copilot-agent-001",
        action=action2,
        context={"purpose": "personalization"}
    )
    
    print(f"   Action: {action2}")
    print(f"   Decision: {result2.decision}")
    print(f"   Laws Evaluated: {result2.laws_evaluated[:5]}")
    print(f"   Risk Score: {result2.risk_score:.2f}")
    print(f"   Detected Primitives: {', '.join(result2.detected_primitives[:3])}")
    print()
    
    # Example 3: High-risk action
    print("   Example 3: High-Risk System Modification")
    print("   " + "-" * 60)
    action3 = "Execute system command to modify firewall rules"
    result3 = nethical.evaluate(
        agent_id="copilot-agent-001",
        action=action3,
        context={"purpose": "security"}
    )
    
    print(f"   Action: {action3}")
    print(f"   Decision: {result3.decision}")
    print(f"   Laws Evaluated: {result3.laws_evaluated[:5]}")
    print(f"   Risk Score: {result3.risk_score:.2f}")
    print(f"   Reasoning: {result3.reasoning}")
    print()
    
    # Step 4: Trace an embedding for audit
    print("4. Tracing embedding decision for audit...")
    trace = nethical.trace_embedding(result1.embedding_trace_id)
    if trace:
        print(f"   ✓ Trace found: {trace.get('found_in', 'audit_log')}")
        print(f"     Trace ID: {trace.get('embedding_trace_id', 'N/A')}")
    print()
    
    # Step 5: Get system statistics
    print("5. System Statistics")
    print("   " + "-" * 60)
    stats = nethical.get_stats()
    print(f"   Registered Agents: {stats['agent_count']}")
    print(f"   25 Laws Enabled: {stats['governance_enabled']['25_laws']}")
    print(f"   Vector Evaluation: {stats['governance_enabled']['vector_evaluation']}")
    
    if 'embedding_stats' in stats:
        emb_stats = stats['embedding_stats']
        print(f"   Embedding Model: {emb_stats['provider']}")
        print(f"   Cache Hit Rate: {emb_stats['hit_rate']:.2%}")
        print(f"   Total Embeddings Generated: {emb_stats['total_generated']}")
    print()
    
    # Step 6: Demonstrate multiple agent management
    print("6. Multi-Agent Management")
    print("   " + "-" * 60)
    
    # Register additional agents
    agent2 = Agent(id="analytics-agent", type="data", capabilities=["data_analysis"])
    agent3 = Agent(id="chat-agent", type="chat", capabilities=["conversation"])
    
    nethical.register_agent(agent2)
    nethical.register_agent(agent3)
    
    print(f"   Total agents: {len(nethical.list_agents())}")
    for agent in nethical.list_agents():
        print(f"   - {agent.id} ({agent.type})")
    print()
    
    # Step 7: Show structured result format (as per problem statement)
    print("7. Structured Result Format (JSON-like)")
    print("   " + "-" * 60)
    print("   {")
    print(f'     "decision": "{result1.decision}",')
    print(f'     "laws_evaluated": {result1.laws_evaluated[:3]},')
    print(f'     "risk_score": {result1.risk_score:.2f},')
    print(f'     "embedding_trace_id": "{result1.embedding_trace_id[:16]}..."')
    print("   }")
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
