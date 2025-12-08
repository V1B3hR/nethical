"""
F5: Simulation & Replay Demo

This demo showcases the time-travel debugging and what-if analysis capabilities
of the Nethical governance system. It demonstrates:

1. Action Stream Persistence - Storing and retrieving historical actions
2. Time-Travel Debugging - Replaying actions from specific time points
3. What-If Analysis - Simulating policy changes on historical data
4. Policy Validation - Comparing outcomes between different policies

Use cases:
- Test new policies before deployment
- Analyze impact of policy changes
- Debug governance decisions
- Validate policy effectiveness

Status: Future Track F5 - Demonstration of planned functionality
"""

import json
import os
import tempfile
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, Dict

# Add parent directory to path for demo utilities
sys.path.insert(0, str(Path(__file__).parent))

try:
    from demo_utils import (
        print_header,
        print_section,
        print_success,
        print_error,
        print_warning,
        print_info,
        print_metric,
        safe_import,
        run_demo_safely,
        print_feature_not_implemented,
        print_next_steps,
        print_key_features,
    )
except ImportError:
    # Fallback implementations
    def print_header(title, width=70):
        print(f"\n{'='*width}\n{title}\n{'='*width}\n")

    def print_section(title, level=1):
        print(
            f"\n{'---' if level==2 else '==='*23} {title} {'---' if level==2 else '==='*23}"
        )

    def print_success(msg):
        print(f"✓ {msg}")

    def print_error(msg):
        print(f"✗ {msg}")

    def print_warning(msg):
        print(f"⚠  {msg}")

    def print_info(msg, indent=0):
        print(f"{'  '*indent}{msg}")

    def print_metric(name, value, unit="", indent=1):
        print(f"{'  '*indent}{name}: {value}{unit}")

    def safe_import(module, cls=None):
        try:
            mod = __import__(module, fromlist=[cls] if cls else [])
            return getattr(mod, cls) if cls else mod
        except:
            return None

    def run_demo_safely(func, name, skip=True):
        try:
            func()
            return True
        except Exception as e:
            print_error(f"Error in {name}: {e}")
            return False

    def print_feature_not_implemented(name, coming=None):
        msg = f"Feature '{name}' not yet implemented"
        if coming:
            msg += f" (coming in {coming})"
        print_warning(msg)

    def print_next_steps(steps, title="Next Steps"):
        print(f"\n{title}:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")

    def print_key_features(features, title="Key Features"):
        print(f"\n{title}:")
        for feature in features:
            print(f"  ✓ {feature}")


# Try to import required modules
ActionReplayer = safe_import("nethical.core.action_replayer", "ActionReplayer")
AgentAction = safe_import("nethical.core.governance", "AgentAction")
ActionType = safe_import("nethical.core.governance", "ActionType")
PersistenceManager = safe_import("nethical.core.governance", "PersistenceManager")
JudgmentResult = safe_import("nethical.core.governance", "JudgmentResult")
Decision = safe_import("nethical.core.governance", "Decision")


def setup_demo_database(db_path: str):
    """
    Create a demo database with sample actions for replay.

    Simulates a week of agent activity with various action types.
    """
    print("🔧 Setting up demo database with historical actions...")

    persistence = PersistenceManager(db_path, retention_days=365)

    # Simulate a week of actions (2024-01-15 to 2024-01-21)
    base_time = datetime(2024, 1, 15, 9, 0, 0)

    # Different types of actions to simulate
    action_templates = [
        ("Query customer database", ActionType.QUERY, 0.2),
        ("Update user profile", ActionType.DATA_ACCESS, 0.3),
        ("Execute sudo command", ActionType.SYSTEM_COMMAND, 0.8),
        ("Call external API", ActionType.EXTERNAL_API, 0.4),
        ("Delete old records", ActionType.SYSTEM_COMMAND, 0.6),
        ("Read sensitive data", ActionType.DATA_ACCESS, 0.5),
        ("Generate report", ActionType.QUERY, 0.1),
        ("Modify configuration", ActionType.SYSTEM_COMMAND, 0.7),
    ]

    agents = ["agent_alice", "agent_bob", "agent_charlie", "agent_diana"]

    action_count = 0
    for day in range(7):  # 7 days
        for hour in range(9, 18):  # Work hours 9am-6pm
            for _ in range(5):  # 5 actions per hour
                template = action_templates[action_count % len(action_templates)]
                agent = agents[action_count % len(agents)]

                timestamp = base_time + timedelta(
                    days=day, hours=hour, minutes=action_count % 60
                )

                action = AgentAction(
                    action_id=f"action_{action_count}",
                    agent_id=agent,
                    action_type=template[1],
                    content=template[0],
                    timestamp=timestamp,
                    intent=f"Business task {action_count}",
                    risk_score=template[2],
                )

                persistence.store_action(action)

                # Store judgment based on risk score
                if template[2] >= 0.7:
                    decision = Decision.BLOCK
                elif template[2] >= 0.5:
                    decision = Decision.WARN
                else:
                    decision = Decision.ALLOW

                judgment = JudgmentResult(
                    judgment_id=f"judgment_{action_count}",
                    action_id=f"action_{action_count}",
                    decision=decision,
                    confidence=0.85,
                    reasoning=f"Risk score: {template[2]}",
                    violations=[],
                    timestamp=timestamp,
                    modifications={},
                    feedback={},
                    remediation_steps=[],
                    follow_up_required=False,
                )

                persistence.store_judgment(judgment)
                action_count += 1

    print(f"✅ Created {action_count} historical actions")
    return action_count


def demo_basic_query(replayer: ActionReplayer):
    """Demo 1: Basic querying of historical actions."""
    print("\n" + "=" * 70)
    print("📊 DEMO 1: Basic Action Querying")
    print("=" * 70)

    # Get overall statistics
    stats = replayer.get_statistics()
    print(f"\n📈 Action Stream Statistics:")
    print(f"  • Total actions: {stats['total_actions']}")
    print(f"  • First action: {stats['first_action_timestamp']}")
    print(f"  • Last action: {stats['last_action_timestamp']}")

    # Query specific agent's actions
    print(f"\n🔍 Querying actions for agent_alice...")
    alice_actions = replayer.get_actions(agent_ids=["agent_alice"], limit=10)
    print(f"  • Found {len(alice_actions)} actions")

    if alice_actions:
        print(f"\n  Sample action:")
        sample = alice_actions[0]
        print(f"    - ID: {sample['action_id']}")
        print(f"    - Type: {sample['action_type']}")
        print(f"    - Content: {sample['content']}")
        print(f"    - Timestamp: {sample['timestamp']}")


def demo_time_travel(replayer: ActionReplayer):
    """Demo 2: Time-travel debugging."""
    print("\n" + "=" * 70)
    print("⏰ DEMO 2: Time-Travel Debugging")
    print("=" * 70)

    # Set time-travel point to January 16, 2024
    travel_time = "2024-01-16T10:00:00"
    print(f"\n🚀 Time-traveling to: {travel_time}")
    replayer.set_timestamp(travel_time)

    # Get actions from that point
    actions = replayer.get_actions(limit=20)
    print(f"  • Retrieved {len(actions)} actions from that time period")

    if actions:
        print(f"\n  📅 First action after time-travel:")
        first = actions[0]
        print(f"    - Timestamp: {first['timestamp']}")
        print(f"    - Agent: {first['agent_id']}")
        print(f"    - Content: {first['content']}")

    # Set a time range
    print(f"\n📆 Setting time range: Jan 16-17, 2024")
    replayer.set_time_range("2024-01-16T00:00:00", "2024-01-17T23:59:59")

    range_actions = replayer.get_actions()
    print(f"  • Found {len(range_actions)} actions in this range")


def demo_policy_replay(replayer: ActionReplayer):
    """Demo 3: Replay with new policy."""
    print("\n" + "=" * 70)
    print("🔄 DEMO 3: Replay with New Policy")
    print("=" * 70)

    # Reset time filter for full replay
    replayer.set_time_range("2024-01-15T00:00:00", "2024-01-21T23:59:59")

    print(f"\n🎯 Replaying actions with 'strict_financial_v2.yaml' policy...")
    results = replayer.replay_with_policy(
        new_policy="strict_financial_v2.yaml", limit=50
    )

    print(f"  • Replayed {len(results)} actions")

    # Analyze decisions
    decisions = {}
    changed_count = 0
    for result in results:
        decision = result.new_decision
        decisions[decision] = decisions.get(decision, 0) + 1

        if result.original_decision and result.original_decision != result.new_decision:
            changed_count += 1

    print(f"\n  📊 Decision Breakdown:")
    for decision, count in sorted(decisions.items()):
        print(f"    - {decision}: {count} ({count/len(results)*100:.1f}%)")

    if len(results) > 0:
        print(
            f"\n  🔀 Decisions changed: {changed_count} ({changed_count/len(results)*100:.1f}%)"
        )
    else:
        print(f"\n  🔀 No actions to replay")

    # Show a sample changed decision
    if changed_count > 0:
        for result in results:
            if (
                result.original_decision
                and result.original_decision != result.new_decision
            ):
                print(f"\n  📝 Example of changed decision:")
                print(f"    - Action: {result.action_id}")
                print(f"    - Original: {result.original_decision}")
                print(f"    - New: {result.new_decision}")
                print(f"    - Reasoning: {result.reasoning}")
                break


def demo_policy_comparison(replayer: ActionReplayer):
    """Demo 4: Compare policies."""
    print("\n" + "=" * 70)
    print("⚖️  DEMO 4: Policy Comparison & What-If Analysis")
    print("=" * 70)

    print(f"\n🔬 Comparing baseline vs. strict policy...")
    comparison = replayer.compare_outcomes(
        baseline_policy="current",
        candidate_policy="strict_financial_v2.yaml",
        limit=100,
    )

    print(f"\n📊 Comparison Results:")
    print(f"  • Total actions analyzed: {comparison.total_actions}")
    print(f"  • Decisions changed: {comparison.decisions_changed}")
    print(f"  • Decisions same: {comparison.decisions_same}")
    print(
        f"  • Change rate: {comparison.decisions_changed/comparison.total_actions*100:.1f}%"
    )

    print(f"\n📈 Restrictiveness Analysis:")
    print(f"  • More restrictive: {comparison.more_restrictive}")
    print(f"  • Less restrictive: {comparison.less_restrictive}")

    print(f"\n⚡ Performance:")
    print(f"  • Execution time: {comparison.execution_time_ms:.2f}ms")
    print(
        f"  • Throughput: {comparison.total_actions / (comparison.execution_time_ms/1000):.0f} actions/sec"
    )

    # Decision breakdown
    print(f"\n🎯 Decision Breakdown:")
    for policy_name, decisions in comparison.decision_breakdown.items():
        print(f"\n  {policy_name}:")
        for decision, count in sorted(decisions.items()):
            print(
                f"    - {decision}: {count} ({count/comparison.total_actions*100:.1f}%)"
            )

    # Show sample of changed actions
    if comparison.changed_actions:
        print(f"\n📝 Sample of Changed Actions:")
        for changed in comparison.changed_actions[:3]:
            print(f"\n  • Action {changed['action_id']}:")
            print(f"    Agent: {changed['agent_id']}")
            print(f"    Content: {changed['content_preview']}...")
            print(
                f"    Baseline → Candidate: {changed['baseline_decision']} → {changed['candidate_decision']}"
            )


def demo_validation_workflow(replayer: ActionReplayer):
    """Demo 5: Pre-deployment validation workflow."""
    print("\n" + "=" * 70)
    print("✅ DEMO 5: Pre-Deployment Policy Validation Workflow")
    print("=" * 70)

    print(f"\n🔍 Validating new policy before deployment...")

    # Step 1: Replay with candidate policy
    print(f"\n  Step 1: Replay historical actions with candidate policy")
    results = replayer.replay_with_policy(new_policy="candidate_policy.yaml", limit=200)
    print(f"    ✓ Replayed {len(results)} actions")

    # Step 2: Compare with current policy
    print(f"\n  Step 2: Compare candidate vs. current policy")
    comparison = replayer.compare_outcomes(
        baseline_policy="current", candidate_policy="candidate_policy.yaml", limit=200
    )

    # Step 3: Analyze impact
    print(f"\n  Step 3: Analyze impact")
    change_rate = comparison.decisions_changed / comparison.total_actions * 100
    print(f"    • Change rate: {change_rate:.1f}%")

    # Decision: Should we deploy?
    print(f"\n  Step 4: Deployment decision")
    if change_rate < 5:
        print(f"    ✅ APPROVED: Low impact, safe to deploy")
    elif change_rate < 20:
        print(f"    ⚠️  CAUTION: Moderate impact, review changes carefully")
    else:
        print(f"    ❌ REJECTED: High impact, requires additional review")

    # Generate validation report
    print(f"\n  Step 5: Generate validation report")
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "candidate_policy": "candidate_policy.yaml",
        "actions_analyzed": comparison.total_actions,
        "change_rate": f"{change_rate:.1f}%",
        "more_restrictive": comparison.more_restrictive,
        "less_restrictive": comparison.less_restrictive,
        "execution_time_ms": comparison.execution_time_ms,
        "decision_breakdown": comparison.decision_breakdown,
    }

    print(f"    ✓ Validation report generated")
    print(f"\n  📄 Sample Report (JSON):")
    print(json.dumps(report, indent=2)[:500] + "...")


def main():
    """Run all demonstration scenarios."""
    print_header("F5: Simulation & Replay System Demo")
    print_info("This demo showcases time-travel debugging and what-if analysis")
    print_info("capabilities for AI agent governance.\n")

    # Check if F5 features are available
    if not ActionReplayer:
        print_feature_not_implemented("Simulation & Replay", "F5 Track")
        print_key_features(
            [
                "Action streams enable time-travel debugging",
                "Policies can be tested on historical data before deployment",
                "What-if analysis helps predict policy impact",
                "Validation workflow ensures safe policy rollouts",
                "High performance: >100 actions/second replay speed",
            ]
        )
        print_next_steps(
            [
                "Review F5 feature specifications",
                "Plan action stream infrastructure",
                "Design replay workflows",
                "Prepare test policies",
            ]
        )
        return

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo_action_streams.db")

        try:
            # Setup demo data
            action_count = setup_demo_database(db_path)

            # Initialize replayer with the same database
            replayer = ActionReplayer(storage_path=db_path)

            # Run demos
            run_demo_safely(lambda: demo_basic_query(replayer), "Basic Query")
            run_demo_safely(lambda: demo_time_travel(replayer), "Time Travel")
            run_demo_safely(lambda: demo_policy_replay(replayer), "Policy Replay")
            run_demo_safely(
                lambda: demo_policy_comparison(replayer), "Policy Comparison"
            )
            run_demo_safely(
                lambda: demo_validation_workflow(replayer), "Validation Workflow"
            )

            print_header("Demo Complete!")

            print_key_features(
                [
                    "Action streams enable time-travel debugging",
                    "Policies can be tested on historical data before deployment",
                    "What-if analysis helps predict policy impact",
                    "Validation workflow ensures safe policy rollouts",
                    "High performance: >100 actions/second replay speed",
                ]
            )

        except KeyboardInterrupt:
            print_warning("\nDemo interrupted by user")
        except Exception as e:
            print_error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
