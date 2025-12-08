"""Example: Using the Policy Formalization tools.

This example demonstrates:
1. Policy validation
2. Policy linting
3. Policy simulation
4. Policy impact analysis
"""

from nethical.core.policy_formalization import (
    PolicyValidator,
    PolicySimulator,
    PolicyImpactAnalyzer,
    PolicyGrammarEBNF,
    PolicyEngineType,
)


def main():
    """Run policy formalization examples."""

    print("=" * 80)
    print("Nethical Phase 2: Policy Formalization Examples")
    print("=" * 80)

    # Example 1: Get EBNF Grammar
    print("\n1. Retrieving EBNF grammar specification...")
    grammar = PolicyGrammarEBNF.get_grammar()
    print(f"   Grammar length: {len(grammar)} characters")
    print(f"   First 200 characters:")
    print(f"   {grammar[:200]}...")

    # Example 2: Validate a policy
    print("\n2. Validating a policy configuration...")
    validator = PolicyValidator(PolicyEngineType.REGION_AWARE)

    valid_policy = {
        "defaults": {"decision": "ALLOW", "deny_overrides": True},
        "rules": [
            {
                "id": "block-malicious-content",
                "enabled": True,
                "priority": 100,
                "when": {"any": ['content.type == "malicious"', "threat_score > 0.8"]},
                "action": {
                    "decision": "BLOCK",
                    "add_disclaimer": "Malicious content detected",
                    "escalate": True,
                    "tags": ["security", "malicious"],
                },
                "metadata": {
                    "description": "Block malicious content",
                    "owner": "security-team",
                },
            }
        ],
    }

    result = validator.validate_policy(valid_policy)
    print(f"   Valid: {result.valid}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")
    print(f"   Rule count: {result.metadata['rule_count']}")

    # Example 3: Lint a policy
    print("\n3. Linting policy for best practices...")
    linting_suggestions = validator.lint_policy(valid_policy)
    if linting_suggestions:
        print(f"   Suggestions:")
        for suggestion in linting_suggestions:
            print(f"     - {suggestion}")
    else:
        print(f"   No linting suggestions - policy follows best practices!")

    # Example 4: Validate an invalid policy
    print("\n4. Validating an invalid policy...")
    invalid_policy = {
        "rules": [
            {
                "id": "incomplete-rule"
                # Missing 'when' and 'action' - should fail validation
            }
        ]
    }

    invalid_result = validator.validate_policy(invalid_policy)
    print(f"   Valid: {invalid_result.valid}")
    print(f"   Errors found:")
    for error in invalid_result.errors[:3]:
        print(f"     - {error}")

    # Example 5: Simulate policy execution
    print("\n5. Simulating policy execution...")
    simulator = PolicySimulator()

    test_cases = [
        {"content": {"type": "safe"}, "threat_score": 0.1},
        {"content": {"type": "malicious"}, "threat_score": 0.9},
        {"content": {"type": "suspicious"}, "threat_score": 0.6},
        {"content": {"type": "malicious"}, "threat_score": 0.85},
        {"content": {"type": "safe"}, "threat_score": 0.2},
    ]

    simulation_results = simulator.simulate_policy(valid_policy, test_cases)
    print(f"   Total test cases: {simulation_results['total_cases']}")
    print(f"   Decision distribution:")
    for decision, percentage in simulation_results["decision_percentages"].items():
        print(f"     - {decision}: {percentage:.1f}%")
    print(f"   Most matched rules:")
    if simulation_results["rule_matches"]:
        for rule_id, count in list(simulation_results["rule_matches"].items())[:3]:
            print(f"     - {rule_id}: matched {count} times")

    # Example 6: Policy impact analysis
    print("\n6. Analyzing policy change impact...")

    current_policy = valid_policy.copy()

    # New policy with stricter threshold
    new_policy = {
        "defaults": {"decision": "ALLOW", "deny_overrides": True},
        "rules": [
            {
                "id": "block-malicious-content",
                "enabled": True,
                "priority": 100,
                "when": {
                    "any": [
                        'content.type == "malicious"',
                        "threat_score > 0.5",  # Lower threshold (stricter)
                    ]
                },
                "action": {
                    "decision": "BLOCK",
                    "add_disclaimer": "Malicious content detected",
                    "escalate": True,
                    "tags": ["security", "malicious"],
                },
            }
        ],
    }

    analyzer = PolicyImpactAnalyzer()
    impact = analyzer.analyze_impact(current_policy, new_policy, test_cases)

    print(f"   Affected rules: {len(impact.affected_rules)}")
    for rule in impact.affected_rules:
        print(f"     - {rule}")
    print(f"   Estimated block rate: {impact.estimated_block_rate:.2%}")
    print(f"   Estimated restrict rate: {impact.estimated_restrict_rate:.2%}")
    print(f"   Risk level: {impact.risk_level}")
    if impact.recommendations:
        print(f"   Recommendations:")
        for rec in impact.recommendations:
            print(f"     - {rec}")

    # Example 7: Check for duplicate rule IDs
    print("\n7. Detecting duplicate rule IDs...")
    duplicate_policy = {
        "rules": [
            {"id": "rule-1", "when": {}, "action": {"decision": "ALLOW"}},
            {"id": "rule-1", "when": {}, "action": {"decision": "BLOCK"}},
        ]
    }

    dup_result = validator.validate_policy(duplicate_policy)
    if not dup_result.valid:
        print(f"   Duplicate detected: {dup_result.errors[0]}")

    print("\n" + "=" * 80)
    print("Policy formalization examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
