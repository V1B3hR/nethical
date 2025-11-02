"""Example: Using the Explainability API for decision explanations.

This example demonstrates:
1. Explaining governance decisions
2. Generating decision trees
3. Creating transparency reports
"""

from nethical.api.explainability_api import ExplainabilityAPI


def main():
    """Run explainability API examples."""
    
    print("=" * 80)
    print("Nethical Phase 2: Explainability API Examples")
    print("=" * 80)
    
    # Initialize API
    api = ExplainabilityAPI()
    
    # Example 1: Explain a BLOCK decision
    print("\n1. Explaining a BLOCK decision...")
    judgment_data = {
        'decision': 'BLOCK',
        'violations': [
            {'type': 'unauthorized_data_access', 'severity': 'HIGH'},
            {'type': 'pii_exposure', 'severity': 'CRITICAL'}
        ],
        'risk_score': 0.92,
        'matched_rules': [
            {'id': 'data-protection-rule', 'priority': 100, 'decision': 'BLOCK'},
            {'id': 'pii-safety-rule', 'priority': 90, 'decision': 'RESTRICT'}
        ]
    }
    
    explain_result = api.explain_decision_endpoint('BLOCK', judgment_data)
    if explain_result['success']:
        data = explain_result['data']
        print(f"   Decision: {data['decision']}")
        print(f"   Primary reason: {data['primary_reason']}")
        print(f"   Natural language explanation:")
        print(f"   \"{data['natural_language']}\"")
        print(f"\n   Contributing factors:")
        for factor in data['contributing_factors']:
            print(f"     - {factor['name']}: {factor['value']}")
    
    # Example 2: Explain an ALLOW decision
    print("\n2. Explaining an ALLOW decision...")
    allow_judgment = {
        'decision': 'ALLOW',
        'violations': [],
        'risk_score': 0.15,
        'matched_rules': []
    }
    
    allow_result = api.explain_decision_endpoint('ALLOW', allow_judgment)
    if allow_result['success']:
        data = allow_result['data']
        print(f"   Decision: {data['decision']}")
        print(f"   Primary reason: {data['primary_reason']}")
        print(f"   Natural language explanation:")
        print(f"   \"{data['natural_language']}\"")
    
    # Example 3: Generate decision tree visualization
    print("\n3. Generating decision tree visualization...")
    tree_result = api.get_decision_tree_endpoint(judgment_data)
    if tree_result['success']:
        tree = tree_result['data']['tree']
        print(f"   Root node: {tree['name']}")
        print(f"   Final decision: {tree['decision']}")
        print(f"   Decision branches:")
        for child in tree['children']:
            print(f"     - {child['name']}: {child.get('value', 'N/A')}")
    
    # Example 4: Explain policy match
    print("\n4. Explaining why a policy rule matched...")
    matched_rule = {
        'id': 'gdpr-compliance-rule',
        'priority': 100,
        'decision': 'RESTRICT',
        'when': {'all': ['region == "EU"', 'data.contains_pii == true']}
    }
    facts = {
        'region': 'EU',
        'data': {'contains_pii': True, 'category': 'health'}
    }
    
    policy_result = api.explain_policy_match_endpoint(matched_rule, facts)
    if policy_result['success']:
        print(f"   Rule: {policy_result['data']['rule_id']}")
        print(f"   Explanation: {policy_result['data']['explanation']}")
    
    # Example 5: Generate transparency report
    print("\n5. Generating transparency report...")
    decisions = [
        {
            'decision': 'BLOCK',
            'violations': ['unauthorized_access'],
            'risk_score': 0.9
        },
        {
            'decision': 'ALLOW',
            'violations': [],
            'risk_score': 0.2
        },
        {
            'decision': 'RESTRICT',
            'violations': ['suspicious_pattern'],
            'risk_score': 0.6
        },
        {
            'decision': 'BLOCK',
            'violations': ['pii_exposure', 'data_breach'],
            'risk_score': 0.95
        }
    ]
    
    report_result = api.generate_transparency_report_endpoint(decisions, "last_hour")
    if report_result['success']:
        report = report_result['data']
        print(f"   Time period: {report['time_period']}")
        print(f"   Total decisions: {report['total_decisions']}")
        print(f"   Decision breakdown:")
        for decision, count in report['decision_breakdown'].items():
            percentage = (count / report['total_decisions']) * 100
            print(f"     - {decision}: {count} ({percentage:.1f}%)")
        print(f"   Total violations detected: {report['total_violations']}")
        print(f"\n   Sample explanations:")
        for i, exp in enumerate(report['sample_explanations'][:2], 1):
            print(f"     {i}. {exp['decision']}: {exp['reason']}")
    
    print("\n" + "=" * 80)
    print("Explainability API examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
