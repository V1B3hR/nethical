"""
Comprehensive Explainability Demo

This example demonstrates all explainability features in Nethical:
- Basic reasoning
- Advanced LIME-like explanations
- SHAP-like feature importance
- Counterfactual explanations
- Decision path visualization

Run with: python examples/explainability_demo.py
"""

import json
from nethical.core.integrated_governance import IntegratedGovernance
from nethical.explainability.advanced_explainer import AdvancedExplainer


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_basic_explanation():
    """Demonstrate basic explanation features."""
    print_section("1. Basic Explanation")
    
    governance = IntegratedGovernance()
    
    # Test case 1: Safe action
    print("Test Case 1: Safe Action")
    print("-" * 80)
    result = governance.process_action(
        agent_id="demo_agent",
        action="Process customer analytics data"
    )
    
    print(f"Decision: {result['decision']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Violations: {len(result.get('violations', []))}")
    print()
    
    # Test case 2: Suspicious action
    print("\nTest Case 2: Action with PII")
    print("-" * 80)
    result = governance.process_action(
        agent_id="demo_agent",
        action="Send email to john@example.com with SSN 123-45-6789"
    )
    
    print(f"Decision: {result['decision']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Violations: {len(result.get('violations', []))}")
    if result.get('pii_detection'):
        pii = result['pii_detection']
        print(f"PII Detected: {pii['matches_count']} matches")
        print(f"PII Risk Score: {pii['pii_risk_score']:.2f}")


def demo_lime_explanation():
    """Demonstrate LIME-like local explanations."""
    print_section("2. LIME-like Local Explanations")
    
    governance = IntegratedGovernance()
    explainer = AdvancedExplainer()
    
    # Evaluate a complex action
    result = governance.process_action(
        agent_id="demo_agent",
        action="Access restricted database and export customer data"
    )
    
    # Generate local explanation
    explanation = explainer.explain_decision_with_features(
        decision=result['decision'],
        judgment_data=result,
        confidence=result.get('confidence', 1.0)
    )
    
    print(f"Decision: {explanation.decision}")
    print(f"Confidence: {explanation.confidence:.2%}")
    print(f"\nMost Influential Features:")
    for i, feature in enumerate(explanation.most_influential_features, 1):
        print(f"  {i}. {feature}")
    
    print("\n" + "-" * 80)
    print(explanation.explanation_text)


def demo_shap_values():
    """Demonstrate SHAP-like feature importance."""
    print_section("3. SHAP-like Feature Importance")
    
    governance = IntegratedGovernance()
    explainer = AdvancedExplainer()
    
    # Evaluate action
    result = governance.process_action(
        agent_id="demo_agent",
        action="Delete user records without authorization"
    )
    
    # Calculate SHAP values
    shap_values = explainer.calculate_shap_values(result)
    
    print("Feature Contributions (SHAP Values):")
    print("-" * 80)
    print(f"{'Feature':<20} {'SHAP Value':<15} {'Direction':<25}")
    print("-" * 80)
    
    for feature, value in sorted(shap_values.items(), 
                                 key=lambda x: abs(x[1]), 
                                 reverse=True):
        direction = "→ blocking" if value > 0 else "→ allowing"
        bar_length = int(abs(value) * 50)  # Scale for visualization
        bar_char = "▓" if value > 0 else "░"
        bar = bar_char * bar_length
        
        print(f"{feature:<20} {value:+.3f}          {direction:<15} {bar}")
    
    print("\nInterpretation:")
    print("  Positive values push towards BLOCK/RESTRICT")
    print("  Negative values push towards ALLOW")
    print("  Magnitude indicates strength of influence")


def demo_counterfactual():
    """Demonstrate counterfactual explanations."""
    print_section("4. Counterfactual Explanations")
    
    governance = IntegratedGovernance()
    explainer = AdvancedExplainer()
    
    # Evaluate action that will be blocked
    result = governance.process_action(
        agent_id="demo_agent",
        action="Execute system command: rm -rf /"
    )
    
    print(f"Current Decision: {result['decision']}")
    print()
    
    # Generate counterfactual
    counterfactual = explainer.generate_counterfactual(
        current_decision=result['decision'],
        judgment_data=result,
        desired_decision='ALLOW'
    )
    
    print("Counterfactual Analysis:")
    print("-" * 80)
    print(counterfactual.explanation_text)
    
    if counterfactual.required_changes:
        print("\n" + "-" * 80)
        print("Required Changes to Get Approval:")
        print("-" * 80)
        for i, change in enumerate(counterfactual.required_changes, 1):
            print(f"\n{i}. {change['description']}")
            print(f"   Feature: {change['feature']}")
            print(f"   Current Value: {change['current_value']}")
            print(f"   Required Value: {change['required_value']}")
            print(f"   Change Type: {change['change_type']}")
        
        print(f"\nMinimal Change: {'Yes' if counterfactual.minimal_change else 'No'}")


def demo_visualization():
    """Demonstrate decision path visualization."""
    print_section("5. Decision Path Visualization")
    
    governance = IntegratedGovernance()
    explainer = AdvancedExplainer()
    
    # Evaluate action
    result = governance.process_action(
        agent_id="demo_agent",
        action="Access confidential financial records"
    )
    
    # Generate visualization data
    viz_data = explainer.generate_decision_path_visualization(result)
    
    print("Decision Tree Structure:")
    print("-" * 80)
    print(json.dumps(viz_data, indent=2))
    
    print("\n" + "-" * 80)
    print("Feature Impact Summary:")
    print("-" * 80)
    
    for child in sorted(viz_data['children'], 
                       key=lambda x: abs(x['impact']), 
                       reverse=True):
        impact_percent = child['impact'] * 100
        print(f"{child['name']:<20} Impact: {impact_percent:+6.2f}%  "
              f"Color: {child['color']:<12} Value: {child['value']:.2f}")


def demo_complete_workflow():
    """Demonstrate complete explainability workflow."""
    print_section("6. Complete Explainability Workflow")
    
    governance = IntegratedGovernance()
    explainer = AdvancedExplainer()
    
    # Test action
    action = "Send marketing email to customer list including john.doe@example.com"
    
    print(f"Action: {action}")
    print("-" * 80)
    
    # 1. Evaluate
    result = governance.process_action(agent_id="demo_agent", action=action)
    
    # 2. Basic explanation
    print("\n[1] Basic Explanation:")
    print(f"  Decision: {result['decision']}")
    print(f"  Reasoning: {result['reasoning'][:100]}...")
    
    # 3. Advanced explanation
    explanation = explainer.explain_decision_with_features(
        decision=result['decision'],
        judgment_data=result
    )
    
    print("\n[2] Feature Importance (Top 3):")
    for feature in explanation.most_influential_features[:3]:
        f_obj = next(f for f in explanation.feature_importances 
                    if f.feature_name == feature)
        print(f"  • {feature}: {f_obj.importance_score:+.3f} - {f_obj.description}")
    
    # 4. SHAP values
    shap_values = explainer.calculate_shap_values(result)
    print("\n[3] SHAP Values (Top 3 by magnitude):")
    top_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    for feature, value in top_shap:
        print(f"  • {feature}: {value:+.3f}")
    
    # 5. Counterfactual (if needed)
    if result['decision'] != 'ALLOW':
        counterfactual = explainer.generate_counterfactual(
            current_decision=result['decision'],
            judgment_data=result
        )
        print("\n[4] Counterfactual:")
        print(f"  {counterfactual.explanation_text[:150]}...")
    
    # 6. Visualization
    viz_data = explainer.generate_decision_path_visualization(result)
    print(f"\n[5] Visualization Data: {len(viz_data['children'])} features")
    print(f"  Decision: {viz_data['decision']}")
    
    print("\n" + "=" * 80)
    print("  Complete explainability report generated!")
    print("=" * 80)


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "NETHICAL EXPLAINABILITY DEMO" + " " * 35 + "║")
    print("║" + " " * 78 + "║")
    print("║  Demonstrating advanced explainability features:" + " " * 28 + "║")
    print("║  • Basic reasoning" + " " * 58 + "║")
    print("║  • LIME-like local explanations" + " " * 45 + "║")
    print("║  • SHAP-like feature importance" + " " * 45 + "║")
    print("║  • Counterfactual explanations" + " " * 46 + "║")
    print("║  • Decision path visualization" + " " * 46 + "║")
    print("╚" + "═" * 78 + "╝")
    
    try:
        demo_basic_explanation()
        demo_lime_explanation()
        demo_shap_values()
        demo_counterfactual()
        demo_visualization()
        demo_complete_workflow()
        
        print("\n" + "=" * 80)
        print("  Demo completed successfully!")
        print("  See docs/ADVANCED_EXPLAINABILITY_GUIDE.md for more information.")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
