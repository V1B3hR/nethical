#!/usr/bin/env python3
"""Phase 5-7 Demo - ML & Anomaly Detection Integration.

This demo showcases the unified Phase 5-7 integration:
1. Phase 5: ML Shadow Mode for passive model validation
2. Phase 6: ML Blended Risk for gray-zone decisions
3. Phase 7: Anomaly & Drift Detection for behavioral monitoring
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.core import Phase567IntegratedGovernance


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_shadow_mode(gov):
    """Demo ML shadow mode functionality."""
    print_section("1. Phase 5: ML Shadow Mode - Passive Prediction")
    
    print("Processing actions with shadow ML predictions...")
    
    test_cases = [
        {
            'agent_id': 'agent_1',
            'action_id': 'action_1',
            'action_type': 'response',
            'features': {'violation_count': 0.2, 'severity_max': 0.3},
            'rule_risk_score': 0.3,
            'rule_classification': 'allow'
        },
        {
            'agent_id': 'agent_2',
            'action_id': 'action_2',
            'action_type': 'system_command',
            'features': {'violation_count': 0.7, 'severity_max': 0.8},
            'rule_risk_score': 0.75,
            'rule_classification': 'deny'
        },
        {
            'agent_id': 'agent_3',
            'action_id': 'action_3',
            'action_type': 'response',
            'features': {'violation_count': 0.5, 'severity_max': 0.5},
            'rule_risk_score': 0.5,
            'rule_classification': 'warn'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        result = gov.process_action(**case)
        
        print(f"Test {i}:")
        print(f"  Rule: {result['rule_risk_score']:.2f} ({result['rule_classification']})")
        if 'shadow' in result:
            shadow = result['shadow']
            print(f"  ML Shadow: {shadow['ml_risk_score']:.2f} ({shadow['ml_classification']})")
            print(f"  Agreement: Scores={shadow['scores_agree']}, "
                  f"Classifications={shadow['classifications_agree']}")
        print()
    
    # Get shadow metrics
    metrics = gov.get_shadow_metrics()
    print("Shadow Mode Metrics:")
    print(f"  Total Predictions: {metrics.get('total_predictions', 0)}")
    print(f"  Agreement Rate: {metrics.get('classification_agreement_rate', 0)*100:.1f}%")
    print(f"  F1 Score: {metrics.get('f1_score', 0):.3f}")


def demo_blended_risk(gov):
    """Demo ML blended risk functionality."""
    print_section("2. Phase 6: ML Blended Risk - Gray Zone Assistance")
    
    print("Testing blended risk in different risk zones...")
    
    test_cases = [
        ('Clear Allow Zone', 0.2, 'allow'),
        ('Gray Zone - Low', 0.45, 'warn'),
        ('Gray Zone - High', 0.55, 'warn'),
        ('Clear Deny Zone', 0.8, 'deny')
    ]
    
    for label, rule_score, classification in test_cases:
        result = gov.process_action(
            agent_id='test_agent',
            action_id=f'action_{int(rule_score*100)}',
            action_type='response',
            features={'risk_indicator': rule_score},
            rule_risk_score=rule_score,
            rule_classification=classification
        )
        
        print(f"{label}:")
        print(f"  Rule Score: {rule_score:.2f}")
        
        if 'blended' in result:
            blended = result['blended']
            print(f"  Blended Score: {blended['blended_risk_score']:.2f}")
            print(f"  Risk Zone: {blended['risk_zone']}")
            print(f"  ML Influenced: {blended['ml_influenced']}")
            if blended['classification_changed']:
                print(f"  ⚠️ Classification Changed!")
        print()
    
    # Get blending metrics
    metrics = gov.get_blending_metrics()
    print("Blending Metrics:")
    print(f"  Total Decisions: {metrics.get('total_decisions', 0)}")
    print(f"  ML Influenced: {metrics.get('ml_influenced_count', 0)}")
    print(f"  ML Influence Rate: {metrics.get('ml_influence_rate', 0)*100:.1f}%")


def demo_anomaly_detection(gov):
    """Demo anomaly and drift detection functionality."""
    print_section("3. Phase 7: Anomaly & Drift Detection")
    
    print("Setting baseline distribution...")
    baseline_scores = [0.2, 0.3, 0.25, 0.35] * 25
    gov.set_baseline_distribution(baseline_scores, cohort="production")
    print(f"✓ Baseline set with {len(baseline_scores)} samples\n")
    
    print("Processing normal actions...")
    for i in range(5):
        result = gov.process_action(
            agent_id='normal_agent',
            action_id=f'normal_{i}',
            action_type='standard_operation',
            features={'indicator': 0.3},
            rule_risk_score=0.3,
            rule_classification='allow',
            cohort='production'
        )
    print("✓ No anomalies detected in normal behavior\n")
    
    print("Simulating anomalous behavior...")
    # Simulate repeated high-risk actions
    for i in range(10):
        result = gov.process_action(
            agent_id='suspicious_agent',
            action_id=f'suspicious_{i}',
            action_type='unusual_pattern',
            features={'indicator': 0.9},
            rule_risk_score=0.9,
            rule_classification='deny',
            cohort='production'
        )
    
    if result.get('anomaly_alert'):
        alert = result['anomaly_alert']
        print(f"⚠️ Anomaly Detected!")
        print(f"  Type: {alert['anomaly_type']}")
        print(f"  Severity: {alert['severity']}")
        print(f"  Score: {alert['anomaly_score']:.3f}")
        print(f"  Quarantine Recommended: {alert['quarantine_recommended']}")
        print(f"  Message: {alert['message']}")
    else:
        print("  (Anomaly threshold not yet reached)")
    
    # Get anomaly statistics
    stats = gov.get_anomaly_statistics()
    print("\nAnomaly Detection Statistics:")
    print(f"  Total Alerts: {stats['alerts']['total']}")
    print(f"  Tracked Agents: {stats.get('tracked_agents', 0)}")
    if stats['alerts']['total'] > 0:
        print(f"  By Type: {stats['alerts']['by_type']}")
        print(f"  By Severity: {stats['alerts']['by_severity']}")


def demo_system_status(gov):
    """Demo system status reporting."""
    print_section("4. System Status & Reporting")
    
    status = gov.get_system_status()
    
    print("Component Status:")
    for component, info in status['components'].items():
        enabled_icon = "✅" if info.get('enabled') else "❌"
        print(f"\n{enabled_icon} {component.replace('_', ' ').title()}")
        for key, value in info.items():
            if key != 'enabled':
                if isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")
    
    print("\n" + "-" * 70)
    print("\nExporting comprehensive report...")
    report = gov.export_phase567_report()
    
    # Print first part of report
    lines = report.split('\n')
    for line in lines[:30]:
        print(line)
    
    if len(lines) > 30:
        print(f"\n... ({len(lines) - 30} more lines)")


def main():
    """Run the complete Phase 5-7 demo."""
    print("\n" + "=" * 70)
    print("  Phase 5-7 Integration Demo")
    print("  ML Shadow Mode + Blended Risk + Anomaly Detection")
    print("=" * 70)
    
    # Initialize with all components enabled
    print("\nInitializing Phase 5-7 Integrated Governance...")
    gov = Phase567IntegratedGovernance(
        storage_dir="./demo_phase567_data",
        enable_shadow_mode=True,
        enable_ml_blending=True,
        enable_anomaly_detection=True,
        gray_zone_lower=0.4,
        gray_zone_upper=0.6,
        rule_weight=0.7,
        ml_weight=0.3
    )
    print("✓ Initialization complete\n")
    
    # Run demos
    demo_shadow_mode(gov)
    demo_blended_risk(gov)
    demo_anomaly_detection(gov)
    demo_system_status(gov)
    
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Phase 5 provides passive ML validation without enforcement risk")
    print("  • Phase 6 enables controlled ML influence in uncertain decisions")
    print("  • Phase 7 monitors for behavioral anomalies and distribution drift")
    print("  • All phases integrate seamlessly through unified governance API")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError running demo: {e}")
        import traceback
        traceback.print_exc()
