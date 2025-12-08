#!/usr/bin/env python3
"""Phase 8-9 Demo: Human-in-the-Loop & Continuous Optimization

This example demonstrates:
1. Phase 8: Escalation queue, human review, SLA tracking
2. Phase 9: Multi-objective optimization, promotion gates
3. Integration: Continuous improvement feedback loop
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.core import Phase89IntegratedGovernance, FeedbackTag


def main():
    """Main demo function."""
    print("=" * 70)
    print("Phase 8-9 Demo: Human-in-the-Loop & Continuous Optimization")
    print("=" * 70)

    # Initialize integrated governance
    print("\n1. Initializing Phase 8-9 Integrated Governance...")
    governance = Phase89IntegratedGovernance(
        storage_dir="./demo_data",
        triage_sla_seconds=3600,  # 1 hour
        resolution_sla_seconds=86400,  # 24 hours
        auto_escalate_on_block=True,
        auto_escalate_on_low_confidence=True,
        low_confidence_threshold=0.7,
    )
    print("✓ Governance initialized")

    # ==================== Phase 8: Human-in-the-Loop ====================
    print("\n" + "=" * 70)
    print("PHASE 8: HUMAN-IN-THE-LOOP OPERATIONS")
    print("=" * 70)

    # Simulate processing actions with escalation
    print("\n2. Processing Actions with Escalation...")

    test_cases = [
        {
            "judgment_id": "judg_001",
            "action_id": "act_001",
            "agent_id": "agent_alpha",
            "decision": "block",
            "confidence": 0.65,
            "violations": [
                {
                    "type": "safety",
                    "severity": 4,
                    "description": "Potential unsafe content",
                }
            ],
        },
        {
            "judgment_id": "judg_002",
            "action_id": "act_002",
            "agent_id": "agent_beta",
            "decision": "allow",
            "confidence": 0.55,
            "violations": [
                {
                    "type": "privacy",
                    "severity": 3,
                    "description": "Privacy concern detected",
                }
            ],
        },
        {
            "judgment_id": "judg_003",
            "action_id": "act_003",
            "agent_id": "agent_gamma",
            "decision": "terminate",
            "confidence": 0.45,
            "violations": [
                {
                    "type": "security",
                    "severity": 5,
                    "description": "Critical security violation",
                }
            ],
        },
    ]

    for i, case in enumerate(test_cases, 1):
        result = governance.process_with_escalation(**case)
        print(f"\nCase {i}:")
        print(f"  Decision: {result['decision']} (confidence: {case['confidence']})")
        print(f"  Escalated: {result['escalated']}")
        if result["escalated"]:
            print(f"  Priority: {result['priority']}")
            print(f"  Case ID: {result['escalation_case_id']}")

    # Human review workflow
    print("\n3. Human Review Workflow...")

    reviewer_id = "reviewer_alice"
    reviewed_count = 0

    while True:
        case = governance.get_next_case(reviewer_id=reviewer_id)
        if not case:
            break

        reviewed_count += 1
        print(f"\nReviewing Case {reviewed_count}:")
        print(f"  Case ID: {case.case_id}")
        print(f"  Agent: {case.agent_id}")
        print(f"  Decision: {case.decision} (confidence: {case.confidence})")
        print(f"  Violations: {len(case.violations)}")
        print(f"  Priority: {case.priority.name}")

        # Simulate human feedback
        # Case 1: False positive
        if reviewed_count == 1:
            feedback = governance.submit_feedback(
                case_id=case.case_id,
                reviewer_id=reviewer_id,
                feedback_tags=[FeedbackTag.FALSE_POSITIVE],
                rationale="Content was actually safe, detector was too aggressive on edge case",
                corrected_decision="allow",
                confidence=0.9,
            )
            print(f"  Feedback: FALSE_POSITIVE - '{feedback.rationale}'")

        # Case 2: Correct decision
        elif reviewed_count == 2:
            feedback = governance.submit_feedback(
                case_id=case.case_id,
                reviewer_id=reviewer_id,
                feedback_tags=[FeedbackTag.CORRECT_DECISION],
                rationale="Decision was appropriate given the context",
                confidence=0.95,
            )
            print(f"  Feedback: CORRECT_DECISION - '{feedback.rationale}'")

        # Case 3: Policy gap
        else:
            feedback = governance.submit_feedback(
                case_id=case.case_id,
                reviewer_id=reviewer_id,
                feedback_tags=[FeedbackTag.POLICY_GAP, FeedbackTag.EDGE_CASE],
                rationale="Existing policy doesn't clearly cover this scenario - needs update",
                confidence=0.85,
                metadata={
                    "suggested_policy": "Add explicit handling for cross-domain requests"
                },
            )
            print(f"  Feedback: POLICY_GAP, EDGE_CASE - '{feedback.rationale}'")

    # SLA Metrics
    print("\n4. SLA Metrics...")
    sla_metrics = governance.get_sla_metrics()
    print(f"  Total Cases: {sla_metrics.total_cases}")
    print(f"  Pending: {sla_metrics.pending_cases}")
    print(f"  Completed: {sla_metrics.completed_cases}")
    print(f"  Median Triage Time: {sla_metrics.median_triage_time_seconds:.2f}s")
    print(
        f"  Median Resolution Time: {sla_metrics.median_resolution_time_seconds:.2f}s"
    )
    print(f"  SLA Breaches: {sla_metrics.sla_breaches}")

    # Feedback Summary
    print("\n5. Feedback Summary (for Continuous Improvement)...")
    summary = governance.get_feedback_summary()
    print(f"  Total Feedback: {summary['total_feedback']}")
    print(f"  Correction Rate: {summary['correction_rate']:.1%}")
    print(f"  False Positive Rate: {summary['false_positive_rate']:.1%}")
    print(f"  Missed Violation Rate: {summary['missed_violation_rate']:.1%}")
    print(f"  Policy Gap Rate: {summary['policy_gap_rate']:.1%}")
    print(f"  Tag Counts: {summary['tag_counts']}")

    # ==================== Phase 9: Continuous Optimization ====================
    print("\n" + "=" * 70)
    print("PHASE 9: CONTINUOUS OPTIMIZATION")
    print("=" * 70)

    # Create baseline configuration
    print("\n6. Creating Baseline Configuration...")
    baseline_config = governance.create_configuration(
        config_version="baseline_v1.0",
        classifier_threshold=0.5,
        gray_zone_lower=0.4,
        gray_zone_upper=0.6,
    )
    print(f"  Config ID: {baseline_config.config_id}")
    print(f"  Version: {baseline_config.config_version}")
    print(f"  Classifier Threshold: {baseline_config.classifier_threshold}")
    print(
        f"  Gray Zone: [{baseline_config.gray_zone_lower}, {baseline_config.gray_zone_upper}]"
    )

    # Record baseline metrics
    baseline_metrics = governance.record_metrics(
        config_id=baseline_config.config_id,
        detection_recall=0.82,
        detection_precision=0.85,
        false_positive_rate=0.08,
        decision_latency_ms=12.0,
        human_agreement=0.86,
        total_cases=1000,
    )
    print(f"  Baseline Metrics:")
    print(f"    Recall: {baseline_metrics.detection_recall:.3f}")
    print(f"    Precision: {baseline_metrics.detection_precision:.3f}")
    print(f"    FP Rate: {baseline_metrics.false_positive_rate:.3f}")
    print(f"    Latency: {baseline_metrics.decision_latency_ms:.1f}ms")
    print(f"    Human Agreement: {baseline_metrics.human_agreement:.3f}")
    print(f"    Fitness Score: {baseline_metrics.fitness_score:.4f}")

    # Run optimization
    print("\n7. Running Random Search Optimization...")
    results = governance.optimize_configuration(
        technique="random_search",
        param_ranges={
            "classifier_threshold": (0.4, 0.7),
            "confidence_threshold": (0.6, 0.9),
            "gray_zone_lower": (0.3, 0.5),
            "gray_zone_upper": (0.5, 0.7),
        },
        n_iterations=20,
    )

    print(f"  Evaluated {len(results)} configurations")
    print(f"\n  Top 5 Configurations:")
    for i, (config, metrics) in enumerate(results[:5], 1):
        print(f"    {i}. {config.config_version}")
        print(f"       Fitness: {metrics.fitness_score:.4f}")
        print(
            f"       Recall: {metrics.detection_recall:.3f}, FP: {metrics.false_positive_rate:.3f}"
        )

    # Check promotion gate for best candidate
    print("\n8. Checking Promotion Gate...")
    best_config, best_metrics = results[0]

    passed, reasons = governance.check_promotion_gate(
        candidate_id=best_config.config_id, baseline_id=baseline_config.config_id
    )

    print(f"  Candidate: {best_config.config_version}")
    print(f"  Promotion Gate: {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"\n  Evaluation Details:")
    for reason in reasons:
        print(f"    {reason}")

    if passed:
        print("\n9. Promoting Configuration to Production...")
        success = governance.promote_configuration(best_config.config_id)
        if success:
            print(
                f"  ✓ Configuration {best_config.config_version} promoted to production"
            )
        else:
            print(f"  ✗ Failed to promote configuration")
    else:
        print("\n9. Configuration Not Promoted (gate criteria not met)")

    # ==================== Continuous Improvement Cycle ====================
    print("\n" + "=" * 70)
    print("CONTINUOUS IMPROVEMENT CYCLE")
    print("=" * 70)

    print("\n10. Running Continuous Improvement Cycle...")
    cycle_result = governance.continuous_improvement_cycle()

    print(f"  Human Agreement: {cycle_result['human_agreement']:.1%}")
    print(f"  Needs Optimization: {cycle_result['needs_optimization']}")
    print(f"\n  Recommendations:")
    if cycle_result["recommendations"]:
        for rec in cycle_result["recommendations"]:
            print(f"    - {rec}")
    else:
        print(f"    - No immediate actions required")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  ✓ Phase 8: Human review workflow with SLA tracking")
    print("  ✓ Phase 9: Multi-objective optimization with promotion gates")
    print("  ✓ Integration: Continuous improvement feedback loop")
    print("\nData stored in: ./demo_data/")
    print()


if __name__ == "__main__":
    main()
