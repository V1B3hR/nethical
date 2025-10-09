"""Demo script for F4: Thresholds, Tuning & Adaptivity features.

This script demonstrates:
1. Bayesian optimization for threshold tuning
2. Adaptive threshold adjustment based on outcomes
3. Agent-specific threshold profiles
4. A/B testing with statistical significance
5. Gradual rollout and rollback
"""

import tempfile
import time
from pathlib import Path

from nethical.core import (
    Phase89IntegratedGovernance,
    PerformanceMetrics,
)


def demo_bayesian_optimization():
    """Demonstrate Bayesian optimization for parameter tuning."""
    print("\n" + "="*70)
    print("1. BAYESIAN OPTIMIZATION FOR THRESHOLD TUNING")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        governance = Phase89IntegratedGovernance(storage_dir=tmpdir)
        
        print("\nRunning Bayesian optimization with 15 iterations...")
        print("Parameter ranges:")
        print("  - classifier_threshold: 0.4 to 0.7")
        print("  - confidence_threshold: 0.6 to 0.9")
        print("  - gray_zone_lower: 0.2 to 0.5")
        print("  - gray_zone_upper: 0.5 to 0.8")
        
        results = governance.optimize_configuration(
            technique="bayesian",
            param_ranges={
                'classifier_threshold': (0.4, 0.7),
                'confidence_threshold': (0.6, 0.9),
                'gray_zone_lower': (0.2, 0.5),
                'gray_zone_upper': (0.5, 0.8)
            },
            n_iterations=15,
            n_initial_random=3
        )
        
        print(f"\n✓ Optimization complete! Evaluated {len(results)} configurations")
        print("\nTop 3 configurations by fitness score:")
        for i, (config, metrics) in enumerate(results[:3], 1):
            print(f"\n  {i}. Config {config.config_version}")
            print(f"     Fitness: {metrics.fitness_score:.4f}")
            print(f"     Classifier threshold: {config.classifier_threshold:.3f}")
            print(f"     Confidence threshold: {config.confidence_threshold:.3f}")
            print(f"     Recall: {metrics.detection_recall:.1%}")
            print(f"     FP rate: {metrics.false_positive_rate:.1%}")


def demo_adaptive_threshold_tuner():
    """Demonstrate adaptive threshold tuning based on outcomes."""
    print("\n" + "="*70)
    print("2. ADAPTIVE THRESHOLD TUNING")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        governance = Phase89IntegratedGovernance(storage_dir=tmpdir)
        
        print("\nInitial thresholds:")
        initial_thresholds = governance.get_adaptive_thresholds()
        print(f"  Classifier: {initial_thresholds['classifier_threshold']:.3f}")
        print(f"  Confidence: {initial_thresholds['confidence_threshold']:.3f}")
        
        # Simulate some outcomes
        print("\n\nSimulating outcome feedback...")
        
        # False positive - should increase threshold
        print("\n1. Recording false positive (should increase threshold)")
        result1 = governance.record_outcome(
            action_id="act_001",
            judgment_id="judg_001",
            predicted_outcome="block",
            actual_outcome="false_positive",
            confidence=0.75,
            human_feedback="This was safe, shouldn't have blocked"
        )
        
        print(f"   → Classifier threshold: {initial_thresholds['classifier_threshold']:.3f} → "
              f"{result1['updated_thresholds']['classifier_threshold']:.3f}")
        
        # False negative - should decrease threshold
        print("\n2. Recording false negative (should decrease threshold)")
        result2 = governance.record_outcome(
            action_id="act_002",
            judgment_id="judg_002",
            predicted_outcome="allow",
            actual_outcome="false_negative",
            confidence=0.60,
            human_feedback="This was risky, should have been blocked"
        )
        
        print(f"   → Classifier threshold: {result1['updated_thresholds']['classifier_threshold']:.3f} → "
              f"{result2['updated_thresholds']['classifier_threshold']:.3f}")
        
        # True positives - correct decisions
        print("\n3. Recording correct decisions...")
        for i in range(5):
            governance.record_outcome(
                action_id=f"act_{100+i}",
                judgment_id=f"judg_{100+i}",
                predicted_outcome="block",
                actual_outcome="correct",
                confidence=0.85
            )
        
        # Get performance statistics
        print("\n\nPerformance Statistics:")
        stats = governance.get_tuning_performance()
        print(f"  Total outcomes: {stats['total_outcomes']}")
        print(f"  Accuracy: {stats['accuracy']:.1%}")
        print(f"  Precision: {stats['precision']:.1%}")
        print(f"  Recall: {stats['recall']:.1%}")
        print(f"  FP rate: {stats['false_positive_rate']:.1%}")
        print(f"  FN rate: {stats['false_negative_rate']:.1%}")


def demo_agent_specific_profiles():
    """Demonstrate agent-specific threshold profiles."""
    print("\n" + "="*70)
    print("3. AGENT-SPECIFIC THRESHOLD PROFILES")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        governance = Phase89IntegratedGovernance(storage_dir=tmpdir)
        
        print("\nSetting up threshold profiles for different agent types:")
        
        # High-risk agent (financial operations)
        print("\n1. Financial Bot (high-risk)")
        governance.set_agent_thresholds(
            agent_id="agent_financial",
            thresholds={
                'classifier_threshold': 0.35,  # More sensitive
                'confidence_threshold': 0.85,
                'gray_zone_lower': 0.3,
                'gray_zone_upper': 0.7
            }
        )
        financial_thresholds = governance.get_adaptive_thresholds("agent_financial")
        print(f"   Classifier: {financial_thresholds['classifier_threshold']:.2f} (more sensitive)")
        print(f"   Confidence: {financial_thresholds['confidence_threshold']:.2f} (higher required)")
        
        # Medium-risk agent (customer service)
        print("\n2. Customer Service Bot (medium-risk)")
        governance.set_agent_thresholds(
            agent_id="agent_customer_service",
            thresholds={
                'classifier_threshold': 0.50,  # Balanced
                'confidence_threshold': 0.75,
                'gray_zone_lower': 0.4,
                'gray_zone_upper': 0.6
            }
        )
        cs_thresholds = governance.get_adaptive_thresholds("agent_customer_service")
        print(f"   Classifier: {cs_thresholds['classifier_threshold']:.2f} (balanced)")
        print(f"   Confidence: {cs_thresholds['confidence_threshold']:.2f}")
        
        # Low-risk agent (information retrieval)
        print("\n3. Info Bot (low-risk)")
        governance.set_agent_thresholds(
            agent_id="agent_info",
            thresholds={
                'classifier_threshold': 0.65,  # Less sensitive
                'confidence_threshold': 0.70,
                'gray_zone_lower': 0.4,
                'gray_zone_upper': 0.6
            }
        )
        info_thresholds = governance.get_adaptive_thresholds("agent_info")
        print(f"   Classifier: {info_thresholds['classifier_threshold']:.2f} (less sensitive)")
        print(f"   Confidence: {info_thresholds['confidence_threshold']:.2f}")
        
        # Global default
        print("\n4. Global Default (fallback)")
        global_thresholds = governance.get_adaptive_thresholds()
        print(f"   Classifier: {global_thresholds['classifier_threshold']:.2f}")
        print(f"   Confidence: {global_thresholds['confidence_threshold']:.2f}")


def demo_ab_testing():
    """Demonstrate A/B testing framework."""
    print("\n" + "="*70)
    print("4. A/B TESTING FRAMEWORK")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        governance = Phase89IntegratedGovernance(storage_dir=tmpdir)
        
        # Create configurations
        print("\nCreating configurations...")
        baseline = governance.create_configuration(
            config_version="baseline_v2.0",
            classifier_threshold=0.5,
            confidence_threshold=0.7
        )
        print(f"✓ Baseline: threshold={baseline.classifier_threshold:.2f}")
        
        candidate = governance.create_configuration(
            config_version="candidate_v2.1",
            classifier_threshold=0.55,  # More conservative
            confidence_threshold=0.75
        )
        print(f"✓ Candidate: threshold={candidate.classifier_threshold:.2f}")
        
        # Start A/B test
        print("\nStarting A/B test with 10% traffic to candidate...")
        control_id, treatment_id = governance.create_ab_test(
            baseline,
            candidate,
            traffic_split=0.1
        )
        print(f"✓ Control variant: {control_id}")
        print(f"✓ Treatment variant: {treatment_id}")
        
        # Simulate data collection
        print("\n\nSimulating data collection...")
        
        # Baseline metrics (90% traffic)
        baseline_metrics = PerformanceMetrics(
            config_id=baseline.config_id,
            detection_recall=0.82,
            detection_precision=0.85,
            false_positive_rate=0.08,
            decision_latency_ms=12.0,
            human_agreement=0.86,
            total_cases=450
        )
        governance.record_ab_metrics(control_id, baseline_metrics)
        print(f"✓ Baseline: {baseline_metrics.total_cases} cases, "
              f"recall={baseline_metrics.detection_recall:.1%}")
        
        # Candidate metrics (10% traffic)
        candidate_metrics = PerformanceMetrics(
            config_id=candidate.config_id,
            detection_recall=0.88,  # 6% improvement
            detection_precision=0.89,
            false_positive_rate=0.05,  # 3% reduction
            decision_latency_ms=13.0,
            human_agreement=0.90,
            total_cases=200
        )
        governance.record_ab_metrics(treatment_id, candidate_metrics)
        print(f"✓ Candidate: {candidate_metrics.total_cases} cases, "
              f"recall={candidate_metrics.detection_recall:.1%}")
        
        # Check statistical significance
        print("\n\nChecking statistical significance...")
        is_sig, p_value, interpretation = governance.check_ab_significance(
            control_id,
            treatment_id,
            metric="detection_recall"
        )
        print(f"  {interpretation}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Result: {'✓ SIGNIFICANT' if is_sig else '✗ Not significant'}")
        
        if is_sig:
            # Gradual rollout
            print("\n\nPerforming gradual rollout...")
            for target in [0.2, 0.4, 0.6]:
                new_traffic = governance.gradual_rollout(
                    treatment_id,
                    target_traffic=target,
                    step_size=0.2
                )
                print(f"  → Traffic increased to {new_traffic:.0%}")
                time.sleep(0.1)  # Simulate time between steps
            
            print("\n✓ Rollout complete!")
        else:
            print("\n→ Continue collecting data before rollout")
        
        # Get summary
        print("\n\nA/B Test Summary:")
        summary = governance.get_ab_summary()
        for variant_id, info in summary['variants'].items():
            variant_type = "Control" if info['is_control'] else "Treatment"
            print(f"\n  {variant_type} ({variant_id}):")
            print(f"    Traffic: {info['traffic']:.1%}")
            if info['metrics']:
                print(f"    Recall: {info['metrics']['detection_recall']:.1%}")
                print(f"    FP Rate: {info['metrics']['false_positive_rate']:.1%}")


def demo_rollback():
    """Demonstrate rollback mechanism."""
    print("\n" + "="*70)
    print("5. ROLLBACK MECHANISM")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        governance = Phase89IntegratedGovernance(storage_dir=tmpdir)
        
        # Create test
        baseline = governance.create_configuration("baseline_v3.0")
        bad_candidate = governance.create_configuration(
            "bad_candidate_v3.1",
            classifier_threshold=0.3  # Too sensitive
        )
        
        control_id, treatment_id = governance.create_ab_test(
            baseline,
            bad_candidate,
            traffic_split=0.2
        )
        
        print(f"\nA/B test started:")
        print(f"  Control traffic: 80%")
        print(f"  Treatment traffic: 20%")
        
        # Simulate poor performance
        print("\n\nSimulating poor performance from treatment...")
        poor_metrics = PerformanceMetrics(
            config_id=bad_candidate.config_id,
            detection_recall=0.95,  # Too high
            false_positive_rate=0.25,  # Way too high!
            human_agreement=0.60,  # Low agreement
            total_cases=100
        )
        governance.record_ab_metrics(treatment_id, poor_metrics)
        
        print(f"  Treatment FP rate: {poor_metrics.false_positive_rate:.1%} ⚠️")
        print(f"  Treatment agreement: {poor_metrics.human_agreement:.1%} ⚠️")
        
        # Rollback
        print("\n\nPerformance degradation detected - rolling back...")
        success = governance.rollback_variant(treatment_id)
        
        if success:
            print("✓ Rollback successful!")
            print(f"  Control traffic: 100%")
            print(f"  Treatment traffic: 0%")
            print("\n  System returned to stable baseline configuration")
        else:
            print("✗ Rollback failed")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("F4: THRESHOLDS, TUNING & ADAPTIVITY - DEMO")
    print("="*70)
    print("\nThis demo showcases the key features of F4:")
    print("  1. Bayesian Optimization")
    print("  2. Adaptive Threshold Tuning")
    print("  3. Agent-Specific Profiles")
    print("  4. A/B Testing Framework")
    print("  5. Rollback Mechanism")
    
    input("\nPress Enter to start...")
    
    # Run demos
    demo_bayesian_optimization()
    input("\nPress Enter to continue...")
    
    demo_adaptive_threshold_tuner()
    input("\nPress Enter to continue...")
    
    demo_agent_specific_profiles()
    input("\nPress Enter to continue...")
    
    demo_ab_testing()
    input("\nPress Enter to continue...")
    
    demo_rollback()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nFor more information, see:")
    print("  - F4_GUIDE.md - Complete feature guide")
    print("  - tests/test_f4_adaptive_tuning.py - Test examples")
    print("  - roadmap.md - Implementation details")
    print("\n")


if __name__ == "__main__":
    main()
