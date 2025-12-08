#!/usr/bin/env python3
"""
Phase 3 Feature Demonstration

This script demonstrates all the Phase 3 features:
- Risk Engine with multi-factor scoring and decay
- Correlation Engine for multi-agent pattern detection
- Fairness Sampler with stratified sampling
- Ethical Drift Reporter with cohort analysis
- Performance Optimizer with risk-based gating
- Integrated Governance combining all components
"""

import time
from datetime import datetime, timedelta

from nethical.core import Phase3IntegratedGovernance, DetectorTier, SamplingStrategy


class MockAction:
    """Mock action for demonstration."""

    def __init__(self, content, action_id=None):
        self.content = content
        self.action_id = action_id or f"action_{int(time.time() * 1000)}"
        self.metadata = {}


def demo_risk_engine():
    """Demonstrate risk engine features."""
    print("\n" + "=" * 70)
    print("DEMO 1: Risk Engine - Multi-Factor Scoring & Decay")
    print("=" * 70)

    from nethical.core import RiskEngine, RiskTier

    engine = RiskEngine(decay_half_life_hours=1.0)

    print("\n1. Calculate risk scores for different agents:")

    # Low risk agent
    score1 = engine.calculate_risk_score("agent_low", 0.2, {})
    print(
        f"   Agent 'agent_low': Risk Score = {score1:.3f}, Tier = {engine.get_tier('agent_low').value}"
    )

    # Medium risk agent
    for _ in range(2):
        score2 = engine.calculate_risk_score("agent_medium", 0.5, {})
    print(
        f"   Agent 'agent_medium': Risk Score = {score2:.3f}, Tier = {engine.get_tier('agent_medium').value}"
    )

    # High risk agent
    for _ in range(3):
        score3 = engine.calculate_risk_score("agent_high", 0.8, {})
    print(
        f"   Agent 'agent_high': Risk Score = {score3:.3f}, Tier = {engine.get_tier('agent_high').value}"
    )

    print(f"\n2. Elevated tier triggers:")
    print(
        f"   agent_low should invoke advanced detectors? {engine.should_invoke_advanced_detectors('agent_low')}"
    )
    print(
        f"   agent_high should invoke advanced detectors? {engine.should_invoke_advanced_detectors('agent_high')}"
    )

    print(f"\n3. Risk tier transitions:")
    for tier in [RiskTier.LOW, RiskTier.NORMAL, RiskTier.HIGH, RiskTier.ELEVATED]:
        print(
            f"   {tier.value.upper()}: score >= {RiskTier.from_score(0.1 if tier == RiskTier.LOW else 0.3 if tier == RiskTier.NORMAL else 0.6 if tier == RiskTier.HIGH else 0.8)}"
        )


def demo_correlation_engine():
    """Demonstrate correlation engine features."""
    print("\n" + "=" * 70)
    print("DEMO 2: Correlation Engine - Multi-Agent Pattern Detection")
    print("=" * 70)

    from nethical.core import CorrelationEngine

    engine = CorrelationEngine()

    print("\n1. Tracking multiple agents for pattern detection:")

    # Simulate escalating probes from multiple agents
    for i in range(5):
        for j in range(i + 1):
            action = MockAction(f"probe_{j}")
            matches = engine.track_action(f"agent_{i}", action, f"payload_{j}")

    print(f"   Tracked {len(engine.agent_windows)} agents")

    print("\n2. Detected correlation patterns:")
    patterns = engine._check_all_patterns()
    if patterns:
        for pattern in patterns:
            print(f"   ✓ {pattern.pattern_name}: {pattern.description}")
            print(
                f"     Severity: {pattern.severity}, Confidence: {pattern.confidence:.2f}"
            )
    else:
        print("   No patterns detected (need more activity)")

    print("\n3. Payload entropy calculation:")
    low_entropy = engine._calculate_entropy("aaaaaaaaaa")
    high_entropy = engine._calculate_entropy("a1b2c3d4e5")
    print(f"   Low entropy text: {low_entropy:.2f} bits")
    print(f"   High entropy text: {high_entropy:.2f} bits")


def demo_fairness_sampling():
    """Demonstrate fairness sampling features."""
    print("\n" + "=" * 70)
    print("DEMO 3: Fairness Sampler - Stratified Sampling")
    print("=" * 70)

    from nethical.core import FairnessSampler

    sampler = FairnessSampler(storage_dir="/tmp/fairness_demo")

    print("\n1. Create sampling job:")
    job_id = sampler.create_sampling_job(
        cohorts=["cohort_a", "cohort_b", "cohort_c"],
        target_sample_size=100,
        strategy=SamplingStrategy.STRATIFIED,
    )
    print(f"   Created job: {job_id}")

    print("\n2. Assign agents to cohorts:")
    for i in range(30):
        cohort = f"cohort_{chr(97 + i % 3)}"  # a, b, c
        sampler.assign_agent_cohort(f"agent_{i}", cohort)
    print(f"   Assigned 30 agents to 3 cohorts")

    print("\n3. Perform stratified sampling:")
    population = {
        "cohort_a": [
            {"agent_id": f"agent_{i}", "action_id": f"action_{i}"} for i in range(15)
        ],
        "cohort_b": [
            {"agent_id": f"agent_{i}", "action_id": f"action_{i}"} for i in range(10)
        ],
        "cohort_c": [
            {"agent_id": f"agent_{i}", "action_id": f"action_{i}"} for i in range(5)
        ],
    }

    collected = sampler.perform_stratified_sampling(job_id, population)
    print(f"   Collected {collected} samples")

    print("\n4. Coverage statistics:")
    stats = sampler.get_coverage_stats(job_id)
    for cohort, coverage in stats.get("cohort_coverage", {}).items():
        print(
            f"   {cohort}: {coverage['count']} samples ({coverage['percentage']:.1f}%)"
        )


def demo_ethical_drift():
    """Demonstrate ethical drift reporting."""
    print("\n" + "=" * 70)
    print("DEMO 4: Ethical Drift Reporter - Cohort Analysis")
    print("=" * 70)

    from nethical.core import EthicalDriftReporter

    reporter = EthicalDriftReporter(report_dir="/tmp/drift_demo")

    print("\n1. Track violations across cohorts:")

    # Cohort A: Low violations
    for i in range(20):
        reporter.track_action(f"agent_a_{i}", "cohort_a", 0.2)
        if i % 5 == 0:
            reporter.track_violation(f"agent_a_{i}", "cohort_a", "safety", "low")

    # Cohort B: High violations (drift)
    for i in range(20):
        reporter.track_action(f"agent_b_{i}", "cohort_b", 0.7)
        if i % 2 == 0:
            reporter.track_violation(f"agent_b_{i}", "cohort_b", "safety", "high")

    print(
        f"   Tracked {sum(p.action_count for p in reporter.cohort_profiles.values())} actions"
    )
    print(f"   Across {len(reporter.cohort_profiles)} cohorts")

    print("\n2. Generate drift report:")
    report = reporter.generate_report(
        start_time=datetime.utcnow() - timedelta(hours=1), end_time=datetime.utcnow()
    )

    print(f"   Report ID: {report.report_id}")
    print(f"   Drift detected: {report.drift_metrics.get('has_drift', False)}")

    print("\n3. Recommendations:")
    for i, rec in enumerate(report.recommendations[:3], 1):
        print(f"   {i}. {rec}")

    print("\n4. Dashboard data:")
    dashboard = reporter.get_dashboard_data()
    for cohort, summary in dashboard["cohort_summary"].items():
        print(
            f"   {cohort}: {summary['violation_count']} violations, "
            f"avg risk = {summary['avg_risk_score']:.2f}"
        )


def demo_performance_optimizer():
    """Demonstrate performance optimizer."""
    print("\n" + "=" * 70)
    print("DEMO 5: Performance Optimizer - Risk-Based Gating")
    print("=" * 70)

    from nethical.core import PerformanceOptimizer, DetectorTier

    optimizer = PerformanceOptimizer(target_cpu_reduction_pct=30.0)

    print("\n1. Register detectors at different tiers:")
    detectors = [
        ("fast_detector", DetectorTier.FAST),
        ("standard_detector", DetectorTier.STANDARD),
        ("advanced_detector", DetectorTier.ADVANCED),
        ("premium_detector", DetectorTier.PREMIUM),
    ]

    for name, tier in detectors:
        optimizer.register_detector(name, tier)
        print(f"   Registered: {name} ({tier.value})")

    print("\n2. Risk-based gating decisions:")
    risk_scores = [0.1, 0.3, 0.6, 0.8]

    for risk in risk_scores:
        print(f"\n   At risk score {risk:.1f}:")
        for name, tier in detectors:
            should_invoke = optimizer.should_invoke_detector(name, risk)
            status = "✓ INVOKE" if should_invoke else "✗ SKIP"
            print(f"     {name}: {status}")

    print("\n3. Track performance:")
    # Simulate baseline
    for i in range(100):
        optimizer.track_action_processing(100.0, 5)

    # Simulate improvement with gating
    for i in range(100):
        optimizer.track_action_processing(70.0, 3)

    reduction = optimizer.get_cpu_reduction_pct()
    print(f"   CPU reduction: {reduction:.1f}%")
    print(f"   Meeting target (30%): {optimizer.is_meeting_target()}")

    print("\n4. Optimization suggestions:")
    for i, suggestion in enumerate(optimizer.suggest_optimizations()[:2], 1):
        print(f"   {i}. {suggestion}")


def demo_integrated_governance():
    """Demonstrate integrated governance."""
    print("\n" + "=" * 70)
    print("DEMO 6: Integrated Governance - All Features Combined")
    print("=" * 70)

    governance = Phase3IntegratedGovernance(
        storage_dir="/tmp/nethical_demo", enable_performance_optimization=True
    )

    print("\n1. Process actions with integrated analysis:")

    # Simulate different agents with varying risk profiles
    scenarios = [
        ("low_risk_agent", "cohort_safe", False, None, None),
        ("medium_risk_agent", "cohort_moderate", True, "privacy", "medium"),
        ("high_risk_agent", "cohort_risky", True, "safety", "high"),
    ]

    for agent_id, cohort, violation, vtype, vsev in scenarios:
        action = MockAction(f"action from {agent_id}")

        results = governance.process_action(
            agent_id=agent_id,
            action=action,
            cohort=cohort,
            violation_detected=violation,
            violation_type=vtype,
            violation_severity=vsev,
            detector_invocations={"detector1": 10.0, "detector2": 5.0},
        )

        print(f"\n   {agent_id}:")
        print(f"     Risk Score: {results['risk_score']:.3f}")
        print(f"     Risk Tier: {results['risk_tier']}")
        print(f"     Invoke Advanced: {results['invoke_advanced_detectors']}")

    print("\n2. System status:")
    status = governance.get_system_status()
    for component, info in status["components"].items():
        enabled = "✓" if info["enabled"] else "✗"
        print(f"   {enabled} {component}: {info}")

    print("\n3. Generate integrated reports:")

    # Drift report
    drift_report = governance.generate_drift_report(days_back=1)
    print(f"   Drift Report: {drift_report['report_id']}")
    print(f"     Cohorts analyzed: {len(drift_report['cohorts'])}")
    print(
        f"     Drift detected: {drift_report['drift_metrics'].get('has_drift', False)}"
    )

    # Performance report
    perf_report = governance.get_performance_report()
    print(f"\n   Performance Report:")
    print(f"     Total actions: {perf_report['action_metrics']['total_actions']}")
    print(
        f"     CPU reduction: {perf_report['optimization']['current_cpu_reduction_pct']:.1f}%"
    )


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("NETHICAL PHASE 3 - FEATURE DEMONSTRATION")
    print("Correlation & Adaptive Risk (Week 8–12)")
    print("=" * 70)

    demo_risk_engine()
    demo_correlation_engine()
    demo_fairness_sampling()
    demo_ethical_drift()
    demo_performance_optimizer()
    demo_integrated_governance()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nAll Phase 3 features demonstrated successfully!")
    print("\nKey achievements:")
    print("  ✓ Risk Engine with multi-factor scoring and decay")
    print("  ✓ Correlation Engine for multi-agent pattern detection")
    print("  ✓ Fairness Sampler with stratified sampling")
    print("  ✓ Ethical Drift Reporter with cohort analysis")
    print("  ✓ Performance Optimizer with >30% CPU reduction")
    print("  ✓ Integrated Governance combining all components")
    print()


if __name__ == "__main__":
    main()
