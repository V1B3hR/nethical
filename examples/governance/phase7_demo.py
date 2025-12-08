"""Phase 7 Demo: Anomaly & Drift Detection

Demonstrates:
- Sequence anomaly scoring (n-gram based)
- Distribution shift detection (PSI / KL divergence)
- Alert pipeline for drift events
- Behavioral anomaly detection
- Statistical drift monitoring
"""

import random
from nethical.core import AnomalyDriftMonitor, AnomalyType, DriftSeverity


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")


def demo_sequence_anomaly():
    """Demonstrate sequence anomaly detection."""
    print_section("1. Sequence Anomaly Detection")

    print("Initializing anomaly detector...\n")
    monitor = AnomalyDriftMonitor(
        sequence_n=3, storage_path="./demo_data/phase7_anomaly"
    )

    # Normal patterns
    print("1. Establishing baseline with normal sequences...")
    normal_sequences = [
        ["read", "process", "write"],
        ["read", "validate", "process"],
        ["fetch", "transform", "load"],
        ["query", "filter", "aggregate"],
    ]

    for i in range(30):
        agent_id = f"normal_agent_{i % 3}"
        sequence = random.choice(normal_sequences)

        for action_type in sequence:
            risk_score = random.uniform(0.1, 0.3)
            monitor.record_action(agent_id, action_type, risk_score)

    print(f"✓ Baseline established")
    stats = monitor.sequence_detector.get_statistics()
    print(f"  - Total n-grams observed: {stats['total_ngrams']}")
    print(f"  - Unique n-grams: {stats['unique_ngrams']}")

    # Anomalous pattern
    print("\n2. Detecting anomalous sequence...")
    anomalous_agent = "suspicious_agent"
    anomalous_sequence = ["delete", "exfiltrate", "cover_tracks"]

    print(f"   Agent '{anomalous_agent}' performing: {anomalous_sequence}")

    alert = None
    for action_type in anomalous_sequence:
        risk_score = random.uniform(0.5, 0.8)
        alert = monitor.record_action(
            anomalous_agent, action_type, risk_score, cohort="production"
        )

    if alert:
        print(f"\n   ⚠️  ALERT TRIGGERED!")
        print(f"   - Type: {alert.anomaly_type.value}")
        print(f"   - Severity: {alert.severity.value}")
        print(f"   - Anomaly Score: {alert.anomaly_score:.3f}")
        print(f"   - Description: {alert.description}")
        print(f"   - Evidence: {alert.evidence}")
        if alert.quarantine_recommended:
            print(f"   - Action: Quarantine recommended")
    else:
        print("   No alert (sequence seen before)")

    return monitor


def demo_behavioral_anomaly():
    """Demonstrate behavioral anomaly detection."""
    print_section("2. Behavioral Anomaly Detection")

    monitor = AnomalyDriftMonitor()

    print("Monitoring agent behavior patterns...\n")

    # Normal diverse behavior
    print("1. Normal agent with diverse actions...")
    normal_agent = "normal_agent"
    normal_actions = ["read", "write", "update", "query", "validate"]

    for i in range(30):
        action = random.choice(normal_actions)
        monitor.record_action(normal_agent, action, random.uniform(0.1, 0.3))

    alert = monitor.check_behavioral_anomaly(normal_agent)
    print(f"   Normal agent: {'✓ No alert' if not alert else '⚠️ Alert!'}")

    # Suspicious repetitive behavior
    print("\n2. Suspicious agent with repetitive pattern...")
    suspicious_agent = "repetitive_agent"

    for i in range(50):
        # 90% same action - suspicious!
        action = "probe_internal_network" if i < 45 else "normal_action"
        monitor.record_action(suspicious_agent, action, random.uniform(0.4, 0.6))

    alert = monitor.check_behavioral_anomaly(suspicious_agent, cohort="production")

    if alert:
        print(f"   ⚠️  BEHAVIORAL ANOMALY DETECTED!")
        print(f"   - Agent: {alert.agent_id}")
        print(f"   - Anomaly Score: {alert.anomaly_score:.3f}")
        print(f"   - Description: {alert.description}")
        print(f"   - Evidence:")
        for key, value in alert.evidence.items():
            print(f"     • {key}: {value}")

    return monitor


def demo_distribution_drift():
    """Demonstrate distribution drift detection."""
    print_section("3. Distribution Drift Detection")

    monitor = AnomalyDriftMonitor(psi_threshold=0.2, kl_threshold=0.1)

    # Set baseline distribution
    print("1. Establishing baseline risk distribution...")
    baseline_scores = [random.betavariate(2, 5) for _ in range(1000)]
    monitor.set_baseline_distribution(baseline_scores)

    print(f"✓ Baseline set with {len(baseline_scores)} samples")
    print(f"  - Mean: {sum(baseline_scores)/len(baseline_scores):.3f}")
    print(f"  - Min: {min(baseline_scores):.3f}")
    print(f"  - Max: {max(baseline_scores):.3f}")

    # Simulate normal operation
    print("\n2. Monitoring current distribution (normal)...")
    for _ in range(100):
        score = random.betavariate(2, 5)  # Same distribution
        monitor.drift_detector.add_score(score)

    alert = monitor.check_drift(cohort="production")
    if alert:
        print(f"   ⚠️ Drift detected: {alert.description}")
    else:
        print(f"   ✓ No drift - distribution stable")

    # Reset and simulate drift
    monitor.drift_detector.reset_current()

    print("\n3. Simulating distribution shift (drift scenario)...")
    print("   Current distribution shifting to higher risk...")

    for _ in range(100):
        # Shifted distribution - higher risk!
        score = random.betavariate(4, 3)  # Different distribution
        monitor.drift_detector.add_score(score)

    alert = monitor.check_drift(cohort="production")

    if alert:
        print(f"\n   ⚠️  DRIFT ALERT!")
        print(f"   - Type: {alert.anomaly_type.value}")
        print(f"   - Severity: {alert.severity.value}")
        print(f"   - Anomaly Score: {alert.anomaly_score:.3f}")
        print(f"   - Description: {alert.description}")
        print(f"   - Evidence:")
        for key, value in alert.evidence.items():
            print(f"     • {key}: {value}")
        print(f"   - Quarantine Recommended: {alert.quarantine_recommended}")

    return monitor


def demo_alert_pipeline():
    """Demonstrate complete alert pipeline."""
    print_section("4. Alert Pipeline & Monitoring")

    monitor = AnomalyDriftMonitor(storage_path="./demo_data/phase7_alerts")

    print("Simulating various anomaly scenarios...\n")

    # Set baseline for drift
    baseline = [random.uniform(0.2, 0.4) for _ in range(500)]
    monitor.set_baseline_distribution(baseline)

    # Scenario 1: Sequence anomaly
    print("1. Triggering sequence anomaly...")
    for action in ["unusual", "sequence", "detected"]:
        monitor.record_action("agent_001", action, 0.5)

    # Scenario 2: Behavioral anomaly
    print("2. Triggering behavioral anomaly...")
    for _ in range(20):
        monitor.record_action("agent_002", "repetitive_action", 0.6)
    monitor.check_behavioral_anomaly("agent_002")

    # Scenario 3: Distribution drift
    print("3. Triggering distribution drift...")
    for _ in range(100):
        monitor.drift_detector.add_score(random.uniform(0.7, 0.9))
    monitor.check_drift()

    # Show alerts
    print("\n4. Alert Summary:")
    stats = monitor.get_statistics()
    alert_stats = stats["alerts"]

    print(f"\n   Total Alerts: {alert_stats['total']}")
    print(f"\n   By Severity:")
    for severity, count in alert_stats["by_severity"].items():
        print(f"   - {severity}: {count}")

    print(f"\n   By Type:")
    for atype, count in alert_stats["by_type"].items():
        print(f"   - {atype}: {count}")

    # Recent alerts
    print("\n5. Recent Alerts (last 5):")
    recent_alerts = monitor.get_alerts(limit=5)

    for i, alert in enumerate(recent_alerts, 1):
        print(f"\n   Alert #{i}:")
        print(f"   - Type: {alert.anomaly_type.value}")
        print(f"   - Severity: {alert.severity.value}")
        print(f"   - Agent: {alert.agent_id or 'N/A'}")
        print(f"   - Score: {alert.anomaly_score:.3f}")
        print(f"   - Auto-escalated: {alert.auto_escalated}")

    # Export for analysis
    print("\n6. Exporting alerts...")
    alerts_export = monitor.export_alerts()
    print(f"✓ Exported {len(alerts_export)} alerts for analysis")

    return monitor


def demo_synthetic_drift_test():
    """Demonstrate synthetic drift test (exit criteria)."""
    print_section("5. Synthetic Drift Test (Exit Criteria)")

    print("Testing drift detection with synthetic data...\n")

    monitor = AnomalyDriftMonitor(psi_threshold=0.15, kl_threshold=0.08)

    # Baseline
    print("1. Creating baseline distribution...")
    baseline = [random.normalvariate(0.3, 0.1) for _ in range(1000)]
    baseline = [max(0, min(1, x)) for x in baseline]  # Clip to [0,1]
    monitor.set_baseline_distribution(baseline)
    print(f"✓ Baseline: mean={sum(baseline)/len(baseline):.3f}")

    # Synthetic drift scenarios
    drift_scenarios = [
        ("Moderate shift", 0.5, 0.15, True),
        ("Severe shift", 0.7, 0.2, True),
        ("No shift", 0.3, 0.1, False),
    ]

    caught_count = 0
    total_synthetic = sum(
        1 for _, _, _, should_detect in drift_scenarios if should_detect
    )

    print("\n2. Testing drift scenarios:")

    for scenario_name, mean, std, should_detect in drift_scenarios:
        monitor.drift_detector.reset_current()

        # Generate shifted distribution
        shifted = [random.normalvariate(mean, std) for _ in range(100)]
        shifted = [max(0, min(1, x)) for x in shifted]

        for score in shifted:
            monitor.drift_detector.add_score(score)

        alert = monitor.check_drift()
        detected = alert is not None

        status = "✓" if detected == should_detect else "✗"
        print(f"   {status} {scenario_name}: ", end="")

        if detected:
            print(
                f"Drift detected (PSI={alert.evidence['psi_score']:.3f}, KL={alert.evidence['kl_divergence']:.3f})"
            )
            if should_detect:
                caught_count += 1
        else:
            print("No drift detected")

    # Calculate detection rate
    detection_rate = (
        (caught_count / total_synthetic * 100) if total_synthetic > 0 else 0
    )

    print(f"\n3. Exit Criteria Check:")
    print(f"   - Synthetic drifts to detect: {total_synthetic}")
    print(f"   - Successfully caught: {caught_count}")
    print(f"   - Detection rate: {detection_rate:.0f}%")
    print(f"   - Target: 100%")
    print(f"   - Status: {'✓ PASSED' if detection_rate == 100 else '✗ FAILED'}")


if __name__ == "__main__":
    # Run all demos
    demo_sequence_anomaly()
    demo_behavioral_anomaly()
    demo_distribution_drift()
    demo_alert_pipeline()
    demo_synthetic_drift_test()

    print("\n" + "=" * 60)
    print("Phase 7 Demo Complete!")
    print("=" * 60)
    print("\nKey Capabilities:")
    print("✓ Sequence anomaly detection (n-gram based)")
    print("✓ Behavioral anomaly detection (repetitive patterns)")
    print("✓ Distribution drift detection (PSI & KL divergence)")
    print("✓ Automated alert pipeline")
    print("✓ Severity-based escalation")
    print("✓ Quarantine recommendations")
    print("\nNext Steps:")
    print("- Review alert thresholds")
    print("- Configure alert routing")
    print("- Set up automated responses")
    print("- Prepare for Phase 8 (Human-in-the-Loop)")
