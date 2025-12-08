"""Demo: Unified Integrated Governance

This demo shows how to use the consolidated IntegratedGovernance class
that brings together ALL phases (3, 4, 5-7, 8-9) into a single unified interface.
"""

from nethical.core import IntegratedGovernance


def main():
    print("=" * 80)
    print("Unified Integrated Governance Demo")
    print("=" * 80)
    print()

    # Initialize with all features enabled
    print("1. Initializing unified governance system...")
    gov = IntegratedGovernance(
        storage_dir="./demo_unified_data",
        # Phase 3
        enable_performance_optimization=True,
        # Phase 4
        enable_merkle_anchoring=True,
        enable_quarantine=True,
        enable_ethical_taxonomy=True,
        enable_sla_monitoring=True,
        # Phase 5-7
        enable_shadow_mode=True,
        enable_ml_blending=True,
        enable_anomaly_detection=True,
    )
    print("   ✓ All phases initialized successfully\n")

    # Check system status
    print("2. System Status:")
    status = gov.get_system_status()
    print(f"   Timestamp: {status['timestamp']}")
    print(
        f"   Components enabled: {sum(status['components_enabled'].values())}/{len(status['components_enabled'])}"
    )
    print()

    # Show phase-by-phase status
    print("   Phase 3 (Risk & Correlation):")
    print(
        f"      - Active risk profiles: {status['phase3']['risk_engine']['active_profiles']}"
    )
    print(
        f"      - Tracked agents: {status['phase3']['correlation_engine']['tracked_agents']}"
    )
    print()

    print("   Phase 4 (Audit & Taxonomy):")
    print(
        f"      - Merkle anchor enabled: {status['phase4']['merkle_anchor']['enabled']}"
    )
    print(
        f"      - Quarantine manager enabled: {status['phase4']['quarantine_manager']['enabled']}"
    )
    print()

    print("   Phase 5-7 (ML & Anomaly Detection):")
    print(
        f"      - Shadow classifier enabled: {status['phase567']['shadow_classifier']['enabled']}"
    )
    print(
        f"      - ML blending enabled: {status['phase567']['blended_engine']['enabled']}"
    )
    print(
        f"      - Anomaly monitoring enabled: {status['phase567']['anomaly_monitor']['enabled']}"
    )
    print()

    print("   Phase 8-9 (Human & Optimization):")
    print(
        f"      - Pending escalation cases: {status['phase89']['escalation_queue']['pending_cases']}"
    )
    print(
        f"      - Tracked configurations: {status['phase89']['optimizer']['tracked_configs']}"
    )
    print()

    # Process a basic action
    print("3. Processing basic action (without ML features)...")
    result1 = gov.process_action(
        agent_id="agent_001",
        action="Hello, how can I help you?",
        cohort="production",
        violation_detected=False,
    )
    print(f"   Risk score: {result1['phase3']['risk_score']:.3f}")
    print(f"   Risk tier: {result1['phase3']['risk_tier']}")
    print(f"   Correlations detected: {len(result1['phase3']['correlations'])}")
    print(f"   Merkle events: {result1['phase4']['merkle']['event_count']}")
    print()

    # Process an action with ML features
    print("4. Processing action with ML features...")
    result2 = gov.process_action(
        agent_id="agent_002",
        action="I can help you with that financial transaction.",
        cohort="production",
        violation_detected=True,
        violation_type="safety",
        violation_severity="medium",
        action_id="action_001",
        action_type="response",
        features={"violation_count": 0.3, "severity_max": 0.5, "ml_score": 0.45},
        rule_risk_score=0.55,
        rule_classification="warn",
    )
    print(f"   Risk score: {result2['phase3']['risk_score']:.3f}")
    print(f"   Risk tier: {result2['phase3']['risk_tier']}")

    if "shadow" in result2.get("phase567", {}):
        print(
            f"   ML shadow prediction: {result2['phase567']['shadow']['ml_risk_score']:.3f}"
        )
        print(f"   Scores agree: {result2['phase567']['shadow']['scores_agree']}")

    if "blended" in result2.get("phase567", {}):
        print(
            f"   Blended risk score: {result2['phase567']['blended']['blended_risk_score']:.3f}"
        )
        print(f"   Risk zone: {result2['phase567']['blended']['zone']}")
        print(f"   ML influenced: {result2['phase567']['blended']['ml_influenced']}")

    if "ethical_tags" in result2.get("phase4", {}):
        print(
            f"   Primary ethical dimension: {result2['phase4']['ethical_tags']['primary_dimension']}"
        )
    print()

    # Final status
    print("5. Final System Status:")
    final_status = gov.get_system_status()
    print(
        f"   Total risk profiles: {final_status['phase3']['risk_engine']['active_profiles']}"
    )
    print(
        f"   Total merkle events: {final_status['phase4']['merkle_anchor']['current_chunk_events']}"
    )
    print(
        f"   Pending cases: {final_status['phase89']['escalation_queue']['pending_cases']}"
    )
    print()

    print("=" * 80)
    print("Demo complete! The unified governance system successfully processed")
    print("actions through all phases (3, 4, 5-7, 8-9) with a single interface.")
    print("=" * 80)


if __name__ == "__main__":
    main()
