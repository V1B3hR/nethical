#!/usr/bin/env python3
"""Phase 4 Demo - Integrity & Ethics Operationalization.

This demo showcases all Phase 4 features:
1. Merkle Anchoring for immutable audit trails
2. Policy Diff Auditing for change management
3. Quarantine Mode for rapid incident response
4. Ethical Taxonomy for multi-dimensional impact analysis
5. SLA Monitoring for performance guarantees
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.core import Phase4IntegratedGovernance


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def demo_merkle_anchoring(gov):
    """Demo Merkle anchoring functionality."""
    print_section("1. Merkle Anchoring - Immutable Audit Trail")
    
    print("Adding events to audit log...")
    for i in range(10):
        gov.process_action(
            agent_id=f'agent_{i}',
            action=f'action_{i}',
            cohort='demo_cohort',
            violation_detected=(i % 3 == 0),
            violation_type='unauthorized_data_access' if (i % 3 == 0) else None
        )
    
    print(f"✓ Added 10 events to current chunk")
    
    # Finalize chunk and get Merkle root
    merkle_root = gov.finalize_audit_chunk()
    print(f"\n✓ Chunk finalized")
    print(f"  Merkle Root: {merkle_root}")
    
    # Verify chunk
    chunk_id = list(gov.merkle_anchor.finalized_chunks.keys())[0]
    is_valid = gov.verify_audit_segment(chunk_id)
    print(f"\n✓ Verification: {'✅ VALID' if is_valid else '❌ INVALID'}")
    
    # Show statistics
    stats = gov.merkle_anchor.get_statistics()
    print(f"\nAudit Statistics:")
    print(f"  Total Chunks: {stats['total_chunks']}")
    print(f"  Total Events: {stats['total_events']}")
    print(f"  Hash Algorithm: {stats['hash_algorithm']}")


def demo_policy_diff(gov):
    """Demo policy diff auditing."""
    print_section("2. Policy Diff Auditing - Change Management")
    
    old_policy = {
        'threshold': 0.5,
        'rate_limit': 100,
        'security': {
            'enabled': True,
            'level': 'standard'
        }
    }
    
    new_policy = {
        'threshold': 0.8,
        'rate_limit': 150,
        'security': {
            'enabled': True,
            'level': 'high'
        },
        'new_feature': 'enabled'
    }
    
    print("Comparing policy versions...")
    diff = gov.compare_policies(old_policy, new_policy)
    
    print(f"\n✓ Policy Comparison Complete")
    print(f"  Risk Level: {diff['risk_level'].upper()}")
    print(f"  Risk Score: {diff['risk_score']:.3f}")
    print(f"\nChanges:")
    print(f"  Added: {diff['summary']['added']}")
    print(f"  Modified: {diff['summary']['modified']}")
    print(f"  Removed: {diff['summary']['removed']}")
    
    print(f"\nRecommendations:")
    for rec in diff['recommendations']:
        print(f"  • {rec}")


def demo_quarantine(gov):
    """Demo quarantine mode."""
    print_section("3. Quarantine Mode - Rapid Incident Response")
    
    # Register agents
    gov.quarantine_manager.register_agent_cohort('agent_1', 'risky_cohort')
    gov.quarantine_manager.register_agent_cohort('agent_2', 'risky_cohort')
    
    print("Simulating synthetic attack...")
    result = gov.simulate_quarantine('risky_cohort')
    
    print(f"\n✓ Quarantine Simulation Complete")
    print(f"  Activation Time: {result['total_time_s']:.3f}s")
    print(f"  Target Time: <{result['target_time_s']}s")
    print(f"  Meets Requirement: {'✅ YES' if result['meets_requirement'] else '❌ NO'}")
    print(f"  Affected Agents: {result['affected_agents']}")
    
    # Try to process action from quarantined cohort
    print("\nTrying to process action from quarantined cohort...")
    action_result = gov.process_action(
        agent_id='agent_1',
        action='test_action',
        cohort='risky_cohort'
    )
    
    print(f"  Action Allowed: {'✅' if action_result['action_allowed'] else '❌'}")
    if not action_result['action_allowed']:
        print(f"  Reason: {action_result.get('reason', 'N/A')}")
    
    # Release cohort
    print("\nReleasing cohort from quarantine...")
    gov.release_cohort('risky_cohort')
    print("✓ Cohort released")


def demo_ethical_taxonomy(gov):
    """Demo ethical taxonomy."""
    print_section("4. Ethical Taxonomy - Multi-Dimensional Impact Analysis")
    
    violations = [
        ('unauthorized_data_access', 'Privacy violation'),
        ('emotional_manipulation', 'Manipulation detected'),
        ('discriminatory_behavior', 'Fairness violation'),
        ('harmful_content', 'Safety violation')
    ]
    
    print("Tagging violations with ethical dimensions...\n")
    
    for violation_type, description in violations:
        result = gov.process_action(
            agent_id='test_agent',
            action='test',
            cohort='test',
            violation_detected=True,
            violation_type=violation_type
        )
        
        if 'ethical_tags' in result:
            tags = result['ethical_tags']['dimensions']
            primary = result['ethical_tags']['primary_dimension']
            
            print(f"{description}:")
            print(f"  Primary: {primary}")
            print(f"  Scores: {', '.join(f'{k}={v:.2f}' for k, v in tags.items())}")
            print()
    
    # Show coverage
    coverage = gov.get_ethical_coverage()
    print(f"Taxonomy Coverage:")
    print(f"  Coverage: {coverage['coverage_percentage']:.1f}%")
    print(f"  Target: {coverage['target_percentage']:.1f}%")
    print(f"  Meets Target: {'✅ YES' if coverage['meets_target'] else '❌ NO'}")


def demo_sla_monitoring(gov):
    """Demo SLA monitoring."""
    print_section("5. SLA Monitoring - Performance Guarantees")
    
    print("Processing 100 actions and tracking latency...")
    
    for i in range(100):
        gov.process_action(
            agent_id=f'agent_{i % 10}',
            action='test',
            cohort='performance_test'
        )
    
    # Get SLA report
    report = gov.get_sla_report()
    
    print(f"\n✓ Performance Test Complete")
    print(f"\nSLA Report:")
    print(f"  Status: {report['overall_status'].upper()}")
    print(f"  SLA Met: {'✅ YES' if report['sla_met'] else '❌ NO'}")
    print(f"\nLatency Metrics:")
    print(f"  P95: {report['p95_latency_ms']:.2f}ms (target: {report['p95_target_ms']}ms)")
    print(f"  P99: {report['metrics']['p99_latency_ms']:.2f}ms")
    print(f"  Average: {report['metrics']['avg_latency_ms']:.2f}ms")
    print(f"  Max: {report['metrics']['max_latency_ms']:.2f}ms")
    
    if report['sla_met']:
        margin_pct = report['p95_margin_pct']
        print(f"\n✅ SLA compliance with {margin_pct:.1f}% margin")


def demo_system_status(gov):
    """Demo system status."""
    print_section("System Status - All Components")
    
    status = gov.get_system_status()
    
    print("Component Status:\n")
    
    for component, info in status['components'].items():
        enabled = "✅" if info.get('enabled') else "❌"
        component_name = component.replace('_', ' ').title()
        print(f"{enabled} {component_name}")
        
        for key, value in info.items():
            if key != 'enabled':
                print(f"    {key}: {value}")
        print()


def main():
    """Run Phase 4 demo."""
    print("\n" + "=" * 60)
    print("  PHASE 4 DEMO")
    print("  Integrity & Ethics Operationalization")
    print("=" * 60)
    
    # Initialize governance system
    print("\nInitializing Phase 4 Integrated Governance...")
    gov = Phase4IntegratedGovernance(
        storage_dir="/tmp/phase4_demo",
        enable_merkle_anchoring=True,
        enable_quarantine=True,
        enable_ethical_taxonomy=True,
        enable_sla_monitoring=True,
        taxonomy_path="taxonomies/ethics_taxonomy.json"
    )
    print("✓ Governance system initialized")
    
    # Run demos
    try:
        demo_merkle_anchoring(gov)
        demo_policy_diff(gov)
        demo_quarantine(gov)
        demo_ethical_taxonomy(gov)
        demo_sla_monitoring(gov)
        demo_system_status(gov)
        
        # Export report
        print_section("Phase 4 Report")
        print(gov.export_phase4_report())
        
        print("\n" + "=" * 60)
        print("  ✅ PHASE 4 DEMO COMPLETE")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
