#!/usr/bin/env python3
"""
Regional Deployment Example

This example demonstrates how to use Nethical's regionalization features
for multi-region, geographically-distributed AI governance deployments.
"""

from nethical.core import IntegratedGovernance
from nethical.core.models import (
    AgentAction,
    ActionType
)


def example_basic_regional_setup():
    """Example 1: Basic regional setup with GDPR compliance."""
    print("\n" + "="*60)
    print("Example 1: Basic Regional Setup (EU GDPR)")
    print("="*60)
    
    # Create a regional governance instance for EU
    governance = IntegratedGovernance(
        storage_dir="./example_eu_data",
        region_id="eu-west-1",
        logical_domain="customer-service",
        data_residency_policy="EU_GDPR"
    )
    
    print(f"\n✓ Created governance for region: {governance.region_id}")
    print(f"  Domain: {governance.logical_domain}")
    print(f"  Policy: {governance.data_residency_policy}")
    print(f"  Compliance requirements: {governance.regional_policies['compliance_requirements']}")
    
    # Process an action with regional context
    result = governance.process_action(
        agent_id="agent_eu_001",
        action="Customer inquiry about data retention policies",
        region_id="eu-west-1",
        compliance_requirements=["GDPR", "data_protection"]
    )
    
    print(f"\n✓ Processed action in region: {result['region_id']}")
    print(f"  Data residency compliant: {result['data_residency']['compliant']}")
    print(f"  Risk score: {result['phase3']['risk_score']:.3f}")


def example_multi_region_deployment():
    """Example 2: Multi-region deployment with different policies."""
    print("\n" + "="*60)
    print("Example 2: Multi-Region Deployment")
    print("="*60)
    
    # US Region with CCPA compliance
    us_governance = IntegratedGovernance(
        storage_dir="./example_us_data",
        region_id="us-west-2",
        logical_domain="sales",
        data_residency_policy="US_CCPA"
    )
    print(f"\n✓ US Region: {us_governance.region_id}")
    print(f"  Policy: {us_governance.data_residency_policy}")
    print(f"  Cross-border transfers allowed: {us_governance.regional_policies['cross_border_transfer_allowed']}")
    
    # EU Region with GDPR compliance
    eu_governance = IntegratedGovernance(
        storage_dir="./example_eu_data",
        region_id="eu-central-1",
        logical_domain="operations",
        data_residency_policy="EU_GDPR"
    )
    print(f"\n✓ EU Region: {eu_governance.region_id}")
    print(f"  Policy: {eu_governance.data_residency_policy}")
    print(f"  Cross-border transfers allowed: {eu_governance.regional_policies['cross_border_transfer_allowed']}")
    
    # Asia-Pacific Region with AI Act compliance
    ap_governance = IntegratedGovernance(
        storage_dir="./example_ap_data",
        region_id="ap-south-1",
        logical_domain="analytics",
        data_residency_policy="AI_ACT"
    )
    print(f"\n✓ AP Region: {ap_governance.region_id}")
    print(f"  Policy: {ap_governance.data_residency_policy}")
    print(f"  Human oversight required: {ap_governance.regional_policies['human_oversight_required']}")


def example_cross_border_validation():
    """Example 3: Cross-border data transfer validation."""
    print("\n" + "="*60)
    print("Example 3: Cross-Border Transfer Validation")
    print("="*60)
    
    # GDPR prevents cross-border transfers
    eu_governance = IntegratedGovernance(
        storage_dir="./example_eu_validation",
        region_id="eu-west-1",
        data_residency_policy="EU_GDPR"
    )
    
    print("\n--- Test 1: Same-region transfer (EU → EU) ---")
    result = eu_governance.process_action(
        agent_id="agent_eu_002",
        action="Query customer data",
        region_id="eu-west-1"
    )
    print(f"✓ Compliant: {result['data_residency']['compliant']}")
    print(f"  Violations: {len(result['data_residency']['violations'])}")
    
    print("\n--- Test 2: Cross-border transfer (EU → US) ---")
    result = eu_governance.process_action(
        agent_id="agent_eu_003",
        action="Transfer data to US",
        region_id="us-east-1"
    )
    print(f"✗ Compliant: {result['data_residency']['compliant']}")
    print(f"  Violations: {result['data_residency']['violations']}")
    
    # CCPA allows cross-border transfers
    us_governance = IntegratedGovernance(
        storage_dir="./example_us_validation",
        region_id="us-west-2",
        data_residency_policy="US_CCPA"
    )
    
    print("\n--- Test 3: Cross-border transfer (US → US) ---")
    result = us_governance.process_action(
        agent_id="agent_us_001",
        action="Transfer within US",
        region_id="us-east-1"
    )
    print(f"✓ Compliant: {result['data_residency']['compliant']}")
    print(f"  Violations: {len(result['data_residency']['violations'])}")


def example_logical_domain_sharding():
    """Example 4: Logical domain sharding for department isolation."""
    print("\n" + "="*60)
    print("Example 4: Logical Domain Sharding")
    print("="*60)
    
    # Customer Service domain
    cs_governance = IntegratedGovernance(
        storage_dir="./example_cs_data",
        region_id="us-east-1",
        logical_domain="customer-service"
    )
    
    # Engineering domain
    eng_governance = IntegratedGovernance(
        storage_dir="./example_eng_data",
        region_id="us-east-1",
        logical_domain="engineering"
    )
    
    # Payment Processing domain
    payment_governance = IntegratedGovernance(
        storage_dir="./example_payment_data",
        region_id="us-east-1",
        logical_domain="payment-processing"
    )
    
    print("\n✓ Created 3 domain-isolated governance instances:")
    print(f"  1. Customer Service: {cs_governance.logical_domain}")
    print(f"  2. Engineering: {eng_governance.logical_domain}")
    print(f"  3. Payment Processing: {payment_governance.logical_domain}")
    
    # Create actions with domain context
    cs_action = AgentAction(
        agent_id="agent_cs_001",
        action_type=ActionType.QUERY,
        content="Customer support inquiry",
        region_id="us-east-1",
        logical_domain="customer-service"
    )
    
    eng_action = AgentAction(
        agent_id="agent_eng_001",
        action_type=ActionType.SYSTEM_COMMAND,
        content="Deploy new feature",
        region_id="us-east-1",
        logical_domain="engineering"
    )
    
    print("\n✓ Created domain-specific actions:")
    print(f"  CS Action: {cs_action.action_id} (domain: {cs_action.logical_domain})")
    print(f"  Eng Action: {eng_action.action_id} (domain: {eng_action.logical_domain})")


def example_cross_region_reporting():
    """Example 5: Cross-region reporting and aggregation."""
    print("\n" + "="*60)
    print("Example 5: Cross-Region Reporting")
    print("="*60)
    
    governance = IntegratedGovernance(storage_dir="./example_global_data")
    
    # Simulate metrics from multiple regions
    metrics = [
        {
            'action_id': 'action_eu_1',
            'region_id': 'eu-west-1',
            'risk_score': 0.5,
            'violation_detected': False
        },
        {
            'action_id': 'action_eu_2',
            'region_id': 'eu-west-1',
            'risk_score': 0.7,
            'violation_detected': True
        },
        {
            'action_id': 'action_us_1',
            'region_id': 'us-east-1',
            'risk_score': 0.3,
            'violation_detected': False
        },
        {
            'action_id': 'action_us_2',
            'region_id': 'us-east-1',
            'risk_score': 0.9,
            'violation_detected': True
        },
        {
            'action_id': 'action_ap_1',
            'region_id': 'ap-south-1',
            'risk_score': 0.6,
            'violation_detected': False
        },
    ]
    
    print("\n--- Aggregate by Region ---")
    regional_summary = governance.aggregate_by_region(metrics, group_by='region_id')
    
    for region, stats in regional_summary.items():
        print(f"\nRegion: {region}")
        print(f"  Total actions: {stats['count']}")
        print(f"  Violations: {stats['violations']}")
        print(f"  Avg risk score: {stats['avg_risk_score']:.2f}")
        print(f"  Action IDs: {', '.join(stats['actions'])}")
    
    # Simulate metrics by domain
    domain_metrics = [
        {
            'action_id': 'action_cs_1',
            'logical_domain': 'customer-service',
            'risk_score': 0.4,
            'violation_detected': False
        },
        {
            'action_id': 'action_cs_2',
            'logical_domain': 'customer-service',
            'risk_score': 0.6,
            'violation_detected': False
        },
        {
            'action_id': 'action_eng_1',
            'logical_domain': 'engineering',
            'risk_score': 0.8,
            'violation_detected': True
        },
        {
            'action_id': 'action_pay_1',
            'logical_domain': 'payment-processing',
            'risk_score': 0.9,
            'violation_detected': True
        },
    ]
    
    print("\n--- Aggregate by Logical Domain ---")
    domain_summary = governance.aggregate_by_region(domain_metrics, group_by='logical_domain')
    
    for domain, stats in domain_summary.items():
        print(f"\nDomain: {domain}")
        print(f"  Total actions: {stats['count']}")
        print(f"  Violations: {stats['violations']}")
        print(f"  Avg risk score: {stats['avg_risk_score']:.2f}")


def example_performance_test():
    """Example 6: Performance test with 5+ regions."""
    print("\n" + "="*60)
    print("Example 6: Multi-Region Performance Test")
    print("="*60)
    
    regions = [
        'eu-west-1',
        'us-east-1',
        'ap-south-1',
        'ap-northeast-1',
        'sa-east-1',
        'ca-central-1'
    ]
    
    print(f"\n✓ Testing with {len(regions)} regions:")
    
    for region in regions:
        gov = IntegratedGovernance(
            storage_dir=f"./example_perf_{region}",
            region_id=region
        )
        
        result = gov.process_action(
            agent_id=f"agent_{region}",
            action=f"Test action for {region}",
            region_id=region
        )
        
        print(f"  ✓ {region}: Processed successfully")
        print(f"     - Risk score: {result['phase3']['risk_score']:.3f}")
        print(f"     - Risk tier: {result['phase3']['risk_tier']}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Nethical Regional Deployment Examples")
    print("="*60)
    print("\nThese examples demonstrate F1: Regionalization & Sharding features")
    print("including geographic distribution, data residency, and cross-region")
    print("reporting capabilities.")
    
    try:
        example_basic_regional_setup()
        example_multi_region_deployment()
        example_cross_border_validation()
        example_logical_domain_sharding()
        example_cross_region_reporting()
        example_performance_test()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Review the code in examples/regional_deployment_demo.py")
        print("  2. Read docs/REGIONAL_DEPLOYMENT_GUIDE.md for detailed documentation")
        print("  3. Check tests/test_regionalization.py for comprehensive test coverage")
        print("  4. Adapt these examples for your production deployment")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
