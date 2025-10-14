#!/usr/bin/env python3
"""
Regional Deployment Example - F1 Feature Track

This example demonstrates how to use Nethical's regionalization features
for multi-region, geographically-distributed AI governance deployments.

Features demonstrated:
- Regional governance instances with data residency policies (GDPR, CCPA, AI Act)
- Multi-region deployment with different compliance requirements
- Cross-border data transfer validation
- Logical domain sharding for department isolation
- Cross-region reporting and metric aggregation
- Performance testing with multiple regions

Status: Future Track F1 - Demonstration of planned functionality
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for demo utilities
sys.path.insert(0, str(Path(__file__).parent))

try:
    from demo_utils import (
        print_header, print_section, print_success, print_error,
        print_warning, print_info, print_metric, safe_import,
        run_demo_safely, print_demo_summary, print_feature_not_implemented,
        print_next_steps
    )
except ImportError:
    # Fallback if demo_utils is not available
    def print_header(title, width=70): print(f"\n{'='*width}\n{title}\n{'='*width}\n")
    def print_section(title, level=1): print(f"\n--- {title} ---")
    def print_success(msg): print(f"✓ {msg}")
    def print_error(msg): print(f"✗ {msg}")
    def print_warning(msg): print(f"⚠ {msg}")
    def print_info(msg, indent=0): print(f"{'  '*indent}{msg}")
    def print_metric(name, value, unit="", indent=1): print(f"{'  '*indent}{name}: {value}{unit}")
    def safe_import(module, cls=None): 
        try:
            mod = __import__(module, fromlist=[cls] if cls else [])
            return getattr(mod, cls) if cls else mod
        except: return None
    def run_demo_safely(func, name, skip=True):
        try: func(); return True
        except Exception as e: print_error(f"Error in {name}: {e}"); return False
    def print_demo_summary(demos): pass
    def print_feature_not_implemented(name, coming=None): print_warning(f"Feature '{name}' not yet implemented")
    def print_next_steps(steps, title="Next Steps"):
        print(f"\n{title}:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")

# Try to import required modules
IntegratedGovernance = safe_import('nethical.core', 'IntegratedGovernance')
AgentAction = safe_import('nethical.core.models', 'AgentAction')
ActionType = safe_import('nethical.core.models', 'ActionType')


def example_basic_regional_setup():
    """Example 1: Basic regional setup with GDPR compliance."""
    print_section("Example 1: Basic Regional Setup (EU GDPR)", level=1)
    
    # Check if required components are available
    if not IntegratedGovernance:
        print_feature_not_implemented("Regional Deployment", "F1 Track")
        print_info("This demo shows how regional governance would work", 1)
        print_info("with EU GDPR compliance requirements", 1)
        return
    
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
    print_section("Example 2: Multi-Region Deployment", level=1)
    
    # Check if required components are available
    if not IntegratedGovernance:
        print_feature_not_implemented("Multi-Region Deployment", "F1 Track")
        print_info("This demo would show deployment across multiple regions:", 1)
        print_info("- US Region with CCPA compliance", 2)
        print_info("- EU Region with GDPR compliance", 2)
        print_info("- Asia-Pacific Region with AI Act compliance", 2)
        return
    
    try:
        # US Region with CCPA compliance
        us_governance = IntegratedGovernance(
            storage_dir="./example_us_data",
            region_id="us-west-2",
            logical_domain="sales",
            data_residency_policy="US_CCPA"
        )
        print_success(f"US Region: {us_governance.region_id}")
        print_info(f"Policy: {us_governance.data_residency_policy}", 1)
        print_info(f"Cross-border transfers allowed: {us_governance.regional_policies['cross_border_transfer_allowed']}", 1)
        
        # EU Region with GDPR compliance
        eu_governance = IntegratedGovernance(
            storage_dir="./example_eu_data",
            region_id="eu-central-1",
            logical_domain="operations",
            data_residency_policy="EU_GDPR"
        )
        print_success(f"EU Region: {eu_governance.region_id}")
        print_info(f"Policy: {eu_governance.data_residency_policy}", 1)
        print_info(f"Cross-border transfers allowed: {eu_governance.regional_policies['cross_border_transfer_allowed']}", 1)
        
        # Asia-Pacific Region with AI Act compliance
        ap_governance = IntegratedGovernance(
            storage_dir="./example_ap_data",
            region_id="ap-south-1",
            logical_domain="analytics",
            data_residency_policy="AI_ACT"
        )
        print_success(f"AP Region: {ap_governance.region_id}")
        print_info(f"Policy: {ap_governance.data_residency_policy}", 1)
        print_info(f"Human oversight required: {ap_governance.regional_policies['human_oversight_required']}", 1)
    except Exception as e:
        print_error(f"Error in multi-region setup: {e}")
        import traceback
        traceback.print_exc()


def example_cross_border_validation():
    """Example 3: Cross-border data transfer validation."""
    print_section("Example 3: Cross-Border Transfer Validation", level=1)
    
    # Check if required components are available
    if not IntegratedGovernance:
        print_feature_not_implemented("Cross-Border Validation", "F1 Track")
        print_info("This demo would show:", 1)
        print_info("- Same-region transfers (compliant)", 2)
        print_info("- Cross-border EU → US transfers (GDPR violation)", 2)
        print_info("- Cross-border US → US transfers (CCPA allows)", 2)
        return
    
    try:
        # GDPR prevents cross-border transfers
        eu_governance = IntegratedGovernance(
            storage_dir="./example_eu_validation",
            region_id="eu-west-1",
            data_residency_policy="EU_GDPR"
        )
        
        print_section("Test 1: Same-region transfer (EU → EU)", level=2)
        result = eu_governance.process_action(
            agent_id="agent_eu_002",
            action="Query customer data",
            region_id="eu-west-1"
        )
        print_success(f"Compliant: {result['data_residency']['compliant']}")
        print_info(f"Violations: {len(result['data_residency']['violations'])}", 1)
        
        print_section("Test 2: Cross-border transfer (EU → US)", level=2)
        result = eu_governance.process_action(
            agent_id="agent_eu_003",
            action="Transfer data to US",
            region_id="us-east-1"
        )
        print_error(f"Compliant: {result['data_residency']['compliant']}")
        print_info(f"Violations: {result['data_residency']['violations']}", 1)
        
        # CCPA allows cross-border transfers
        us_governance = IntegratedGovernance(
            storage_dir="./example_us_validation",
            region_id="us-west-2",
            data_residency_policy="US_CCPA"
        )
        
        print_section("Test 3: Cross-border transfer (US → US)", level=2)
        result = us_governance.process_action(
            agent_id="agent_us_001",
            action="Transfer within US",
            region_id="us-east-1"
        )
        print_success(f"Compliant: {result['data_residency']['compliant']}")
        print_info(f"Violations: {len(result['data_residency']['violations'])}", 1)
    except Exception as e:
        print_error(f"Error in cross-border validation: {e}")


def example_logical_domain_sharding():
    """Example 4: Logical domain sharding for department isolation."""
    print_section("Example 4: Logical Domain Sharding", level=1)
    
    # Check if required components are available
    if not IntegratedGovernance or not AgentAction or not ActionType:
        print_feature_not_implemented("Logical Domain Sharding", "F1 Track")
        print_info("This demo would show department isolation with:", 1)
        print_info("- Customer Service domain", 2)
        print_info("- Engineering domain", 2)
        print_info("- Payment Processing domain", 2)
        return
    
    try:
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
        
        print_success("Created 3 domain-isolated governance instances:")
        print_info(f"1. Customer Service: {cs_governance.logical_domain}", 1)
        print_info(f"2. Engineering: {eng_governance.logical_domain}", 1)
        print_info(f"3. Payment Processing: {payment_governance.logical_domain}", 1)
        
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
        
        print_success("Created domain-specific actions:")
        print_info(f"CS Action: {cs_action.action_id} (domain: {cs_action.logical_domain})", 1)
        print_info(f"Eng Action: {eng_action.action_id} (domain: {eng_action.logical_domain})", 1)
    except Exception as e:
        print_error(f"Error in logical domain sharding: {e}")


def example_cross_region_reporting():
    """Example 5: Cross-region reporting and aggregation."""
    print_section("Example 5: Cross-Region Reporting", level=1)
    
    # Check if required components are available
    if not IntegratedGovernance:
        print_feature_not_implemented("Cross-Region Reporting", "F1 Track")
        print_info("This demo would show metrics aggregation across regions", 1)
        return
    
    try:
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
        
        print_section("Aggregate by Region", level=2)
        regional_summary = governance.aggregate_by_region(metrics, group_by='region_id')
        
        for region, stats in regional_summary.items():
            print(f"\nRegion: {region}")
            print_info(f"Total actions: {stats['count']}", 1)
            print_info(f"Violations: {stats['violations']}", 1)
            print_info(f"Avg risk score: {stats['avg_risk_score']:.2f}", 1)
            print_info(f"Action IDs: {', '.join(stats['actions'])}", 1)
        
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
        
        print_section("Aggregate by Logical Domain", level=2)
        domain_summary = governance.aggregate_by_region(domain_metrics, group_by='logical_domain')
        
        for domain, stats in domain_summary.items():
            print(f"\nDomain: {domain}")
            print_info(f"Total actions: {stats['count']}", 1)
            print_info(f"Violations: {stats['violations']}", 1)
            print_info(f"Avg risk score: {stats['avg_risk_score']:.2f}", 1)
    except Exception as e:
        print_error(f"Error in cross-region reporting: {e}")


def example_performance_test():
    """Example 6: Performance test with 5+ regions."""
    print_section("Example 6: Multi-Region Performance Test", level=1)
    
    # Check if required components are available
    if not IntegratedGovernance:
        print_feature_not_implemented("Multi-Region Performance", "F1 Track")
        print_info("This demo would test performance across 6 regions", 1)
        return
    
    try:
        regions = [
            'eu-west-1',
            'us-east-1',
            'ap-south-1',
            'ap-northeast-1',
            'sa-east-1',
            'ca-central-1'
        ]
        
        print_success(f"Testing with {len(regions)} regions:")
        
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
            
            print_success(f"{region}: Processed successfully")
            print_info(f"Risk score: {result['phase3']['risk_score']:.3f}", 2)
            print_info(f"Risk tier: {result['phase3']['risk_tier']}", 2)
    except Exception as e:
        print_error(f"Error in performance test: {e}")


def main():
    """Run all examples."""
    print_header("Nethical Regional Deployment Examples")
    print_info("These examples demonstrate F1: Regionalization & Sharding features")
    print_info("including geographic distribution, data residency, and cross-region")
    print_info("reporting capabilities.\n")
    
    # Track demo results
    demos = []
    
    try:
        success = run_demo_safely(example_basic_regional_setup, "Basic Regional Setup")
        demos.append({"name": "Basic Regional Setup", "success": success})
        
        success = run_demo_safely(example_multi_region_deployment, "Multi-Region Deployment")
        demos.append({"name": "Multi-Region Deployment", "success": success})
        
        success = run_demo_safely(example_cross_border_validation, "Cross-Border Validation")
        demos.append({"name": "Cross-Border Validation", "success": success})
        
        success = run_demo_safely(example_logical_domain_sharding, "Logical Domain Sharding")
        demos.append({"name": "Logical Domain Sharding", "success": success})
        
        success = run_demo_safely(example_cross_region_reporting, "Cross-Region Reporting")
        demos.append({"name": "Cross-Region Reporting", "success": success})
        
        success = run_demo_safely(example_performance_test, "Performance Test")
        demos.append({"name": "Performance Test", "success": success})
        
        print_header("All examples completed!")
        
        print_next_steps([
            "Review the code in examples/advanced/regional_deployment_demo.py",
            "Read docs/REGIONAL_DEPLOYMENT_GUIDE.md for detailed documentation",
            "Check tests/test_regionalization.py for comprehensive test coverage",
            "Adapt these examples for your production deployment"
        ])
        
    except KeyboardInterrupt:
        print_warning("\nDemo interrupted by user")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if demos:
            print_demo_summary(demos)


if __name__ == "__main__":
    main()
