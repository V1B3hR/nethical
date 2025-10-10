"""
Demo: F2 Detector & Policy Extensibility

This script demonstrates the complete plugin and policy DSL system.
Shows how to:
1. Register custom detector plugins
2. Load and evaluate policies from YAML/JSON files
3. Run plugins and policies on actions
4. Monitor plugin health and performance
"""

import asyncio
import logging
from pathlib import Path

import sys
from pathlib import Path

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent))

from nethical.core.plugin_interface import get_plugin_manager
from nethical.core.policy_dsl import get_policy_engine
from custom_detectors import (
    FinancialComplianceDetector,
    HealthcareComplianceDetector,
    CustomPolicyDetector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Sample action classes for testing
class FinancialAction:
    """Sample action with financial data."""
    def __init__(self):
        self.content = "Processing payment with credit card 4532-1234-5678-9010 for $500"
        self.context = {
            'encryption_enabled': False,
            'authorized': False,
            'audit_logging': False
        }


class HealthcareAction:
    """Sample action with healthcare data."""
    def __init__(self):
        self.content = "Patient MRN: 12345678 with diagnosis A01.1 requires treatment"
        self.context = {
            'patient_consent': False,
            'encryption_enabled': False,
            'minimum_necessary_reviewed': False
        }


class SecureFinancialAction:
    """Sample action with proper security."""
    def __init__(self):
        self.content = "Processing financial_data transaction"
        self.context = {
            'encryption_enabled': True,
            'authorized': True,
            'audit_logging': True,
            'tls_enabled': True,
            'user_role': 'financial_officer'
        }


class CustomPolicyAction:
    """Sample action for custom policy testing."""
    def __init__(self):
        self.content = "This action contains a secret password value"
        self.context = {}


def print_separator(title=""):
    """Print a separator line."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def print_violations(violations, title="Violations"):
    """Print violations in a formatted way."""
    print(f"\n{title}: {len(violations)}")
    for i, v in enumerate(violations, 1):
        print(f"\n  {i}. [{v.severity.upper()}] {v.description}")
        print(f"     Detector: {v.detector}")
        print(f"     Confidence: {v.confidence:.2f}")
        if v.recommendations:
            print(f"     Recommendations:")
            for rec in v.recommendations[:2]:  # Show first 2
                print(f"       - {rec}")


async def demo_plugin_registration():
    """Demonstrate plugin registration and usage."""
    print_separator("1. PLUGIN REGISTRATION")
    
    plugin_manager = get_plugin_manager()
    
    # Register custom detectors
    print("\nRegistering plugins...")
    detectors = [
        FinancialComplianceDetector(),
        HealthcareComplianceDetector(),
        CustomPolicyDetector(
            policy_name="no_secrets",
            forbidden_patterns=[r'\bsecret\b', r'\bpassword\b']
        )
    ]
    
    for detector in detectors:
        plugin_manager.register_plugin(detector)
        print(f"  ✓ Registered: {detector.name}")
    
    # List registered plugins
    print("\nRegistered plugins:")
    plugins = plugin_manager.list_plugins()
    for name, info in plugins.items():
        metadata = info['metadata']
        print(f"  - {name}")
        print(f"    Version: {metadata['version']}")
        print(f"    Status: {info['status']}")
        print(f"    Tags: {', '.join(metadata['tags'])}")


async def demo_plugin_execution():
    """Demonstrate plugin execution on different actions."""
    print_separator("2. PLUGIN EXECUTION")
    
    plugin_manager = get_plugin_manager()
    
    # Test 1: Financial action (should have violations)
    print("\n[Test 1] Running plugins on financial action (insecure)...")
    action = FinancialAction()
    results = await plugin_manager.run_all_plugins(action)
    
    total_violations = sum(len(v) for v in results.values())
    print(f"Total violations detected: {total_violations}")
    
    for plugin_name, violations in results.items():
        if violations:
            print(f"\n{plugin_name}:")
            for v in violations[:2]:  # Show first 2 per plugin
                print(f"  - [{v.severity}] {v.description}")
    
    # Test 2: Secure financial action (should have fewer violations)
    print("\n[Test 2] Running plugins on financial action (secure)...")
    secure_action = SecureFinancialAction()
    results = await plugin_manager.run_all_plugins(secure_action)
    
    total_violations = sum(len(v) for v in results.values())
    print(f"Total violations detected: {total_violations}")
    if total_violations == 0:
        print("  ✓ No violations - properly secured!")


async def demo_policy_loading():
    """Demonstrate policy loading from files."""
    print_separator("3. POLICY LOADING")
    
    policy_engine = get_policy_engine()
    
    # Check if policy files exist
    policy_dir = Path("examples/policies")
    if not policy_dir.exists():
        print(f"\nPolicy directory not found: {policy_dir}")
        print("Creating sample policies in memory...")
        
        # Create sample policy programmatically
        from nethical.core.policy_dsl import Policy, PolicyRule, RuleSeverity, PolicyAction
        
        rule = PolicyRule(
            condition="contains(action.content, 'credit card')",
            severity=RuleSeverity.HIGH,
            actions=[PolicyAction.REQUIRE_ENCRYPTION, PolicyAction.AUDIT_LOG],
            description="Credit card data requires encryption"
        )
        
        policy = Policy(
            name="demo_financial_policy",
            version="1.0.0",
            enabled=True,
            rules=[rule],
            description="Demo financial compliance policy",
            tags={"demo", "finance"}
        )
        
        policy_engine.add_policy(policy)
        print(f"  ✓ Created policy: {policy.name}")
    else:
        # Load from files
        print("\nLoading policies from files...")
        for policy_file in policy_dir.glob("*.yaml"):
            try:
                loaded = policy_engine.load_policy_file(str(policy_file))
                print(f"  ✓ Loaded from {policy_file.name}: {', '.join(loaded)}")
            except Exception as e:
                print(f"  ✗ Failed to load {policy_file.name}: {e}")
        
        for policy_file in policy_dir.glob("*.json"):
            try:
                loaded = policy_engine.load_policy_file(str(policy_file))
                print(f"  ✓ Loaded from {policy_file.name}: {', '.join(loaded)}")
            except Exception as e:
                print(f"  ✗ Failed to load {policy_file.name}: {e}")
    
    # List loaded policies
    print("\nLoaded policies:")
    policies = policy_engine.list_policies()
    for name, info in policies.items():
        print(f"  - {name} (v{info['version']})")
        print(f"    Enabled: {info['enabled']}")
        print(f"    Rules: {len(info['rules'])}")


async def demo_policy_evaluation():
    """Demonstrate policy evaluation."""
    print_separator("4. POLICY EVALUATION")
    
    policy_engine = get_policy_engine()
    
    # Test with financial action
    print("\n[Test] Evaluating policies on financial action...")
    action = FinancialAction()
    violations = policy_engine.evaluate_policies(action)
    
    print_violations(violations, "Policy Violations")


async def demo_health_monitoring():
    """Demonstrate plugin health monitoring."""
    print_separator("5. HEALTH MONITORING")
    
    plugin_manager = get_plugin_manager()
    
    print("\nPerforming health checks on all plugins...")
    health_results = await plugin_manager.health_check_all()
    
    for plugin_name, is_healthy in health_results.items():
        status = "✓ HEALTHY" if is_healthy else "✗ UNHEALTHY"
        print(f"  {plugin_name}: {status}")


async def demo_performance_metrics():
    """Demonstrate performance metrics."""
    print_separator("6. PERFORMANCE METRICS")
    
    plugin_manager = get_plugin_manager()
    
    print("\nPlugin Performance Metrics:")
    plugins = plugin_manager.list_plugins()
    
    for name, info in plugins.items():
        metrics = info['metrics']
        print(f"\n  {name}:")
        print(f"    Total runs: {metrics['total_runs']}")
        print(f"    Success rate: {metrics['success_rate']:.1f}%")
        print(f"    Violations detected: {metrics['violations_detected']}")


async def demo_integration():
    """Demonstrate integrated plugin and policy system."""
    print_separator("7. INTEGRATED DETECTION")
    
    plugin_manager = get_plugin_manager()
    policy_engine = get_policy_engine()
    
    # Test action
    action = FinancialAction()
    
    print("\nRunning complete detection pipeline...")
    
    # Run plugins
    plugin_violations = await plugin_manager.run_all_plugins(action)
    plugin_violation_count = sum(len(v) for v in plugin_violations.values())
    
    # Run policies
    policy_violations = policy_engine.evaluate_policies(action)
    policy_violation_count = len(policy_violations)
    
    # Combine results
    all_violations = []
    for violations in plugin_violations.values():
        all_violations.extend(violations)
    all_violations.extend(policy_violations)
    
    print(f"\nDetection Summary:")
    print(f"  Plugin violations: {plugin_violation_count}")
    print(f"  Policy violations: {policy_violation_count}")
    print(f"  Total violations: {len(all_violations)}")
    
    # Group by severity
    by_severity = {}
    for v in all_violations:
        severity = v.severity if hasattr(v, 'severity') else 'unknown'
        by_severity[severity] = by_severity.get(severity, 0) + 1
    
    print(f"\nViolations by severity:")
    for severity in ['critical', 'high', 'medium', 'low', 'info']:
        if severity in by_severity:
            print(f"  {severity.upper()}: {by_severity[severity]}")


async def main():
    """Run all demonstrations."""
    print_separator("F2: DETECTOR & POLICY EXTENSIBILITY DEMO")
    print("\nThis demo showcases the plugin and policy DSL system.")
    print("It demonstrates custom detectors, policy evaluation, and monitoring.")
    
    try:
        # Run all demo sections
        await demo_plugin_registration()
        await demo_plugin_execution()
        await demo_policy_loading()
        await demo_policy_evaluation()
        await demo_health_monitoring()
        await demo_performance_metrics()
        await demo_integration()
        
        print_separator("DEMO COMPLETE")
        print("\nKey Features Demonstrated:")
        print("  ✓ Custom detector plugins")
        print("  ✓ Plugin registration and discovery")
        print("  ✓ Policy DSL (YAML/JSON)")
        print("  ✓ Policy evaluation engine")
        print("  ✓ Health monitoring")
        print("  ✓ Performance metrics")
        print("  ✓ Integrated detection pipeline")
        
        print("\nNext Steps:")
        print("  - Create your own custom detectors")
        print("  - Define organization-specific policies")
        print("  - Load policies from configuration files")
        print("  - Monitor plugin health and performance")
        print("  - See docs/PLUGIN_DEVELOPER_GUIDE.md for details")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
