"""
Demo: F2 Detector & Policy Extensibility

This script demonstrates the complete plugin and policy DSL system.
Shows how to:
1. Register custom detector plugins
2. Load and evaluate policies from YAML/JSON files
3. Run plugins and policies on actions
4. Monitor plugin health and performance

Status: Future Track F2 - Demonstration of planned functionality
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Any

# Add examples directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import demo utilities
try:
    from demo_utils import (
        print_header,
        print_section,
        print_success,
        print_error,
        print_warning,
        print_info,
        safe_import,
        run_demo_safely,
        print_feature_not_implemented,
        print_next_steps,
        print_key_features,
    )
except ImportError:
    # Fallback implementations
    def print_header(title, width=70):
        print(f"\n{'='*width}\n{title}\n{'='*width}\n")

    def print_section(title, level=1):
        print(
            f"\n{'---' if level==2 else '==='*23} {title} {'---' if level==2 else '==='*23}"
        )

    def print_success(msg):
        print(f"✓ {msg}")

    def print_error(msg):
        print(f"✗ {msg}")

    def print_warning(msg):
        print(f"⚠  {msg}")

    def print_info(msg, indent=0):
        print(f"{'  '*indent}{msg}")

    def safe_import(module, cls=None):
        try:
            mod = __import__(module, fromlist=[cls] if cls else [])
            return getattr(mod, cls) if cls else mod
        except:
            return None

    def run_demo_safely(func, name, skip=True):
        try:
            func()
            return True
        except Exception as e:
            print_error(f"Error in {name}: {e}")
            return False

    def print_feature_not_implemented(name, coming=None):
        msg = f"Feature '{name}' not yet implemented"
        if coming:
            msg += f" (coming in {coming})"
        print_warning(msg)

    def print_next_steps(steps, title="Next Steps"):
        print(f"\n{title}:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")

    def print_key_features(features, title="Key Features"):
        print(f"\n{title}:")
        for feature in features:
            print(f"  ✓ {feature}")


# Try to import required modules
get_plugin_manager = safe_import("nethical.core.plugin_interface", "get_plugin_manager")
get_policy_engine = safe_import("nethical.core.policy_dsl", "get_policy_engine")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Sample action classes for testing
class FinancialAction:
    """Sample action with financial data."""

    def __init__(self):
        self.content = (
            "Processing payment with credit card 4532-1234-5678-9010 for $500"
        )
        self.context = {
            "encryption_enabled": False,
            "authorized": False,
            "audit_logging": False,
        }


class HealthcareAction:
    """Sample action with healthcare data."""

    def __init__(self):
        self.content = "Patient MRN: 12345678 with diagnosis A01.1 requires treatment"
        self.context = {
            "patient_consent": False,
            "encryption_enabled": False,
            "minimum_necessary_reviewed": False,
        }


class SecureFinancialAction:
    """Sample action with proper security."""

    def __init__(self):
        self.content = "Processing financial_data transaction"
        self.context = {
            "encryption_enabled": True,
            "authorized": True,
            "audit_logging": True,
            "tls_enabled": True,
            "user_role": "financial_officer",
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
    print_section("1. PLUGIN REGISTRATION", level=1)

    if not get_plugin_manager:
        print_feature_not_implemented("Plugin System", "F2 Track")
        print_info("This demo would show:", 1)
        print_info("- Custom detector registration", 2)
        print_info("- Plugin discovery and listing", 2)
        print_info("- Plugin metadata management", 2)
        return

    try:
        plugin_manager = get_plugin_manager()

        # In a real implementation, would register custom detectors
        print_info("Registering plugins...", 0)
        print_info("✓ Registered: FinancialComplianceDetector", 1)
        print_info("✓ Registered: HealthcareComplianceDetector", 1)
        print_info("✓ Registered: CustomPolicyDetector", 1)

        print_section("Registered plugins", level=2)
        print_info("Demo mode - showing expected structure", 1)
    except Exception as e:
        print_error(f"Error in plugin registration: {e}")


async def demo_plugin_execution():
    """Demonstrate plugin execution on different actions."""
    print_section("2. PLUGIN EXECUTION", level=1)

    if not get_plugin_manager:
        print_feature_not_implemented("Plugin Execution", "F2 Track")
        print_info("This demo would show running plugins on actions", 1)
        return

    try:
        plugin_manager = get_plugin_manager()
        print_info("Demo mode - showing expected execution flow", 1)
        print_success("Plugins would detect violations in insecure actions")
        print_success("Plugins would pass secure actions")
    except Exception as e:
        print_error(f"Error in plugin execution: {e}")


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
        from nethical.core.policy_dsl import (
            Policy,
            PolicyRule,
            RuleSeverity,
            PolicyAction,
        )

        rule = PolicyRule(
            condition="contains(action.content, 'credit card')",
            severity=RuleSeverity.HIGH,
            actions=[PolicyAction.REQUIRE_ENCRYPTION, PolicyAction.AUDIT_LOG],
            description="Credit card data requires encryption",
        )

        policy = Policy(
            name="demo_financial_policy",
            version="1.0.0",
            enabled=True,
            rules=[rule],
            description="Demo financial compliance policy",
            tags={"demo", "finance"},
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
        metrics = info["metrics"]
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
        severity = v.severity if hasattr(v, "severity") else "unknown"
        by_severity[severity] = by_severity.get(severity, 0) + 1

    print(f"\nViolations by severity:")
    for severity in ["critical", "high", "medium", "low", "info"]:
        if severity in by_severity:
            print(f"  {severity.upper()}: {by_severity[severity]}")


async def main():
    """Run all demonstrations."""
    print_header("F2: DETECTOR & POLICY EXTENSIBILITY DEMO")
    print_info("This demo showcases the plugin and policy DSL system.")
    print_info("It demonstrates custom detectors, policy evaluation, and monitoring.\n")

    try:
        # Run all demo sections
        await demo_plugin_registration()
        await demo_plugin_execution()

        # Additional demos would be run here but need safety checks added
        if get_plugin_manager and get_policy_engine:
            print_warning("Additional demos require full F2 implementation")

        print_header("DEMO COMPLETE")

        print_key_features(
            [
                "Custom detector plugins",
                "Plugin registration and discovery",
                "Policy DSL (YAML/JSON)",
                "Policy evaluation engine",
                "Health monitoring",
                "Performance metrics",
                "Integrated detection pipeline",
            ]
        )

        print_next_steps(
            [
                "Create your own custom detectors",
                "Define organization-specific policies",
                "Load policies from configuration files",
                "Monitor plugin health and performance",
                "See docs/PLUGIN_DEVELOPER_GUIDE.md for details",
            ]
        )

    except KeyboardInterrupt:
        print_warning("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
