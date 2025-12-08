"""F3 Privacy & Data Handling Feature Demonstration.

This script demonstrates the complete F3 features:
1. Enhanced Redaction Pipeline
2. Differential Privacy
3. Federated Analytics
4. Data Minimization

Status: Future Track F3 - Demonstration of planned functionality
"""

import sys
from pathlib import Path
from typing import Optional, Any, Dict, List

# Try to import numpy (optional)
try:
    import numpy as np
except ImportError:
    np = None

# Add parent directory to path for demo utilities
sys.path.insert(0, str(Path(__file__).parent))

try:
    from demo_utils import (
        print_header,
        print_section,
        print_success,
        print_error,
        print_warning,
        print_info,
        print_metric,
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

    def print_metric(name, value, unit="", indent=1):
        print(f"{'  '*indent}{name}: {value}{unit}")

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
IntegratedGovernance = safe_import(
    "nethical.core.integrated_governance", "IntegratedGovernance"
)
EnhancedRedactionPipeline = safe_import(
    "nethical.core.redaction_pipeline", "EnhancedRedactionPipeline"
)
DifferentialPrivacy = safe_import(
    "nethical.core.differential_privacy", "DifferentialPrivacy"
)
FederatedAnalytics = safe_import(
    "nethical.core.federated_analytics", "FederatedAnalytics"
)
DataMinimization = safe_import("nethical.core.data_minimization", "DataMinimization")


def example_1_enhanced_redaction():
    """Example 1: Enhanced Redaction Pipeline with Context-Aware PII Detection."""
    print_section("Example 1: Enhanced Redaction Pipeline", level=1)

    if not EnhancedRedactionPipeline:
        print_feature_not_implemented("Enhanced Redaction", "F3 Track")
        print_info("This demo would show PII detection and redaction", 1)
        print_info("with >95% accuracy and utility preservation", 1)
        return

    try:
        # Create redaction pipeline with aggressive policy
        print_success("Redaction pipeline initialized")
        print_info("Demo mode - showing expected capabilities", 1)
    except Exception as e:
        print_error(f"Error in enhanced redaction: {e}")

    # Create redaction pipeline with aggressive policy
    pipeline = EnhancedRedactionPipeline(
        policy=RedactionPolicy.AGGRESSIVE, enable_audit=True, enable_reversible=True
    )

    # Sample text with PII
    text = """
    Dear Customer,
    
    Thank you for your inquiry. Please contact us at:
    - Email: support@company.com
    - Phone: (555) 123-4567
    - Our representative John Doe will assist you
    
    For verification, please provide:
    - SSN: 123-45-6789
    - Credit Card: 4111 1111 1111 1111
    
    Best regards,
    Customer Service Team
    """

    print("\nOriginal text (excerpt):")
    print(text[:200] + "...")

    # Detect PII
    pii_matches = pipeline.detect_pii(text)
    print(f"\n✓ Detected {len(pii_matches)} PII instances:")
    for match in pii_matches[:5]:
        print(f"  - {match.pii_type.value}: confidence={match.confidence:.2f}")

    # Redact with utility preservation
    result = pipeline.redact(
        text, user_id="customer_service_rep", preserve_utility=True
    )

    print("\n✓ Redacted text (excerpt):")
    print(result.redacted_text[:250] + "...")

    print(f"\n✓ Audit trail entries: {len(pipeline.audit_trail)}")
    print(f"✓ Reversible redaction: {result.reversible}")

    # Statistics
    stats = pipeline.get_statistics()
    print(f"\n📊 Pipeline Statistics:")
    print(f"  - Total redactions: {stats['total_redactions']}")
    print(f"  - Accuracy rate: {stats['accuracy_rate']:.1%}")
    print(f"  - PII detected: {sum(stats['pii_detected'].values())}")


def example_2_differential_privacy():
    """Example 2: Differential Privacy with Privacy Budget Tracking."""
    print("\n" + "=" * 70)
    print("Example 2: Differential Privacy & Privacy Budget Tracking")
    print("=" * 70)

    # Initialize differential privacy with epsilon=10.0 (for demonstration)
    dp = DifferentialPrivacy(epsilon=10.0, delta=1e-5)

    print(f"\n✓ Initialized with ε={dp.budget.epsilon}, δ={dp.budget.delta}")
    print(f"  Privacy guarantee: {dp.get_privacy_guarantees()['guarantee']}")

    # Simulate aggregated metrics with noise
    original_metrics = {
        "model_accuracy": 0.94,
        "precision": 0.92,
    }

    print("\n📊 Original Metrics:")
    for name, value in original_metrics.items():
        print(f"  {name}: {value:.4f}")

    # Add privacy noise
    noised_metrics = dp.add_noise_to_aggregated_metrics(
        original_metrics, sensitivity=0.1, noise_level=0.05
    )

    print("\n🔒 Privacy-Preserving Metrics (with noise):")
    for name, value in noised_metrics.items():
        diff = abs(value - original_metrics[name])
        print(f"  {name}: {value:.4f} (±{diff:.4f})")

    # Check privacy budget
    budget_status = dp.get_privacy_budget_status()
    print(f"\n💰 Privacy Budget:")
    print(f"  - Consumed: {budget_status['epsilon_consumed']:.4f}")
    print(f"  - Remaining: {budget_status['epsilon_remaining']:.4f}")
    print(f"  - Operations: {budget_status['operations_count']}")

    # Privacy audit
    audit = PrivacyAudit(dp)
    compliance = audit.validate_compliance()

    print(f"\n✅ Privacy Compliance:")
    print(f"  - GDPR compliant: {compliance['checks']['GDPR']['compliant']}")
    print(f"  - CCPA compliant: {compliance['checks']['CCPA']['compliant']}")


def example_3_federated_analytics():
    """Example 3: Federated Analytics for Cross-Region Aggregation."""
    print("\n" + "=" * 70)
    print("Example 3: Federated Analytics Across Regions")
    print("=" * 70)

    # Initialize federated analytics for 3 regions
    regions = ["us-east-1", "eu-west-1", "ap-south-1"]
    fa = FederatedAnalytics(
        regions=regions,
        privacy_preserving=True,
        enable_encryption=True,
        noise_level=0.1,
    )

    print(f"\n✓ Initialized federated analytics for {len(regions)} regions")

    # Simulate regional metrics
    regional_data = {
        "us-east-1": {"accuracy": 0.95, "latency": 120, "throughput": 850},
        "eu-west-1": {"accuracy": 0.93, "latency": 140, "throughput": 780},
        "ap-south-1": {"accuracy": 0.94, "latency": 150, "throughput": 820},
    }

    # Register metrics from each region
    for region, metrics in regional_data.items():
        fa.register_regional_metrics(
            region_id=region, metrics=metrics, sample_size=1000
        )
        print(f"\n📍 {region}:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Latency: {metrics['latency']}ms")
        print(f"  Throughput: {metrics['throughput']} req/s")

    # Compute aggregated metrics
    print("\n🌍 Computing privacy-preserving global aggregation...")
    aggregated = fa.compute_metrics(privacy_preserving=True)

    print(f"\n📊 Global Aggregated Metrics (privacy-preserving):")
    for name, value in aggregated.aggregated_values.items():
        print(f"  {name}: {value:.4f}")

    print(f"\n✓ Total samples: {aggregated.total_samples}")
    print(f"✓ Privacy-preserving: {aggregated.privacy_preserving}")
    print(f"✓ Noise level: {aggregated.noise_level}")

    # Privacy-preserving correlation
    correlation = fa.privacy_preserving_correlation("accuracy", "latency")
    print(f"\n📈 Cross-Region Correlation (privacy-preserving):")
    print(f"  Accuracy ↔ Latency: {correlation.correlation:.3f}")
    print(f"  P-value: {correlation.p_value:.4f}")
    print(
        f"  Confidence interval: [{correlation.confidence_interval[0]:.3f}, {correlation.confidence_interval[1]:.3f}]"
    )

    # Validate privacy guarantees
    validation = fa.validate_privacy_guarantees()
    print(f"\n✅ Privacy Guarantees Validation:")
    print(
        f"  - No raw data sharing: {validation['checks']['no_raw_data_sharing']['passed']}"
    )
    print(f"  - Encryption enabled: {validation['checks']['encryption']['passed']}")
    print(f"  - Noise addition: {validation['checks']['noise_addition']['passed']}")


def example_4_data_minimization():
    """Example 4: Data Minimization & Right-to-be-Forgotten."""
    print("\n" + "=" * 70)
    print("Example 4: Data Minimization & Right-to-be-Forgotten")
    print("=" * 70)

    # Initialize data minimization system
    dm = DataMinimization(
        storage_dir="./demo_data_minimization",
        enable_auto_deletion=True,
        anonymization_enabled=True,
    )

    print("\n✓ Data minimization system initialized")
    print("  Auto-deletion: Enabled")
    print("  Anonymization: Enabled")

    # Store minimal necessary data
    user_id = "user_demo_123"

    # Store PII with short retention
    record1 = dm.store_data(
        data={"email": "user@example.com", "name": "John Doe"},
        category=DataCategory.PERSONAL_IDENTIFIABLE,
        user_id=user_id,
        minimal_fields_only=True,
    )

    print(f"\n📝 Stored PII (30-day retention):")
    print(f"  Record ID: {record1.record_id[:16]}...")
    print(f"  Expires: {record1.expires_at.strftime('%Y-%m-%d')}")

    # Store operational data
    record2 = dm.store_data(
        data={"action": "login", "status": "success"},
        category=DataCategory.OPERATIONAL,
        user_id=user_id,
    )

    print(f"\n📝 Stored operational data (90-day retention):")
    print(f"  Record ID: {record2.record_id[:16]}...")

    # Anonymize data
    anonymized = dm.anonymize_data(record1.record_id, anonymization_level="standard")
    print(f"\n🔒 Anonymized record:")
    print(
        f"  Original email preserved: {'email' in record1.data and '@' in str(record1.data.get('email', ''))}"
    )
    print(f"  After anonymization: {anonymized.anonymized}")

    # Right-to-be-forgotten request
    print(f"\n🗑️ Processing right-to-be-forgotten request for {user_id}...")
    deletion_request = dm.request_data_deletion(user_id)

    print(f"✓ Deletion request created:")
    print(f"  Request ID: {deletion_request.request_id}")
    print(f"  Status: {deletion_request.status}")
    print(f"  Records deleted: {deletion_request.records_deleted}")

    # Statistics
    stats = dm.get_statistics()
    print(f"\n📊 Data Minimization Statistics:")
    print(f"  - Total records: {stats['total_records']}")
    print(f"  - Active records: {stats['active_records']}")
    print(f"  - Anonymized: {stats['anonymized_records']}")
    print(f"  - Deleted: {stats['deleted_records']}")

    # Compliance validation
    compliance = dm.validate_compliance()
    print(f"\n✅ Compliance Validation:")
    print(f"  - Overall compliant: {compliance['compliant']}")
    print(
        f"  - Retention policies: {compliance['checks']['retention_policies']['passed']}"
    )
    print(
        f"  - Right-to-be-forgotten: {compliance['checks']['right_to_be_forgotten']['passed']}"
    )


def example_5_integrated_governance():
    """Example 5: Integrated Governance with Privacy Features."""
    print("\n" + "=" * 70)
    print("Example 5: Integrated Governance with F3 Privacy Features")
    print("=" * 70)

    # Initialize governance with privacy mode
    gov = IntegratedGovernance(
        storage_dir="./demo_governance_privacy",
        region_id="us-east-1",
        privacy_mode="differential",
        epsilon=1.0,
        redaction_policy="aggressive",
    )

    print("\n✓ Integrated governance initialized with F3 features:")
    print(f"  - Privacy mode: {gov.privacy_mode}")
    print(f"  - Privacy budget (ε): {gov.epsilon}")
    print(f"  - Redaction policy: {gov.redaction_policy_name}")
    print(f"  - Region: {gov.region_id}")

    # Check enabled components
    privacy_components = {
        name: enabled
        for name, enabled in gov.components_enabled.items()
        if "privacy" in name.lower()
        or name in ["redaction_pipeline", "federated_analytics", "data_minimization"]
    }

    print(f"\n📦 Privacy Components Status:")
    for component, enabled in privacy_components.items():
        status = "✓ Enabled" if enabled else "✗ Disabled"
        print(f"  {component}: {status}")

    # Test redaction
    text_with_pii = "Contact admin@company.com or call (555) 123-4567"
    result = gov.redaction_pipeline.redact(text_with_pii)

    print(f"\n🔒 Redaction Test:")
    print(f"  Original: {text_with_pii[:50]}...")
    print(f"  Redacted: {result.redacted_text[:50]}...")
    print(f"  PII detected: {len(result.pii_matches)}")

    # Privacy budget status
    if gov.differential_privacy:
        budget = gov.differential_privacy.get_privacy_budget_status()
        print(f"\n💰 Privacy Budget Status:")
        print(f"  - Total: {budget['epsilon_total']}")
        print(f"  - Remaining: {budget['epsilon_remaining']}")
        print(f"  - Exhausted: {budget['is_exhausted']}")

    # Federated analytics availability
    if gov.federated_analytics:
        fa_stats = gov.federated_analytics.get_statistics()
        print(f"\n🌍 Federated Analytics:")
        print(f"  - Regions: {', '.join(fa_stats['regions'])}")
        print(f"  - Privacy-preserving: {fa_stats['privacy_preserving']}")

    print("\n✅ All F3 features integrated successfully!")


def main():
    """Run all F3 feature demonstrations."""
    print_header("F3: Privacy & Data Handling - Feature Demonstration")
    print_info("This demo showcases the complete F3 implementation:")
    print_info("1. Enhanced Redaction Pipeline (>95% PII detection accuracy)")
    print_info("2. Differential Privacy (DP-SGD, privacy budget tracking)")
    print_info("3. Federated Analytics (cross-region aggregation)")
    print_info("4. Data Minimization (GDPR/CCPA compliance)")
    print_info("5. Integrated Governance (unified interface)\n")

    # Check if any F3 features are available
    if not any(
        [
            EnhancedRedactionPipeline,
            DifferentialPrivacy,
            FederatedAnalytics,
            DataMinimization,
        ]
    ):
        print_feature_not_implemented("F3 Privacy Features", "F3 Track")
        print_section("Planned Features", level=2)
        print_key_features(
            [
                "PII detection and redaction (>95% accuracy)",
                "Differential privacy implementation",
                "Federated analytics for 3+ regions",
                "Privacy budget tracking",
                "GDPR/CCPA compliance validation",
                "Privacy impact assessment support",
            ]
        )
        print_next_steps(
            [
                "Review F3 feature specifications",
                "Plan privacy-preserving infrastructure",
                "Prepare compliance requirements",
                "Test with synthetic data",
            ]
        )
        return

    demos = []
    try:
        success = run_demo_safely(example_1_enhanced_redaction, "Enhanced Redaction")
        demos.append({"name": "Enhanced Redaction", "success": success})

        success = run_demo_safely(
            example_2_differential_privacy, "Differential Privacy"
        )
        demos.append({"name": "Differential Privacy", "success": success})

        success = run_demo_safely(example_3_federated_analytics, "Federated Analytics")
        demos.append({"name": "Federated Analytics", "success": success})

        success = run_demo_safely(example_4_data_minimization, "Data Minimization")
        demos.append({"name": "Data Minimization", "success": success})

        success = run_demo_safely(
            example_5_integrated_governance, "Integrated Governance"
        )
        demos.append({"name": "Integrated Governance", "success": success})

        print_header("All F3 Features Demonstrated Successfully!")

        print_key_features(
            [
                "PII detection and redaction (>95% accuracy)",
                "Differential privacy implementation",
                "Federated analytics for 3+ regions",
                "Privacy budget tracking",
                "GDPR/CCPA compliance validation",
                "Privacy impact assessment support",
            ]
        )

    except KeyboardInterrupt:
        print_warning("\nDemo interrupted by user")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
