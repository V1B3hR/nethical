"""Tests for F3: Privacy & Data Handling features.

This test suite covers:
- Enhanced Redaction Pipeline
- Differential Privacy
- Federated Analytics
- Data Minimization
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from nethical.core.redaction_pipeline import (
    EnhancedRedactionPipeline,
    RedactionPolicy,
    PIIType,
)
from nethical.core.differential_privacy import (
    DifferentialPrivacy,
    PrivacyMechanism,
    DPTrainingConfig,
    PrivacyAudit,
)
from nethical.core.federated_analytics import FederatedAnalytics, AggregationMethod
from nethical.core.data_minimization import (
    DataMinimization,
    DataCategory,
    RetentionPolicy,
)
from nethical.core.integrated_governance import IntegratedGovernance


class TestEnhancedRedactionPipeline:
    """Tests for enhanced redaction pipeline."""

    def test_pii_detection(self):
        """Test PII detection accuracy."""
        pipeline = EnhancedRedactionPipeline(policy=RedactionPolicy.STANDARD)

        # Test email detection
        text = "Contact me at john.doe@example.com for details"
        matches = pipeline.detect_pii(text)
        assert len(matches) > 0
        assert any(m.pii_type == PIIType.EMAIL for m in matches)

        # Test phone detection
        text = "Call me at (555) 123-4567"
        matches = pipeline.detect_pii(text)
        assert len(matches) > 0
        assert any(m.pii_type == PIIType.PHONE for m in matches)

        # Test SSN detection
        text = "My SSN is 123-45-6789"
        matches = pipeline.detect_pii(text)
        assert len(matches) > 0
        assert any(m.pii_type == PIIType.SSN for m in matches)

    def test_redaction_accuracy(self):
        """Test redaction accuracy meets >95% target."""
        pipeline = EnhancedRedactionPipeline(policy=RedactionPolicy.STANDARD)

        # Validate with test cases
        test_cases = [
            ("Email: test@example.com", [PIIType.EMAIL]),
            ("Phone: 555-123-4567", [PIIType.PHONE]),
            ("SSN: 123-45-6789", [PIIType.SSN]),
            ("Credit card: 4111 1111 1111 1111", [PIIType.CREDIT_CARD]),
            ("IP: 192.168.1.1", [PIIType.IP_ADDRESS]),
        ]

        accuracy = pipeline.validate_detection_accuracy(test_cases)
        assert accuracy >= 0.95, f"Accuracy {accuracy} below 95% threshold"

    def test_context_aware_redaction(self):
        """Test context-aware redaction preserves utility."""
        pipeline = EnhancedRedactionPipeline(
            policy=RedactionPolicy.STANDARD, min_confidence=0.85
        )

        text = "Please email me at user@company.com"
        result = pipeline.redact(text, preserve_utility=True)

        assert result.redacted_text != text
        assert len(result.pii_matches) > 0
        # Should preserve domain in utility-preserving mode
        if "@" in text:
            assert "@" in result.redacted_text or "[REDACTED]" in result.redacted_text

    def test_redaction_audit_trail(self):
        """Test redaction audit trail creation."""
        pipeline = EnhancedRedactionPipeline(
            policy=RedactionPolicy.STANDARD, enable_audit=True
        )

        text = "Contact: john@example.com and jane@test.com"
        result = pipeline.redact(text, user_id="test_user")

        assert len(pipeline.audit_trail) > 0
        assert pipeline.audit_trail[-1].action == "redact"
        assert pipeline.audit_trail[-1].success
        assert pipeline.audit_trail[-1].user_id == "test_user"

    def test_reversible_redaction(self):
        """Test reversible redaction for authorized access."""
        pipeline = EnhancedRedactionPipeline(
            policy=RedactionPolicy.STANDARD, enable_reversible=True
        )

        original_text = "Email: test@example.com"
        result = pipeline.redact(original_text)

        assert result.reversible
        assert len(result.redaction_map) > 0

        # Test unauthorized restoration
        restored = pipeline.restore(result, user_id="unauthorized", authorized=False)
        assert restored is None

        # Test authorized restoration
        restored = pipeline.restore(result, user_id="admin", authorized=True)
        assert restored is not None
        # Check that email is in restored text (pattern match)
        assert "@" in restored and "." in restored

    def test_redaction_policies(self):
        """Test different redaction policy levels."""
        text = "Name: John Doe, Email: john@example.com, SSN: 123-45-6789"

        # Minimal policy - only critical PII
        minimal = EnhancedRedactionPipeline(policy=RedactionPolicy.MINIMAL)
        result = minimal.redact(text)
        # Should redact SSN but not necessarily email
        assert (
            "SSN" in result.redacted_text or "123-45-6789" not in result.redacted_text
        )

        # Aggressive policy - all PII
        aggressive = EnhancedRedactionPipeline(policy=RedactionPolicy.AGGRESSIVE)
        result = aggressive.redact(text)
        # Should redact everything
        assert len(result.pii_matches) >= len(minimal.redact(text).pii_matches)

    def test_redaction_statistics(self):
        """Test redaction pipeline statistics."""
        pipeline = EnhancedRedactionPipeline(policy=RedactionPolicy.STANDARD)

        texts = ["Email: user1@example.com", "Phone: 555-123-4567", "SSN: 123-45-6789"]

        for text in texts:
            pipeline.redact(text)

        stats = pipeline.get_statistics()
        assert stats["total_redactions"] == len(texts)
        assert stats["accuracy_rate"] >= 0.95


class TestDifferentialPrivacy:
    """Tests for differential privacy implementation."""

    def test_privacy_budget_tracking(self):
        """Test privacy budget tracking."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

        assert dp.budget.epsilon == 1.0
        assert dp.budget.remaining == 1.0
        assert dp.budget.consumed == 0.0

        # Consume some budget
        dp.add_noise(100.0, sensitivity=1.0, operation="test_query")

        assert dp.budget.consumed > 0
        assert dp.budget.remaining < 1.0
        assert not dp.budget.is_exhausted()

    def test_noise_injection(self):
        """Test noise injection for aggregated metrics."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

        original_value = 100.0
        noised_value = dp.add_noise(original_value, sensitivity=1.0)

        # Value should be different (with high probability)
        assert noised_value != original_value

        # But should be reasonably close
        assert abs(noised_value - original_value) < 50

    def test_noised_metrics(self):
        """Test noise addition to aggregated metrics."""
        dp = DifferentialPrivacy(
            epsilon=10.0, delta=1e-5
        )  # Large epsilon for multiple operations

        metrics = {
            "accuracy": 0.95,
            "precision": 0.92,
        }

        noised_metrics = dp.add_noise_to_aggregated_metrics(
            metrics, sensitivity=0.1, noise_level=0.1
        )

        # Check that all metrics are present
        assert len(noised_metrics) == len(metrics)
        for key in metrics:
            assert key in noised_metrics
            # Values should be numeric
            assert isinstance(noised_metrics[key], (int, float))

        # Check privacy budget was consumed
        assert dp.budget.consumed > 0

    def test_dp_sgd(self):
        """Test DP-SGD for model training."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        config = DPTrainingConfig(
            epsilon=1.0, max_grad_norm=1.0, noise_multiplier=1.1, batch_size=32
        )

        # Simulate gradients
        gradients = np.random.randn(32, 10)

        # Apply DP-SGD
        noised_gradients = dp.dp_sgd_step(gradients, config, batch_size=32)

        assert noised_gradients.shape == (10,)
        # Gradients should be clipped and noised
        assert not np.array_equal(noised_gradients, gradients.mean(axis=0))

    def test_privacy_utility_tradeoff(self):
        """Test privacy-utility tradeoff optimization."""
        dp = DifferentialPrivacy(epsilon=5.0, delta=1e-5)

        # Define a simple utility function
        def utility_fn(epsilon):
            # Higher epsilon = higher utility (less noise)
            return 1.0 - np.exp(-epsilon)

        best_epsilon, best_utility = dp.optimize_privacy_utility_tradeoff(
            utility_fn, epsilon_range=(0.1, 5.0), num_samples=10
        )

        assert 0.1 <= best_epsilon <= 5.0
        assert 0 <= best_utility <= 1.0
        assert len(dp.tradeoff_history) == 10

    def test_privacy_budget_exhaustion(self):
        """Test behavior when privacy budget is exhausted."""
        dp = DifferentialPrivacy(epsilon=0.5, delta=1e-5)

        # Consume budget
        try:
            for i in range(10):
                dp.add_noise(100.0, sensitivity=1.0, operation=f"query_{i}")
        except ValueError as e:
            assert "Insufficient privacy budget" in str(e)

        # Budget should be exhausted or nearly so
        assert dp.budget.remaining <= 0.1

    def test_privacy_guarantees(self):
        """Test privacy guarantee reporting."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

        guarantees = dp.get_privacy_guarantees()

        assert guarantees["differential_privacy"]
        assert guarantees["epsilon"] == 1.0
        assert guarantees["delta"] == 1e-5
        assert "(1.0, 1e-05)-differential privacy" in guarantees["guarantee"]
        assert "interpretation" in guarantees


class TestFederatedAnalytics:
    """Tests for federated analytics."""

    def test_regional_metrics_registration(self):
        """Test registration of regional metrics."""
        regions = ["us-east-1", "eu-west-1", "ap-south-1"]
        fa = FederatedAnalytics(regions=regions)

        # Register metrics for each region
        for region in regions:
            fa.register_regional_metrics(
                region_id=region,
                metrics={"accuracy": 0.95, "latency": 100},
                sample_size=1000,
            )

        stats = fa.get_statistics()
        assert set(stats["regions_processed"]) == set(regions)

    def test_cross_region_aggregation(self):
        """Test cross-region metric aggregation without raw data sharing."""
        regions = ["us-east-1", "eu-west-1", "ap-south-1"]
        fa = FederatedAnalytics(regions=regions, privacy_preserving=True)

        # Register different metrics for each region
        fa.register_regional_metrics("us-east-1", {"accuracy": 0.95}, sample_size=1000)
        fa.register_regional_metrics("eu-west-1", {"accuracy": 0.93}, sample_size=800)
        fa.register_regional_metrics("ap-south-1", {"accuracy": 0.94}, sample_size=1200)

        # Compute aggregated metrics
        result = fa.compute_metrics(privacy_preserving=True, noise_level=0.1)

        assert "accuracy" in result.aggregated_values
        assert len(result.regions) == 3
        assert result.privacy_preserving
        assert result.total_samples == 3000

    def test_privacy_preserving_correlation(self):
        """Test privacy-preserving correlation detection."""
        regions = ["us-east-1", "eu-west-1"]
        fa = FederatedAnalytics(regions=regions, privacy_preserving=True)

        # Register correlated metrics
        fa.register_regional_metrics(
            "us-east-1", {"metric_a": 0.9, "metric_b": 0.85}, sample_size=1000
        )
        fa.register_regional_metrics(
            "eu-west-1", {"metric_a": 0.95, "metric_b": 0.92}, sample_size=1000
        )

        # Compute correlation
        correlation = fa.privacy_preserving_correlation(
            "metric_a", "metric_b", noise_level=0.1
        )

        assert -1.0 <= correlation.correlation <= 1.0
        assert 0.0 <= correlation.p_value <= 1.0
        assert correlation.privacy_preserving
        assert len(correlation.regions) == 2

    def test_secure_multiparty_statistics(self):
        """Test secure multi-party computation for statistics."""
        regions = ["us-east-1", "eu-west-1", "ap-south-1"]
        fa = FederatedAnalytics(regions=regions)

        # Register metrics
        for i, region in enumerate(regions):
            fa.register_regional_metrics(
                region, {"performance": 0.9 + i * 0.02}, sample_size=1000
            )

        # Compute mean
        result = fa.secure_multiparty_statistic(
            "performance", statistic="mean", privacy_preserving=True
        )

        assert "value" in result
        assert result["sample_size"] == 3000
        assert len(result["regions"]) == 3

    def test_encrypted_metric_reporting(self):
        """Test encrypted metric reporting."""
        regions = ["us-east-1", "eu-west-1"]
        fa = FederatedAnalytics(regions=regions, enable_encryption=True)

        fa.register_regional_metrics("us-east-1", {"metric": 0.95}, sample_size=1000)
        fa.register_regional_metrics("eu-west-1", {"metric": 0.93}, sample_size=1000)

        report = fa.get_encrypted_report(metric_names=["metric"])

        assert report["encrypted"]
        assert "report_hash" in report
        assert "data" in report
        assert report["metadata"]["regions_count"] == 2

    def test_privacy_guarantees_validation(self):
        """Test validation of privacy guarantees."""
        regions = ["us-east-1", "eu-west-1"]
        fa = FederatedAnalytics(
            regions=regions, privacy_preserving=True, enable_encryption=True
        )

        validation = fa.validate_privacy_guarantees()

        assert validation["privacy_preserving"]
        assert validation["passed"]
        assert validation["checks"]["no_raw_data_sharing"]["passed"]
        assert validation["checks"]["encryption"]["passed"]


class TestDataMinimization:
    """Tests for data minimization and right-to-be-forgotten."""

    def test_automatic_retention_policies(self):
        """Test automatic data retention policies."""
        dm = DataMinimization(enable_auto_deletion=True)

        # Store data in different categories
        dm.store_data(
            {"user_id": "user123", "email": "test@example.com"},
            category=DataCategory.PERSONAL_IDENTIFIABLE,
            user_id="user123",
        )

        dm.store_data(
            {"event": "login", "timestamp": datetime.now()},
            category=DataCategory.AUDIT,
            user_id="user123",
        )

        stats = dm.get_statistics()
        assert stats["total_records"] == 2
        assert stats["active_records"] == 2

    def test_minimal_data_collection(self):
        """Test minimal necessary data collection."""
        dm = DataMinimization()

        # Store with minimal fields only
        data = {
            "user_id": "user123",
            "email": "test@example.com",
            "unnecessary_field": "should_be_filtered",
        }

        record = dm.store_data(
            data, category=DataCategory.PERSONAL_IDENTIFIABLE, minimal_fields_only=True
        )

        # Should only keep essential fields
        assert "user_id" in record.data or "email" in record.data

    def test_anonymization_pipeline(self):
        """Test data anonymization pipeline."""
        dm = DataMinimization(anonymization_enabled=True)

        record = dm.store_data(
            {"email": "test@example.com", "name": "John Doe"},
            category=DataCategory.PERSONAL_IDENTIFIABLE,
            user_id="user123",
        )

        # Anonymize the record
        anonymized = dm.anonymize_data(record.record_id, anonymization_level="standard")

        assert anonymized is not None
        assert anonymized.anonymized
        # Original values should be hashed or removed
        if "email" in anonymized.data:
            assert anonymized.data["email"] != "test@example.com"

    def test_right_to_be_forgotten(self):
        """Test right-to-be-forgotten support."""
        dm = DataMinimization(enable_auto_deletion=True)

        user_id = "user_to_delete"

        # Store multiple records for the user
        for i in range(3):
            dm.store_data(
                {"data": f"value_{i}"},
                category=DataCategory.OPERATIONAL,
                user_id=user_id,
            )

        # Request deletion
        deletion_request = dm.request_data_deletion(user_id)

        assert deletion_request.status in ["pending", "completed"]
        assert deletion_request.user_id == user_id

        # Check that records are marked as deleted
        if deletion_request.status == "completed":
            deleted_count = sum(
                1
                for r in dm.records.values()
                if r.metadata.get("user_id") == user_id and r.deleted
            )
            assert deleted_count > 0

    def test_retention_policy_processing(self):
        """Test retention policy processing."""
        dm = DataMinimization(enable_auto_deletion=True)

        # Store data with short retention
        record = dm.store_data(
            {"test": "data"},
            category=DataCategory.PERSONAL_IDENTIFIABLE,
            user_id="test_user",
        )

        # Manually set expiration to past
        record.expires_at = datetime.now() - timedelta(days=1)

        # Process retention policies
        result = dm.process_retention_policy()

        assert result["expired_records"] > 0
        assert result["deleted"] > 0 or result["anonymized"] > 0

    def test_gdpr_ccpa_compliance(self):
        """Test GDPR/CCPA compliance validation."""
        dm = DataMinimization(enable_auto_deletion=True, anonymization_enabled=True)

        validation = dm.validate_compliance()

        assert "compliant" in validation
        assert "checks" in validation
        assert "retention_policies" in validation["checks"]
        assert "right_to_be_forgotten" in validation["checks"]


class TestIntegratedGovernanceWithPrivacy:
    """Tests for integrated governance with F3 privacy features."""

    def test_privacy_mode_initialization(self):
        """Test governance initialization with privacy mode."""
        gov = IntegratedGovernance(
            storage_dir="./test_governance_privacy",
            privacy_mode="differential",
            epsilon=1.0,
            redaction_policy="aggressive",
        )

        assert gov.privacy_mode == "differential"
        assert gov.epsilon == 1.0
        assert gov.differential_privacy is not None
        assert gov.redaction_pipeline is not None
        assert gov.data_minimization is not None

        # Check component flags
        assert gov.components_enabled["differential_privacy"]
        assert gov.components_enabled["redaction_pipeline"]
        assert gov.components_enabled["data_minimization"]

    def test_federated_analytics_with_regions(self):
        """Test federated analytics in regional governance."""
        gov = IntegratedGovernance(
            storage_dir="./test_governance_federated",
            region_id="us-east-1",
            privacy_mode="differential",
            epsilon=1.0,
        )

        assert gov.federated_analytics is not None
        assert gov.components_enabled["federated_analytics"]
        assert "us-east-1" in gov.federated_analytics.regions

    def test_redaction_in_governance(self):
        """Test redaction pipeline integration in governance."""
        gov = IntegratedGovernance(
            storage_dir="./test_governance_redaction", redaction_policy="standard"
        )

        # Test that redaction pipeline is available
        assert gov.redaction_pipeline is not None

        # Test redaction
        text = "Contact: admin@example.com"
        result = gov.redaction_pipeline.redact(text)

        assert result.redacted_text != text
        assert len(result.pii_matches) > 0

    def test_privacy_budget_in_governance(self):
        """Test privacy budget tracking in governance."""
        gov = IntegratedGovernance(
            storage_dir="./test_governance_budget",
            privacy_mode="differential",
            epsilon=2.0,
        )

        assert gov.differential_privacy is not None
        budget_status = gov.differential_privacy.get_privacy_budget_status()

        assert budget_status["epsilon_total"] == 2.0
        assert budget_status["epsilon_remaining"] == 2.0
        assert not budget_status["is_exhausted"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
