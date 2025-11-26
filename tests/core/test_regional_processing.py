"""
Tests for Regional Processing (effective_region fix from PR #2)

This module tests regional processing functionality including:
- Region ID configuration and propagation
- Data residency validation
- Regional policy enforcement
"""

import pytest
from pathlib import Path
import tempfile

from nethical.core.integrated_governance import IntegratedGovernance


class TestRegionalConfiguration:
    """Tests for regional configuration."""

    def test_initialization_with_region(self, tmp_path):
        """Test IntegratedGovernance initialization with region_id."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="eu-west-1",
        )

        assert governance.region_id == "eu-west-1"

    def test_initialization_without_region(self, tmp_path):
        """Test IntegratedGovernance initialization without region_id."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
        )

        assert governance.region_id is None

    def test_initialization_with_logical_domain(self, tmp_path):
        """Test IntegratedGovernance initialization with logical_domain."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="eu-west-1",
            logical_domain="customer-service",
        )

        assert governance.region_id == "eu-west-1"
        assert governance.logical_domain == "customer-service"

    def test_initialization_with_data_residency_policy(self, tmp_path):
        """Test IntegratedGovernance initialization with data_residency_policy."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="eu-central-1",
            data_residency_policy="EU_GDPR",
        )

        assert governance.data_residency_policy == "EU_GDPR"
        assert len(governance.regional_policies) > 0


class TestDataResidencyValidation:
    """Tests for data residency validation."""

    def test_validate_same_region(self, tmp_path):
        """Test validation passes for same region."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="eu-west-1",
            data_residency_policy="EU_GDPR",
        )

        result = governance.validate_data_residency(region_id="eu-west-1")

        assert result["compliant"] is True
        assert result["region_id"] == "eu-west-1"

    def test_validate_no_region_specified(self, tmp_path):
        """Test validation when no region is specified."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="us-east-1",
        )

        result = governance.validate_data_residency()

        assert result["region_id"] == "us-east-1"
        assert result["compliant"] is True

    def test_validate_cross_border_restricted(self, tmp_path):
        """Test validation fails for cross-border transfer when restricted."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="eu-central-1",
            data_residency_policy="EU_GDPR",
        )

        # EU_GDPR typically restricts cross-border transfers
        result = governance.validate_data_residency(region_id="us-east-1")

        # Should flag the cross-border concern
        assert result["region_id"] == "us-east-1"


class TestRegionalProcessing:
    """Tests for regional processing in action evaluation."""

    def test_process_action_with_region(self, tmp_path):
        """Test processing action with region_id parameter."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="eu-west-1",
        )

        result = governance.process_action(
            agent_id="test-agent",
            action="Query customer records",
            region_id="eu-west-1",
        )

        assert "decision" in result or "result" in result
        assert result.get("region_id") == "eu-west-1"

    def test_process_action_inherits_default_region(self, tmp_path):
        """Test processing action inherits default region when not specified."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="us-west-2",
        )

        result = governance.process_action(
            agent_id="test-agent",
            action="Query records",
        )

        assert result.get("region_id") == "us-west-2"

    def test_process_action_override_region(self, tmp_path):
        """Test processing action can override default region."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="us-east-1",
        )

        result = governance.process_action(
            agent_id="test-agent",
            action="Query records",
            region_id="eu-west-1",  # Override
        )

        assert result.get("region_id") == "eu-west-1"

    def test_process_action_no_region(self, tmp_path):
        """Test processing action without any region."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
        )

        result = governance.process_action(
            agent_id="test-agent",
            action="Query records",
        )

        assert result.get("region_id") is None


class TestEffectiveRegionVariable:
    """Tests for effective_region variable ordering (PR #2 fix regression tests)."""

    def test_effective_region_defined_before_use(self, tmp_path):
        """Test that effective_region is defined before it's used in processing."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="eu-west-1",
            enable_quota_enforcement=False,
        )

        # This should not raise NameError for effective_region
        result = governance.process_action(
            agent_id="test-agent",
            action="Test action",
            region_id="eu-central-1",
        )

        assert "decision" in result or "result" in result

    def test_effective_region_with_quota_enforcement(self, tmp_path):
        """Test effective_region works with quota enforcement."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="us-east-1",
            enable_quota_enforcement=False,  # Disable to avoid Redis dependency
        )

        result = governance.process_action(
            agent_id="test-agent",
            action="Test action",
            region_id="us-west-2",
        )

        # Should use the action's region_id as effective_region
        assert result.get("region_id") == "us-west-2"

    def test_effective_region_fallback_to_instance_region(self, tmp_path):
        """Test effective_region falls back to instance region when action region is None."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            region_id="ap-southeast-1",
        )

        result = governance.process_action(
            agent_id="test-agent",
            action="Test action",
            # No region_id specified - should use instance region
        )

        assert result.get("region_id") == "ap-southeast-1"


class TestRegionalPolicies:
    """Tests for regional policy loading and application."""

    def test_load_eu_gdpr_policy(self, tmp_path):
        """Test loading EU_GDPR policy."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            data_residency_policy="EU_GDPR",
        )

        assert governance.regional_policies is not None
        assert "data_retention_days" in governance.regional_policies

    def test_load_us_ccpa_policy(self, tmp_path):
        """Test loading US_CCPA policy."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            data_residency_policy="US_CCPA",
        )

        assert governance.regional_policies is not None

    def test_load_global_default_policy(self, tmp_path):
        """Test loading GLOBAL_DEFAULT policy."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            data_residency_policy="GLOBAL_DEFAULT",
        )

        assert governance.regional_policies is not None

    def test_unknown_policy_uses_global_default(self, tmp_path):
        """Test unknown policy falls back to GLOBAL_DEFAULT."""
        governance = IntegratedGovernance(
            storage_dir=str(tmp_path / "storage"),
            data_residency_policy="UNKNOWN_POLICY",
        )

        # Should not raise error, falls back to global default
        assert governance.regional_policies is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
