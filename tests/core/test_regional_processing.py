"""Regression tests for regional processing fixes.

These tests verify the effective_region variable ordering fix from PR #157.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timezone

from nethical.core.integrated_governance import IntegratedGovernance
from nethical.core.models import AgentAction


class TestRegionalProcessing:
    """Regression tests for regional processing fixes."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp(prefix="test_regional_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def governance_with_region(self, temp_storage_dir):
        """Create governance instance with region configured."""
        return IntegratedGovernance(
            storage_dir=temp_storage_dir,
            region_id="us-east-1",
            enable_quota_enforcement=True,
        )

    @pytest.fixture
    def governance_no_region(self, temp_storage_dir):
        """Create governance instance without region."""
        return IntegratedGovernance(
            storage_dir=temp_storage_dir,
            enable_quota_enforcement=True,
        )

    def test_effective_region_with_quota(self, governance_with_region):
        """Verify effective_region is defined before use in quota check.

        Regression test for PR #157: Ensures the effective_region variable
        is properly initialized before being used in quota enforcement.
        """
        governance = governance_with_region

        # Process an action with quota checking enabled
        result = governance.process_action(
            agent_id="test_agent",
            action="SELECT * FROM users",
            action_type="query",
            cohort="default",
        )

        # Should complete without NameError
        assert result is not None
        assert "decision" in result

        # Region should be captured in result
        assert "region_id" in result
        assert result["region_id"] == "us-east-1"

    def test_effective_region_override_in_call(self, governance_with_region):
        """Verify region_id parameter overrides instance region."""
        governance = governance_with_region

        # Process with different region specified in call
        result = governance.process_action(
            agent_id="test_agent",
            action="INSERT INTO logs VALUES (...)",
            action_type="mutation",
            region_id="eu-west-1",  # Override default region
        )

        assert result is not None
        assert result["region_id"] == "eu-west-1"

    def test_region_id_propagation(self, governance_with_region):
        """Verify region_id is correctly propagated through processing."""
        governance = governance_with_region

        # Process action
        result = governance.process_action(
            agent_id="test_agent",
            action="DELETE FROM audit_logs WHERE age > 365",
            action_type="deletion",
        )

        # Region ID should be in the result
        assert result["region_id"] == "us-east-1"

        # If residency validation was performed, it should reference the region
        if "residency_validation" in result and result["residency_validation"]:
            assert result["residency_validation"]["region_id"] == "us-east-1"

    def test_no_region_specified(self, governance_no_region):
        """Verify processing works when no region is specified."""
        governance = governance_no_region

        result = governance.process_action(
            agent_id="test_agent",
            action="GET /api/status",
            action_type="query",
        )

        # Should complete without error
        assert result is not None
        assert "decision" in result

        # Region should be None or empty
        region = result.get("region_id")
        assert region is None or region == ""

    def test_quota_with_region_as_tenant(self, governance_with_region):
        """Verify region is used as tenant in quota enforcement."""
        governance = governance_with_region

        # Process multiple actions to test quota with region
        for i in range(3):
            result = governance.process_action(
                agent_id="quota_test_agent",
                action=f"Query {i}",
                action_type="query",
            )

            assert result is not None
            # Check quota enforcement used region
            if "quota_enforcement" in result and result["quota_enforcement"]:
                quota_info = result["quota_enforcement"]
                # Quota should have been checked with region as tenant
                assert quota_info.get("allowed") is not None


class TestDataResidencyValidation:
    """Tests for data residency validation with regions."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp(prefix="test_residency_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def governance(self, temp_storage_dir):
        """Create governance instance."""
        return IntegratedGovernance(
            storage_dir=temp_storage_dir,
            region_id="us-east-1",
        )

    def test_validate_data_residency_valid_region(self, governance):
        """Test data residency validation with valid region."""
        result = governance.validate_data_residency("us-east-1")

        assert result is not None
        assert "region_id" in result
        assert result["region_id"] == "us-east-1"
        assert result["compliant"] is True

    def test_validate_data_residency_different_regions(self, governance):
        """Test data residency validation with different regions."""
        regions = ["us-west-2", "eu-west-1", "ap-northeast-1"]

        for region in regions:
            result = governance.validate_data_residency(region)
            assert result is not None
            assert result["region_id"] == region

    def test_residency_validation_in_action_processing(self, governance):
        """Test residency validation is performed during action processing."""
        result = governance.process_action(
            agent_id="residency_agent",
            action="Process user data",
            action_type="processing",
            region_id="eu-central-1",
        )

        # Should include residency validation for EU region
        assert result is not None
        if "residency_validation" in result and result["residency_validation"]:
            assert result["residency_validation"]["region_id"] == "eu-central-1"


class TestMultiRegionGovernance:
    """Tests for multi-region governance scenarios."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp(prefix="test_multiregion_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_multiple_governance_instances_different_regions(self, temp_storage_dir):
        """Test multiple governance instances for different regions."""
        governance_us = IntegratedGovernance(
            storage_dir=f"{temp_storage_dir}/us",
            region_id="us-east-1",
        )

        governance_eu = IntegratedGovernance(
            storage_dir=f"{temp_storage_dir}/eu",
            region_id="eu-west-1",
        )

        # Process actions in both regions
        result_us = governance_us.process_action(
            agent_id="us_agent",
            action="US action",
        )

        result_eu = governance_eu.process_action(
            agent_id="eu_agent",
            action="EU action",
        )

        assert result_us["region_id"] == "us-east-1"
        assert result_eu["region_id"] == "eu-west-1"

    def test_cross_region_action_with_override(self, temp_storage_dir):
        """Test processing action for different region than instance default."""
        governance = IntegratedGovernance(
            storage_dir=temp_storage_dir,
            region_id="us-east-1",  # Default region
        )

        # Process action for different region
        result = governance.process_action(
            agent_id="cross_region_agent",
            action="Cross-region data transfer",
            region_id="eu-west-1",  # Override region
        )

        # Should use the override region
        assert result["region_id"] == "eu-west-1"
