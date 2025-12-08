"""Tests for Phase 3 compliance modules.

Tests the compliance validation capabilities including:
- GDPR compliance validation
- EU AI Act compliance validation
- Data Residency management
- Right to Explanation (GDPR Article 22)
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from nethical.compliance import (
    ComplianceValidator,
    ComplianceFramework,
    GDPRComplianceValidator,
    EUAIActValidator,
    DataResidencyManager,
    DataRegion,
    DataClassification,
    DataType,
    AIRiskLevel,
    LawfulBasis,
)


class TestGDPRComplianceValidator:
    """Test GDPR compliance validation."""

    def test_validator_initialization(self):
        """Test GDPR validator initialization."""
        validator = GDPRComplianceValidator(response_deadline_days=30)
        assert validator.response_deadline_days == 30
        assert len(validator.validation_results) == 0

    def test_validate_article_5(self):
        """Test Article 5 (Principles) validation."""
        validator = GDPRComplianceValidator()
        result = validator.validate_article_5(
            {
                "lawful_basis": True,
                "purpose_documented": True,
                "data_minimization": True,
                "accuracy_controls": True,
                "retention_policy": True,
                "security_controls": True,
            }
        )
        assert result.status.value == "compliant"
        assert len(result.gaps) == 0

    def test_validate_article_5_partial(self):
        """Test Article 5 with partial compliance."""
        validator = GDPRComplianceValidator()
        result = validator.validate_article_5(
            {
                "lawful_basis": True,
                "purpose_documented": True,
                "data_minimization": False,
                "accuracy_controls": False,
            }
        )
        assert result.status.value in ["partial", "non_compliant"]
        assert len(result.gaps) > 0

    def test_validate_article_22(self):
        """Test Article 22 (Automated Decision-Making) validation."""
        validator = GDPRComplianceValidator()
        result = validator.validate_article_22(
            {
                "human_intervention_available": True,
                "explanation_capability": True,
                "logic_documented": True,
                "significance_explained": True,
                "appeal_mechanism": True,
                "safeguards_implemented": True,
            }
        )
        assert result.status.value == "compliant"

    def test_validate_article_6_consent(self):
        """Test Article 6 (Lawful Basis) validation with consent."""
        validator = GDPRComplianceValidator()
        result = validator.validate_article_6(
            lawful_basis=LawfulBasis.CONSENT,
            supporting_evidence={
                "consent_record": True,
                "consent_freely_given": True,
                "withdrawal_mechanism": True,
            },
        )
        assert result.status.value == "compliant"

    def test_generate_automated_decision_explanation(self):
        """Test Article 22 compliant explanation generation."""
        validator = GDPRComplianceValidator()
        explanation = validator.generate_automated_decision_explanation(
            decision_id="test-123",
            decision="BLOCK",
            judgment_data={
                "risk_score": 0.85,
                "violations": [{"type": "safety", "description": "Test violation"}],
                "action": "test action",
            },
        )
        assert explanation.decision_id == "test-123"
        assert explanation.decision == "BLOCK"
        assert len(explanation.factors) > 0
        assert len(explanation.human_readable) > 0
        assert "appeal" in explanation.appeal_mechanism.lower()


class TestEUAIActValidator:
    """Test EU AI Act compliance validation."""

    def test_validator_initialization(self):
        """Test EU AI Act validator initialization."""
        validator = EUAIActValidator()
        assert validator.risk_level is None

    def test_classify_minimal_risk(self):
        """Test minimal risk classification."""
        validator = EUAIActValidator({})
        risk = validator.classify_risk_level()
        assert risk == AIRiskLevel.MINIMAL

    def test_classify_high_risk(self):
        """Test high risk classification for critical infrastructure."""
        validator = EUAIActValidator({"critical_infrastructure": True})
        risk = validator.classify_risk_level()
        assert risk == AIRiskLevel.HIGH

    def test_classify_unacceptable_risk(self):
        """Test unacceptable risk classification for social scoring."""
        validator = EUAIActValidator({"social_scoring": True})
        risk = validator.classify_risk_level()
        assert risk == AIRiskLevel.UNACCEPTABLE

    def test_classify_limited_risk(self):
        """Test limited risk classification for chatbots."""
        validator = EUAIActValidator({"chatbot": True})
        risk = validator.classify_risk_level()
        assert risk == AIRiskLevel.LIMITED

    def test_validate_article_9(self):
        """Test Article 9 (Risk Management) validation."""
        validator = EUAIActValidator({"critical_infrastructure": True})
        result = validator.validate_article_9(
            {
                "risk_process_established": True,
                "risks_identified": True,
                "mitigation_measures": True,
                "continuous_monitoring": True,
                "lifecycle_coverage": True,
            }
        )
        assert result.status.value == "compliant"

    def test_run_conformity_assessment(self):
        """Test full conformity assessment."""
        validator = EUAIActValidator({"critical_infrastructure": True})
        result = validator.run_conformity_assessment(
            {
                "risk_management": {
                    "risk_process_established": True,
                    "risks_identified": True,
                    "mitigation_measures": True,
                    "continuous_monitoring": True,
                },
                "data_governance": {
                    "governance_practices": True,
                    "data_quality_controls": True,
                    "bias_detection": True,
                },
                "logging": {
                    "automatic_logging": True,
                    "traceability": True,
                    "tamper_evident": True,
                },
            }
        )
        assert result.risk_level == AIRiskLevel.HIGH
        assert len(result.article_results) > 0


class TestDataResidencyManager:
    """Test data residency management."""

    def test_manager_initialization(self):
        """Test data residency manager initialization."""
        manager = DataResidencyManager()
        assert len(manager.policies) > 0

    def test_validate_pii_in_eu(self):
        """Test PII validation in EU region."""
        manager = DataResidencyManager()
        is_valid, violation = manager.validate_storage_location(
            data_type=DataType.PII,
            target_region=DataRegion.EU_WEST_1,
        )
        assert is_valid is True
        assert violation is None

    def test_validate_pii_in_us_blocked(self):
        """Test PII validation blocked in US region."""
        manager = DataResidencyManager()
        is_valid, violation = manager.validate_storage_location(
            data_type=DataType.PII,
            target_region=DataRegion.US_EAST_1,
        )
        assert is_valid is False
        assert violation is not None
        assert violation.blocked is True

    def test_cross_region_transfer_blocked(self):
        """Test cross-region transfer blocking for PII."""
        manager = DataResidencyManager()
        is_valid, violation = manager.validate_cross_region_transfer(
            data_type=DataType.PII,
            source_region=DataRegion.EU_WEST_1,
            target_region=DataRegion.US_EAST_1,
        )
        assert is_valid is False
        assert violation is not None
        assert "cross_region" in violation.violation_type

    def test_same_region_transfer_allowed(self):
        """Test same-region transfer allowed."""
        manager = DataResidencyManager()
        is_valid, violation = manager.validate_cross_region_transfer(
            data_type=DataType.PII,
            source_region=DataRegion.EU_WEST_1,
            target_region=DataRegion.EU_WEST_1,
        )
        assert is_valid is True

    def test_global_data_allowed_everywhere(self):
        """Test global data (policies) allowed in any region."""
        manager = DataResidencyManager()
        is_valid, violation = manager.validate_storage_location(
            data_type=DataType.POLICIES,
            target_region=DataRegion.US_EAST_1,
        )
        assert is_valid is True

    def test_classify_data_pii(self):
        """Test PII data classification."""
        manager = DataResidencyManager()
        data_type, classification = manager.classify_data(
            content={"email": "user@example.com", "name": "John Doe"},
        )
        assert data_type == DataType.PII
        assert classification in [
            DataClassification.PII,
            DataClassification.SENSITIVE_PII,
        ]

    def test_record_data_movement(self):
        """Test data movement recording."""
        manager = DataResidencyManager()
        record = manager.record_data_movement(
            data_id="test-data-123",
            data_type=DataType.GENERAL,
            classification=DataClassification.INTERNAL,
            source_region=DataRegion.EU_WEST_1,
            target_region=DataRegion.EU_CENTRAL_1,
            movement_type="copy",
            reason="Disaster recovery",
        )
        assert record.record_id is not None
        assert record.authorized is True


class TestComplianceValidator:
    """Test main compliance validator orchestrator."""

    def test_validator_initialization(self):
        """Test compliance validator initialization."""
        validator = ComplianceValidator()
        assert validator.gdpr_validator is not None
        assert validator.eu_ai_act_validator is not None
        assert validator.data_residency_manager is not None

    def test_validate_all_frameworks(self):
        """Test validation of all frameworks."""
        validator = ComplianceValidator({"critical_infrastructure": True})
        report = validator.validate(ComplianceFramework.ALL)
        assert report.report_id is not None
        assert len(report.frameworks_validated) > 0
        assert report.compliance_score >= 0

    def test_validate_gdpr_only(self):
        """Test GDPR-only validation."""
        validator = ComplianceValidator()
        report = validator.validate(ComplianceFramework.GDPR)
        assert ComplianceFramework.GDPR in report.frameworks_validated
        assert ComplianceFramework.EU_AI_ACT not in report.frameworks_validated

    def test_validate_eu_ai_act_only(self):
        """Test EU AI Act-only validation."""
        validator = ComplianceValidator({"critical_infrastructure": True})
        report = validator.validate(ComplianceFramework.EU_AI_ACT)
        assert ComplianceFramework.EU_AI_ACT in report.frameworks_validated
        assert report.eu_ai_act_summary is not None

    def test_report_to_dict(self):
        """Test report serialization."""
        validator = ComplianceValidator()
        report = validator.validate(ComplianceFramework.ALL)
        report_dict = report.to_dict()
        assert "report_id" in report_dict
        assert "compliance_score" in report_dict
        assert "validation_results" in report_dict

    def test_report_to_json(self):
        """Test report JSON serialization."""
        validator = ComplianceValidator()
        report = validator.validate(ComplianceFramework.ALL)
        json_str = report.to_json()
        assert isinstance(json_str, str)
        assert "report_id" in json_str

    def test_report_save(self):
        """Test report saving to file."""
        temp_dir = tempfile.mkdtemp()
        try:
            validator = ComplianceValidator()
            report = validator.validate(ComplianceFramework.ALL)
            output_path = Path(temp_dir) / "report.json"
            report.save(str(output_path))
            assert output_path.exists()
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
