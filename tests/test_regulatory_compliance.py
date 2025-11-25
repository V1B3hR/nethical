"""
Tests for the Regulatory Compliance Framework.

Tests EU AI Act, UK Law, and US Standards compliance modules.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
import tempfile

from nethical.security.regulatory_compliance import (
    AIRiskLevel,
    RegulatoryFramework,
    ComplianceStatus,
    ControlCategory,
    RegulatoryRequirement,
    EUAIActCompliance,
    UKLawCompliance,
    USStandardsCompliance,
    RegulatoryMappingGenerator,
    generate_regulatory_mapping_table,
)


class TestAIRiskLevel:
    """Tests for AI risk level classification."""

    def test_risk_levels_exist(self):
        """Test all risk levels are defined."""
        assert AIRiskLevel.UNACCEPTABLE.value == "unacceptable"
        assert AIRiskLevel.HIGH.value == "high"
        assert AIRiskLevel.LIMITED.value == "limited"
        assert AIRiskLevel.MINIMAL.value == "minimal"


class TestRegulatoryFramework:
    """Tests for regulatory framework enumeration."""

    def test_all_frameworks_defined(self):
        """Test all supported frameworks are defined."""
        frameworks = [f.value for f in RegulatoryFramework]
        assert "eu_ai_act" in frameworks
        assert "uk_gdpr" in frameworks
        assert "uk_dpa_2018" in frameworks
        assert "uk_nhs_dspt" in frameworks
        assert "us_nist_ai_rmf" in frameworks
        assert "us_hipaa" in frameworks
        assert "us_soc2" in frameworks


class TestEUAIActCompliance:
    """Tests for EU AI Act compliance module."""

    @pytest.fixture
    def eu_compliance(self):
        """Create EU AI Act compliance instance."""
        return EUAIActCompliance()

    def test_requirements_initialized(self, eu_compliance):
        """Test requirements are initialized."""
        assert len(eu_compliance.requirements) > 0
        assert "EU-AI-9.1" in eu_compliance.requirements
        assert "EU-AI-13.1" in eu_compliance.requirements
        assert "EU-AI-15.3" in eu_compliance.requirements

    def test_article_9_requirements(self, eu_compliance):
        """Test Article 9 (Risk Management) requirements."""
        req = eu_compliance.requirements.get("EU-AI-9.1")
        assert req is not None
        assert req.framework == RegulatoryFramework.EU_AI_ACT
        assert req.article == "Article 9"
        assert req.category == ControlCategory.RISK_MANAGEMENT
        assert req.high_risk_only is True

    def test_article_13_requirements(self, eu_compliance):
        """Test Article 13 (Transparency) requirements."""
        req = eu_compliance.requirements.get("EU-AI-13.1")
        assert req is not None
        assert req.category == ControlCategory.TRANSPARENCY
        assert len(req.code_modules) > 0

    def test_article_14_requirements(self, eu_compliance):
        """Test Article 14 (Human Oversight) requirements."""
        req = eu_compliance.requirements.get("EU-AI-14.1")
        assert req is not None
        assert req.category == ControlCategory.HUMAN_OVERSIGHT

    def test_classify_risk_unacceptable(self, eu_compliance):
        """Test unacceptable risk classification."""
        # Social scoring is banned
        result = eu_compliance.classify_risk_level({"social_scoring": True})
        assert result == AIRiskLevel.UNACCEPTABLE

        # Subliminal manipulation is banned
        result = eu_compliance.classify_risk_level({"subliminal_manipulation": True})
        assert result == AIRiskLevel.UNACCEPTABLE

    def test_classify_risk_high(self, eu_compliance):
        """Test high risk classification."""
        high_risk_cases = [
            {"biometric_identification": True},
            {"critical_infrastructure": True},
            {"education_vocational": True},
            {"employment_recruitment": True},
            {"law_enforcement": True},
        ]
        
        for case in high_risk_cases:
            result = eu_compliance.classify_risk_level(case)
            assert result == AIRiskLevel.HIGH, f"Failed for {case}"

    def test_classify_risk_limited(self, eu_compliance):
        """Test limited risk classification."""
        limited_cases = [
            {"chatbot": True},
            {"emotion_recognition": True},
            {"deep_fake": True},
        ]
        
        for case in limited_cases:
            result = eu_compliance.classify_risk_level(case)
            assert result == AIRiskLevel.LIMITED, f"Failed for {case}"

    def test_classify_risk_minimal(self, eu_compliance):
        """Test minimal risk classification."""
        result = eu_compliance.classify_risk_level({})
        assert result == AIRiskLevel.MINIMAL

        result = eu_compliance.classify_risk_level({"generic_ai": True})
        assert result == AIRiskLevel.MINIMAL

    def test_get_applicable_requirements_high_risk(self, eu_compliance):
        """Test getting requirements for high-risk systems."""
        requirements = eu_compliance.get_applicable_requirements(AIRiskLevel.HIGH)
        assert len(requirements) > 0
        
        # High-risk should include all requirements
        req_ids = [r.id for r in requirements]
        assert "EU-AI-9.1" in req_ids
        assert "EU-AI-13.1" in req_ids

    def test_get_applicable_requirements_unacceptable(self, eu_compliance):
        """Test that unacceptable risk returns no requirements."""
        requirements = eu_compliance.get_applicable_requirements(AIRiskLevel.UNACCEPTABLE)
        assert len(requirements) == 0

    def test_assess_requirement(self, eu_compliance):
        """Test updating requirement status."""
        result = eu_compliance.assess_requirement(
            "EU-AI-9.1",
            ComplianceStatus.COMPLIANT,
            "Verified through testing"
        )
        assert result is True
        
        req = eu_compliance.requirements["EU-AI-9.1"]
        assert req.implementation_status == ComplianceStatus.COMPLIANT
        assert req.remediation_notes == "Verified through testing"

    def test_assess_nonexistent_requirement(self, eu_compliance):
        """Test assessing non-existent requirement."""
        result = eu_compliance.assess_requirement(
            "NONEXISTENT",
            ComplianceStatus.COMPLIANT
        )
        assert result is False


class TestUKLawCompliance:
    """Tests for UK Law compliance module."""

    @pytest.fixture
    def uk_compliance(self):
        """Create UK Law compliance instance."""
        return UKLawCompliance()

    def test_requirements_initialized(self, uk_compliance):
        """Test requirements are initialized."""
        assert len(uk_compliance.requirements) > 0

    def test_uk_gdpr_requirements(self, uk_compliance):
        """Test UK GDPR requirements exist."""
        assert "UK-GDPR-5" in uk_compliance.requirements
        assert "UK-GDPR-25" in uk_compliance.requirements
        assert "UK-GDPR-32" in uk_compliance.requirements

    def test_dpa_2018_requirements(self, uk_compliance):
        """Test DPA 2018 requirements exist."""
        assert "UK-DPA-64" in uk_compliance.requirements
        req = uk_compliance.requirements["UK-DPA-64"]
        assert req.framework == RegulatoryFramework.UK_DPA_2018
        assert req.category == ControlCategory.TRANSPARENCY

    def test_nhs_dspt_requirements(self, uk_compliance):
        """Test NHS DSPT requirements exist."""
        assert "NHS-DSPT-1" in uk_compliance.requirements
        assert "NHS-DSPT-7" in uk_compliance.requirements
        
        req = uk_compliance.requirements["NHS-DSPT-7"]
        assert req.framework == RegulatoryFramework.UK_NHS_DSPT
        assert req.category == ControlCategory.ACCESS_CONTROL

    def test_data_subject_rights_coverage(self, uk_compliance):
        """Test data subject rights requirement."""
        req = uk_compliance.requirements["UK-GDPR-12-22"]
        assert req.category == ControlCategory.PRIVACY
        assert "data_compliance.py" in str(req.code_modules)


class TestUSStandardsCompliance:
    """Tests for US Standards compliance module."""

    @pytest.fixture
    def us_compliance(self):
        """Create US Standards compliance instance."""
        return USStandardsCompliance()

    def test_requirements_initialized(self, us_compliance):
        """Test requirements are initialized."""
        assert len(us_compliance.requirements) > 0

    def test_nist_rmf_govern_function(self, us_compliance):
        """Test NIST RMF GOVERN function requirements."""
        assert "NIST-RMF-GV1" in us_compliance.requirements
        req = us_compliance.requirements["NIST-RMF-GV1"]
        assert req.framework == RegulatoryFramework.US_NIST_AI_RMF
        assert req.category == ControlCategory.RISK_MANAGEMENT

    def test_nist_rmf_measure_function(self, us_compliance):
        """Test NIST RMF MEASURE function requirements."""
        assert "NIST-RMF-MS1" in us_compliance.requirements
        assert "NIST-RMF-MS2" in us_compliance.requirements

    def test_nist_rmf_manage_function(self, us_compliance):
        """Test NIST RMF MANAGE function requirements."""
        assert "NIST-RMF-MG1" in us_compliance.requirements
        assert "NIST-RMF-MG3" in us_compliance.requirements

    def test_soc2_requirements(self, us_compliance):
        """Test SOC2 requirements exist."""
        assert "SOC2-CC6.1" in us_compliance.requirements
        assert "SOC2-CC7.4" in us_compliance.requirements
        
        req = us_compliance.requirements["SOC2-CC6.1"]
        assert req.framework == RegulatoryFramework.US_SOC2
        assert req.category == ControlCategory.ACCESS_CONTROL


class TestRegulatoryMappingGenerator:
    """Tests for regulatory mapping table generation."""

    @pytest.fixture
    def generator(self):
        """Create mapping generator instance."""
        return RegulatoryMappingGenerator()

    def test_generator_initialization(self, generator):
        """Test generator initializes with all compliance modules."""
        assert generator.eu_ai_act is not None
        assert generator.uk_law is not None
        assert generator.us_standards is not None

    def test_generate_mapping_table(self, generator):
        """Test mapping table generation."""
        mapping = generator.generate_mapping_table()
        
        assert "metadata" in mapping
        assert "summary" in mapping
        assert "requirements" in mapping
        
        # Check metadata
        assert mapping["metadata"]["total_requirements"] > 0
        assert "frameworks" in mapping["metadata"]
        
        # Check summary
        assert "by_framework" in mapping["summary"]
        assert "by_category" in mapping["summary"]
        assert "by_status" in mapping["summary"]
        
        # Check requirements
        assert len(mapping["requirements"]) > 0

    def test_requirements_structure(self, generator):
        """Test requirement entries have correct structure."""
        mapping = generator.generate_mapping_table()
        
        for req in mapping["requirements"]:
            assert "id" in req
            assert "framework" in req
            assert "article" in req
            assert "title" in req
            assert "description" in req
            assert "category" in req
            assert "status" in req
            assert "code_modules" in req
            assert "test_evidence" in req
            assert "documentation" in req

    def test_generate_markdown_report(self, generator):
        """Test markdown report generation."""
        md = generator.generate_markdown_report()
        
        assert "# Regulatory Compliance Mapping Table" in md
        assert "## Summary by Framework" in md
        assert "## EU AI ACT" in md
        assert "## UK GDPR" in md
        assert "## Cross-Reference Matrix" in md
        assert "Last Updated:" in md

    def test_generate_json_report(self, generator):
        """Test JSON report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_mapping.json"
            json_str = generator.generate_json_report(str(output_path))
            
            # Verify JSON is valid
            data = json.loads(json_str)
            assert "metadata" in data
            assert "requirements" in data
            
            # Verify file was created
            assert output_path.exists()

    def test_generate_audit_report(self, generator):
        """Test audit report generation."""
        report = generator.generate_audit_report(auditor_name="TestAuditor")
        
        assert "report_id" in report
        assert "generated_at" in report
        assert report["auditor"] == "TestAuditor"
        assert "summary" in report
        assert "findings" in report
        assert "recommendations" in report
        
        # Check summary fields
        summary = report["summary"]
        assert "total_requirements" in summary
        assert "compliance_score" in summary
        assert "certification_ready" in summary


class TestGenerateRegulatoryMappingTable:
    """Tests for the convenience function."""

    def test_generate_to_temp_directory(self):
        """Test generating mapping table to temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_regulatory_mapping_table(tmpdir)
            
            assert "markdown" in result
            assert "json" in result
            assert "audit" in result
            
            # Verify files exist
            assert Path(result["markdown"]).exists()
            assert Path(result["json"]).exists()
            assert Path(result["audit"]).exists()

    def test_generated_files_content(self):
        """Test content of generated files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_regulatory_mapping_table(tmpdir)
            
            # Check markdown
            md_content = Path(result["markdown"]).read_text()
            assert "Regulatory Compliance" in md_content
            
            # Check JSON
            json_content = Path(result["json"]).read_text()
            data = json.loads(json_content)
            assert data["metadata"]["total_requirements"] > 0
            
            # Check audit
            audit_content = Path(result["audit"]).read_text()
            audit_data = json.loads(audit_content)
            assert "summary" in audit_data


class TestControlCategory:
    """Tests for control category enumeration."""

    def test_all_categories_defined(self):
        """Test all control categories are defined."""
        categories = [c.value for c in ControlCategory]
        
        expected = [
            "transparency",
            "explainability",
            "human_oversight",
            "data_governance",
            "risk_management",
            "technical_documentation",
            "conformity_assessment",
            "incident_response",
            "audit_logging",
            "access_control",
            "security",
            "privacy",
            "fairness",
        ]
        
        for cat in expected:
            assert cat in categories, f"Missing category: {cat}"


class TestComplianceStatus:
    """Tests for compliance status enumeration."""

    def test_all_statuses_defined(self):
        """Test all compliance statuses are defined."""
        statuses = [s.value for s in ComplianceStatus]
        
        assert "compliant" in statuses
        assert "partial" in statuses
        assert "non_compliant" in statuses
        assert "not_applicable" in statuses
        assert "pending_review" in statuses


class TestCrossFrameworkCoverage:
    """Tests for cross-framework control mapping."""

    def test_risk_management_across_frameworks(self):
        """Test risk management controls exist across frameworks."""
        eu = EUAIActCompliance()
        us = USStandardsCompliance()
        
        eu_risk = [r for r in eu.requirements.values() 
                   if r.category == ControlCategory.RISK_MANAGEMENT]
        us_risk = [r for r in us.requirements.values() 
                   if r.category == ControlCategory.RISK_MANAGEMENT]
        
        assert len(eu_risk) > 0
        assert len(us_risk) > 0

    def test_transparency_across_frameworks(self):
        """Test transparency controls exist across frameworks."""
        eu = EUAIActCompliance()
        uk = UKLawCompliance()
        
        eu_transparency = [r for r in eu.requirements.values() 
                          if r.category == ControlCategory.TRANSPARENCY]
        uk_transparency = [r for r in uk.requirements.values() 
                          if r.category == ControlCategory.TRANSPARENCY]
        
        assert len(eu_transparency) > 0
        assert len(uk_transparency) > 0

    def test_security_across_frameworks(self):
        """Test security controls exist across frameworks."""
        eu = EUAIActCompliance()
        uk = UKLawCompliance()
        us = USStandardsCompliance()
        
        eu_security = [r for r in eu.requirements.values() 
                      if r.category == ControlCategory.SECURITY]
        uk_security = [r for r in uk.requirements.values() 
                      if r.category == ControlCategory.SECURITY]
        us_security = [r for r in us.requirements.values() 
                      if r.category == ControlCategory.SECURITY]
        
        assert len(eu_security) > 0
        assert len(uk_security) > 0
        assert len(us_security) > 0
