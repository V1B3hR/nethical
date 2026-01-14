"""
Tests for Cyber Resilience Act (CRA) Compliance Module.

Tests product classification, requirements validation, SBOM generation,
and incident response compliance.
"""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nethical.compliance.cra import (
    CRAProductRisk,
    RequirementStatus,
    VulnerabilityProcessStatus,
    SecureByDefaultLevel,
    ProductInfo,
    CRAProductRiskResult,
    RequirementsValidation,
    SBOM,
    VulnerabilityProcessValidation,
    SecureByDefaultAssessment,
    ConformityDeclaration,
    SecurityIncident,
    IncidentResponseValidation,
    CyberResilienceActCompliance,
)


class TestProductInfo:
    """Tests for ProductInfo dataclass."""
    
    def test_product_info_creation(self):
        """Test ProductInfo creation."""
        info = ProductInfo(
            product_name="Nethical",
            product_version="2.0.0",
            manufacturer="Test Org",
            product_type="software",
            intended_use="AI safety governance"
        )
        
        assert info.product_name == "Nethical"
        assert info.product_version == "2.0.0"
        assert info.is_security_component is False


class TestCRACompliance:
    """Tests for CRA compliance validator."""
    
    @pytest.fixture
    def product_info_default(self):
        """Create product info for default risk classification."""
        return ProductInfo(
            product_name="TestApp",
            product_version="1.0.0",
            manufacturer="Test Company",
            product_type="software",
            intended_use="General purpose application"
        )
    
    @pytest.fixture
    def product_info_important(self):
        """Create product info for important risk classification."""
        return ProductInfo(
            product_name="Nethical AI Safety",
            product_version="2.0.0",
            manufacturer="Nethical",
            product_type="software",
            intended_use="AI governance and safety validation",
            is_security_component=True
        )
    
    @pytest.fixture
    def product_info_critical(self):
        """Create product info for critical risk classification."""
        return ProductInfo(
            product_name="SCADA Control",
            product_version="3.0.0",
            manufacturer="Industrial Corp",
            product_type="software",
            intended_use="Critical infrastructure control",
            is_security_component=True,
            used_in_critical_infrastructure=True
        )
    
    @pytest.fixture
    def cra_compliance_default(self, product_info_default):
        """Create CRA compliance instance for default product."""
        return CyberResilienceActCompliance(product_info_default)
    
    @pytest.fixture
    def cra_compliance_important(self, product_info_important):
        """Create CRA compliance instance for important product."""
        return CyberResilienceActCompliance(product_info_important)
    
    @pytest.fixture
    def cra_compliance_critical(self, product_info_critical):
        """Create CRA compliance instance for critical product."""
        return CyberResilienceActCompliance(product_info_critical)
    
    def test_initialization(self, cra_compliance_default, product_info_default):
        """Test CRA compliance initialization."""
        assert cra_compliance_default.product_info == product_info_default
        assert cra_compliance_default.risk_classification is None
    
    def test_classify_product_risk_default(self, cra_compliance_default):
        """Test product risk classification for default product."""
        result = cra_compliance_default.classify_product_risk()
        
        assert isinstance(result, CRAProductRiskResult)
        assert result.risk_level == CRAProductRisk.DEFAULT
        assert len(result.applicable_requirements) >= 2
        assert result.assessment_id is not None
    
    def test_classify_product_risk_important(self, cra_compliance_important):
        """Test product risk classification for important product."""
        result = cra_compliance_important.classify_product_risk()
        
        assert result.risk_level == CRAProductRisk.IMPORTANT
        assert "security" in result.classification_rationale.lower() or \
               "ai" in result.classification_rationale.lower() or \
               "governance" in result.classification_rationale.lower()
        assert len(result.applicable_requirements) > 2
    
    def test_classify_product_risk_critical(self, cra_compliance_critical):
        """Test product risk classification for critical product."""
        result = cra_compliance_critical.classify_product_risk()
        
        assert result.risk_level == CRAProductRisk.CRITICAL
        assert "critical infrastructure" in result.classification_rationale.lower()
        assert "Third-party security audit" in result.applicable_requirements
    
    def test_classify_product_risk_ai_governance(self):
        """Test that AI governance products are classified as Important."""
        info = ProductInfo(
            product_name="AI Governance Framework",
            product_version="1.0.0",
            manufacturer="Test",
            product_type="software",
            intended_use="AI safety"
        )
        cra = CyberResilienceActCompliance(info)
        result = cra.classify_product_risk()
        
        assert result.risk_level == CRAProductRisk.IMPORTANT
    
    def test_validate_essential_requirements(self, cra_compliance_important):
        """Test essential requirements validation."""
        validation = cra_compliance_important.validate_essential_requirements()
        
        assert isinstance(validation, RequirementsValidation)
        assert validation.overall_status in [
            RequirementStatus.COMPLIANT,
            RequirementStatus.PARTIAL
        ]
        assert len(validation.requirements_checked) > 0
        assert validation.compliant_count > 0
        assert len(validation.findings) > 0
    
    def test_validate_essential_requirements_details(self, cra_compliance_important):
        """Test essential requirements validation includes key checks."""
        validation = cra_compliance_important.validate_essential_requirements()
        
        # Check that key requirements are validated
        assert "secure_by_default" in validation.requirements_checked
        assert "access_control" in validation.requirements_checked
        assert "cia_triad" in validation.requirements_checked
        assert "vulnerability_disclosure" in validation.requirements_checked
    
    def test_generate_sbom(self, cra_compliance_important):
        """Test SBOM generation."""
        sbom = cra_compliance_important.generate_sbom()
        
        assert isinstance(sbom, SBOM)
        assert sbom.sbom_format == "CycloneDX"
        assert len(sbom.components) > 0
        assert sbom.components[0]["name"] == "Nethical AI Safety"
        assert sbom.sbom_id is not None
        assert isinstance(sbom.generated_at, datetime)
    
    def test_sbom_includes_metadata(self, cra_compliance_important):
        """Test SBOM includes required metadata."""
        sbom = cra_compliance_important.generate_sbom()
        
        assert "timestamp" in sbom.metadata
        assert "manufacturer" in sbom.metadata
        assert sbom.metadata["manufacturer"] == "Nethical"
    
    def test_validate_vulnerability_disclosure(self, cra_compliance_important):
        """Test vulnerability disclosure process validation."""
        validation = cra_compliance_important.validate_vulnerability_disclosure()
        
        assert isinstance(validation, VulnerabilityProcessValidation)
        assert validation.status in [
            VulnerabilityProcessStatus.ADEQUATE,
            VulnerabilityProcessStatus.NEEDS_IMPROVEMENT,
            VulnerabilityProcessStatus.INADEQUATE
        ]
        assert len(validation.findings) > 0
    
    def test_vulnerability_disclosure_checks_security_md(self, cra_compliance_important):
        """Test that vulnerability disclosure checks for SECURITY.md."""
        validation = cra_compliance_important.validate_vulnerability_disclosure()
        
        # Should check for SECURITY.md existence
        has_security_md = Path("SECURITY.md").exists()
        assert validation.has_disclosure_policy == has_security_md
    
    def test_assess_secure_by_default(self, cra_compliance_important):
        """Test secure-by-default assessment."""
        assessment = cra_compliance_important.assess_secure_by_default()
        
        assert isinstance(assessment, SecureByDefaultAssessment)
        assert assessment.level in [
            SecureByDefaultLevel.FULL,
            SecureByDefaultLevel.SUBSTANTIAL,
            SecureByDefaultLevel.BASIC,
            SecureByDefaultLevel.INSUFFICIENT
        ]
        assert assessment.secure_defaults_count >= 0
        assert len(assessment.findings) > 0
    
    def test_secure_by_default_checks_multiple_aspects(self, cra_compliance_important):
        """Test that secure-by-default checks multiple security aspects."""
        assessment = cra_compliance_important.assess_secure_by_default()
        
        # Should check multiple security defaults
        assert len(assessment.findings) >= 5
        
        # Check for key security defaults
        finding_names = [f["name"] for f in assessment.findings]
        assert "Authentication required" in finding_names
        assert "HTTPS/TLS" in finding_names
    
    def test_generate_conformity_declaration(self, cra_compliance_important):
        """Test conformity declaration generation."""
        declaration = cra_compliance_important.generate_conformity_declaration()
        
        assert isinstance(declaration, ConformityDeclaration)
        assert declaration.product_name == "Nethical AI Safety"
        assert declaration.product_version == "2.0.0"
        assert declaration.conformity_standard == "Cyber Resilience Act (CRA)"
        assert len(declaration.essential_requirements) > 0
        assert len(declaration.harmonized_standards) > 0
        assert declaration.declaration_id is not None
    
    def test_conformity_declaration_includes_key_requirements(self, cra_compliance_important):
        """Test that conformity declaration lists key requirements."""
        declaration = cra_compliance_important.generate_conformity_declaration()
        
        requirements_text = " ".join(declaration.essential_requirements)
        assert "Secure by default" in requirements_text
        assert "unauthorized access" in requirements_text
        assert "Vulnerability disclosure" in requirements_text
    
    def test_validate_incident_response_critical(self, cra_compliance_important):
        """Test incident response validation for critical incident."""
        incident = SecurityIncident(
            incident_id="INC-001",
            incident_type="vulnerability",
            severity="critical",
            description="Critical security vulnerability",
            affected_versions=["2.0.0"],
            detection_date=datetime.now(timezone.utc),
            exploitation_observed=True
        )
        
        validation = cra_compliance_important.validate_incident_response(incident)
        
        assert isinstance(validation, IncidentResponseValidation)
        assert validation.incident_id == "INC-001"
        assert validation.notification_required is True
        assert len(validation.findings) > 0
    
    def test_validate_incident_response_24hr_deadline(self, cra_compliance_important):
        """Test that actively exploited vulnerabilities have 24hr deadline."""
        incident = SecurityIncident(
            incident_id="INC-002",
            incident_type="vulnerability",
            severity="critical",
            description="Actively exploited vulnerability",
            affected_versions=["2.0.0"],
            detection_date=datetime.now(timezone.utc),
            exploitation_observed=True
        )
        
        validation = cra_compliance_important.validate_incident_response(incident)
        
        # Should have 24-hour deadline
        expected_deadline = incident.detection_date + timedelta(hours=24)
        assert abs((validation.notification_deadline - expected_deadline).total_seconds()) < 60
    
    def test_validate_incident_response_72hr_deadline(self, cra_compliance_important):
        """Test that non-exploited incidents have 72hr deadline."""
        incident = SecurityIncident(
            incident_id="INC-003",
            incident_type="vulnerability",
            severity="high",
            description="High severity vulnerability",
            affected_versions=["2.0.0"],
            detection_date=datetime.now(timezone.utc),
            exploitation_observed=False
        )
        
        validation = cra_compliance_important.validate_incident_response(incident)
        
        # Should have 72-hour deadline
        expected_deadline = incident.detection_date + timedelta(hours=72)
        assert abs((validation.notification_deadline - expected_deadline).total_seconds()) < 60
    
    def test_validate_incident_response_low_severity(self, cra_compliance_important):
        """Test that low severity incidents don't require notification."""
        incident = SecurityIncident(
            incident_id="INC-004",
            incident_type="bug",
            severity="low",
            description="Minor security issue",
            affected_versions=["2.0.0"],
            detection_date=datetime.now(timezone.utc)
        )
        
        validation = cra_compliance_important.validate_incident_response(incident)
        
        assert validation.notification_required is False


class TestDataclasses:
    """Test dataclass instantiation and defaults."""
    
    def test_product_info_defaults(self):
        """Test ProductInfo default values."""
        info = ProductInfo(
            product_name="Test",
            product_version="1.0",
            manufacturer="TestCo",
            product_type="software",
            intended_use="Testing"
        )
        
        assert info.is_security_component is False
        assert info.used_in_critical_infrastructure is False
        assert info.network_connectivity is True
    
    def test_cra_product_risk_result_creation(self):
        """Test CRAProductRiskResult dataclass."""
        result = CRAProductRiskResult(
            risk_level=CRAProductRisk.IMPORTANT,
            classification_rationale="Security component",
            applicable_requirements=["Req1", "Req2"]
        )
        
        assert result.risk_level == CRAProductRisk.IMPORTANT
        assert result.assessment_id is not None
        assert isinstance(result.timestamp, datetime)
    
    def test_sbom_creation(self):
        """Test SBOM dataclass."""
        sbom = SBOM(
            sbom_format="CycloneDX",
            sbom_version="1.5",
            components=[],
            dependencies=[],
            vulnerabilities=[],
            metadata={}
        )
        
        assert sbom.sbom_format == "CycloneDX"
        assert sbom.sbom_id is not None
        assert isinstance(sbom.generated_at, datetime)
    
    def test_conformity_declaration_creation(self):
        """Test ConformityDeclaration dataclass."""
        declaration = ConformityDeclaration(
            product_name="Test Product",
            product_version="1.0.0",
            manufacturer="TestCo",
            manufacturer_address="123 Test St",
            declaration_date=datetime.now(timezone.utc)
        )
        
        assert declaration.product_name == "Test Product"
        assert declaration.conformity_standard == "Cyber Resilience Act (CRA)"
        assert declaration.ce_marking_affixed is False
        assert declaration.declaration_id is not None
    
    def test_security_incident_creation(self):
        """Test SecurityIncident dataclass."""
        incident = SecurityIncident(
            incident_id="INC-001",
            incident_type="vulnerability",
            severity="high",
            description="Test incident",
            affected_versions=["1.0.0"],
            detection_date=datetime.now(timezone.utc)
        )
        
        assert incident.incident_id == "INC-001"
        assert incident.exploitation_observed is False
        assert incident.remediation_status == "investigating"
