"""
Tests for UK Online Safety Act (OSA) 2023 Compliance Module.

Tests duty of care framework, child safety, illegal content detection,
and transparency reporting.
"""

import pytest
from datetime import datetime

from nethical.compliance.uk_osa import (
    ServiceType,
    RiskLevel,
    Action,
    ChildSafetyRiskLevel,
    AgeVerificationStatus,
    AlgorithmicHarmLevel,
    OSARiskAssessment,
    CSAEDetectionResult,
    ChildSafetyRisk,
    OSATransparencyReport,
    AgeVerificationResult,
    AlgorithmicHarmAssessment,
    UKOSACompliance,
)


class TestServiceType:
    """Tests for service type enumeration."""
    
    def test_service_types_exist(self):
        """Test all service types are defined."""
        assert ServiceType.CATEGORY_1.value == "category_1"
        assert ServiceType.CATEGORY_2A.value == "category_2a"
        assert ServiceType.CATEGORY_2B.value == "category_2b"
        assert ServiceType.USER_TO_USER.value == "user_to_user"


class TestRiskLevel:
    """Tests for risk level enumeration."""
    
    def test_risk_levels_exist(self):
        """Test all risk levels are defined."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestUKOSACompliance:
    """Tests for UK OSA compliance validator."""
    
    @pytest.fixture
    def osa_compliance(self):
        """Create OSA compliance instance for Category 1 service."""
        return UKOSACompliance(
            service_type=ServiceType.CATEGORY_1,
            user_base_size=50_000_000  # 50M UK users
        )
    
    def test_initialization(self, osa_compliance):
        """Test compliance validator initializes correctly."""
        assert osa_compliance.service_type == ServiceType.CATEGORY_1
        assert osa_compliance.user_base_size == 50_000_000
        assert osa_compliance._metrics["total_assessments"] == 0
    
    def test_assess_safe_content(self, osa_compliance):
        """Test assessment of safe content."""
        result = osa_compliance.assess_illegal_content_risk(
            content="Hello, how are you today?",
            context={}
        )
        
        assert isinstance(result, OSARiskAssessment)
        assert result.risk_level == RiskLevel.LOW
        assert result.illegal_content_detected is False
        assert result.recommended_action == Action.ALLOW
        assert len(result.content_categories) == 0
    
    def test_assess_terrorism_content(self, osa_compliance):
        """Test detection of terrorism-related content."""
        result = osa_compliance.assess_illegal_content_risk(
            content="This message promotes terrorism and violent extremism",
            context={}
        )
        
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.illegal_content_detected is True
        assert "terrorism" in result.content_categories
        assert result.recommended_action == Action.REPORT_AUTHORITIES
        assert result.confidence > 0.8
    
    def test_assess_csae_content(self, osa_compliance):
        """Test detection of CSAE content."""
        result = osa_compliance.assess_illegal_content_risk(
            content="Content involving child exploitation",
            context={}
        )
        
        assert result.risk_level == RiskLevel.CRITICAL
        assert result.illegal_content_detected is True
        assert "csae" in result.content_categories
        assert result.recommended_action == Action.REPORT_AUTHORITIES
        assert result.confidence > 0.8
    
    def test_assess_fraud_content(self, osa_compliance):
        """Test detection of fraud/scam content."""
        result = osa_compliance.assess_illegal_content_risk(
            content="Get rich quick! Guaranteed returns! This is definitely not a scam!",
            context={}
        )
        
        assert result.risk_level == RiskLevel.HIGH
        assert result.illegal_content_detected is True
        assert "fraud" in result.content_categories
        assert result.recommended_action == Action.REMOVE
    
    def test_assess_hate_speech(self, osa_compliance):
        """Test detection of hate speech."""
        result = osa_compliance.assess_illegal_content_risk(
            content="This content contains hate speech and discriminatory language",
            context={}
        )
        
        assert result.risk_level == RiskLevel.HIGH
        assert result.illegal_content_detected is True
        assert "hate_speech" in result.content_categories
        assert result.recommended_action == Action.REMOVE
    
    def test_detect_csae_content_safe(self, osa_compliance):
        """Test CSAE detection with safe content."""
        result = osa_compliance.detect_csae_content(
            content="Family-friendly content",
            metadata={}
        )
        
        assert isinstance(result, CSAEDetectionResult)
        assert result.csae_detected is False
        assert result.recommended_action == Action.ALLOW
        assert result.requires_law_enforcement_report is False
    
    def test_detect_csae_content_with_keywords(self, osa_compliance):
        """Test CSAE detection with keyword matches."""
        result = osa_compliance.detect_csae_content(
            content="Content related to child abuse",
            metadata={}
        )
        
        assert result.csae_detected is True
        assert result.confidence > 0.8
        assert len(result.indicators) > 0
        assert result.recommended_action == Action.REPORT_AUTHORITIES
        assert result.requires_law_enforcement_report is True
    
    def test_detect_csae_content_with_hash(self, osa_compliance):
        """Test CSAE detection with known hash."""
        result = osa_compliance.detect_csae_content(
            content="Some content",
            metadata={"hash": "known_csae_hash_1"}
        )
        
        assert result.csae_detected is True
        assert result.confidence > 0.9
        assert "hash database" in result.indicators[0].lower()
        assert result.requires_law_enforcement_report is True
    
    def test_detect_csae_content_with_metadata(self, osa_compliance):
        """Test CSAE detection with concerning metadata."""
        result = osa_compliance.detect_csae_content(
            content="Image content",
            metadata={
                "contains_minors": True,
                "sexually_explicit": True
            }
        )
        
        assert result.csae_detected is True
        assert result.confidence > 0.85
        assert result.requires_law_enforcement_report is True
    
    def test_assess_child_user_safe_content(self, osa_compliance):
        """Test child user assessment with safe content."""
        result = osa_compliance.assess_child_user_risk(
            user_age=10,
            content_type="educational"
        )
        
        assert isinstance(result, ChildSafetyRisk)
        assert result.age_appropriate is True
        assert result.parental_controls_required is True  # Always for under 13
        assert result.risk_level == ChildSafetyRiskLevel.RESTRICTED
    
    def test_assess_child_user_adult_content(self, osa_compliance):
        """Test child user blocked from adult content."""
        result = osa_compliance.assess_child_user_risk(
            user_age=15,
            content_type="adult_content"
        )
        
        assert result.age_appropriate is False
        assert result.risk_level == ChildSafetyRiskLevel.BLOCKED
        assert "Age restriction: 18+" in result.content_restrictions
    
    def test_assess_child_user_gambling(self, osa_compliance):
        """Test child user blocked from gambling content."""
        result = osa_compliance.assess_child_user_risk(
            user_age=16,
            content_type="gambling"
        )
        
        assert result.age_appropriate is False
        assert result.risk_level == ChildSafetyRiskLevel.BLOCKED
    
    def test_assess_adult_user(self, osa_compliance):
        """Test adult user has no restrictions."""
        result = osa_compliance.assess_child_user_risk(
            user_age=25,
            content_type="adult_content"
        )
        
        assert result.age_appropriate is True
        assert result.parental_controls_required is False
        assert result.risk_level == ChildSafetyRiskLevel.SAFE
        assert len(result.content_restrictions) == 0
    
    def test_assess_young_child_protections(self, osa_compliance):
        """Test enhanced protections for children under 13."""
        result = osa_compliance.assess_child_user_risk(
            user_age=10,
            content_type="social_media"
        )
        
        assert result.parental_controls_required is True
        assert "Parental consent required" in result.content_restrictions
        assert "Limited data collection" in result.content_restrictions
        assert "No targeted advertising" in result.content_restrictions
    
    def test_generate_transparency_report(self, osa_compliance):
        """Test transparency report generation."""
        # Generate some activity
        osa_compliance.assess_illegal_content_risk("Safe content", {})
        osa_compliance.assess_illegal_content_risk("Content with terrorism", {})
        osa_compliance.detect_csae_content("Safe content", {})
        
        report = osa_compliance.generate_transparency_report("Q1 2024")
        
        assert isinstance(report, OSATransparencyReport)
        assert report.reporting_period == "Q1 2024"
        assert report.total_content_assessments == 2
        assert report.illegal_content_detected >= 1
        assert report.report_id is not None
        assert isinstance(report.generated_at, datetime)
    
    def test_validate_age_verification_government_id(self, osa_compliance):
        """Test age verification with government ID."""
        result = osa_compliance.validate_age_verification("government_id")
        
        assert isinstance(result, AgeVerificationResult)
        assert result.status == AgeVerificationStatus.VERIFIED
        assert result.confidence >= 0.90
        assert result.verified_age is not None
    
    def test_validate_age_verification_credit_card(self, osa_compliance):
        """Test age verification with credit card."""
        result = osa_compliance.validate_age_verification("credit_card")
        
        assert result.status == AgeVerificationStatus.VERIFIED
        assert result.confidence >= 0.80
    
    def test_validate_age_verification_facial(self, osa_compliance):
        """Test age verification with facial estimation."""
        result = osa_compliance.validate_age_verification("facial_age_estimation")
        
        assert result.status == AgeVerificationStatus.INSUFFICIENT_EVIDENCE
        assert result.confidence < 0.80
    
    def test_validate_age_verification_self_declaration(self, osa_compliance):
        """Test age verification with self declaration (insufficient)."""
        result = osa_compliance.validate_age_verification("self_declaration")
        
        assert result.status == AgeVerificationStatus.FAILED
        assert result.confidence < 0.60
    
    def test_validate_age_verification_unknown_method(self, osa_compliance):
        """Test age verification with unknown method."""
        result = osa_compliance.validate_age_verification("unknown_method")
        
        assert result.status == AgeVerificationStatus.FAILED
        assert result.confidence == 0.0
        assert "Unrecognized" in result.explanation
    
    def test_assess_algorithmic_harm_minimal(self, osa_compliance):
        """Test algorithmic harm assessment with minimal risk."""
        result = osa_compliance.assess_algorithmic_harm({
            "personalization_strength": 0.3,
            "engagement_optimization": False,
            "bias_score": 0.2,
            "explainable": True,
        })
        
        assert isinstance(result, AlgorithmicHarmAssessment)
        assert result.harm_level == AlgorithmicHarmLevel.MINIMAL
        assert result.mitigation_required is False
    
    def test_assess_algorithmic_harm_filter_bubble(self, osa_compliance):
        """Test detection of filter bubble risks."""
        result = osa_compliance.assess_algorithmic_harm({
            "personalization_strength": 0.9,
        })
        
        assert result.harm_level in [AlgorithmicHarmLevel.MODERATE, AlgorithmicHarmLevel.SIGNIFICANT]
        assert any("filter bubble" in issue.lower() for issue in result.issues_identified)
    
    def test_assess_algorithmic_harm_engagement_optimization(self, osa_compliance):
        """Test detection of engagement optimization risks."""
        result = osa_compliance.assess_algorithmic_harm({
            "engagement_optimization": True,
        })
        
        assert result.harm_level == AlgorithmicHarmLevel.SIGNIFICANT
        assert result.mitigation_required is True
        assert any("engagement" in issue.lower() for issue in result.issues_identified)
    
    def test_assess_algorithmic_harm_bias(self, osa_compliance):
        """Test detection of algorithmic bias."""
        result = osa_compliance.assess_algorithmic_harm({
            "bias_score": 0.7,
        })
        
        assert result.harm_level == AlgorithmicHarmLevel.SIGNIFICANT
        assert result.mitigation_required is True
        assert any("bias" in issue.lower() for issue in result.issues_identified)
    
    def test_assess_algorithmic_harm_addictive_patterns(self, osa_compliance):
        """Test detection of addictive design patterns."""
        result = osa_compliance.assess_algorithmic_harm({
            "uses_infinite_scroll": True,
            "autoplay_enabled": True,
        })
        
        assert result.harm_level >= AlgorithmicHarmLevel.MODERATE
        assert any("excessive use" in issue.lower() for issue in result.issues_identified)
    
    def test_assess_algorithmic_harm_child_safety(self, osa_compliance):
        """Test detection of child safety issues."""
        result = osa_compliance.assess_algorithmic_harm({
            "child_users_present": True,
            "child_safety_mode": False,
        })
        
        assert result.harm_level == AlgorithmicHarmLevel.SEVERE
        assert result.mitigation_required is True
        assert any("child" in issue.lower() for issue in result.issues_identified)
    
    def test_handle_user_complaint(self, osa_compliance):
        """Test user complaint handling."""
        complaint = {
            "user_id": "user123",
            "content_id": "content456",
            "reason": "Inappropriate content",
            "resolved": True,
        }
        
        result = osa_compliance.handle_user_complaint(complaint, response_time_hours=2.5)
        
        assert result["status"] == "resolved"
        assert result["response_time_hours"] == 2.5
        assert "complaint_id" in result
        assert osa_compliance._metrics["user_complaints"] == 1
        assert osa_compliance._metrics["complaints_resolved"] == 1
    
    def test_metrics_tracking(self, osa_compliance):
        """Test that metrics are properly tracked."""
        # Perform various operations
        osa_compliance.assess_illegal_content_risk("terrorism content", {})
        osa_compliance.assess_illegal_content_risk("fraud scheme", {})
        osa_compliance.detect_csae_content("child abuse", {})
        osa_compliance.assess_child_user_risk(10, "adult_content")
        
        # Check metrics
        assert osa_compliance._metrics["total_assessments"] == 2
        assert osa_compliance._metrics["illegal_content_detected"] >= 2
        assert osa_compliance._metrics["terrorism_content"] >= 1
        assert osa_compliance._metrics["fraud_content"] >= 1
        assert osa_compliance._metrics["csae_detections"] >= 1
        assert osa_compliance._metrics["child_users_protected"] >= 1
    
    def test_transparency_report_metrics(self, osa_compliance):
        """Test transparency report includes all metrics."""
        # Generate activity
        osa_compliance.assess_illegal_content_risk("terrorism", {})
        osa_compliance.detect_csae_content("child abuse", {})
        osa_compliance.handle_user_complaint({"resolved": True}, 1.5)
        osa_compliance.handle_user_complaint({"resolved": True}, 2.5)
        
        report = osa_compliance.generate_transparency_report("2024")
        
        assert report.terrorism_content_removed >= 1
        assert report.csae_detections >= 1
        assert report.user_complaints_received >= 2
        assert report.complaints_resolved >= 2
        assert report.average_response_time_hours > 0


class TestDataclasses:
    """Test dataclass instantiation and defaults."""
    
    def test_osa_risk_assessment_creation(self):
        """Test OSARiskAssessment dataclass."""
        assessment = OSARiskAssessment(
            risk_level=RiskLevel.HIGH,
            illegal_content_detected=True,
            content_categories=["terrorism"],
            recommended_action=Action.REMOVE,
            explanation="Test",
            confidence=0.9,
        )
        
        assert assessment.risk_level == RiskLevel.HIGH
        assert assessment.assessment_id is not None
        assert isinstance(assessment.timestamp, datetime)
    
    def test_csae_detection_result_creation(self):
        """Test CSAEDetectionResult dataclass."""
        result = CSAEDetectionResult(
            csae_detected=True,
            confidence=0.95,
            indicators=["test"],
            recommended_action=Action.REPORT_AUTHORITIES,
            requires_law_enforcement_report=True,
        )
        
        assert result.csae_detected is True
        assert result.detection_id is not None
    
    def test_transparency_report_creation(self):
        """Test OSATransparencyReport dataclass."""
        report = OSATransparencyReport(
            reporting_period="Q1 2024",
            total_content_assessments=100,
            illegal_content_detected=10,
            content_removed=8,
            law_enforcement_reports=2,
            user_complaints_received=50,
            complaints_resolved=45,
            average_response_time_hours=2.5,
        )
        
        assert report.reporting_period == "Q1 2024"
        assert report.report_id is not None
        assert isinstance(report.generated_at, datetime)
