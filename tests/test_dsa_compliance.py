"""
Tests for Digital Services Act (DSA) Compliance Module.

Tests content moderation, transparency reporting, systemic risk assessment,
and crisis response protocols.
"""

import pytest
from datetime import datetime

from nethical.compliance.dsa import (
    PlatformType,
    ModerationAction,
    CrisisType,
    SystemicRiskCategory,
    ComplaintStatus,
    ContentNotice,
    NoticeProcessingResult,
    StatementOfReasons,
    UserComplaint,
    ComplaintResolution,
    SystemicRiskAssessment,
    DSATransparencyReport,
    RecommenderAssessment,
    CrisisResponse,
    DSACompliance,
)


class TestPlatformType:
    """Tests for platform type enumeration."""
    
    def test_platform_types_exist(self):
        """Test all platform types are defined."""
        assert PlatformType.HOSTING.value == "hosting"
        assert PlatformType.ONLINE_PLATFORM.value == "online_platform"
        assert PlatformType.VLOP.value == "vlop"
        assert PlatformType.VLOSE.value == "vlose"


class TestDSACompliance:
    """Tests for DSA compliance validator."""
    
    @pytest.fixture
    def dsa_compliance_platform(self):
        """Create DSA compliance instance for regular platform."""
        return DSACompliance(
            platform_type=PlatformType.ONLINE_PLATFORM,
            monthly_active_users=10_000_000  # 10M users
        )
    
    @pytest.fixture
    def dsa_compliance_vlop(self):
        """Create DSA compliance instance for VLOP."""
        return DSACompliance(
            platform_type=PlatformType.VLOP,
            monthly_active_users=50_000_000  # 50M users - VLOP threshold
        )
    
    def test_initialization_regular_platform(self, dsa_compliance_platform):
        """Test compliance validator initializes correctly for regular platform."""
        assert dsa_compliance_platform.platform_type == PlatformType.ONLINE_PLATFORM
        assert dsa_compliance_platform.monthly_active_users == 10_000_000
        assert dsa_compliance_platform.is_vlop is False
    
    def test_initialization_vlop(self, dsa_compliance_vlop):
        """Test VLOP platform is correctly identified."""
        assert dsa_compliance_vlop.platform_type == PlatformType.VLOP
        assert dsa_compliance_vlop.is_vlop is True
    
    def test_process_content_notice_safe(self, dsa_compliance_platform):
        """Test processing of notice with safe content."""
        notice = ContentNotice(
            reporter_id="user123",
            content_id="content456",
            alleged_illegality="Spam content",
            explanation="This is spam"
        )
        
        result = dsa_compliance_platform.process_content_notice(notice)
        
        assert isinstance(result, NoticeProcessingResult)
        assert result.notice_id == notice.notice_id
        assert result.action_taken == ModerationAction.NO_ACTION
        assert result.processing_time_hours > 0
    
    def test_process_content_notice_illegal(self, dsa_compliance_platform):
        """Test processing of notice with illegal content."""
        notice = ContentNotice(
            reporter_id="user123",
            content_id="content789",
            alleged_illegality="illegal terrorism content",
            explanation="Contains terrorist propaganda"
        )
        
        result = dsa_compliance_platform.process_content_notice(notice)
        
        assert result.action_taken == ModerationAction.CONTENT_REMOVED
        assert dsa_compliance_platform._metrics["content_removed"] >= 1
        assert dsa_compliance_platform._metrics["notices_actioned"] >= 1
    
    def test_process_content_notice_generates_statement(self, dsa_compliance_platform):
        """Test that processing generates statement of reasons."""
        notice = ContentNotice(
            reporter_id="user123",
            content_id="content999",
            alleged_illegality="illegal content",
            explanation="Violates terms"
        )
        
        result = dsa_compliance_platform.process_content_notice(notice)
        
        assert result.statement_of_reasons is not None
        assert isinstance(result.statement_of_reasons, StatementOfReasons)
        assert result.statement_of_reasons.content_id == notice.content_id
    
    def test_generate_statement_of_reasons(self, dsa_compliance_platform):
        """Test statement of reasons generation."""
        decision = {
            "content_id": "content123",
            "action": ModerationAction.CONTENT_REMOVED,
            "legal_basis": "DSA Article 14",
            "facts": ["Contains illegal content", "Violates community standards"],
            "automated": False
        }
        
        statement = dsa_compliance_platform.generate_statement_of_reasons(decision)
        
        assert isinstance(statement, StatementOfReasons)
        assert statement.content_id == "content123"
        assert statement.action_taken == ModerationAction.CONTENT_REMOVED
        assert statement.legal_basis == "DSA Article 14"
        assert len(statement.facts_considered) == 2
        assert statement.automated_decision is False
        assert len(statement.complaint_options) > 0
    
    def test_statement_of_reasons_automated(self, dsa_compliance_platform):
        """Test statement tracks automated decisions."""
        decision = {
            "content_id": "content456",
            "action": ModerationAction.CONTENT_RESTRICTED,
            "legal_basis": "Automated filter",
            "facts": ["Pattern match"],
            "automated": True
        }
        
        initial_automated = dsa_compliance_platform._metrics["automated_decisions"]
        statement = dsa_compliance_platform.generate_statement_of_reasons(decision)
        
        assert statement.automated_decision is True
        assert dsa_compliance_platform._metrics["automated_decisions"] == initial_automated + 1
    
    def test_handle_internal_complaint(self, dsa_compliance_platform):
        """Test internal complaint handling."""
        complaint = UserComplaint(
            user_id="user789",
            decision_id="decision123",
            complaint_reason="I disagree with the decision"
        )
        
        resolution = dsa_compliance_platform.handle_internal_complaint(complaint)
        
        assert isinstance(resolution, ComplaintResolution)
        assert resolution.complaint_id == complaint.complaint_id
        assert resolution.status in [ComplaintStatus.RESOLVED, ComplaintStatus.REJECTED]
        assert resolution.processing_time_hours > 0
        assert dsa_compliance_platform._metrics["complaints_received"] >= 1
    
    def test_handle_complaint_with_merit(self, dsa_compliance_platform):
        """Test complaint that has merit."""
        complaint = UserComplaint(
            user_id="user789",
            decision_id="decision456",
            complaint_reason="This was an error in the automated system"
        )
        
        resolution = dsa_compliance_platform.handle_internal_complaint(complaint)
        
        assert resolution.status == ComplaintStatus.RESOLVED
        assert resolution.action_taken is not None
        assert "upheld" in resolution.resolution_explanation.lower()
        assert dsa_compliance_platform._metrics["complaints_resolved"] >= 1
    
    def test_assess_systemic_risk_regular_platform(self, dsa_compliance_platform):
        """Test systemic risk assessment for regular platform."""
        assessment = dsa_compliance_platform.assess_systemic_risk()
        
        assert isinstance(assessment, SystemicRiskAssessment)
        assert len(assessment.risks_identified) > 0
        assert len(assessment.mitigation_measures) > 0
        assert assessment.assessment_id is not None
        assert isinstance(assessment.assessment_date, datetime)
    
    def test_assess_systemic_risk_vlop(self, dsa_compliance_vlop):
        """Test systemic risk assessment for VLOP."""
        assessment = dsa_compliance_vlop.assess_systemic_risk()
        
        assert isinstance(assessment, SystemicRiskAssessment)
        assert len(assessment.risks_identified) >= 3
        assert len(assessment.risk_categories) > 0
        assert SystemicRiskCategory.ILLEGAL_CONTENT in assessment.risk_categories
        assert SystemicRiskCategory.CHILD_SAFETY in assessment.risk_categories
    
    def test_assess_systemic_risk_includes_categories(self, dsa_compliance_vlop):
        """Test that systemic risk assessment includes required categories."""
        assessment = dsa_compliance_vlop.assess_systemic_risk()
        
        # Check for key risk categories
        category_names = [risk["category"] for risk in assessment.risks_identified]
        assert SystemicRiskCategory.ILLEGAL_CONTENT in category_names
        assert SystemicRiskCategory.FUNDAMENTAL_RIGHTS in category_names
        assert SystemicRiskCategory.CHILD_SAFETY in category_names
    
    def test_generate_transparency_report(self, dsa_compliance_platform):
        """Test transparency report generation."""
        # Generate some activity
        notice1 = ContentNotice(
            reporter_id="user1",
            content_id="content1",
            alleged_illegality="illegal content",
            explanation="test"
        )
        dsa_compliance_platform.process_content_notice(notice1)
        
        complaint = UserComplaint(
            user_id="user2",
            decision_id="decision1",
            complaint_reason="test"
        )
        dsa_compliance_platform.handle_internal_complaint(complaint)
        
        report = dsa_compliance_platform.generate_transparency_report("Q1 2024")
        
        assert isinstance(report, DSATransparencyReport)
        assert report.reporting_period == "Q1 2024"
        assert report.total_content_notices >= 1
        assert report.complaints_received >= 1
        assert report.report_id is not None
    
    def test_activate_crisis_protocol_public_health(self, dsa_compliance_platform):
        """Test crisis protocol activation for public health."""
        response = dsa_compliance_platform.activate_crisis_protocol(
            CrisisType.PUBLIC_HEALTH
        )
        
        assert isinstance(response, CrisisResponse)
        assert response.crisis_type == CrisisType.PUBLIC_HEALTH
        assert len(response.measures_activated) > 0
        assert len(response.coordination_entities) > 0
        assert response.status == "active"
        assert "health" in " ".join(response.measures_activated).lower()
    
    def test_activate_crisis_protocol_public_security(self, dsa_compliance_platform):
        """Test crisis protocol activation for public security."""
        response = dsa_compliance_platform.activate_crisis_protocol(
            CrisisType.PUBLIC_SECURITY
        )
        
        assert response.crisis_type == CrisisType.PUBLIC_SECURITY
        assert "law enforcement" in " ".join(response.coordination_entities).lower() or \
               "law enforcement" in str(response.coordination_entities).lower()
    
    def test_activate_crisis_protocol_armed_conflict(self, dsa_compliance_platform):
        """Test crisis protocol activation for armed conflict."""
        response = dsa_compliance_platform.activate_crisis_protocol(
            CrisisType.ARMED_CONFLICT
        )
        
        assert response.crisis_type == CrisisType.ARMED_CONFLICT
        assert len(response.measures_activated) > 0
    
    def test_activate_crisis_protocol_natural_disaster(self, dsa_compliance_platform):
        """Test crisis protocol activation for natural disaster."""
        response = dsa_compliance_platform.activate_crisis_protocol(
            CrisisType.NATURAL_DISASTER
        )
        
        assert response.crisis_type == CrisisType.NATURAL_DISASTER
        assert "emergency" in " ".join(response.measures_activated).lower()
    
    def test_get_active_crises(self, dsa_compliance_platform):
        """Test retrieving active crises."""
        # Activate multiple crises
        response1 = dsa_compliance_platform.activate_crisis_protocol(
            CrisisType.PUBLIC_HEALTH
        )
        response2 = dsa_compliance_platform.activate_crisis_protocol(
            CrisisType.PUBLIC_SECURITY
        )
        
        active_crises = dsa_compliance_platform.get_active_crises()
        
        assert len(active_crises) >= 2
        assert all(crisis.status == "active" for crisis in active_crises)
    
    def test_resolve_crisis(self, dsa_compliance_platform):
        """Test resolving a crisis."""
        response = dsa_compliance_platform.activate_crisis_protocol(
            CrisisType.PUBLIC_HEALTH
        )
        
        # Resolve the crisis
        result = dsa_compliance_platform.resolve_crisis(response.response_id)
        
        assert result is True
        assert dsa_compliance_platform._active_crises[response.response_id].status == "resolved"
        
        # Check it's no longer in active crises
        active_crises = dsa_compliance_platform.get_active_crises()
        assert response.response_id not in [c.response_id for c in active_crises]
    
    def test_resolve_nonexistent_crisis(self, dsa_compliance_platform):
        """Test resolving a nonexistent crisis."""
        result = dsa_compliance_platform.resolve_crisis("nonexistent_id")
        assert result is False
    
    def test_assess_recommender_system_basic(self, dsa_compliance_platform):
        """Test recommender system assessment with basic parameters."""
        algorithm = {
            "name": "content_recommender",
            "uses_engagement": True,
            "uses_popularity": True,
        }
        
        assessment = dsa_compliance_platform.assess_recommender_system(algorithm)
        
        assert isinstance(assessment, RecommenderAssessment)
        assert assessment.system_name == "content_recommender"
        assert len(assessment.main_parameters) >= 2
        assert assessment.transparency_level == "basic"
    
    def test_assess_recommender_system_with_controls(self, dsa_compliance_platform):
        """Test recommender system with user controls."""
        algorithm = {
            "name": "personalized_feed",
            "uses_personalization": True,
            "user_can_disable": True,
            "alternative_available": True,
            "user_can_customize": True,
        }
        
        assessment = dsa_compliance_platform.assess_recommender_system(algorithm)
        
        assert len(assessment.user_control_options) >= 2
        assert assessment.user_can_modify is True
        assert assessment.alternative_available is True
    
    def test_assess_recommender_system_transparency(self, dsa_compliance_platform):
        """Test recommender system transparency levels."""
        algorithm_detailed = {
            "name": "test",
            "detailed_explanations": True,
        }
        
        assessment = dsa_compliance_platform.assess_recommender_system(algorithm_detailed)
        assert assessment.transparency_level == "detailed"
        
        algorithm_full = {
            "name": "test",
            "full_parameter_disclosure": True,
        }
        
        assessment = dsa_compliance_platform.assess_recommender_system(algorithm_full)
        assert assessment.transparency_level == "full"
    
    def test_assess_recommender_system_risk(self, dsa_compliance_platform):
        """Test recommender system risk assessment."""
        algorithm_medium = {
            "name": "test",
            "engagement_optimization": True,
        }
        
        assessment = dsa_compliance_platform.assess_recommender_system(algorithm_medium)
        assert assessment.risk_level == "medium"
        
        algorithm_high = {
            "name": "test",
            "no_human_oversight": True,
        }
        
        assessment = dsa_compliance_platform.assess_recommender_system(algorithm_high)
        assert assessment.risk_level == "high"
    
    def test_metrics_tracking(self, dsa_compliance_platform):
        """Test that metrics are properly tracked."""
        # Process notices
        notice1 = ContentNotice(
            reporter_id="user1",
            content_id="content1",
            alleged_illegality="illegal terrorism",
            explanation="test"
        )
        dsa_compliance_platform.process_content_notice(notice1)
        
        notice2 = ContentNotice(
            reporter_id="user2",
            content_id="content2",
            alleged_illegality="illegal csae",
            explanation="test"
        )
        dsa_compliance_platform.process_content_notice(notice2)
        
        # Handle complaints
        complaint = UserComplaint(
            user_id="user3",
            decision_id="decision1",
            complaint_reason="error"
        )
        dsa_compliance_platform.handle_internal_complaint(complaint)
        
        # Check metrics
        assert dsa_compliance_platform._metrics["content_notices"] >= 2
        assert dsa_compliance_platform._metrics["content_removed"] >= 2
        assert dsa_compliance_platform._metrics["complaints_received"] >= 1
        assert dsa_compliance_platform._metrics["complaints_resolved"] >= 1


class TestDataclasses:
    """Test dataclass instantiation and defaults."""
    
    def test_content_notice_creation(self):
        """Test ContentNotice dataclass."""
        notice = ContentNotice(
            reporter_id="user123",
            content_id="content456",
            alleged_illegality="Illegal content",
            explanation="Test explanation"
        )
        
        assert notice.reporter_id == "user123"
        assert notice.notice_id is not None
        assert isinstance(notice.timestamp, datetime)
    
    def test_statement_of_reasons_creation(self):
        """Test StatementOfReasons dataclass."""
        statement = StatementOfReasons(
            content_id="content123",
            action_taken=ModerationAction.CONTENT_REMOVED,
            legal_basis="DSA Article 14",
            facts_considered=["Fact 1", "Fact 2"],
            automated_decision=False
        )
        
        assert statement.content_id == "content123"
        assert statement.decision_id is not None
        assert statement.redress_period_days == 30
    
    def test_user_complaint_creation(self):
        """Test UserComplaint dataclass."""
        complaint = UserComplaint(
            user_id="user456",
            decision_id="decision789",
            complaint_reason="I disagree"
        )
        
        assert complaint.user_id == "user456"
        assert complaint.complaint_id is not None
        assert isinstance(complaint.timestamp, datetime)
    
    def test_systemic_risk_assessment_creation(self):
        """Test SystemicRiskAssessment dataclass."""
        assessment = SystemicRiskAssessment(
            risks_identified=[{"test": "risk"}],
            mitigation_measures=["measure1"],
            residual_risks=["residual1"]
        )
        
        assert assessment.assessment_id is not None
        assert isinstance(assessment.assessment_date, datetime)
        assert len(assessment.risks_identified) == 1
    
    def test_dsa_transparency_report_creation(self):
        """Test DSATransparencyReport dataclass."""
        report = DSATransparencyReport(
            reporting_period="Q1 2024",
            total_content_notices=100,
            content_removed=50,
            complaints_received=25
        )
        
        assert report.reporting_period == "Q1 2024"
        assert report.report_id is not None
        assert isinstance(report.generated_at, datetime)
    
    def test_crisis_response_creation(self):
        """Test CrisisResponse dataclass."""
        response = CrisisResponse(
            crisis_type=CrisisType.PUBLIC_HEALTH,
            measures_activated=["measure1", "measure2"],
            coordination_entities=["WHO"]
        )
        
        assert response.crisis_type == CrisisType.PUBLIC_HEALTH
        assert response.response_id is not None
        assert response.status == "active"
        assert isinstance(response.activated_at, datetime)
