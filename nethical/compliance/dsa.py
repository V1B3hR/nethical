"""Digital Services Act (DSA) Compliance Module for Nethical.

This module provides comprehensive EU Digital Services Act (DSA)
compliance validation capabilities including:
- Content Moderation (Articles 14-16)
- Transparency Obligations (Articles 15, 24, 42)
- Risk Assessment for VLOPs/VLOSEs (Article 34)
- Crisis Response Mechanism (Article 36)

Adheres to the 25 Fundamental Laws:
- Law 6: Decision Authority - Clear decision-making authority
- Law 10: Reasoning Transparency - Explainable decision-making
- Law 15: Audit Compliance - Cooperation with auditing
- Law 21: Human Safety Priority - Primacy of human safety

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlatformType(str, Enum):
    """DSA platform categories."""
    
    HOSTING = "hosting"  # Hosting services (Article 14)
    ONLINE_PLATFORM = "online_platform"  # Online platforms (Article 15-16)
    VLOP = "vlop"  # Very Large Online Platform (45M+ EU users)
    VLOSE = "vlose"  # Very Large Online Search Engine


class ModerationAction(str, Enum):
    """Content moderation action types."""
    
    NO_ACTION = "no_action"
    CONTENT_REMOVED = "content_removed"
    CONTENT_RESTRICTED = "content_restricted"
    ACCOUNT_SUSPENDED = "account_suspended"
    CONTENT_LABELED = "content_labeled"


class CrisisType(str, Enum):
    """Types of public emergencies/crises."""
    
    PUBLIC_HEALTH = "public_health"
    PUBLIC_SECURITY = "public_security"
    NATURAL_DISASTER = "natural_disaster"
    ARMED_CONFLICT = "armed_conflict"


class SystemicRiskCategory(str, Enum):
    """Categories of systemic risks (Article 34)."""
    
    ILLEGAL_CONTENT = "illegal_content"
    FUNDAMENTAL_RIGHTS = "fundamental_rights"
    DEMOCRATIC_PROCESS = "democratic_process"
    GENDER_BASED_VIOLENCE = "gender_based_violence"
    CHILD_SAFETY = "child_safety"
    PUBLIC_HEALTH = "public_health"


class ComplaintStatus(str, Enum):
    """Internal complaint handling status."""
    
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


@dataclass
class ContentNotice:
    """Notice of allegedly illegal content (Article 16)."""
    
    notice_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reporter_id: str = ""
    content_id: str = ""
    alleged_illegality: str = ""
    explanation: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Optional fields
    reporter_contact: Optional[str] = None
    legal_basis: Optional[str] = None
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class NoticeProcessingResult:
    """Result of processing a content notice."""
    
    notice_id: str
    action_taken: ModerationAction
    decision_timestamp: datetime
    processing_time_hours: float
    explanation: str
    
    # Statement of reasons provided to user
    statement_of_reasons: Optional[StatementOfReasons] = None


@dataclass
class StatementOfReasons:
    """Statement of reasons for content moderation decision (Article 17)."""
    
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content_id: str = ""
    action_taken: ModerationAction = ModerationAction.NO_ACTION
    legal_basis: str = ""
    facts_considered: List[str] = field(default_factory=list)
    automated_decision: bool = False
    complaint_options: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Additional information
    territorial_scope: str = "EU"
    redress_period_days: int = 30


@dataclass
class UserComplaint:
    """Internal complaint against moderation decision (Article 20)."""
    
    complaint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    decision_id: str = ""
    complaint_reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Optional supporting information
    additional_evidence: List[str] = field(default_factory=list)


@dataclass
class ComplaintResolution:
    """Resolution of internal complaint."""
    
    complaint_id: str
    status: ComplaintStatus
    resolution_explanation: str
    action_taken: Optional[ModerationAction] = None
    resolved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_hours: float = 0.0


@dataclass
class SystemicRiskAssessment:
    """Systemic risk assessment for VLOPs/VLOSEs (Article 34)."""
    
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    assessment_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    risks_identified: List[Dict[str, Any]] = field(default_factory=list)
    mitigation_measures: List[str] = field(default_factory=list)
    residual_risks: List[str] = field(default_factory=list)
    next_assessment_date: Optional[datetime] = None
    
    # Risk categories
    risk_categories: List[SystemicRiskCategory] = field(default_factory=list)


@dataclass
class DSATransparencyReport:
    """Transparency report (Article 15, 24, 42)."""
    
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reporting_period: str = ""
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Content moderation metrics
    total_content_notices: int = 0
    notices_actioned: int = 0
    content_removed: int = 0
    content_restricted: int = 0
    accounts_suspended: int = 0
    
    # Complaint metrics
    complaints_received: int = 0
    complaints_resolved: int = 0
    average_complaint_resolution_hours: float = 0.0
    
    # Automated decision metrics
    automated_decisions: int = 0
    automated_decisions_overturned: int = 0
    
    # Recommender system transparency
    recommender_systems: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RecommenderAssessment:
    """Assessment of recommender system (Article 27)."""
    
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_name: str = ""
    main_parameters: List[str] = field(default_factory=list)
    user_control_options: List[str] = field(default_factory=list)
    transparency_level: str = "basic"  # basic, detailed, full
    risk_level: str = "low"  # low, medium, high
    
    # Compliance checks
    user_can_modify: bool = False
    alternative_available: bool = False
    explanation_provided: bool = False


@dataclass
class CrisisResponse:
    """Crisis response protocol activation (Article 36)."""
    
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    crisis_type: CrisisType = CrisisType.PUBLIC_HEALTH
    activated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    measures_activated: List[str] = field(default_factory=list)
    coordination_entities: List[str] = field(default_factory=list)
    status: str = "active"  # active, monitoring, resolved
    
    # Impact metrics
    content_flagged: int = 0
    urgent_removals: int = 0


class DSACompliance:
    """Digital Services Act compliance implementation.
    
    Implements content moderation, transparency reporting, risk assessment,
    and crisis response mechanisms for DSA compliance.
    """
    
    def __init__(self, platform_type: PlatformType, monthly_active_users: int):
        """Initialize DSA compliance validator.
        
        Args:
            platform_type: Type of platform (HOSTING, ONLINE_PLATFORM, VLOP, VLOSE)
            monthly_active_users: EU monthly active users (VLOP/VLOSE threshold: 45M)
        """
        self.platform_type = platform_type
        self.monthly_active_users = monthly_active_users
        
        # Determine if platform is VLOP/VLOSE
        self.is_vlop = monthly_active_users >= 45_000_000 and platform_type in [
            PlatformType.VLOP, PlatformType.ONLINE_PLATFORM
        ]
        
        # Metrics for transparency reporting
        self._metrics = {
            "content_notices": 0,
            "notices_actioned": 0,
            "content_removed": 0,
            "content_restricted": 0,
            "accounts_suspended": 0,
            "complaints_received": 0,
            "complaints_resolved": 0,
            "complaint_resolution_times": [],
            "automated_decisions": 0,
            "automated_decisions_overturned": 0,
        }
        
        # Active crisis protocols
        self._active_crises: Dict[str, CrisisResponse] = {}
        
        logger.info(
            f"Initialized DSA Compliance for {platform_type.value} "
            f"platform with {monthly_active_users} EU users "
            f"(VLOP: {self.is_vlop})"
        )
    
    def process_content_notice(self, notice: ContentNotice) -> NoticeProcessingResult:
        """Process notice of allegedly illegal content (Article 16).
        
        Args:
            notice: Content notice from user
            
        Returns:
            Notice processing result with action taken
        """
        self._metrics["content_notices"] += 1
        
        # Simulate processing time (would be actual time in production)
        processing_time = 2.5  # hours
        
        # Analyze notice and determine action
        action_taken = ModerationAction.NO_ACTION
        explanation = "Content reviewed and found compliant"
        
        # Check alleged illegality
        if "illegal" in notice.alleged_illegality.lower():
            # Serious illegality claims
            if any(term in notice.alleged_illegality.lower() 
                   for term in ["terrorism", "csae", "violence"]):
                action_taken = ModerationAction.CONTENT_REMOVED
                explanation = "Content removed due to serious illegal content"
                self._metrics["content_removed"] += 1
            else:
                action_taken = ModerationAction.CONTENT_RESTRICTED
                explanation = "Content restricted pending further review"
                self._metrics["content_restricted"] += 1
            
            self._metrics["notices_actioned"] += 1
        
        # Generate statement of reasons
        statement = self._generate_statement_of_reasons(
            content_id=notice.content_id,
            action=action_taken,
            legal_basis=notice.legal_basis or "DSA Article 14",
            facts=[notice.alleged_illegality, notice.explanation],
            automated=False
        )
        
        return NoticeProcessingResult(
            notice_id=notice.notice_id,
            action_taken=action_taken,
            decision_timestamp=datetime.now(timezone.utc),
            processing_time_hours=processing_time,
            explanation=explanation,
            statement_of_reasons=statement,
        )
    
    def generate_statement_of_reasons(
        self, decision: Dict[str, Any]
    ) -> StatementOfReasons:
        """Generate statement of reasons for moderation decision (Article 17).
        
        Args:
            decision: Moderation decision details
            
        Returns:
            Statement of reasons
        """
        return self._generate_statement_of_reasons(
            content_id=decision.get("content_id", ""),
            action=decision.get("action", ModerationAction.NO_ACTION),
            legal_basis=decision.get("legal_basis", "Terms of Service"),
            facts=decision.get("facts", []),
            automated=decision.get("automated", False)
        )
    
    def _generate_statement_of_reasons(
        self,
        content_id: str,
        action: ModerationAction,
        legal_basis: str,
        facts: List[str],
        automated: bool
    ) -> StatementOfReasons:
        """Internal method to generate statement of reasons."""
        # Define complaint options
        complaint_options = {
            "internal_complaint": "Submit complaint via platform complaint system",
            "out_of_court_dispute": "Contact certified dispute resolution body",
            "judicial_redress": "File complaint with competent authority"
        }
        
        if automated:
            self._metrics["automated_decisions"] += 1
        
        return StatementOfReasons(
            content_id=content_id,
            action_taken=action,
            legal_basis=legal_basis,
            facts_considered=facts,
            automated_decision=automated,
            complaint_options=complaint_options,
        )
    
    def handle_internal_complaint(self, complaint: UserComplaint) -> ComplaintResolution:
        """Handle internal complaint against moderation decision (Article 20).
        
        Args:
            complaint: User complaint
            
        Returns:
            Complaint resolution
        """
        self._metrics["complaints_received"] += 1
        
        # Simulate complaint review process
        processing_time = 48.0  # hours - DSA requires timely handling
        
        # Determine resolution
        status = ComplaintStatus.RESOLVED
        action_taken = None
        explanation = "Complaint reviewed. Original decision upheld."
        
        # Check if complaint has merit
        if "error" in complaint.complaint_reason.lower() or \
           "mistake" in complaint.complaint_reason.lower():
            status = ComplaintStatus.RESOLVED
            action_taken = ModerationAction.NO_ACTION  # Reverse original decision
            explanation = "Complaint upheld. Original decision reversed."
            
            # Track automated decision overturn
            # (would check if original was automated in production)
            self._metrics["automated_decisions_overturned"] += 1
        
        if status == ComplaintStatus.RESOLVED:
            self._metrics["complaints_resolved"] += 1
            self._metrics["complaint_resolution_times"].append(processing_time)
        
        return ComplaintResolution(
            complaint_id=complaint.complaint_id,
            status=status,
            resolution_explanation=explanation,
            action_taken=action_taken,
            processing_time_hours=processing_time,
        )
    
    def assess_systemic_risk(self) -> SystemicRiskAssessment:
        """Assess systemic risks for VLOPs/VLOSEs (Article 34).
        
        Returns:
            Systemic risk assessment
        """
        if not self.is_vlop:
            logger.warning("Systemic risk assessment only required for VLOPs/VLOSEs")
        
        risks_identified = []
        risk_categories = []
        mitigation_measures = []
        residual_risks = []
        
        # Assess illegal content dissemination risk
        risks_identified.append({
            "category": SystemicRiskCategory.ILLEGAL_CONTENT,
            "description": "Risk of rapid dissemination of illegal content",
            "severity": "medium",
            "likelihood": "medium"
        })
        risk_categories.append(SystemicRiskCategory.ILLEGAL_CONTENT)
        mitigation_measures.append("Automated content filtering with human review")
        
        # Assess fundamental rights risks
        risks_identified.append({
            "category": SystemicRiskCategory.FUNDAMENTAL_RIGHTS,
            "description": "Algorithmic systems may impact freedom of expression",
            "severity": "medium",
            "likelihood": "low"
        })
        risk_categories.append(SystemicRiskCategory.FUNDAMENTAL_RIGHTS)
        mitigation_measures.append("Human oversight of automated decisions")
        residual_risks.append("Some false positives in content moderation")
        
        # Assess democratic process risks (for large platforms)
        if self.monthly_active_users > 100_000_000:
            risks_identified.append({
                "category": SystemicRiskCategory.DEMOCRATIC_PROCESS,
                "description": "Risk of manipulation of civic discourse",
                "severity": "high",
                "likelihood": "medium"
            })
            risk_categories.append(SystemicRiskCategory.DEMOCRATIC_PROCESS)
            mitigation_measures.append("Election integrity monitoring")
            mitigation_measures.append("Fact-checking partnerships")
        
        # Assess child safety risks
        risks_identified.append({
            "category": SystemicRiskCategory.CHILD_SAFETY,
            "description": "Risk of exposure to harmful content for minors",
            "severity": "high",
            "likelihood": "medium"
        })
        risk_categories.append(SystemicRiskCategory.CHILD_SAFETY)
        mitigation_measures.append("Age verification and parental controls")
        
        # Calculate next assessment date (annually as per DSA)
        from datetime import timedelta
        next_assessment = datetime.now(timezone.utc) + timedelta(days=365)
        
        return SystemicRiskAssessment(
            risks_identified=risks_identified,
            mitigation_measures=mitigation_measures,
            residual_risks=residual_risks,
            next_assessment_date=next_assessment,
            risk_categories=list(set(risk_categories)),
        )
    
    def generate_transparency_report(self, period: str) -> DSATransparencyReport:
        """Generate transparency report (Article 15, 24, 42).
        
        Args:
            period: Reporting period (e.g., "Q1 2024")
            
        Returns:
            DSA transparency report
        """
        # Calculate average complaint resolution time
        avg_complaint_time = 0.0
        if self._metrics["complaint_resolution_times"]:
            avg_complaint_time = sum(self._metrics["complaint_resolution_times"]) / \
                                len(self._metrics["complaint_resolution_times"])
        
        return DSATransparencyReport(
            reporting_period=period,
            total_content_notices=self._metrics["content_notices"],
            notices_actioned=self._metrics["notices_actioned"],
            content_removed=self._metrics["content_removed"],
            content_restricted=self._metrics["content_restricted"],
            accounts_suspended=self._metrics["accounts_suspended"],
            complaints_received=self._metrics["complaints_received"],
            complaints_resolved=self._metrics["complaints_resolved"],
            average_complaint_resolution_hours=avg_complaint_time,
            automated_decisions=self._metrics["automated_decisions"],
            automated_decisions_overturned=self._metrics["automated_decisions_overturned"],
        )
    
    def activate_crisis_protocol(self, crisis_type: CrisisType) -> CrisisResponse:
        """Activate crisis response mechanism (Article 36).
        
        Args:
            crisis_type: Type of crisis
            
        Returns:
            Crisis response details
        """
        measures = []
        coordination = []
        
        # Define crisis-specific measures
        if crisis_type == CrisisType.PUBLIC_HEALTH:
            measures.extend([
                "Enhanced monitoring of health misinformation",
                "Prioritized review of public health content",
                "Partnership with health authorities for accurate information"
            ])
            coordination.append("WHO")
            coordination.append("National health authorities")
        
        elif crisis_type == CrisisType.PUBLIC_SECURITY:
            measures.extend([
                "Expedited removal of violence-inciting content",
                "Real-time monitoring of coordinated harmful activity",
                "Collaboration with law enforcement"
            ])
            coordination.append("Law enforcement agencies")
            coordination.append("Digital Services Coordinator")
        
        elif crisis_type == CrisisType.ARMED_CONFLICT:
            measures.extend([
                "Monitoring of war propaganda and disinformation",
                "Protection of civilian information",
                "Coordination with international bodies"
            ])
            coordination.append("UN agencies")
            coordination.append("International humanitarian organizations")
        
        else:  # Natural disaster
            measures.extend([
                "Verification of emergency information",
                "Prevention of disaster-related scams",
                "Promotion of official emergency communications"
            ])
            coordination.append("Emergency services")
            coordination.append("Government disaster response agencies")
        
        response = CrisisResponse(
            crisis_type=crisis_type,
            measures_activated=measures,
            coordination_entities=coordination,
        )
        
        # Store active crisis
        self._active_crises[response.response_id] = response
        
        logger.info(
            f"Crisis protocol activated: {crisis_type.value} "
            f"(ID: {response.response_id})"
        )
        
        return response
    
    def assess_recommender_system(self, algorithm: Dict[str, Any]) -> RecommenderAssessment:
        """Assess recommender system for transparency (Article 27).
        
        Args:
            algorithm: Algorithm parameters and configuration
            
        Returns:
            Recommender system assessment
        """
        system_name = algorithm.get("name", "unnamed_recommender")
        
        # Extract main parameters
        main_parameters = []
        if algorithm.get("uses_engagement"):
            main_parameters.append("User engagement metrics")
        if algorithm.get("uses_personalization"):
            main_parameters.append("Personalization based on user history")
        if algorithm.get("uses_popularity"):
            main_parameters.append("Content popularity")
        if algorithm.get("uses_recency"):
            main_parameters.append("Content recency")
        
        # Check user control options
        user_controls = []
        if algorithm.get("user_can_disable"):
            user_controls.append("Disable personalization")
        if algorithm.get("alternative_available"):
            user_controls.append("Switch to chronological feed")
        if algorithm.get("user_can_customize"):
            user_controls.append("Customize recommendation preferences")
        
        # Determine transparency level
        transparency = "basic"
        if algorithm.get("detailed_explanations"):
            transparency = "detailed"
        if algorithm.get("full_parameter_disclosure"):
            transparency = "full"
        
        # Assess risk level
        risk = "low"
        if algorithm.get("engagement_optimization"):
            risk = "medium"
        if algorithm.get("no_human_oversight"):
            risk = "high"
        
        return RecommenderAssessment(
            system_name=system_name,
            main_parameters=main_parameters,
            user_control_options=user_controls,
            transparency_level=transparency,
            risk_level=risk,
            user_can_modify=algorithm.get("user_can_customize", False),
            alternative_available=algorithm.get("alternative_available", False),
            explanation_provided=algorithm.get("detailed_explanations", False),
        )
    
    def get_active_crises(self) -> List[CrisisResponse]:
        """Get list of active crisis responses.
        
        Returns:
            List of active crisis responses
        """
        return [
            crisis for crisis in self._active_crises.values()
            if crisis.status == "active"
        ]
    
    def resolve_crisis(self, response_id: str) -> bool:
        """Mark a crisis response as resolved.
        
        Args:
            response_id: Crisis response ID
            
        Returns:
            True if crisis was found and resolved
        """
        if response_id in self._active_crises:
            self._active_crises[response_id].status = "resolved"
            logger.info(f"Crisis response {response_id} marked as resolved")
            return True
        return False
