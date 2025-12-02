"""GDPR Compliance Validator for Nethical.

This module provides comprehensive GDPR (General Data Protection Regulation)
compliance validation capabilities including:
- Data Subject Rights validation (Articles 12-22)
- Lawful Basis assessment (Article 6)
- Data Protection by Design and Default (Article 25)
- Right to Explanation for automated decision-making (Article 22)

Adheres to the 25 Fundamental Laws:
- Law 10: Reasoning Transparency - Explainable decision-making
- Law 12: Limitation Disclosure - Disclosure of known limitations  
- Law 15: Audit Compliance - Cooperation with auditing
- Law 22: Digital Security - Protection of privacy

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GDPRArticle(str, Enum):
    """GDPR Articles relevant to AI systems."""
    
    ARTICLE_5 = "article_5"  # Principles of processing
    ARTICLE_6 = "article_6"  # Lawful basis
    ARTICLE_12 = "article_12"  # Transparent information
    ARTICLE_13 = "article_13"  # Information at collection
    ARTICLE_14 = "article_14"  # Information when not collected from subject
    ARTICLE_15 = "article_15"  # Right of access
    ARTICLE_16 = "article_16"  # Right to rectification
    ARTICLE_17 = "article_17"  # Right to erasure
    ARTICLE_18 = "article_18"  # Right to restriction
    ARTICLE_20 = "article_20"  # Right to data portability
    ARTICLE_21 = "article_21"  # Right to object
    ARTICLE_22 = "article_22"  # Automated decision-making
    ARTICLE_25 = "article_25"  # Data protection by design
    ARTICLE_32 = "article_32"  # Security of processing
    ARTICLE_33 = "article_33"  # Breach notification to authority
    ARTICLE_34 = "article_34"  # Breach notification to subject
    ARTICLE_35 = "article_35"  # DPIA


class DataSubjectRight(str, Enum):
    """Data Subject Rights under GDPR."""
    
    ACCESS = "access"  # Article 15
    RECTIFICATION = "rectification"  # Article 16
    ERASURE = "erasure"  # Article 17 (Right to be forgotten)
    RESTRICTION = "restriction"  # Article 18
    PORTABILITY = "portability"  # Article 20
    OBJECTION = "objection"  # Article 21
    AUTOMATED_DECISION = "automated_decision"  # Article 22


class LawfulBasis(str, Enum):
    """Lawful Bases for Processing under GDPR Article 6."""
    
    CONSENT = "consent"  # Article 6(1)(a)
    CONTRACT = "contract"  # Article 6(1)(b)
    LEGAL_OBLIGATION = "legal_obligation"  # Article 6(1)(c)
    VITAL_INTERESTS = "vital_interests"  # Article 6(1)(d)
    PUBLIC_TASK = "public_task"  # Article 6(1)(e)
    LEGITIMATE_INTERESTS = "legitimate_interests"  # Article 6(1)(f)


class ComplianceStatus(str, Enum):
    """Compliance validation status."""
    
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    PENDING_REVIEW = "pending_review"


@dataclass
class GDPRValidationResult:
    """Result of a GDPR compliance validation check."""
    
    article: GDPRArticle
    status: ComplianceStatus
    requirement: str
    evidence: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "article": self.article.value,
            "status": self.status.value,
            "requirement": self.requirement,
            "evidence": self.evidence,
            "gaps": self.gaps,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DataSubjectRequest:
    """Request for data subject rights exercise."""
    
    request_id: str
    subject_id: str
    right: DataSubjectRight
    request_time: datetime
    status: str = "pending"
    response_deadline: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None


@dataclass
class AutomatedDecisionExplanation:
    """Explanation for automated decision-making (Article 22).
    
    This implements the Right to Explanation requirement under GDPR
    Article 22, ensuring data subjects can understand significant
    decisions made by automated systems.
    """
    
    decision_id: str
    decision: str
    logic_involved: str
    significance: str
    consequences: str
    data_used: List[str]
    factors: List[Dict[str, Any]]
    human_readable: str
    appeal_mechanism: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "decision_id": self.decision_id,
            "decision": self.decision,
            "logic_involved": self.logic_involved,
            "significance": self.significance,
            "consequences": self.consequences,
            "data_used": self.data_used,
            "factors": self.factors,
            "human_readable": self.human_readable,
            "appeal_mechanism": self.appeal_mechanism,
            "timestamp": self.timestamp.isoformat(),
        }


class GDPRComplianceValidator:
    """GDPR Compliance Validator for Nethical governance system.
    
    Validates compliance with GDPR requirements including:
    - Data Subject Rights (Articles 12-22)
    - Lawful Basis for Processing (Article 6)
    - Data Protection by Design (Article 25)
    - Right to Explanation (Article 22)
    
    Attributes:
        response_deadline_days: Days to respond to DSR requests (default: 30)
    """
    
    # GDPR response deadline for Data Subject Requests
    DEFAULT_RESPONSE_DAYS = 30
    
    def __init__(
        self,
        response_deadline_days: int = DEFAULT_RESPONSE_DAYS,
    ) -> None:
        """Initialize GDPR Compliance Validator.
        
        Args:
            response_deadline_days: Days to respond to DSR requests
        """
        self.response_deadline_days = response_deadline_days
        self.validation_results: List[GDPRValidationResult] = []
        self.dsr_requests: Dict[str, DataSubjectRequest] = {}
        
        logger.info(
            "GDPRComplianceValidator initialized with %d day response deadline",
            response_deadline_days,
        )
    
    def validate_article_5(
        self,
        processing_activities: Dict[str, Any],
    ) -> GDPRValidationResult:
        """Validate Article 5: Principles of Processing.
        
        Checks:
        - Lawfulness, fairness, transparency
        - Purpose limitation
        - Data minimization
        - Accuracy
        - Storage limitation
        - Integrity and confidentiality
        - Accountability
        
        Args:
            processing_activities: Dictionary describing processing activities
            
        Returns:
            GDPRValidationResult with compliance status
        """
        evidence = []
        gaps = []
        recommendations = []
        
        # Check lawfulness
        if processing_activities.get("lawful_basis"):
            evidence.append("Lawful basis documented")
        else:
            gaps.append("Lawful basis not documented")
            recommendations.append("Document lawful basis for all processing activities")
        
        # Check purpose limitation
        if processing_activities.get("purpose_documented"):
            evidence.append("Processing purposes documented")
        else:
            gaps.append("Processing purposes not clearly defined")
            recommendations.append("Document specific, explicit purposes for data processing")
        
        # Check data minimization
        if processing_activities.get("data_minimization"):
            evidence.append("Data minimization controls implemented")
        else:
            gaps.append("Data minimization not verified")
            recommendations.append("Implement data minimization controls")
        
        # Check accuracy
        if processing_activities.get("accuracy_controls"):
            evidence.append("Data accuracy controls in place")
        else:
            gaps.append("No accuracy verification controls")
            recommendations.append("Implement data accuracy verification procedures")
        
        # Check storage limitation
        if processing_activities.get("retention_policy"):
            evidence.append("Data retention policy defined")
        else:
            gaps.append("No data retention policy")
            recommendations.append("Define and implement data retention policies")
        
        # Check security
        if processing_activities.get("security_controls"):
            evidence.append("Security controls implemented")
        else:
            gaps.append("Security controls not verified")
            recommendations.append("Implement appropriate security measures")
        
        # Determine status
        if len(gaps) == 0:
            status = ComplianceStatus.COMPLIANT
        elif len(gaps) <= 2:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        result = GDPRValidationResult(
            article=GDPRArticle.ARTICLE_5,
            status=status,
            requirement="Principles relating to processing of personal data",
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
        )
        
        self.validation_results.append(result)
        return result
    
    def validate_article_6(
        self,
        lawful_basis: LawfulBasis,
        supporting_evidence: Dict[str, Any],
    ) -> GDPRValidationResult:
        """Validate Article 6: Lawful Basis for Processing.
        
        Args:
            lawful_basis: The claimed lawful basis
            supporting_evidence: Evidence supporting the lawful basis
            
        Returns:
            GDPRValidationResult with compliance status
        """
        evidence = []
        gaps = []
        recommendations = []
        
        # Validate based on the claimed lawful basis
        if lawful_basis == LawfulBasis.CONSENT:
            if supporting_evidence.get("consent_record"):
                evidence.append("Consent record exists")
            else:
                gaps.append("No consent record found")
                recommendations.append("Implement consent management system")
            
            if supporting_evidence.get("consent_freely_given"):
                evidence.append("Consent freely given (not bundled)")
            else:
                gaps.append("Cannot verify consent was freely given")
                recommendations.append("Review consent collection mechanism")
            
            if supporting_evidence.get("withdrawal_mechanism"):
                evidence.append("Withdrawal mechanism available")
            else:
                gaps.append("No consent withdrawal mechanism")
                recommendations.append("Implement easy consent withdrawal")
        
        elif lawful_basis == LawfulBasis.CONTRACT:
            if supporting_evidence.get("contract_exists"):
                evidence.append("Contract documented")
            else:
                gaps.append("No contract documentation")
                recommendations.append("Document contractual basis")
            
            if supporting_evidence.get("processing_necessary"):
                evidence.append("Processing necessary for contract")
            else:
                gaps.append("Processing may not be necessary for contract")
                recommendations.append("Review processing necessity")
        
        elif lawful_basis == LawfulBasis.LEGITIMATE_INTERESTS:
            if supporting_evidence.get("lia_conducted"):
                evidence.append("Legitimate Interests Assessment conducted")
            else:
                gaps.append("No LIA conducted")
                recommendations.append("Conduct Legitimate Interests Assessment")
            
            if supporting_evidence.get("balancing_test"):
                evidence.append("Balancing test documented")
            else:
                gaps.append("Balancing test not documented")
                recommendations.append("Document balancing test results")
        
        elif lawful_basis == LawfulBasis.LEGAL_OBLIGATION:
            if supporting_evidence.get("legal_reference"):
                evidence.append("Legal obligation referenced")
            else:
                gaps.append("Legal obligation not documented")
                recommendations.append("Document specific legal obligation")
        
        elif lawful_basis == LawfulBasis.VITAL_INTERESTS:
            if supporting_evidence.get("vital_interest_documented"):
                evidence.append("Vital interest documented")
            else:
                gaps.append("Vital interest not documented")
                recommendations.append("Document vital interest justification")
        
        elif lawful_basis == LawfulBasis.PUBLIC_TASK:
            if supporting_evidence.get("public_task_authority"):
                evidence.append("Public task authority documented")
            else:
                gaps.append("Public task authority not documented")
                recommendations.append("Document public task legal basis")
        
        # Determine status
        if len(gaps) == 0:
            status = ComplianceStatus.COMPLIANT
        elif len(gaps) == 1:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        result = GDPRValidationResult(
            article=GDPRArticle.ARTICLE_6,
            status=status,
            requirement=f"Lawful basis: {lawful_basis.value}",
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
        )
        
        self.validation_results.append(result)
        return result
    
    def validate_article_22(
        self,
        automated_decision_config: Dict[str, Any],
    ) -> GDPRValidationResult:
        """Validate Article 22: Automated Individual Decision-Making.
        
        Validates that automated decision-making systems provide:
        - Right to human intervention
        - Right to express point of view
        - Right to contest the decision
        - Meaningful information about logic involved
        - Significance and consequences explanation
        
        Args:
            automated_decision_config: Configuration for automated decisions
            
        Returns:
            GDPRValidationResult with compliance status
        """
        evidence = []
        gaps = []
        recommendations = []
        
        # Check for human intervention capability
        if automated_decision_config.get("human_intervention_available"):
            evidence.append("Human intervention mechanism available")
        else:
            gaps.append("No human intervention mechanism")
            recommendations.append("Implement human review/override capability")
        
        # Check for explanation capability
        if automated_decision_config.get("explanation_capability"):
            evidence.append("Decision explanation capability implemented")
        else:
            gaps.append("No explanation capability for automated decisions")
            recommendations.append("Implement Right to Explanation")
        
        # Check for logic transparency
        if automated_decision_config.get("logic_documented"):
            evidence.append("Decision logic documented")
        else:
            gaps.append("Decision logic not documented")
            recommendations.append("Document decision-making logic")
        
        # Check for significance explanation
        if automated_decision_config.get("significance_explained"):
            evidence.append("Significance and consequences explained")
        else:
            gaps.append("Significance/consequences not explained")
            recommendations.append("Document decision significance and consequences")
        
        # Check for appeal/contest mechanism
        if automated_decision_config.get("appeal_mechanism"):
            evidence.append("Appeal/contest mechanism available")
        else:
            gaps.append("No appeal mechanism")
            recommendations.append("Implement decision appeal process")
        
        # Check for safeguards
        if automated_decision_config.get("safeguards_implemented"):
            evidence.append("Safeguards implemented")
        else:
            gaps.append("Safeguards not verified")
            recommendations.append("Implement appropriate safeguards")
        
        # Determine status
        if len(gaps) == 0:
            status = ComplianceStatus.COMPLIANT
        elif len(gaps) <= 2:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        result = GDPRValidationResult(
            article=GDPRArticle.ARTICLE_22,
            status=status,
            requirement="Automated individual decision-making, including profiling",
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
        )
        
        self.validation_results.append(result)
        return result
    
    def validate_article_25(
        self,
        design_controls: Dict[str, Any],
    ) -> GDPRValidationResult:
        """Validate Article 25: Data Protection by Design and Default.
        
        Args:
            design_controls: Dictionary describing privacy by design controls
            
        Returns:
            GDPRValidationResult with compliance status
        """
        evidence = []
        gaps = []
        recommendations = []
        
        # Check pseudonymization
        if design_controls.get("pseudonymization"):
            evidence.append("Pseudonymization implemented")
        else:
            gaps.append("No pseudonymization")
            recommendations.append("Implement pseudonymization where appropriate")
        
        # Check data minimization by design
        if design_controls.get("data_minimization_by_design"):
            evidence.append("Data minimization built into design")
        else:
            gaps.append("Data minimization not in design")
            recommendations.append("Redesign to minimize data collection by default")
        
        # Check privacy by default
        if design_controls.get("privacy_by_default"):
            evidence.append("Privacy-protective defaults configured")
        else:
            gaps.append("Privacy not default setting")
            recommendations.append("Set privacy-protective defaults")
        
        # Check access controls
        if design_controls.get("access_controls"):
            evidence.append("Access controls implemented")
        else:
            gaps.append("Access controls not verified")
            recommendations.append("Implement role-based access controls")
        
        # Check encryption
        if design_controls.get("encryption"):
            evidence.append("Encryption implemented")
        else:
            gaps.append("Encryption not verified")
            recommendations.append("Implement encryption at rest and in transit")
        
        # Determine status
        if len(gaps) == 0:
            status = ComplianceStatus.COMPLIANT
        elif len(gaps) <= 2:
            status = ComplianceStatus.PARTIAL
        else:
            status = ComplianceStatus.NON_COMPLIANT
        
        result = GDPRValidationResult(
            article=GDPRArticle.ARTICLE_25,
            status=status,
            requirement="Data protection by design and by default",
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
        )
        
        self.validation_results.append(result)
        return result
    
    def generate_automated_decision_explanation(
        self,
        decision_id: str,
        decision: str,
        judgment_data: Dict[str, Any],
    ) -> AutomatedDecisionExplanation:
        """Generate Article 22 compliant explanation for automated decision.
        
        This implements the Right to Explanation requirement, providing
        data subjects with meaningful information about automated decisions.
        
        Args:
            decision_id: Unique identifier for the decision
            decision: The decision made (e.g., ALLOW, BLOCK)
            judgment_data: Data from the judgment process
            
        Returns:
            AutomatedDecisionExplanation with all required information
        """
        # Extract factors from judgment data
        factors = []
        
        if "risk_score" in judgment_data:
            factors.append({
                "name": "Risk Score",
                "value": judgment_data["risk_score"],
                "weight": 0.3,
                "explanation": "Overall risk assessment based on action analysis",
            })
        
        if "violations" in judgment_data:
            violations = judgment_data["violations"]
            factors.append({
                "name": "Violations Detected",
                "value": len(violations),
                "weight": 0.4,
                "explanation": "Number of policy violations detected",
                "details": violations[:5] if isinstance(violations, list) else [],
            })
        
        if "matched_rules" in judgment_data:
            rules = judgment_data["matched_rules"]
            factors.append({
                "name": "Policy Rules",
                "value": len(rules) if isinstance(rules, list) else 0,
                "weight": 0.2,
                "explanation": "Policy rules that matched the action",
            })
        
        # Determine data used
        data_used = []
        if judgment_data.get("action"):
            data_used.append("Action content")
        if judgment_data.get("agent_id"):
            data_used.append("Agent identifier")
        if judgment_data.get("context"):
            data_used.append("Action context")
        if judgment_data.get("action_type"):
            data_used.append("Action type classification")
        
        # Generate human-readable explanation
        human_readable = self._generate_human_readable_explanation(
            decision=decision,
            factors=factors,
            judgment_data=judgment_data,
        )
        
        # Determine significance based on decision
        significance_map = {
            "ALLOW": "The action was permitted to proceed",
            "RESTRICT": "The action was limited or modified",
            "BLOCK": "The action was prevented from executing",
            "TERMINATE": "The session or agent was terminated",
        }
        significance = significance_map.get(
            decision,
            "The action was evaluated by the governance system",
        )
        
        # Determine consequences
        consequences_map = {
            "ALLOW": "No restrictions applied. Action proceeds normally.",
            "RESTRICT": "Action proceeds with limitations or additional oversight.",
            "BLOCK": "Action is prevented. May require review or modification.",
            "TERMINATE": "Agent or session is terminated. Manual intervention required.",
        }
        consequences = consequences_map.get(
            decision,
            "Standard governance evaluation applied.",
        )
        
        # Logic involved
        logic_involved = (
            "The decision was made using Nethical's governance system which evaluates "
            "actions against the 25 Fundamental Laws, policy rules, and risk assessment. "
            "The system uses rule-based evaluation combined with risk scoring to determine "
            "the appropriate action."
        )
        
        return AutomatedDecisionExplanation(
            decision_id=decision_id,
            decision=decision,
            logic_involved=logic_involved,
            significance=significance,
            consequences=consequences,
            data_used=data_used,
            factors=factors,
            human_readable=human_readable,
            appeal_mechanism=(
                "To contest this decision, submit an appeal via the /v2/appeals endpoint "
                "or contact the system administrator. Human review is available for all "
                "automated decisions."
            ),
        )
    
    def _generate_human_readable_explanation(
        self,
        decision: str,
        factors: List[Dict[str, Any]],
        judgment_data: Dict[str, Any],
    ) -> str:
        """Generate human-readable explanation text.
        
        Args:
            decision: The decision made
            factors: Contributing factors
            judgment_data: Full judgment data
            
        Returns:
            Human-readable explanation string
        """
        # Start with decision summary
        parts = [f"The governance system evaluated your action and made a '{decision}' decision."]
        
        # Add factor explanations
        if factors:
            parts.append("\nKey factors in this decision:")
            for factor in factors:
                parts.append(f"- {factor['name']}: {factor['explanation']}")
        
        # Add violation details if present
        violations = judgment_data.get("violations", [])
        if violations and isinstance(violations, list):
            parts.append("\nViolations detected:")
            for v in violations[:3]:
                if isinstance(v, dict):
                    parts.append(f"- {v.get('type', 'Unknown')}: {v.get('description', 'N/A')}")
                else:
                    parts.append(f"- {v}")
        
        # Add risk score context
        risk_score = judgment_data.get("risk_score", 0)
        if risk_score > 0:
            risk_level = (
                "low" if risk_score < 0.3
                else "moderate" if risk_score < 0.6
                else "high" if risk_score < 0.8
                else "critical"
            )
            parts.append(f"\nRisk assessment: {risk_level} ({risk_score:.2f})")
        
        # Add human oversight note
        parts.append(
            "\nYou have the right to request human review of this decision "
            "and to express your point of view."
        )
        
        return "\n".join(parts)
    
    def register_dsr(
        self,
        subject_id: str,
        right: DataSubjectRight,
    ) -> DataSubjectRequest:
        """Register a Data Subject Request.
        
        Args:
            subject_id: Identifier for the data subject
            right: The right being exercised
            
        Returns:
            DataSubjectRequest object with deadline
        """
        from datetime import timedelta
        
        request_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        request = DataSubjectRequest(
            request_id=request_id,
            subject_id=subject_id,
            right=right,
            request_time=now,
            status="pending",
            response_deadline=now + timedelta(days=self.response_deadline_days),
        )
        
        self.dsr_requests[request_id] = request
        
        logger.info(
            "DSR registered: %s for subject %s (right: %s)",
            request_id,
            subject_id,
            right.value,
        )
        
        return request
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get summary of compliance validation results.
        
        Returns:
            Dictionary with compliance summary
        """
        if not self.validation_results:
            return {
                "total_checks": 0,
                "status": "No validations performed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        
        # Count by status
        status_counts = {
            "compliant": 0,
            "partial": 0,
            "non_compliant": 0,
            "pending_review": 0,
        }
        
        for result in self.validation_results:
            status_counts[result.status.value] = (
                status_counts.get(result.status.value, 0) + 1
            )
        
        total = len(self.validation_results)
        compliant = status_counts["compliant"]
        partial = status_counts["partial"]
        
        # Calculate compliance score
        score = ((compliant + (partial * 0.5)) / total * 100) if total > 0 else 0
        
        # Overall status
        if status_counts["non_compliant"] > 0:
            overall_status = "Non-Compliant"
        elif status_counts["partial"] > 0:
            overall_status = "Partially Compliant"
        else:
            overall_status = "Compliant"
        
        return {
            "total_checks": total,
            "status_breakdown": status_counts,
            "compliance_score": round(score, 2),
            "overall_status": overall_status,
            "dsr_pending": sum(
                1 for r in self.dsr_requests.values() if r.status == "pending"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


__all__ = [
    "GDPRComplianceValidator",
    "GDPRArticle",
    "DataSubjectRight",
    "LawfulBasis",
    "ComplianceStatus",
    "GDPRValidationResult",
    "DataSubjectRequest",
    "AutomatedDecisionExplanation",
]
