# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Impact Assessment Pipeline (DPIA / AIIA / FRIA).

Implements structured impact assessment workflows mandated by:

- **GDPR Art. 35:** Data Protection Impact Assessment (DPIA) for high-risk processing
- **EU AI Act Art. 9, 27:** AI Impact Assessment (AIIA) and Fundamental Rights Impact
  Assessment (FRIA) for high-risk AI systems
- **ISO/IEC 42001 §6.1.2:** AI risk assessment and management
- **NIST AI RMF:** Map, Measure, Manage, Govern functions

Pipeline stages:
1. System Classification (risk tier determination)
2. Stakeholder Identification
3. Rights & Freedoms Analysis
4. Risk Scoring (likelihood × severity matrix)
5. Mitigation Planning
6. Authority Consultation (GDPR Art. 36)
7. Ongoing Monitoring Commitment

Gap Addressed: 2.4 (Impact Assessment Pipeline)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.governance.impact_assessment")


# ============== Enums ==============


class AssessmentType(str, Enum):
    """Type of impact assessment."""
    DPIA = "dpia"                   # Data Protection Impact Assessment (GDPR Art. 35)
    AIIA = "aiia"                   # AI Impact Assessment (EU AI Act)
    FRIA = "fria"                   # Fundamental Rights Impact Assessment (EU AI Act Art. 27)
    COMBINED = "combined"           # Combined DPIA + AIIA + FRIA
    EQUALITY = "equality"           # Equality Impact Assessment
    HUMAN_RIGHTS = "human_rights"   # Human Rights Impact Assessment
    CUSTOM = "custom"


class RiskTier(str, Enum):
    """EU AI Act risk classification tiers."""
    UNACCEPTABLE = "unacceptable"   # Prohibited (Art. 5)
    HIGH = "high"                   # Requires conformity assessment (Annex III)
    LIMITED = "limited"             # Transparency obligations only (Art. 52)
    MINIMAL = "minimal"             # No specific obligations


class AssessmentStatus(str, Enum):
    """Status of an impact assessment."""
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    UNDER_REVIEW = "under_review"
    AWAITING_AUTHORITY_CONSULTATION = "awaiting_authority_consultation"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"
    ARCHIVED = "archived"


class RiskLevel(str, Enum):
    """Risk level (likelihood × severity)."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FundamentalRight(str, Enum):
    """EU Charter of Fundamental Rights categories."""
    DIGNITY = "dignity"                     # Title I (Arts. 1-5)
    FREEDOMS = "freedoms"                   # Title II (Arts. 6-19)
    EQUALITY = "equality"                   # Title III (Arts. 20-26)
    SOLIDARITY = "solidarity"               # Title IV (Arts. 27-38)
    CITIZENS_RIGHTS = "citizens_rights"     # Title V (Arts. 39-46)
    JUSTICE = "justice"                     # Title VI (Arts. 47-50)
    PRIVACY = "privacy"                     # Art. 7 + Art. 8 (data protection)
    NON_DISCRIMINATION = "non_discrimination"  # Art. 21
    CHILD_RIGHTS = "child_rights"           # Art. 24


# ============== Assessment Data Models ==============


class StakeholderGroup(BaseModel):
    """An identified stakeholder group affected by the AI system."""
    group_id: str = Field(default_factory=lambda: f"sg_{uuid.uuid4().hex[:8]}")
    name: str = Field(..., description="Stakeholder group name (e.g., 'Citizens applying for benefits')")
    category: str = Field(..., description="Category: 'data_subjects', 'operators', 'affected_persons', 'society'")
    estimated_size: Optional[int] = Field(default=None, description="Estimated number of individuals affected")
    vulnerability_level: str = Field(
        default="standard",
        description="'standard', 'vulnerable', 'highly_vulnerable' (children, elderly, minorities)"
    )
    rights_at_risk: List[FundamentalRight] = Field(default_factory=list)
    description: str = Field(default="")


class IdentifiedRisk(BaseModel):
    """A specific risk identified during the assessment."""
    risk_id: str = Field(default_factory=lambda: f"risk_{uuid.uuid4().hex[:8]}")
    title: str = Field(..., description="Short risk title")
    description: str = Field(..., description="Detailed risk description")
    risk_category: str = Field(
        ..., description="Category: 'bias', 'privacy', 'safety', 'autonomy', 'transparency', 'accountability'"
    )
    affected_rights: List[FundamentalRight] = Field(default_factory=list)
    affected_stakeholders: List[str] = Field(
        default_factory=list, description="Stakeholder group IDs"
    )
    likelihood: int = Field(..., ge=1, le=5, description="1=Very Unlikely, 5=Almost Certain")
    severity: int = Field(..., ge=1, le=5, description="1=Negligible, 5=Catastrophic")
    risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    existing_controls: List[str] = Field(default_factory=list)
    residual_risk_acceptable: bool = Field(default=False)

    def calculate_risk_score(self) -> int:
        """Calculate risk score as likelihood × severity."""
        return self.likelihood * self.severity

    def determine_risk_level(self) -> RiskLevel:
        """Determine risk level from score."""
        score = self.calculate_risk_score()
        if score <= 4:
            return RiskLevel.NEGLIGIBLE
        elif score <= 8:
            return RiskLevel.LOW
        elif score <= 14:
            return RiskLevel.MEDIUM
        elif score <= 20:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL


class MitigationMeasure(BaseModel):
    """A proposed mitigation measure for an identified risk."""
    measure_id: str = Field(default_factory=lambda: f"mit_{uuid.uuid4().hex[:8]}")
    title: str = Field(..., description="Mitigation measure title")
    description: str = Field(..., description="Detailed description of the measure")
    target_risk_ids: List[str] = Field(..., description="Risk IDs this measure addresses")
    measure_type: str = Field(
        ..., description="'technical', 'organizational', 'legal', 'governance', 'training'"
    )
    implementation_status: str = Field(
        default="planned",
        description="'planned', 'in_progress', 'implemented', 'verified'"
    )
    responsible_role: Optional[str] = Field(default=None, description="Role responsible for implementation")
    deadline: Optional[str] = Field(default=None, description="Implementation deadline (ISO 8601)")
    estimated_risk_reduction_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Estimated percentage reduction in risk"
    )
    nethical_feature_mapping: Optional[str] = Field(
        default=None,
        description="Nethical module/feature that implements this measure (e.g., 'detectors.bias', 'core.kill_switch')"
    )


class AuthorityConsultation(BaseModel):
    """Record of supervisory authority consultation (GDPR Art. 36)."""
    consultation_id: str = Field(default_factory=lambda: f"cons_{uuid.uuid4().hex[:8]}")
    authority_name: str = Field(..., description="Supervisory authority name")
    authority_jurisdiction: str = Field(..., description="Jurisdiction (e.g., 'PL-UODO', 'IE-DPC')")
    submitted_at: Optional[str] = Field(default=None)
    response_received_at: Optional[str] = Field(default=None)
    authority_decision: Optional[str] = Field(default=None, description="'approved', 'conditions', 'rejected'")
    conditions_imposed: List[str] = Field(default_factory=list)
    reference_number: Optional[str] = Field(default=None)


# ============== Impact Assessment ==============


class ImpactAssessment(BaseModel):
    """A complete impact assessment document."""
    assessment_id: str = Field(default_factory=lambda: f"ia_{uuid.uuid4().hex[:12]}")
    assessment_type: AssessmentType = Field(default=AssessmentType.COMBINED)
    status: AssessmentStatus = Field(default=AssessmentStatus.DRAFT)
    version: str = Field(default="1.0")

    # System identification
    system_name: str = Field(..., description="Name of the AI system being assessed")
    system_description: str = Field(default="", description="Description of purpose and functionality")
    system_version: Optional[str] = Field(default=None)
    deployment_context: str = Field(
        default="", description="Where and how the system will be deployed"
    )
    risk_tier: RiskTier = Field(default=RiskTier.LIMITED)

    # Organizational context
    controller_organization: str = Field(..., description="Data controller / deployer organization")
    dpo_contact: Optional[str] = Field(default=None, description="DPO contact information")
    assessor_id: Optional[str] = Field(default=None, description="User who conducted the assessment")
    reviewer_ids: List[str] = Field(default_factory=list, description="Users who reviewed")
    tenant_id: Optional[str] = Field(default=None)

    # Assessment content
    stakeholders: List[StakeholderGroup] = Field(default_factory=list)
    data_categories_processed: List[str] = Field(
        default_factory=list,
        description="Types of personal data (e.g., 'biometric', 'health', 'financial', 'behavioral')"
    )
    legal_basis: List[str] = Field(
        default_factory=list,
        description="Legal bases for processing (GDPR Art. 6/9)"
    )
    identified_risks: List[IdentifiedRisk] = Field(default_factory=list)
    mitigation_measures: List[MitigationMeasure] = Field(default_factory=list)
    authority_consultations: List[AuthorityConsultation] = Field(default_factory=list)

    # Scores and outcomes
    overall_risk_level: RiskLevel = Field(default=RiskLevel.MEDIUM)
    overall_risk_score: float = Field(default=0.0, ge=0.0, le=25.0)
    rights_impact_score: float = Field(default=0.0, ge=0.0, le=10.0)
    mitigation_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    requires_authority_consultation: bool = Field(default=False)
    assessment_conclusion: str = Field(default="")

    # Timestamps
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    approved_at: Optional[str] = Field(default=None)
    next_review_date: Optional[str] = Field(default=None)

    # Monitoring commitment
    ongoing_monitoring_plan: str = Field(
        default="",
        description="Description of how the system will be monitored post-deployment"
    )
    review_frequency_months: int = Field(
        default=12, ge=1,
        description="How often the assessment should be reviewed"
    )


# ============== Impact Assessment Engine ==============


class ImpactAssessmentEngine:
    """Engine for managing and evaluating impact assessments.

    Provides automated risk scoring, gap detection, EU AI Act risk tier
    classification, and authority consultation triggers.
    """

    def __init__(self) -> None:
        self._assessments: Dict[str, ImpactAssessment] = {}

    # ---------- Assessment Lifecycle ----------

    def create_assessment(
        self,
        system_name: str,
        controller_organization: str,
        assessment_type: AssessmentType = AssessmentType.COMBINED,
        assessor_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        system_description: str = "",
        deployment_context: str = "",
    ) -> ImpactAssessment:
        """Create a new impact assessment."""
        assessment = ImpactAssessment(
            system_name=system_name,
            controller_organization=controller_organization,
            assessment_type=assessment_type,
            assessor_id=assessor_id,
            tenant_id=tenant_id,
            system_description=system_description,
            deployment_context=deployment_context,
        )
        self._assessments[assessment.assessment_id] = assessment
        logger.info(
            f"Created {assessment_type.value} assessment '{assessment.assessment_id}' "
            f"for system '{system_name}'"
        )
        return assessment

    def get_assessment(self, assessment_id: str) -> Optional[ImpactAssessment]:
        """Retrieve an assessment by ID."""
        return self._assessments.get(assessment_id)

    # ---------- Risk Classification ----------

    def classify_risk_tier(self, assessment_id: str) -> RiskTier:
        """Automatically classify the EU AI Act risk tier based on system characteristics."""
        assessment = self._get_assessment(assessment_id)

        # Check for unacceptable risk indicators (EU AI Act Art. 5)
        unacceptable_indicators = {
            "social_scoring", "real_time_biometric_public", "subliminal_manipulation",
            "vulnerability_exploitation", "predictive_policing_individual",
        }
        system_features = set(assessment.data_categories_processed)

        if system_features & unacceptable_indicators:
            assessment.risk_tier = RiskTier.UNACCEPTABLE
            return RiskTier.UNACCEPTABLE

        # Check for high-risk indicators (Annex III)
        high_risk_domains = {
            "biometric", "critical_infrastructure", "education_access",
            "employment", "essential_services", "law_enforcement",
            "migration_asylum", "justice_democratic", "health_diagnosis",
        }
        high_risk_data = {"biometric", "health", "genetic", "criminal_records", "children_data"}

        if system_features & high_risk_domains or system_features & high_risk_data:
            assessment.risk_tier = RiskTier.HIGH
            return RiskTier.HIGH

        # Check for vulnerability of stakeholders
        has_vulnerable = any(
            s.vulnerability_level in ("vulnerable", "highly_vulnerable")
            for s in assessment.stakeholders
        )
        if has_vulnerable:
            assessment.risk_tier = RiskTier.HIGH
            return RiskTier.HIGH

        # Limited risk (transparency obligations)
        limited_indicators = {"chatbot", "emotion_recognition", "deepfake", "ai_generated_content"}
        if system_features & limited_indicators:
            assessment.risk_tier = RiskTier.LIMITED
            return RiskTier.LIMITED

        assessment.risk_tier = RiskTier.MINIMAL
        return RiskTier.MINIMAL

    # ---------- Risk Analysis ----------

    def calculate_risk_scores(self, assessment_id: str) -> Dict[str, Any]:
        """Calculate aggregate risk scores for the assessment."""
        assessment = self._get_assessment(assessment_id)

        if not assessment.identified_risks:
            return {"overall_score": 0.0, "risk_level": RiskLevel.NEGLIGIBLE.value, "risks_by_level": {}}

        total_score = 0.0
        risk_levels = {level: 0 for level in RiskLevel}
        rights_impact: Dict[str, int] = {}

        for risk in assessment.identified_risks:
            score = risk.calculate_risk_score()
            level = risk.determine_risk_level()
            risk.risk_level = level
            total_score += score
            risk_levels[level] += 1

            for right in risk.affected_rights:
                rights_impact[right.value] = rights_impact.get(right.value, 0) + 1

        avg_score = total_score / len(assessment.identified_risks)
        max_score = max(r.calculate_risk_score() for r in assessment.identified_risks)

        # Determine overall risk level
        if risk_levels.get(RiskLevel.CRITICAL, 0) > 0 or max_score >= 20:
            overall = RiskLevel.CRITICAL
        elif risk_levels.get(RiskLevel.HIGH, 0) > 0 or avg_score > 14:
            overall = RiskLevel.HIGH
        elif avg_score > 8:
            overall = RiskLevel.MEDIUM
        elif avg_score > 4:
            overall = RiskLevel.LOW
        else:
            overall = RiskLevel.NEGLIGIBLE

        assessment.overall_risk_level = overall
        assessment.overall_risk_score = round(avg_score, 2)
        assessment.rights_impact_score = round(len(rights_impact) / max(len(FundamentalRight), 1) * 10, 2)

        # Determine if authority consultation is required (GDPR Art. 36)
        assessment.requires_authority_consultation = (
            overall in (RiskLevel.HIGH, RiskLevel.CRITICAL)
            and any(not r.residual_risk_acceptable for r in assessment.identified_risks if r.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL))
        )

        # Calculate mitigation coverage
        mitigated_risks = set()
        for measure in assessment.mitigation_measures:
            mitigated_risks.update(measure.target_risk_ids)
        total_risks = {r.risk_id for r in assessment.identified_risks}
        if total_risks:
            assessment.mitigation_coverage_pct = round(
                len(mitigated_risks & total_risks) / len(total_risks) * 100, 1
            )

        assessment.last_updated = datetime.now(timezone.utc).isoformat()

        return {
            "overall_score": avg_score,
            "max_risk_score": max_score,
            "risk_level": overall.value,
            "total_risks": len(assessment.identified_risks),
            "risks_by_level": {k.value: v for k, v in risk_levels.items() if v > 0},
            "rights_impacted": rights_impact,
            "mitigation_coverage_pct": assessment.mitigation_coverage_pct,
            "requires_authority_consultation": assessment.requires_authority_consultation,
        }

    # ---------- Gap Detection ----------

    def detect_gaps(self, assessment_id: str) -> List[str]:
        """Detect completeness gaps in an impact assessment."""
        assessment = self._get_assessment(assessment_id)
        gaps = []

        if not assessment.system_description:
            gaps.append("Missing system description")
        if not assessment.deployment_context:
            gaps.append("Missing deployment context")
        if not assessment.stakeholders:
            gaps.append("No stakeholder groups identified")
        if not assessment.data_categories_processed:
            gaps.append("No data categories specified")
        if not assessment.legal_basis:
            gaps.append("No legal basis specified (GDPR Art. 6/9 required)")
        if not assessment.identified_risks:
            gaps.append("No risks identified (minimum risk analysis required)")
        if not assessment.dpo_contact:
            gaps.append("No DPO contact information provided")

        # Check mitigation coverage
        if assessment.identified_risks and not assessment.mitigation_measures:
            gaps.append("No mitigation measures defined for identified risks")

        high_risks_unmitigated = []
        mitigated_ids = set()
        for m in assessment.mitigation_measures:
            mitigated_ids.update(m.target_risk_ids)
        for risk in assessment.identified_risks:
            if risk.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and risk.risk_id not in mitigated_ids:
                high_risks_unmitigated.append(risk.title)
        if high_risks_unmitigated:
            gaps.append(f"High/Critical risks without mitigation: {', '.join(high_risks_unmitigated)}")

        # Check if authority consultation is needed but not initiated
        if assessment.requires_authority_consultation and not assessment.authority_consultations:
            gaps.append("Authority consultation required (GDPR Art. 36) but not initiated")

        # Ongoing monitoring
        if not assessment.ongoing_monitoring_plan:
            gaps.append("No ongoing monitoring plan defined")

        # FRIA-specific
        if assessment.assessment_type in (AssessmentType.FRIA, AssessmentType.COMBINED):
            if not any(s.rights_at_risk for s in assessment.stakeholders):
                gaps.append("FRIA: No fundamental rights at risk identified for any stakeholder group")

        return gaps

    # ---------- Nethical Feature Mapping ----------

    def suggest_nethical_mitigations(self, assessment_id: str) -> List[Dict[str, str]]:
        """Suggest Nethical features that can serve as mitigation measures."""
        assessment = self._get_assessment(assessment_id)
        suggestions = []

        risk_categories = {r.risk_category for r in assessment.identified_risks}

        mapping = {
            "bias": {
                "module": "nethical.core.fairness_sampler",
                "description": "Fairness sampling and bias detection across demographic groups",
            },
            "privacy": {
                "module": "nethical.core.differential_privacy + nethical.core.redaction_pipeline",
                "description": "Differential privacy guarantees and PII redaction",
            },
            "safety": {
                "module": "nethical.core.kill_switch + nethical.edge.kinetic_safety",
                "description": "Kill-switch circuit breaker and kinetic safety interlock",
            },
            "transparency": {
                "module": "nethical.explainability.decision_explainer",
                "description": "Natural-language decision explanations for affected persons",
            },
            "accountability": {
                "module": "nethical.security.merkle_ledger + nethical.security.audit_logging",
                "description": "Post-quantum Merkle-DAG audit trail with non-repudiation",
            },
            "autonomy": {
                "module": "nethical.gateway.hitl + nethical.governance.human_review",
                "description": "Human-in-the-loop gateway with SLA-tracked review queue",
            },
            "manipulation": {
                "module": "nethical.detectors.manipulation_detector + nethical.ethics.covert_persuasion_shield",
                "description": "Manipulation detection and covert persuasion shield",
            },
        }

        for category in risk_categories:
            if category in mapping:
                suggestions.append({
                    "risk_category": category,
                    "nethical_module": mapping[category]["module"],
                    "description": mapping[category]["description"],
                })

        return suggestions

    # ---------- Report Generation ----------

    def generate_summary_report(self, assessment_id: str) -> Dict[str, Any]:
        """Generate a summary report suitable for executive review."""
        assessment = self._get_assessment(assessment_id)
        self.calculate_risk_scores(assessment_id)
        gaps = self.detect_gaps(assessment_id)

        return {
            "assessment_id": assessment.assessment_id,
            "assessment_type": assessment.assessment_type.value,
            "system_name": assessment.system_name,
            "controller": assessment.controller_organization,
            "risk_tier": assessment.risk_tier.value,
            "overall_risk_level": assessment.overall_risk_level.value,
            "overall_risk_score": assessment.overall_risk_score,
            "total_identified_risks": len(assessment.identified_risks),
            "total_mitigation_measures": len(assessment.mitigation_measures),
            "mitigation_coverage_pct": assessment.mitigation_coverage_pct,
            "stakeholder_groups": len(assessment.stakeholders),
            "rights_impact_score": assessment.rights_impact_score,
            "requires_authority_consultation": assessment.requires_authority_consultation,
            "completeness_gaps": gaps,
            "status": assessment.status.value,
            "created_at": assessment.created_at,
            "last_updated": assessment.last_updated,
        }

    # ---------- Internal Helpers ----------

    def _get_assessment(self, assessment_id: str) -> ImpactAssessment:
        """Retrieve assessment, raising if not found."""
        assessment = self._assessments.get(assessment_id)
        if not assessment:
            raise ValueError(f"Assessment '{assessment_id}' not found.")
        return assessment
