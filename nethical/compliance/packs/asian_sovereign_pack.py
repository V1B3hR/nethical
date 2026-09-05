"""Asian Sovereign AI Regulatory Pack (nethical.compliance.packs.asian_sovereign_pack).

Implements comprehensive compliance evaluations for:
1. Japan METI AI Guidelines for Business ver 1.0 (Ministry of Economy, Trade and Industry)
   - 5 Fundamental Principles: Human-Centricity, Safety, Fairness/Privacy, Transparency, Security.
   - Risk assessment lifecycle & incident disclosure to METI / IPA (Information-technology Promotion Agency).
2. Singapore IMDA Model AI Governance Framework for Generative AI (IMDA & AI Verify Foundation)
   - 9 Core Dimensions: Accountability, Data, Trusted Development, Incident Reporting,
     Testing & Assurance, Security, Content Provenance (C2PA), Safety Alignment, Human-in-the-Loop.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.compliance.packs.asian_sovereign")


class METIPrinciple(str, Enum):
    """Core principles under METI AI Guidelines for Business ver 1.0."""
    HUMAN_CENTRICITY = "HUMAN_CENTRICITY"
    SAFETY = "SAFETY"
    FAIRNESS_PRIVACY = "FAIRNESS_PRIVACY"
    TRANSPARENCY = "TRANSPARENCY"
    SECURITY = "SECURITY"


class JapanMETIEvaluationResult(BaseModel):
    """Evaluation result for Japan METI AI Guidelines ver 1.0."""
    is_compliant: bool = Field(..., description="Czy system spełnia wytyczne biznesowe METI")
    compliance_score: float = Field(..., ge=0.0, le=1.0, description="Wskaźnik dojrzałości zarządzania AI wg METI")
    principles_evaluated: Dict[str, bool] = Field(default_factory=dict)
    ipa_reporting_ready: bool = Field(default=False, description="Gotowość procedur raportowania incydentów do IPA")
    governance_charter_active: bool = Field(default=False)
    missing_controls: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class SingaporeIMDAEvaluationResult(BaseModel):
    """Evaluation result for Singapore IMDA Model AI Governance Framework for GenAI."""
    is_compliant: bool = Field(..., description="Czy model spełnia 9 wymiarów IMDA Model Framework")
    readiness_score: float = Field(..., ge=0.0, le=1.0)
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    c2pa_provenance_verified: bool = Field(default=False, description="Weryfikacja metadanych proweniencji C2PA / znaku wodnego")
    ai_verify_testing_conducted: bool = Field(default=False, description="Przeprowadzenie testów w standardzie AI Verify")
    human_in_the_loop_adequate: bool = Field(default=False)
    missing_dimensions: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AsianComplianceReport(BaseModel):
    """Composite Asian Sovereign AI Compliance Report."""
    is_fully_compliant: bool
    japan_meti: JapanMETIEvaluationResult
    singapore_imda: SingaporeIMDAEvaluationResult
    overall_asian_trust_score: float = Field(..., ge=0.0, le=1.0)
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JapanMETIEvaluator:
    """Evaluates adherence to Japan METI AI Guidelines for Business ver 1.0."""

    REQUIRED_CONTROLS = [
        ("has_ai_governance_charter", "Brak oficjalnej polityki i karty etyki AI zatwierdzonej przez kierownictwo (Principle: Human-Centricity)."),
        ("has_safety_lifecycle_assessment", "Brak procedury ciągłej oceny ryzyka bezpieczeństwa w cyklu życia modelu (Principle: Safety)."),
        ("has_privacy_fairness_review", "Brak formalnego przeglądu ochrony prywatności i sprawiedliwości algorytmicznej (Principle: Fairness/Privacy)."),
        ("has_system_transparency_log", "Brak transparentnych logów wyjaśnialności decyzji (Principle: Transparency)."),
        ("has_security_vulnerability_management", "Brak zarządzania podatnościami modeli i infrastruktury AI (Principle: Security)."),
        ("has_ipa_incident_notification_sla", "Brak procedury eskalacji incydentów AI do IPA w terminie 30 dni (METI Incident Disclosure)."),
    ]

    def evaluate(self, metadata: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> JapanMETIEvaluationResult:
        missing: List[str] = []
        recommendations: List[str] = []
        principles_status = {
            METIPrinciple.HUMAN_CENTRICITY.value: bool(metadata.get("has_ai_governance_charter")),
            METIPrinciple.SAFETY.value: bool(metadata.get("has_safety_lifecycle_assessment")),
            METIPrinciple.FAIRNESS_PRIVACY.value: bool(metadata.get("has_privacy_fairness_review")),
            METIPrinciple.TRANSPARENCY.value: bool(metadata.get("has_system_transparency_log")),
            METIPrinciple.SECURITY.value: bool(metadata.get("has_security_vulnerability_management")),
        }

        for field, err in self.REQUIRED_CONTROLS:
            if not metadata.get(field):
                missing.append(err)

        ipa_ready = bool(metadata.get("has_ipa_incident_notification_sla"))
        charter_active = bool(metadata.get("has_ai_governance_charter"))

        total_reqs = len(self.REQUIRED_CONTROLS)
        passed_reqs = total_reqs - len(missing)
        score = round(passed_reqs / total_reqs, 3)

        if score < 0.85:
            recommendations.append("Dostosuj polityki korporacyjne do wytycznych METI AI ver 1.0 (wymagany próg 85%).")
        if not ipa_ready:
            recommendations.append("Ustanów kanał zgłaszania incydentów do IPA (Information-technology Promotion Agency Japan).")

        return JapanMETIEvaluationResult(
            is_compliant=(len(missing) == 0),
            compliance_score=score,
            principles_evaluated=principles_status,
            ipa_reporting_ready=ipa_ready,
            governance_charter_active=charter_active,
            missing_controls=missing,
            recommendations=recommendations,
        )


class SingaporeIMDAEvaluator:
    """Evaluates adherence to Singapore IMDA Model AI Governance Framework for GenAI."""

    DIMENSIONS = [
        "accountability",
        "data_governance",
        "trusted_development",
        "incident_reporting",
        "testing_and_assurance",
        "security_safeguards",
        "content_provenance",
        "safety_alignment",
        "human_in_the_loop",
    ]

    def evaluate(self, metadata: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> SingaporeIMDAEvaluationResult:
        missing: List[str] = []
        recommendations: List[str] = []
        dim_scores: Dict[str, float] = {}

        # 1. Accountability
        acc = 1.0 if metadata.get("has_accountability_chain") else 0.0
        dim_scores["accountability"] = acc
        if acc == 0.0:
            missing.append("Wymiar 1 (Accountability): Brak jasnego podziału odpowiedzialności w łańcuchu GenAI.")

        # 2. Data Governance
        data_sc = 1.0 if metadata.get("has_training_data_provenance") else 0.0
        dim_scores["data_governance"] = data_sc
        if data_sc == 0.0:
            missing.append("Wymiar 2 (Data): Brak udokumentowanej proweniencji i legalności danych treningowych.")

        # 3. Trusted Development & Deployment
        dev_sc = 1.0 if metadata.get("has_model_eval_and_sbom") else 0.0
        dim_scores["trusted_development"] = dev_sc
        if dev_sc == 0.0:
            missing.append("Wymiar 3 (Trusted Development): Brak SBOM i systematycznej ewaluacji modelu.")

        # 4. Incident Reporting
        inc_sc = 1.0 if metadata.get("has_singapore_incident_protocol") else 0.0
        dim_scores["incident_reporting"] = inc_sc
        if inc_sc == 0.0:
            missing.append("Wymiar 4 (Incident Reporting): Brak procedury raportowania anomalii GenAI do organów w Singapurze.")

        # 5. Testing & Assurance (AI Verify Foundation)
        ai_verify = bool(metadata.get("has_ai_verify_test_suite"))
        dim_scores["testing_and_assurance"] = 1.0 if ai_verify else 0.0
        if not ai_verify:
            missing.append("Wymiar 5 (Testing & Assurance): Brak poświadczenia testów w ekosystemie AI Verify Foundation.")

        # 6. Security Safeguards
        sec_sc = 1.0 if metadata.get("has_adversarial_jailbreak_defense") else 0.0
        dim_scores["security_safeguards"] = sec_sc
        if sec_sc == 0.0:
            missing.append("Wymiar 6 (Security): Brak obrony przed atakami jailbreak i zatruwaniem promptów.")

        # 7. Content Provenance (C2PA / Watermarking)
        c2pa = bool(metadata.get("has_c2pa_provenance_watermark"))
        dim_scores["content_provenance"] = 1.0 if c2pa else 0.0
        if not c2pa:
            missing.append("Wymiar 7 (Content Provenance): Brak kryptograficznych metadanych proweniencji C2PA lub znaków wodnych.")

        # 8. Safety Alignment
        safe_sc = 1.0 if metadata.get("has_safety_red_teaming") else 0.0
        dim_scores["safety_alignment"] = safe_sc
        if safe_sc == 0.0:
            missing.append("Wymiar 8 (Safety Alignment): Brak procedury Red-Teamingu przeciwko szkodliwym treściom.")

        # 9. Human-In-The-Loop
        hitl = bool(metadata.get("has_human_oversight_channel"))
        dim_scores["human_in_the_loop"] = 1.0 if hitl else 0.0
        if not hitl:
            missing.append("Wymiar 9 (Human-In-The-Loop): Brak kanału eskalacji do operatora ludzkiego w decyzjach krytycznych.")

        overall_score = round(sum(dim_scores.values()) / len(dim_scores), 3)

        if not c2pa:
            recommendations.append("Wdrożyć standard C2PA (Coalition for Content Provenance and Authenticity) dla generowanych treści.")
        if not ai_verify:
            recommendations.append("Uruchomić testy ewaluacyjne w środowisku AI Verify Toolkit rekomendowanym przez IMDA.")

        return SingaporeIMDAEvaluationResult(
            is_compliant=(len(missing) == 0),
            readiness_score=overall_score,
            dimension_scores=dim_scores,
            c2pa_provenance_verified=c2pa,
            ai_verify_testing_conducted=ai_verify,
            human_in_the_loop_adequate=hitl,
            missing_dimensions=missing,
            recommendations=recommendations,
        )


class AsianSovereignPack:
    """Composite Asian Sovereign Pack covering Japan METI and Singapore IMDA."""

    def __init__(self) -> None:
        self.meti_evaluator = JapanMETIEvaluator()
        self.imda_evaluator = SingaporeIMDAEvaluator()

    def evaluate(self, system_metadata: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> AsianComplianceReport:
        meti_res = self.meti_evaluator.evaluate(system_metadata, payload)
        imda_res = self.imda_evaluator.evaluate(system_metadata, payload)

        avg_score = round((meti_res.compliance_score + imda_res.readiness_score) / 2.0, 3)
        fully_compliant = meti_res.is_compliant and imda_res.is_compliant

        return AsianComplianceReport(
            is_fully_compliant=fully_compliant,
            japan_meti=meti_res,
            singapore_imda=imda_res,
            overall_asian_trust_score=avg_score,
        )
