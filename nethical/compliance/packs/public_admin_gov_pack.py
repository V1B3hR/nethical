"""Public Administration and Sovereign Government Governance Pack.

Implements statutory compliance and administrative law invariants for public sector institutions,
ministries, government agencies, municipal offices, and sovereign public services:
- Kodeks Postępowania Administracyjnego (KPA - Polish Code of Administrative Procedure):
  * Art. 7: Zasada prawdy obiektywnej (Objective truth - facts must be proven, not probabilistic).
  * Art. 8: Zasada pogłębiania zaufania obywateli do organów władzy publicznej.
  * Art. 10: Prawo strony do czynnego udziału w postępowaniu.
  * Art. 77 § 1: Obowiązek wyczerpującego zebrania i rozpatrzenia materiału dowodowego.
  * Art. 107 § 3: Bezwzględny wymóg pełnego uzasadnienia faktycznego i prawnego (Zakaz "czarnej skrzynki").
- Rozporządzenie ws. Krajowych Ram Interoperacyjności (KRI):
  * Archiwizacja EZD, formaty otwarte (XML, PDF/A), dostępność cyfrowa WCAG 2.1 AA.
- Ustawa o ochronie informacji niejawnych (UOIN):
  * Klauzule: ZASTRZEŻONE, POUFNE, TAJNE, ŚCIŚLE TAJNE.
  * Zasada wiedzy koniecznej (Need-to-Know), zakaz transferu poza certyfikowane systemy teleinformatyczne.
- Ustawa o dostępie do informacji publicznej (UDIP) & Biuletyn Informacji Publicznej (BIP).

Hard Invariants Enforced:
1. Prohibition of Unsigned Autonomous Administrative Decisions (Art. 107 KPA & GDPR Art. 22).
2. Prohibition of "Black Box" Algorithmic Decisions (Plain-language administrative reasoning required).
3. Prohibition of Probabilistic Sanctions (Art. 7 KPA - Administrative penalties require proven facts).
4. Classified Information Leakage Guard (Compartmentalization & ABW/SKW clearance enforcement).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.compliance.packs.public_admin_gov_pack")


class ClearanceLevel(str, Enum):
    JAWNE = "JAWNE"                         # Unclassified / Public
    ZASTRZEZONE = "ZASTRZEZONE"             # Restricted
    POUFNE = "POUFNE"                       # Confidential
    TAJNE = "TAJNE"                         # Secret
    SCISLE_TAJNE = "SCISLE_TAJNE"           # Top Secret


class AdminDecisionStatus(str, Enum):
    LAWFUL_AND_ACTIONABLE = "LAWFUL_AND_ACTIONABLE"
    REQUIRES_EVIDENTIARY_COMPLETION = "REQUIRES_EVIDENTIARY_COMPLETION"
    DEFECTIVE_BLACK_BOX = "DEFECTIVE_BLACK_BOX"
    PROHIBITED_ARBITRARY_DECISION = "PROHIBITED_ARBITRARY_DECISION"


class PublicAdminComplianceResult(BaseModel):
    """Evaluation result for Public Administration governance compliance."""
    evaluation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agency_or_system_name: str
    clearance_level: ClearanceLevel
    decision_status: AdminDecisionStatus
    is_compliant: bool
    compliance_score: float = Field(..., ge=0.0, le=1.0)
    kpa_art7_objective_truth_verified: bool
    kpa_art107_reasoning_provided: bool
    qualified_human_signature_verified: bool
    classified_data_airgapped: bool
    kri_interoperability_compliant: bool
    violations: List[str] = Field(default_factory=list)
    missing_administrative_elements: List[str] = Field(default_factory=list)
    remediation_recommendations: List[str] = Field(default_factory=list)


class PublicAdminGovPack:
    """Evaluates public sector and municipal AI workflows against KPA, KRI, and state secrets law."""

    def __init__(self) -> None:
        self.jurisdiction = "Poland (KPA, KRI, UOIN) & European Union (Charter of Fundamental Rights Art. 41)"
        self.supervisory_bodies = [
            "Naczelny Sąd Administracyjny (NSA)",
            "Wojewódzkie Sądy Administracyjne (WSA)",
            "Ministerstwo Cyfryzacji (MC)",
            "Najwyższa Izba Kontroli (NIK)",
            "Agencja Bezpieczeństwa Wewnętrznego (ABW)",
        ]

    def evaluate(self, payload: Dict[str, Any]) -> PublicAdminComplianceResult:
        """Evaluates an administrative AI action against KPA statutory requirements."""
        system_name = payload.get("system_name", "Public-Service-Gateway")
        eval_id = f"GOV-{uuid.uuid4().hex[:12].upper()}"

        # 1. Determine Clearance Level
        raw_clearance = payload.get("clearance_level", "JAWNE")
        try:
            clearance = ClearanceLevel(raw_clearance)
        except ValueError:
            clearance = ClearanceLevel.JAWNE

        violations: List[str] = []
        missing_elements: List[str] = []
        remediations: List[str] = []

        # 2. Hard Invariant I: Human Official Signature (KPA Art. 107 § 1 pkt 8)
        is_autonomous_final_decision = payload.get("is_autonomous_final_decision", False)
        has_qualified_signature = payload.get("qualified_human_signature", False) or payload.get("official_epuap_signed", False)

        if is_autonomous_final_decision and not has_qualified_signature:
            violations.append(
                "CRITICAL: System attempted to issue a final, legally binding administrative decision (decyzja administracyjna) "
                "without qualified electronic signature of authorized public official (Art. 107 § 1 pkt 8 KPA & Art. 22 RODO). "
                "Decyzja obarczona wadą nieważności z mocy prawa (Art. 156 § 1 pkt 2 KPA)."
            )
            remediations.append("Decyzja administracyjna musi być podpisana profilem zaufanym lub podpisem kwalifikowanym przez urzędnika.")

        # 3. Hard Invariant II: Full Legal and Factual Reasoning (KPA Art. 107 § 3 - Anti Black Box)
        provides_legal_reasoning = payload.get("provides_legal_basis_and_reasoning", True)
        is_black_box = payload.get("is_black_box_model", False) or not provides_legal_reasoning

        if is_black_box:
            violations.append(
                "VIOLATION: Omission of intelligible legal and factual reasoning (Uzasadnienie faktyczne i prawne). "
                "Obywatelowi nie można odmówić uprawnienia lub nałożyć kary powołując się wyłącznie na wyliczenie algorytmu (Art. 107 § 3 KPA)."
            )
            remediations.append("Wygeneruj pełne uzasadnienie ze wskazaniem faktów, dowodów oraz dokładnych artykułów prawa.")

        # 4. Hard Invariant III: Objective Truth vs. Probabilistic Speculation (KPA Art. 7 & 77)
        relies_on_unproven_probability = payload.get("relies_on_unproven_probabilistic_model", False)
        art7_verified = True
        if relies_on_unproven_probability:
            art7_verified = False
            violations.append(
                "VIOLATION: Administrative sanction or denial based on probabilistic prediction without exhaustive evidentiary proof "
                "(Złamanie zasady prawdy obiektywnej Art. 7 i obowiązku wyczerpującego zebrania dowodów Art. 77 § 1 KPA)."
            )
            remediations.append("Wstrzymaj orzekanie do czasu zebrania dokumentacji źródłowej i protokołów oględzin.")

        # 5. Classified Information & State Secrets Protection (UOIN)
        classified_leak = payload.get("unauthorized_classified_egress", False)
        classified_airgapped = True
        if clearance in (ClearanceLevel.POUFNE, ClearanceLevel.TAJNE, ClearanceLevel.SCISLE_TAJNE):
            is_airgapped = payload.get("is_airgapped_node", True)
            has_abw_cert = payload.get("has_abw_skw_accreditation", False)
            if not is_airgapped or not has_abw_cert or classified_leak:
                classified_airgapped = False
                violations.append(
                    f"CRITICAL: Przetwarzanie informacji niejawnych o klauzuli {clearance.value} w systemie nieposiadającym "
                    f"akredytacji bezpieczeństwa teleinformatycznego ABW/SKW lub poza węzłem Air-Gapped (Art. 48 UOIN)."
                )
                remediations.append("Natychmiast skieruj przetwarzanie do izolowanego węzła AirGappedSovereignNode z certyfikacją ABW.")

        # 6. Interoperability & Accessibility (KRI & WCAG)
        kri_compliant = payload.get("kri_interoperability_compliant", False)
        if not kri_compliant:
            missing_elements.append("Brak poświadczenia zgodności z Krajowymi Ramami Interoperacyjności (KRI - formaty otwarte i archiwizacja EZD).")
            remediations.append("Dostosuj eksport decyzji do formatu PDF/A-2a oraz schematu XML zgodnego z e-PUAP.")

        wcag_accessible = payload.get("wcag_21_aa_accessible", False)
        if not wcag_accessible:
            missing_elements.append("Brak deklaracji dostępności cyfrowej dla osób z niepełnosprawnościami (standard WCAG 2.1 AA).")
            remediations.append("Wdróż syntezę mowy i alternatywne kontrasty w interfejsie obywatelskim.")

        # 7. Scoring and Decision Status
        deductions = (len(violations) * 0.35) + (len(missing_elements) * 0.10)
        score = max(0.0, min(1.0, 1.0 - deductions))

        if not classified_airgapped or (is_autonomous_final_decision and not has_qualified_signature):
            decision_status = AdminDecisionStatus.PROHIBITED_ARBITRARY_DECISION
            is_compliant = False
        elif is_black_box or not art7_verified:
            decision_status = AdminDecisionStatus.DEFECTIVE_BLACK_BOX
            is_compliant = False
        elif missing_elements:
            decision_status = AdminDecisionStatus.REQUIRES_EVIDENTIARY_COMPLETION
            is_compliant = score >= 0.70
        else:
            decision_status = AdminDecisionStatus.LAWFUL_AND_ACTIONABLE
            is_compliant = True

        logger.info(
            "PublicAdminGovPack evaluation complete: %s, Clearance: %s, Compliant: %s, Score: %.2f",
            eval_id, clearance.value, is_compliant, score
        )

        return PublicAdminComplianceResult(
            evaluation_id=eval_id,
            agency_or_system_name=system_name,
            clearance_level=clearance,
            decision_status=decision_status,
            is_compliant=is_compliant,
            compliance_score=round(score, 3),
            kpa_art7_objective_truth_verified=art7_verified,
            kpa_art107_reasoning_provided=not is_black_box,
            qualified_human_signature_verified=has_qualified_signature or not is_autonomous_final_decision,
            classified_data_airgapped=classified_airgapped,
            kri_interoperability_compliant=kri_compliant,
            violations=violations,
            missing_administrative_elements=missing_elements,
            remediation_recommendations=remediations,
        )
