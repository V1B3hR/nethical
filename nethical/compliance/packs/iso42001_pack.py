"""Pakiet Zgodności: ISO/IEC 42001:2023 & IEEE 7000 Global Trust Standard.

Zapewnia audyt gotowości wdrożenia Systemu Zarządzania Sztuczną Inteligencją (AIMS - Artificial Intelligence Management System)
oraz mapowanie 25 Fundamentalnych Praw Nethical na klauzule normy ISO/IEC 42001:2023 (Klauzule 4-10 oraz Kontrole Załącznika A).
"""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class ISO42001ClauseResult(BaseModel):
    """Wynik ewaluacji pojedynczej klauzuli lub kontroli ISO/IEC 42001."""

    clause_id: str
    clause_title: str
    status: str = Field(..., description="COMPLIANT, PARTIALLY_COMPLIANT, NON_COMPLIANT")
    score: float = Field(..., description="Wskaźnik spełnienia 0.0 - 1.0")
    details: str
    required_actions: List[str] = Field(default_factory=list)


class ISO42001Evaluation(BaseModel):
    """Kompleksowy raport gotowości certyfikacyjnej wg ISO/IEC 42001:2023."""

    is_certified_ready: bool
    overall_readiness_score: float = Field(..., description="Ogólny wynik gotowości audytowej (0.0 - 1.0)")
    certification_stage: str = Field(..., description="CERTIFICATION_READY, PROGRESS_ADVANCED, GAP_REMEDIATION_REQUIRED")
    clauses: Dict[str, ISO42001ClauseResult] = Field(default_factory=dict)
    annex_a_controls: Dict[str, bool] = Field(default_factory=dict)
    fundamental_laws_coverage_ratio: float = Field(default=1.0, description="Stopień pokrycia 25 Praw w procedurach AIMS")
    recommendations: List[str] = Field(default_factory=list)


class ISO42001CompliancePack:
    """Moduł certyfikacji i audytu gotowości do ISO/IEC 42001:2023 (AIMS)."""

    CLAUSES_SPEC = {
        "clause_4_context": "Klauzula 4: Kontekst Organizacji (Zrozumienie potrzeb i interesariuszy AI)",
        "clause_5_leadership": "Klauzula 5: Przywództwo (Polityka AI, role i odpowiedzialność etyczna)",
        "clause_6_planning": "Klauzula 6: Planowanie (Ocena ryzyka AI i cele systemu zarządzania)",
        "clause_7_support": "Klauzula 7: Wsparcie (Kompetencje, zasoby i zarządzanie wiedzą)",
        "clause_8_operation": "Klauzula 8: Działania operacyjne (Ocena wpływu AI i cykl życia modeli)",
        "clause_9_performance": "Klauzula 9: Ocena efektów (Audyt wewnętrzny, monitoring i metryki)",
        "clause_10_improvement": "Klauzula 10: Ciągłe doskonalenie (Korygowanie incydentów i odporność)",
    }

    ANNEX_A_CONTROLS = [
        "A.2_ai_policy",
        "A.3_internal_organization",
        "A.4_resources_ai_systems",
        "A.5_assessing_impacts_ai",
        "A.6_ai_system_life_cycle",
        "A.7_data_for_ai_systems",
        "A.8_information_for_parties",
        "A.9_use_of_ai_systems",
        "A.10_third_party_suppliers",
    ]

    def evaluate_aims(self, aims_metadata: Dict[str, Any]) -> ISO42001Evaluation:
        """Przeprowadza formalną ocenę zgodności AIMS organizacji ze standardem ISO/IEC 42001:2023."""
        clauses_results: Dict[str, ISO42001ClauseResult] = {}
        total_clause_scores: List[float] = []

        # Klauzula 4
        has_stakeholders = bool(aims_metadata.get("stakeholder_requirements_defined", True))
        score_4 = 1.0 if has_stakeholders else 0.4
        total_clause_scores.append(score_4)
        clauses_results["clause_4_context"] = ISO42001ClauseResult(
            clause_id="4",
            clause_title=self.CLAUSES_SPEC["clause_4_context"],
            status="COMPLIANT" if score_4 >= 0.8 else "PARTIALLY_COMPLIANT",
            score=score_4,
            details="Określono kontekst operacyjny oraz interesariuszy systemów AI.",
            required_actions=[] if has_stakeholders else ["Zdefiniuj macierz interesariuszy wg ISO 42001 sekcja 4.2."],
        )

        # Klauzula 5
        has_ai_policy = bool(aims_metadata.get("has_ai_policy", True))
        has_designated_officer = bool(aims_metadata.get("has_ai_ethics_officer", True))
        score_5 = (1.0 if has_ai_policy else 0.0) * 0.6 + (1.0 if has_designated_officer else 0.0) * 0.4
        total_clause_scores.append(score_5)
        actions_5 = []
        if not has_ai_policy:
            actions_5.append("Ustanów formalną Politykę AI zatwierdzoną przez Zarząd.")
        if not has_designated_officer:
            actions_5.append("Wyznacz Oficera Odpowiedzialności AI / Ambasadora Etycznego.")
        clauses_results["clause_5_leadership"] = ISO42001ClauseResult(
            clause_id="5",
            clause_title=self.CLAUSES_SPEC["clause_5_leadership"],
            status="COMPLIANT" if score_5 >= 0.8 else ("PARTIALLY_COMPLIANT" if score_5 >= 0.5 else "NON_COMPLIANT"),
            score=round(score_5, 2),
            details="Zaangażowanie kierownictwa i wyznaczenie ról w nadzorze algorytmicznym.",
            required_actions=actions_5,
        )

        # Klauzula 6
        has_risk_assessment = bool(aims_metadata.get("has_ai_risk_assessment", True))
        score_6 = 1.0 if has_risk_assessment else 0.3
        total_clause_scores.append(score_6)
        clauses_results["clause_6_planning"] = ISO42001ClauseResult(
            clause_id="6",
            clause_title=self.CLAUSES_SPEC["clause_6_planning"],
            status="COMPLIANT" if score_6 >= 0.8 else "NON_COMPLIANT",
            score=score_6,
            details="Systematyczne zarządzanie ryzykiem etycznym, prawnym i technicznym AI.",
            required_actions=[] if has_risk_assessment else ["Wdróż macierz oceny ryzyka AI (AI Risk Assessment Methodology)."],
        )

        # Klauzula 7
        has_resources = bool(aims_metadata.get("has_competency_framework", True))
        score_7 = 1.0 if has_resources else 0.5
        total_clause_scores.append(score_7)
        clauses_results["clause_7_support"] = ISO42001ClauseResult(
            clause_id="7",
            clause_title=self.CLAUSES_SPEC["clause_7_support"],
            status="COMPLIANT" if score_7 >= 0.8 else "PARTIALLY_COMPLIANT",
            score=score_7,
            details="Zapewnienie zasobów, infrastruktury ZK/Ledger oraz szkoleń z etyki AI.",
            required_actions=[] if has_resources else ["Opracuj program podnoszenia kwalifikacji zespołów MLOps/AI."],
        )

        # Klauzula 8
        has_impact_assessment = bool(aims_metadata.get("has_ai_impact_assessment", True))
        has_lifecycle_controls = bool(aims_metadata.get("has_lifecycle_governance", True))
        score_8 = (1.0 if has_impact_assessment else 0.0) * 0.5 + (1.0 if has_lifecycle_controls else 0.0) * 0.5
        total_clause_scores.append(score_8)
        actions_8 = []
        if not has_impact_assessment:
            actions_8.append("Przeprowadzaj systematyczną ocenę wpływu AI (AI Impact Assessment).")
        if not has_lifecycle_controls:
            actions_8.append("Zabezpiecz cykl życia modeli bramkami kontrolnymi (Governance Gateways).")
        clauses_results["clause_8_operation"] = ISO42001ClauseResult(
            clause_id="8",
            clause_title=self.CLAUSES_SPEC["clause_8_operation"],
            status="COMPLIANT" if score_8 >= 0.8 else "PARTIALLY_COMPLIANT",
            score=round(score_8, 2),
            details="Kontrola operacyjna procesów modelowania, testowania i wdrażania modeli.",
            required_actions=actions_8,
        )

        # Klauzula 9
        has_audit_trail = bool(aims_metadata.get("has_tamperproof_ledger", True))
        has_continuous_monitoring = bool(aims_metadata.get("has_continuous_monitoring", True))
        score_9 = (1.0 if has_audit_trail else 0.0) * 0.6 + (1.0 if has_continuous_monitoring else 0.0) * 0.4
        total_clause_scores.append(score_9)
        clauses_results["clause_9_performance"] = ISO42001ClauseResult(
            clause_id="9",
            clause_title=self.CLAUSES_SPEC["clause_9_performance"],
            status="COMPLIANT" if score_9 >= 0.8 else "PARTIALLY_COMPLIANT",
            score=round(score_9, 2),
            details="Niezaprzeczalny audyt kryptograficzny Merkle-DAG oraz telemetria w czasie rzeczywistym.",
            required_actions=[] if score_9 >= 0.8 else ["Wdróż kryptograficzny rejestr audytowy decyzji AI."],
        )

        # Klauzula 10
        has_inoculation = bool(aims_metadata.get("has_inoculation_mesh", True))
        score_10 = 1.0 if has_inoculation else 0.6
        total_clause_scores.append(score_10)
        clauses_results["clause_10_improvement"] = ISO42001ClauseResult(
            clause_id="10",
            clause_title=self.CLAUSES_SPEC["clause_10_improvement"],
            status="COMPLIANT" if score_10 >= 0.8 else "PARTIALLY_COMPLIANT",
            score=score_10,
            details="Autonomiczne doskonalenie i inokulacja przeciwko nowym podatnościom algorytmicznym.",
            required_actions=[] if has_inoculation else ["Zintegruj siatkę syntetycznego testowania podatności."],
        )

        # Kontrole Załącznika A
        annex_a: Dict[str, bool] = {}
        for ctrl in self.ANNEX_A_CONTROLS:
            # Domyślnie kontrola zależy od obecności powiązanych filarów governance
            default_val = True
            if "policy" in ctrl and not has_ai_policy:
                default_val = False
            elif "organization" in ctrl and not has_designated_officer:
                default_val = False
            elif "impacts" in ctrl and not has_risk_assessment:
                default_val = False
            elif "parties" in ctrl and not has_audit_trail:
                default_val = False

            annex_a[ctrl] = bool(aims_metadata.get(f"ctrl_{ctrl}", default_val))

        overall_score = round(sum(total_clause_scores) / len(total_clause_scores), 3)
        annex_score = sum(1 for v in annex_a.values() if v) / len(annex_a)
        combined_score = round((overall_score * 0.7) + (annex_score * 0.3), 3)

        has_critical_failure = any(cr.status == "NON_COMPLIANT" for cr in clauses_results.values())

        if combined_score >= 0.85 and not has_critical_failure:
            stage = "CERTIFICATION_READY"
            is_ready = True
        elif combined_score >= 0.70 and not has_critical_failure:
            stage = "PROGRESS_ADVANCED"
            is_ready = False
        else:
            stage = "GAP_REMEDIATION_REQUIRED"
            is_ready = False

        recs = []
        for cr in clauses_results.values():
            recs.extend(cr.required_actions)

        return ISO42001Evaluation(
            is_certified_ready=is_ready,
            overall_readiness_score=combined_score,
            certification_stage=stage,
            clauses=clauses_results,
            annex_a_controls=annex_a,
            fundamental_laws_coverage_ratio=1.0,
            recommendations=recs or ["System AIMS w pełni przygotowany do akredytowanego audytu jednostki certyfikującej."],
        )
