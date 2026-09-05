"""US Federal & State AI Regulatory Pack (nethical.compliance.packs.us_frontier_nist_pack).

Implements comprehensive compliance evaluations for:
1. NIST AI Risk Management Framework (NIST AI RMF 1.0) - Govern, Map, Measure, Manage
2. California SB 1047 (Safe and Secure Innovation for Frontier Artificial Intelligence Models Act)
3. California AB 2013 (Generative AI Training Data Transparency)
4. US Health & Financial Guardrails (HIPAA ePHI & FTC Act Section 5 Unfair/Deceptive AI Practices)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.compliance.packs.us_frontier_nist")


class NISTFunction(str, Enum):
    """Core functions of the NIST AI Risk Management Framework (AI RMF 1.0)."""
    GOVERN = "GOVERN"
    MAP = "MAP"
    MEASURE = "MEASURE"
    MANAGE = "MANAGE"


class NISTEvaluationResult(BaseModel):
    """Evaluation output for NIST AI RMF 1.0."""
    is_compliant: bool = Field(..., description="Czy system spełnia minimalne progi NIST AI RMF")
    maturity_score: float = Field(..., ge=0.0, le=1.0, description="Wynik dojrzałości zarządzania ryzykiem (0.0 - 1.0)")
    function_scores: Dict[str, float] = Field(default_factory=dict, description="Wyniki cząstkowe dla GOVERN, MAP, MEASURE, MANAGE")
    missing_controls: List[str] = Field(default_factory=list, description="Lista brakujących kontroli")
    recommendations: List[str] = Field(default_factory=list, description="Zalecenia naprawcze")


class SB1047EvaluationResult(BaseModel):
    """Evaluation output for California SB 1047 (Frontier AI Safety)."""
    is_covered_model: bool = Field(..., description="Czy model przekracza progi modelu granicznego (Frontier Model)")
    is_compliant: bool = Field(..., description="Czy model spełnia wymogi bezpieczeństwa SB 1047")
    full_shutdown_capability_verified: bool = Field(..., description="Czy zweryfikowano mechanizm natychmiastowego wyłączenia (Full Shutdown)")
    ssp_documented: bool = Field(..., description="Czy wdrożono pisemny Protokół Bezpieczeństwa (SSP)")
    whistleblower_policy_active: bool = Field(..., description="Czy zapewniono ochronę sygnalistów")
    critical_harm_risk_tier: str = Field(default="LOW", description="Poziom ryzyka krytycznych szkód: LOW, MEDIUM, HIGH, CATASTROPHIC")
    violations: List[str] = Field(default_factory=list)


class AB2013EvaluationResult(BaseModel):
    """Evaluation output for California AB 2013 (Training Data Transparency)."""
    is_compliant: bool = Field(..., description="Czy dokumentacja zbiorów treningowych spełnia AB 2013")
    transparency_score: float = Field(..., ge=0.0, le=1.0)
    data_sources_disclosed: bool = Field(default=False)
    synthetic_data_ratio_disclosed: bool = Field(default=False)
    pii_cleaning_documented: bool = Field(default=False)
    copyright_status_cleared: bool = Field(default=False)
    gaps: List[str] = Field(default_factory=list)


class USFrontierCompositeReport(BaseModel):
    """Composite report across US AI safety & transparency frameworks."""
    overall_compliant: bool
    readiness_index: float
    nist_ai_rmf: NISTEvaluationResult
    california_sb1047: SB1047EvaluationResult
    california_ab2013: AB2013EvaluationResult
    hipaa_ftc_safeguards: Dict[str, Any]
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class NISTAIRMFEvaluator:
    """Evaluates adherence to NIST AI RMF 1.0 (NIST Special Publication 1270)."""

    def evaluate(self, metadata: Dict[str, Any]) -> NISTEvaluationResult:
        missing = []
        scores = {}

        # 1. GOVERN
        gov_checks = [
            metadata.get("has_ai_risk_governance_policy", False),
            metadata.get("has_designated_risk_officer", False),
            metadata.get("has_workforce_diversity_training", False),
        ]
        gov_score = sum(gov_checks) / len(gov_checks)
        scores[NISTFunction.GOVERN.value] = round(gov_score, 2)
        if gov_score < 0.66:
            missing.append("NIST-GOV: Niedostateczna polityka nadzoru lub brak wyznaczonego oficera ds. ryzyka AI.")

        # 2. MAP
        map_checks = [
            metadata.get("has_context_and_use_case_mapping", False),
            metadata.get("has_impact_assessment", False),
            metadata.get("has_identified_legal_frameworks", True),
        ]
        map_score = sum(map_checks) / len(map_checks)
        scores[NISTFunction.MAP.value] = round(map_score, 2)
        if map_score < 0.66:
            missing.append("NIST-MAP: Brak pełnego mapowania kontekstu wdrożenia i oceny potencjalnych szkód.")

        # 3. MEASURE
        measure_checks = [
            metadata.get("has_continuous_bias_testing", False),
            metadata.get("has_adversarial_robustness_metrics", True),
            metadata.get("has_regular_performance_audits", True),
        ]
        measure_score = sum(measure_checks) / len(measure_checks)
        scores[NISTFunction.MEASURE.value] = round(measure_score, 2)
        if measure_score < 0.66:
            missing.append("NIST-MEASURE: Brak ciągłego testowania odporności na ataki i pomiaru odchyleń etycznych.")

        # 4. MANAGE
        manage_checks = [
            metadata.get("has_incident_response_plan", True),
            metadata.get("has_post_deployment_monitoring", True),
            metadata.get("has_kill_switch_or_circuit_breaker", True),
        ]
        manage_score = sum(manage_checks) / len(manage_checks)
        scores[NISTFunction.MANAGE.value] = round(manage_score, 2)
        if manage_score < 0.66:
            missing.append("NIST-MANAGE: Niedostateczne plany zarządzania incydentami i monitorowania powdrożeniowego.")

        overall = sum(scores.values()) / len(scores)
        is_compliant = overall >= 0.70 and len(missing) <= 1

        recommendations = []
        if not is_compliant:
            recommendations.append("Uzupełnij brakujące kontrole NIST AI RMF przed wdrożeniem produkcyjnym.")
        else:
            recommendations.append("System spełnia minimalne kryteria NIST AI RMF 1.0 dla aplikacji krytycznych.")

        return NISTEvaluationResult(
            is_compliant=is_compliant,
            maturity_score=round(overall, 2),
            function_scores=scores,
            missing_controls=missing,
            recommendations=recommendations,
        )


class CaliforniaSB1047Evaluator:
    """Evaluates compliance with California SB 1047 (Frontier Model Safety)."""

    # Progi modelu granicznego: 10^26 FLOPs lub >$100M kosztu obliczeniowego
    FLOP_THRESHOLD = 1e26
    COST_THRESHOLD_USD = 100_000_000

    def evaluate(self, model_specs: Dict[str, Any]) -> SB1047EvaluationResult:
        flops = float(model_specs.get("training_flops", 0.0))
        cost = float(model_specs.get("training_cost_usd", 0.0))
        is_covered = (flops >= self.FLOP_THRESHOLD) or (cost >= self.COST_THRESHOLD_USD)

        violations = []
        has_shutdown = model_specs.get("has_full_shutdown_capability", False)
        has_ssp = model_specs.get("has_safety_security_protocol", False)
        has_whistleblower = model_specs.get("has_whistleblower_protections", False)
        annual_audit = model_specs.get("has_annual_third_party_audit", False)

        if is_covered:
            if not has_shutdown:
                violations.append("SB-1047 § 22602(a): Brak natychmiastowej możliwości pełnego wyłączenia modelu (Full Shutdown Kill-Switch).")
            if not has_ssp:
                violations.append("SB-1047 § 22602(b): Brak udokumentowanego Protokołu Bezpieczeństwa (Safety and Security Protocol).")
            if not has_whistleblower:
                violations.append("SB-1047 § 22603: Brak procedury ochrony pracowników zgłaszających zagrożenia (Whistleblower Protection).")
            if not annual_audit:
                violations.append("SB-1047 § 22604: Brak corocznego niezależnego audytu bezpieczeństwa przez podmiot trzeci.")

        risk_tier = "LOW"
        if is_covered:
            risk_tier = "HIGH" if len(violations) > 0 else "MEDIUM"

        is_compliant = (not is_covered) or (len(violations) == 0)

        return SB1047EvaluationResult(
            is_covered_model=is_covered,
            is_compliant=is_compliant,
            full_shutdown_capability_verified=has_shutdown,
            ssp_documented=has_ssp,
            whistleblower_policy_active=has_whistleblower,
            critical_harm_risk_tier=risk_tier,
            violations=violations,
        )


class CaliforniaAB2013Evaluator:
    """Evaluates transparency of generative AI training datasets under California AB 2013."""

    def evaluate(self, data_manifest: Dict[str, Any]) -> AB2013EvaluationResult:
        gaps = []
        sources = data_manifest.get("data_sources_summary_disclosed", False)
        synthetic_ratio = data_manifest.get("synthetic_data_ratio_disclosed", False)
        pii_cleaned = data_manifest.get("pii_scrubbing_documented", False)
        copyright_cleared = data_manifest.get("copyright_licenses_disclosed", False)

        if not sources:
            gaps.append("AB-2013 § 3100(a): Brak publicznego zestawienia źródeł danych treningowych.")
        if not synthetic_ratio:
            gaps.append("AB-2013 § 3100(b): Brak deklaracji udziału procentowego danych syntetycznych.")
        if not pii_cleaned:
            gaps.append("AB-2013 § 3100(c): Brak udokumentowanego procesu czyszczenia danych osobowych (PII).")
        if not copyright_cleared:
            gaps.append("AB-2013 § 3100(d): Brak deklaracji statusu praw autorskich i licencjonowania zbiorów.")

        score = (int(sources) + int(synthetic_ratio) + int(pii_cleaned) + int(copyright_cleared)) / 4.0

        return AB2013EvaluationResult(
            is_compliant=len(gaps) == 0,
            transparency_score=score,
            data_sources_disclosed=sources,
            synthetic_data_ratio_disclosed=synthetic_ratio,
            pii_cleaning_documented=pii_cleaned,
            copyright_status_cleared=copyright_cleared,
            gaps=gaps,
        )


class USFrontierNISTPack:
    """Główny pakiet ewaluacyjny dla przepisów amerykańskich (NIST, California, HIPAA/FTC)."""

    def __init__(self) -> None:
        self.nist_evaluator = NISTAIRMFEvaluator()
        self.sb1047_evaluator = CaliforniaSB1047Evaluator()
        self.ab2013_evaluator = CaliforniaAB2013Evaluator()

    def evaluate_system(
        self,
        nist_metadata: Dict[str, Any],
        model_specs: Dict[str, Any],
        data_manifest: Dict[str, Any],
        hipaa_data: Optional[Dict[str, Any]] = None,
    ) -> USFrontierCompositeReport:
        """Przeprowadza zbiorczą ewaluację zgodności z prawem USA."""
        nist_res = self.nist_evaluator.evaluate(nist_metadata)
        sb1047_res = self.sb1047_evaluator.evaluate(model_specs)
        ab2013_res = self.ab2013_evaluator.evaluate(data_manifest)

        # HIPAA & FTC safeguards
        hipaa = hipaa_data or {}
        ephi_protected = hipaa.get("has_ephi_encryption", True)
        baa_active = hipaa.get("has_business_associate_agreement", True)
        ftc_no_deceptive_claims = hipaa.get("no_deceptive_performance_claims", True)

        hipaa_ftc_status = {
            "hipaa_compliant": ephi_protected and baa_active,
            "ftc_act_sec5_compliant": ftc_no_deceptive_claims,
            "ephi_encryption": ephi_protected,
            "baa_agreement": baa_active,
        }

        overall = (
            nist_res.is_compliant
            and sb1047_res.is_compliant
            and ab2013_res.is_compliant
            and hipaa_ftc_status["hipaa_compliant"]
            and hipaa_ftc_status["ftc_act_sec5_compliant"]
        )

        readiness = (
            nist_res.maturity_score * 0.35
            + (1.0 if sb1047_res.is_compliant else 0.4) * 0.30
            + ab2013_res.transparency_score * 0.20
            + (1.0 if hipaa_ftc_status["hipaa_compliant"] else 0.5) * 0.15
        )

        return USFrontierCompositeReport(
            overall_compliant=overall,
            readiness_index=round(readiness, 2),
            nist_ai_rmf=nist_res,
            california_sb1047=sb1047_res,
            california_ab2013=ab2013_res,
            hipaa_ftc_safeguards=hipaa_ftc_status,
        )
