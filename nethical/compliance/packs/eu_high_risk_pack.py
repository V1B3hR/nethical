"""Pakiet Zgodności: EU High-Risk AI Pack (EU AI Act Artykuły 6, 9-15).

Zapewnia zautomatyzowaną weryfikację systemów AI wysokiego ryzyka
zgodnie z wymogami unijnego rozporządzenia o sztucznej inteligencji.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


class EUHighRiskEvaluation(BaseModel):
    """Wynik ewaluacji zgodności z EU AI Act dla systemu wysokiego ryzyka."""
    is_compliant: bool
    risk_tier: str = "HIGH_RISK"
    annex_category: str
    checks: Dict[str, bool] = Field(default_factory=dict)
    missing_requirements: List[str] = Field(default_factory=list)
    human_oversight_verified: bool
    ce_marking_readiness_score: float  # 0.0 - 1.0


class EUHighRiskPack:
    """Moduł weryfikacji wymogów EU AI Act dla systemów wysokiego ryzyka."""

    HIGH_RISK_DOMAINS = [
        "critical_infrastructure",
        "education_vocational_training",
        "employment_worker_management",
        "essential_private_public_services",
        "law_enforcement",
        "migration_asylum_border",
        "administration_of_justice",
        "biometric_identification",
    ]

    def evaluate_system(self, system_metadata: Dict[str, Any]) -> EUHighRiskEvaluation:
        """Ocenia architekturę systemu AI pod kątem wymogów EU AI Act."""
        domain = system_metadata.get("domain", "general")
        annex_match = domain in self.HIGH_RISK_DOMAINS or system_metadata.get("is_high_risk", False)
        
        checks = {
            "risk_management_system": bool(system_metadata.get("has_risk_management", False)),
            "data_governance_documented": bool(system_metadata.get("has_data_governance", False)),
            "technical_documentation_annex_iv": bool(system_metadata.get("has_tech_docs", False)),
            "record_keeping_logging": bool(system_metadata.get("has_automatic_logging", True)),
            "transparency_user_information": bool(system_metadata.get("has_user_disclosure", True)),
            "human_oversight_hitl": bool(system_metadata.get("has_human_oversight", False)),
            "accuracy_robustness_cybersecurity": bool(system_metadata.get("has_security_testing", False)),
        }

        missing = [k for k, v in checks.items() if not v]
        score = sum(1 for v in checks.values() if v) / len(checks)
        is_compliant = len(missing) == 0

        return EUHighRiskEvaluation(
            is_compliant=is_compliant,
            risk_tier="HIGH_RISK" if annex_match else "STANDARD",
            annex_category=domain if annex_match else "non_high_risk_tier",
            checks=checks,
            missing_requirements=missing,
            human_oversight_verified=checks["human_oversight_hitl"],
            ce_marking_readiness_score=round(score, 2),
        )
