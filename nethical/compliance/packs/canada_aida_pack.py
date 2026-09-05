"""Canada Artificial Intelligence and Data Act (AIDA - Bill C-27) Compliance Pack.

Implements statutory compliance evaluation for the Canadian federal jurisdiction:
- High-Impact AI System determination across sensitive commercial sectors.
- Harm & Systemic Risk assessment (physical, psychological, property, economic harm).
- Biased Output mitigation under Canadian Human Rights Act (CHRA).
- Confidential Commercial Data & De-identified Data protections.
- Plain-Language Transparency & Public Disclosure verification.
- Enforcement and Penalties tracking: Administrative Monetary Penalties (AMPs up to 3%
  of global gross revenues or $10M CAD) and Criminal Offence provisions for knowing harm.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.compliance.packs.canada_aida_pack")


class AIDASector(str, Enum):
    EMPLOYMENT_AND_HR = "employment_and_hr"
    ESSENTIAL_SERVICES_CREDIT = "essential_services_credit"
    BIOMETRICS_AND_SURVEILLANCE = "biometrics_and_surveillance"
    AUTONOMOUS_SYSTEMS_KINETIC = "autonomous_systems_kinetic"
    HEALTHCARE_AND_TRIAGE = "healthcare_and_triage"
    GENERAL_COMMERCIAL = "general_commercial"


class HarmCategory(str, Enum):
    PHYSICAL_HARM = "physical_harm"
    PSYCHOLOGICAL_HARM = "psychological_harm"
    PROPERTY_DAMAGE = "property_damage"
    ECONOMIC_LOSS = "economic_loss"


class AIDARiskLevel(str, Enum):
    LOW_IMPACT = "LOW_IMPACT"
    HIGH_IMPACT_COMPLIANT = "HIGH_IMPACT_COMPLIANT"
    HIGH_IMPACT_NON_COMPLIANT = "HIGH_IMPACT_NON_COMPLIANT"
    PROHIBITED_CRIMINAL_RISK = "PROHIBITED_CRIMINAL_RISK"


class AIDAComplianceResult(BaseModel):
    """Evaluation result for Canada AIDA compliance."""
    evaluation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    system_name: str
    sector: AIDASector
    is_high_impact: bool
    risk_level: AIDARiskLevel
    is_compliant: bool
    compliance_score: float = Field(..., ge=0.0, le=1.0)
    identified_harms: List[HarmCategory] = Field(default_factory=list)
    biased_output_risk: bool = False
    plain_language_published: bool = False
    confidential_data_protected: bool = False
    missing_obligations: List[str] = Field(default_factory=list)
    max_statutory_penalty_cad: str = Field(..., description="Potential maximum statutory AMP")
    remediation_recommendations: List[str] = Field(default_factory=list)


class CanadaAIDAPack:
    """Evaluates AI workflows against Canada's Artificial Intelligence and Data Act (AIDA)."""

    def __init__(self) -> None:
        self.jurisdiction = "Canada (Federal - Bill C-27 / AIDA)"
        self.regulator = "AI and Data Commissioner (Innovation, Science and Economic Development Canada - ISED)"

    def evaluate(self, system_metadata: Dict[str, Any]) -> AIDAComplianceResult:
        """Evaluates a system profile against AIDA statutory requirements."""
        system_name = system_metadata.get("system_name", "Autonomous-Agent-Service")
        sector_raw = system_metadata.get("sector", "general_commercial")
        try:
            sector = AIDASector(sector_raw)
        except ValueError:
            sector = AIDASector.GENERAL_COMMERCIAL

        # 1. High Impact Determination (Schedule of High-Impact Systems)
        high_impact_sectors = {
            AIDASector.EMPLOYMENT_AND_HR,
            AIDASector.ESSENTIAL_SERVICES_CREDIT,
            AIDASector.BIOMETRICS_AND_SURVEILLANCE,
            AIDASector.AUTONOMOUS_SYSTEMS_KINETIC,
            AIDASector.HEALTHCARE_AND_TRIAGE,
        }
        is_high_impact = sector in high_impact_sectors or system_metadata.get("is_safety_critical", False)

        # 2. Harm Risk Assessment
        harms: List[HarmCategory] = []
        if system_metadata.get("potential_physical_harm", False):
            harms.append(HarmCategory.PHYSICAL_HARM)
        if system_metadata.get("potential_psychological_harm", False):
            harms.append(HarmCategory.PSYCHOLOGICAL_HARM)
        if system_metadata.get("potential_property_damage", False):
            harms.append(HarmCategory.PROPERTY_DAMAGE)
        if system_metadata.get("potential_economic_loss", False):
            harms.append(HarmCategory.ECONOMIC_LOSS)

        # 3. Biased Output & Human Rights
        has_bias_audit = system_metadata.get("has_bias_audit", False)
        biased_output_risk = not has_bias_audit and is_high_impact

        # 4. Plain-Language Transparency
        plain_lang = system_metadata.get("has_plain_language_summary", False)

        # 5. Confidential Commercial Data
        confidential_data_protected = system_metadata.get("confidential_data_protected", True)

        missing: List[str] = []
        remediation: List[str] = []

        if is_high_impact:
            if not has_bias_audit:
                missing.append("AIDA_SEC_6: Lack of documented bias mitigation measures (Canadian Human Rights Act)")
                remediation.append("Execute demographic disparity and algorithmic fairness testing.")
            if not plain_lang:
                missing.append("AIDA_SEC_11: Failure to publish plain-language description of high-impact system")
                remediation.append("Publish accessible summary of system objectives, risk mitigations, and data sources.")
            if not system_metadata.get("has_risk_management_policy", False):
                missing.append("AIDA_SEC_8: Absence of formal risk mitigation and incident monitoring policy")
                remediation.append("Implement pre-execution governance gate with continuous logging.")
            if not confidential_data_protected:
                missing.append("AIDA_SEC_5: Inadequate protection of confidential commercial information")
                remediation.append("Deploy Reversible Token Vault and zero-egress enclaves.")

        # Criminal Offence Trigger: Knowing creation of substantial harm or fraud
        is_criminal = (
            HarmCategory.PHYSICAL_HARM in harms and
            not system_metadata.get("has_safety_interlock", False)
        )

        if is_criminal:
            risk_level = AIDARiskLevel.PROHIBITED_CRIMINAL_RISK
            is_compliant = False
            compliance_score = 0.0
            max_penalty = "CRIMINAL_LIABILITY (Fines up to $25,000,000 CAD or imprisonment up to 5 years)"
            remediation.insert(0, "EMERGENCY: Immediate kill-switch shutdown required to avoid criminal prosecution.")
        elif is_high_impact:
            passed_checks = 4 - len(missing)
            compliance_score = max(0.0, passed_checks / 4.0)
            is_compliant = len(missing) == 0
            risk_level = AIDARiskLevel.HIGH_IMPACT_COMPLIANT if is_compliant else AIDARiskLevel.HIGH_IMPACT_NON_COMPLIANT
            max_penalty = "Up to 3% of global gross revenues or $10,000,000 CAD (AIDA AMPs)"
        else:
            is_compliant = True
            compliance_score = 1.0
            risk_level = AIDARiskLevel.LOW_IMPACT
            max_penalty = "Not applicable (Standard Commercial Oversight)"

        eval_id = f"AIDA-EVAL-{int(datetime.now(timezone.utc).timestamp())}"

        return AIDAComplianceResult(
            evaluation_id=eval_id,
            system_name=system_name,
            sector=sector,
            is_high_impact=is_high_impact,
            risk_level=risk_level,
            is_compliant=is_compliant,
            compliance_score=round(compliance_score, 2),
            identified_harms=harms,
            biased_output_risk=biased_output_risk,
            plain_language_published=plain_lang,
            confidential_data_protected=confidential_data_protected,
            missing_obligations=missing,
            max_statutory_penalty_cad=max_penalty,
            remediation_recommendations=remediation,
        )
