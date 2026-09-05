"""Pakiet Zgodności: UK Cyber, Data & Infrastructure Compliance Pack.

Obejmuje 3 kluczowe brytyjskie ramy prawne:
1. Computer Misuse Act 1990 (CMA 1990):
   - Sec 1: Unauthorized access to computer material
   - Sec 2: Unauthorized access with intent to commit or facilitate commission of further offences
   - Sec 3: Unauthorized acts with intent to impair, or with recklessness as to impairing, operation of computer
   - Sec 3A: Making, supplying or obtaining articles for use in offence under section 1 or 3
2. UK GDPR and Data Protection Act 2018 (UK DPA 2018):
   - Art. 5: Core Data Protection Principles & Accountability
   - Art. 9: Special Category Data processing safeguards
   - Art. 22: Automated Individual Decision-Making & Profiling (Right to Human Intervention)
   - Chapter V: International transfers (UK Adequacy & IDTA)
3. Network and Information Systems (NIS) Regulations 2018 (UK NIS):
   - Operators of Essential Services (OES) & Relevant Digital Service Providers (RDSPs)
   - Mandatory statutory incident reporting (<72h) to Competent Authorities (ICO, Ofgem, DfT)
   - Security duties and resilience risk management
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.compliance.packs.uk_cyber_data")


# ==============================================================================
# 1. COMPUTER MISUSE ACT 1990
# ==============================================================================

class CMAViolationType:
    SECTION_1_UNAUTHORIZED_ACCESS = "CMA_SEC_1_UNAUTHORIZED_ACCESS"
    SECTION_2_ACCESS_WITH_FURTHER_INTENT = "CMA_SEC_2_ACCESS_WITH_FURTHER_INTENT"
    SECTION_3_UNAUTHORIZED_IMPAIRMENT = "CMA_SEC_3_UNAUTHORIZED_IMPAIRMENT"
    SECTION_3A_SUPPLYING_MALICIOUS_ARTICLES = "CMA_SEC_3A_SUPPLYING_ARTICLES"


class CMAEvaluation(BaseModel):
    """Wynik ewaluacji zgodności z brytyjską ustawą Computer Misuse Act 1990."""

    is_compliant: bool
    offences_detected: List[str] = Field(default_factory=list)
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL_CRIMINAL")
    offence_summaries: List[str] = Field(default_factory=list)
    statutory_citations: List[str] = Field(default_factory=list)
    interlock_action: str = Field(..., description="ALLOW, RESTRICT, BLOCK, TERMINATE")


class ComputerMisuseActEvaluator:
    """Ewaluator zapobiegający popełnieniu przestępstw komputerowych pod CMA 1990."""

    SEC1_PATTERNS = [
        r"(?i)\b(unauthorized\s+access|bypass\s+auth|brute\s*force|credential\s*dump|crack\s*password)\b",
        r"(?i)\b(dump\s+sam|mimikatz|shadow\s+copy\s+dump|extract\s+ntds\.dit)\b",
    ]
    SEC2_PATTERNS = [
        r"(?i)\b(exfiltrate\s+data|steal\s+database|ransomware|blackmail|corporate\s+espionage)\b",
        r"(?i)\b(transfer\s+unauthorized\s+funds|wire\s+fraud)\b",
    ]
    SEC3_PATTERNS = [
        r"(?i)\b(ddos|dos\s+attack|fork\s*bomb|wipe\s+disk|format\s+c:|rm\s+-rf\s+/|disable\s+firewall)\b",
        r"(?i)\b(corrupt\s+data|system\s+sabotage|halt\s+industrial\s+control|scada\s+override)\b",
    ]
    SEC3A_PATTERNS = [
        r"(?i)\b(generate\s+exploit|build\s+botnet|payload\s+generator|c2\s+framework|keylogger\s+source)\b",
        r"(?i)\b(reverse\s+shell\s+generator|rootkit\s+builder)\b",
    ]

    @classmethod
    def evaluate(cls, action_payload: Dict[str, Any]) -> CMAEvaluation:
        text = str(action_payload.get("input", "")) + " " + str(action_payload.get("command", "")) + " " + str(action_payload.get("tool_name", ""))
        detected = []
        summaries = []
        citations = []

        # Sec 1 Check
        if any(re.search(p, text) for p in cls.SEC1_PATTERNS):
            detected.append(CMAViolationType.SECTION_1_UNAUTHORIZED_ACCESS)
            summaries.append("Wykryto próbę nieautoryzowanego dostępu do materiałów komputerowych.")
            citations.append("Computer Misuse Act 1990 Section 1 (Unauthorized access)")

        # Sec 2 Check
        if any(re.search(p, text) for p in cls.SEC2_PATTERNS):
            detected.append(CMAViolationType.SECTION_2_ACCESS_WITH_FURTHER_INTENT)
            summaries.append("Wykryto zamiar popełnienia dalszego przestępstwa (kradzież/szantaż/oszustwo).")
            citations.append("Computer Misuse Act 1990 Section 2 (Access with intent to commit further offence)")

        # Sec 3 Check
        if any(re.search(p, text) for p in cls.SEC3_PATTERNS):
            detected.append(CMAViolationType.SECTION_3_UNAUTHORIZED_IMPAIRMENT)
            summaries.append("Wykryto próbę bezprawnego zakłócenia lub zniszczenia działania systemu/danych.")
            citations.append("Computer Misuse Act 1990 Section 3 (Unauthorized acts with intent to impair)")

        # Sec 3A Check
        if any(re.search(p, text) for p in cls.SEC3A_PATTERNS):
            detected.append(CMAViolationType.SECTION_3A_SUPPLYING_MALICIOUS_ARTICLES)
            summaries.append("Wykryto wytwarzanie lub dostarczanie narzędzi/artykułów do cyberprzestępstw.")
            citations.append("Computer Misuse Act 1990 Section 3A (Making, supplying or obtaining articles)")

        is_compliant = len(detected) == 0
        if not is_compliant:
            if CMAViolationType.SECTION_3_UNAUTHORIZED_IMPAIRMENT in detected or CMAViolationType.SECTION_3A_SUPPLYING_MALICIOUS_ARTICLES in detected:
                risk_level = "CRITICAL_CRIMINAL"
                interlock_action = "TERMINATE"
            else:
                risk_level = "HIGH"
                interlock_action = "BLOCK"
        else:
            risk_level = "LOW"
            interlock_action = "ALLOW"

        return CMAEvaluation(
            is_compliant=is_compliant,
            offences_detected=detected,
            risk_level=risk_level,
            offence_summaries=summaries,
            statutory_citations=citations,
            interlock_action=interlock_action,
        )


# ==============================================================================
# 2. UK GDPR & DATA PROTECTION ACT 2018
# ==============================================================================

class UKGDPREvaluation(BaseModel):
    """Wynik ewaluacji zgodności z UK GDPR i Data Protection Act 2018."""

    is_compliant: bool
    lawful_basis_documented: bool
    special_category_data_cleared: bool
    automated_decision_safeguards_active: bool
    data_minimisation_score: float  # 0.0 - 1.0
    international_transfer_mechanism: str  # ADEQUATE_UK, IDTA, BCR, DOMESTIC
    recommendations: List[str] = Field(default_factory=list)
    statutory_basis: str = "UK GDPR & UK Data Protection Act 2018 (c. 12)"


class UKGDPRPack:
    """Moduł weryfikujący zgodność z brytyjskim reżimem ochrony danych."""

    SPECIAL_CATEGORY_KEYWORDS = [
        "health_record", "medical_diagnosis", "genetic_data", "biometric_template",
        "political_affiliation", "religious_belief", "trade_union", "criminal_convictions"
    ]

    def evaluate_processing(self, processing_spec: Dict[str, Any]) -> UKGDPREvaluation:
        has_lawful_basis = bool(processing_spec.get("has_lawful_basis", True))
        contains_special = any(k in str(processing_spec) for k in self.SPECIAL_CATEGORY_KEYWORDS)
        special_cleared = (not contains_special) or bool(processing_spec.get("explicit_consent_art9", False))
        
        is_solely_automated = bool(processing_spec.get("is_solely_automated_decision", False))
        has_human_oversight = bool(processing_spec.get("has_human_intervention_right", True))
        automated_safeguards = (not is_solely_automated) or has_human_oversight

        transfer_dest = processing_spec.get("transfer_destination_country", "UK")
        if transfer_dest in ["UK", "EU", "EEA"]:
            transfer_mech = "ADEQUATE_UK"
        else:
            transfer_mech = processing_spec.get("transfer_mechanism", "IDTA_REQUIRED")

        recommendations = []
        if not has_lawful_basis:
            recommendations.append("Brak udokumentowanej podstawy prawnej (Art. 6 UK GDPR).")
        if not special_cleared:
            recommendations.append("Naruszenie Art. 9 UK GDPR: brak wyraźnej zgody na przetwarzanie danych szczególnej kategorii.")
        if not automated_safeguards:
            recommendations.append("Naruszenie Art. 22 UK GDPR: brak mechanizmu interwencji ludzkiej w zautomatyzowanej decyzji.")
        if transfer_mech == "IDTA_REQUIRED":
            recommendations.append("Wymagane wdrożenie International Data Transfer Agreement (IDTA) pod DPA 2018.")

        is_compliant = has_lawful_basis and special_cleared and automated_safeguards and (transfer_mech != "IDTA_REQUIRED")
        
        return UKGDPREvaluation(
            is_compliant=is_compliant,
            lawful_basis_documented=has_lawful_basis,
            special_category_data_cleared=special_cleared,
            automated_decision_safeguards_active=automated_safeguards,
            data_minimisation_score=0.98 if is_compliant else 0.60,
            international_transfer_mechanism=transfer_mech,
            recommendations=recommendations or ["Przetwarzanie w pełni zgodne z UK GDPR i DPA 2018."],
        )


# ==============================================================================
# 3. NETWORK AND INFORMATION SYSTEMS (NIS) REGULATIONS 2018 (UK)
# ==============================================================================

class UKNISIncidentNotification(BaseModel):
    """Zgłoszenie incydentu na mocy UK NIS Regulations 2018."""

    notification_id: str = Field(default_factory=lambda: f"nis_uk_{uuid.uuid4().hex[:10]}")
    entity_type: str = Field(..., description="OES (Operator of Essential Service) lub RDSP (Digital Service Provider)")
    competent_authority: str = Field(..., description="ICO, Ofgem, DfT, DHSC")
    severity: str = Field(..., description="SIGNIFICANT_IMPACT, MAJOR, SYSTEMIC")
    affected_users_estimate: int
    duration_hours: float
    statutory_deadline_hours: int = 72
    notification_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    statutory_citation: str = "Network and Information Systems Regulations 2018 (SI 2018/506) Regulation 11/12"


class UKNISPack:
    """Pakiet oceny odporności i obowiązków sprawozdawczych pod UK NIS Regulations 2018."""

    OES_SECTORS = ["energy", "transport", "health", "water", "digital_infrastructure"]
    RDSP_TYPES = ["cloud_computing_service", "online_marketplace", "search_engine"]

    def evaluate_entity_posture(self, entity_meta: Dict[str, Any]) -> Dict[str, Any]:
        sector = entity_meta.get("sector", "general")
        is_oes = sector in self.OES_SECTORS
        is_rdsp = sector in self.RDSP_TYPES

        has_incident_plan = bool(entity_meta.get("has_incident_response_plan", True))
        has_72h_sla = bool(entity_meta.get("has_statutory_72h_reporting_sla", True))
        has_business_continuity = bool(entity_meta.get("has_bcp_and_dr", True))

        in_scope = is_oes or is_rdsp
        compliant = (not in_scope) or (has_incident_plan and has_72h_sla and has_business_continuity)

        return {
            "regime": "UK NIS Regulations 2018",
            "in_scope": in_scope,
            "entity_classification": "OES" if is_oes else ("RDSP" if is_rdsp else "NON_NIS_ENTITY"),
            "is_compliant": compliant,
            "has_72h_reporting_sla": has_72h_sla,
            "security_measures_verified": has_incident_plan and has_business_continuity,
            "competent_authority": "ICO" if is_rdsp else f"Designated Sector Authority for {sector}",
        }

    def generate_incident_notification(
        self,
        entity_type: str,
        sector: str,
        affected_users: int,
        duration_hours: float,
    ) -> UKNISIncidentNotification:
        authority = "ICO" if entity_type == "RDSP" else f"Sector Authority for {sector}"
        severity = "MAJOR" if affected_users > 50000 else "SIGNIFICANT_IMPACT"
        return UKNISIncidentNotification(
            entity_type=entity_type,
            competent_authority=authority,
            severity=severity,
            affected_users_estimate=affected_users,
            duration_hours=duration_hours,
        )
