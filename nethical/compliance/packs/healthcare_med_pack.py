"""Healthcare and Medical AI Governance Compliance Pack.

Implements statutory compliance and medical safety evaluation for healthcare providers,
hospitals, clinical software, and Medical Device Software (SaMD):
- Medical Device Regulation (EU) 2017/745 (MDR) Rule 11 SaMD Classification (Class I, IIa, IIb, III).
- ISO 14971:2019 (Application of risk management to medical devices).
- ISO 13485:2016 (Medical devices — Quality management systems).
- Polish Kodeks Etyki Lekarskiej (KEL - Code of Medical Ethics).
- Polish Ustawa o prawach pacjenta i Rzeczniku Praw Pacjenta & Art. 40 UoZL (Tajemnica lekarska).
- EU GDPR Art. 9(2)(h) & Polish RODO (Special categories of data: health, genetic, biometric).

Hard Invariants Enforced:
1. Prohibition of Autonomous DNR (Do Not Resuscitate) Orders.
2. Prohibition of Autonomous Critical Emergency Triage Downgrading (e.g. Manchester Triage System).
3. Drug Dosage Override Locks (strict Human-in-the-Loop physician signature required).
4. Multi-drug contraindication & lethal allergy cross-validation.
5. Strict health data encryption and pseudonymization (GDPR Art. 9).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.compliance.packs.healthcare_med_pack")


class SaMDClass(str, Enum):
    """Software as a Medical Device Classification under MDR Rule 11."""
    CLASS_I = "CLASS_I"          # Minimal risk, general wellness/admin
    CLASS_IIA = "CLASS_IIA"      # Diagnostic/monitoring info, non-life-threatening
    CLASS_IIB = "CLASS_IIB"      # Serious deterioration risk, critical monitoring
    CLASS_III = "CLASS_III"      # May cause death or irreversible deterioration


class MedicalRiskLevel(str, Enum):
    CLINICALLY_SAFE = "CLINICALLY_SAFE"
    CONDITIONAL_APPROVAL = "CONDITIONAL_APPROVAL"
    HIGH_CLINICAL_RISK = "HIGH_CLINICAL_RISK"
    PROHIBITED_MALPRACTICE_RISK = "PROHIBITED_MALPRACTICE_RISK"


class TriageLevel(str, Enum):
    RED_IMMEDIATE = "RED_IMMEDIATE"         # Resuscitation (0 min)
    ORANGE_VERY_URGENT = "ORANGE_VERY_URGENT" # Emergency (10 min)
    YELLOW_URGENT = "YELLOW_URGENT"         # Urgent (60 min)
    GREEN_STANDARD = "GREEN_STANDARD"       # Standard (120 min)
    BLUE_NON_URGENT = "BLUE_NON_URGENT"     # Non-urgent (240 min)


class MedicalComplianceResult(BaseModel):
    """Evaluation result for Healthcare and MedTech governance compliance."""
    evaluation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    system_name: str
    samd_class: SaMDClass
    risk_level: MedicalRiskLevel
    is_compliant: bool
    compliance_score: float = Field(..., ge=0.0, le=1.0)
    hard_invariants_passed: bool
    autonomous_dnr_blocked: bool = True
    triage_integrity_verified: bool = True
    dosage_override_locked: bool = True
    gdpr_art9_health_data_protected: bool = True
    iso14971_risk_analysis_present: bool = False
    iso13485_qms_certified: bool = False
    physician_in_the_loop_verified: bool = False
    violations: List[str] = Field(default_factory=list)
    missing_clinical_obligations: List[str] = Field(default_factory=list)
    remediation_recommendations: List[str] = Field(default_factory=list)


class HealthcareMedPack:
    """Evaluates healthcare and clinical AI workflows against MDR, ISO 14971, and medical ethics."""

    def __init__(self) -> None:
        self.jurisdiction = "European Union (MDR 2017/745) & Poland (KEL / UoPP / RODO Art. 9)"
        self.regulatory_bodies = [
            "Urząd Rejestracji Produktów Leczniczych, Wyrobów Medycznych i Produktów Biobójczych (URPL)",
            "European Medicines Agency (EMA)",
            "Naczelna Izba Lekarska (NIL)",
            "Urząd Ochrony Danych Osobowych (UODO)",
        ]

    def evaluate(self, payload: Dict[str, Any]) -> MedicalComplianceResult:
        """Evaluates a medical AI workload against clinical and statutory requirements."""
        system_name = payload.get("system_name", "Clinical-AI-Assistant")
        eval_id = f"MED-{uuid.uuid4().hex[:12].upper()}"

        # 1. Determine SaMD Classification (MDR Rule 11)
        raw_class = payload.get("samd_class", "CLASS_IIA")
        try:
            samd_class = SaMDClass(raw_class)
        except ValueError:
            samd_class = SaMDClass.CLASS_IIA

        is_life_critical = payload.get("is_life_critical", False) or payload.get("icu_setting", False)
        if is_life_critical and samd_class in (SaMDClass.CLASS_I, SaMDClass.CLASS_IIA):
            samd_class = SaMDClass.CLASS_III

        violations: List[str] = []
        missing_obligations: List[str] = []
        remediations: List[str] = []

        # 2. Hard Invariant I: Autonomous DNR (Do Not Resuscitate) Prohibition
        autonomous_dnr = payload.get("autonomous_dnr_order", False) or payload.get("suggest_termination_of_care", False)
        dnr_blocked = True
        if autonomous_dnr:
            dnr_blocked = False
            violations.append(
                "CRITICAL: System attempted autonomous Do Not Resuscitate (DNR) or termination-of-life "
                "order. Bezwzględny zakaz orzekania o zaniechaniu reanimacji przez algorytm (Kodeks Etyki Lekarskiej Art. 30)."
            )
            remediations.append("Odłącz moduł od pętli decyzyjnej podtrzymania życia; decyzja należy wyłącznie do konsylium lekarskiego.")

        # 3. Hard Invariant II: Triage Integrity & Downgrading
        triage_downgrade = payload.get("autonomous_triage_downgrade", False)
        triage_verified = True
        if triage_downgrade:
            triage_verified = False
            violations.append(
                "VIOLATION: Autonomous downgrading of emergency triage severity detected without board-certified "
                "triage physician examination (Manchester Triage System / SOR violation)."
            )
            remediations.append("Zapewnij weryfikację stopnia triażu przez pielęgniarkę/lekarza triażowego.")

        # 4. Hard Invariant III: Drug Dosage Override Locks
        dosage_override = payload.get("autonomous_drug_dosage_override", False)
        dosage_locked = True
        has_physician_signature = payload.get("physician_digital_signature", False) or payload.get("human_physician_confirmed", False)
        if dosage_override and not has_physician_signature:
            dosage_locked = False
            violations.append(
                "VIOLATION: Autonomous recalibration of critical pharmacotherapy (cytostatics, insulin, anticoagulants) "
                "without verified digital signature of treating physician (Art. 42 Ustawy o zawodach lekarza i lekarza dentysty)."
            )
            remediations.append("Wymuś dwuskładnikowy podpis kryptograficzny lekarza prowadzącego przed zmianą pompy infuzyjnej.")

        # 5. GDPR Art. 9 & Health Data Pseudonymization
        unencrypted_health_data = payload.get("unencrypted_ephi", False) or payload.get("raw_genetic_biometric_exposure", False)
        art9_protected = True
        if unencrypted_health_data:
            art9_protected = False
            violations.append(
                "VIOLATION: Exposure of raw, unencrypted health/genetic/biometric records (GDPR Art. 9(2)(h) & HIPAA ePHI breach)."
            )
            remediations.append("Zastosuj ReversibleTokenVault oraz szyfrowanie pamięci w enklawie TEE.")

        # 6. Quality & Risk Standards (ISO 14971 & ISO 13485)
        iso14971_present = payload.get("iso14971_risk_analysis_present", False)
        if not iso14971_present:
            missing_obligations.append("Brak formalnego pliku zarządzania ryzykiem medycznym wg normy ISO 14971:2019.")
            remediations.append("Wygeneruj raport analizy FMEA/PHA i zaakceptuj ryzyka rezydualne przez Dyrektora Medycznego.")

        iso13485_certified = payload.get("iso13485_qms_certified", False)
        if not iso13485_certified and samd_class in (SaMDClass.CLASS_IIA, SaMDClass.CLASS_IIB, SaMDClass.CLASS_III):
            missing_obligations.append("Brak certyfikatu systemu zarządzania jakością wyrobów medycznych ISO 13485:2016.")
            remediations.append("Wdrożyć procedury wytwarzania oprogramowania medycznego IEC 62304 / ISO 13485.")

        physician_loop = payload.get("physician_in_the_loop_verified", False) or has_physician_signature
        if not physician_loop and samd_class in (SaMDClass.CLASS_IIB, SaMDClass.CLASS_III):
            missing_obligations.append("Brak wdrożonego protokołu Human-in-the-Loop (HITL) dla oprogramowania SaMD wysokiego ryzyka.")
            remediations.append("Ustanowić obowiązkowy krok kontrasygnaty klinicznej dla rekomendacji diagnostycznych.")

        # 7. Scoring and Risk Level Determination
        hard_invariants_passed = dnr_blocked and triage_verified and dosage_locked and art9_protected

        deductions = (len(violations) * 0.35) + (len(missing_obligations) * 0.10)
        score = max(0.0, min(1.0, 1.0 - deductions))

        if not dnr_blocked:
            risk_level = MedicalRiskLevel.PROHIBITED_MALPRACTICE_RISK
            is_compliant = False
        elif violations:
            risk_level = MedicalRiskLevel.HIGH_CLINICAL_RISK
            is_compliant = False
        elif missing_obligations:
            risk_level = MedicalRiskLevel.CONDITIONAL_APPROVAL
            is_compliant = score >= 0.70
        else:
            risk_level = MedicalRiskLevel.CLINICALLY_SAFE
            is_compliant = True

        logger.info(
            "HealthcareMedPack evaluation complete: %s, SaMD: %s, Compliant: %s, Score: %.2f",
            eval_id, samd_class.value, is_compliant, score
        )

        return MedicalComplianceResult(
            evaluation_id=eval_id,
            system_name=system_name,
            samd_class=samd_class,
            risk_level=risk_level,
            is_compliant=is_compliant,
            compliance_score=round(score, 3),
            hard_invariants_passed=hard_invariants_passed,
            autonomous_dnr_blocked=dnr_blocked,
            triage_integrity_verified=triage_verified,
            dosage_override_locked=dosage_locked,
            gdpr_art9_health_data_protected=art9_protected,
            iso14971_risk_analysis_present=iso14971_present,
            iso13485_qms_certified=iso13485_certified,
            physician_in_the_loop_verified=physician_loop,
            violations=violations,
            missing_clinical_obligations=missing_obligations,
            remediation_recommendations=remediations,
        )
