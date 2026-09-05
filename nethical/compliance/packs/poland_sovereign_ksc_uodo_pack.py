"""Pakiet Zgodności: Polish Sovereign Cyber, Jurisdiction & Data Governance Pack.

Obejmuje 5 kluczowych polskich ram prawnych i regulacyjnych:
1. Krajowy System Cyberbezpieczeństwa (KSC - Dz.U. 2018 poz. 1560 z późn. zm. / NIS2):
   - Operatorzy Usług Kluczowych (OUK) oraz Dostawcy Usług Cyfrowych (DUC)
   - Obowiązkowe zgłaszanie incydentów w czasie <24h do CSIRT NASK / CSIRT GOV / CSIRT MON
   - Obowiązkowy audyt bezpieczeństwa systemu teleinformatycznego co 2 lata
2. Rozszerzona Jurysdykcja Polska & Kodeks Karny (Rozdział XXXIII k.k.):
   - Art. 267 k.k.: Bezprawne uzyskanie informacji (hacking / ominięcie zabezpieczeń)
   - Art. 268 & 268a k.k.: Niszczenie danych i udaremnianie dostępu do danych
   - Art. 269 k.k.: Sabotaż informatyczny infrastruktury o istotnym znaczeniu
   - Art. 269a k.k.: Zakłócenie pracy systemu komputerowego (DDoS / zapychanie zasobów)
   - Art. 269b k.k.: Wytwarzanie i dystrybucja exploitów, malware i narzędzi hakerskich
   - Eksterytorializm prawa karnego (Art. 110-112 k.k.) chroniący interesy i obywateli RP
3. Odpowiedzialność Zarządu i Kadry Kierowniczej (Executive Liability in Poland):
   - Art. 293 / Art. 483 k.s.h.: Należyta staranność zawodowa i ochrona Business Judgment Rule
   - Art. 296 k.k.: Odpowiedzialność karna za niegospodarność przez zaniechanie cyberbezpieczeństwa
   - Kary osobiste KSC nakładane bezpośrednio na kierowników podmiotów (do 100 000 PLN)
4. Krajowy System Certyfikacji Cyberbezpieczeństwa:
   - Poziomy zaufania certyfikatów: Podstawowy (Basic), Znaczny (Substantial), Wysoki (High)
   - Certyfikacja komponentów dla polskiego sektora publicznego i infrastruktury krytycznej
5. Urząd Ochrony Danych Osobowych (UODO):
   - Ustawa z 10 maja 2018 r. o ochronie danych osobowych (Dz.U. 2018 poz. 1000)
   - Rejestracja IOD, 72-godzinny formularz zgłoszeniowy naruszenia do Prezesa UODO
   - Prowadzenie Rejestru Naruszeń oraz ochrona przed karami administracyjnymi (do 20M EUR / 4%)
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.compliance.packs.poland_sovereign_ksc_uodo")


# ==============================================================================
# 1. KRAJOWY SYSTEM CYBERBEZPIECZEŃSTWA (KSC)
# ==============================================================================

class KSCIncidentNotification(BaseModel):
    """Zgłoszenie incydentu do CSIRT poziomu krajowego na mocy ustawy o KSC."""

    incident_id: str = Field(default_factory=lambda: f"ksc_inc_{uuid.uuid4().hex[:10]}")
    target_csirt: str = Field(..., description="CSIRT NASK, CSIRT GOV (ABW) lub CSIRT MON")
    incident_classification: str = Field(..., description="KRYTYCZNY, POWAZNY, ZWYKLY")
    affected_sector: str
    detection_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    statutory_deadline_hours: int = 24  # Maksymalnie 24h na zgłoszenie incydentu poważnego
    legal_basis: str = "Ustawa z dnia 5 lipca 2018 r. o krajowym systemie cyberbezpieczeństwa (Dz.U. 2018 poz. 1560) Art. 11/12"
    technical_description: str


class KSCAuditReadiness(BaseModel):
    """Ocena gotowości podmiotu do obowiązkowego 2-letniego audytu KSC."""

    is_compliant: bool
    entity_type: str = Field(..., description="OUK (Operator Usługi Kluczowej), DUC (Dostawca Usługi Cyfrowej), PODMIOT_KLUCZOWY")
    last_audit_date: Optional[str] = None
    audit_overdue: bool = False
    security_documentation_complete: bool = True
    csirt_communication_channel_verified: bool = True
    readiness_score: float = 0.98
    statutory_citation: str = "Ustawa o KSC Art. 15 (Audyt bezpieczeństwa systemu)"


class PolishKSCPack:
    """Pakiet weryfikacji i obsługi obowiązków Krajowego Systemu Cyberbezpieczeństwa."""

    def classify_and_dispatch_incident(
        self,
        sector: str,
        is_public_admin: bool,
        is_military_defense: bool,
        impact_critical: bool,
        description: str,
    ) -> KSCIncidentNotification:
        if is_military_defense:
            csirt = "CSIRT MON"
        elif is_public_admin:
            csirt = "CSIRT GOV (Agencja Bezpieczeństwa Wewnętrznego)"
        else:
            csirt = "CSIRT NASK (Naukowa i Akademicka Sieć Komputerowa - PIB)"

        classification = "KRYTYCZNY" if impact_critical else "POWAZNY"

        return KSCIncidentNotification(
            target_csirt=csirt,
            incident_classification=classification,
            affected_sector=sector,
            technical_description=description,
        )

    def evaluate_audit_posture(
        self,
        entity_type: str,
        days_since_last_audit: int,
    ) -> KSCAuditReadiness:
        # KSC wymaga audytu co najmniej raz na 2 lata (730 dni)
        overdue = days_since_last_audit > 730
        is_comp = not overdue
        return KSCAuditReadiness(
            is_compliant=is_comp,
            entity_type=entity_type,
            audit_overdue=overdue,
            readiness_score=0.98 if is_comp else 0.40,
        )


# ==============================================================================
# 2. ROZSZERZONA POLSKA JURYSDYKCJA & KODEKS KARNY (ART. 267 - 269b k.k.)
# ==============================================================================

class PolishPenalCodeEvaluation(BaseModel):
    """Ocena operacji pod kątem przestępstw przeciwko ochronie informacji (Rozdział XXXIII k.k.)."""

    is_lawful: bool
    offences_flagged: List[str] = Field(default_factory=list)
    statutory_articles: List[str] = Field(default_factory=list)
    penal_risk_level: str = Field(..., description="BRAK, WYSOKI, KRYTYCZNY_ZBRODNIA")
    interlock_decision: str = Field(..., description="ALLOW, BLOCK, TERMINATE")
    prosecution_jurisdiction: str = "Prokuratura Rzeczypospolitej Polskiej (Jurysdykcja Eksterytorialna Art. 110-112 k.k.)"


class PolishPenalCodeEvaluator:
    """Detektor i interlock dla czynów zabronionych z Kodeksu Karnego RP."""

    PATTERNS_267 = [
        r"(?i)\b(przełamanie\s+haseł|podsłuch\s+sieciowy|nieuprawnione\s+uzyskanie\s+dostępu|obchodzenie\s+zabezpieczeń)\b",
        r"(?i)\b(sniffing|keylogger|dump\s+pamięci|przechwycenie\s+komunikacji)\b",
    ]
    PATTERNS_268 = [
        r"(?i)\b(usunięcie\s+bazy|kasowanie\s+logów|niszczenie\s+danych|udaremnienie\s+dostępu)\b",
        r"(?i)\b(delete\s+audit|truncate\s+table|wipe\s+drive)\b",
    ]
    PATTERNS_269 = [
        r"(?i)\b(sabotaż|infrastruktura\s+krytyczna|paraliż\s+energetyki|zakłócenie\s+systemu\s+obronnego)\b",
        r"(?i)\b(scada\s+attack|stuxnet|blackout\s+trigger)\b",
    ]
    PATTERNS_269A = [
        r"(?i)\b(zakłócenie\s+pracy\s+systemu|ddos\s+atak|zapychanie\s+łącza|flood\s+serwera)\b",
    ]
    PATTERNS_269B = [
        r"(?i)\b(wirus|trojan|ransomware|tworzenie\s+exploita|kod\s+złośliwy|przystosowanie\s+do\s+przestępstwa)\b",
    ]

    @classmethod
    def evaluate_intent_and_payload(cls, payload: Dict[str, Any]) -> PolishPenalCodeEvaluation:
        text = str(payload.get("input", "")) + " " + str(payload.get("command", "")) + " " + str(payload.get("tool_name", ""))
        offences = []
        articles = []

        if any(re.search(p, text) for p in cls.PATTERNS_267):
            offences.append("Bezprawne uzyskanie informacji / hacking")
            articles.append("Art. 267 § 1-3 k.k. (kara pozbawienia wolności do lat 2)")

        if any(re.search(p, text) for p in cls.PATTERNS_268):
            offences.append("Niszczenie lub uszkadzanie danych informatycznych")
            articles.append("Art. 268 / Art. 268a k.k. (kara pozbawienia wolności do lat 5)")

        if any(re.search(p, text) for p in cls.PATTERNS_269):
            offences.append("Sabotaż informatyczny infrastruktury o istotnym znaczeniu")
            articles.append("Art. 269 k.k. (Zbrodnia/Występek: kara pozbawienia wolności od 6 miesięcy do lat 8)")

        if any(re.search(p, text) for p in cls.PATTERNS_269A):
            offences.append("Zakłócenie pracy systemu komputerowego")
            articles.append("Art. 269a k.k. (kara pozbawienia wolności od 3 miesięcy do lat 5)")

        if any(re.search(p, text) for p in cls.PATTERNS_269B):
            offences.append("Wytwarzanie lub dystrybucja narzędzi i programów hakerskich")
            articles.append("Art. 269b k.k. (kara pozbawienia wolności od 3 miesięcy do lat 5)")

        is_lawful = len(offences) == 0
        if not is_lawful:
            if "Art. 269 k.k." in str(articles) or "Art. 269b k.k." in str(articles):
                risk = "KRYTYCZNY_ZBRODNIA"
                decision = "TERMINATE"
            else:
                risk = "WYSOKI"
                decision = "BLOCK"
        else:
            risk = "BRAK"
            decision = "ALLOW"

        return PolishPenalCodeEvaluation(
            is_lawful=is_lawful,
            offences_flagged=offences,
            statutory_articles=articles,
            penal_risk_level=risk,
            interlock_decision=decision,
        )


# ==============================================================================
# 3. ODPOWIEDZIALNOŚĆ ZARZĄDU I KADRY KIEROWNICZEJ (EXECUTIVE LIABILITY)
# ==============================================================================

class ExecutiveLiabilityShieldReport(BaseModel):
    """Certyfikat ochrony i dochowania należytej staranności przez Zarząd (Business Judgment Rule)."""

    shield_id: str = Field(default_factory=lambda: f"exec_shield_{uuid.uuid4().hex[:10]}")
    board_due_diligence_verified: bool
    business_judgment_rule_applicable: bool
    personal_fine_exposure_ksc_pln: int = 0  # Maksymalnie 100 000 PLN z KSC
    penal_liability_risk_art296_kk: str = "WYELIMINOWANE"  # Brak niegospodarności
    cryptographic_audit_trail_proven: bool = True
    corporate_code_articles: List[str] = Field(default_factory=lambda: [
        "Art. 293 § 3 k.s.h. (Zasada Biznesowej Oceny Sytuacji - Business Judgment Rule dla Sp. z o.o.)",
        "Art. 483 § 3 k.s.h. (Zasada Biznesowej Oceny Sytuacji dla Spółki Akcyjnej)",
        "Art. 296 k.k. (Wyłączenie winy z tytułu należytego nadzoru nad majątkiem spółki)",
        "Ustawa o KSC (Zwolnienie z kar osobistych dzięki certyfikowanemu systemowi governance)",
    ])
    issued_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PolishExecutiveLiabilityPack:
    """Moduł audytu ochrony zarządu i kadry zarządzającej przed osobistą odpowiedzialnością cywilną i karną."""

    def evaluate_board_due_diligence(
        self,
        has_merkle_ledger_active: bool,
        has_formal_risk_policy: bool,
        has_hitl_escalation_active: bool,
        has_periodic_audits: bool,
    ) -> ExecutiveLiabilityShieldReport:
        due_diligence = has_merkle_ledger_active and has_formal_risk_policy and has_hitl_escalation_active and has_periodic_audits

        fine_exposure = 0 if due_diligence else 100000
        penal_risk = "WYELIMINOWANE" if due_diligence else "PODWYŻSZONE_RYZYKO_ZANIECHANIA"

        return ExecutiveLiabilityShieldReport(
            board_due_diligence_verified=due_diligence,
            business_judgment_rule_applicable=due_diligence,
            personal_fine_exposure_ksc_pln=fine_exposure,
            penal_liability_risk_art296_kk=penal_risk,
            cryptographic_audit_trail_proven=has_merkle_ledger_active,
        )


# ==============================================================================
# 4. KRAJOWY SYSTEM CERTYFIKACJI CYBERBEZPIECZEŃSTWA
# ==============================================================================

class PolishCyberCertificationEvaluation(BaseModel):
    """Ocena poziomu zaufania w Krajowym Systemie Certyfikacji Cyberbezpieczeństwa."""

    confidence_level: str = Field(..., description="PODSTAWOWY (Basic), ZNACZNY (Substantial), WYSOKI (High)")
    is_certified_for_critical_infrastructure: bool
    eu_cybersecurity_act_aligned: bool = True
    quantum_resistant_cryptography_verified: bool
    accredited_body_pca_ready: bool = True
    score: float = 1.0


class PolishCyberCertificationPack:
    """Weryfikator zgodności z Krajowym Systemem Certyfikacji Cyberbezpieczeństwa."""

    def evaluate_component_confidence(
        self,
        has_pqc_signatures: bool,
        has_formal_smt_proofs: bool,
        has_tee_enclave_attestation: bool,
    ) -> PolishCyberCertificationEvaluation:
        if has_pqc_signatures and has_formal_smt_proofs and has_tee_enclave_attestation:
            level = "WYSOKI (High)"
            infra_allowed = True
            score = 1.0
        elif has_pqc_signatures:
            level = "ZNACZNY (Substantial)"
            infra_allowed = True
            score = 0.85
        else:
            level = "PODSTAWOWY (Basic)"
            infra_allowed = False
            score = 0.65

        return PolishCyberCertificationEvaluation(
            confidence_level=level,
            is_certified_for_critical_infrastructure=infra_allowed,
            quantum_resistant_cryptography_verified=has_pqc_signatures,
            score=score,
        )


# ==============================================================================
# 5. URZĄD OCHRONY DANYCH OSOBOWYCH (UODO)
# ==============================================================================

class UODOBreachNotice(BaseModel):
    """Oficjalne zgłoszenie naruszenia ochrony danych osobowych do Prezesa UODO."""

    notice_id: str = Field(default_factory=lambda: f"uodo_{uuid.uuid4().hex[:10]}")
    controller_name: str
    dpo_full_name: str
    dpo_email: str
    breach_datetime: str
    detection_datetime: str
    statutory_deadline_hours: int = 72
    affected_persons_estimated: int
    data_scope_pesel_included: bool
    risk_to_rights_level: str = Field(..., description="NISKIE, SREDNIE, WYSOKIE")
    remedial_actions_summary: str
    formal_statutory_basis: str = "Art. 33 Rozporządzenia RODO oraz Ustawa z dnia 10 maja 2018 r. o ochronie danych osobowych (Prezesa UODO)"


class PolishUODOPack:
    """Pakiet procedur nadzorczych i zgłoszeniowych przed Prezesem UODO."""

    def draft_uodo_notification(
        self,
        controller: str,
        dpo_name: str,
        dpo_email: str,
        affected_count: int,
        includes_pesel: bool,
        remedial_actions: str,
    ) -> UODOBreachNotice:
        now = datetime.now(timezone.utc).isoformat()
        risk = "WYSOKIE" if (includes_pesel or affected_count > 1000) else "SREDNIE"
        return UODOBreachNotice(
            controller_name=controller,
            dpo_full_name=dpo_name,
            dpo_email=dpo_email,
            breach_datetime=now,
            detection_datetime=now,
            affected_persons_estimated=affected_count,
            data_scope_pesel_included=includes_pesel,
            risk_to_rights_level=risk,
            remedial_actions_summary=remedial_actions,
        )
