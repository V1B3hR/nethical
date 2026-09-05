"""Pakiet Zgodności: EU Resilience, Cybersecurity & Data Protection Pack.

Obejmuje 3 kluczowe unijne rozporządzenia:
1. Digital Operational Resilience Act (DORA - Regulation EU 2022/2554):
   - Zarządzanie ryzykiem ICT dla instytucji finansowych i dostawców chmurowych (CTPP)
   - Klasyfikacja i zgłaszanie incydentów ICT (Major Incident: <4h wstępne / <24h formalne)
   - Testy operacyjnej odporności cyfrowej i Threat-Led Penetration Testing (TLPT)
   - Zarządzanie ryzykiem stron trzecich i strategie wyjścia
2. Cyber Resilience Act (CRA - Regulation EU 2024/2847):
   - Wymogi bezpieczeństwa produktów z elementami cyfrowymi (Hardware & Software AI)
   - Obowiązkowe prowadzenie Software Bill of Materials (SBOM - CycloneDX / SPDX)
   - Zgłaszanie aktywnie wykorzystywanych podatności do CSIRT i ENISA w czasie <24h
   - 5-letni cykl wsparcia bezpieczeństwa i automatyczne aktualizacje
3. EU GDPR (Regulation EU 2016/679):
   - Podstawy prawne (Art. 6) oraz kategorie szczególne (Art. 9)
   - Ocena Skutków dla Ochrony Danych (DPIA - Art. 35) dla zautomatyzowanych systemów AI
   - Zgłaszanie naruszeń w terminie 72 godzin do organu nadzorczego (Art. 33)
   - Gwarancje przeciwko wyłącznie zautomatyzowanemu podejmowaniu decyzji (Art. 22)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.compliance.packs.eu_resilience_cra_dora")


# ==============================================================================
# 1. DIGITAL OPERATIONAL RESILIENCE ACT (DORA - EU 2022/2554)
# ==============================================================================

class DORAIncidentReport(BaseModel):
    """Oficjalny raport incydentu ICT zgodnie z DORA Artykuł 19."""

    incident_id: str = Field(default_factory=lambda: f"dora_inc_{uuid.uuid4().hex[:10]}")
    classification: str = Field(..., description="MAJOR_ICT_INCIDENT, SIGNIFICANT_INCIDENT")
    impacted_services: List[str] = Field(default_factory=list)
    initial_notification_deadline_hours: int = 4
    full_report_deadline_hours: int = 24
    reported_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    competent_supervisory_authority: str = Field(..., description="EBA, ESMA, EIOPA lub krajowy KNF")
    statutory_citation: str = "Regulation (EU) 2022/2554 (DORA) Articles 18-20"


class DORAEvaluation(BaseModel):
    """Wynik ewaluacji odporności cyfrowej zgodnie z DORA."""

    is_compliant: bool
    ict_risk_framework_active: bool
    tlpt_testing_up_to_date: bool
    third_party_concentration_risk_acceptable: bool
    major_incident_management_verified: bool
    readiness_score: float  # 0.0 - 1.0
    gap_findings: List[str] = Field(default_factory=list)
    statutory_basis: str = "Regulation (EU) 2022/2554 of the European Parliament and of the Council (DORA)"


class DORAPack:
    """Moduł weryfikujący wymogi operacyjnej odporności cyfrowej DORA."""

    def evaluate_financial_entity(self, entity_posture: Dict[str, Any]) -> DORAEvaluation:
        ict_framework = bool(entity_posture.get("has_ict_risk_framework", True))
        tlpt_tested = bool(entity_posture.get("has_threat_led_penetration_test", True))
        third_party_safe = bool(entity_posture.get("has_exit_strategy_for_cloud", True))
        incident_proc = bool(entity_posture.get("has_dora_incident_procedures", True))

        gaps = []
        if not ict_framework:
            gaps.append("Brak formalnych ram zarządzania ryzykiem ICT (DORA Art. 6).")
        if not tlpt_tested:
            gaps.append("Brak aktualnego testu penetracyjnego opartego na zagrożeniach TLPT (DORA Art. 26).")
        if not third_party_safe:
            gaps.append("Brak strategii wyjścia i monitoringu koncentracji dostawców ICT (DORA Art. 28).")
        if not incident_proc:
            gaps.append("Brak procedury raportowania incydentów w reżimie 4h/24h (DORA Art. 19).")

        score = (int(ict_framework) + int(tlpt_tested) + int(third_party_safe) + int(incident_proc)) / 4.0
        is_compliant = len(gaps) == 0

        return DORAEvaluation(
            is_compliant=is_compliant,
            ict_risk_framework_active=ict_framework,
            tlpt_testing_up_to_date=tlpt_tested,
            third_party_concentration_risk_acceptable=third_party_safe,
            major_incident_management_verified=incident_proc,
            readiness_score=round(score, 2),
            gap_findings=gaps,
        )

    def create_incident_report(
        self,
        impacted_services: List[str],
        authority: str = "KNF (Komisja Nadzoru Finansowego)",
        is_major: bool = True,
    ) -> DORAIncidentReport:
        return DORAIncidentReport(
            classification="MAJOR_ICT_INCIDENT" if is_major else "SIGNIFICANT_INCIDENT",
            impacted_services=impacted_services,
            competent_supervisory_authority=authority,
        )


# ==============================================================================
# 2. CYBER RESILIENCE ACT (CRA - EU 2024/2847)
# ==============================================================================

class CRAVulnerabilityNotification(BaseModel):
    """Zgłoszenie aktywnie wykorzystywanej podatności pod CRA Artykuł 11."""

    notification_id: str = Field(default_factory=lambda: f"cra_vuln_{uuid.uuid4().hex[:10]}")
    product_name: str
    product_version: str
    cve_id: Optional[str] = None
    vulnerability_description: str
    is_actively_exploited: bool = True
    statutory_deadline_hours: int = 24
    notified_authorities: List[str] = Field(default_factory=lambda: ["Designated CSIRT", "ENISA"])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    statutory_citation: str = "Regulation (EU) 2024/2847 (CRA) Article 11"


class CRAEvaluation(BaseModel):
    """Ocena zgodności produktu z elementami cyfrowymi z wymogami CRA."""

    is_compliant: bool
    secure_by_default: bool
    sbom_present: bool
    vulnerability_handling_procedure: bool
    automatic_security_updates_supported: bool
    support_period_years: int = 5
    readiness_score: float
    recommendations: List[str] = Field(default_factory=list)
    statutory_basis: str = "Regulation (EU) 2024/2847 of the European Parliament and of the Council (CRA)"


class CRAPack:
    """Moduł oceny zgodności Cyber Resilience Act (CRA)."""

    def evaluate_product_cyber_resilience(self, product_meta: Dict[str, Any]) -> CRAEvaluation:
        sec_default = bool(product_meta.get("is_secure_by_default", True))
        sbom_ok = bool(product_meta.get("has_sbom", True))
        vuln_proc = bool(product_meta.get("has_24h_vulnerability_reporting", True))
        auto_updates = bool(product_meta.get("supports_secure_updates", True))

        recs = []
        if not sec_default:
            recs.append("Produkt wymaga domyślnej bezpiecznej konfiguracji bez podatnych haseł i portów.")
        if not sbom_ok:
            recs.append("Brak Software Bill of Materials (SBOM) w standardzie CycloneDX/SPDX (CRA Załącznik I).")
        if not vuln_proc:
            recs.append("Wymagana procedura 24-godzinnego zgłaszania aktywnie wykorzystywanych podatności do CSIRT/ENISA.")
        if not auto_updates:
            recs.append("Brak bezpiecznego mechanizmu automatycznych aktualizacji krytycznych.")

        score = (int(sec_default) + int(sbom_ok) + int(vuln_proc) + int(auto_updates)) / 4.0
        return CRAEvaluation(
            is_compliant=(score >= 1.0),
            secure_by_default=sec_default,
            sbom_present=sbom_ok,
            vulnerability_handling_procedure=vuln_proc,
            automatic_security_updates_supported=auto_updates,
            readiness_score=round(score, 2),
            recommendations=recs or ["Produkt w pełni spełnia wymogi Cyber Resilience Act."],
        )

    def generate_vulnerability_notification(
        self,
        product_name: str,
        product_version: str,
        description: str,
        cve_id: Optional[str] = None,
    ) -> CRAVulnerabilityNotification:
        return CRAVulnerabilityNotification(
            product_name=product_name,
            product_version=product_version,
            vulnerability_description=description,
            cve_id=cve_id,
        )


# ==============================================================================
# 3. EU GDPR (REGULATION EU 2016/679)
# ==============================================================================

class GDPRBreachNotification(BaseModel):
    """Zgłoszenie naruszenia ochrony danych osobowych pod Art. 33 RODO/GDPR."""

    notification_id: str = Field(default_factory=lambda: f"gdpr_br_{uuid.uuid4().hex[:10]}")
    data_controller: str
    nature_of_breach: str
    categories_of_data: List[str]
    approximate_individuals_count: int
    likely_consequences: str
    measures_taken: str
    dpo_contact: str
    statutory_deadline_hours: int = 72
    reported_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    statutory_citation: str = "Regulation (EU) 2016/679 (GDPR) Article 33"


class EUGDPREvaluation(BaseModel):
    """Ocena zgodności przetwarzania z unijnym rozporządzeniem RODO/GDPR."""

    is_compliant: bool
    lawful_basis_article_6: bool
    special_categories_article_9_cleared: bool
    dpia_required: bool
    dpia_conducted: bool
    article_22_human_oversight_guaranteed: bool
    breach_notification_sla_72h: bool
    compliance_score: float
    violations_found: List[str] = Field(default_factory=list)


class EUGDPRPack:
    """Moduł weryfikujący zgodność z RODO / EU GDPR."""

    def evaluate_ai_processing(self, processing_ctx: Dict[str, Any]) -> EUGDPREvaluation:
        art6 = bool(processing_ctx.get("lawful_basis_established", True))
        contains_special = bool(processing_ctx.get("contains_biometric_or_health_data", False))
        art9_cleared = (not contains_special) or bool(processing_ctx.get("explicit_consent_obtained", False))

        is_high_risk_ai = bool(processing_ctx.get("is_automated_profiling", False))
        dpia_required = is_high_risk_ai or contains_special
        dpia_done = bool(processing_ctx.get("dpia_completed", False)) if dpia_required else True

        art22_hitl = bool(processing_ctx.get("human_oversight_available", True))
        sla_72h = bool(processing_ctx.get("has_72h_breach_sla", True))

        violations = []
        if not art6:
            violations.append("Naruszenie Art. 6 RODO: brak legalnej podstawy przetwarzania.")
        if not art9_cleared:
            violations.append("Naruszenie Art. 9 RODO: niedozwolone przetwarzanie danych wrażliwych/biometrycznych.")
        if dpia_required and not dpia_done:
            violations.append("Naruszenie Art. 35 RODO: brak obowiązkowej oceny skutków DPIA dla systemu AI.")
        if not art22_hitl:
            violations.append("Naruszenie Art. 22 RODO: brak prawa do interwencji ludzkiej i wyjaśnienia decyzji.")

        score = 1.0 - (len(violations) * 0.25)
        is_compliant = len(violations) == 0

        return EUGDPREvaluation(
            is_compliant=is_compliant,
            lawful_basis_article_6=art6,
            special_categories_article_9_cleared=art9_cleared,
            dpia_required=dpia_required,
            dpia_conducted=dpia_done,
            article_22_human_oversight_guaranteed=art22_hitl,
            breach_notification_sla_72h=sla_72h,
            compliance_score=max(0.0, round(score, 2)),
            violations_found=violations,
        )

    def draft_breach_notification(
        self,
        controller: str,
        nature: str,
        data_categories: List[str],
        count: int,
        consequences: str,
        measures: str,
        dpo_email: str = "dpo@nethical.org",
    ) -> GDPRBreachNotification:
        return GDPRBreachNotification(
            data_controller=controller,
            nature_of_breach=nature,
            categories_of_data=data_categories,
            approximate_individuals_count=count,
            likely_consequences=consequences,
            measures_taken=measures,
            dpo_contact=dpo_email,
        )
