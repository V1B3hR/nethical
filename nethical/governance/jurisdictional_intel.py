# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Global Institutional & Jurisdictional Intelligence Engine (nethical.governance.jurisdictional_intel).

Integruje empiryczne wskaźniki ładu państwowego i jakości regulacji:
1. World Bank Worldwide Governance Indicators (WGI) & GovData360:
   - 6 wymiarów: Rule of Law, Regulatory Quality, Government Effectiveness, Control of Corruption,
     Voice & Accountability, Political Stability.
2. OECD Indicators of Regulatory Policy and Governance (iREG):
   - Metodologia Regulatory Impact Assessment (RIA), ocena proporcjonalności i konsultacji publicznych.
3. UK Government i.AI (Cabinet Office) Open Gov Architecture:
   - Zgodność z Crown Commercial Service (CCS) AI Framework, Contracts Finder i standardem ATRS.
4. University of Gothenburg Quality of Government (QoG) Institute:
   - Indeks bezstronności biurokratycznej i prewencji korupcji w zamówieniach publicznych.
5. GDPR / RODO Art. 44-49 & NATO Interoperability:
   - Zautomatyzowana ocena ryzyka transgranicznego transferu danych i modeli AI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.governance.jurisdictional_intel")


class GovernanceDimension(str, Enum):
    """6 fundamentalnych wymiarów ładu instytucjonalnego Banku Światowego (WGI)."""
    RULE_OF_LAW = "RULE_OF_LAW"                          # Praworządność i niezależność sądownictwa
    REGULATORY_QUALITY = "REGULATORY_QUALITY"            # Jakość i przewidywalność prawa
    GOVERNMENT_EFFECTIVENESS = "GOVERNMENT_EFFECTIVENESS"# Efektywność administracji i usług publicznych
    CONTROL_OF_CORRUPTION = "CONTROL_OF_CORRUPTION"      # Kontrola korupcji i przejrzystość
    VOICE_AND_ACCOUNTABILITY = "VOICE_AND_ACCOUNTABILITY"# Odpowiedzialność władzy i wolności obywatelskie
    POLITICAL_STABILITY = "POLITICAL_STABILITY"          # Stabilność polityczna i brak przemocy


class DataClassification(str, Enum):
    """Klasyfikacja wrażliwości danych zgodnie z doktrynami państwowymi."""
    PUBLIC_OPEN = "PUBLIC_OPEN"                          # Dane jawne (Open Data)
    OFFICIAL_INTERNAL = "OFFICIAL_INTERNAL"              # Służbowe / Poufne korporacyjne
    PERSONAL_GDPR = "PERSONAL_GDPR"                      # Dane osobowe chronione RODO (PII)
    SPECIAL_CATEGORY_HEALTH_BIOMETRIC = "SPECIAL_HEALTH" # Dane medyczne, biometryczne, genetyczne (Art. 9 RODO)
    CRITICAL_INFRASTRUCTURE_OT = "CRITICAL_OT"           # Telemetria sterowania przemysłowego (SCADA/ICS)
    STATE_SECURITY_DEFENSE = "DEFENSE_RESTRICTED"        # Bezpieczeństwo państwa / NATO RESTRICTED


class TransferVerdict(str, Enum):
    """Orzeczenie bramy jurysdykcyjnej dla transferu danych i wywołania modeli."""
    ALLOW = "ALLOW"                                      # Bezpieczny transfer (wspólny obszar prawny)
    ALLOW_WITH_TEE_ENCLAVE = "ALLOW_WITH_TEE_ENCLAVE"    # Dozwolony wyłącznie w szyfrowanej enklawie z tokenizacją PII
    BLOCK_INADEQUATE_RULE_OF_LAW = "BLOCK_RULE_OF_LAW"   # Blokada: brak gwarancji praworządności (Art. 45 RODO)
    BLOCK_CORRUPTION_OR_INSTABILITY = "BLOCK_CORRUPTION" # Blokada: wysokie ryzyko przejęcia lub niestabilności
    BLOCK_NATIONAL_SOVEREIGNTY = "BLOCK_SOVEREIGNTY"     # Blokada: wymóg suwerenności na terytorium RP / UK / NATO


class JurisdictionProfile(BaseModel):
    """Profil państwa oparty na empirycznych danych Banku Światowego, OECD i QoG."""
    country_code: str = Field(..., description="Kod ISO 3166-1 alpha-2")
    country_name: str
    is_eu_eea: bool = False
    is_nato_member: bool = False
    has_gdpr_adequacy: bool = False                      # Oficjalna decyzja Komisji Europejskiej o adekwatności (Art. 45)
    wgi_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Wskaźniki WGI w skali od -2.5 (bardzo słaby) do +2.5 (bardzo silny)",
    )
    qog_impartiality_score: float = Field(default=0.5, ge=0.0, le=1.0) # Quality of Government Institute (0-1)
    oecd_ireg_score: float = Field(default=2.0, ge=0.0, le=4.0)        # OECD Regulatory Quality (0-4)


class CrossBorderAuditResult(BaseModel):
    """Wynik formalnego audytu transferu transgranicznego danych."""
    verdict: TransferVerdict
    jurisdiction_trust_score: float = Field(..., description="Syntetyczny wskaźnik zaufania jurysdykcyjnego (0-100)")
    rule_of_law_score: float
    is_adequate_gdpr: bool
    reasons: List[str] = Field(default_factory=list)
    enforced_safeguards: List[str] = Field(default_factory=list)
    audited_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OECDRegulatoryImpactResult(BaseModel):
    """Wynik oceny skutków regulacji i przejrzystości algorytmicznej wg OECD iREG & UK DSIT."""
    is_compliant: bool
    ria_score: float = Field(..., ge=0.0, le=4.0, description="Wynik OECD Regulatory Impact Assessment")
    transparency_tier: str = "ATRS_TIER_2"
    proportionality_confirmed: bool
    recommendations: List[str] = Field(default_factory=list)


# =============================================================================
# EMPIRICAL GLOBAL JURISDICTION KNOWLEDGE REPOSITORY
# =============================================================================

DEFAULT_JURISDICTIONS: Dict[str, JurisdictionProfile] = {
    "PL": JurisdictionProfile(
        country_code="PL",
        country_name="Poland",
        is_eu_eea=True,
        is_nato_member=True,
        has_gdpr_adequacy=True,
        wgi_scores={
            GovernanceDimension.RULE_OF_LAW.value: 0.58,
            GovernanceDimension.REGULATORY_QUALITY.value: 0.85,
            GovernanceDimension.GOVERNMENT_EFFECTIVENESS.value: 0.62,
            GovernanceDimension.CONTROL_OF_CORRUPTION.value: 0.61,
            GovernanceDimension.VOICE_AND_ACCOUNTABILITY.value: 0.72,
            GovernanceDimension.POLITICAL_STABILITY.value: 0.45,
        },
        qog_impartiality_score=0.74,
        oecd_ireg_score=2.85,
    ),
    "GB": JurisdictionProfile(
        country_code="GB",
        country_name="United Kingdom",
        is_eu_eea=False,
        is_nato_member=True,
        has_gdpr_adequacy=True,  # UK Adequacy decision under EU GDPR
        wgi_scores={
            GovernanceDimension.RULE_OF_LAW.value: 1.48,
            GovernanceDimension.REGULATORY_QUALITY.value: 1.62,
            GovernanceDimension.GOVERNMENT_EFFECTIVENESS.value: 1.35,
            GovernanceDimension.CONTROL_OF_CORRUPTION.value: 1.72,
            GovernanceDimension.VOICE_AND_ACCOUNTABILITY.value: 1.25,
            GovernanceDimension.POLITICAL_STABILITY.value: 0.42,
        },
        qog_impartiality_score=0.91,
        oecd_ireg_score=3.45,
    ),
    "US": JurisdictionProfile(
        country_code="US",
        country_name="United States",
        is_eu_eea=False,
        is_nato_member=True,
        has_gdpr_adequacy=True,  # EU-US Data Privacy Framework
        wgi_scores={
            GovernanceDimension.RULE_OF_LAW.value: 1.32,
            GovernanceDimension.REGULATORY_QUALITY.value: 1.45,
            GovernanceDimension.GOVERNMENT_EFFECTIVENESS.value: 1.41,
            GovernanceDimension.CONTROL_OF_CORRUPTION.value: 1.28,
            GovernanceDimension.VOICE_AND_ACCOUNTABILITY.value: 1.05,
            GovernanceDimension.POLITICAL_STABILITY.value: 0.12,
        },
        qog_impartiality_score=0.86,
        oecd_ireg_score=3.10,
    ),
    "DE": JurisdictionProfile(
        country_code="DE",
        country_name="Germany",
        is_eu_eea=True,
        is_nato_member=True,
        has_gdpr_adequacy=True,
        wgi_scores={
            GovernanceDimension.RULE_OF_LAW.value: 1.61,
            GovernanceDimension.REGULATORY_QUALITY.value: 1.58,
            GovernanceDimension.GOVERNMENT_EFFECTIVENESS.value: 1.52,
            GovernanceDimension.CONTROL_OF_CORRUPTION.value: 1.85,
            GovernanceDimension.VOICE_AND_ACCOUNTABILITY.value: 1.38,
            GovernanceDimension.POLITICAL_STABILITY.value: 0.72,
        },
        qog_impartiality_score=0.93,
        oecd_ireg_score=3.20,
    ),
    "CH": JurisdictionProfile(
        country_code="CH",
        country_name="Switzerland",
        is_eu_eea=False,
        is_nato_member=False,
        has_gdpr_adequacy=True,
        wgi_scores={
            GovernanceDimension.RULE_OF_LAW.value: 1.88,
            GovernanceDimension.REGULATORY_QUALITY.value: 1.74,
            GovernanceDimension.GOVERNMENT_EFFECTIVENESS.value: 1.89,
            GovernanceDimension.CONTROL_OF_CORRUPTION.value: 2.01,
            GovernanceDimension.VOICE_AND_ACCOUNTABILITY.value: 1.51,
            GovernanceDimension.POLITICAL_STABILITY.value: 1.15,
        },
        qog_impartiality_score=0.96,
        oecd_ireg_score=3.35,
    ),
    # High-Risk / Authoritarian Benchmark
    "RU": JurisdictionProfile(
        country_code="RU",
        country_name="Russian Federation",
        is_eu_eea=False,
        is_nato_member=False,
        has_gdpr_adequacy=False,
        wgi_scores={
            GovernanceDimension.RULE_OF_LAW.value: -0.85,
            GovernanceDimension.REGULATORY_QUALITY.value: -0.52,
            GovernanceDimension.GOVERNMENT_EFFECTIVENESS.value: -0.21,
            GovernanceDimension.CONTROL_OF_CORRUPTION.value: -0.92,
            GovernanceDimension.VOICE_AND_ACCOUNTABILITY.value: -1.25,
            GovernanceDimension.POLITICAL_STABILITY.value: -1.15,
        },
        qog_impartiality_score=0.22,
        oecd_ireg_score=0.80,
    ),
    "CN": JurisdictionProfile(
        country_code="CN",
        country_name="People's Republic of China",
        is_eu_eea=False,
        is_nato_member=False,
        has_gdpr_adequacy=False,
        wgi_scores={
            GovernanceDimension.RULE_OF_LAW.value: -0.32,
            GovernanceDimension.REGULATORY_QUALITY.value: -0.18,
            GovernanceDimension.GOVERNMENT_EFFECTIVENESS.value: 0.55,
            GovernanceDimension.CONTROL_OF_CORRUPTION.value: -0.15,
            GovernanceDimension.VOICE_AND_ACCOUNTABILITY.value: -1.68,
            GovernanceDimension.POLITICAL_STABILITY.value: -0.42,
        },
        qog_impartiality_score=0.41,
        oecd_ireg_score=1.40,
    ),
}


class JurisdictionalTrustEngine:
    """Silnik analityczny oceny zaufania jurysdykcyjnego i nadzoru transgranicznego."""

    def __init__(self, profiles: Optional[Dict[str, JurisdictionProfile]] = None) -> None:
        self.profiles = profiles or DEFAULT_JURISDICTIONS

    def get_profile(self, country_code: str) -> Optional[JurisdictionProfile]:
        return self.profiles.get(country_code.upper())

    def compute_trust_score(self, country_code: str) -> float:
        """Oblicza syntetyczny wskaźnik zaufania jurysdykcyjnego (JTS) w skali 0-100."""
        profile = self.get_profile(country_code)
        if not profile:
            return 25.0  # Domyślny konserwatywny próg dla nieznanej jurysdykcji

        # Normalizacja wag WGI z przedziału [-2.5, +2.5] do [0, 100]
        scores = profile.wgi_scores
        rl = scores.get(GovernanceDimension.RULE_OF_LAW.value, 0.0)
        rq = scores.get(GovernanceDimension.REGULATORY_QUALITY.value, 0.0)
        cc = scores.get(GovernanceDimension.CONTROL_OF_CORRUPTION.value, 0.0)
        ge = scores.get(GovernanceDimension.GOVERNMENT_EFFECTIVENESS.value, 0.0)

        # Wymiar praworządności i kontroli korupcji ma najwyższą wagę (60%)
        weighted_wgi = (rl * 0.35) + (cc * 0.25) + (rq * 0.20) + (ge * 0.20)
        # Przeliczenie na 0-100: (val + 2.5) / 5.0 * 100
        wgi_normalized = max(0.0, min(100.0, (weighted_wgi + 2.5) / 5.0 * 100.0))

        # Wpływ QoG Institute (15%) i OECD iREG (15%)
        qog_component = profile.qog_impartiality_score * 100.0
        oecd_component = (profile.oecd_ireg_score / 4.0) * 100.0

        # Wskaźnik końcowy
        final_jts = (wgi_normalized * 0.70) + (qog_component * 0.15) + (oecd_component * 0.15)
        return round(final_jts, 2)

    def evaluate_cross_border_transfer(
        self,
        source_country: str,
        destination_country: str,
        classification: DataClassification,
        cloud_vendor: str = "AWS / Azure / Sovereign Node",
    ) -> CrossBorderAuditResult:
        """Audytuje transfer danych lub zapytania AI przez granice państwowe (Art. 44-49 RODO / NATO)."""
        src = self.get_profile(source_country)
        dst = self.get_profile(destination_country)

        reasons: List[str] = []
        safeguards: List[str] = []

        if not dst:
            return CrossBorderAuditResult(
                verdict=TransferVerdict.BLOCK_INADEQUATE_RULE_OF_LAW,
                jurisdiction_trust_score=20.0,
                rule_of_law_score=-1.5,
                is_adequate_gdpr=False,
                reasons=[f"Nieznana jurysdykcja docelowa [{destination_country}]. Brak weryfikacji w bazach Banku Światowego i OECD."],
                enforced_safeguards=["AIRGAP_LOCKDOWN", "FAIL_CLOSED"],
            )

        jts = self.compute_trust_score(destination_country)
        rl_score = dst.wgi_scores.get(GovernanceDimension.RULE_OF_LAW.value, -1.0)
        cc_score = dst.wgi_scores.get(GovernanceDimension.CONTROL_OF_CORRUPTION.value, -1.0)

        # 1. Bezwzględny wymóg suwerenności: Dane obronne i infrastruktura krytyczna
        if classification in (DataClassification.CRITICAL_INFRASTRUCTURE_OT, DataClassification.STATE_SECURITY_DEFENSE):
            if destination_country != source_country:
                # Dopuszczalne wyłącznie między zweryfikowanymi sojusznikami NATO o JTS >= 75
                if not (dst.is_nato_member and jts >= 75.0):
                    return CrossBorderAuditResult(
                        verdict=TransferVerdict.BLOCK_NATIONAL_SOVEREIGNTY,
                        jurisdiction_trust_score=jts,
                        rule_of_law_score=rl_score,
                        is_adequate_gdpr=dst.has_gdpr_adequacy,
                        reasons=[
                            f"Dane krytyczne OT/Obronne ({classification.value}) podlegają suwerennej blokadzie terytorialnej.",
                            f"Kraj docelowy [{dst.country_name}] nie spełnia sojuszniczego progu bezpieczeństwa NATO CNI.",
                        ],
                        enforced_safeguards=["PURDUE_DATA_DIODE_LOCKDOWN", "LOCAL_SOVEREIGN_EXECUTION_ONLY"],
                    )

        # 2. Naruszenie praworządności: Rule of Law < 0.0 w skali WGI
        if rl_score < 0.0 or cc_score < -0.5:
            return CrossBorderAuditResult(
                verdict=TransferVerdict.BLOCK_INADEQUATE_RULE_OF_LAW,
                jurisdiction_trust_score=jts,
                rule_of_law_score=rl_score,
                is_adequate_gdpr=False,
                reasons=[
                    f"Kraj docelowy [{dst.country_name}] wykazuje negatywny indeks praworządności (Rule of Law: {rl_score:.2f} < 0.0).",
                    "Brak gwarancji ochrony przed nieautoryzowaną inwigilacją obcych służb państwowych.",
                    "Naruszenie wyroku TSUE Schrems II oraz Art. 44 i 45 RODO.",
                ],
                enforced_safeguards=["TRANSFER_PROHIBITION", "ENCLAVE_REJECTION"],
            )

        # 3. Transfer PII / Danych medycznych bez decyzji o adekwatności
        if classification in (DataClassification.PERSONAL_GDPR, DataClassification.SPECIAL_CATEGORY_HEALTH_BIOMETRIC):
            if not dst.has_gdpr_adequacy and not dst.is_eu_eea:
                # Wymóg enklawy TEE i tokenizacji w locie
                reasons.append(f"Kraj [{dst.country_name}] nie posiada bezpośredniej decyzji o adekwatności KE (Art. 45 RODO).")
                safeguards.extend(["REVERSIBLE_TOKEN_VAULT_PSEUDONYMIZATION", "AES_256_GCM_ENCLAVE_ONLY"])
                return CrossBorderAuditResult(
                    verdict=TransferVerdict.ALLOW_WITH_TEE_ENCLAVE,
                    jurisdiction_trust_score=jts,
                    rule_of_law_score=rl_score,
                    is_adequate_gdpr=False,
                    reasons=reasons,
                    enforced_safeguards=safeguards,
                )

        # 4. Dozwolony transfer w bezpiecznym korytarzu
        reasons.append(f"Bezpieczny korytarz jurysdykcyjny ({source_country} -> {destination_country}).")
        reasons.append(f"Wysoki wskaźnik zaufania instytucjonalnego JTS = {jts:.1f} / 100 (WGI Rule of Law: {rl_score:.2f}).")
        return CrossBorderAuditResult(
            verdict=TransferVerdict.ALLOW,
            jurisdiction_trust_score=jts,
            rule_of_law_score=rl_score,
            is_adequate_gdpr=dst.has_gdpr_adequacy,
            reasons=reasons,
            enforced_safeguards=["TLS_1_3_CHACHA20_ENCRYPTION", "MERKLE_RECEIPT_LOGGED"],
        )

    def evaluate_oecd_regulatory_impact(
        self,
        system_name: str,
        high_risk_domain: bool,
        stakeholder_consultations_conducted: bool,
        ex_post_monitoring_enabled: bool,
    ) -> OECDRegulatoryImpactResult:
        """Ewaluuje jakość Oceny Skutków Regulacji (RIA) wg standardów OECD iREG i UK DSIT."""
        score = 0.0
        recommendations: List[str] = []

        if high_risk_domain:
            score += 1.0
        if stakeholder_consultations_conducted:
            score += 1.5
        else:
            recommendations.append("Brak udokumentowanych konsultacji ze stronami społecznymi (OECD iREG Filar Konsultacji).")

        if ex_post_monitoring_enabled:
            score += 1.5
        else:
            recommendations.append("Brak mechanizmu ewaluacji skutków ex-post modelu po wdrożeniu.")

        is_compliant = score >= 3.0 or (not high_risk_domain and score >= 1.5)

        return OECDRegulatoryImpactResult(
            is_compliant=is_compliant,
            ria_score=round(score, 2),
            transparency_tier="ATRS_TIER_2" if high_risk_domain else "ATRS_TIER_1",
            proportionality_confirmed=is_compliant,
            recommendations=recommendations or ["Wdrożenie spełnia wytyczne OECD iREG w zakresie przejrzystości regulacyjnej."],
        )
