# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Test suite for Global Institutional & Jurisdictional Intelligence Engine.

Weryfikuje:
1. World Bank Worldwide Governance Indicators (WGI) & Jurisdictional Trust Score (JTS).
2. Nadzór nad transferami transgranicznymi (GDPR Art. 44-49, Schrems II, NATO CNI).
3. OECD Indicators of Regulatory Policy and Governance (iREG) & ATRS.
4. University of Gothenburg Quality of Government (QoG) Institute (bezstronność biurokratyczna).
"""

from __future__ import annotations

import pytest

from nethical.governance.jurisdictional_intel import (
    DataClassification,
    GovernanceDimension,
    JurisdictionProfile,
    JurisdictionalTrustEngine,
    OECDRegulatoryImpactResult,
    TransferVerdict,
)


def test_world_bank_wgi_trust_score_calculation() -> None:
    """Weryfikuje, że wskaźniki WGI poprawnie różnicują zaufanie do jurysdykcji."""
    engine = JurisdictionalTrustEngine()

    # Państwa o wysokiej praworządności (Rule of Law > 1.3)
    jts_gb = engine.compute_trust_score("GB")
    jts_de = engine.compute_trust_score("DE")
    jts_ch = engine.compute_trust_score("CH")
    jts_pl = engine.compute_trust_score("PL")

    assert jts_gb >= 80.0, f"UK JTS zbyt niski: {jts_gb}"
    assert jts_de >= 80.0, f"Niemcy JTS zbyt niski: {jts_de}"
    assert jts_ch >= 85.0, f"Szwajcaria JTS zbyt niski: {jts_ch}"
    assert jts_pl >= 65.0, f"Polska JTS zbyt niski: {jts_pl}"

    # Państwa autorytarne z ujemną praworządnością (Rule of Law < 0)
    jts_ru = engine.compute_trust_score("RU")
    jts_cn = engine.compute_trust_score("CN")

    assert jts_ru < 40.0, f"Rosja JTS zbyt wysoki: {jts_ru}"
    assert jts_cn < 50.0, f"Chiny JTS zbyt wysoki: {jts_cn}"
    assert jts_gb > jts_ru + 40.0, "Różnica wskaźnika praworządności musi być jednoznaczna!"


def test_gdpr_cross_border_transfer_schrems_ii_blockade() -> None:
    """Weryfikuje blokadę transferu PII do jurysdykcji z deficytem praworządności (Schrems II / RODO Art. 44-45)."""
    engine = JurisdictionalTrustEngine()

    # Transfer danych osobowych z Polski do Rosji (Rule of Law = -0.85 < 0)
    res_bad = engine.evaluate_cross_border_transfer(
        source_country="PL",
        destination_country="RU",
        classification=DataClassification.PERSONAL_GDPR,
    )
    assert res_bad.verdict == TransferVerdict.BLOCK_INADEQUATE_RULE_OF_LAW
    assert res_bad.is_adequate_gdpr is False
    assert "TRANSFER_PROHIBITION" in res_bad.enforced_safeguards
    assert any("Schrems II" in r for r in res_bad.reasons)

    # Bezpieczny transfer wewnątrz UE (PL -> DE)
    res_ok = engine.evaluate_cross_border_transfer(
        source_country="PL",
        destination_country="DE",
        classification=DataClassification.PERSONAL_GDPR,
    )
    assert res_ok.verdict == TransferVerdict.ALLOW
    assert res_ok.is_adequate_gdpr is True
    assert "TLS_1_3_CHACHA20_ENCRYPTION" in res_ok.enforced_safeguards


def test_critical_infrastructure_ot_sovereignty_lockdown() -> None:
    """Weryfikuje suwerenną blokadę transferu telemetrii SCADA/OT poza sojusz NATO."""
    engine = JurisdictionalTrustEngine()

    # Próba eksportu telemetrii OT z Wielkiej Brytanii do Rosji
    res_ot_bad = engine.evaluate_cross_border_transfer(
        source_country="GB",
        destination_country="RU",
        classification=DataClassification.CRITICAL_INFRASTRUCTURE_OT,
    )
    assert res_ot_bad.verdict == TransferVerdict.BLOCK_NATIONAL_SOVEREIGNTY
    assert "PURDUE_DATA_DIODE_LOCKDOWN" in res_ot_bad.enforced_safeguards
    assert "LOCAL_SOVEREIGN_EXECUTION_ONLY" in res_ot_bad.enforced_safeguards

    # Dozwolona współpraca sojusznicza NATO (PL -> US z wysokim JTS)
    res_ot_nato = engine.evaluate_cross_border_transfer(
        source_country="PL",
        destination_country="US",
        classification=DataClassification.STATE_SECURITY_DEFENSE,
    )
    assert res_ot_nato.verdict == TransferVerdict.ALLOW
    assert res_ot_nato.jurisdiction_trust_score >= 75.0


def test_oecd_ireg_regulatory_impact_assessment_compliance() -> None:
    """Weryfikuje zgodność Oceny Skutków Regulacji z metodologią OECD iREG i UK ATRS."""
    engine = JurisdictionalTrustEngine()

    # Wdrożenie wysokiego ryzyka z konsultacjami i monitoringiem ex-post
    res_full = engine.evaluate_oecd_regulatory_impact(
        system_name="Autonomous_Triage_AI_NHS",
        high_risk_domain=True,
        stakeholder_consultations_conducted=True,
        ex_post_monitoring_enabled=True,
    )
    assert res_full.is_compliant is True
    assert res_full.ria_score == 4.0
    assert res_full.transparency_tier == "ATRS_TIER_2"
    assert res_full.proportionality_confirmed is True

    # Wdrożenie wysokiego ryzyka z pominięciem konsultacji społecznych
    res_partial = engine.evaluate_oecd_regulatory_impact(
        system_name="Police_Predictive_Bail_AI",
        high_risk_domain=True,
        stakeholder_consultations_conducted=False,
        ex_post_monitoring_enabled=True,
    )
    assert res_partial.is_compliant is False
    assert any("konsultacji" in r for r in res_partial.recommendations)


def test_qog_institute_impartiality_integration() -> None:
    """Weryfikuje, że wskaźniki bezstronności University of Gothenburg QoG Institute wpływają na profil."""
    engine = JurisdictionalTrustEngine()
    profile_ch = engine.get_profile("CH")
    profile_ru = engine.get_profile("RU")

    assert profile_ch is not None and profile_ru is not None
    assert profile_ch.qog_impartiality_score == 0.96
    assert profile_ru.qog_impartiality_score == 0.22
    assert profile_ch.qog_impartiality_score > profile_ru.qog_impartiality_score * 4.0
