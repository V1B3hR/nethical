# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Ambassador RF / EMF & Network Flow Symbiotic Extensions.

Verifies:
1. BlyskawicaAmbassador.evaluate_rf_emission client & daemon fallback.
2. BlyskawicaAmbassador.evaluate_network_flow client & daemon fallback.
3. AntiHallucinationGovernor recognizes ICNIRP, RED, and Prawo Telekomunikacyjne statutory references.
4. Fake telecom standards are flagged as hallucinations.
5. Sparing dilemma archetypes include SPAR-33 and SPAR-34.
"""

import pytest
from nethical.ambassador.client import BlyskawicaAmbassador
from nethical.ambassador.co_training import (
    AntiHallucinationGovernor,
    SymbioticCoTrainingEngine,
)


@pytest.fixture
def ambassador() -> BlyskawicaAmbassador:
    """Provides a BlyskawicaAmbassador client instance."""
    return BlyskawicaAmbassador()


def test_ambassador_evaluate_rf_emission_safe(ambassador: BlyskawicaAmbassador) -> None:
    """Verifies that safe RF telemetry passes ambassador evaluation."""
    payload = {
        "emitter_id": "iot_sensor_5g",
        "frequency_hz": 3.5e9,
        "tx_power_dbm": 12.0,
        "estimated_sar_w_kg": 0.25,
        "human_distance_meters": 1.5,
    }

    res = ambassador.evaluate_rf_emission(payload)

    assert res["is_safe"] is True
    assert res["decision"] == "ALLOW"
    assert len(res["violations"]) == 0


def test_ambassador_evaluate_rf_emission_violation(ambassador: BlyskawicaAmbassador) -> None:
    """Verifies that severe SAR triggers violation and elevates cortisol."""
    payload = {
        "emitter_id": "tactical_jammer_node",
        "frequency_hz": 28.0e9,
        "tx_power_dbm": 35.0,
        "estimated_sar_w_kg": 3.8,  # Excessive SAR
    }

    res = ambassador.evaluate_rf_emission(payload)

    assert res["is_safe"] is False
    assert res["decision"] in ("RESTRICT", "BLOCK")
    assert len(res["violations"]) >= 1
    assert res["cortisol_level"] > 0.04


def test_ambassador_evaluate_network_flow_syn_flood(ambassador: BlyskawicaAmbassador) -> None:
    """Verifies that malicious SYN flood flow triggers block and cortisol surge."""
    flow = {
        "source_ip": "198.51.100.77",
        "destination_ip": "10.0.0.1",
        "destination_port": 80,
        "syn_flag_count": 200,
        "ack_flag_count": 0,
    }

    res = ambassador.evaluate_network_flow(flow)

    assert res["is_safe"] is False
    assert res["decision"] == "BLOCK"
    assert res["primary_mitigation"] == "DROP_FLOW"
    assert res["cortisol_level"] > 0.04


def test_anti_hallucination_with_telecom_standards() -> None:
    """Verifies that ICNIRP and RED standards are grounded, but fake ones are rejected."""
    gov = AntiHallucinationGovernor()

    # Valid citation
    valid_text = "Zgodnie z ICNIRP 2020 Basic Restrictions oraz RED 2014/53/EU Art. 3.1(a) nakazuję redukcję mocy Tx."
    is_grounded, violations, _ = gov.verify_epistemic_grounding(valid_text)
    assert is_grounded is True
    assert len(violations) == 0

    # Hallucinated article
    fake_text = "Na podstawie ICNIRP 2020 Art. 9999 oraz RED 2014/53/EU Art. 777 nakazuję wyłączenie."
    is_grounded_fake, violations_fake, _ = gov.verify_epistemic_grounding(fake_text)
    assert is_grounded_fake is False
    assert len(violations_fake) >= 1


def test_sparing_dilemmas_include_rf_and_cyber_immunity() -> None:
    """Verifies that SPAR-33 and SPAR-34 are part of the generated dilemma suite."""
    engine = SymbioticCoTrainingEngine()
    dilemmas = engine.generate_sparing_dilemmas(count=40)

    dilemma_ids = [d.dilemma_id for d in dilemmas]
    assert any(d_id.startswith("SPAR-33-RF-SAR-BIOLOGICAL-OVERRIDE") for d_id in dilemma_ids)
    assert any(d_id.startswith("SPAR-34-NET-CICIDS-DEFENSE-QUARANTINE") for d_id in dilemma_ids)
