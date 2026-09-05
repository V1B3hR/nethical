"""Zestaw testów jednostkowych i integracyjnych: Kinetic Safety OS & ISO/IEC 42001 (Faza 4).

Weryfikuje:
1. Gubernator kinetyczny (KineticSafetyGovernor): interlock przestrzenny, clamping prędkości, E-STOP latch.
2. Pakiet ISO/IEC 42001:2023 & IEEE 7000: audyt AIMS, klauzule 4-10, kontrole Załącznika A.
3. Integrację GovernanceGateway z poleceniami aktuacji robotycznej.
4. Endpointy FastAPI (Kinetic OS, E-STOP, ISO 42001, Portal stats).
"""

import pytest
from fastapi.testclient import TestClient

from nethical.edge.kinetic_safety import (
    KineticSafetyGovernor,
    KineticSafetyEnvelope,
    RoboticSensorTelemetry,
)
from nethical.compliance.packs.iso42001_pack import ISO42001CompliancePack
from nethical.gateway.proxy import GovernanceGateway
from nethical.api import app


@pytest.fixture
def governor():
    """Inicjalizuje czystego gubernatora kinetycznego."""
    return KineticSafetyGovernor()


@pytest.fixture
def client():
    """Inicjalizuje klienta testowego FastAPI."""
    return TestClient(app)


# ==============================================================================
# 1. TESTY KINETIC SAFETY OS (EMBODIED AI)
# ==============================================================================

def test_kinetic_safe_actuation(governor):
    """Polecenie aktuacji w bezpiecznym korytarzu z dalekim człowiekiem zostaje dozwolone."""
    telemetry = RoboticSensorTelemetry(
        human_distance_meters=2.5,
        current_velocity_mps=0.5,
        applied_torque_nm=10.0,
        active_zone="production_zone_A",
        obstacle_detected=False,
    )
    decision = governor.evaluate_actuation(
        tool_name="actuate_joint_motor",
        arguments={"velocity_mps": 0.8, "torque_nm": 15.0},
        telemetry=telemetry,
    )

    assert decision.decision == "ALLOW"
    assert not decision.estop_engaged
    assert len(decision.violations) == 0
    assert decision.latency_microseconds < 5000.0  # sub-millisecond expected


def test_kinetic_proximity_warning_velocity_clamp(governor):
    """Gdy człowiek znajduje się w strefie ostrzegawczej (<0.8 m), prędkość zostaje ograniczona."""
    telemetry = RoboticSensorTelemetry(
        human_distance_meters=0.6,  # między 0.3 m a 0.8 m
        current_velocity_mps=1.0,
        applied_torque_nm=12.0,
        active_zone="production_zone_A",
    )
    decision = governor.evaluate_actuation(
        tool_name="actuate_manipulator",
        arguments={"velocity_mps": 1.2},
        telemetry=telemetry,
    )

    assert decision.decision == "RESTRICT"
    assert decision.clamped_velocity_mps == governor.envelope.reduced_velocity_near_human_mps
    assert any("HumanProximityWarning" in v for v in decision.violations)
    assert not decision.estop_engaged


def test_kinetic_critical_proximity_estop_latch(governor):
    """Krytyczne naruszenie bąbla człowieka (<0.3 m) natychmiast zatrzaskuje E-STOP (Prawo 1)."""
    telemetry = RoboticSensorTelemetry(
        human_distance_meters=0.15,  # krytyczne zagrożenie
        current_velocity_mps=0.4,
        applied_torque_nm=10.0,
        active_zone="production_zone_A",
    )
    decision = governor.evaluate_actuation(
        tool_name="move_robot_base",
        arguments={"velocity_mps": 0.5},
        telemetry=telemetry,
    )

    assert decision.decision == "EMERGENCY_STOP"
    assert decision.estop_engaged is True
    assert governor.estop_active is True
    assert any("Human Proximity Breach" in v for v in decision.violations)

    # Kolejne zapytania muszą być natychmiast blokowane przez aktywny E-STOP
    second_decision = governor.evaluate_actuation(
        tool_name="move_robot_base",
        arguments={},
        telemetry=RoboticSensorTelemetry(human_distance_meters=5.0),
    )
    assert second_decision.decision == "EMERGENCY_STOP"
    assert second_decision.estop_engaged is True


def test_kinetic_estop_reset_security(governor):
    """Wyłącznik E-STOP może zostać odblokowany wyłącznie właściwym kluczem PIN."""
    governor.trigger_estop("Testowy alarm bezpieczeństwa")
    assert governor.estop_active is True

    # Próba nieautoryzowanego resetu
    success, msg = governor.reset_estop("INVALID_PIN")
    assert success is False
    assert governor.estop_active is True

    # Prawidłowy autoryzowany reset
    success, msg = governor.reset_estop(KineticSafetyGovernor.RESET_PIN)
    assert success is True
    assert governor.estop_active is False
    assert governor.estop_reason is None


def test_kinetic_fail_closed_missing_telemetry(governor):
    """Zasada Fail-Closed: polecenie aktuacji bez telemetrii sensorowej jest bezwzględnie blokowane."""
    decision = governor.evaluate_actuation(
        tool_name="actuate_gripper",
        arguments={"force_n": 50},
        telemetry=None,
    )
    assert decision.decision == "BLOCK"
    assert any("Telemetry Missing" in v for v in decision.violations)


def test_kinetic_spatial_and_torque_violations(governor):
    """Naruszenie strefy operacyjnej oraz przekroczenie dopuszczalnego momentu obrotowego."""
    # Strefa niedozwolona
    telemetry_wrong_zone = RoboticSensorTelemetry(
        human_distance_meters=2.0,
        active_zone="unauthorized_public_corridor",
    )
    res_zone = governor.evaluate_actuation("actuate_arm", {}, telemetry_wrong_zone)
    assert res_zone.decision == "BLOCK"
    assert any("SpatialZoneViolation" in v for v in res_zone.violations)

    # Zbyt wysoki moment obrotowy
    telemetry_valid_zone = RoboticSensorTelemetry(
        human_distance_meters=2.0,
        active_zone="production_zone_A",
    )
    res_torque = governor.evaluate_actuation(
        "actuate_arm",
        {"torque_nm": 45.0},  # max to 25.0 Nm
        telemetry_valid_zone,
    )
    assert res_torque.decision == "BLOCK"
    assert any("TorqueLimitViolation" in v for v in res_torque.violations)


# ==============================================================================
# 2. TESTY ISO/IEC 42001:2023 & IEEE 7000 COMPLIANCE PACK
# ==============================================================================

def test_iso42001_full_compliance():
    """Pełna konfiguracja AIMS spełnia kryteria certyfikacji ISO/IEC 42001:2023."""
    pack = ISO42001CompliancePack()
    metadata = {
        "has_ai_policy": True,
        "has_ai_ethics_officer": True,
        "has_ai_risk_assessment": True,
        "has_competency_framework": True,
        "has_ai_impact_assessment": True,
        "has_lifecycle_governance": True,
        "has_tamperproof_ledger": True,
        "has_continuous_monitoring": True,
        "has_inoculation_mesh": True,
        "stakeholder_requirements_defined": True,
    }
    evaluation = pack.evaluate_aims(metadata)

    assert evaluation.is_certified_ready is True
    assert evaluation.overall_readiness_score >= 0.85
    assert evaluation.certification_stage == "CERTIFICATION_READY"
    assert len(evaluation.clauses) == 7
    assert len(evaluation.annex_a_controls) == 9
    assert evaluation.fundamental_laws_coverage_ratio == 1.0


def test_iso42001_gap_remediation_required():
    """Brak polityki AI, oficerów i rejestru audytowego wymaga działań korygujących."""
    pack = ISO42001CompliancePack()
    metadata = {
        "has_ai_policy": False,
        "has_ai_ethics_officer": False,
        "has_ai_risk_assessment": False,
        "has_tamperproof_ledger": False,
    }
    evaluation = pack.evaluate_aims(metadata)

    assert evaluation.is_certified_ready is False
    assert evaluation.certification_stage == "GAP_REMEDIATION_REQUIRED"
    assert len(evaluation.recommendations) > 0


# ==============================================================================
# 3. TESTY INTEGRACJI GOVERNANCE GATEWAY Z INTERLOCKIEM KINETYCZNYM
# ==============================================================================

def test_gateway_kinetic_actuation_interlock():
    """Gateway automatycznie rozpoznaje wywołanie robotyczne i egzekwuje E-STOP."""
    gov = KineticSafetyGovernor()
    gateway = GovernanceGateway(kinetic_governor=gov)

    # Symulacja krytycznego bąbla człowieka
    context = {
        "telemetry": {
            "human_distance_meters": 0.2,
            "current_velocity_mps": 0.8,
            "applied_torque_nm": 15.0,
            "active_zone": "production_zone_A",
        }
    }

    decision = gateway.intercept_tool_call(
        agent_id="industrial_arm_agent",
        tool_name="actuate_welding_torch",
        arguments={"velocity_mps": 1.0},
        context=context,
    )

    assert decision.decision == "TERMINATE"
    assert decision.estop_engaged is True
    assert decision.kinetic_evaluation is not None
    assert any("Human Proximity Breach" in v for v in decision.violations)


# ==============================================================================
# 4. TESTY ENDPOINTÓW FASTAPI (KINETIC & ISO 42001)
# ==============================================================================

def test_api_kinetic_endpoints(client):
    """Weryfikuje endpointy /api/v1/kinetic/evaluate, /estop, /reset, /telemetry."""
    # 1. Telemetria początkowa
    tel_res = client.get("/api/v1/kinetic/telemetry")
    assert tel_res.status_code == 200
    assert "estop_active" in tel_res.json()

    # 2. Ewaluacja bezpiecznego ruchu
    eval_res = client.post(
        "/api/v1/kinetic/evaluate",
        json={
            "tool_name": "actuate_conveyor",
            "arguments": {"speed": 0.5},
            "telemetry": {
                "human_distance_meters": 2.0,
                "current_velocity_mps": 0.2,
                "applied_torque_nm": 5.0,
                "active_zone": "production_zone_A",
                "obstacle_detected": False,
            },
        },
    )
    assert eval_res.status_code == 200
    assert eval_res.json()["decision"] == "ALLOW"

    # 3. Ręczne wywołanie E-STOP
    estop_res = client.post(
        "/api/v1/kinetic/estop",
        json={"reason": "Operator panel trigger"},
    )
    assert estop_res.status_code == 200
    assert estop_res.json()["estop_active"] is True

    # 4. Próba resetu nieprawidłowym PIN-em (403)
    bad_reset = client.post(
        "/api/v1/kinetic/reset",
        json={"auth_pin": "WRONG_SECRET"},
    )
    assert bad_reset.status_code == 403

    # 5. Prawidłowy reset E-STOP
    good_reset = client.post(
        "/api/v1/kinetic/reset",
        json={"auth_pin": KineticSafetyGovernor.RESET_PIN},
    )
    assert good_reset.status_code == 200
    assert good_reset.json()["estop_active"] is False


def test_api_iso42001_endpoints(client):
    """Weryfikuje endpointy audytu ISO/IEC 42001."""
    # Domyślny profil Nethical Enterprise
    res_default = client.get("/api/v1/compliance/iso42001")
    assert res_default.status_code == 200
    data = res_default.json()
    assert data["is_certified_ready"] is True
    assert data["overall_readiness_score"] >= 0.85

    # Własny audyt klienta
    res_custom = client.post(
        "/api/v1/compliance/iso42001/audit",
        json={"metadata": {"has_ai_policy": False}},
    )
    assert res_custom.status_code == 200
    assert "clauses" in res_custom.json()


def test_portal_stats_includes_kinetic_and_iso(client):
    """Weryfikuje czy portal stats zwraca metryki kinetyczne oraz ISO 42001."""
    res = client.get("/api/v1/portal/stats")
    assert res.status_code == 200
    stats = res.json()
    assert "kinetic_safety" in stats
    assert "iso42001_readiness_score" in stats
    assert stats["iso42001_readiness_score"] > 0.9
