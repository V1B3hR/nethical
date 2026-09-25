# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Biological EMF & RF Radiation Safety Detector (nethical.detectors.emf_radiation_detector).

Verifies:
1. Safe emissions under standard thresholds (Decision: ALLOW).
2. SAR limit violations (< 2.0 W/kg) and automatic Tx power throttling calculation.
3. Millimeter-wave (mmWave) power density limits (10 W/m² public, 50 W/m² occupational).
4. ALARA protocol enforcement when humans are in close proximity (< 0.5m).
5. Neuromodulation resonance guard against uncertified Alpha/Theta pulse modulation.
6. Async integration with governance action pipelines.
"""

import pytest
from nethical.detectors.emf_radiation_detector import (
    EmfRadiationDetector,
    EmfEmissionTelemetry,
    EmfExposureZone,
    EmfMitigationAction,
)


@pytest.fixture
def detector() -> EmfRadiationDetector:
    """Provides a cleanly initialized EmfRadiationDetector instance."""
    return EmfRadiationDetector()


def test_safe_emission_allowed(detector: EmfRadiationDetector) -> None:
    """Verifies that typical low-power WiFi/BLE emissions are allowed."""
    telemetry = EmfEmissionTelemetry(
        emitter_id="wifi7_iot_gateway",
        frequency_hz=5.8e9,
        tx_power_dbm=12.0,  # ~16 mW (low power)
        estimated_sar_w_kg=0.15,
        power_density_w_m2=0.5,
        human_distance_meters=2.0,
        pulse_modulation_hz=100.0,
        exposure_zone=EmfExposureZone.GENERAL_PUBLIC,
    )

    result = detector.evaluate_telemetry(telemetry)

    assert result.is_safe is True
    assert result.decision == "ALLOW"
    assert len(result.violations) == 0
    assert result.primary_mitigation == EmfMitigationAction.NONE
    assert result.latency_microseconds < 1000.0  # Must be sub-millisecond


def test_sar_limit_violation_triggers_throttling(detector: EmfRadiationDetector) -> None:
    """Verifies that SAR exceeding 2.0 W/kg triggers RESTRICT and computes required dB drop."""
    telemetry = EmfEmissionTelemetry(
        emitter_id="5g_edge_transceiver",
        frequency_hz=3.5e9,
        tx_power_dbm=30.0,  # 1000 mW (high)
        estimated_sar_w_kg=3.2,  # Exceeds 2.0 W/kg
        power_density_w_m2=4.0,
        human_distance_meters=0.8,
    )

    result = detector.evaluate_telemetry(telemetry)

    assert result.is_safe is False
    assert result.decision == "RESTRICT"
    assert len(result.violations) >= 1
    sar_v = next(v for v in result.violations if v.violation_type == "EXCESSIVE_SAR_EXPOSURE")
    assert sar_v.measured_value == 3.2
    assert sar_v.allowed_limit == 2.0
    assert sar_v.mitigation in (EmfMitigationAction.THROTTLE_TX_POWER, EmfMitigationAction.EMERGENCY_RADIO_CUTOFF)
    assert result.suggested_tx_power_dbm is not None
    assert result.suggested_tx_power_dbm < 30.0


def test_extreme_sar_triggers_emergency_cutoff(detector: EmfRadiationDetector) -> None:
    """Verifies that severe SAR (> 1.5x limit, e.g. 4.5 W/kg) triggers BLOCK & radio cutoff."""
    telemetry = EmfEmissionTelemetry(
        emitter_id="tactical_repeater",
        frequency_hz=2.4e9,
        tx_power_dbm=36.0,
        estimated_sar_w_kg=4.8,  # > 2.0 * 1.5 = 3.0 W/kg
    )

    result = detector.evaluate_telemetry(telemetry)

    assert result.is_safe is False
    assert result.decision == "BLOCK"
    assert result.primary_mitigation == EmfMitigationAction.EMERGENCY_RADIO_CUTOFF


def test_mmwave_power_density_public_vs_occupational(detector: EmfRadiationDetector) -> None:
    """Verifies public limit is 10 W/m² and occupational is 50 W/m²."""
    # Public zone with 18 W/m² -> Violation
    public_telemetry = EmfEmissionTelemetry(
        emitter_id="mmwave_radar_60ghz",
        frequency_hz=60.0e9,
        tx_power_dbm=20.0,
        power_density_w_m2=18.0,
        exposure_zone=EmfExposureZone.GENERAL_PUBLIC,
    )
    res_pub = detector.evaluate_telemetry(public_telemetry)
    assert res_pub.is_safe is False
    assert any(v.violation_type == "EXCESSIVE_POWER_DENSITY" for v in res_pub.violations)

    # Same power density in Occupational zone -> Allowed (limit is 50 W/m²)
    occ_telemetry = EmfEmissionTelemetry(
        emitter_id="mmwave_radar_60ghz",
        frequency_hz=60.0e9,
        tx_power_dbm=20.0,
        power_density_w_m2=18.0,
        exposure_zone=EmfExposureZone.OCCUPATIONAL,
    )
    res_occ = detector.evaluate_telemetry(occ_telemetry)
    assert res_occ.is_safe is True
    assert res_occ.decision == "ALLOW"


def test_alara_protocol_near_field_throttling(detector: EmfRadiationDetector) -> None:
    """Verifies that transmit power is clamped when human is within 0.5m."""
    telemetry = EmfEmissionTelemetry(
        emitter_id="wearable_transmitter",
        frequency_hz=2.4e9,
        tx_power_dbm=23.0,  # 200 mW, too high for near contact under ALARA
        human_distance_meters=0.15,  # 15 cm
        estimated_sar_w_kg=1.2,  # SAR is below 2.0, but ALARA mandates lower power
    )

    result = detector.evaluate_telemetry(telemetry)

    assert result.is_safe is False
    assert result.decision == "RESTRICT"
    alara_v = next(v for v in result.violations if v.violation_type == "ALARA_PROXIMITY_VIOLATION")
    assert alara_v.mitigation == EmfMitigationAction.THROTTLE_TX_POWER
    assert result.suggested_tx_power_dbm <= 14.0


def test_medical_device_exemption_from_alara(detector: EmfRadiationDetector) -> None:
    """Verifies that certified medical devices are exempt from general consumer ALARA throttling."""
    telemetry = EmfEmissionTelemetry(
        emitter_id="certified_neuro_telemetry_implant",
        frequency_hz=402.0e6,  # MICS band
        tx_power_dbm=18.0,
        human_distance_meters=0.05,
        estimated_sar_w_kg=0.8,
        is_medical_device=True,
    )

    result = detector.evaluate_telemetry(telemetry)

    assert result.is_safe is True
    assert not any(v.violation_type == "ALARA_PROXIMITY_VIOLATION" for v in result.violations)


def test_neuromodulation_resonance_guard(detector: EmfRadiationDetector) -> None:
    """Verifies that RF pulsing at human Alpha frequency (10 Hz) triggers FREQUENCY_HOP."""
    telemetry = EmfEmissionTelemetry(
        emitter_id="unauthorized_bci_probe",
        frequency_hz=915.0e6,
        tx_power_dbm=10.0,
        pulse_modulation_hz=10.0,  # Exactly in the 8-13 Hz Alpha band
        estimated_sar_w_kg=0.2,
    )

    result = detector.evaluate_telemetry(telemetry)

    assert result.is_safe is False
    assert result.decision == "RESTRICT"
    res_v = next(v for v in result.violations if v.violation_type == "NEUROMODULATION_FREQUENCY_RISK")
    assert res_v.mitigation == EmfMitigationAction.FREQUENCY_HOP
    assert result.primary_mitigation == EmfMitigationAction.FREQUENCY_HOP


@pytest.mark.asyncio
async def test_async_action_pipeline_integration(detector: EmfRadiationDetector) -> None:
    """Verifies detect_violations async method with action mock."""
    class DummyAction:
        agent_id = "drone_comm_agent"
        context = {
            "emitter_id": "drone_5g_relay",
            "frequency_hz": 28.0e9,
            "tx_power_dbm": 33.0,
            "estimated_sar_w_kg": 2.7,
        }

    action = DummyAction()
    violations = await detector.detect_violations(action)

    assert len(violations) >= 1
    assert violations[0]["emitter_id"] == "drone_5g_relay"
    assert violations[0]["violation_type"] == "EXCESSIVE_SAR_EXPOSURE"
