# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Protobuf and Dataclass Telemetry Contracts (nethical.proto).

Verifies serialization, fields, and message mirroring for:
1. CellularTelemetry (5G / 4G link telemetry)
2. EmfRadiationTelemetry (biological exposure telemetry)
3. NetworkFlowTelemetry (8-dimensional flow telemetry)
4. EvaluateRequest with nested telemetry payloads
5. EvaluateResponse with primary mitigation instruction
"""

from nethical.proto import (
    CellularTelemetry,
    EmfRadiationTelemetry,
    EvaluateRequest,
    EvaluateResponse,
    NetworkFlowTelemetry,
)


def test_cellular_telemetry_proto_fields() -> None:
    """Verifies that CellularTelemetry contains all required 3GPP fields."""
    cell = CellularTelemetry(
        cell_id="GNB_KRAKOW_04",
        generation="5G_mmWave",
        frequency_mhz=28000.0,
        bandwidth_mhz=400.0,
        rsrp_dbm=-78.0,
        rsrq_db=-9.5,
        sinr_db=22.0,
        cqi=14,
        latency_ms=3.2,
        signal_grade="EXCELLENT",
    )

    assert cell.cell_id == "GNB_KRAKOW_04"
    assert cell.generation == "5G_mmWave"
    assert cell.cqi == 14
    assert cell.latency_ms == 3.2


def test_emf_radiation_telemetry_proto_fields() -> None:
    """Verifies EmfRadiationTelemetry fields and optional SAR thresholds."""
    emf = EmfRadiationTelemetry(
        emitter_id="tactical_relay",
        frequency_hz=3.5e9,
        tx_power_dbm=28.0,
        estimated_sar_w_kg=1.8,
        power_density_w_m2=6.5,
        human_distance_meters=0.45,
        pulse_modulation_hz=10.0,
        exposure_zone="general_public",
        is_medical_device=False,
    )

    assert emf.emitter_id == "tactical_relay"
    assert emf.estimated_sar_w_kg == 1.8
    assert emf.pulse_modulation_hz == 10.0


def test_network_flow_telemetry_proto_fields() -> None:
    """Verifies NetworkFlowTelemetry 8-dim fields."""
    flow = NetworkFlowTelemetry(
        flow_id="FLOW_TEST_01",
        source_ip="192.168.1.10",
        destination_ip="10.0.0.1",
        destination_port=443,
        flow_duration_ms=45.0,
        total_fwd_packets=25,
        total_bwd_packets=30,
        flow_bytes_per_sec=150000.0,
        flow_packets_per_sec=1200.0,
        syn_flag_count=1,
        ack_flag_count=1,
        dst_port_entropy=0.8,
    )

    assert flow.flow_id == "FLOW_TEST_01"
    assert flow.destination_port == 443
    assert flow.dst_port_entropy == 0.8


def test_evaluate_request_and_response_with_telemetry() -> None:
    """Verifies EvaluateRequest packaging and EvaluateResponse mitigation."""
    cell = CellularTelemetry(cell_id="CELL_99")
    emf = EmfRadiationTelemetry(estimated_sar_w_kg=2.5)
    flow = NetworkFlowTelemetry(dst_port_entropy=3.8)

    req = EvaluateRequest(
        agent_id="sentinel_agent",
        action="transmit_high_power_telemetry",
        cellular_telemetry=cell,
        emf_telemetry=emf,
        network_flow_telemetry=flow,
    )

    assert req.cellular_telemetry is not None
    assert req.emf_telemetry is not None
    assert req.network_flow_telemetry is not None
    assert req.emf_telemetry.estimated_sar_w_kg == 2.5

    resp = EvaluateResponse(
        decision="RESTRICT",
        decision_id="DEC_001",
        risk_score=0.85,
        primary_mitigation="THROTTLE_TX_POWER",
    )

    assert resp.decision == "RESTRICT"
    assert resp.primary_mitigation == "THROTTLE_TX_POWER"
