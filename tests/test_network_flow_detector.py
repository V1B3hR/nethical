# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Network Flow Telemetry & Cyber-Immunity Detector (nethical.detectors.network_flow_detector).

Verifies:
1. Normal benign flow traffic allowed.
2. SYN flood signature detection triggering BLOCK / DROP_FLOW.
3. Port scan reconnaissance detection via destination port entropy.
4. Volumetric DDoS bursts detection.
5. Data exfiltration detection.
6. Quarantine Buffer ('The Garden') provenance watermarking and release.
7. Rhythm deviation / network entropy tracking.
8. Async detector pipeline compliance.
"""

import pytest
from nethical.detectors.network_flow_detector import (
    NetworkFlowDetector,
    FlowMitigationAction,
)
from nethical.security.flow_telemetry import (
    GardenQuarantineBuffer,
    NetworkAttackType,
    NetworkFlowSample,
    NetworkRhythmTracker,
)


@pytest.fixture
def detector() -> NetworkFlowDetector:
    """Provides a fresh NetworkFlowDetector instance."""
    return NetworkFlowDetector()


def test_benign_flow_allowed(detector: NetworkFlowDetector) -> None:
    """Verifies that normal network traffic passes without violation."""
    sample = NetworkFlowSample(
        source_ip="192.168.1.50",
        destination_ip="10.0.0.1",
        destination_port=443,
        flow_duration_ms=120.0,
        total_fwd_packets=8,
        total_bwd_packets=12,
        flow_bytes_per_sec=4500.0,
        flow_packets_per_sec=30.0,
        syn_flag_count=1,
        ack_flag_count=1,
        dst_port_entropy=0.2,
    )

    result = detector.evaluate_flows([sample])

    assert result.is_safe is True
    assert result.decision == "ALLOW"
    assert len(result.violations) == 0
    assert result.primary_mitigation == FlowMitigationAction.NONE
    assert result.latency_microseconds < 1500.0


def test_syn_flood_triggers_block_and_drop(detector: NetworkFlowDetector) -> None:
    """Verifies that SYN flood traffic is immediately blocked."""
    flood_sample = NetworkFlowSample(
        source_ip="198.51.100.23",
        destination_ip="10.0.0.1",
        destination_port=80,
        flow_duration_ms=50.0,
        total_fwd_packets=500,
        total_bwd_packets=0,
        syn_flag_count=250,  # Massive SYN burst
        ack_flag_count=0,
    )

    result = detector.evaluate_flows([flood_sample])

    assert result.is_safe is False
    assert result.decision == "BLOCK"
    assert result.primary_mitigation == FlowMitigationAction.DROP_FLOW
    assert any(v.attack_type == NetworkAttackType.SYN_FLOOD for v in result.violations)


def test_port_scan_detection(detector: NetworkFlowDetector) -> None:
    """Verifies that high port entropy reconnaissance is caught."""
    scan_sample = NetworkFlowSample(
        source_ip="203.0.113.88",
        destination_ip="10.0.0.5",
        destination_port=0,
        total_fwd_packets=80,
        total_bwd_packets=2,
        dst_port_entropy=4.2,  # High port entropy indicates port sweep
    )

    result = detector.evaluate_flows([scan_sample])

    assert result.is_safe is False
    assert result.decision in ("RESTRICT", "BLOCK")
    assert any(v.attack_type == NetworkAttackType.PORT_SCAN for v in result.violations)
    assert result.primary_mitigation == FlowMitigationAction.QUARANTINE_SOURCE_IP


def test_ddos_volumetric_burst(detector: NetworkFlowDetector) -> None:
    """Verifies that volumetric bandwidth attacks trigger DDoS detection."""
    ddos_sample = NetworkFlowSample(
        source_ip="198.51.100.99",
        destination_ip="10.0.0.1",
        flow_packets_per_sec=50_000.0,
        flow_bytes_per_sec=25_000_000.0,
    )

    result = detector.evaluate_flows([ddos_sample])

    assert result.is_safe is False
    assert result.decision == "BLOCK"
    assert any(v.attack_type == NetworkAttackType.DDOS_BURST for v in result.violations)


def test_data_exfiltration_detection(detector: NetworkFlowDetector) -> None:
    """Verifies detection of asymmetrical outbound data exfiltration."""
    exfil_sample = NetworkFlowSample(
        source_ip="10.0.0.15",
        destination_ip="198.51.100.4",
        destination_port=443,
        total_fwd_packets=350,
        total_bwd_packets=1,  # Almost no inbound response
        flow_bytes_per_sec=1_200_000.0,
    )

    result = detector.evaluate_flows([exfil_sample])

    assert result.is_safe is False
    assert any(v.attack_type == NetworkAttackType.DATA_EXFILTRATION for v in result.violations)


def test_garden_quarantine_watermarking() -> None:
    """Verifies 'The Garden' quarantine buffer watermarking and verification."""
    quarantine = GardenQuarantineBuffer(capacity=10)
    sample = NetworkFlowSample(source_ip="1.2.3.4")

    # Ingest untrusted observation
    record = quarantine.ingest(sample, metadata={"agent": "external_sensor"})
    assert record["provenance"] == "Observation (External)"
    assert record["passed_safety_check"] is False

    # Verify and release
    verified = quarantine.release_verified(record["record_id"])
    assert verified is not None
    assert verified["provenance"] == "Verified (Internal)"
    assert verified["passed_safety_check"] is True


def test_rhythm_entropy_tracking() -> None:
    """Verifies that sudden shifts in network breathing register an entropy spike."""
    tracker = NetworkRhythmTracker(momentum=0.8)

    # Establish normal quiet rhythm
    normal_vectors = [[2.0, 1.0, 1.0, 5.0, 2.0, 1.0, 1.0, 0.1] for _ in range(20)]
    tracker.update(normal_vectors)

    # Inject sudden chaotic burst
    burst_vectors = [[10.0, 9.0, 0.0, 16.0, 12.0, 50.0, 0.0, 4.5] for _ in range(5)]
    spike = tracker.update(burst_vectors)

    assert spike > 1.0  # Significant deviation from established baseline


@pytest.mark.asyncio
async def test_async_detector_pipeline(detector: NetworkFlowDetector) -> None:
    """Verifies async detect_violations integration."""
    class DummyAction:
        agent_id = "network_sentinel"
        context = {
            "source_ip": "192.0.2.1",
            "destination_ip": "10.0.0.1",
            "destination_port": 80,
            "syn_flag_count": 100,
            "ack_flag_count": 0,
        }

    action = DummyAction()
    violations = await detector.detect_violations(action)

    assert len(violations) >= 1
    assert violations[0]["attack_type"] == "SYN_FLOOD"
