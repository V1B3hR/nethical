# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Zestaw testów jednostkowych dla modułu obrony przed atakiem 'Cichy Cel' (SilentTargetSteppingStoneGuard)."""

import pytest

from nethical.security.stepping_stone_guard import (
    SilentTargetSteppingStoneGuard,
    NetworkTier,
    NetworkHop,
    SteppingStoneAlert,
)
from nethical.security.merkle_ledger import MerkleLedger


def test_purdue_model_level_3_4_isolation() -> None:
    """Weryfikuje, że ruch z sieci domowych/CDN nie może bezpośrednio trafiać do SCADA/PLC."""
    ledger = MerkleLedger()
    guard = SilentTargetSteppingStoneGuard(ledger=ledger)

    # 1. Próba połączenia z sieci domowej do sterownika PLC kotła elektrociepłowni
    allowed, action, alert = guard.evaluate_traffic_flow(
        source_tier=NetworkTier.RESIDENTIAL_CONSUMER,
        destination_tier=NetworkTier.CONTROL_PLC_L1_L2,
        destination_port=502,  # Modbus
        protocol="MODBUS_TCP",
        asset_name="Elektrociepłownia EC-2 Kocioł Parowy K1",
    )

    assert allowed is False
    assert "HARDWARE_DATA_DIODE_LOCKDOWN" in action
    assert alert is not None
    assert alert.threat_type == "PURDUE_MODEL_BREACH"
    assert alert.severity == "CRITICAL"
    assert alert.receipt_id is not None
    assert alert.merkle_root is not None
    assert len(guard.alerts_history) == 1

    # 2. Ruch dozwolony: stacja inżynierska w DMZ do SCADA L3
    allowed_dmz, action_dmz, alert_dmz = guard.evaluate_traffic_flow(
        source_tier=NetworkTier.INDUSTRIAL_DMZ,
        destination_tier=NetworkTier.OPERATIONS_SCADA_L3,
        destination_port=4840,
        protocol="OPC_UA",
    )
    assert allowed_dmz is True
    assert alert_dmz is None


def test_stepping_stone_residential_corridor_detection() -> None:
    """Weryfikuje wykrywanie korytarza przeskoku przez budynki osiedla zbiegające się ku elektrociepłowni."""
    ledger = MerkleLedger()
    guard = SilentTargetSteppingStoneGuard(ledger=ledger)

    # Symulacja łańcucha skoków: Budynek 1 -> 4 -> 7 -> 21 -> 77 -> 98 (dom pracownika EC)
    chain = [
        NetworkHop(node_id="router_bldg_01", tier=NetworkTier.RESIDENTIAL_CONSUMER, ip_address="192.168.1.10", geo_location="Osiedle Słoneczne 1", protocol="UDP/QUIC"),
        NetworkHop(node_id="router_bldg_04", tier=NetworkTier.RESIDENTIAL_CONSUMER, ip_address="192.168.4.15", geo_location="Osiedle Słoneczne 4", protocol="UDP/QUIC"),
        NetworkHop(node_id="router_bldg_07", tier=NetworkTier.RESIDENTIAL_CONSUMER, ip_address="192.168.7.20", geo_location="Osiedle Słoneczne 7", protocol="UDP/QUIC"),
        NetworkHop(node_id="router_bldg_21", tier=NetworkTier.RESIDENTIAL_CONSUMER, ip_address="192.168.21.35", geo_location="Osiedle Słoneczne 21", protocol="UDP/QUIC"),
        NetworkHop(node_id="router_bldg_77", tier=NetworkTier.RESIDENTIAL_CONSUMER, ip_address="192.168.77.50", geo_location="Osiedle Słoneczne 77", protocol="UDP/QUIC"),
        NetworkHop(
            node_id="router_bldg_98_worker",
            tier=NetworkTier.RESIDENTIAL_CONSUMER,
            ip_address="192.168.98.100",
            geo_location="Osiedle Słoneczne 98",
            protocol="UDP/QUIC",
            metadata={"proximity": "district_heating_plant", "is_employee_residential_node": True}
        ),
    ]

    is_corridor, alert = guard.detect_stepping_stone_corridor(
        chain=chain,
        target_proximity_tag="district_heating_plant",
    )

    assert is_corridor is True
    assert alert is not None
    assert alert.threat_type == "STEPPING_STONE_CORRIDOR"
    assert alert.severity == "HIGH"
    assert alert.target_asset == "district_heating_plant"
    assert len(alert.source_chain) == 6
    assert alert.receipt_id is not None
    assert alert.merkle_root is not None


def test_streaming_media_covert_tunnel_inspection() -> None:
    """Weryfikuje wykrywanie ukrytych komend SCADA/shell w strumieniu wideo/audio."""
    guard = SilentTargetSteppingStoneGuard()

    # Pakiet wideo zawierający wstrzyknięte polecenie modyfikacji zaworu
    malicious_video_chunk = (
        b"\x00\x00\x00\x01\x67\x42\x00\x1f"
        b"METADATA_FRAME: set_valve_pressure boiler=1 target=999bar override_boiler"
        b"\x00\x00\x00\x01\x68\xce\x06\xe2"
    )

    is_clean, alert = guard.inspect_streaming_packet(
        stream_id="stream_user_netflix_bldg98",
        payload_bytes=malicious_video_chunk,
        declared_codec="H265_AV1",
    )

    assert is_clean is False
    assert alert is not None
    assert alert.threat_type == "STREAM_COVERT_TUNNEL"
    assert alert.severity == "CRITICAL"
    assert any("set_valve_pressure" in ind for ind in alert.detected_indicators)


def test_legitimate_civilian_streaming_permitted() -> None:
    """Weryfikuje, że legalny, czysty ruch streamingu mieszkańców osiedla nie jest blokowany."""
    guard = SilentTargetSteppingStoneGuard()

    clean_video_chunk = b"\x00\x00\x00\x01\x67\x42\x00\x1f" + (b"\xaa\xbb\xcc\xdd" * 128)
    is_clean, alert = guard.inspect_streaming_packet(
        stream_id="legit_stream_4k_user12",
        payload_bytes=clean_video_chunk,
    )

    assert is_clean is True
    assert alert is None
