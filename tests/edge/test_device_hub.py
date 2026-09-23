# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Universal Edge Device Admission Hub (tests.edge.test_device_hub).

Validates vendor-agnostic device admission, capability tier assignment,
cryptographic handshake tokens, watchdog heartbeat enforcement, and socket server discovery.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Generator

import pytest

from nethical.edge.device_hub import (
    ActuationBus,
    DeviceAdmissionResult,
    DeviceType,
    EdgeCapabilityTier,
    EdgeDeviceHub,
    EdgeDeviceProfile,
)
from nethical.edge.industrial_fieldbus import IndustrialFieldbusInterlock


@pytest.fixture
def fieldbus() -> IndustrialFieldbusInterlock:
    """Provide a fresh industrial fieldbus interlock instance."""
    return IndustrialFieldbusInterlock()


@pytest.fixture
def hub(fieldbus: IndustrialFieldbusInterlock) -> EdgeDeviceHub:
    """Provide a configured EdgeDeviceHub with 100ms watchdog timeout."""
    return EdgeDeviceHub(fieldbus_interlock=fieldbus, default_heartbeat_timeout_ms=100)


class TestEdgeDeviceProfile:
    """Unit tests for EdgeDeviceProfile schema and validation."""

    def test_profile_creation_defaults(self) -> None:
        """Verify profile creation with default and mandatory fields."""
        profile = EdgeDeviceProfile(
            device_id="arm_kuka_01",
            manufacturer="KUKA",
            model="LBR iiwa 14 R820",
            device_type=DeviceType.ROBOT_6AXIS,
            mac_address="00:0F:4B:88:12:34",
            ip_address="192.168.1.50",
            actuation_bus=ActuationBus.ETHERCAT_FSOE,
        )
        assert profile.device_id == "arm_kuka_01"
        assert profile.manufacturer == "KUKA"
        assert profile.device_type == DeviceType.ROBOT_6AXIS
        assert profile.actuation_bus == ActuationBus.ETHERCAT_FSOE
        assert profile.has_hardware_watchdog is True
        assert profile.tpm_attestation_status == "HARDWARE_TPM_VERIFIED"
        assert profile.declared_safety_envelope == {}


class TestDeviceAdmission:
    """Unit tests for device admission and capability tier negotiation."""

    def test_admit_scada_plc_tier1(self, hub: EdgeDeviceHub) -> None:
        """SCADA PLC must be classified as TIER1_MICRO."""
        profile = EdgeDeviceProfile(
            device_id="plc_siemens_s7",
            manufacturer="Siemens",
            model="S7-1500",
            device_type=DeviceType.SCADA_PLC,
            mac_address="00:1C:06:12:34:56",
            actuation_bus=ActuationBus.MODBUS_TCP,
        )
        res = hub.admit_device(profile)
        assert res.admitted is True
        assert res.assigned_tier == EdgeCapabilityTier.TIER1_MICRO
        assert res.bound_cutoff_channel == ActuationBus.MODBUS_TCP
        assert len(res.handshake_token) == 64
        assert len(res.merkle_proof_hash) == 64
        assert "Default sovereign safety envelope assigned" in res.reasons

        # Verify device stored
        stored = hub.get_device("plc_siemens_s7")
        assert stored is not None
        assert stored.declared_safety_envelope["max_pressure_bar"] == 10.0

    def test_admit_robot_and_drone_tier2(self, hub: EdgeDeviceHub) -> None:
        """Robots and UAVs must be classified as TIER2_ROBOTICS."""
        robot = EdgeDeviceProfile(
            device_id="fanuc_crx",
            manufacturer="FANUC",
            model="CRX-10iA",
            device_type=DeviceType.ROBOT_6AXIS,
            mac_address="08:00:27:AA:BB:CC",
            actuation_bus=ActuationBus.CAN_BUS,
        )
        drone = EdgeDeviceProfile(
            device_id="uav_skydio",
            manufacturer="Skydio",
            model="X2D",
            device_type=DeviceType.UAV_DRONE,
            mac_address="00:1E:58:11:22:33",
            actuation_bus=ActuationBus.ROS2_DDS,
        )

        res_robot = hub.admit_device(robot)
        res_drone = hub.admit_device(drone)

        assert res_robot.assigned_tier == EdgeCapabilityTier.TIER2_ROBOTICS
        assert res_drone.assigned_tier == EdgeCapabilityTier.TIER2_ROBOTICS
        assert hub.get_device("fanuc_crx").declared_safety_envelope["max_tcp_velocity_mps"] == 0.25
        assert hub.get_device("uav_skydio").declared_safety_envelope["max_altitude_agl_m"] == 120.0

    def test_admit_medical_and_general_tier3(self, hub: EdgeDeviceHub) -> None:
        """Medical and general devices must receive TIER3_AI_EDGE."""
        med = EdgeDeviceProfile(
            device_id="surgical_robot_01",
            manufacturer="Intuitive",
            model="daVinci_SP",
            device_type=DeviceType.MEDICAL_DEVICE,
            mac_address="70:85:C2:55:66:77",
            actuation_bus=ActuationBus.IP_SOCKET,
        )
        res = hub.admit_device(med)
        assert res.assigned_tier == EdgeCapabilityTier.TIER3_AI_EDGE

    def test_custom_declared_envelope_preserved(self, hub: EdgeDeviceHub) -> None:
        """Custom safety bounds declared by device must not be overwritten by defaults."""
        custom_envelope = {"max_speed_kmh": 25.0, "special_zone": "warehouse_b"}
        profile = EdgeDeviceProfile(
            device_id="agv_kivabot",
            manufacturer="AmazonRobotics",
            model="Pegasus",
            device_type=DeviceType.AUTONOMOUS_VEHICLE,
            mac_address="10:BF:48:99:88:77",
            declared_safety_envelope=custom_envelope,
        )
        res = hub.admit_device(profile)
        assert res.admitted is True
        assert hub.get_device("agv_kivabot").declared_safety_envelope == custom_envelope

    def test_list_devices_and_filtering(self, hub: EdgeDeviceHub) -> None:
        """Listing devices returns all devices or filters by category."""
        hub.admit_device(
            EdgeDeviceProfile(
                device_id="d1", manufacturer="M1", model="X", device_type=DeviceType.ROBOT_6AXIS, mac_address="01:00:00:00:00:01"
            )
        )
        hub.admit_device(
            EdgeDeviceProfile(
                device_id="d2", manufacturer="M2", model="Y", device_type=DeviceType.UAV_DRONE, mac_address="01:00:00:00:00:02"
            )
        )
        hub.admit_device(
            EdgeDeviceProfile(
                device_id="d3", manufacturer="M3", model="Z", device_type=DeviceType.ROBOT_6AXIS, mac_address="01:00:00:00:00:03"
            )
        )

        all_devices = hub.list_devices()
        assert len(all_devices) == 3

        robots = hub.list_devices(device_type=DeviceType.ROBOT_6AXIS)
        assert len(robots) == 2
        assert {d.device_id for d in robots} == {"d1", "d3"}


class TestHeartbeatAndWatchdog:
    """Unit tests for watchdog monitoring and fail-closed cutoff."""

    def test_heartbeat_unknown_device(self, hub: EdgeDeviceHub) -> None:
        """Heartbeat for an unadmitted device must return False."""
        assert hub.heartbeat("non_existent_dev") is False

    def test_heartbeat_admitted_device(self, hub: EdgeDeviceHub) -> None:
        """Heartbeat for an admitted device updates tick and returns True."""
        profile = EdgeDeviceProfile(
            device_id="node_42",
            manufacturer="Bosch",
            model="Rexroth",
            device_type=DeviceType.GENERAL_ACTUATOR,
            mac_address="02:42:AC:11:00:02",
        )
        hub.admit_device(profile)
        assert hub.heartbeat("node_42") is True

    def test_watchdog_timeout_triggers_emergency_cutoff(self, hub: EdgeDeviceHub, fieldbus: IndustrialFieldbusInterlock) -> None:
        """Watchdog check detects stale heartbeat and trips physical fieldbus interlock."""
        profile = EdgeDeviceProfile(
            device_id="node_timeout",
            manufacturer="ABB",
            model="IRB 6700",
            device_type=DeviceType.ROBOT_6AXIS,
            mac_address="00:0A:F7:12:34:56",
            actuation_bus=ActuationBus.CAN_BUS,
        )
        hub.admit_device(profile)

        # Fresh tick: no timeout
        assert hub.check_watchdogs() == []
        assert fieldbus.is_interlocked is False

        # Simulate time passage beyond 100ms
        hub._last_heartbeat["node_timeout"] = time.perf_counter() - 0.2

        # Watchdog trigger
        timed_out = hub.check_watchdogs()
        assert "node_timeout" in timed_out
        assert fieldbus.is_interlocked is True
        assert "WATCHDOG_HEARTBEAT_TIMEOUT" in fieldbus.last_trip_reason


class TestEmergencyCutoffChannels:
    """Unit tests for physical cutoff dispatch across actuation buses."""

    def test_can_bus_cutoff(self, hub: EdgeDeviceHub, fieldbus: IndustrialFieldbusInterlock) -> None:
        """CAN_BUS cutoff dispatches EMCY frame and NMT stop."""
        profile = EdgeDeviceProfile(
            device_id="dev_can",
            manufacturer="KUKA",
            model="KR",
            device_type=DeviceType.ROBOT_6AXIS,
            mac_address="00:11:22:33:44:55",
            actuation_bus=ActuationBus.CAN_BUS,
        )
        hub.admit_device(profile)
        report = hub.trigger_emergency_cutoff("dev_can", reason="TEST_ESTOP")

        assert report["action"] == "EMERGENCY_CUTOFF_EXECUTED"
        assert report["bus"] == ActuationBus.CAN_BUS.value
        assert fieldbus.is_interlocked is True
        assert any(f.arbitration_id == 0x080 for f in fieldbus.can_frames_log)

    def test_modbus_tcp_cutoff(self, hub: EdgeDeviceHub, fieldbus: IndustrialFieldbusInterlock) -> None:
        """MODBUS_TCP cutoff writes coil zero and emergency register."""
        profile = EdgeDeviceProfile(
            device_id="dev_mb",
            manufacturer="Schneider",
            model="Modicon",
            device_type=DeviceType.SCADA_PLC,
            mac_address="00:22:33:44:55:66",
            actuation_bus=ActuationBus.MODBUS_TCP,
        )
        hub.admit_device(profile)
        report = hub.trigger_emergency_cutoff("dev_mb", reason="OVERPRESSURE")

        assert report["bus"] == ActuationBus.MODBUS_TCP.value
        assert any(cmd.value == 0x0000 for cmd in fieldbus.modbus_commands_log)
        assert any(cmd.value == 0xDEAD for cmd in fieldbus.modbus_commands_log)


class TestSocketServerDiscovery:
    """Integration tests for background socket server handling admission over TCP."""

    def test_socket_server_handshake_and_heartbeat(self, hub: EdgeDeviceHub) -> None:
        """Test TCP handshake protocol for dynamic device admission and heartbeats."""
        port = 18992
        hub.start_socket_server(host="127.0.0.1", port=port)
        time.sleep(0.05)  # Allow socket thread to bind

        try:
            # 1. Admit device via socket
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(("127.0.0.1", port))

            admission_req = {
                "action": "admit",
                "profile": {
                    "device_id": "socket_cobot_01",
                    "manufacturer": "Universal Robots",
                    "model": "UR10e",
                    "device_type": "ROBOT_6AXIS",
                    "mac_address": "00:04:4B:99:11:22",
                    "actuation_bus": "ETHERCAT_FSOE",
                },
            }
            client.sendall(json.dumps(admission_req).encode("utf-8"))
            resp_data = json.loads(client.recv(4096).decode("utf-8"))
            client.close()

            assert resp_data["status"] == "SUCCESS"
            adm = resp_data["admission"]
            assert adm["device_id"] == "socket_cobot_01"
            assert adm["assigned_tier"] == "TIER2_ROBOTICS"
            assert hub.get_device("socket_cobot_01") is not None

            # 2. Heartbeat via socket
            client_hb = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_hb.connect(("127.0.0.1", port))
            hb_req = {"action": "heartbeat", "device_id": "socket_cobot_01"}
            client_hb.sendall(json.dumps(hb_req).encode("utf-8"))
            hb_resp = json.loads(client_hb.recv(4096).decode("utf-8"))
            client_hb.close()

            assert hb_resp["status"] == "SUCCESS"

            # 3. Invalid action handling
            client_err = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_err.connect(("127.0.0.1", port))
            client_err.sendall(json.dumps({"action": "invalid_op"}).encode("utf-8"))
            err_resp = json.loads(client_err.recv(4096).decode("utf-8"))
            client_err.close()

            assert err_resp["status"] == "ERROR"
            assert "Unsupported action" in err_resp["message"]

        finally:
            hub.stop_socket_server()
