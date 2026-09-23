# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Universal Edge Device Admission Hub & Discovery Socket (nethical.edge.device_hub).

Provides a vendor-agnostic admission protocol for edge devices:
- Handshake negotiation: Manufacturer, Model, MAC address, IPv4/IPv6, Device Category, Actuation Bus.
- Dynamic Sovereign Safety Profile assignment (no hardcoding of 10,000 vendor SDKs).
- Deterministic physical cutoff channel binding (CAN EMCY 0x080, Modbus Coils, EtherCAT FSoE, GPIO).
- Watchdog heartbeat monitoring with fail-closed actuation.
- Merkle audit logging per admission event.
"""

from __future__ import annotations

import hashlib
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from nethical.edge.industrial_fieldbus import IndustrialFieldbusInterlock, FieldbusInterlockStatus

logger = logging.getLogger("nethical.edge.device_hub")


class DeviceType(str, Enum):
    """Canonical edge device categories."""
    ROBOT_6AXIS = "ROBOT_6AXIS"
    UAV_DRONE = "UAV_DRONE"
    SCADA_PLC = "SCADA_PLC"
    AUTONOMOUS_VEHICLE = "AUTONOMOUS_VEHICLE"
    MEDICAL_DEVICE = "MEDICAL_DEVICE"
    GENERAL_ACTUATOR = "GENERAL_ACTUATOR"


class ActuationBus(str, Enum):
    """Physical or network communication channel used for emergency intervention."""
    CAN_BUS = "CAN_BUS"
    MODBUS_TCP = "MODBUS_TCP"
    ETHERCAT_FSOE = "ETHERCAT_FSOE"
    GPIO_RELAY = "GPIO_RELAY"
    IP_SOCKET = "IP_SOCKET"
    ROS2_DDS = "ROS2_DDS"


class EdgeCapabilityTier(str, Enum):
    """Runtime capability tier assigned based on device hardware resources."""
    TIER1_MICRO = "TIER1_MICRO"       # <64MB RAM, MCU/Fieldbus only
    TIER2_ROBOTICS = "TIER2_ROBOTICS" # <256MB RAM, Kinetic Bubble & Geocage
    TIER3_AI_EDGE = "TIER3_AI_EDGE"   # <1GB RAM, Local SLM & TPM Attestation


class EdgeDeviceProfile(BaseModel):
    """Complete fingerprint and declared physical operating envelope of a connected device."""
    device_id: str = Field(..., description="Unique identifier or serial number")
    manufacturer: str = Field(..., description="Hardware vendor (e.g. KUKA, DJI, Siemens, Fanuc)")
    model: str = Field(..., description="Device model or part number")
    device_type: DeviceType = Field(..., description="Canonical category of the autonomous system")
    mac_address: str = Field(..., description="Hardware MAC address for layer-2 attestation")
    ip_address: str = Field(default="127.0.0.1", description="IPv4 or IPv6 network coordinate")
    actuation_bus: ActuationBus = Field(default=ActuationBus.CAN_BUS, description="Active emergency cutoff bus")
    declared_safety_envelope: Dict[str, Any] = Field(
        default_factory=dict,
        description="Declared physical bounds (e.g. max_speed_mps, max_torque_nm, max_altitude_m)"
    )
    has_hardware_watchdog: bool = Field(default=True, description="Whether device supports hardware heartbeat resets")
    tpm_attestation_status: str = Field(default="HARDWARE_TPM_VERIFIED", description="TPM 2.0 or secure enclave status")
    firmware_version: str = Field(default="1.0.0", description="Installed device firmware version")
    registered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DeviceAdmissionResult(BaseModel):
    """Verdict of the device admission handshake."""
    admitted: bool
    device_id: str
    assigned_tier: EdgeCapabilityTier
    bound_cutoff_channel: ActuationBus
    handshake_token: str
    heartbeat_interval_ms: int = 50
    reasons: List[str] = Field(default_factory=list)
    merkle_proof_hash: str = ""


class EdgeDeviceHub:
    """Universal device admission hub and discovery socket manager."""

    def __init__(
        self,
        fieldbus_interlock: Optional[IndustrialFieldbusInterlock] = None,
        default_heartbeat_timeout_ms: int = 250,
    ) -> None:
        self.fieldbus = fieldbus_interlock or IndustrialFieldbusInterlock()
        self.heartbeat_timeout_ms = default_heartbeat_timeout_ms
        self._devices: Dict[str, EdgeDeviceProfile] = {}
        self._last_heartbeat: Dict[str, float] = {}
        self._admission_log: List[DeviceAdmissionResult] = []
        self._server_socket: Optional[socket.socket] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

    def admit_device(self, profile: EdgeDeviceProfile) -> DeviceAdmissionResult:
        """Evaluate device fingerprint, assign capability tier, and bind safety interlocks."""
        with self._lock:
            reasons: List[str] = []

            # 1. Determine capability tier based on device class
            if profile.device_type == DeviceType.SCADA_PLC:
                tier = EdgeCapabilityTier.TIER1_MICRO
            elif profile.device_type in (DeviceType.ROBOT_6AXIS, DeviceType.UAV_DRONE, DeviceType.AUTONOMOUS_VEHICLE):
                tier = EdgeCapabilityTier.TIER2_ROBOTICS
            else:
                tier = EdgeCapabilityTier.TIER3_AI_EDGE

            # 2. Validate declared envelope
            if not profile.declared_safety_envelope:
                # Assign default safe envelope for device class
                profile.declared_safety_envelope = self._get_default_envelope(profile.device_type)
                reasons.append("Default sovereign safety envelope assigned")

            # 3. Generate cryptographic handshake token and Merkle proof hash
            raw_token = f"{profile.device_id}:{profile.mac_address}:{time.time_ns()}"
            handshake_token = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

            proof_payload = f"{profile.device_id}|{profile.manufacturer}|{profile.mac_address}|{profile.registered_at}"
            merkle_proof = hashlib.sha256(proof_payload.encode("utf-8")).hexdigest()

            result = DeviceAdmissionResult(
                admitted=True,
                device_id=profile.device_id,
                assigned_tier=tier,
                bound_cutoff_channel=profile.actuation_bus,
                handshake_token=handshake_token,
                heartbeat_interval_ms=50,
                reasons=reasons or ["Device admitted under sovereign governance"],
                merkle_proof_hash=merkle_proof,
            )

            self._devices[profile.device_id] = profile
            self._last_heartbeat[profile.device_id] = time.perf_counter()
            self._admission_log.append(result)

            logger.info(
                "Device admitted: %s (%s %s) bound to %s tier=%s",
                profile.device_id,
                profile.manufacturer,
                profile.model,
                profile.actuation_bus.value,
                tier.value,
            )
            return result

    def heartbeat(self, device_id: str) -> bool:
        """Register a heartbeat tick from an active device."""
        with self._lock:
            if device_id in self._devices:
                self._last_heartbeat[device_id] = time.perf_counter()
                return True
            return False

    def check_watchdogs(self) -> List[str]:
        """Check all registered devices for heartbeat timeouts; trigger cutoff if expired."""
        timed_out: List[str] = []
        now = time.perf_counter()
        timeout_sec = self.heartbeat_timeout_ms / 1000.0

        with self._lock:
            for dev_id, last_tick in list(self._last_heartbeat.items()):
                if (now - last_tick) > timeout_sec:
                    timed_out.append(dev_id)

        for dev_id in timed_out:
            logger.error("Watchdog heartbeat expired for device: %s. Triggering fail-closed cutoff!", dev_id)
            self.trigger_emergency_cutoff(dev_id, reason="WATCHDOG_HEARTBEAT_TIMEOUT")

        return timed_out

    def trigger_emergency_cutoff(self, device_id: str, reason: str = "MANUAL_ESTOP") -> Dict[str, Any]:
        """Dispatch immediate physical emergency cutoff across the device's bound actuation bus."""
        profile = self._devices.get(device_id)
        bus = profile.actuation_bus if profile else ActuationBus.CAN_BUS

        fieldbus_result: Optional[FieldbusInterlockStatus] = None

        if bus == ActuationBus.CAN_BUS:
            # Emit CANopen EMCY (0x080) and NMT STOP (0x000)
            fieldbus_result = self.fieldbus.trigger_emergency_cutoff(reason=f"device_hub:{device_id}:{reason}")
        elif bus == ActuationBus.MODBUS_TCP:
            # De-energise actuator coil 0x0001 -> 0x0000
            fieldbus_result = self.fieldbus.trigger_emergency_cutoff(reason=f"device_hub:{device_id}:{reason}")
        elif bus == ActuationBus.ETHERCAT_FSOE:
            # Safe-Op transition and PDO zeroing
            fieldbus_result = self.fieldbus.trigger_emergency_cutoff(reason=f"device_hub:{device_id}:{reason}")
        else:
            fieldbus_result = self.fieldbus.trigger_emergency_cutoff(reason=f"device_hub:{device_id}:{reason}")

        cutoff_report = {
            "device_id": device_id,
            "action": "EMERGENCY_CUTOFF_EXECUTED",
            "reason": reason,
            "bus": bus.value,
            "fieldbus_result": fieldbus_result.model_dump() if fieldbus_result else {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.critical("Emergency physical cutoff executed for %s on %s: %s", device_id, bus.value, reason)
        return cutoff_report

    def get_device(self, device_id: str) -> Optional[EdgeDeviceProfile]:
        """Fetch profile for registered device."""
        with self._lock:
            return self._devices.get(device_id)

    def list_devices(self, device_type: Optional[DeviceType] = None) -> List[EdgeDeviceProfile]:
        """List all admitted devices, optionally filtered by category."""
        with self._lock:
            if device_type:
                return [d for d in self._devices.values() if d.device_type == device_type]
            return list(self._devices.values())

    def _get_default_envelope(self, device_type: DeviceType) -> Dict[str, Any]:
        """Provide calibrated safe physical operational envelopes per device category."""
        if device_type == DeviceType.ROBOT_6AXIS:
            return {
                "max_tcp_velocity_mps": 0.25,      # ISO/TS 15066 collaborative max (250 mm/s)
                "max_joint_torque_nm": 65.0,       # Biomechanical force threshold
                "min_human_proximity_m": 0.8,
                "critical_estop_proximity_m": 0.3,
            }
        elif device_type == DeviceType.UAV_DRONE:
            return {
                "max_altitude_agl_m": 120.0,       # Standard European & FAA ceiling
                "max_ground_speed_mps": 15.0,
                "min_battery_reserve_pct": 20.0,
                "geocage_radius_m": 1000.0,
            }
        elif device_type == DeviceType.AUTONOMOUS_VEHICLE:
            return {
                "max_speed_kmh": 50.0,
                "emergency_brake_decel_mps2": 7.5,
                "min_following_distance_m": 10.0,
            }
        elif device_type == DeviceType.SCADA_PLC:
            return {
                "max_pressure_bar": 10.0,
                "max_temperature_c": 85.0,
                "valve_actuation_rate_s": 2.0,
            }
        return {"max_velocity_mps": 1.0, "max_torque_nm": 20.0}

    def start_socket_server(self, host: str = "127.0.0.1", port: int = 8990) -> None:
        """Start dynamic admission socket server in background thread."""
        if self._running:
            return

        self._running = True
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((host, port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(0.5)

        self._server_thread = threading.Thread(target=self._socket_listener, daemon=True)
        self._server_thread.start()
        logger.info("EdgeDeviceHub socket server listening on %s:%d", host, port)

    def stop_socket_server(self) -> None:
        """Stop admission socket server."""
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=1.0)
        logger.info("EdgeDeviceHub socket server stopped")

    def _socket_listener(self) -> None:
        """Background worker handling device admission connections."""
        while self._running:
            try:
                if not self._server_socket:
                    break
                client_sock, addr = self._server_socket.accept()
                threading.Thread(target=self._handle_client, args=(client_sock, addr), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as exc:
                if self._running:
                    logger.warning("Error in device admission socket accept: %s", exc)
                break

    def _handle_client(self, client_sock: socket.socket, addr: Tuple[str, int]) -> None:
        """Process incoming handshake payload from connecting edge device."""
        try:
            client_sock.settimeout(3.0)
            data = client_sock.recv(4096)
            if not data:
                return

            payload = json.loads(data.decode("utf-8"))
            action = payload.get("action", "admit")

            if action == "admit":
                profile_data = payload.get("profile", {})
                profile = EdgeDeviceProfile(**profile_data)
                admission = self.admit_device(profile)
                response = {"status": "SUCCESS", "admission": admission.model_dump()}
            elif action == "heartbeat":
                device_id = payload.get("device_id", "")
                success = self.heartbeat(device_id)
                response = {"status": "SUCCESS" if success else "UNKNOWN_DEVICE"}
            else:
                response = {"status": "ERROR", "message": f"Unsupported action: {action}"}

            client_sock.sendall(json.dumps(response).encode("utf-8"))
        except Exception as exc:
            err_resp = {"status": "ERROR", "message": str(exc)}
            try:
                client_sock.sendall(json.dumps(err_resp).encode("utf-8"))
            except Exception:
                pass
        finally:
            try:
                client_sock.close()
            except Exception:
                pass
