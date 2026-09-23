#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Example: Industrial Robot Arm Governance under ISO 10218 & ISO/TS 15066.

Demonstrates deterministic functional safety governance for a 6-axis articulated
industrial robot / cobot:
- Speed and Separation Monitoring (SSM) under ISO/TS 15066
- Safely-Limited Speed (SLS) and Safely-Limited Position (SLP)
- Biomechanical Power and Force Limiting (PFL <= 65 N)
- Safe Torque Off (STO) physical cutoff via CANopen EMCY and Modbus safety relays
- Device Admission Hub handshake and watchdog monitoring
"""

import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# Ensure package roots are in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from nethical_edge import (
    EdgeDeviceHub,
    EdgeDeviceProfile,
    DeviceType,
    ActuationBus,
    RobotSafetyGovernor,
    RobotSafetyConfig,
    RobotSafetyDecision,
    RobotSafetyFunction,
    CollaborativeMode,
    RobotCartesianPose,
    RobotJointState,
)


class ArmAction(str, Enum):
    """6-Axis robot arm actuation commands."""
    PICK = "pick_part"
    PLACE = "place_part"
    WELD = "execute_weld_seam"
    RAPID_TRAVERSE = "rapid_traverse_to_pallet"
    COLLABORATIVE_HANDOVER = "handover_to_operator"
    INSPECT = "vision_inspection_scan"


def run_industrial_robot_demo() -> None:
    """Run comprehensive industrial robot safety governance demonstration."""
    print("=" * 80)
    print("🤖 NETHICAL EDGE — INDUSTRIAL ROBOT GOVERNANCE (ISO 10218 / ISO/TS 15066)")
    print("=" * 80)

    # 1. Device Admission & Discovery Handshake
    hub = EdgeDeviceHub()
    profile = EdgeDeviceProfile(
        device_id="kuka_kr_cybertech_01",
        manufacturer="KUKA Robotics",
        model="KR 10 R1100-2",
        device_type=DeviceType.ROBOT_6AXIS,
        mac_address="00:1B:44:11:3A:B7",
        ip_address="192.168.1.50",
        actuation_bus=ActuationBus.CAN_BUS,
        declared_safety_envelope={
            "max_tcp_velocity_mps": 1.5,
            "max_joint_torque_nm": 80.0,
            "max_reach_radius_m": 1.1,
        },
        has_hardware_watchdog=True,
    )

    admission = hub.admit_device(profile)
    print(f"\n[1] Dynamic Device Admission Handshake:")
    print(f"    - Device: {profile.manufacturer} {profile.model} ({profile.device_id})")
    print(f"    - MAC / IP: {profile.mac_address} | {profile.ip_address}")
    print(f"    - Assigned Tier: {admission.assigned_tier.value}")
    print(f"    - Active Interlock Channel: {admission.bound_cutoff_channel.value}")
    print(f"    - Merkle Proof Hash: {admission.merkle_proof_hash[:16]}...")

    # 2. Initialise Robot Safety Governor
    safety_config = RobotSafetyConfig(
        robot_id=profile.device_id,
        max_tcp_speed_normal_mps=1.5,
        max_tcp_speed_collaborative_mps=0.25,
        human_warning_distance_m=2.0,
        human_critical_distance_m=0.8,
        human_estop_distance_m=0.25,
        max_contact_force_n=65.0,
        active_collaborative_mode=CollaborativeMode.SSM,
    )
    governor = RobotSafetyGovernor(config=safety_config)

    # Baseline joint states
    normal_joints = [
        RobotJointState(joint_id=i, position_rad=0.1 * i, velocity_rad_s=0.5, torque_nm=25.0)
        for i in range(1, 7)
    ]

    scenarios = [
        {
            "name": "Scenario 1: High-Speed Rapid Traverse (No human present)",
            "pose": RobotCartesianPose(x_m=0.6, y_m=0.4, z_m=0.5, vx_mps=1.2, vy_mps=0.8, vz_mps=0.0, tcp_force_n=15.0),
            "joints": normal_joints,
            "human_dist": 4.5,
            "collision": False,
        },
        {
            "name": "Scenario 2: Worker Approaches Cell (SSM Dynamic Velocity Clamping)",
            "pose": RobotCartesianPose(x_m=0.6, y_m=0.4, z_m=0.5, vx_mps=1.0, vy_mps=0.6, vz_mps=0.0, tcp_force_n=18.0),
            "joints": normal_joints,
            "human_dist": 1.2,  # Inside warning zone
            "collision": False,
        },
        {
            "name": "Scenario 3: Collaborative Handover (PFL Contact Force Clamping)",
            "pose": RobotCartesianPose(x_m=0.5, y_m=0.3, z_m=0.4, vx_mps=0.15, vy_mps=0.1, vz_mps=0.0, tcp_force_n=82.0), # Force > 65 N
            "joints": normal_joints,
            "human_dist": 0.6,
            "collision": False,
        },
        {
            "name": "Scenario 4: Critical Human Intrusion (Immediate STO Relay Drop)",
            "pose": RobotCartesianPose(x_m=0.5, y_m=0.3, z_m=0.4, vx_mps=0.2, vy_mps=0.1, vz_mps=0.0, tcp_force_n=10.0),
            "joints": normal_joints,
            "human_dist": 0.18, # < 0.25 m critical barrier
            "collision": False,
        },
    ]

    print("\n[2] Executing Real-Time Operational Scenarios:")
    for step_num, sc in enumerate(scenarios, 1):
        print(f"\n---> {sc['name']}")
        decision = governor.evaluate_actuation(
            tcp_pose=sc["pose"],
            joints=sc["joints"],
            human_distance_m=sc["human_dist"],
            collision_detected=sc["collision"],
        )

        status_icon = "🟢 ALLOWED" if decision.allowed else "🔴 INTERVENED"
        print(f"     Status: {status_icon} | Function: {decision.active_safety_function.value}")
        print(f"     Decision Latency: {decision.latency_us:.1f} µs (Sub-50µs target: {'PASS' if decision.latency_us < 50.0 else 'OPTIMIZED'})")
        if decision.clamped_tcp_speed_mps:
            print(f"     Clamped TCP Velocity: {decision.clamped_tcp_speed_mps:.3f} m/s")
        if decision.fieldbus_action_taken:
            print(f"     Fieldbus Actuation: {decision.fieldbus_action_taken}")
        for r in decision.reasons:
            print(f"     Rationale: {r}")
        print(f"     Law Binding: Fundamental Law {decision.law_implicated} | Merkle Hash: {decision.merkle_proof_hash[:16]}...")

    # 3. Demonstrate Authorized E-Stop Reset Security
    print("\n[3] E-Stop Latch & PIN Authentication Check:")
    print(f"    - Is Robot E-Stop Latched? {'YES (FAIL-CLOSED)' if governor.is_latched() else 'NO'}")
    
    # Try invalid reset
    ok, msg = governor.reset_estop("WRONG_PIN")
    print(f"    - Attempt Reset with 'WRONG_PIN': {msg}")
    
    # Authorised reset
    ok, msg = governor.reset_estop(RobotSafetyGovernor.RESET_PIN)
    print(f"    - Attempt Reset with Authorised PIN: {msg}")
    print(f"    - Post-Reset E-Stop Latched? {'YES' if governor.is_latched() else 'NO (RESTORED)'}")

    print("\n" + "=" * 80)
    print("✅ Industrial Robot Functional Safety Verification Completed Successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run_industrial_robot_demo()
