#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Example: Autonomous Drone BVLOS Safety Governance under NATO AEP-107 & EU 2019/947.

Demonstrates deterministic flight safety governance for uncrewed aerial systems (UAS)
operating Beyond Visual Line of Sight (BVLOS):
- 3D Geocaging (lateral containment and 120m AGL sovereign altitude ceiling)
- Detect and Avoid (DAA) airspace integration with ADS-B In collision avoidance
- Electronic warfare resilience: GNSS jamming / spoofing detection & dead reckoning
- Multi-stage failsafes: Autonomous Return-to-Home (RTH), Safe2Ditch landing, and FTS
- Universal Edge Device Admission Hub integration
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
        if hasattr(sys.stdout, "reconfigure"):
            getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            getattr(sys.stderr, "reconfigure")(encoding="utf-8", errors="replace")
    except Exception:
        pass

from nethical_edge import (
    EdgeDeviceHub,
    EdgeDeviceProfile,
    DeviceType,
    ActuationBus,
    DroneSafetyGovernor,
    DroneSafetyConfig,
    DroneSafetyDecision,
    DroneFlightState,
    FailsafeAction,
    DAATrafficAlert,
    DroneTelemetry,
    ADSBTrafficTarget,
)


def run_autonomous_drone_demo() -> None:
    """Run comprehensive autonomous drone safety governance demonstration."""
    print("=" * 80)
    print("🛸 NETHICAL EDGE — AUTONOMOUS DRONE BVLOS SAFETY (NATO AEP-107 / EU 2019/947)")
    print("=" * 80)

    # 1. Device Admission & Discovery Handshake
    hub = EdgeDeviceHub()
    profile = EdgeDeviceProfile(
        device_id="uas_skydio_x2d_09",
        manufacturer="Skydio Defense",
        model="X2D Autonomous UAS",
        device_type=DeviceType.UAV_DRONE,
        mac_address="00:1E:58:99:A2:10",
        ip_address="192.168.10.101",
        actuation_bus=ActuationBus.ROS2_DDS,
        declared_safety_envelope={
            "max_altitude_agl_m": 120.0,
            "max_ground_speed_mps": 18.0,
            "max_geocage_radius_m": 2500.0,
            "battery_rth_threshold_pct": 25.0,
        },
        has_hardware_watchdog=True,
    )

    admission = hub.admit_device(profile)
    print(f"\n[1] Dynamic Device Admission Handshake:")
    print(f"    - UAS Model: {profile.manufacturer} {profile.model} ({profile.device_id})")
    print(f"    - MAC / IP: {profile.mac_address} | {profile.ip_address}")
    print(f"    - Assigned Tier: {admission.assigned_tier.value}")
    print(f"    - Cutoff / Failsafe Channel: {admission.bound_cutoff_channel.value}")
    print(f"    - Merkle Proof Hash: {admission.merkle_proof_hash[:16]}...")

    # 2. Initialise Drone Safety Governor
    safety_config = DroneSafetyConfig(
        drone_id=profile.device_id,
        max_altitude_agl_m=120.0,
        max_geocage_radius_m=2000.0,
        home_latitude=52.2297,
        home_longitude=21.0122,
        battery_rth_threshold_pct=25.0,
        battery_forced_land_pct=12.0,
        daa_traffic_advisory_dist_m=1000.0,
        daa_resolution_advisory_dist_m=300.0,
        max_tolerable_jamming_ratio=0.7,
    )
    governor = DroneSafetyGovernor(config=safety_config)

    scenarios = [
        {
            "name": "Scenario 1: Nominal BVLOS Infrastructure Inspection Flight",
            "telemetry": DroneTelemetry(
                latitude=52.2310,
                longitude=21.0150,
                altitude_agl_m=85.0,     # Within 120m ceiling
                ground_speed_mps=12.0,
                heading_deg=45.0,
                battery_percentage=88.0,
                gps_satellites=16,
                gnss_jamming_indicator=0.05,
                imu_gyro_consistency_score=0.99,
                c2_link_quality=0.95,
                nearby_traffic=[],
            ),
        },
        {
            "name": "Scenario 2: Altitude Ceiling Breach Attempt (Clamping to 120m AGL)",
            "telemetry": DroneTelemetry(
                latitude=52.2350,
                longitude=21.0200,
                altitude_agl_m=142.0,    # > 120m ceiling breach!
                ground_speed_mps=14.0,
                heading_deg=45.0,
                battery_percentage=75.0,
                gps_satellites=15,
                gnss_jamming_indicator=0.08,
                imu_gyro_consistency_score=0.98,
                c2_link_quality=0.92,
                nearby_traffic=[],
            ),
        },
        {
            "name": "Scenario 3: Non-Cooperative Airspace Intruder (ADS-B DAA Resolution Advisory)",
            "telemetry": DroneTelemetry(
                latitude=52.2370,
                longitude=21.0250,
                altitude_agl_m=95.0,
                ground_speed_mps=14.0,
                heading_deg=90.0,
                battery_percentage=60.0,
                gps_satellites=14,
                gnss_jamming_indicator=0.10,
                imu_gyro_consistency_score=0.97,
                c2_link_quality=0.88,
                nearby_traffic=[
                    ADSBTrafficTarget(
                        icao_address="48D2A1",
                        callsign="CESSNA_172",
                        distance_meters=210.0,  # < 300m Resolution Advisory barrier!
                        altitude_relative_m=10.0,
                        bearing_deg=85.0,
                        closing_speed_mps=45.0,
                    )
                ],
            ),
        },
        {
            "name": "Scenario 4: Electronic Warfare GNSS Jamming Attack + Low Battery (Safe2Ditch)",
            "telemetry": DroneTelemetry(
                latitude=52.2400,
                longitude=21.0300,
                altitude_agl_m=60.0,
                ground_speed_mps=8.0,
                heading_deg=175.0,
                battery_percentage=11.5, # < 12% critical battery
                gps_satellites=4,        # Jamming degraded
                gnss_jamming_indicator=0.89, # Severe EW jamming!
                imu_gyro_consistency_score=0.95,
                c2_link_quality=0.40,
                nearby_traffic=[],
            ),
        },
    ]

    print("\n[2] Executing Real-Time BVLOS Flight Operations:")
    for step_num, sc in enumerate(scenarios, 1):
        print(f"\n---> {sc['name']}")
        decision = governor.evaluate_flight_step(telemetry=sc["telemetry"])

        status_icon = "🟢 NOMINAL" if decision.allowed else "🔴 INTERVENED"
        print(f"     Status: {status_icon} | Flight State: {decision.flight_state.value}")
        print(f"     DAA Airspace Alert: {decision.daa_alert.value}")
        print(f"     Failsafe Action: {decision.failsafe_action.value}")
        print(f"     Decision Latency: {decision.latency_us:.1f} µs")
        if decision.evasion_heading_delta_deg:
            print(f"     Evasive Vector Heading: {decision.evasion_heading_delta_deg:.0f} deg")
        for r in decision.reasons:
            print(f"     Rationale: {r}")
        print(f"     Law Binding: Fundamental Law {decision.law_implicated} | Merkle Hash: {decision.merkle_proof_hash[:16]}...")

    # 3. Emergency Flight Termination System (FTS)
    print("\n[3] Emergency Flight Termination System (FTS) Verification:")
    print("    - Simulating structural catastrophic rotor failure command...")
    governor.trigger_flight_termination(reason="STRUCTURAL_ROTOR_FAILURE_DETECTION")
    fts_decision = governor.evaluate_flight_step(telemetry=scenarios[0]["telemetry"])
    print(f"    - Post-Termination State: {fts_decision.flight_state.value}")
    print(f"    - FTS Action: {fts_decision.failsafe_action.value}")
    print(f"    - Rationale: {fts_decision.reasons[0]}")

    print("\n" + "=" * 80)
    print("✅ Autonomous Drone BVLOS Safety Verification Completed Successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run_autonomous_drone_demo()
