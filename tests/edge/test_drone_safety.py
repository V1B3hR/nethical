# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Autonomous Drone & UAV BVLOS Safety Governor (tests.edge.test_drone_safety).

Validates NATO AEP-107 and EU 2019/947 flight safety functions:
3D geocaging, 120m AGL ceiling, ADS-B Detect-and-Avoid (DAA),
GNSS anti-jamming, Safe2Ditch emergency landing, and Flight Termination System (FTS).
"""

from __future__ import annotations

import time
import pytest

from nethical.edge.drone_safety import (
    ADSBTrafficTarget,
    DAATrafficAlert,
    DroneFlightState,
    DroneSafetyConfig,
    DroneSafetyDecision,
    DroneSafetyGovernor,
    DroneTelemetry,
    FailsafeAction,
)


@pytest.fixture
def nominal_telemetry() -> DroneTelemetry:
    """Nominal UAS flight telemetry within legal and physical bounds."""
    return DroneTelemetry(
        latitude=52.2298,
        longitude=21.0125,
        altitude_agl_m=65.0,
        ground_speed_mps=12.0,
        heading_deg=90.0,
        battery_percentage=85.0,
        gps_satellites=16,
        gnss_jamming_indicator=0.05,
        imu_gyro_consistency_score=0.98,
        c2_link_quality=0.95,
        nearby_traffic=[],
    )


@pytest.fixture
def governor() -> DroneSafetyGovernor:
    """Standard UAV safety governor configured for Warsaw test envelope."""
    config = DroneSafetyConfig(
        drone_id="uas_skydio_test",
        max_altitude_agl_m=120.0,
        min_altitude_agl_m=3.0,
        max_geocage_radius_m=2000.0,
        home_latitude=52.2297,
        home_longitude=21.0122,
        battery_rth_threshold_pct=25.0,
        battery_forced_land_pct=12.0,
        daa_traffic_advisory_dist_m=1000.0,
        daa_resolution_advisory_dist_m=300.0,
        max_tolerable_jamming_ratio=0.7,
        min_imu_consistency=0.75,
    )
    return DroneSafetyGovernor(config=config)


class TestDroneNominalOperations:
    """Tests for nominal BVLOS mission execution."""

    def test_nominal_flight_step(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """Nominal flight parameters evaluate as allowed with Law 21 (Self-Correction/Mission)."""
        decision = governor.evaluate_flight_step(nominal_telemetry)
        assert decision.allowed is True
        assert decision.flight_state == DroneFlightState.IN_FLIGHT_MISSION
        assert decision.failsafe_action == FailsafeAction.NONE
        assert decision.daa_alert == DAATrafficAlert.CLEAR
        assert decision.law_implicated == 21
        assert len(decision.merkle_proof_hash) == 64


class TestAirspaceContainmentAndGeocaging:
    """Tests for 120m AGL ceiling and 3D lateral geocage enforcement."""

    def test_altitude_ceiling_breach_forces_hover(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """Altitude exceeding 120m AGL triggers failsafe HOVER and clamps climb."""
        breached_telemetry = nominal_telemetry.model_copy(update={"altitude_agl_m": 138.5})
        decision = governor.evaluate_flight_step(breached_telemetry)
        assert decision.allowed is False
        assert decision.failsafe_action == FailsafeAction.HOVER
        assert decision.flight_state == DroneFlightState.HOVERING
        assert any("Altitude Ceiling Breach: 138.5 m" in r for r in decision.reasons)

    def test_geocage_boundary_breach_forces_rth(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """Flight coordinates outside 2000m radial geocage trigger Return-To-Home."""
        # 52.2700, 21.0122 is ~4.5 km North of home (52.2297, 21.0122)
        out_of_bounds_telemetry = nominal_telemetry.model_copy(update={"latitude": 52.2700})
        decision = governor.evaluate_flight_step(out_of_bounds_telemetry)
        assert decision.allowed is False
        assert decision.failsafe_action == FailsafeAction.RETURN_TO_HOME
        assert decision.flight_state == DroneFlightState.RETURN_TO_HOME
        assert any("Geocage Hard Boundary Breached" in r for r in decision.reasons)


class TestDetectAndAvoidNATOAEP107:
    """Tests for ADS-B In airspace awareness and collision evasion vectors."""

    def test_daa_traffic_advisory(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """Traffic within 1000m triggers TRAFFIC_ADVISORY while remaining allowed."""
        intruder = ADSBTrafficTarget(
            icao_address="4840B2",
            callsign="LOT3801",
            distance_meters=850.0,
            altitude_relative_m=50.0,
            bearing_deg=45.0,
            closing_speed_mps=30.0,
        )
        traffic_telemetry = nominal_telemetry.model_copy(update={"nearby_traffic": [intruder]})
        decision = governor.evaluate_flight_step(traffic_telemetry)
        assert decision.allowed is True
        assert decision.daa_alert == DAATrafficAlert.TRAFFIC_ADVISORY
        assert any("DAA Traffic Advisory: Airspace intruder LOT3801" in r for r in decision.reasons)

    def test_daa_resolution_advisory_evasive_maneuver(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """Traffic within 300m triggers RESOLUTION_ADVISORY and evasion vector."""
        intruder = ADSBTrafficTarget(
            icao_address="4855C1",
            callsign="CESSNA_172",
            distance_meters=210.0,
            altitude_relative_m=-10.0,
            bearing_deg=90.0,
            closing_speed_mps=45.0,
        )
        critical_traffic = nominal_telemetry.model_copy(update={"nearby_traffic": [intruder]})
        decision = governor.evaluate_flight_step(critical_traffic)
        assert decision.allowed is False
        assert decision.daa_alert == DAATrafficAlert.RESOLUTION_ADVISORY
        assert decision.flight_state == DroneFlightState.AVOIDANCE_MANEUVER
        # Bearing 90 + 90 = 180 deg
        assert decision.evasion_heading_delta_deg == 180.0
        assert any("Executing immediate evasive vector to 180 deg" in r for r in decision.reasons)


class TestElectronicWarfareAndCyberMitigation:
    """Tests for GNSS jamming, sensor spoofing, and C2 loss."""

    def test_gnss_jamming_attack_triggers_safe2ditch(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """Jamming metric > 0.7 triggers dead reckoning and Safe2Ditch landing."""
        jammed_telemetry = nominal_telemetry.model_copy(update={"gnss_jamming_indicator": 0.88})
        decision = governor.evaluate_flight_step(jammed_telemetry)
        assert decision.allowed is False
        assert decision.failsafe_action == FailsafeAction.SAFE2DITCH
        assert decision.flight_state == DroneFlightState.SAFE2DITCH_LANDING
        assert any("Severe GNSS Jamming/Spoofing detected" in r for r in decision.reasons)

    def test_imu_gyro_inconsistency_triggers_safe2ditch(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """IMU consistency drop below 0.75 triggers Safe2Ditch."""
        compromised_telemetry = nominal_telemetry.model_copy(update={"imu_gyro_consistency_score": 0.62})
        decision = governor.evaluate_flight_step(compromised_telemetry)
        assert decision.allowed is False
        assert decision.failsafe_action == FailsafeAction.SAFE2DITCH
        assert any("IMU sensor inconsistency" in r for r in decision.reasons)

    def test_c2_link_loss_triggers_rth(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """Complete loss of C2 link triggers autonomous Return-To-Home."""
        lost_c2_telemetry = nominal_telemetry.model_copy(update={"c2_link_quality": 0.02})
        decision = governor.evaluate_flight_step(lost_c2_telemetry)
        assert decision.allowed is False
        assert decision.failsafe_action == FailsafeAction.RETURN_TO_HOME
        assert any("Command & Control (C2) link lost" in r for r in decision.reasons)


class TestBatteryReserveFailsafes:
    """Tests for multi-stage battery exhaustion failsafes."""

    def test_battery_low_triggers_rth(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """Battery dropping to 22% (<= 25%) triggers RTH."""
        low_bat = nominal_telemetry.model_copy(update={"battery_percentage": 22.0})
        decision = governor.evaluate_flight_step(low_bat)
        assert decision.allowed is False
        assert decision.failsafe_action == FailsafeAction.RETURN_TO_HOME
        assert decision.flight_state == DroneFlightState.RETURN_TO_HOME

    def test_battery_critical_triggers_immediate_safe2ditch(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """Battery dropping to 9% (<= 12%) triggers immediate Safe2Ditch landing."""
        crit_bat = nominal_telemetry.model_copy(update={"battery_percentage": 9.5})
        decision = governor.evaluate_flight_step(crit_bat)
        assert decision.allowed is False
        assert decision.failsafe_action == FailsafeAction.SAFE2DITCH
        assert decision.flight_state == DroneFlightState.SAFE2DITCH_LANDING


class TestFlightTerminationSystem:
    """Tests for independent Flight Termination System (FTS)."""

    def test_fts_deployment_cuts_power_and_stops_all_flight(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """Triggering FTS latches TERMINATED state and halts further operations."""
        governor.trigger_flight_termination(reason="MANUAL_ABORT_SAFETY_RANGE")

        decision = governor.evaluate_flight_step(nominal_telemetry)
        assert decision.allowed is False
        assert decision.flight_state == DroneFlightState.TERMINATED
        assert decision.failsafe_action == FailsafeAction.FLIGHT_TERMINATION_SYSTEM
        assert any("FTS Activated: Motor power severed" in r for r in decision.reasons)


class TestDroneRealTimeLatency:
    """Benchmark tests to verify deterministic sub-50 µs evaluation latency."""

    def test_sub_50_microsecond_evaluation_latency(
        self,
        governor: DroneSafetyGovernor,
        nominal_telemetry: DroneTelemetry,
    ) -> None:
        """Verify that UAV safety evaluation completes well within the 50 µs deadline."""
        # Warmup
        for _ in range(20):
            governor.evaluate_flight_step(nominal_telemetry)

        latencies = []
        for _ in range(100):
            dec = governor.evaluate_flight_step(nominal_telemetry)
            latencies.append(dec.latency_us)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 50.0, f"Average latency {avg_latency:.2f} µs exceeds 50 µs target"
