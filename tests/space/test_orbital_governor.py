# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for the Sovereign Orbital Governor (tests.space.test_orbital_governor)."""

from datetime import datetime, timezone
import pytest

from nethical.space.detectors import (
    BeamPointingCommand,
    BeamTargetType,
    SecondaryHazardObject,
)
from nethical.space.models import (
    ConstellationTopology,
    InterSatelliteLink,
    ISLLinkStatus,
    LinkBudget,
    OrbitalState,
    Vector3D,
)
from nethical.space.orbital_governor import (
    OrbitalAction,
    OrbitalGovernor,
    OrbitalGovernorConfig,
    OrbitalSafetyDecision,
)


class TestOrbitalGovernor:
    """Test suite for the thread-safe onboard satellite decision core."""

    def test_nominal_orbital_verification_loop(self) -> None:
        governor = OrbitalGovernor()
        state = OrbitalState(
            satellite_id="SAT_SOVEREIGN_01",
            position_eci_km=Vector3D(x=6878.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )
        decision: OrbitalSafetyDecision = governor.evaluate_telemetry_and_conjunction(
            current_state=state,
        )

        assert decision.allowed
        assert decision.action == OrbitalAction.EXECUTE_COMMAND
        assert decision.law_implicated == 21
        assert decision.merkle_proof_hash is not None
        assert len(decision.merkle_proof_hash) == 64
        assert decision.latency_us > 0.0

    def test_conjunction_triggers_avoidance_maneuver(self) -> None:
        governor = OrbitalGovernor()
        primary = OrbitalState(
            satellite_id="SAT_SOVEREIGN_01",
            position_eci_km=Vector3D(x=6878.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )
        debris = OrbitalState(
            satellite_id="DEBRIS_TRACK_44",
            position_eci_km=Vector3D(x=6878.02, y=0.01, z=0.0),  # Close proximity
            velocity_eci_kms=Vector3D(x=0.0, y=-7.6, z=0.0),
        )
        decision = governor.evaluate_telemetry_and_conjunction(
            current_state=primary,
            conjunction_target=debris,
        )

        assert not decision.allowed
        assert decision.action == OrbitalAction.EXECUTE_AVOIDANCE_BURN
        assert decision.recommended_delta_v_ms is not None
        assert decision.law_implicated == 21
        assert any("Critical Conjunction Detected" in r for r in decision.reasons)

    def test_secondary_debris_forces_safe_hold_under_law_22(self) -> None:
        governor = OrbitalGovernor()
        primary = OrbitalState(
            satellite_id="SAT_SOVEREIGN_01",
            position_eci_km=Vector3D(x=6878.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )
        debris = OrbitalState(
            satellite_id="DEBRIS_TRACK_A",
            position_eci_km=Vector3D(x=6878.02, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=-7.6, z=0.0),
        )
        secondary_catalog = [
            SecondaryHazardObject(
                object_id="DEBRIS_CLOUD_KOSMOS_FRAGMENT_1",
                position_eci_km=Vector3D(x=6878.1, y=0.0, z=0.0),
                velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
            )
        ]
        decision = governor.evaluate_telemetry_and_conjunction(
            current_state=primary,
            conjunction_target=debris,
            secondary_catalog=secondary_catalog,
        )

        assert not decision.allowed
        assert decision.action == OrbitalAction.ENTER_SAFE_HOLD
        assert decision.law_implicated == 22  # Prevention of secondary debris cascade
        assert governor.is_in_safe_hold

        # Re-evaluating while locked in safe hold stays locked
        subsequent = governor.evaluate_telemetry_and_conjunction(current_state=primary)
        assert not subsequent.allowed
        assert subsequent.action == OrbitalAction.ENTER_SAFE_HOLD

        # Release safe hold
        governor.release_safe_hold()
        assert not governor.is_in_safe_hold

    def test_gnss_spoofing_switches_to_celestial(self) -> None:
        governor = OrbitalGovernor()
        gnss_spoofed = OrbitalState(
            satellite_id="SAT_SOVEREIGN_01",
            position_eci_km=Vector3D(x=6900.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )
        celestial_truth = OrbitalState(
            satellite_id="SAT_SOVEREIGN_01",
            position_eci_km=Vector3D(x=6878.0, y=0.0, z=0.0),  # 22 km divergence
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )

        decision = governor.evaluate_telemetry_and_conjunction(
            current_state=gnss_spoofed,
            celestial_imu_state=celestial_truth,
        )

        assert not decision.allowed
        assert decision.action == OrbitalAction.SWITCH_TO_CELESTIAL_NAV
        assert any("Severe GNSS navigation solution spoofing" in r for r in decision.reasons)

    def test_rf_jamming_triggers_optical_isl_reroute(self) -> None:
        governor = OrbitalGovernor()
        state = OrbitalState(
            satellite_id="SAT_SOVEREIGN_01",
            position_eci_km=Vector3D(x=6878.0, y=0.0, z=0.0),
            velocity_eci_kms=Vector3D(x=0.0, y=7.6, z=0.0),
        )
        jammed_lb = LinkBudget(
            link_id="SAT_PRIMARY_RF_LINK",
            carrier_frequency_ghz=14.0,
            tx_power_dbw=10.0,
            tx_antenna_gain_dbi=30.0,
            rx_antenna_gain_dbi=35.0,
            slant_range_km=1000.0,
            jammer_power_at_rx_dbw=-60.0,
        )
        topology = ConstellationTopology(
            satellite_id="SAT_SOVEREIGN_01",
            inter_satellite_links={
                "SAT_NEIGHBOR_LASER": InterSatelliteLink(
                    target_satellite_id="SAT_NEIGHBOR_LASER",
                    link_type="LASER_OPTICAL",
                    range_km=1400.0,
                    azimuth_deg=45.0,
                    elevation_deg=0.0,
                    status=ISLLinkStatus.ACTIVE,
                    latency_ms=4.8,
                )
            },
            active_route_table={
                "GROUND_STATION_MAIN": ["SAT_PRIMARY_RF_LINK", "GROUND_STATION_MAIN"]
            },
        )

        decision = governor.evaluate_telemetry_and_conjunction(
            current_state=state,
            link_budget=jammed_lb,
            topology=topology,
        )

        assert not decision.allowed
        assert decision.action == OrbitalAction.SWITCH_TO_OPTICAL_ISL
        assert decision.rerouted_isl_hop == "SAT_NEIGHBOR_LASER"
        assert any("Autonomous laser ISL reroute" in r for r in decision.reasons)

    def test_beam_pointing_audit(self) -> None:
        governor = OrbitalGovernor()
        valid_cmd = BeamPointingCommand(
            beam_id="BEAM_GW",
            target_type=BeamTargetType.GROUND_STATION,
            target_lat=52.23,
            target_lon=21.01,
            slant_range_km=700.0,
            off_axis_angle_to_geo_arc_deg=18.0,
        )
        decision = governor.audit_beam_pointing(valid_cmd)
        assert decision.allowed
        assert decision.action == OrbitalAction.EXECUTE_COMMAND

        invalid_cmd = BeamPointingCommand(
            beam_id="BEAM_PROHIBITED",
            target_type=BeamTargetType.TERRESTRIAL_COMMUNITY,
            target_lat=-26.696,
            target_lon=116.637,  # Square Kilometre Array (SKA)
            slant_range_km=700.0,
        )
        veto_decision = governor.audit_beam_pointing(invalid_cmd)
        assert not veto_decision.allowed
        assert veto_decision.action == OrbitalAction.VETO_COMMAND
        assert veto_decision.law_implicated == 20  # Coexistence
