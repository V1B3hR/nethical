# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Industrial Robot & Cobot Safety Governor (tests.edge.test_robot_safety).

Validates ISO 10218-1/2 functional safety functions (STO, SS1, SS2, SLS, SLP, SLT)
and ISO/TS 15066 collaborative modes (SRMS, PFL, SSM) with sub-50 µs cutoff.
"""

from __future__ import annotations

import time
import pytest

from nethical.edge.industrial_fieldbus import IndustrialFieldbusInterlock
from nethical.edge.robot_safety import (
    BODY_REGION_FORCE_LIMITS,
    BodyRegion,
    CollaborativeMode,
    RobotCartesianPose,
    RobotJointState,
    RobotSafetyConfig,
    RobotSafetyDecision,
    RobotSafetyFunction,
    RobotSafetyGovernor,
)


@pytest.fixture
def fieldbus() -> IndustrialFieldbusInterlock:
    """Provide a fresh industrial fieldbus interlock instance."""
    return IndustrialFieldbusInterlock()


@pytest.fixture
def nominal_pose() -> RobotCartesianPose:
    """Safe, centered Cartesian TCP pose within table workspace."""
    return RobotCartesianPose(x_m=0.4, y_m=0.3, z_m=0.5, vx_mps=0.2, vy_mps=0.1, vz_mps=0.0, tcp_force_n=15.0)


@pytest.fixture
def nominal_joints() -> list[RobotJointState]:
    """Safe 6-axis joint states well within torque limits."""
    return [
        RobotJointState(joint_id=1, position_rad=0.0, velocity_rad_s=0.5, torque_nm=25.0),
        RobotJointState(joint_id=2, position_rad=0.5, velocity_rad_s=0.4, torque_nm=30.0),
        RobotJointState(joint_id=3, position_rad=-0.5, velocity_rad_s=0.3, torque_nm=20.0),
        RobotJointState(joint_id=4, position_rad=0.0, velocity_rad_s=0.2, torque_nm=15.0),
        RobotJointState(joint_id=5, position_rad=0.2, velocity_rad_s=0.1, torque_nm=10.0),
        RobotJointState(joint_id=6, position_rad=0.0, velocity_rad_s=0.1, torque_nm=5.0),
    ]


@pytest.fixture
def governor(fieldbus: IndustrialFieldbusInterlock) -> RobotSafetyGovernor:
    """Provide a standard RobotSafetyGovernor configured for SSM collaborative mode."""
    config = RobotSafetyConfig(
        robot_id="kuka_iiwa_test",
        max_tcp_speed_normal_mps=1.5,
        max_tcp_speed_collaborative_mps=0.25,
        human_warning_distance_m=2.0,
        human_critical_distance_m=0.8,
        human_estop_distance_m=0.25,
        max_contact_force_n=65.0,
        max_joint_torque_nm=80.0,
        min_z_m=0.0,
        max_reach_radius_m=1.2,
        active_collaborative_mode=CollaborativeMode.SSM,
    )
    return RobotSafetyGovernor(config=config, fieldbus_interlock=fieldbus)


class TestRobotNominalSafety:
    """Tests for nominal operation within the verified safety envelope."""

    def test_nominal_evaluation_allowed(
        self,
        governor: RobotSafetyGovernor,
        nominal_pose: RobotCartesianPose,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """Nominal parameters with human far away should yield ALLOW and law 23."""
        decision = governor.evaluate_actuation(
            tcp_pose=nominal_pose,
            joints=nominal_joints,
            human_distance_m=3.5,
            collision_detected=False,
        )
        assert decision.allowed is True
        assert decision.active_safety_function == RobotSafetyFunction.NONE
        assert decision.clamped_tcp_speed_mps is None
        assert decision.applied_torque_clamped_nm is None
        assert len(decision.merkle_proof_hash) == 64
        assert decision.law_implicated == 23


class TestReflexCollisionAndSTO:
    """Tests for ISO 10218 Safe Torque Off and reflex impact response."""

    def test_direct_collision_triggers_sto(
        self,
        governor: RobotSafetyGovernor,
        fieldbus: IndustrialFieldbusInterlock,
        nominal_pose: RobotCartesianPose,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """Reflex collision trigger must instantly trip fieldbus interlock and latch STO."""
        decision = governor.evaluate_actuation(
            tcp_pose=nominal_pose,
            joints=nominal_joints,
            human_distance_m=1.5,
            collision_detected=True,
        )
        assert decision.allowed is False
        assert decision.active_safety_function == RobotSafetyFunction.STO
        assert decision.fieldbus_action_taken == "EMCY_0x080_STO_RELAY_OPEN"
        assert decision.law_implicated == 1
        assert governor.is_latched() is True
        assert fieldbus.is_interlocked is True

    def test_excessive_tcp_force_triggers_sto(
        self,
        governor: RobotSafetyGovernor,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """Excessive force (over 1.5x max contact force = 97.5 N) triggers STO reflex."""
        high_force_pose = RobotCartesianPose(
            x_m=0.5, y_m=0.5, z_m=0.4, tcp_force_n=110.0
        )
        decision = governor.evaluate_actuation(
            tcp_pose=high_force_pose,
            joints=nominal_joints,
            human_distance_m=2.0,
            collision_detected=False,
        )
        assert decision.allowed is False
        assert decision.active_safety_function == RobotSafetyFunction.STO
        assert governor.is_latched() is True

    def test_critical_proximity_barrier_triggers_sto(
        self,
        governor: RobotSafetyGovernor,
        nominal_pose: RobotCartesianPose,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """Human penetrating 0.25m envelope triggers immediate latched STO."""
        decision = governor.evaluate_actuation(
            tcp_pose=nominal_pose,
            joints=nominal_joints,
            human_distance_m=0.15,
            collision_detected=False,
        )
        assert decision.allowed is False
        assert decision.active_safety_function == RobotSafetyFunction.STO
        assert governor.is_latched() is True


class TestSpatialAndTorqueLimits:
    """Tests for SLP (Safely-Limited Position) and SLT (Safely-Limited Torque)."""

    def test_radial_work_envelope_breach_ss1(
        self,
        governor: RobotSafetyGovernor,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """TCP exceeding radial reach envelope (1.2m) engages SS1 deceleration."""
        out_of_bounds_pose = RobotCartesianPose(x_m=1.0, y_m=1.0, z_m=0.5)  # r = sqrt(2) = 1.414m > 1.2m
        decision = governor.evaluate_actuation(
            tcp_pose=out_of_bounds_pose,
            joints=nominal_joints,
            human_distance_m=2.5,
        )
        assert decision.allowed is False
        assert decision.active_safety_function == RobotSafetyFunction.SS1
        assert any("SLP Breach: Radial reach" in r for r in decision.reasons)

    def test_table_plane_penetration_ss1(
        self,
        governor: RobotSafetyGovernor,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """TCP penetrating table level (z < 0.0m) engages SS1 stop."""
        penetrating_pose = RobotCartesianPose(x_m=0.4, y_m=0.3, z_m=-0.05)
        decision = governor.evaluate_actuation(
            tcp_pose=penetrating_pose,
            joints=nominal_joints,
            human_distance_m=2.5,
        )
        assert decision.allowed is False
        assert decision.active_safety_function == RobotSafetyFunction.SS1
        assert any("penetrates table plane" in r for r in decision.reasons)

    def test_joint_torque_ceiling_slt(
        self,
        governor: RobotSafetyGovernor,
        nominal_pose: RobotCartesianPose,
    ) -> None:
        """Joint torque exceeding 80 Nm triggers SLT clamping."""
        overtorque_joints = [
            RobotJointState(joint_id=1, position_rad=0.0, velocity_rad_s=0.5, torque_nm=25.0),
            RobotJointState(joint_id=2, position_rad=0.5, velocity_rad_s=0.4, torque_nm=-92.0),  # Exceeds 80 Nm
        ]
        decision = governor.evaluate_actuation(
            tcp_pose=nominal_pose,
            joints=overtorque_joints,
            human_distance_m=2.5,
        )
        assert decision.allowed is False
        assert decision.active_safety_function == RobotSafetyFunction.SLT
        assert decision.applied_torque_clamped_nm == -80.0
        assert any("SLT Breach: Joint 2 torque -92.0 Nm" in r for r in decision.reasons)


class TestCollaborativeModesAndPFL:
    """Tests for ISO/TS 15066 collaborative modes (SSM, PFL, SRMS)."""

    def test_ssm_speed_clamping_critical_proximity(
        self,
        governor: RobotSafetyGovernor,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """Human within 0.8m triggers SSM speed clamp to 250 mm/s (0.25 m/s)."""
        fast_pose = RobotCartesianPose(x_m=0.4, y_m=0.3, z_m=0.5, vx_mps=0.8, vy_mps=0.6, vz_mps=0.0)  # speed = 1.0 m/s
        decision = governor.evaluate_actuation(
            tcp_pose=fast_pose,
            joints=nominal_joints,
            human_distance_m=0.6,  # Inside critical distance
        )
        assert decision.active_safety_function == RobotSafetyFunction.SLS
        assert decision.clamped_tcp_speed_mps == 0.25
        assert any("SSM/PFL Active" in r for r in decision.reasons)

    def test_srms_collaborative_mode_stop(
        self,
        fieldbus: IndustrialFieldbusInterlock,
        nominal_pose: RobotCartesianPose,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """SRMS mode with human nearby enforces Safe Operational Stop (SOS)."""
        config = RobotSafetyConfig(
            active_collaborative_mode=CollaborativeMode.SRMS,
            human_critical_distance_m=0.8,
        )
        gov = RobotSafetyGovernor(config=config, fieldbus_interlock=fieldbus)
        decision = gov.evaluate_actuation(
            tcp_pose=nominal_pose,
            joints=nominal_joints,
            human_distance_m=0.6,
        )
        assert decision.allowed is False
        assert decision.active_safety_function == RobotSafetyFunction.SOS
        assert any("SRMS Active" in r for r in decision.reasons)

    def test_pfl_contact_force_limit(
        self,
        governor: RobotSafetyGovernor,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """PFL contact force between 65N and 97.5N triggers PROTECTIVE_STOP."""
        excessive_force_pose = RobotCartesianPose(
            x_m=0.4, y_m=0.3, z_m=0.5, tcp_force_n=75.0  # > 65 N, but < 97.5 N
        )
        decision = governor.evaluate_actuation(
            tcp_pose=excessive_force_pose,
            joints=nominal_joints,
            human_distance_m=1.5,
        )
        assert decision.allowed is False
        assert decision.active_safety_function == RobotSafetyFunction.PROTECTIVE_STOP
        assert any("PFL Force Violation" in r for r in decision.reasons)


class TestEstopLatchAndReset:
    """Tests for latched emergency stop behaviour and PIN reset."""

    def test_latched_estop_blocks_future_cycles(
        self,
        governor: RobotSafetyGovernor,
        nominal_pose: RobotCartesianPose,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """Once STO is latched, subsequent calls remain in EMERGENCY_STOP."""
        # Trigger STO
        governor.evaluate_actuation(
            tcp_pose=nominal_pose,
            joints=nominal_joints,
            human_distance_m=1.0,
            collision_detected=True,
        )
        assert governor.is_latched() is True

        # Subsequent evaluation without collision must be rejected
        blocked_decision = governor.evaluate_actuation(
            tcp_pose=nominal_pose,
            joints=nominal_joints,
            human_distance_m=3.0,
            collision_detected=False,
        )
        assert blocked_decision.allowed is False
        assert blocked_decision.active_safety_function == RobotSafetyFunction.EMERGENCY_STOP

    def test_estop_reset_pin_validation(
        self,
        governor: RobotSafetyGovernor,
        nominal_pose: RobotCartesianPose,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """Resetting latched E-Stop requires correct PIN."""
        governor.evaluate_actuation(
            tcp_pose=nominal_pose,
            joints=nominal_joints,
            human_distance_m=0.1,
        )
        assert governor.is_latched() is True

        # Invalid PIN
        success, msg = governor.reset_estop("WRONG_PIN")
        assert success is False
        assert governor.is_latched() is True

        # Valid PIN
        success, msg = governor.reset_estop(RobotSafetyGovernor.RESET_PIN)
        assert success is True
        assert governor.is_latched() is False

        # Now nominal evaluation passes
        resumed = governor.evaluate_actuation(
            tcp_pose=nominal_pose,
            joints=nominal_joints,
            human_distance_m=3.0,
        )
        assert resumed.allowed is True


class TestRobotRealTimeLatency:
    """Benchmark tests to verify deterministic sub-50 µs evaluation latency."""

    def test_sub_50_microsecond_evaluation_latency(
        self,
        governor: RobotSafetyGovernor,
        nominal_pose: RobotCartesianPose,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """Verify that evaluation completes well within the 50 µs real-time deadline."""
        # Warmup
        for _ in range(20):
            governor.evaluate_actuation(nominal_pose, nominal_joints, human_distance_m=2.0)

        latencies = []
        for _ in range(100):
            dec = governor.evaluate_actuation(nominal_pose, nominal_joints, human_distance_m=2.0)
            latencies.append(dec.latency_us)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 50.0, f"Average latency {avg_latency:.2f} µs exceeds 50 µs target"


class TestParameterTestingISO10218_2:
    """Normative Parameter Testing per ISO 10218-2:2025.

    Verifies that safety function behaviours scale deterministically and proportionally
    when safety-critical parameters are varied across their operational spectrum.
    """

    def test_ssm_distance_parameter_scaling(
        self,
        nominal_pose: RobotCartesianPose,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """Verifies that widening human_critical_distance_m shifts speed clamping boundary accordingly."""
        # Baseline with critical_dist = 0.8m
        gov_base = RobotSafetyGovernor(config=RobotSafetyConfig(human_critical_distance_m=0.8))
        dec_at_1m = gov_base.evaluate_actuation(nominal_pose, nominal_joints, human_distance_m=1.0)
        assert dec_at_1m.active_safety_function != RobotSafetyFunction.SOS

        # Widened envelope with critical_dist = 1.2m
        gov_wide = RobotSafetyGovernor(
            config=RobotSafetyConfig(
                human_critical_distance_m=1.2,
                active_collaborative_mode=CollaborativeMode.SRMS,
            )
        )
        dec_wide = gov_wide.evaluate_actuation(nominal_pose, nominal_joints, human_distance_m=1.0)
        assert dec_wide.allowed is False
        assert dec_wide.active_safety_function == RobotSafetyFunction.SOS

    def test_force_limit_parameter_scaling(
        self,
        nominal_joints: list[RobotJointState],
    ) -> None:
        """Verifies that adjusting max_contact_force_n calibrates PFL trigger proportionally."""
        pose_80n = RobotCartesianPose(x_m=0.4, y_m=0.3, z_m=0.5, tcp_force_n=80.0)

        # Baseline: 65 N limit -> 80 N is rejected
        gov_standard = RobotSafetyGovernor(config=RobotSafetyConfig(max_contact_force_n=65.0))
        res_standard = gov_standard.evaluate_actuation(pose_80n, nominal_joints, human_distance_m=2.0)
        assert res_standard.allowed is False
        assert res_standard.active_safety_function == RobotSafetyFunction.PROTECTIVE_STOP

        # Heavy-duty tool configuration: 120 N limit -> 80 N is permitted
        gov_heavy = RobotSafetyGovernor(config=RobotSafetyConfig(max_contact_force_n=120.0))
        res_heavy = gov_heavy.evaluate_actuation(pose_80n, nominal_joints, human_distance_m=2.0)
        assert res_heavy.allowed is True


class TestBiofidelicBodyRegionsISO15066:
    """Tests for ISO/TS 15066 Annex A biomechanical body region thresholds."""

    @pytest.mark.parametrize(
        ("region", "test_force_n", "expect_allowed"),
        [
            (BodyRegion.FACE, 60.0, True),     # Limit: 65 N -> 60 N allowed
            (BodyRegion.FACE, 70.0, False),    # Limit: 65 N -> 70 N blocked
            (BodyRegion.CHEST, 130.0, True),   # Limit: 140 N -> 130 N allowed
            (BodyRegion.CHEST, 150.0, False),  # Limit: 140 N -> 150 N blocked
            (BodyRegion.HANDS_FINGERS, 130.0, True),  # Limit: 140 N -> 130 N allowed
            (BodyRegion.HANDS_FINGERS, 145.0, False), # Limit: 140 N -> 145 N blocked
            (BodyRegion.BACK_SHOULDERS, 200.0, True), # Limit: 210 N -> 200 N allowed
            (BodyRegion.BACK_SHOULDERS, 225.0, False),# Limit: 210 N -> 225 N blocked
        ],
    )
    def test_anatomical_body_region_thresholds(
        self,
        governor: RobotSafetyGovernor,
        nominal_joints: list[RobotJointState],
        region: BodyRegion,
        test_force_n: float,
        expect_allowed: bool,
    ) -> None:
        """Verifies exact compliance with ISO/TS 15066 Annex A biofidelic limits per body zone."""
        pose = RobotCartesianPose(x_m=0.4, y_m=0.3, z_m=0.5, tcp_force_n=test_force_n)
        dec = governor.evaluate_actuation(
            tcp_pose=pose,
            joints=nominal_joints,
            human_distance_m=1.5,
            target_body_region=region,
        )
        assert dec.allowed is expect_allowed
        if not expect_allowed:
            assert dec.active_safety_function == RobotSafetyFunction.PROTECTIVE_STOP
            assert any(f"for body region {region.value}" in r for r in dec.reasons)

