# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for ISO 13849-1 (PL-e / Cat 4) and ISO 26262 (ASIL-D) Functional Safety (tests.edge.test_iso13849_iso26262).

Validates quantitative machinery safety metrics:
- Diagnostic Coverage (DC >= 99%)
- Mean Time to Dangerous Failure (MTTFd >= 30 years)
- Common Cause Failure (CCF >= 65 points per Annex F)
- Automotive HARA ASIL rating and Autonomous Emergency Braking (AEB) overrides.
"""

from __future__ import annotations

import time
import pytest

from nethical.edge.industrial_fieldbus import IndustrialFieldbusInterlock
from nethical.edge.iso13849_watchdog import (
    CobotSafetyMode,
    HardwareWatchdogTimer,
    ISO13849Evaluation,
    ISO13849SafetyEvaluator,
    PerformanceLevel,
)
from nethical.edge.iso26262_asil import (
    ASILRating,
    Controllability,
    Exposure,
    ISO26262EvaluationResult,
    ISO26262SafetyEvaluator,
    Severity,
    VehicleControlState,
)


class TestISO13849MachinerySafety:
    """Quantitative evaluation of machinery safety functions per ISO 13849-1."""

    def test_ple_category4_quantified_compliance(self) -> None:
        """Category 4 with DC >= 99%, MTTFd >= 30y, CCF >= 65 achieves PL-e (SIL 3 equivalent)."""
        evaluator = ISO13849SafetyEvaluator()
        result = evaluator.evaluate_performance_level(
            category="Cat 4",
            mttf_d_years=35.0,
            dc_avg_pct=99.4,
            ccf_score=80,
            required_pl=PerformanceLevel.PL_E,
        )

        assert isinstance(result, ISO13849Evaluation)
        assert result.achieved_pl == PerformanceLevel.PL_E
        assert result.target_pl_met is True
        assert result.safety_integrity_level == "SIL_3"
        assert result.is_compliant_for_human_shared_space is True
        assert len(result.violations) == 0

    def test_insufficient_diagnostic_coverage_downgrades_pl(self) -> None:
        """Diagnostic Coverage below 99% fails to satisfy PL-e requirement for Cat 4."""
        evaluator = ISO13849SafetyEvaluator()
        result = evaluator.evaluate_performance_level(
            category="Cat 4",
            mttf_d_years=35.0,
            dc_avg_pct=88.0,  # Below 90% threshold for PL-d/e
            ccf_score=75,
            required_pl=PerformanceLevel.PL_E,
        )

        assert result.achieved_pl == PerformanceLevel.PL_C
        assert result.target_pl_met is False
        assert any("nie spełnia wymaganego" in v for v in result.violations)

    def test_low_ccf_score_flags_annex_f_violation(self) -> None:
        """Common Cause Failure (CCF) score < 65 fails redundant channel separation."""
        evaluator = ISO13849SafetyEvaluator()
        result = evaluator.evaluate_performance_level(
            category="Cat 4",
            mttf_d_years=35.0,
            dc_avg_pct=99.5,
            ccf_score=55,  # Insufficient CCF per Annex F
            required_pl=PerformanceLevel.PL_E,
        )

        assert result.target_pl_met is False
        assert any("Wynik CCF=55 < 65" in v for v in result.violations)


class TestHardwareWatchdogTimer:
    """Sub-millisecond hardware watchdog timer fail-closed verification."""

    def test_watchdog_nominal_kicks(self) -> None:
        """Regular kicks keep hardware relay energized."""
        wdt = HardwareWatchdogTimer(timeout_us=20000.0)  # 20ms
        assert wdt.kick(agent_id="test_node", sequence_id=1) is True
        status = wdt.check_and_enforce()
        assert status.hardware_relay_energized is True
        assert status.is_tripped is False

    def test_watchdog_timeout_trips_relay_and_fieldbus(self) -> None:
        """Timeout de-energizes relay and invokes industrial fieldbus callback."""
        tripped_reasons: list[str] = []

        def on_cutoff(reason: str) -> None:
            tripped_reasons.append(reason)

        wdt = HardwareWatchdogTimer(timeout_us=100.0)  # 100 µs
        wdt.register_fieldbus_callback(on_cutoff)

        # Force timeout
        time.sleep(0.002)  # 2 ms >> 100 µs
        status = wdt.check_and_enforce()

        assert status.hardware_relay_energized is False
        assert status.is_tripped is True
        assert len(tripped_reasons) == 1
        assert "Przekroczono limit pulsu" in tripped_reasons[0]

        # Reset without override key is blocked
        assert wdt.manual_reset("INVALID_KEY") is False
        assert wdt.is_tripped is True

        # Authorized reset succeeds
        assert wdt.manual_reset("NETHICAL_HARDWARE_OVERRIDE_AUTH") is True
        assert wdt.is_tripped is False
        assert wdt.hardware_relay_energized is True


class TestISO26262AutomotiveSafety:
    """Tests for Road Vehicles Functional Safety (ISO 26262 ASIL-D & HARA)."""

    def test_asild_determination(self) -> None:
        """S3 (fatal) + E4 (continuous) + C3 (uncontrollable) yields ASIL-D."""
        fieldbus = IndustrialFieldbusInterlock()
        evaluator = ISO26262SafetyEvaluator(fieldbus=fieldbus)

        result = evaluator.evaluate_motion_command(
            commanded_speed_mps=25.0,
            commanded_steering_deg_per_sec=35.0,
            time_to_collision_seconds=2.5,
            hazard_description="Unintended high-speed steering divergence on motorway",
            severity=Severity.S3,
            exposure=Exposure.E4,
            controllability=Controllability.C3,
        )

        assert isinstance(result, ISO26262EvaluationResult)
        assert result.asil_level == ASILRating.ASIL_D
        assert result.control_state == VehicleControlState.NORMAL_AUTONOMOUS
        assert result.is_actuation_permitted is True

    def test_aeb_emergency_brake_override(self) -> None:
        """Time-To-Collision <= 0.6s engages Automatic Emergency Braking (AEB)."""
        fieldbus = IndustrialFieldbusInterlock()
        evaluator = ISO26262SafetyEvaluator(fieldbus=fieldbus)

        result = evaluator.evaluate_motion_command(
            commanded_speed_mps=15.0,
            commanded_steering_deg_per_sec=20.0,
            time_to_collision_seconds=0.45,  # <= 0.6s
            hazard_description="Imminent pedestrian collision in trajectory",
            severity=Severity.S3,
            exposure=Exposure.E4,
            controllability=Controllability.C3,
        )

        assert result.control_state == VehicleControlState.AEB_EMERGENCY_BRAKE_ACTIVE
        assert result.is_actuation_permitted is False
        assert any("Automatic Emergency Braking" in mech for mech in result.safety_mechanisms_active)

    def test_steering_rate_exceedance_trips_fieldbus(self) -> None:
        """Steering angular velocity > 450 deg/s trips emergency fieldbus interlock."""
        fieldbus = IndustrialFieldbusInterlock()
        evaluator = ISO26262SafetyEvaluator(fieldbus=fieldbus)

        result = evaluator.evaluate_motion_command(
            commanded_speed_mps=20.0,
            commanded_steering_deg_per_sec=520.0,  # Exceeds 450 deg/s threshold
            time_to_collision_seconds=2.0,
            hazard_description="Erratic steering actuator runaway command",
            severity=Severity.S3,
            exposure=Exposure.E4,
            controllability=Controllability.C3,
        )

        assert result.control_state == VehicleControlState.STEERING_RATE_OVERRIDE_STOP
        assert result.is_actuation_permitted is False
        assert result.hardware_interlock_tripped is True
        assert fieldbus.is_interlocked is True
