# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Orbital Hardware-in-the-Loop (HIL) Simulator (tests.space.test_hil_simulator)."""

import pytest

from nethical.space.hil_simulator import (
    OrbitalHILResult,
    OrbitalHILSimulator,
    SpaceFaultType,
)
from nethical.space.orbital_governor import OrbitalAction


class TestOrbitalHILSimulator:
    """Test suite for in-the-loop stress testing and fault injection."""

    def test_hil_jamming_ramp_mitigation(self) -> None:
        hil = OrbitalHILSimulator()
        res: OrbitalHILResult = hil.run_jamming_ramp_test(peak_jammer_power_dbw=-55.0)

        assert res.injection_successful
        assert res.fault_type == SpaceFaultType.RF_BARRAGE_JAMMING
        assert res.governor_action == OrbitalAction.SWITCH_TO_OPTICAL_ISL
        assert res.mitigation_verified
        assert res.test_passed
        assert res.governor_reaction_latency_us > 0.0

    def test_hil_spoofing_step_reversion(self) -> None:
        hil = OrbitalHILSimulator()
        res = hil.run_spoofing_step_test(spoof_divergence_km=45.0)

        assert res.injection_successful
        assert res.fault_type == SpaceFaultType.GNSS_SPOOFING_INJECTION
        assert res.governor_action == OrbitalAction.SWITCH_TO_CELESTIAL_NAV
        assert res.mitigation_verified
        assert res.test_passed

    def test_hil_radiation_seu_bitflip(self) -> None:
        hil = OrbitalHILSimulator()
        res = hil.run_radiation_seu_bitflip_test()

        assert res.injection_successful
        assert res.fault_type == SpaceFaultType.RADIATION_SEU_BIT_FLIP
        assert res.test_passed
        assert res.mitigation_verified
