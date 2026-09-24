# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for Space Operations Compliance Pack (tests.compliance.test_space_operations_pack)."""

import pytest

from nethical.compliance.packs.space_operations_pack import (
    SpaceComplianceResult,
    SpaceMissionProfile,
    SpaceOperationsPack,
    SpaceStandard,
)


class TestSpaceOperationsPack:
    """Test suite for international space law and orbital standards compliance."""

    def test_nominal_satellite_clearance(self) -> None:
        pack = SpaceOperationsPack()
        profile = {
            "satellite_or_platform_id": "SOVEREIGN_SAT_LEO_01",
            "mission_profile": "SPACE_OPERATIONS_PROFILE",
            "itu_epfd_and_spectrum_coordinated": True,
            "ost_due_regard_and_non_contamination": True,
            "eu_space_act_stm_compliant": True,
            "ecss_e40_q80_software_assured": True,
            "resilient_to_adversarial_jamming_and_spoofing": True,
            "post_mission_disposal_under_5years": True,
            "autonomous_conjunction_avoidance_active": True,
        }
        result: SpaceComplianceResult = pack.evaluate(profile)
        assert result.is_space_certified_ready
        assert result.readiness_score == 1.0
        assert result.operational_clearance_status == "CLEARED_FOR_LAUNCH"
        assert len(result.missing_capabilities) == 0

    def test_outer_space_treaty_violation_rejects_clearance(self) -> None:
        pack = SpaceOperationsPack()
        profile = {
            "satellite_or_platform_id": "RENEGADE_PROBE_01",
            "ost_due_regard_and_non_contamination": False,  # Treaty violation!
        }
        result = pack.evaluate(profile)
        assert not result.is_space_certified_ready
        assert any("TREATY VIOLATION" in m for m in result.missing_capabilities)

    def test_missing_conjunction_avoidance_flags_eu_space_act(self) -> None:
        pack = SpaceOperationsPack()
        profile = {
            "satellite_or_platform_id": "CUBESAT_NO_MANEUVER",
            "autonomous_conjunction_avoidance_active": False,
        }
        result = pack.evaluate(profile)
        assert not result.conjunction_avoidance_verified
        assert any("EU SPACE ACT NON-COMPLIANCE" in m for m in result.missing_capabilities)
        assert any("CollisionCourseDetector" in r for r in result.regulatory_recommendations)

    def test_zero_debris_5year_deorbit_enforcement(self) -> None:
        pack = SpaceOperationsPack()
        profile = {
            "satellite_or_platform_id": "SAT_LONG_LIVED",
            "post_mission_disposal_under_5years": False,  # Will remain as space debris!
        }
        result = pack.evaluate(profile)
        assert not result.post_mission_disposal_compliant
        assert any("ZERO DEBRIS CHARTER BREACH" in m for m in result.missing_capabilities)
