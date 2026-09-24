# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Sovereign Space Operations & Orbital Compliance Pack (nethical.compliance.packs.space_operations_pack).

Implements regulatory compliance evaluations for orbital spacecraft, satellite constellations,
and High-Altitude Platform Stations (HAPS) adhering to:
- ITU Radio Regulations (Articles 21 & 22 EPFD power flux limits)
- Outer Space Treaty (1967) (Articles VI, VII, and IX: Due Regard & Non-Contamination)
- EU Space Act COM(2025) 335 (Space Traffic Management & Zero Debris Charter)
- ECSS Space Software Engineering & Product Assurance (ECSS-E-ST-40C / ECSS-Q-ST-80C)
- NATO AEP-107 Sense-and-Avoid (Orbital & Stratospheric Domain Extension)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.compliance.packs.space_operations_pack")


class SpaceMissionProfile(str, Enum):
    """Mission operational classifications in orbital and stratospheric domains."""
    SPACE_OPERATIONS_PROFILE = "SPACE_OPERATIONS_PROFILE"
    CONSTELLATION_OPERATOR_PROFILE = "CONSTELLATION_OPERATOR_PROFILE"
    HAPS_MISSION_PROFILE = "HAPS_MISSION_PROFILE"
    DEFENSE_ORBITAL_SECURITY_PROFILE = "DEFENSE_ORBITAL_SECURITY_PROFILE"


class SpaceStandard(str, Enum):
    """Treaties, international regulations, and engineering standards for space flight."""
    ITU_RADIO_REGULATIONS = "ITU_Radio_Regulations_Articles_21_22"
    OUTER_SPACE_TREATY = "Outer_Space_Treaty_1967_Due_Regard"
    EU_SPACE_ACT = "EU_Space_Act_COM_2025_335_STM"
    ECSS_SOFTWARE_ASSURANCE = "ECSS_E_ST_40C_Q_ST_80C"
    NATO_ORBITAL_DAA = "NATO_AEP_107_Space_Sense_And_Avoid"
    ZERO_DEBRIS_DISPOSAL = "ESA_Zero_Debris_5Year_Deorbit"


class SpaceComplianceResult(BaseModel):
    """Result of regulatory and treaty evaluation for space systems."""
    evaluation_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    satellite_or_platform_id: str
    mission_profile: SpaceMissionProfile
    is_space_certified_ready: bool
    readiness_score: float = Field(..., ge=0.0, le=1.0)
    standards_breakdown: Dict[str, bool]
    conjunction_avoidance_verified: bool
    epfd_itu_coordination_verified: bool
    post_mission_disposal_compliant: bool
    missing_capabilities: List[str] = Field(default_factory=list)
    operational_clearance_status: str = Field(
        ...,
        description="CLEARED_FOR_LAUNCH, CLEARED_FOR_ORBITAL_OPS, RESTRICTED_SIMULATION_ONLY, REJECTED_NON_COMPLIANT",
    )
    regulatory_recommendations: List[str] = Field(default_factory=list)


class SpaceOperationsPack:
    """Evaluates spacecraft and constellation systems against international space laws and technical standards."""

    def __init__(self) -> None:
        self.doctrine = "Sovereign Orbital & Stratospheric AI Governance Doctrine"
        self.certifying_frameworks = [
            "ITU Radiocommunication Bureau",
            "UN Office for Outer Space Affairs (UNOOSA)",
            "European Space Agency (ESA) Zero Debris Charter",
            "European Commission Directorate-General for Defence Industry and Space (DG DEFIS)",
            "NATO Allied Command Transformation (Space Center)",
        ]

    def evaluate(self, system_profile: Dict[str, Any]) -> SpaceComplianceResult:
        """Evaluate a spacecraft or HAPS operational profile against international space requirements."""
        platform_id = system_profile.get("satellite_or_platform_id", "SOVEREIGN_NODE_01")
        profile_raw = system_profile.get("mission_profile", "SPACE_OPERATIONS_PROFILE")
        try:
            mission_profile = SpaceMissionProfile(profile_raw)
        except ValueError:
            mission_profile = SpaceMissionProfile.SPACE_OPERATIONS_PROFILE

        # Evaluate the 6 core space standards
        standards_status: Dict[str, bool] = {
            SpaceStandard.ITU_RADIO_REGULATIONS.value: system_profile.get("itu_epfd_and_spectrum_coordinated", True),
            SpaceStandard.OUTER_SPACE_TREATY.value: system_profile.get("ost_due_regard_and_non_contamination", True),
            SpaceStandard.EU_SPACE_ACT.value: system_profile.get("eu_space_act_stm_compliant", True),
            SpaceStandard.ECSS_SOFTWARE_ASSURANCE.value: system_profile.get("ecss_e40_q80_software_assured", True),
            SpaceStandard.NATO_ORBITAL_DAA.value: system_profile.get("resilient_to_adversarial_jamming_and_spoofing", True),
            SpaceStandard.ZERO_DEBRIS_DISPOSAL.value: system_profile.get("post_mission_disposal_under_5years", True),
        }

        missing: List[str] = []
        recommendations: List[str] = []

        # 1. Outer Space Treaty due regard check
        if not standards_status[SpaceStandard.OUTER_SPACE_TREATY.value]:
            missing.append(
                "TREATY VIOLATION: Non-compliance with Outer Space Treaty Article IX (Due Regard / Harmful Contamination)."
            )

        # 2. Debris mitigation and conjunction avoidance check
        conjunction_verified = system_profile.get("autonomous_conjunction_avoidance_active", True)
        if not conjunction_verified:
            standards_status[SpaceStandard.EU_SPACE_ACT.value] = False
            missing.append(
                "EU SPACE ACT NON-COMPLIANCE: Missing autonomous collision avoidance capability (Pc > 1e-4 CARA)."
            )
            recommendations.append("Integrate nethical.space.detectors.CollisionCourseDetector into onboard flight avionics.")

        # 3. Post-mission de-orbit disposal
        disposal_compliant = standards_status[SpaceStandard.ZERO_DEBRIS_DISPOSAL.value]
        if not disposal_compliant and mission_profile != SpaceMissionProfile.HAPS_MISSION_PROFILE:
            missing.append(
                "ZERO DEBRIS CHARTER BREACH: No certified 5-year post-mission orbital de-orbit or graveyard burn plan."
            )
            recommendations.append("Implement automated passivation and propulsion reserve allocation for end-of-life deorbit.")

        # 4. ITU frequency and EPFD limits
        itu_verified = standards_status[SpaceStandard.ITU_RADIO_REGULATIONS.value]
        if not itu_verified:
            missing.append(
                "ITU ARTICLE 22 BREACH: Potential uncoordinated EPFD interference with Geostationary Orbital Arc."
            )
            recommendations.append("Deploy BeamSteeringAuditor to gate steerable phased-array beams.")

        # 5. Electronic warfare & jamming resilience
        ew_resilient = standards_status[SpaceStandard.NATO_ORBITAL_DAA.value]
        if not ew_resilient:
            missing.append(
                "DEFENSE DAA DEFICIENCY: Insufficient anti-jamming and anti-spoofing resilience under contested conditions."
            )
            recommendations.append("Integrate multi-constellation GNSS / celestial star-tracker cross-validation.")

        # Calculate readiness score
        passed_count = sum(1 for val in standards_status.values() if val)
        total_count = len(standards_status)
        readiness_score = round(passed_count / float(total_count), 3)

        is_certified_ready = readiness_score >= 0.83 and not any("TREATY VIOLATION" in m for m in missing)

        if is_certified_ready and readiness_score == 1.0:
            clearance = "CLEARED_FOR_LAUNCH"
        elif is_certified_ready:
            clearance = "CLEARED_FOR_ORBITAL_OPS"
        elif readiness_score >= 0.5:
            clearance = "RESTRICTED_SIMULATION_ONLY"
        else:
            clearance = "REJECTED_NON_COMPLIANT"

        eval_id = f"SPACE-EVAL-{platform_id}-{int(datetime.now(timezone.utc).timestamp())}"

        return SpaceComplianceResult(
            evaluation_id=eval_id,
            satellite_or_platform_id=platform_id,
            mission_profile=mission_profile,
            is_space_certified_ready=is_certified_ready,
            readiness_score=readiness_score,
            standards_breakdown=standards_status,
            conjunction_avoidance_verified=conjunction_verified,
            epfd_itu_coordination_verified=itu_verified,
            post_mission_disposal_compliant=disposal_compliant,
            missing_capabilities=missing,
            operational_clearance_status=clearance,
            regulatory_recommendations=recommendations,
        )
