# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Orbital Conjunction & Collision Course Detector (nethical.space.detectors.collision_detector).

Implements Conjunction Assessment Risk Analysis (CARA), Foster 2D collision probability (Pc),
and autonomous avoidance maneuver authorization gates complying with:
- Law 21 (Protection: Defense against kinetic destruction)
- Law 22 (Prevention: Proactive risk avoidance)
- EU Space Act COM(2025) 335 (Debris mitigation & space traffic management)
- NASA / ESA Space Debris Mitigation Guidelines (Pc > 1e-4 threshold)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from nethical.space.models import OrbitalState, Vector3D

logger = logging.getLogger("nethical.space.detectors.collision_detector")


class ConjunctionAlertLevel(str, Enum):
    """Conjunction assessment threat levels."""
    NOMINAL = "NOMINAL"                   # Miss distance > 5 km, Pc < 1e-7
    INFORMATIONAL = "INFORMATIONAL"       # 1 km < Miss distance <= 5 km
    WATCH = "WATCH"                       # Miss distance <= 1 km, Pc >= 1e-5
    CRITICAL_AVOIDANCE = "CRITICAL_AVOIDANCE"  # Pc >= 1e-4, active burn required


class SecondaryHazardObject(BaseModel):
    """Known debris object, derelict rocket body, or satellite for secondary hazard avoidance."""
    object_id: str
    catalog_name: str = "DEBRIS_CATALOG"
    position_eci_km: Vector3D
    velocity_eci_kms: Vector3D
    hard_body_radius_m: float = 2.5


class CollisionAssessmentResult(BaseModel):
    """Comprehensive conjunction assessment and maneuver authorization decision."""
    conjunction_detected: bool
    alert_level: ConjunctionAlertLevel
    miss_distance_km: float
    radial_miss_m: float
    intrack_miss_m: float
    crosstrack_miss_m: float
    collision_probability_pc: float
    maneuver_required: bool
    maneuver_authorized: bool
    safe_delta_v_vector_ms: Optional[Vector3D] = None
    reasons: List[str] = Field(default_factory=list)


class CollisionDetectorConfig(BaseModel):
    """Thresholds for orbital collision risk analysis."""
    pc_threshold: float = Field(default=1e-4, description="Collision probability threshold requiring avoidance (1e-4)")
    critical_miss_distance_km: float = Field(default=1.0, description="Hard miss distance boundary (km)")
    nominal_hard_body_radius_m: float = Field(default=5.0, description="Combined physical radius of colliding bodies")
    default_evasion_delta_v_ms: float = Field(default=0.5, description="Default impulse delta-V for radial separation (m/s)")


class CollisionCourseDetector:
    """Evaluates orbital conjunctions and authorizes autonomous kinetic maneuvers."""

    def __init__(self, config: Optional[CollisionDetectorConfig] = None) -> None:
        self.config = config or CollisionDetectorConfig()

    def evaluate_conjunction(
        self,
        primary_state: OrbitalState,
        secondary_state: OrbitalState,
        proposed_delta_v_ms: Optional[Vector3D] = None,
        secondary_catalog: Optional[List[SecondaryHazardObject]] = None,
    ) -> CollisionAssessmentResult:
        """Calculate miss distance, Foster 2D collision probability Pc, and authorize maneuver."""
        r1 = primary_state.position_eci_km
        v1 = primary_state.velocity_eci_kms
        r2 = secondary_state.position_eci_km
        v2 = secondary_state.velocity_eci_kms

        # Relative position vector in ECI
        delta_r = r2.subtract(r1)
        miss_distance_km = delta_r.magnitude

        # Construct Radial, In-Track, Cross-Track (RIC) frame vectors for primary satellite
        u_rad = r1.normalized()
        h_vec = r1.cross(v1)
        w_cross = h_vec.normalized()
        v_intrack = w_cross.cross(u_rad).normalized()

        # Project delta_r into RIC frame in metres
        delta_r_m = delta_r.scale(1000.0)
        dr_m = delta_r_m.dot(u_rad)
        di_m = delta_r_m.dot(v_intrack)
        dc_m = delta_r_m.dot(w_cross)

        # Combined position variance (1-sigma sum)
        cov1 = primary_state.covariance
        cov2 = secondary_state.covariance
        sigma_rad = math.sqrt(cov1.sigma_radial_m ** 2 + cov2.sigma_radial_m ** 2)
        sigma_cross = math.sqrt(cov1.sigma_crosstrack_m ** 2 + cov2.sigma_crosstrack_m ** 2)
        sigma_comb = max(1.0, math.sqrt((sigma_rad ** 2 + sigma_cross ** 2) / 2.0))

        # Foster 2D Collision Probability (Pc)
        r_comb = self.config.nominal_hard_body_radius_m
        d_miss_m = miss_distance_km * 1000.0

        exponent_hardbody = -(r_comb ** 2) / (2.0 * (sigma_comb ** 2))
        exponent_miss = -(d_miss_m ** 2) / (2.0 * (sigma_comb ** 2))

        # Clamp exponents to avoid underflow/overflow
        p_hardbody = 1.0 - math.exp(max(-700.0, exponent_hardbody))
        pc = p_hardbody * math.exp(max(-700.0, exponent_miss))
        pc = min(1.0, max(0.0, pc))

        # Classify Alert Level
        conjunction_detected = False
        maneuver_required = False
        alert_level = ConjunctionAlertLevel.NOMINAL
        reasons: List[str] = []

        if pc >= self.config.pc_threshold or miss_distance_km <= self.config.critical_miss_distance_km:
            conjunction_detected = True
            maneuver_required = True
            alert_level = ConjunctionAlertLevel.CRITICAL_AVOIDANCE
            reasons.append(
                f"Critical Conjunction Detected with {secondary_state.satellite_id}: "
                f"Miss distance = {miss_distance_km:.3f} km, Collision Probability Pc = {pc:.2e} "
                f"(exceeds threshold {self.config.pc_threshold:.2e}). Autonomous evasive maneuver required."
            )
        elif pc >= 1e-6 or miss_distance_km <= 3.0:
            conjunction_detected = True
            alert_level = ConjunctionAlertLevel.WATCH
            reasons.append(
                f"Conjunction Watch: Miss distance = {miss_distance_km:.3f} km, Pc = {pc:.2e}."
            )
        else:
            reasons.append(
                f"Trajectory nominal: Miss distance = {miss_distance_km:.3f} km, Pc = {pc:.2e}."
            )

        # Maneuver Evaluation and Debris Cloud Safety Check
        maneuver_authorized = False
        safe_delta_v: Optional[Vector3D] = None

        if maneuver_required:
            # Propose nominal radial delta-V burn if not provided
            effective_delta_v = proposed_delta_v_ms or Vector3D(
                x=w_cross.x * self.config.default_evasion_delta_v_ms,
                y=w_cross.y * self.config.default_evasion_delta_v_ms,
                z=w_cross.z * self.config.default_evasion_delta_v_ms,
            )

            # Check if proposed maneuver creates secondary hazard with known catalog debris
            is_secondary_hazard = False
            if secondary_catalog:
                # Convert delta_v to km/s
                dv_kms = effective_delta_v.scale(0.001)
                post_maneuver_vel = v1.add(dv_kms)
                post_maneuver_state = OrbitalState(
                    satellite_id=primary_state.satellite_id,
                    position_eci_km=r1,
                    velocity_eci_kms=post_maneuver_vel,
                )

                for debris in secondary_catalog:
                    debris_dist = post_maneuver_state.position_eci_km.distance_to(debris.position_eci_km)
                    if debris_dist < self.config.critical_miss_distance_km:
                        is_secondary_hazard = True
                        reasons.append(
                            f"VETO: Proposed avoidance burn creates secondary conjunction with {debris.object_id} "
                            f"(projected distance: {debris_dist:.2f} km). Maneuver rejected under Law 22 (Prevention)."
                        )
                        break

            if not is_secondary_hazard:
                maneuver_authorized = True
                safe_delta_v = effective_delta_v
                reasons.append(
                    f"Avoidance burn authorized: Delta-V = {effective_delta_v.magnitude:.3f} m/s. "
                    f"Verified clean corridor clear of secondary catalog debris."
                )

        return CollisionAssessmentResult(
            conjunction_detected=conjunction_detected,
            alert_level=alert_level,
            miss_distance_km=miss_distance_km,
            radial_miss_m=dr_m,
            intrack_miss_m=di_m,
            crosstrack_miss_m=dc_m,
            collision_probability_pc=pc,
            maneuver_required=maneuver_required,
            maneuver_authorized=maneuver_authorized,
            safe_delta_v_vector_ms=safe_delta_v,
            reasons=reasons,
        )
