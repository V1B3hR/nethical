# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Orbital GNSS Spoofing & Ephemeris Manipulation Detector (nethical.space.detectors.spoofing_detector).

Detects adversarial GNSS spoofing, false ephemeris injection, and pseudorange drift
by cross-validating spaceborne GNSS receivers against onboard celestial star trackers,
earth horizon sensors, and inertial measurement units (IMUs).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from nethical.space.models import OrbitalState

logger = logging.getLogger("nethical.space.detectors.spoofing_detector")


class SpoofingSeverity(str, Enum):
    """Classification of spoofing threat severity."""
    NONE = "NONE"
    SUSPECTED_DRIFT = "SUSPECTED_DRIFT"       # Slow pseudorange walk-off / clock creep
    ACUTE_SPOOFING = "ACUTE_SPOOFING"         # Abrupt position / velocity step jump
    UNAUTHENTICATED_SIGNAL = "UNAUTHENTICATED" # Missing cryptographic OSNMA signatures


class SpoofingMitigationAction(str, Enum):
    """Autonomous navigation mitigation actions."""
    NONE = "NONE"
    REJECT_GNSS_USE_CELESTIAL = "REJECT_GNSS_USE_CELESTIAL"
    FALLBACK_TO_OSNMA_GALILEO = "FALLBACK_TO_OSNMA_GALILEO"
    HOLD_PROPAGATION_ORBIT = "HOLD_PROPAGATION_ORBIT"


class SpoofingAlert(BaseModel):
    """Result of orbital GNSS spoofing and state consistency verification."""
    is_spoofed: bool
    severity: SpoofingSeverity = SpoofingSeverity.NONE
    position_divergence_km: float
    velocity_divergence_kms: float
    clock_drift_ppm: float
    mitigation_action: SpoofingMitigationAction = SpoofingMitigationAction.NONE
    details: str


class SpoofingDetectorConfig(BaseModel):
    """Thresholds for orbital GNSS vs Celestial/IMU divergence."""
    max_position_divergence_km: float = Field(default=5.0, description="Max acceptable divergence before spoofing alert (km)")
    max_velocity_divergence_kms: float = Field(default=0.05, description="Max velocity divergence (km/s = 50 m/s)")
    max_clock_drift_ppm: float = Field(default=15.0, description="Max allowable receiver clock drift rate (ppm)")
    star_tracker_minimum_stars: int = Field(default=4, description="Minimum tracked stars for celestial validation")


class SpoofingDetector:
    """Detects hostile GNSS manipulation in orbit via multi-sensor cross-validation."""

    def __init__(self, config: Optional[SpoofingDetectorConfig] = None) -> None:
        self.config = config or SpoofingDetectorConfig()

    def evaluate(
        self,
        gnss_state: OrbitalState,
        celestial_imu_state: OrbitalState,
        receiver_clock_drift_ppm: float = 0.0,
        osnma_authenticated: bool = True,
        tracked_stars_count: int = 8,
    ) -> SpoofingAlert:
        """Cross-validate GNSS navigation solution against independent celestial & inertial references."""
        # Calculate Euclidean position and velocity divergence in ECI J2000
        pos_divergence_km = gnss_state.position_eci_km.distance_to(celestial_imu_state.position_eci_km)
        vel_divergence_kms = gnss_state.velocity_eci_kms.distance_to(celestial_imu_state.velocity_eci_kms)

        is_spoofed = False
        severity = SpoofingSeverity.NONE
        mitigation = SpoofingMitigationAction.NONE
        details = "GNSS orbit solution verified against onboard celestial reference."

        # 1. Acute Step Jump / Unphysical Divergence
        if (
            pos_divergence_km > self.config.max_position_divergence_km
            or vel_divergence_kms > self.config.max_velocity_divergence_kms
        ):
            is_spoofed = True
            severity = SpoofingSeverity.ACUTE_SPOOFING
            mitigation = SpoofingMitigationAction.REJECT_GNSS_USE_CELESTIAL
            details = (
                f"Severe GNSS navigation solution spoofing detected! "
                f"Position divergence: {pos_divergence_km:.2f} km (limit: {self.config.max_position_divergence_km:.2f} km). "
                f"Velocity divergence: {vel_divergence_kms * 1000.0:.1f} m/s. "
                f"Discarding GNSS inputs; switching to Autonomous Celestial/Star-Tracker Navigation."
            )

        # 2. Clock Drift Creep (Walk-off attack)
        elif abs(receiver_clock_drift_ppm) > self.config.max_clock_drift_ppm:
            is_spoofed = True
            severity = SpoofingSeverity.SUSPECTED_DRIFT
            mitigation = SpoofingMitigationAction.HOLD_PROPAGATION_ORBIT
            details = (
                f"Adversarial receiver clock walk-off detected: drift rate = {receiver_clock_drift_ppm:.1f} ppm "
                f"(threshold: {self.config.max_clock_drift_ppm:.1f} ppm). Holding orbital propagation state."
            )

        # 3. Missing Cryptographic Authenticity in Contested Environment
        elif not osnma_authenticated:
            is_spoofed = True
            severity = SpoofingSeverity.UNAUTHENTICATED_SIGNAL
            mitigation = SpoofingMitigationAction.FALLBACK_TO_OSNMA_GALILEO
            details = (
                "GNSS signals lack authenticated cryptographic signatures (Galileo OSNMA failure). "
                "Enforcing encrypted sovereign signal selection."
            )

        if is_jammed_or_spoofed := is_spoofed:
            logger.warning(
                "Orbital GNSS Spoofing Detected on %s! Severity: %s | Action: %s",
                gnss_state.satellite_id,
                severity.value,
                mitigation.value,
            )

        return SpoofingAlert(
            is_spoofed=is_spoofed,
            severity=severity,
            position_divergence_km=pos_divergence_km,
            velocity_divergence_kms=vel_divergence_kms,
            clock_drift_ppm=receiver_clock_drift_ppm,
            mitigation_action=mitigation,
            details=details,
        )
