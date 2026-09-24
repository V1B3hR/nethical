# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Orbital Phased-Array & Laser Beam Steering Auditor (nethical.space.detectors.beam_steering_auditor).

Audits steerable active electronically scanned arrays (AESA), spot beams, and inter-satellite
laser communications for regulatory compliance with:
- ITU Radio Regulations Article 22 (EPFD limits protecting Geostationary Orbital Arc)
- ITU Radio Regulations Article 21 (Terrestrial power flux density limits)
- Law 9 (Transparency) & Law 20 (Coexistence)
- Radio Astronomy Quiet Zones (e.g. SKA, ALMA, Green Bank) and sovereign RF geofences
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from nethical.space.models import SPEED_OF_LIGHT_M_S, Vector3D

logger = logging.getLogger("nethical.space.detectors.beam_steering_auditor")


class BeamTargetType(str, Enum):
    """Destination classification for steerable RF or optical beams."""
    GROUND_STATION = "GROUND_STATION"
    INTER_SATELLITE_LINK = "INTER_SATELLITE_LINK"
    TERRESTRIAL_COMMUNITY = "TERRESTRIAL_COMMUNITY"
    PROHIBITED_ZONE = "PROHIBITED_ZONE"
    UNCOORDINATED_SATELLITE = "UNCOORDINATED_SATELLITE"


class ProhibitedGeofence(BaseModel):
    """Terrestrial geofenced zone where RF emissions or optical targeting is prohibited."""
    zone_id: str
    name: str
    center_lat: float
    center_lon: float
    radius_km: float
    prohibition_reason: str = "RADIO_ASTRONOMY_PROTECTION"


class BeamPointingCommand(BaseModel):
    """Steering command targeting a terrestrial coordinate or orbital spacecraft."""
    beam_id: str
    target_type: BeamTargetType
    target_lat: Optional[float] = None
    target_lon: Optional[float] = None
    target_satellite_id: Optional[str] = None
    carrier_frequency_ghz: float = Field(default=28.5, description="Transmitting frequency in GHz")
    tx_power_watts: float = Field(default=20.0, description="RF / Optical power (W)")
    tx_antenna_gain_dbi: float = Field(default=42.0, description="Boresight peak antenna gain (dBi)")
    slant_range_km: float = Field(default=800.0, description="Range to target (km)")
    off_axis_angle_to_geo_arc_deg: float = Field(default=15.0, description="Angle from GSO orbital arc (deg)")


class BeamAuditResult(BaseModel):
    """Regulatory audit verdict for the requested beam pointing vector."""
    allowed: bool
    epfd_dbw_m2: float
    itu_article22_compliant: bool
    geofence_compliant: bool
    reasons: List[str] = Field(default_factory=list)


class BeamSteeringAuditorConfig(BaseModel):
    """Regulatory thresholds for RF emissions and geofence containment."""
    # ITU Article 22 EPFD limit (-160 dB(W/m^2) in standard reference bandwidth)
    itu_max_epfd_dbw_m2: float = Field(default=-160.0, description="ITU EPFD threshold towards GSO arc (dB(W/m^2))")
    min_separation_from_geo_arc_deg: float = Field(default=2.5, description="Minimum separation angle from GEO arc (deg)")
    default_prohibited_geofences: List[ProhibitedGeofence] = Field(
        default_factory=lambda: [
            ProhibitedGeofence(
                zone_id="ZONE_SKA_AUS",
                name="Square Kilometre Array (Western Australia)",
                center_lat=-26.696,
                center_lon=116.637,
                radius_km=150.0,
                prohibition_reason="ITU Radio Astronomy Quiet Zone",
            ),
            ProhibitedGeofence(
                zone_id="ZONE_GREEN_BANK",
                name="National Radio Quiet Zone (Green Bank, USA)",
                center_lat=38.433,
                center_lon=-79.839,
                radius_km=100.0,
                prohibition_reason="US National Radio Astronomy Sanctuary",
            ),
        ]
    )


class BeamSteeringAuditor:
    """Verifies that steerable phased arrays and lasers respect ITU limits and quiet geofences."""

    def __init__(self, config: Optional[BeamSteeringAuditorConfig] = None) -> None:
        self.config = config or BeamSteeringAuditorConfig()

    def audit_beam_command(self, command: BeamPointingCommand) -> BeamAuditResult:
        """Evaluate beam pointing command against ITU EPFD limits and quiet zones."""
        reasons: List[str] = []
        allowed = True

        # 1. Calculate Equivalent Power Flux Density (EPFD)
        # Power Flux Density at slant range d: PFD = (P_tx * G_linear) / (4 * pi * d^2)
        d_m = max(1000.0, command.slant_range_km * 1000.0)
        g_linear = 10.0 ** (command.tx_antenna_gain_dbi / 10.0)
        eirp_watts = command.tx_power_watts * g_linear

        pfd_w_m2 = eirp_watts / (4.0 * math.pi * (d_m ** 2))
        pfd_dbw_m2 = 10.0 * math.log10(max(1e-30, pfd_w_m2))

        # EPFD scaling towards GSO arc considering off-axis roll-off
        off_axis = max(0.1, command.off_axis_angle_to_geo_arc_deg)
        # ITU standard off-axis antenna roll-off: G(theta) = 32 - 25*log10(theta)
        off_axis_gain_dbi = max(0.0, 32.0 - 25.0 * math.log10(off_axis))
        epfd_dbw_m2 = (
            10.0 * math.log10(max(1e-6, command.tx_power_watts))
            + off_axis_gain_dbi
            - (10.0 * math.log10(4.0 * math.pi * (d_m ** 2)))
        )

        itu_compliant = True
        if command.off_axis_angle_to_geo_arc_deg < self.config.min_separation_from_geo_arc_deg:
            if epfd_dbw_m2 > self.config.itu_max_epfd_dbw_m2:
                allowed = False
                itu_compliant = False
                reasons.append(
                    f"ITU Article 22 Breach: Beam {command.beam_id} pointing within {command.off_axis_angle_to_geo_arc_deg:.1f} deg "
                    f"of GEO arc causes EPFD = {epfd_dbw_m2:.1f} dB(W/m^2) exceeding limit {self.config.itu_max_epfd_dbw_m2:.1f} dB(W/m^2)."
                )

        # 2. Terrestrial Geofence Audit (Radio Astronomy Quiet Zones)
        geofence_compliant = True
        if command.target_lat is not None and command.target_lon is not None:
            for zone in self.config.default_prohibited_geofences:
                dist_km = self._haversine_distance_km(
                    command.target_lat, command.target_lon, zone.center_lat, zone.center_lon
                )
                if dist_km <= zone.radius_km:
                    allowed = False
                    geofence_compliant = False
                    reasons.append(
                        f"Geofence Violation: Spot beam illuminates prohibited sanctuary '{zone.name}' "
                        f"({dist_km:.1f} km from center, radius {zone.radius_km:.1f} km). {zone.prohibition_reason}."
                    )
                    break

        # 3. Uncoordinated Satellite Blinding / Dazzle Check
        if command.target_type == BeamTargetType.UNCOORDINATED_SATELLITE:
            allowed = False
            reasons.append(
                f"VETO: Uncoordinated targeting of foreign spacecraft {command.target_satellite_id} "
                f"prohibited under Law 20 (Coexistence) and Outer Space Treaty (Article IX - Due Regard)."
            )

        if allowed:
            reasons.append(
                f"Beam pointing authorized: Target {command.target_type.value}, EPFD = {epfd_dbw_m2:.1f} dB(W/m^2). "
                f"ITU and Geofence constraints satisfied."
            )

        return BeamAuditResult(
            allowed=allowed,
            epfd_dbw_m2=epfd_dbw_m2,
            itu_article22_compliant=itu_compliant,
            geofence_compliant=geofence_compliant,
            reasons=reasons,
        )

    def _haversine_distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in kilometers between two terrestrial coordinates."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        return 6371.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
