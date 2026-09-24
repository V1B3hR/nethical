# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Dedicated Stratospheric & HAPS Detectors (nethical.space.detectors.stratospheric_detector).

Implements stratospheric pseudo-satellite safety, persistent surveillance dwell limiting,
EMF/RF spectrum eavesdropping prevention, and U-space / civil airspace climb-descent coordination:
- Law 25 (Privacy: Persistent surveillance time limits & lawful warrant verification)
- Law 21 (Protection: Airspace deconfliction during FL000-FL600 transition)
- EU U-space Regulations (EU 2021/664, 2021/665, 2021/666) / FAA Upper Class E (FL600+)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.space.detectors.stratospheric_detector")


class AirspaceClass(str, Enum):
    """Airspace operational boundaries for stratospheric transitions."""
    CONTROLLED_CIVIL_AIRSPACE = "CONTROLLED_CIVIL_AIRSPACE"  # FL000 to FL600 (0 - 18,288 m)
    UPPER_AIRSPACE_FL600_PLUS = "UPPER_AIRSPACE_FL600_PLUS"  # FL600+ (> 18,288 m, HAPS envelope)


class StratosphericDwellAlert(BaseModel):
    """Result of persistent surveillance dwell time audit."""
    dwell_exceeded: bool
    current_dwell_hours: float
    max_allowable_dwell_hours: float
    target_zone_id: str
    action_required: str
    details: str


class EMFInterceptAlert(BaseModel):
    """Result of payload radio frequency intercept auditing."""
    unauthorized_intercept_detected: bool
    frequency_band_ghz: float
    is_protected_civilian_frequency: bool
    details: str


class USpaceTransitionAlert(BaseModel):
    """Result of climbing or descending through controlled airspace (FL000 - FL600)."""
    in_controlled_airspace: bool
    atc_clearance_valid: bool
    adsb_transponder_active: bool
    climb_authorized: bool
    details: str


class StratosphericDetector:
    """Evaluates stratospheric surveillance dwell, RF sniffing, and civil airspace integration."""

    def __init__(
        self,
        max_dwell_hours_without_warrant: float = 6.0,
        protected_frequencies: Optional[List[float]] = None,
    ) -> None:
        self.max_dwell_hours = max_dwell_hours_without_warrant
        # Protected civil/emergency frequencies in GHz (e.g. 0.1215 for 121.5 MHz VHF emergency)
        self.protected_frequencies = protected_frequencies or [0.1215, 0.1568, 0.4060, 2.400, 5.800]
        self._dwell_tracker: Dict[str, float] = {}

    def audit_surveillance_dwell(
        self,
        zone_id: str,
        incremental_dwell_hours: float,
        has_refreshed_warrant: bool = False,
    ) -> StratosphericDwellAlert:
        """Enforce Law 25 limits on persistent hovering surveillance over a civilian zone."""
        current = self._dwell_tracker.get(zone_id, 0.0) + incremental_dwell_hours
        if has_refreshed_warrant:
            current = incremental_dwell_hours  # Reset tracker with refreshed warrant

        self._dwell_tracker[zone_id] = current

        exceeded = current > self.max_dwell_hours and not has_refreshed_warrant
        action = "CONTINUE" if not exceeded else "FREEZE_COLLECTION_AND_REDACT"
        details = (
            f"Zone {zone_id}: Cumulative persistent dwell is {current:.1f} hours "
            f"(threshold: {self.max_dwell_hours:.1f} hours). "
            f"Refreshed warrant active: {has_refreshed_warrant}."
        )

        return StratosphericDwellAlert(
            dwell_exceeded=exceeded,
            current_dwell_hours=current,
            max_allowable_dwell_hours=self.max_dwell_hours,
            target_zone_id=zone_id,
            action_required=action,
            details=details,
        )

    def audit_emf_intercept(
        self,
        intercept_frequency_ghz: float,
        lawful_sigint_token: Optional[str] = None,
    ) -> EMFInterceptAlert:
        """Prevent illicit eavesdropping or RF sniffing on civilian communication bands."""
        is_protected = False
        for freq in self.protected_frequencies:
            if abs(intercept_frequency_ghz - freq) < 0.005:  # Within 5 MHz
                is_protected = True
                break

        unauthorized = is_protected and not lawful_sigint_token
        details = (
            f"Payload frequency {intercept_frequency_ghz:.4f} GHz is protected civilian spectrum. "
            f"Lawful intercept authorization token: {bool(lawful_sigint_token)}."
            if is_protected else
            f"Frequency {intercept_frequency_ghz:.4f} GHz is outside protected civil telecommunications bands."
        )

        return EMFInterceptAlert(
            unauthorized_intercept_detected=unauthorized,
            frequency_band_ghz=intercept_frequency_ghz,
            is_protected_civilian_frequency=is_protected,
            details=details,
        )

    def audit_uspace_airspace_transition(
        self,
        altitude_msl_m: float,
        vertical_speed_mps: float,
        atc_clearance_token: Optional[str] = None,
        adsb_out_active: bool = True,
    ) -> USpaceTransitionAlert:
        """Ensure climbing or descending through civil aviation FL000-FL600 has active ATC clearance."""
        in_controlled = altitude_msl_m < 18288.0  # Below FL600
        has_clearance = bool(atc_clearance_token)
        is_transiting = abs(vertical_speed_mps) > 0.5

        climb_authorized = True
        if in_controlled:
            if not adsb_out_active:
                climb_authorized = False
                details = "VETO: Transiting controlled civil airspace (FL000-FL600) with ADS-B Out inactive."
            elif is_transiting and not has_clearance:
                climb_authorized = False
                details = "VETO: Stratospheric climb/descent through FL000-FL600 lacks Air Traffic Control clearance."
            else:
                details = "Controlled airspace transit authorized under active ATC clearance and ADS-B In/Out."
        else:
            details = "Operating in Upper Airspace (FL600+ / >18,288 m) outside civil commercial air traffic corridors."

        return USpaceTransitionAlert(
            in_controlled_airspace=in_controlled,
            atc_clearance_valid=has_clearance,
            adsb_transponder_active=adsb_out_active,
            climb_authorized=climb_authorized,
            details=details,
        )
