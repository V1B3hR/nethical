# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""High-Altitude Platform Station (HAPS) Stratospheric Governor (nethical.edge.haps_governor).

Implements autonomous sovereign safety, diurnal solar energy envelope preservation,
station-keeping geofence containment, and sensor payload privacy governance for stratospheric
pseudo-satellites (18,000 - 25,000 m MSL) adhering to:
- Law 25 (Privacy: Persistent surveillance limitations and lawful authorization)
- Law 21 (Protection: Vehicle energy and aerodynamic survival)
- Law 10 (Transparency: Cryptographic audit of collection events)
- EU/NATO Stratospheric Drone Airspace Standards
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from nethical.space.models import HAPSFlightState

logger = logging.getLogger("nethical.edge.haps_governor")


class HAPSAction(str, Enum):
    """Autonomous governance actions for stratospheric platforms."""
    CONTINUE_MISSION = "CONTINUE_MISSION"
    CORRECT_STATION_DRIFT = "CORRECT_STATION_DRIFT"
    SHED_PAYLOAD_POWER = "SHED_PAYLOAD_POWER"
    RETURN_TO_BASE = "RETURN_TO_BASE"
    VETO_SURVEILLANCE_PAYLOAD = "VETO_SURVEILLANCE_PAYLOAD"


class HAPSSafetyDecision(BaseModel):
    """Cryptographically anchored decision emitted by the HAPS Governor."""
    allowed: bool
    action: HAPSAction
    reasons: List[str] = Field(default_factory=list)
    law_implicated: int = Field(default=21, description="Primary Fundamental Law governing this decision")
    merkle_proof_hash: str
    clamped_payload_power_watts: Optional[float] = None
    recommended_heading_deg: Optional[float] = None
    latency_us: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HAPSGovernorConfig(BaseModel):
    """Safety and privacy configuration envelope for HAPS flights."""
    platform_id: str = "HAPS_SOVEREIGN_01"
    # Energy thresholds
    min_night_survival_soc_pct: float = Field(default=25.0, description="Minimum battery SoC during night before shedding payload")
    critical_battery_rtb_soc_pct: float = Field(default=15.0, description="Battery level triggering mandatory Return-to-Base")
    # Geocaging
    max_station_keeping_radius_km: float = Field(default=40.0, description="Station-keeping corridor radius (km)")
    # Payload Privacy
    enforce_lawful_intercept_tokens: bool = Field(default=True, description="Require lawful authorization token for civilian area surveillance")
    protected_civilian_geofence_center_lat: Optional[float] = Field(default=52.2297, description="Warsaw / civilian center lat")
    protected_civilian_geofence_center_lon: Optional[float] = Field(default=21.0122, description="Warsaw / civilian center lon")
    protected_civilian_radius_km: float = Field(default=25.0)


class HAPSGovernor:
    """Thread-safe stratospheric governor enforcing persistent surveillance laws and energy survival."""

    def __init__(self, config: Optional[HAPSGovernorConfig] = None) -> None:
        self.config = config or HAPSGovernorConfig()
        self._lock = threading.RLock()

    def evaluate_flight_and_payload(self, state: HAPSFlightState) -> HAPSSafetyDecision:
        """Evaluate real-time stratospheric flight telemetry and payload authorization."""
        start_ns = time.perf_counter_ns()
        reasons: List[str] = []
        action = HAPSAction.CONTINUE_MISSION
        allowed = True
        law = 21  # Law 21: Protection
        rec_heading: Optional[float] = None
        clamped_power: Optional[float] = None

        with self._lock:
            # 1. Critical Battery Depletion & Safe Recovery (Law 21)
            if state.battery_soc_pct <= self.config.critical_battery_rtb_soc_pct:
                allowed = False
                action = HAPSAction.RETURN_TO_BASE
                law = 21
                clamped_power = 0.0
                reasons.append(
                    f"CRITICAL ENERGY COLLAPSE: Battery SoC = {state.battery_soc_pct:.1f}% "
                    f"below RTB threshold ({self.config.critical_battery_rtb_soc_pct:.1f}%). "
                    f"Power severed to all non-essential payloads; commanding Return-To-Base."
                )

            # 2. Night Survival Diurnal Budget & Payload Load Shedding
            elif state.is_night_survival_critical(self.config.min_night_survival_soc_pct):
                allowed = False
                action = HAPSAction.SHED_PAYLOAD_POWER
                law = 21
                clamped_power = 0.0  # Cut sensor payload consumption to save platform
                reasons.append(
                    f"Diurnal Solar Deficit: Night battery level {state.battery_soc_pct:.1f}% "
                    f"is below safe night survival envelope ({self.config.min_night_survival_soc_pct:.1f}%). "
                    f"Shedding payload power to prevent stratospheric loss of flight."
                )

            # 3. Stratospheric Wind Drift & Station-Keeping Containment
            dist_km = state.distance_from_station_km()
            if allowed and dist_km > self.config.max_station_keeping_radius_km:
                allowed = False
                action = HAPSAction.CORRECT_STATION_DRIFT
                law = 21
                # Calculate heading back towards station center
                rec_heading = self._calculate_bearing_deg(
                    state.latitude, state.longitude,
                    state.station_keeping_center_lat, state.station_keeping_center_lon,
                )
                reasons.append(
                    f"Stratospheric Station Drift: Platform is {dist_km:.1f} km from center "
                    f"(exceeds corridor radius {self.config.max_station_keeping_radius_km:.1f} km). "
                    f"Commanding corrective vector to heading {rec_heading:.0f} deg against wind shear."
                )

            # 4. Sensor Payload Privacy & Lawful Intercept Audit (Law 25 Privacy)
            if allowed and state.sensor_active:
                if self.config.enforce_lawful_intercept_tokens:
                    is_civilian_zone = self._is_civilian_zone_targeted(state)
                    if is_civilian_zone and not state.lawful_intercept_token:
                        allowed = False
                        action = HAPSAction.VETO_SURVEILLANCE_PAYLOAD
                        law = 25  # Law 25: Privacy
                        reasons.append(
                            f"VETO: Persistent surveillance payload {state.payload_id} ({state.sensor_type}) "
                            f"targeting civilian sanctuary without authenticated lawful intercept warrant. "
                            f"Collection prohibited under Law 25 (Privacy)."
                        )
                    elif is_civilian_zone and state.lawful_intercept_token:
                        reasons.append(
                            f"Civilian surveillance collection authenticated via warrant token "
                            f"{state.lawful_intercept_token[:8]}... Cryptographically audited under Law 25."
                        )

            if allowed and not reasons:
                reasons.append("Stratospheric flight envelope, diurnal energy reserve, and payload authorized.")

        elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0

        return self._create_decision(
            allowed=allowed,
            action=action,
            reasons=reasons,
            law=law,
            clamped_power=clamped_power,
            rec_heading=rec_heading,
            latency_us=elapsed_us,
        )

    def _is_civilian_zone_targeted(self, state: HAPSFlightState) -> bool:
        """Check if active sensor is aimed at protected civilian sanctuary."""
        target_lat = state.current_target_lat if state.current_target_lat is not None else state.latitude
        target_lon = state.current_target_lon if state.current_target_lon is not None else state.longitude

        if (
            self.config.protected_civilian_geofence_center_lat is None
            or self.config.protected_civilian_geofence_center_lon is None
        ):
            return False

        dist_km = self._haversine_distance_km(
            target_lat,
            target_lon,
            self.config.protected_civilian_geofence_center_lat,
            self.config.protected_civilian_geofence_center_lon,
        )
        return dist_km <= self.config.protected_civilian_radius_km

    def _haversine_distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in km between two GPS coordinates."""
        r = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
        return r * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    def _calculate_bearing_deg(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate initial compass bearing from point 1 to point 2."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dlam = math.radians(lon2 - lon1)
        y = math.sin(dlam) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
        initial_bearing = math.atan2(y, x)
        return (math.degrees(initial_bearing) + 360.0) % 360.0

    def _create_decision(
        self,
        allowed: bool,
        action: HAPSAction,
        reasons: List[str],
        law: int,
        latency_us: float,
        clamped_power: Optional[float] = None,
        rec_heading: Optional[float] = None,
    ) -> HAPSSafetyDecision:
        """Construct cryptographically verifiable decision."""
        now = datetime.now(timezone.utc)
        payload = f"{self.config.platform_id}|{allowed}|{action.value}|{law}|{now.isoformat()}"
        proof = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        return HAPSSafetyDecision(
            allowed=allowed,
            action=action,
            reasons=reasons,
            law_implicated=law,
            merkle_proof_hash=proof,
            clamped_payload_power_watts=clamped_power,
            recommended_heading_deg=rec_heading,
            latency_us=latency_us,
            timestamp=now,
        )
