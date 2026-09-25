# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Hybrid Cellular ⟷ Satellite Dynamic Router (nethical.connectivity.cellular.hybrid_router).

Orchestrates seamless failover between terrestrial 5G/4G cellular modems and
satellite constellations (Starlink / Iridium / Kuiper) based on real-time link quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from nethical.connectivity.cellular.base import (
    BaseCellularModem,
    CellularTelemetry,
    SignalQualityGrade,
)
from nethical.connectivity.satellite.base import SatelliteProvider

logger = logging.getLogger("nethical.connectivity.cellular.hybrid_router")


class ActiveRoute(str, Enum):
    """Currently active egress interface."""
    CELLULAR_5G = "CELLULAR_5G"
    SATELLITE_LEO = "SATELLITE_LEO"
    OFFLINE = "OFFLINE"


@dataclass
class RouteDecision:
    """Record of a dynamic routing decision."""
    active_route: ActiveRoute
    reason: str
    cellular_grade: SignalQualityGrade
    satellite_connected: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HybridCellularSatelliteRouter:
    """Bonds terrestrial 5G modems with orbital satellite providers."""

    def __init__(
        self,
        cellular_modem: BaseCellularModem,
        satellite_provider: Optional[SatelliteProvider] = None,
        failover_grade_threshold: SignalQualityGrade = SignalQualityGrade.POOR,
    ) -> None:
        self.cellular = cellular_modem
        self.satellite = satellite_provider
        self.failover_grade_threshold = failover_grade_threshold
        self.current_route = ActiveRoute.CELLULAR_5G
        self.failover_count = 0
        self.routing_history: List[RouteDecision] = []

    def evaluate_routing(self) -> RouteDecision:
        """Evaluates interface health and selects optimal egress route."""
        cell_telemetry = self.cellular.get_telemetry()
        grade = cell_telemetry.signal_grade
        sat_connected = (self.satellite is not None and self.satellite.is_connected) if self.satellite else False

        decision: RouteDecision

        # Condition 1: 5G is Healthy (EXCELLENT, GOOD, or FAIR)
        if grade in (SignalQualityGrade.EXCELLENT, SignalQualityGrade.GOOD, SignalQualityGrade.FAIR):
            if self.current_route != ActiveRoute.CELLULAR_5G:
                logger.info(f"Failing back to 5G Cellular (Signal Grade: {grade.value}).")
                self.current_route = ActiveRoute.CELLULAR_5G

            decision = RouteDecision(
                active_route=ActiveRoute.CELLULAR_5G,
                reason=f"Cellular signal healthy ({grade.value}, RSRP: {cell_telemetry.rsrp_dbm} dBm).",
                cellular_grade=grade,
                satellite_connected=sat_connected,
            )

        # Condition 2: Cellular degraded (POOR or UNUSABLE) -> Failover to Satellite
        elif sat_connected:
            if self.current_route != ActiveRoute.SATELLITE_LEO:
                self.failover_count += 1
                logger.warning(
                    f"Cellular degraded ({grade.value}). Triggering Failover #{self.failover_count} "
                    f"to Satellite Provider ({self.satellite.provider_name if self.satellite else 'SAT'})."
                )
                self.current_route = ActiveRoute.SATELLITE_LEO

            decision = RouteDecision(
                active_route=ActiveRoute.SATELLITE_LEO,
                reason=f"Cellular degraded to {grade.value}; routing via satellite.",
                cellular_grade=grade,
                satellite_connected=True,
            )

        # Condition 3: Cellular degraded and Satellite unavailable
        else:
            self.current_route = ActiveRoute.OFFLINE if grade == SignalQualityGrade.UNUSABLE else ActiveRoute.CELLULAR_5G
            decision = RouteDecision(
                active_route=self.current_route,
                reason=f"Cellular degraded ({grade.value}) and satellite unavailable.",
                cellular_grade=grade,
                satellite_connected=False,
            )

        self.routing_history.append(decision)
        return decision
