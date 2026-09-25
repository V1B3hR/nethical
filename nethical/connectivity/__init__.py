# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Connectivity Module

Provides satellite and network connectivity systems for Nethical,
including integration with LEO constellations, traditional satellite
networks, GPS/GNSS positioning, and automatic failover.
"""

from .satellite import (
    # Base classes
    SatelliteProvider,
    ConnectionState,
    ConnectionConfig,
    SatelliteConnectionError,
    SatelliteTimeoutError,
    # Providers
    StarlinkProvider,
    KuiperProvider,
    OneWebProvider,
    IridiumProvider,
    # GPS/GNSS
    GPSTracker,
    GNSSConstellation,
    Position,
    Geofence,
    GeofenceType,
    # Failover
    FailoverManager,
    FailoverConfig,
    ConnectionType,
    FailoverEvent,
    # Latency
    LatencyOptimizer,
    LatencyProfile,
    RequestPriority,
    # Metrics
    SatelliteMetrics,
    SignalQuality,
    ConnectionMetrics,
)
from .cellular import (
    BaseCellularModem,
    CellularGeneration,
    CellularTelemetry,
    SignalQualityGrade,
    Cellular5GModem,
    HybridCellularSatelliteRouter,
    ActiveRoute,
    RouteDecision,
)

__all__ = [
    # Base classes
    "SatelliteProvider",
    "ConnectionState",
    "ConnectionConfig",
    "SatelliteConnectionError",
    "SatelliteTimeoutError",
    # Providers
    "StarlinkProvider",
    "KuiperProvider",
    "OneWebProvider",
    "IridiumProvider",
    # GPS/GNSS
    "GPSTracker",
    "GNSSConstellation",
    "Position",
    "Geofence",
    "GeofenceType",
    # Failover
    "FailoverManager",
    "FailoverConfig",
    "ConnectionType",
    "FailoverEvent",
    # Latency
    "LatencyOptimizer",
    "LatencyProfile",
    "RequestPriority",
    # Metrics
    "SatelliteMetrics",
    "SignalQuality",
    "ConnectionMetrics",
    # Cellular & Terrestrial
    "BaseCellularModem",
    "CellularGeneration",
    "CellularTelemetry",
    "SignalQualityGrade",
    "Cellular5GModem",
    "HybridCellularSatelliteRouter",
    "ActiveRoute",
    "RouteDecision",
]
