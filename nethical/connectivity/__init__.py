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
]
