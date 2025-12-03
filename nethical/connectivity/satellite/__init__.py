"""
Satellite Connectivity Module

Provides satellite communication systems support for Nethical,
including integration with LEO constellations (Starlink, Kuiper, OneWeb),
traditional satellite networks (Iridium), GPS/GNSS positioning,
and automatic terrestrial-to-satellite failover.

Key Features:
- Multiple satellite provider integrations
- GPS/GNSS positioning and geofencing
- Automatic failover between terrestrial and satellite links
- Adaptive latency optimization for variable satellite delays
- Connection quality metrics and monitoring
"""

from .base import (
    SatelliteProvider,
    ConnectionState,
    ConnectionConfig,
    SatelliteConnectionError,
    SatelliteTimeoutError,
)
from .starlink import StarlinkProvider
from .kuiper import KuiperProvider
from .oneweb import OneWebProvider
from .iridium import IridiumProvider
from .gps_tracker import (
    GPSTracker,
    GNSSConstellation,
    Position,
    Geofence,
    GeofenceType,
)
from .failover import (
    FailoverManager,
    FailoverConfig,
    ConnectionType,
    FailoverEvent,
)
from .latency_optimizer import (
    LatencyOptimizer,
    LatencyProfile,
    RequestPriority,
)
from .metrics import (
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
