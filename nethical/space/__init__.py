# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Sovereign Orbital & Stratospheric Space Operations (nethical.space).

Provides autonomous space governance, astrodynamical state modeling,
RF/optical link budget verification, electronic warfare detectors, and orbital collision avoidance.
"""

from nethical.space.detectors import (
    BeamAuditResult,
    BeamPointingCommand,
    BeamSteeringAuditor,
    BeamSteeringAuditorConfig,
    BeamTargetType,
    CollisionAssessmentResult,
    CollisionCourseDetector,
    CollisionDetectorConfig,
    ConjunctionAlertLevel,
    JammingAlert,
    JammingDetector,
    JammingDetectorConfig,
    JammingMitigationAction,
    JammingType,
    ProhibitedGeofence,
    SecondaryHazardObject,
    SpoofingAlert,
    SpoofingDetector,
    SpoofingDetectorConfig,
    SpoofingMitigationAction,
    SpoofingSeverity,
)
from nethical.space.models import (
    BOLTZMANN_CONSTANT_J_K,
    EARTH_EQUATORIAL_RADIUS_KM,
    GEO_ALTITUDE_KM,
    MU_EARTH_KM3_S2,
    SPEED_OF_LIGHT_M_S,
    ConstellationTopology,
    Covariance3D,
    GroundStationContact,
    HAPSFlightState,
    InterSatelliteLink,
    ISLLinkStatus,
    LinkBudget,
    OrbitalRegime,
    OrbitalState,
    Vector3D,
)
from nethical.space.orbital_governor import (
    OrbitalAction,
    OrbitalGovernor,
    OrbitalGovernorConfig,
    OrbitalSafetyDecision,
)

__all__ = [
    # Physical Constants
    "MU_EARTH_KM3_S2",
    "EARTH_EQUATORIAL_RADIUS_KM",
    "SPEED_OF_LIGHT_M_S",
    "BOLTZMANN_CONSTANT_J_K",
    "GEO_ALTITUDE_KM",
    # Domain Models
    "OrbitalRegime",
    "Vector3D",
    "Covariance3D",
    "OrbitalState",
    "LinkBudget",
    "ISLLinkStatus",
    "InterSatelliteLink",
    "GroundStationContact",
    "ConstellationTopology",
    "HAPSFlightState",
    # Detectors
    "JammingDetector",
    "JammingDetectorConfig",
    "JammingAlert",
    "JammingType",
    "JammingMitigationAction",
    "SpoofingDetector",
    "SpoofingDetectorConfig",
    "SpoofingAlert",
    "SpoofingSeverity",
    "SpoofingMitigationAction",
    "CollisionCourseDetector",
    "CollisionDetectorConfig",
    "CollisionAssessmentResult",
    "ConjunctionAlertLevel",
    "SecondaryHazardObject",
    "BeamSteeringAuditor",
    "BeamSteeringAuditorConfig",
    "BeamPointingCommand",
    "BeamAuditResult",
    "BeamTargetType",
    "ProhibitedGeofence",
    # Governor
    "OrbitalGovernor",
    "OrbitalGovernorConfig",
    "OrbitalSafetyDecision",
    "OrbitalAction",
]
