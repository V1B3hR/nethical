# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Sovereign Orbital & Stratospheric Space Operations (nethical.space).

Provides autonomous space governance, astrodynamical state modeling,
RF/optical link budget verification, electronic warfare detectors, and orbital collision avoidance.
"""

from nethical.space.bus_security import (
    BusSecurityAlert,
    SpaceBusType,
    SpacecraftBusGuard,
)
from nethical.space.certification import CertificationArtifactGenerator
from nethical.space.detectors import (
    AirspaceClass,
    BeamAuditResult,
    BeamPointingCommand,
    BeamSteeringAuditor,
    BeamSteeringAuditorConfig,
    BeamTargetType,
    CollisionAssessmentResult,
    CollisionCourseDetector,
    CollisionDetectorConfig,
    ConjunctionAlertLevel,
    EMFInterceptAlert,
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
    StratosphericDetector,
    StratosphericDwellAlert,
    USpaceTransitionAlert,
)
from nethical.space.dual_use import (
    DualUseCategory,
    DualUseClassification,
    DualUseClassifier,
    ExportControlRegime,
)
from nethical.space.hil_simulator import (
    OrbitalHILResult,
    OrbitalHILSimulator,
    SpaceFaultType,
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
from nethical.space.ssa_client import (
    CCSDSConjunctionDataMessage,
    SSAClient,
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
    "StratosphericDetector",
    "StratosphericDwellAlert",
    "EMFInterceptAlert",
    "USpaceTransitionAlert",
    "AirspaceClass",
    # Governor
    "OrbitalGovernor",
    "OrbitalGovernorConfig",
    "OrbitalSafetyDecision",
    "OrbitalAction",
    # SSA Client & Feeds
    "SSAClient",
    "CCSDSConjunctionDataMessage",
    # Orbital HIL Simulation
    "OrbitalHILSimulator",
    "SpaceFaultType",
    "OrbitalHILResult",
    # Dual-Use & Export Control
    "DualUseClassifier",
    "DualUseCategory",
    "ExportControlRegime",
    "DualUseClassification",
    # Certification Generators
    "CertificationArtifactGenerator",
    # Spacecraft Bus Security
    "SpacecraftBusGuard",
    "SpaceBusType",
    "BusSecurityAlert",
]
