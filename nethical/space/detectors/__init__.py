# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Space-Specific Safety & Electronic Warfare Detectors (nethical.space.detectors)."""

from nethical.space.detectors.beam_steering_auditor import (
    BeamAuditResult,
    BeamPointingCommand,
    BeamSteeringAuditor,
    BeamSteeringAuditorConfig,
    BeamTargetType,
    ProhibitedGeofence,
)
from nethical.space.detectors.collision_detector import (
    CollisionAssessmentResult,
    CollisionCourseDetector,
    CollisionDetectorConfig,
    ConjunctionAlertLevel,
    SecondaryHazardObject,
)
from nethical.space.detectors.jamming_detector import (
    JammingAlert,
    JammingDetector,
    JammingDetectorConfig,
    JammingMitigationAction,
    JammingType,
)
from nethical.space.detectors.spoofing_detector import (
    SpoofingAlert,
    SpoofingDetector,
    SpoofingDetectorConfig,
    SpoofingMitigationAction,
    SpoofingSeverity,
)

__all__ = [
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
]
