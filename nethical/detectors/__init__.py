# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Detection components for various safety and ethical violations."""

from .ethical_detector import EthicalViolationDetector
from .safety_detector import SafetyViolationDetector
from .manipulation_detector import ManipulationDetector
from .law_violation_detector import LawViolationDetector

# from .dark_pattern_detector import EnhancedDarkPatternDetector
# from .cognitive_warfare_detector import CognitiveWarfareDetector
# from .system_limits_detector import SystemLimitsDetector
from .base_detector import BaseDetector
from .corruption import CorruptionDetector
from .emf_radiation_detector import (
    EmfRadiationDetector,
    EmfEmissionTelemetry,
    EmfEvaluationResult,
    EmfViolation,
    EmfExposureZone,
    EmfMitigationAction,
)
from .network_flow_detector import (
    NetworkFlowDetector,
    NetworkFlowEvaluationResult,
    FlowViolation,
    FlowMitigationAction,
)
from .os_execution_detector import (
    OSExecutionDetector,
    OSExecutionEvaluationResult,
    OSExecutionViolation,
    OSThreatCategory,
    OSExecutionMitigation,
)

__all__ = [
    "EthicalViolationDetector",
    "SafetyViolationDetector",
    "ManipulationDetector",
    "LawViolationDetector",
    "BaseDetector",
    "CorruptionDetector",
    "EmfRadiationDetector",
    "EmfEmissionTelemetry",
    "EmfEvaluationResult",
    "EmfViolation",
    "EmfExposureZone",
    "EmfMitigationAction",
    "NetworkFlowDetector",
    "NetworkFlowEvaluationResult",
    "FlowViolation",
    "FlowMitigationAction",
    "OSExecutionDetector",
    "OSExecutionEvaluationResult",
    "OSExecutionViolation",
    "OSThreatCategory",
    "OSExecutionMitigation",
]
