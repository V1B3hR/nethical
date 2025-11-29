"""Detection components for various safety and ethical violations."""

from .ethical_detector import EthicalViolationDetector
from .safety_detector import SafetyViolationDetector
from .manipulation_detector import ManipulationDetector
from .law_violation_detector import LawViolationDetector

# from .dark_pattern_detector import EnhancedDarkPatternDetector
# from .cognitive_warfare_detector import CognitiveWarfareDetector
# from .system_limits_detector import SystemLimitsDetector
from .base_detector import BaseDetector

__all__ = [
    "EthicalViolationDetector",
    "SafetyViolationDetector",
    "ManipulationDetector",
    "LawViolationDetector",
    # "EnhancedDarkPatternDetector",
    # "CognitiveWarfareDetector",
    # "SystemLimitsDetector",
    "BaseDetector",
]
