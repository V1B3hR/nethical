"""
Model Security Detection Suite

This module provides detection for model-level security attacks
as defined in Roadmap_Maturity.md Phase 2.3.

Detectors:
- ExtractionDetector (MS-001): Model extraction via API queries
- MembershipInferenceDetector (MS-002): Training data membership inference
- InversionDetector (MS-003): Model inversion attacks
- BackdoorDetector (MS-004): Backdoor activation attempts

Author: Nethical Core Team
Version: 1.0.0
"""

from .extraction_detector import ExtractionDetector
from .membership_inference_detector import MembershipInferenceDetector
from .inversion_detector import InversionDetector
from .backdoor_detector import BackdoorDetector

__all__ = [
    "ExtractionDetector",
    "MembershipInferenceDetector",
    "InversionDetector",
    "BackdoorDetector",
]
