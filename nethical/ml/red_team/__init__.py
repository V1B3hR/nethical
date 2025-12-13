"""
Autonomous Red Team System for Nethical

This module implements the Phase 4 Autonomous Red Team capability that
continuously tests and improves Nethical's detection capabilities.

Components:
- AttackGenerator: ML-based generation of novel attack variants
- CoverageOptimizer: Identifies gaps in detection coverage
- DetectorChallenger: Continuously probes detectors for weaknesses

Phase 4 Objective: Self-updating detection with minimal human intervention

Author: Nethical Core Team
Version: 1.0.0
"""

from .attack_generator import (
    AttackGenerator,
    AttackCategory,
    GenerationMethod,
    AttackVariant,
)
from .coverage_optimizer import (
    CoverageOptimizer,
    CoverageGap,
    CoverageReport,
)
from .detector_challenger import (
    DetectorChallenger,
    ChallengeType,
    DetectorWeakness,
    ChallengeResult,
    DetectorProfile,
)

__all__ = [
    "AttackGenerator",
    "AttackCategory",
    "GenerationMethod",
    "AttackVariant",
    "CoverageOptimizer",
    "CoverageGap",
    "CoverageReport",
    "DetectorChallenger",
    "ChallengeType",
    "DetectorWeakness",
    "ChallengeResult",
    "DetectorProfile",
]
