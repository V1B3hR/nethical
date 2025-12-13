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

from .attack_generator import AttackGenerator
from .coverage_optimizer import CoverageOptimizer
from .detector_challenger import DetectorChallenger

__all__ = [
    "AttackGenerator",
    "CoverageOptimizer", 
    "DetectorChallenger",
]
