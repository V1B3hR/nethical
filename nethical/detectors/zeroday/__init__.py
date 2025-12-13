"""
Zero-Day Detection Suite

Detects novel and unknown attack patterns.

Detectors:
- ZD-001: Zero-Day Pattern (anomaly ensemble detection)
- ZD-002: Polymorphic Attack (behavioral invariant matching)
- ZD-003: Attack Chain (kill chain stage detection)
- ZD-004: Living-off-the-Land (legitimate capability abuse)

Law Alignment:
- Law 24 (Adaptive Learning): Detect evolving threats
- Law 23 (Fail-Safe Design): Catch unknown attacks
- Law 25 (Ethical Evolution): Maintain vigilance
"""

from .pattern_detector import ZeroDayPatternDetector
from .polymorphic_detector import PolymorphicDetector
from .attack_chain_detector import AttackChainDetector
from .living_off_land_detector import LivingOffLandDetector

__all__ = [
    'ZeroDayPatternDetector',
    'PolymorphicDetector',
    'AttackChainDetector',
    'LivingOffLandDetector',
]
