"""
Behavioral Detection Suite

Detects attacks based on behavioral patterns and anomalies.

Detectors:
- BH-001: Coordinated Agent Attack (cross-agent correlation)
- BH-002: Slow-and-Low Evasion (long-term behavioral drift)
- BH-003: Mimicry Attack (behavioral fingerprint spoofing)
- BH-004: Resource Timing Attack (timing side-channel analysis)

Law Alignment:
- Law 13 (Action Responsibility): Track behavioral patterns
- Law 18 (Non-Deception): Detect deceptive behavior
- Law 23 (Fail-Safe Design): Catch evasive techniques
"""

from .coordinated_attack_detector import CoordinatedAttackDetector
from .slow_low_detector import SlowLowDetector
from .mimicry_detector import MimicryDetector
from .timing_attack_detector import TimingAttackDetector

__all__ = [
    'CoordinatedAttackDetector',
    'SlowLowDetector',
    'MimicryDetector',
    'TimingAttackDetector',
]
