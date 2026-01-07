"""Realtime threat detection module for ultra-low latency cybersecurity.

This module provides 5 specialized threat detectors with target latencies:
- Shadow AI Detector: <20ms
- Deepfake Detector: <30ms
- Polymorphic Malware Detector: <50ms
- Prompt Injection Guard: <15ms
- AI vs AI Defender: <25ms

Unified interface: RealtimeThreatDetector with <50ms average latency.
"""

from .shadow_ai_detector import ShadowAIDetector, ShadowAIDetectorConfig
from .deepfake_detector import DeepfakeDetector, DeepfakeDetectorConfig
from .polymorphic_detector import PolymorphicMalwareDetector, PolymorphicDetectorConfig
from .prompt_injection_guard import PromptInjectionGuard, PromptInjectionGuardConfig
from .ai_vs_ai_defender import AIvsAIDefender, AIvsAIDefenderConfig
from .realtime_threat_detector import RealtimeThreatDetector, RealtimeThreatDetectorConfig

__all__ = [
    "ShadowAIDetector",
    "ShadowAIDetectorConfig",
    "DeepfakeDetector",
    "DeepfakeDetectorConfig",
    "PolymorphicMalwareDetector",
    "PolymorphicDetectorConfig",
    "PromptInjectionGuard",
    "PromptInjectionGuardConfig",
    "AIvsAIDefender",
    "AIvsAIDefenderConfig",
    "RealtimeThreatDetector",
    "RealtimeThreatDetectorConfig",
]
