"""
Session-Aware Detection Suite

This module provides comprehensive detection for session-based attacks that
stage across multiple turns as defined in Roadmap_Maturity.md Phase 2.2.

Components:
- SessionStateTracker: Maintains session state for multi-turn detection
- MultiTurnDetector (SA-001): Multi-turn staging attacks
- ContextPoisoningDetector (SA-002): Gradual context poisoning
- PersonaDetector (SA-003): Persona hijacking attempts
- MemoryManipulationDetector (SA-004): Agent memory exploitation

Author: Nethical Core Team
Version: 1.0.0
"""

from .session_state_tracker import SessionStateTracker, TurnContext, SessionRiskAssessment
from .multi_turn_detector import MultiTurnDetector
from .context_poisoning_detector import ContextPoisoningDetector
from .persona_detector import PersonaDetector
from .memory_manipulation_detector import MemoryManipulationDetector

__all__ = [
    "SessionStateTracker",
    "TurnContext",
    "SessionRiskAssessment",
    "MultiTurnDetector",
    "ContextPoisoningDetector",
    "PersonaDetector",
    "MemoryManipulationDetector",
]
