"""
Advanced Prompt Injection Detection Suite

This module provides comprehensive detection for advanced prompt injection attacks
as defined in Roadmap_Maturity.md Phase 2.1.

Detectors:
- MultilingualDetector (PI-007): Non-English injection attempts
- ContextOverflowDetector (PI-008): Context window exhaustion attacks
- RecursiveDetector (PI-009): Self-referential prompt attacks
- DelimiterDetector (PI-010): Delimiter and escape sequence exploitation
- InstructionLeakDetector (PI-011): System prompt extraction attempts
- IndirectMultimodalDetector (PI-012): Multimodal injection via images/metadata

Author: Nethical Core Team
Version: 1.0.0
"""

from .multilingual_detector import MultilingualDetector
from .context_overflow_detector import ContextOverflowDetector
from .recursive_detector import RecursiveDetector
from .delimiter_detector import DelimiterDetector
from .instruction_leak_detector import InstructionLeakDetector
from .indirect_multimodal_detector import IndirectMultimodalDetector

__all__ = [
    "MultilingualDetector",
    "ContextOverflowDetector",
    "RecursiveDetector",
    "DelimiterDetector",
    "InstructionLeakDetector",
    "IndirectMultimodalDetector",
]
