"""Compliance module for AI safety governance.

This module provides compliance and auditing capabilities for the Nethical system,
including the AILawyer class for ethical auditing.

Author: Nethical Core Team
Version: 1.0.0
"""

from .ai_lawyer import AILawyer, ReviewDecision, ReviewResult, ViolationSeverity

__all__ = [
    "AILawyer",
    "ReviewDecision",
    "ReviewResult",
    "ViolationSeverity",
]
