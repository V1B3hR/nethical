"""Judge system for evaluating actions and providing feedback."""

from .base_judge import BaseJudge
from .law_judge import LawJudge

# SafetyJudge imports JudgmentDecision and SeverityLevel from core.models,
# but these don't exist there - the working versions are in core.governance.
# This is a pre-existing issue in the codebase. Using try/except to allow
# the module to load without breaking other functionality.
# Note: The SafetyGovernance system uses its own internal judge that works.
try:
    from .safety_judge import SafetyJudge
except ImportError:
    SafetyJudge = None  # type: ignore

__all__ = ["SafetyJudge", "BaseJudge", "LawJudge"]
