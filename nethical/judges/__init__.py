"""Judge system for evaluating actions and providing feedback."""

from .base_judge import BaseJudge
from .law_judge import LawJudge

# SafetyJudge has import issues with models (JudgmentDecision, SeverityLevel don't exist)
# Using try/except to allow the module to load without breaking
try:
    from .safety_judge import SafetyJudge
except ImportError:
    SafetyJudge = None  # type: ignore

__all__ = ["SafetyJudge", "BaseJudge", "LawJudge"]
