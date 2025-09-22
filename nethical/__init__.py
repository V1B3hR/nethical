"""
Nethical: Safety governance system for AI agents.

This package provides comprehensive monitoring and safety governance
for AI agents, including:
- Intent vs action deviation monitoring
- Ethical and safety constraint violation detection
- Manipulation technique recognition
- Judge system for action evaluation and feedback
"""

from .core.governance import SafetyGovernance
from .core.models import AgentAction, SafetyViolation, JudgmentResult, MonitoringConfig

__version__ = "0.1.0"
__all__ = ["SafetyGovernance", "AgentAction", "SafetyViolation", "JudgmentResult", "MonitoringConfig"]