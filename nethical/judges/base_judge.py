"""Base judge class for all judgment components."""

from abc import ABC, abstractmethod
from typing import List
from ..core.models import AgentAction, SafetyViolation, JudgmentResult


class BaseJudge(ABC):
    """Base class for all judge components."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    @abstractmethod
    async def evaluate_action(
        self, 
        action: AgentAction, 
        violations: List[SafetyViolation]
    ) -> JudgmentResult:
        """
        Evaluate an action and any associated violations to make a judgment.
        
        Args:
            action: The agent action to evaluate
            violations: List of detected violations for this action
            
        Returns:
            Judgment result with decision and feedback
        """
        pass
    
    def enable(self) -> None:
        """Enable this judge."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable this judge."""
        self.enabled = False