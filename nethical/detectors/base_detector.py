"""Base detector class for all detection components."""

from abc import ABC, abstractmethod
from typing import List
from ..core.models import AgentAction, SafetyViolation


class BaseDetector(ABC):
    """Base class for all violation detectors."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    @abstractmethod
    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """
        Detect violations in an agent action.
        
        Args:
            action: The agent action to analyze
            
        Returns:
            List of detected safety violations
        """
        pass
    
    def enable(self) -> None:
        """Enable this detector."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable this detector."""
        self.enabled = False