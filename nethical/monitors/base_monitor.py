"""Base monitor class for all monitoring components."""

from abc import ABC, abstractmethod
from typing import List, Optional
from ..core.models import AgentAction, SafetyViolation


class BaseMonitor(ABC):
    """Base class for all monitoring components."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    @abstractmethod
    async def analyze_action(self, action: AgentAction) -> List[SafetyViolation]:
        """
        Analyze an agent action and return any detected violations.
        
        Args:
            action: The agent action to analyze
            
        Returns:
            List of detected safety violations
        """
        pass
    
    def enable(self) -> None:
        """Enable this monitor."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable this monitor."""
        self.enabled = False