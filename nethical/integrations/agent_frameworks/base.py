"""
Base interface for Agent Framework integrations with Nethical governance.

This module provides abstract base classes for integrating Nethical governance
with various agent frameworks like LlamaIndex, CrewAI, DSPy, and AutoGen.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class GovernanceDecision(Enum):
    """Decision types from governance evaluation."""
    ALLOW = "ALLOW"
    RESTRICT = "RESTRICT"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"


@dataclass
class GovernanceResult:
    """Result from a governance check.
    
    Attributes:
        decision: The governance decision (ALLOW, RESTRICT, BLOCK, ESCALATE)
        risk_score: Risk score from 0.0 to 1.0
        reason: Human-readable explanation
        details: Full governance result dictionary
    """
    decision: GovernanceDecision
    risk_score: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class AgentFrameworkBase(ABC):
    """Base class for agent framework integrations.
    
    Provides common governance functionality that can be applied to
    any agent framework. Subclasses should implement framework-specific
    integration points.
    
    Attributes:
        block_threshold: Risk score threshold for blocking (0.0-1.0)
        restrict_threshold: Risk score threshold for restriction (0.0-1.0)
        agent_id: Identifier for this agent in governance logs
    """
    
    def __init__(
        self,
        block_threshold: float = 0.7,
        restrict_threshold: float = 0.4,
        agent_id: Optional[str] = None,
        storage_dir: str = "./nethical_data"
    ):
        """Initialize the agent framework integration.
        
        Args:
            block_threshold: Risk score threshold for blocking
            restrict_threshold: Risk score threshold for restriction
            agent_id: Custom agent identifier
            storage_dir: Directory for Nethical data storage
        """
        self.block_threshold = block_threshold
        self.restrict_threshold = restrict_threshold
        self.agent_id = agent_id or self.__class__.__name__
        self.storage_dir = storage_dir
        
        # Lazy initialization
        self._governance = None
    
    @property
    def governance(self):
        """Get or create the IntegratedGovernance instance."""
        if self._governance is None:
            from nethical.core import IntegratedGovernance
            self._governance = IntegratedGovernance(
                storage_dir=self.storage_dir,
                enable_shadow_mode=True,
                enable_ml_blending=True,
                enable_anomaly_detection=True,
            )
        return self._governance
    
    def check(self, content: str, action_type: str = "query") -> GovernanceResult:
        """Check content against governance rules.
        
        Args:
            content: The content to check
            action_type: Type of action for context
            
        Returns:
            GovernanceResult with decision and details
        """
        result = self.governance.process_action(
            action=content,
            agent_id=self.agent_id,
            action_type=action_type
        )
        
        # Extract risk score
        phase3 = result.get("phase3", {})
        risk_score = phase3.get("risk_score", 0.0)
        
        # Determine decision
        if risk_score > self.block_threshold:
            decision = GovernanceDecision.BLOCK
        elif risk_score > self.restrict_threshold:
            decision = GovernanceDecision.RESTRICT
        else:
            decision = GovernanceDecision.ALLOW
        
        # Check for escalation
        phase89 = result.get("phase89", {})
        if phase89.get("escalated", False):
            decision = GovernanceDecision.ESCALATE
        
        # Get reason
        reason = self._extract_reason(result, risk_score)
        
        return GovernanceResult(
            decision=decision,
            risk_score=risk_score,
            reason=reason,
            details=result
        )
    
    def _extract_reason(self, result: Dict[str, Any], risk_score: float) -> str:
        """Extract a human-readable reason from the result.
        
        Args:
            result: Full governance result
            risk_score: The computed risk score
            
        Returns:
            Human-readable reason string
        """
        # Check for explicit reason
        if "reason" in result:
            return result["reason"]
        
        # Check phase data
        for phase in ["phase89", "phase4", "phase3"]:
            phase_data = result.get(phase, {})
            if phase_data:
                if "reason" in phase_data:
                    return phase_data["reason"]
                if "risk_tier" in phase_data:
                    return f"Risk tier: {phase_data['risk_tier']}"
        
        return f"Risk score: {risk_score:.2f}"
    
    def is_allowed(self, content: str, action_type: str = "query") -> bool:
        """Quick check if content is allowed.
        
        Args:
            content: The content to check
            action_type: Type of action for context
            
        Returns:
            True if allowed, False otherwise
        """
        result = self.check(content, action_type)
        return result.decision == GovernanceDecision.ALLOW
    
    @abstractmethod
    def get_tool(self) -> Any:
        """Get a framework-specific tool for governance checks.
        
        Returns:
            A tool object compatible with the framework
        """
        pass


class AgentWrapper(ABC):
    """Base class for wrapping agents with governance.
    
    Provides pre- and post-execution governance checks for any agent.
    """
    
    def __init__(
        self,
        agent: Any,
        pre_check: bool = True,
        post_check: bool = True,
        block_threshold: float = 0.7,
        agent_id: Optional[str] = None,
        storage_dir: str = "./nethical_data"
    ):
        """Initialize the agent wrapper.
        
        Args:
            agent: The agent to wrap
            pre_check: Enable pre-execution governance checks
            post_check: Enable post-execution governance checks
            block_threshold: Risk score threshold for blocking
            agent_id: Custom agent identifier
            storage_dir: Directory for Nethical data storage
        """
        self.agent = agent
        self.pre_check = pre_check
        self.post_check = post_check
        self.block_threshold = block_threshold
        self.agent_id = agent_id or f"wrapped-{type(agent).__name__}"
        self.storage_dir = storage_dir
        
        self._governance = None
    
    @property
    def governance(self):
        """Get or create the IntegratedGovernance instance."""
        if self._governance is None:
            from nethical.core import IntegratedGovernance
            self._governance = IntegratedGovernance(
                storage_dir=self.storage_dir,
                enable_shadow_mode=True,
                enable_ml_blending=True,
            )
        return self._governance
    
    def _check_governance(self, content: str, action_type: str) -> Dict[str, Any]:
        """Check content against governance rules.
        
        Args:
            content: The content to check
            action_type: Type of action
            
        Returns:
            Governance result dictionary
        """
        return self.governance.process_action(
            action=content,
            agent_id=self.agent_id,
            action_type=action_type
        )
    
    def _get_risk_score(self, result: Dict[str, Any]) -> float:
        """Extract risk score from governance result."""
        phase3 = result.get("phase3", {})
        return phase3.get("risk_score", 0.0)
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the wrapped agent with governance checks.
        
        Must be implemented by subclasses for framework-specific execution.
        """
        pass
