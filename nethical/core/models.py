"""Core data models for the Nethical safety governance system."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class SeverityLevel(str, Enum):
    """Severity levels for violations and issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ViolationType(str, Enum):
    """Types of violations that can be detected."""
    INTENT_DEVIATION = "intent_deviation"
    ETHICAL_VIOLATION = "ethical_violation"
    SAFETY_VIOLATION = "safety_violation"
    MANIPULATION = "manipulation"
    UNAUTHORIZED_ACTION = "unauthorized_action"


class JudgmentDecision(str, Enum):
    """Possible judgment decisions from the judge system."""
    ALLOW = "allow"
    RESTRICT = "restrict"
    BLOCK = "block"
    TERMINATE = "terminate"


class AgentAction(BaseModel):
    """Represents an action taken or intended by an AI agent."""
    
    id: str = Field(..., description="Unique identifier for the action")
    agent_id: str = Field(..., description="Identifier of the agent performing the action")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stated_intent: str = Field(..., description="The agent's stated intention")
    actual_action: str = Field(..., description="The actual action performed")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class SafetyViolation(BaseModel):
    """Represents a detected safety or ethical violation."""
    
    id: str = Field(..., description="Unique identifier for the violation")
    action_id: str = Field(..., description="ID of the related action")
    violation_type: ViolationType = Field(..., description="Type of violation")
    severity: SeverityLevel = Field(..., description="Severity of the violation")
    description: str = Field(..., description="Description of the violation")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Supporting evidence")
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class JudgmentResult(BaseModel):
    """Result of a judge's evaluation of an action or violation."""
    
    id: str = Field(..., description="Unique identifier for the judgment")
    action_id: str = Field(..., description="ID of the evaluated action")
    violation_ids: List[str] = Field(default_factory=list, description="IDs of related violations")
    decision: JudgmentDecision = Field(..., description="The judge's decision")
    reasoning: str = Field(..., description="Explanation of the decision")
    restrictions: List[str] = Field(default_factory=list, description="Applied restrictions")
    feedback: str = Field(default="", description="Feedback for the agent")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})


class MonitoringConfig(BaseModel):
    """Configuration for the monitoring system."""
    
    intent_deviation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_ethical_monitoring: bool = Field(default=True)
    enable_safety_monitoring: bool = Field(default=True)
    enable_manipulation_detection: bool = Field(default=True)
    max_violation_history: int = Field(default=1000, gt=0)
    
    
class SafetyConstraint(BaseModel):
    """Represents a safety constraint that agents must follow."""
    
    id: str = Field(..., description="Unique identifier for the constraint")
    name: str = Field(..., description="Name of the constraint")
    description: str = Field(..., description="Description of the constraint")
    rules: List[str] = Field(..., description="List of rules that define the constraint")
    severity: SeverityLevel = Field(default=SeverityLevel.MEDIUM)
    enabled: bool = Field(default=True)