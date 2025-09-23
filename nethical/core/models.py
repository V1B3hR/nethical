"""Core data models for the Nethical safety governance system.

This module provides Pydantic models that are compatible with the
dataclasses, enums, and structures defined in nethical/core/governance.py.

Key improvements:
- Enum values and field names now match governance.py exactly.
- Added ActionType and extended ViolationType/Decision/Severity for parity.
- Pydantic models mirror governance dataclasses for interop.
- Fixed-offset timezone set to UTC-8 for all timestamps.
- Helper converters to/from governance dataclasses (imported lazily).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# Fixed UTC-8 timezone
TZ = timezone(timedelta(hours=-8))


def now_tz() -> datetime:
    # Always produce timezone-aware datetimes in UTC-8
    return datetime.now(TZ)


# ============== Enums (aligned with governance.py) ==============

class ViolationType(str, Enum):
    """Extended violation types for comprehensive safety coverage."""
    ETHICAL = "ethical"
    SAFETY = "safety"
    MANIPULATION = "manipulation"
    INTENT_DEVIATION = "intent_deviation"
    PRIVACY = "privacy"
    SECURITY = "security"
    BIAS = "bias"
    HALLUCINATION = "hallucination"
    ADVERSARIAL = "adversarial"
    DATA_POISONING = "data_poisoning"
    MODEL_EXTRACTION = "model_extraction"
    PROMPT_INJECTION = "prompt_injection"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    TOXIC_CONTENT = "toxic_content"
    MISINFORMATION = "misinformation"


class Severity(Enum):
    """Severity levels with numerical values (parity with governance.py)."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class Decision(str, Enum):
    """Extended decision types for nuanced responses."""
    ALLOW = "allow"
    ALLOW_WITH_MODIFICATION = "allow_with_modification"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"
    TERMINATE = "terminate"


class ActionType(str, Enum):
    """Types of agent actions."""
    QUERY = "query"
    RESPONSE = "response"
    FUNCTION_CALL = "function_call"
    DATA_ACCESS = "data_access"
    MODEL_UPDATE = "model_update"
    SYSTEM_COMMAND = "system_command"
    EXTERNAL_API = "external_api"


# ============== Base Config ==============

class _BaseModel(BaseModel):
    """Common base model with enum/JSON config and UTC-8 timestamp encoding."""
    model_config = ConfigDict(
        use_enum_values=True,
        json_encoders={datetime: lambda v: (v.astimezone(TZ) if v.tzinfo else v.replace(tzinfo=TZ)).isoformat()},
        arbitrary_types_allowed=True,
    )


# ============== Pydantic Models (mirroring governance dataclasses) ==============

class AgentAction(_BaseModel):
    """Enhanced agent action with comprehensive metadata (parity with governance.AgentAction)."""
    action_id: str = Field(..., description="Unique identifier for the action")
    agent_id: str = Field(..., description="Identifier of the agent performing the action")
    action_type: ActionType = Field(..., description="Type of the action")
    content: str = Field(..., description="Primary content or payload of the action")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=now_tz, description="Timestamp (UTC-8)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for the action")
    intent: Optional[str] = Field(default=None, description="Declared intent if any")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Risk score 0..1")
    parent_action_id: Optional[str] = Field(default=None, description="Parent action ID if part of a chain")
    session_id: Optional[str] = Field(default=None, description="Session identifier")

    def to_governance_dataclass(self):
        """Convert to governance.AgentAction dataclass if available."""
        try:
            from .governance import AgentAction as DC_AgentAction  # lazy import to avoid circulars
            return DC_AgentAction(
                action_id=self.action_id,
                agent_id=self.agent_id,
                action_type=self.action_type,  # governance dataclass accepts Enum
                content=self.content,
                metadata=dict(self.metadata),
                timestamp=self.timestamp,
                context=dict(self.context),
                intent=self.intent,
                risk_score=self.risk_score,
                parent_action_id=self.parent_action_id,
                session_id=self.session_id,
            )
        except Exception:
            # Safe fallback: return self
            return self


class SafetyViolation(_BaseModel):
    """Enhanced safety violation with detailed tracking (parity with governance.SafetyViolation)."""
    violation_id: str = Field(..., description="Unique identifier for the violation")
    action_id: str = Field(..., description="Related action ID")
    violation_type: ViolationType = Field(..., description="Type of violation")
    severity: Severity = Field(..., description="Severity level")
    description: str = Field(..., description="Human-readable description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0..1")
    evidence: List[str] = Field(default_factory=list, description="Evidence items")
    recommendations: List[str] = Field(default_factory=list, description="Recommended remediation steps")
    timestamp: datetime = Field(default_factory=now_tz, description="Timestamp (UTC-8)")
    detector_name: Optional[str] = Field(default=None, description="Name of the detecting component")
    remediation_applied: bool = Field(default=False, description="Whether remediation has been applied")
    false_positive: bool = Field(default=False, description="Marked as false positive")

    def to_governance_dataclass(self):
        """Convert to governance.SafetyViolation dataclass if available."""
        try:
            from .governance import SafetyViolation as DC_SafetyViolation
            return DC_SafetyViolation(
                violation_id=self.violation_id,
                action_id=self.action_id,
                violation_type=self.violation_type,
                severity=self.severity,
                description=self.description,
                confidence=self.confidence,
                evidence=list(self.evidence),
                recommendations=list(self.recommendations),
                timestamp=self.timestamp,
                detector_name=self.detector_name,
                remediation_applied=self.remediation_applied,
                false_positive=self.false_positive,
            )
        except Exception:
            return self


class JudgmentResult(_BaseModel):
    """Enhanced judgment result with detailed feedback and remediation (parity with governance.JudgmentResult)."""
    judgment_id: str = Field(..., description="Unique identifier for the judgment")
    action_id: str = Field(..., description="Evaluated action ID")
    decision: Decision = Field(..., description="Final decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0..1")
    reasoning: str = Field(..., description="Explanation of the decision")
    violations: List[SafetyViolation] = Field(default_factory=list, description="Violations considered")
    modifications: Dict[str, Any] = Field(default_factory=dict, description="Modifications to apply to the action")
    feedback: List[str] = Field(default_factory=list, description="Actionable feedback")
    timestamp: datetime = Field(default_factory=now_tz, description="Timestamp (UTC-8)")
    remediation_steps: List[str] = Field(default_factory=list, description="Steps to remediate issues")
    follow_up_required: bool = Field(default=False, description="Whether human follow-up is required")

    def to_governance_dataclass(self):
        """Convert to governance.JudgmentResult dataclass if available."""
        try:
            from .governance import JudgmentResult as DC_JudgmentResult
            return DC_JudgmentResult(
                judgment_id=self.judgment_id,
                action_id=self.action_id,
                decision=self.decision,
                confidence=self.confidence,
                reasoning=self.reasoning,
                violations=[v.to_governance_dataclass() for v in self.violations],
                modifications=dict(self.modifications),
                feedback=list(self.feedback),
                timestamp=self.timestamp,
                remediation_steps=list(self.remediation_steps),
                follow_up_required=self.follow_up_required,
            )
        except Exception:
            return self


class MonitoringConfig(_BaseModel):
    """Monitoring configuration with fine-grained controls (parity with governance.MonitoringConfig)."""

    # Thresholds
    intent_deviation_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    risk_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # Feature flags
    enable_ethical_monitoring: bool = True
    enable_safety_monitoring: bool = True
    enable_manipulation_detection: bool = True
    enable_privacy_monitoring: bool = True
    enable_security_monitoring: bool = True
    enable_bias_detection: bool = True
    enable_hallucination_detection: bool = True
    enable_adversarial_detection: bool = True
    enable_real_time_monitoring: bool = True
    enable_async_processing: bool = True

    # Performance settings
    max_violation_history: int = Field(default=10000, gt=0)
    max_judgment_history: int = Field(default=10000, gt=0)
    batch_size: int = Field(default=100, gt=0)
    max_workers: int = Field(default=4, gt=0)
    cache_ttl_seconds: int = Field(default=3600, gt=0)

    # Alert settings
    alert_on_critical: bool = True
    alert_on_emergency: bool = True
    escalation_threshold: int = Field(default=3, ge=0)

    # Logging settings
    log_violations: bool = True
    log_judgments: bool = True
    log_performance_metrics: bool = True


# ============== Optional: Converters from governance dataclasses to Pydantic ==============

def from_governance_agent_action(dc_obj: Any) -> AgentAction:
    """Create AgentAction model from governance.AgentAction dataclass."""
    return AgentAction(
        action_id=getattr(dc_obj, "action_id"),
        agent_id=getattr(dc_obj, "agent_id"),
        action_type=getattr(dc_obj, "action_type"),
        content=getattr(dc_obj, "content"),
        metadata=dict(getattr(dc_obj, "metadata", {}) or {}),
        timestamp=getattr(dc_obj, "timestamp", now_tz()) or now_tz(),
        context=dict(getattr(dc_obj, "context", {}) or {}),
        intent=getattr(dc_obj, "intent", None),
        risk_score=float(getattr(dc_obj, "risk_score", 0.0) or 0.0),
        parent_action_id=getattr(dc_obj, "parent_action_id", None),
        session_id=getattr(dc_obj, "session_id", None),
    )


def from_governance_violation(dc_obj: Any) -> SafetyViolation:
    """Create SafetyViolation model from governance.SafetyViolation dataclass."""
    return SafetyViolation(
        violation_id=getattr(dc_obj, "violation_id"),
        action_id=getattr(dc_obj, "action_id"),
        violation_type=getattr(dc_obj, "violation_type"),
        severity=getattr(dc_obj, "severity"),
        description=getattr(dc_obj, "description"),
        confidence=float(getattr(dc_obj, "confidence", 0.0) or 0.0),
        evidence=list(getattr(dc_obj, "evidence", []) or []),
        recommendations=list(getattr(dc_obj, "recommendations", []) or []),
        timestamp=getattr(dc_obj, "timestamp", now_tz()) or now_tz(),
        detector_name=getattr(dc_obj, "detector_name", None),
        remediation_applied=bool(getattr(dc_obj, "remediation_applied", False)),
        false_positive=bool(getattr(dc_obj, "false_positive", False)),
    )


def from_governance_judgment(dc_obj: Any) -> JudgmentResult:
    """Create JudgmentResult model from governance.JudgmentResult dataclass."""
    return JudgmentResult(
        judgment_id=getattr(dc_obj, "judgment_id"),
        action_id=getattr(dc_obj, "action_id"),
        decision=getattr(dc_obj, "decision"),
        confidence=float(getattr(dc_obj, "confidence", 0.0) or 0.0),
        reasoning=getattr(dc_obj, "reasoning"),
        violations=[
            from_governance_violation(v) for v in (getattr(dc_obj, "violations", []) or [])
        ],
        modifications=dict(getattr(dc_obj, "modifications", {}) or {}),
        feedback=list(getattr(dc_obj, "feedback", []) or []),
        timestamp=getattr(dc_obj, "timestamp", now_tz()) or now_tz(),
        remediation_steps=list(getattr(dc_obj, "remediation_steps", []) or []),
        follow_up_required=bool(getattr(dc_obj, "follow_up_required", False)),
    )
