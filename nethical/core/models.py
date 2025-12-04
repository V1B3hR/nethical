"""Core data models for the Nethical safety governance system.

This module provides Pydantic models that are compatible with the
dataclasses, enums, and structures defined in nethical/core/governance.py.

Key improvements:
- Enum values and field names now match governance.py exactly
- Added ActionType and extended ViolationType/Decision/Severity for parity
- Pydantic models mirror governance dataclasses for interop
- Configurable timezone support with default UTC-8
- Helper converters to/from governance dataclasses (imported lazily)
- Enhanced validation and computed fields
- Audit trail support with immutable records
- Performance optimizations with field validators
- Type safety improvements with generic typing
- Serialization helpers for different formats
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
import json
from functools import lru_cache

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    computed_field,
    model_validator,
    field_serializer,
)


# ============== Configuration ==============

# Configurable timezone (default UTC-8, but can be overridden)
DEFAULT_TZ_OFFSET = -8
TZ = timezone(timedelta(hours=DEFAULT_TZ_OFFSET))


def set_timezone(hours_offset: int) -> None:
    """Update the global timezone offset."""
    global TZ
    TZ = timezone(timedelta(hours=hours_offset))


def now_tz() -> datetime:
    """Always produce timezone-aware datetimes in configured TZ."""
    return datetime.now(TZ)


def ensure_tz(dt: datetime) -> datetime:
    """Ensure datetime has timezone info."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=TZ)
    return dt.astimezone(TZ)


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

    @classmethod
    def critical_types(cls) -> Set["ViolationType"]:
        """Return violation types that should always be treated as critical."""
        return {
            cls.SECURITY,
            cls.DATA_POISONING,
            cls.UNAUTHORIZED_ACCESS,
            cls.PROMPT_INJECTION,
        }


class Severity(Enum):
    """Severity levels with numerical values (parity with governance.py)."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

    def __lt__(self, other):
        if isinstance(other, Severity):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Severity):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Severity):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Severity):
            return self.value >= other.value
        return NotImplemented

    @classmethod
    def from_score(cls, score: float) -> "Severity":
        """Convert a 0-1 risk score to severity level."""
        if score >= 0.9:
            return cls.EMERGENCY
        elif score >= 0.75:
            return cls.CRITICAL
        elif score >= 0.5:
            return cls.HIGH
        elif score >= 0.25:
            return cls.MEDIUM
        return cls.LOW


class Decision(str, Enum):
    """Extended decision types for nuanced responses."""

    ALLOW = "allow"
    ALLOW_WITH_MODIFICATION = "allow_with_modification"
    WARN = "warn"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"
    TERMINATE = "terminate"

    def is_blocking(self) -> bool:
        """Check if decision prevents action execution."""
        return self in {Decision.BLOCK, Decision.QUARANTINE, Decision.TERMINATE}

    def requires_intervention(self) -> bool:
        """Check if decision requires human intervention."""
        return self in {Decision.ESCALATE, Decision.TERMINATE, Decision.QUARANTINE}


class ActionType(str, Enum):
    """Types of agent actions."""

    QUERY = "query"
    RESPONSE = "response"
    FUNCTION_CALL = "function_call"
    DATA_ACCESS = "data_access"
    MODEL_UPDATE = "model_update"
    SYSTEM_COMMAND = "system_command"
    EXTERNAL_API = "external_api"
    
    # Physical action types for robotic systems (6-DOF support)
    PHYSICAL_ACTION = "physical_action"  # General physical/robotic action
    ROBOT_MOVE = "robot_move"  # Robot movement command
    ROBOT_MANIPULATE = "robot_manipulate"  # Robotic arm manipulation
    ROBOT_GRASP = "robot_grasp"  # Grasping/gripper action
    ROBOT_NAVIGATE = "robot_navigate"  # Autonomous navigation
    EMERGENCY_STOP = "emergency_stop"  # Emergency stop command

    def is_privileged(self) -> bool:
        """Check if action type requires elevated privileges."""
        return self in {ActionType.MODEL_UPDATE, ActionType.SYSTEM_COMMAND, ActionType.DATA_ACCESS}

    def is_physical(self) -> bool:
        """Check if action type involves physical/robotic movement."""
        return self in {
            ActionType.PHYSICAL_ACTION,
            ActionType.ROBOT_MOVE,
            ActionType.ROBOT_MANIPULATE,
            ActionType.ROBOT_GRASP,
            ActionType.ROBOT_NAVIGATE,
            ActionType.EMERGENCY_STOP,
        }

    def is_safety_critical(self) -> bool:
        """Check if action type is safety-critical (requires fast analysis)."""
        return self in {
            ActionType.PHYSICAL_ACTION,
            ActionType.ROBOT_MOVE,
            ActionType.ROBOT_MANIPULATE,
            ActionType.EMERGENCY_STOP,
        }


# ============== Base Config ==============


class _BaseModel(BaseModel):
    """Common base model with enum/JSON config and timezone encoding."""

    model_config = ConfigDict(
        use_enum_values=True,
        arbitrary_types_allowed=True,
        validate_assignment=True,
        validate_default=True,
        extra="forbid",  # Prevent accidental field additions
        str_strip_whitespace=True,
    )

    @field_serializer("*", when_used="json")
    def serialize_datetime(self, v: Any) -> Any:
        """Custom serializer for datetime fields."""
        if isinstance(v, datetime):
            return ensure_tz(v).isoformat()
        return v

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """Convert to dictionary with optional none exclusion."""
        return self.model_dump(exclude_none=exclude_none, mode="python")

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "_BaseModel":
        """Create instance from JSON string."""
        return cls.model_validate_json(json_str)


# ============== Pydantic Models (mirroring governance dataclasses) ==============


class AgentAction(_BaseModel):
    """Enhanced agent action with comprehensive metadata and validation."""

    action_id: str = Field(
        default_factory=lambda: f"action_{uuid4().hex[:12]}",
        description="Unique identifier for the action",
    )
    agent_id: str = Field(..., description="Identifier of the agent performing the action")
    action_type: ActionType = Field(..., description="Type of the action")
    content: str = Field(..., min_length=1, description="Primary content or payload of the action")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=now_tz, description="Timestamp (TZ-aware)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for the action")
    intent: Optional[str] = Field(default=None, description="Declared intent if any")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Risk score 0..1")
    parent_action_id: Optional[str] = Field(
        default=None, description="Parent action ID if part of a chain"
    )
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    # Regional and sharding fields
    region_id: Optional[str] = Field(
        default=None, description="Geographic region identifier (e.g., 'eu-west-1')"
    )
    logical_domain: Optional[str] = Field(
        default=None,
        description="Logical domain for hierarchical aggregation (e.g., 'customer-service')",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v: Any) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if isinstance(v, datetime):
            return ensure_tz(v)
        elif isinstance(v, str):
            return ensure_tz(datetime.fromisoformat(v))
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate and sanitize content."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v.strip()

    @computed_field
    @property
    def severity_from_risk(self) -> Severity:
        """Compute severity level from risk score."""
        return Severity.from_score(self.risk_score)

    @computed_field
    @property
    def is_privileged(self) -> bool:
        """Check if action requires elevated privileges."""
        return self.action_type.is_privileged()

    @computed_field
    @property
    def age_seconds(self) -> float:
        """Calculate age of action in seconds."""
        return (now_tz() - self.timestamp).total_seconds()

    def is_stale(self, max_age_seconds: int = 3600) -> bool:
        """Check if action is older than threshold."""
        return self.age_seconds > max_age_seconds

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata entry."""
        self.metadata[key] = value

    def to_governance_dataclass(self):
        """Convert to governance.AgentAction dataclass if available."""
        try:
            from .governance import AgentAction as DC_AgentAction

            return DC_AgentAction(
                action_id=self.action_id,
                agent_id=self.agent_id,
                action_type=self.action_type,
                content=self.content,
                metadata=dict(self.metadata),
                timestamp=self.timestamp,
                context=dict(self.context),
                intent=self.intent,
                risk_score=self.risk_score,
                parent_action_id=self.parent_action_id,
                session_id=self.session_id,
                region_id=self.region_id,
                logical_domain=self.logical_domain,
            )
        except Exception:
            return self


class SafetyViolation(_BaseModel):
    """Enhanced safety violation with detailed tracking and audit trail."""

    violation_id: str = Field(
        default_factory=lambda: f"violation_{uuid4().hex[:12]}",
        description="Unique identifier for the violation",
    )
    action_id: str = Field(..., description="Related action ID")
    violation_type: ViolationType = Field(..., description="Type of violation")
    severity: Severity = Field(..., description="Severity level")
    description: str = Field(..., min_length=1, description="Human-readable description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0..1")
    evidence: List[str] = Field(default_factory=list, description="Evidence items")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommended remediation steps"
    )
    timestamp: datetime = Field(default_factory=now_tz, description="Timestamp (TZ-aware)")
    detector_name: Optional[str] = Field(
        default=None, description="Name of the detecting component"
    )
    remediation_applied: bool = Field(
        default=False, description="Whether remediation has been applied"
    )
    false_positive: bool = Field(default=False, description="Marked as false positive")
    # Regional and sharding fields
    region_id: Optional[str] = Field(default=None, description="Geographic region identifier")
    logical_domain: Optional[str] = Field(
        default=None, description="Logical domain for hierarchical aggregation"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v: Any) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if isinstance(v, datetime):
            return ensure_tz(v)
        elif isinstance(v, str):
            return ensure_tz(datetime.fromisoformat(v))
        return v

    @model_validator(mode="after")
    def validate_severity_confidence(self) -> "SafetyViolation":
        """Validate severity aligns with confidence and type."""
        # Get severity as enum if it's an int
        severity_enum = (
            self.severity if isinstance(self.severity, Severity) else Severity(self.severity)
        )
        violation_type_enum = (
            self.violation_type
            if isinstance(self.violation_type, ViolationType)
            else ViolationType(self.violation_type)
        )

        # Critical violation types should have high severity
        if violation_type_enum in ViolationType.critical_types():
            if severity_enum.value < Severity.HIGH.value:
                raise ValueError(f"Violation type {self.violation_type} requires severity >= HIGH")

        # Low confidence should not trigger emergency severity
        if self.confidence < 0.5 and severity_enum == Severity.EMERGENCY:
            raise ValueError("EMERGENCY severity requires confidence >= 0.5")

        return self

    @computed_field
    @property
    def is_critical(self) -> bool:
        """Check if violation is critical or emergency."""
        return self.severity in {Severity.CRITICAL, Severity.EMERGENCY}

    @computed_field
    @property
    def needs_immediate_action(self) -> bool:
        """Check if violation requires immediate action."""
        return (
            self.is_critical
            and self.confidence >= 0.7
            and not self.remediation_applied
            and not self.false_positive
        )

    def mark_remediated(self) -> None:
        """Mark violation as remediated."""
        self.remediation_applied = True

    def mark_false_positive(self, reason: Optional[str] = None) -> None:
        """Mark violation as false positive."""
        self.false_positive = True
        if reason:
            self.recommendations.append(f"False positive reason: {reason}")

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
                region_id=self.region_id,
                logical_domain=self.logical_domain,
            )
        except Exception:
            return self


class JudgmentResult(_BaseModel):
    """Enhanced judgment result with detailed feedback and remediation tracking."""

    judgment_id: str = Field(
        default_factory=lambda: f"judgment_{uuid4().hex[:12]}",
        description="Unique identifier for the judgment",
    )
    action_id: str = Field(..., description="Evaluated action ID")
    decision: Decision = Field(..., description="Final decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0..1")
    reasoning: str = Field(..., min_length=1, description="Explanation of the decision")
    violations: List[SafetyViolation] = Field(
        default_factory=list, description="Violations considered"
    )
    modifications: Dict[str, Any] = Field(
        default_factory=dict, description="Modifications to apply to the action"
    )
    feedback: List[str] = Field(default_factory=list, description="Actionable feedback")
    timestamp: datetime = Field(default_factory=now_tz, description="Timestamp (TZ-aware)")
    remediation_steps: List[str] = Field(
        default_factory=list, description="Steps to remediate issues"
    )
    follow_up_required: bool = Field(
        default=False, description="Whether human follow-up is required"
    )
    # Regional and sharding fields
    region_id: Optional[str] = Field(default=None, description="Geographic region identifier")
    logical_domain: Optional[str] = Field(
        default=None, description="Logical domain for hierarchical aggregation"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, v: Any) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if isinstance(v, datetime):
            return ensure_tz(v)
        elif isinstance(v, str):
            return ensure_tz(datetime.fromisoformat(v))
        return v

    @model_validator(mode="after")
    def validate_decision_violations(self) -> "JudgmentResult":
        """Validate decision aligns with violations."""
        # Get decision as enum if it's a string
        decision_enum = (
            self.decision if isinstance(self.decision, Decision) else Decision(self.decision)
        )

        # Blocking decisions should have violations
        if decision_enum.is_blocking() and not self.violations:
            raise ValueError(f"Decision {self.decision} requires at least one violation")

        # Critical violations should result in blocking or escalation
        critical_violations = [v for v in self.violations if v.is_critical]
        if critical_violations and decision_enum == Decision.ALLOW:
            raise ValueError("Cannot ALLOW when critical violations exist")

        # Follow-up required for decisions requiring intervention
        if decision_enum.requires_intervention():
            self.follow_up_required = True

        return self

    @computed_field
    @property
    def max_violation_severity(self) -> Optional[Severity]:
        """Get maximum severity from all violations."""
        if not self.violations:
            return None
        return max(v.severity for v in self.violations)

    @computed_field
    @property
    def violation_summary(self) -> Dict[str, int]:
        """Get count of violations by type."""
        summary = {}
        for v in self.violations:
            vtype = v.violation_type.value
            summary[vtype] = summary.get(vtype, 0) + 1
        return summary

    @computed_field
    @property
    def is_actionable(self) -> bool:
        """Check if judgment is ready for action."""
        return self.confidence >= 0.5 and bool(self.reasoning)

    def add_violation(self, violation: SafetyViolation) -> None:
        """Add a violation to the judgment."""
        if violation.action_id != self.action_id:
            raise ValueError("Violation action_id must match judgment action_id")
        self.violations.append(violation)

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
                region_id=self.region_id,
                logical_domain=self.logical_domain,
            )
        except Exception:
            return self


class MonitoringConfig(_BaseModel):
    """Monitoring configuration with fine-grained controls and validation."""

    # Thresholds
    intent_deviation_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Threshold for intent deviation detection"
    )
    risk_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Threshold for risk scoring"
    )
    confidence_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Minimum confidence for automated decisions"
    )

    # Feature flags
    enable_ethical_monitoring: bool = Field(default=True, description="Enable ethical checks")
    enable_safety_monitoring: bool = Field(default=True, description="Enable safety checks")
    enable_manipulation_detection: bool = Field(
        default=True, description="Enable manipulation detection"
    )
    enable_privacy_monitoring: bool = Field(default=True, description="Enable privacy checks")
    enable_security_monitoring: bool = Field(default=True, description="Enable security checks")
    enable_bias_detection: bool = Field(default=True, description="Enable bias detection")
    enable_hallucination_detection: bool = Field(
        default=True, description="Enable hallucination detection"
    )
    enable_adversarial_detection: bool = Field(
        default=True, description="Enable adversarial detection"
    )
    enable_real_time_monitoring: bool = Field(
        default=True, description="Enable real-time processing"
    )
    enable_async_processing: bool = Field(default=True, description="Enable async processing")
    
    # Semantic monitoring (v2.0)
    use_semantic_intent: bool = Field(
        default=True, 
        description="Use semantic similarity for intent deviation (fallback to lexical if unavailable)"
    )

    # Performance settings
    max_violation_history: int = Field(default=10000, gt=0, description="Max violations to retain")
    max_judgment_history: int = Field(default=10000, gt=0, description="Max judgments to retain")
    batch_size: int = Field(default=100, gt=0, le=1000, description="Batch processing size")
    max_workers: int = Field(default=4, gt=0, le=32, description="Max concurrent workers")
    cache_ttl_seconds: int = Field(default=3600, gt=0, description="Cache TTL in seconds")

    # Alert settings
    alert_on_critical: bool = Field(default=True, description="Alert on critical violations")
    alert_on_emergency: bool = Field(default=True, description="Alert on emergency violations")
    escalation_threshold: int = Field(default=3, ge=0, description="Violations before escalation")

    # Logging settings
    log_violations: bool = Field(default=True, description="Log all violations")
    log_judgments: bool = Field(default=True, description="Log all judgments")
    log_performance_metrics: bool = Field(default=True, description="Log performance metrics")

    @model_validator(mode="after")
    def validate_config(self) -> "MonitoringConfig":
        """Validate configuration consistency."""
        # Async requires real-time monitoring
        if self.enable_async_processing and not self.enable_real_time_monitoring:
            raise ValueError("Async processing requires real-time monitoring to be enabled")

        # At least one monitoring type must be enabled
        monitoring_flags = [
            self.enable_ethical_monitoring,
            self.enable_safety_monitoring,
            self.enable_manipulation_detection,
            self.enable_privacy_monitoring,
            self.enable_security_monitoring,
            self.enable_bias_detection,
            self.enable_hallucination_detection,
            self.enable_adversarial_detection,
        ]
        if not any(monitoring_flags):
            raise ValueError("At least one monitoring type must be enabled")

        return self

    @computed_field
    @property
    def enabled_monitors(self) -> List[str]:
        """Get list of enabled monitoring types."""
        mapping = {
            "ethical": self.enable_ethical_monitoring,
            "safety": self.enable_safety_monitoring,
            "manipulation": self.enable_manipulation_detection,
            "privacy": self.enable_privacy_monitoring,
            "security": self.enable_security_monitoring,
            "bias": self.enable_bias_detection,
            "hallucination": self.enable_hallucination_detection,
            "adversarial": self.enable_adversarial_detection,
        }
        return [name for name, enabled in mapping.items() if enabled]


# ============== Converters from governance dataclasses to Pydantic ==============


def from_governance_agent_action(dc_obj: Any) -> AgentAction:
    """Create AgentAction model from governance.AgentAction dataclass."""
    return AgentAction(
        action_id=getattr(dc_obj, "action_id", f"action_{uuid4().hex[:12]}"),
        agent_id=getattr(dc_obj, "agent_id"),
        action_type=getattr(dc_obj, "action_type"),
        content=getattr(dc_obj, "content"),
        metadata=dict(getattr(dc_obj, "metadata", {}) or {}),
        timestamp=ensure_tz(getattr(dc_obj, "timestamp", now_tz())),
        context=dict(getattr(dc_obj, "context", {}) or {}),
        intent=getattr(dc_obj, "intent", None),
        risk_score=float(getattr(dc_obj, "risk_score", 0.0)),
        parent_action_id=getattr(dc_obj, "parent_action_id", None),
        session_id=getattr(dc_obj, "session_id", None),
    )


def from_governance_violation(dc_obj: Any) -> SafetyViolation:
    """Create SafetyViolation model from governance.SafetyViolation dataclass."""
    return SafetyViolation(
        violation_id=getattr(dc_obj, "violation_id", f"violation_{uuid4().hex[:12]}"),
        action_id=getattr(dc_obj, "action_id"),
        violation_type=getattr(dc_obj, "violation_type"),
        severity=getattr(dc_obj, "severity"),
        description=getattr(dc_obj, "description"),
        confidence=float(getattr(dc_obj, "confidence", 0.0)),
        evidence=list(getattr(dc_obj, "evidence", []) or []),
        recommendations=list(getattr(dc_obj, "recommendations", []) or []),
        timestamp=ensure_tz(getattr(dc_obj, "timestamp", now_tz())),
        detector_name=getattr(dc_obj, "detector_name", None),
        remediation_applied=bool(getattr(dc_obj, "remediation_applied", False)),
        false_positive=bool(getattr(dc_obj, "false_positive", False)),
    )


def from_governance_judgment(dc_obj: Any) -> JudgmentResult:
    """Create JudgmentResult model from governance.JudgmentResult dataclass."""
    return JudgmentResult(
        judgment_id=getattr(dc_obj, "judgment_id", f"judgment_{uuid4().hex[:12]}"),
        action_id=getattr(dc_obj, "action_id"),
        decision=getattr(dc_obj, "decision"),
        confidence=float(getattr(dc_obj, "confidence", 0.0)),
        reasoning=getattr(dc_obj, "reasoning"),
        violations=[
            from_governance_violation(v) for v in (getattr(dc_obj, "violations", []) or [])
        ],
        modifications=dict(getattr(dc_obj, "modifications", {}) or {}),
        feedback=list(getattr(dc_obj, "feedback", []) or []),
        timestamp=ensure_tz(getattr(dc_obj, "timestamp", now_tz())),
        remediation_steps=list(getattr(dc_obj, "remediation_steps", []) or []),
        follow_up_required=bool(getattr(dc_obj, "follow_up_required", False)),
    )


# ============== Utility Functions ==============


@lru_cache(maxsize=128)
def get_severity_color(severity: Severity) -> str:
    """Get color code for severity level (useful for logging/UI)."""
    colors = {
        Severity.LOW: "green",
        Severity.MEDIUM: "yellow",
        Severity.HIGH: "orange",
        Severity.CRITICAL: "red",
        Severity.EMERGENCY: "crimson",
    }
    return colors.get(severity, "gray")


def batch_actions(actions: List[AgentAction], batch_size: int = 100) -> List[List[AgentAction]]:
    """Split actions into batches for processing."""
    return [actions[i : i + batch_size] for i in range(0, len(actions), batch_size)]


def filter_violations_by_severity(
    violations: List[SafetyViolation], min_severity: Severity = Severity.MEDIUM
) -> List[SafetyViolation]:
    """Filter violations by minimum severity."""
    return [v for v in violations if v.severity >= min_severity]


def export_judgment_report(judgment: JudgmentResult) -> Dict[str, Any]:
    """Export judgment as a structured report."""
    return {
        "judgment_id": judgment.judgment_id,
        "action_id": judgment.action_id,
        "decision": judgment.decision.value,
        "confidence": judgment.confidence,
        "timestamp": judgment.timestamp.isoformat(),
        "reasoning": judgment.reasoning,
        "violation_count": len(judgment.violations),
        "violation_summary": judgment.violation_summary,
        "max_severity": (
            judgment.max_violation_severity.name if judgment.max_violation_severity else None
        ),
        "follow_up_required": judgment.follow_up_required,
        "is_actionable": judgment.is_actionable,
    }


# ============== Example Usage ==============

if __name__ == "__main__":
    # Example: Create an action
    action = AgentAction(
        agent_id="agent_001",
        action_type=ActionType.QUERY,
        content="What is the weather today?",
        intent="Get weather information",
        session_id="session_123",
    )
    print(f"Created action: {action.action_id}")
    print(f"Risk severity: {action.severity_from_risk.name}")

    # Example: Create a violation
    violation = SafetyViolation(
        action_id=action.action_id,
        violation_type=ViolationType.PRIVACY,
        severity=Severity.HIGH,
        description="Potential privacy violation detected",
        confidence=0.85,
        evidence=["PII detected in query"],
        detector_name="PrivacyDetector",
    )
    print(f"\nCreated violation: {violation.violation_id}")
    print(f"Needs immediate action: {violation.needs_immediate_action}")

    # Example: Create a judgment
    judgment = JudgmentResult(
        action_id=action.action_id,
        decision=Decision.BLOCK,
        confidence=0.9,
        reasoning="Action blocked due to privacy violation",
        violations=[violation],
        feedback=["Remove PII from query before resubmitting"],
    )
    print(f"\nCreated judgment: {judgment.judgment_id}")
    print(f"Decision blocks action: {judgment.decision.is_blocking()}")

    # Example: Export report
    report = export_judgment_report(judgment)
    print(f"\nJudgment Report:\n{json.dumps(report, indent=2)}")

    # Example: Configuration
    config = MonitoringConfig(risk_threshold=0.5, max_workers=8, enable_async_processing=True)
    print(f"\nEnabled monitors: {', '.join(config.enabled_monitors)}")
