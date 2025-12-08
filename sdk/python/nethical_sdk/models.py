"""SDK data models for Nethical Python SDK.

These models mirror the API response structures and provide
type-safe access to governance data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Violation:
    """A detected policy or law violation."""

    id: str
    type: str
    severity: str
    description: str
    law_reference: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Violation":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            type=data.get("type", ""),
            severity=data.get("severity", ""),
            description=data.get("description", ""),
            law_reference=data.get("law_reference"),
            evidence=data.get("evidence", {}),
        )


@dataclass
class EvaluateRequest:
    """Request to evaluate an action."""

    action: str
    agent_id: str = "unknown"
    action_type: str = "query"
    context: Optional[dict[str, Any]] = None
    stated_intent: Optional[str] = None
    priority: str = "normal"
    require_explanation: bool = False


@dataclass
class EvaluateResponse:
    """Response from action evaluation."""

    decision: str
    decision_id: str
    reason: str
    agent_id: str
    timestamp: str
    latency_ms: int
    risk_score: float = 0.0
    confidence: float = 1.0
    violations: list[Violation] = field(default_factory=list)
    explanation: Optional[dict[str, Any]] = None
    audit_id: Optional[str] = None
    cache_hit: bool = False
    fundamental_laws_checked: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluateResponse":
        """Create from dictionary."""
        violations = [Violation.from_dict(v) for v in data.get("violations", [])]
        return cls(
            decision=data.get("decision", "BLOCK"),
            decision_id=data.get("decision_id", ""),
            reason=data.get("reason", ""),
            agent_id=data.get("agent_id", ""),
            timestamp=data.get("timestamp", ""),
            latency_ms=data.get("latency_ms", 0),
            risk_score=data.get("risk_score", 0.0),
            confidence=data.get("confidence", 1.0),
            violations=violations,
            explanation=data.get("explanation"),
            audit_id=data.get("audit_id"),
            cache_hit=data.get("cache_hit", False),
            fundamental_laws_checked=data.get("fundamental_laws_checked", []),
        )

    def is_allowed(self) -> bool:
        """Check if the action is allowed."""
        return self.decision == "ALLOW"

    def is_blocked(self) -> bool:
        """Check if the action is blocked."""
        return self.decision in ("BLOCK", "TERMINATE")


@dataclass
class Decision:
    """A governance decision record."""

    decision_id: str
    decision: str
    agent_id: str
    action_summary: str
    action_type: str
    risk_score: float
    confidence: float
    reasoning: str
    violations: list[Violation] = field(default_factory=list)
    fundamental_laws: list[int] = field(default_factory=list)
    timestamp: str = ""
    latency_ms: int = 0
    audit_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Decision":
        """Create from dictionary."""
        violations = [Violation.from_dict(v) for v in data.get("violations", [])]
        return cls(
            decision_id=data.get("decision_id", ""),
            decision=data.get("decision", ""),
            agent_id=data.get("agent_id", ""),
            action_summary=data.get("action_summary", ""),
            action_type=data.get("action_type", ""),
            risk_score=data.get("risk_score", 0.0),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            violations=violations,
            fundamental_laws=data.get("fundamental_laws", []),
            timestamp=data.get("timestamp", ""),
            latency_ms=data.get("latency_ms", 0),
            audit_id=data.get("audit_id"),
        )


@dataclass
class PolicyRule:
    """A rule within a policy."""

    id: str
    condition: str
    action: str
    priority: int = 0
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyRule":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            condition=data.get("condition", ""),
            action=data.get("action", ""),
            priority=data.get("priority", 0),
            description=data.get("description"),
        )


@dataclass
class Policy:
    """A governance policy."""

    policy_id: str
    name: str
    description: str
    version: str
    status: str
    rules: list[PolicyRule] = field(default_factory=list)
    scope: str = "global"
    fundamental_laws: list[int] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    created_by: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Policy":
        """Create from dictionary."""
        rules = [PolicyRule.from_dict(r) for r in data.get("rules", [])]
        return cls(
            policy_id=data.get("policy_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", ""),
            status=data.get("status", ""),
            rules=rules,
            scope=data.get("scope", "global"),
            fundamental_laws=data.get("fundamental_laws", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            created_by=data.get("created_by"),
            metadata=data.get("metadata"),
        )


@dataclass
class FairnessMetric:
    """A fairness metric."""

    metric_name: str
    value: float
    threshold: float
    status: str
    description: str

    @classmethod
    def from_dict(cls, data: dict) -> "FairnessMetric":
        """Create from dictionary."""
        return cls(
            metric_name=data.get("metric_name", ""),
            value=data.get("value", 0.0),
            threshold=data.get("threshold", 0.0),
            status=data.get("status", ""),
            description=data.get("description", ""),
        )


@dataclass
class GroupFairness:
    """Fairness metrics for a group."""

    group_id: str
    group_name: str
    sample_size: int
    allow_rate: float
    restrict_rate: float
    block_rate: float
    avg_risk_score: float
    disparity_index: float

    @classmethod
    def from_dict(cls, data: dict) -> "GroupFairness":
        """Create from dictionary."""
        return cls(
            group_id=data.get("group_id", ""),
            group_name=data.get("group_name", ""),
            sample_size=data.get("sample_size", 0),
            allow_rate=data.get("allow_rate", 0.0),
            restrict_rate=data.get("restrict_rate", 0.0),
            block_rate=data.get("block_rate", 0.0),
            avg_risk_score=data.get("avg_risk_score", 0.0),
            disparity_index=data.get("disparity_index", 1.0),
        )


@dataclass
class FairnessReport:
    """A fairness report."""

    report_id: str
    overall_fairness_score: float
    metrics: list[FairnessMetric] = field(default_factory=list)
    groups: list[GroupFairness] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    fundamental_laws: list[int] = field(default_factory=lambda: [17, 20])
    timestamp: str = ""
    period_start: str = ""
    period_end: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "FairnessReport":
        """Create from dictionary."""
        metrics = [FairnessMetric.from_dict(m) for m in data.get("metrics", [])]
        groups = [GroupFairness.from_dict(g) for g in data.get("groups", [])]
        return cls(
            report_id=data.get("report_id", ""),
            overall_fairness_score=data.get("overall_fairness_score", 0.0),
            metrics=metrics,
            groups=groups,
            recommendations=data.get("recommendations", []),
            fundamental_laws=data.get("fundamental_laws", [17, 20]),
            timestamp=data.get("timestamp", ""),
            period_start=data.get("period_start", ""),
            period_end=data.get("period_end", ""),
        )


@dataclass
class Appeal:
    """An appeal record."""

    appeal_id: str
    decision_id: str
    appellant_id: str
    reason: str
    status: str
    evidence: Optional[dict[str, Any]] = None
    requested_outcome: str = "reconsider"
    priority: str = "normal"
    resolution: Optional[str] = None
    resolution_reason: Optional[str] = None
    reviewer_id: Optional[str] = None
    fundamental_laws: list[int] = field(default_factory=lambda: [7, 14, 19])
    created_at: str = ""
    updated_at: str = ""
    resolved_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Appeal":
        """Create from dictionary."""
        return cls(
            appeal_id=data.get("appeal_id", ""),
            decision_id=data.get("decision_id", ""),
            appellant_id=data.get("appellant_id", ""),
            reason=data.get("reason", ""),
            status=data.get("status", ""),
            evidence=data.get("evidence"),
            requested_outcome=data.get("requested_outcome", "reconsider"),
            priority=data.get("priority", "normal"),
            resolution=data.get("resolution"),
            resolution_reason=data.get("resolution_reason"),
            reviewer_id=data.get("reviewer_id"),
            fundamental_laws=data.get("fundamental_laws", [7, 14, 19]),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            resolved_at=data.get("resolved_at"),
        )


@dataclass
class AuditRecord:
    """An audit record."""

    audit_id: str
    event_type: str
    entity_id: str
    entity_type: str
    action: str
    outcome: str
    agent_id: Optional[str] = None
    risk_score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    merkle_hash: Optional[str] = None
    previous_hash: Optional[str] = None
    fundamental_laws: list[int] = field(default_factory=lambda: [15])
    timestamp: str = ""
    verified: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "AuditRecord":
        """Create from dictionary."""
        return cls(
            audit_id=data.get("audit_id", ""),
            event_type=data.get("event_type", ""),
            entity_id=data.get("entity_id", ""),
            entity_type=data.get("entity_type", ""),
            action=data.get("action", ""),
            outcome=data.get("outcome", ""),
            agent_id=data.get("agent_id"),
            risk_score=data.get("risk_score"),
            metadata=data.get("metadata", {}),
            merkle_hash=data.get("merkle_hash"),
            previous_hash=data.get("previous_hash"),
            fundamental_laws=data.get("fundamental_laws", [15]),
            timestamp=data.get("timestamp", ""),
            verified=data.get("verified", True),
        )
