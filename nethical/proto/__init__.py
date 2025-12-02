"""Protocol Buffer definitions for Nethical.

This package contains the gRPC protocol definitions for
low-latency inter-service communication.

The proto files can be compiled using:
    protoc --python_out=. --grpc_python_out=. governance.proto

For now, we provide Python dataclasses that mirror the proto messages
until the proto files are compiled.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Violation:
    """Violation message (mirrors proto)."""
    
    id: str
    type: str
    severity: str
    description: str
    law_reference: Optional[str] = None
    evidence: dict = field(default_factory=dict)


@dataclass
class Explanation:
    """Explanation message (mirrors proto)."""
    
    summary: str
    risk_factors: list[str] = field(default_factory=list)
    decision_rationale: str = ""
    laws_applied: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class EvaluateRequest:
    """Evaluate request message (mirrors proto)."""
    
    agent_id: str
    action: str
    action_type: str = "query"
    context: dict = field(default_factory=dict)
    stated_intent: Optional[str] = None
    priority: str = "normal"
    require_explanation: bool = False
    request_id: Optional[str] = None


@dataclass
class EvaluateResponse:
    """Evaluate response message (mirrors proto)."""
    
    decision: str
    decision_id: str
    risk_score: float = 0.0
    confidence: float = 1.0
    latency_ms: int = 0
    violations: list[Violation] = field(default_factory=list)
    reason: str = ""
    explanation: Optional[Explanation] = None
    audit_id: Optional[str] = None
    cache_hit: bool = False
    fundamental_laws_checked: list[int] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class Decision:
    """Decision record message (mirrors proto)."""
    
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


@dataclass
class Policy:
    """Policy message (mirrors proto)."""
    
    policy_id: str
    name: str
    description: str
    version: str
    status: str
    scope: str
    fundamental_laws: list[int] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


__all__ = [
    "Violation",
    "Explanation",
    "EvaluateRequest",
    "EvaluateResponse",
    "Decision",
    "Policy",
]
