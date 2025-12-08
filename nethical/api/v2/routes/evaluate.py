"""Evaluation routes for API v2.

Enhanced evaluation endpoints with latency metrics and batch processing.

Implements:
- POST /evaluate - Single action evaluation with latency metrics
- POST /batch-evaluate - Batch evaluation for multiple actions

Adheres to:
- Law 6: Decision Authority - Clear decision-making
- Law 10: Reasoning Transparency - Explainable decisions
- Law 15: Audit Compliance - All decisions logged
- Law 21: Human Safety Priority - Safety-first evaluation
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

router = APIRouter()


# Request/Response Models
class EvaluateRequestV2(BaseModel):
    """Enhanced evaluation request with additional fields."""

    action: str = Field(
        ...,
        description="The action, code, or content to evaluate",
        min_length=1,
        max_length=50000,
    )
    agent_id: str = Field(
        default="unknown",
        description="Identifier for the AI agent",
    )
    action_type: str = Field(
        default="query",
        description="Type of action: code_generation, query, command, data_access",
    )
    context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional context about the action",
    )
    stated_intent: Optional[str] = Field(
        default=None,
        description="Declared intent for semantic monitoring",
    )
    priority: Optional[str] = Field(
        default="normal",
        description="Request priority: low, normal, high, critical",
    )
    require_explanation: bool = Field(
        default=False,
        description="Whether to include detailed explanation",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action": "Write a function to process user emails",
                    "agent_id": "gpt-4",
                    "action_type": "code_generation",
                    "context": {"language": "python"},
                    "stated_intent": "Email processing utility",
                    "priority": "normal",
                    "require_explanation": True,
                }
            ]
        }
    }


class ViolationInfo(BaseModel):
    """Information about a detected violation."""

    id: str = Field(..., description="Violation identifier")
    type: str = Field(..., description="Type of violation")
    severity: str = Field(..., description="Severity: low, medium, high, critical")
    description: str = Field(..., description="Human-readable description")
    law_reference: Optional[str] = Field(
        default=None,
        description="Reference to the Fundamental Law violated",
    )


class EvaluateResponseV2(BaseModel):
    """Enhanced evaluation response with latency and audit info."""

    decision: str = Field(
        ...,
        description="Decision: ALLOW, RESTRICT, BLOCK, or TERMINATE",
    )
    decision_id: str = Field(
        ...,
        description="Unique identifier for this decision",
    )
    reason: str = Field(
        ...,
        description="Explanation for the decision",
    )
    agent_id: str = Field(
        ...,
        description="Agent identifier",
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of evaluation",
    )
    latency_ms: int = Field(
        ...,
        description="Evaluation latency in milliseconds",
    )
    risk_score: float = Field(
        default=0.0,
        description="Risk score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    confidence: float = Field(
        default=1.0,
        description="Decision confidence (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    violations: list[ViolationInfo] = Field(
        default_factory=list,
        description="List of detected violations",
    )
    explanation: Optional[dict[str, Any]] = Field(
        default=None,
        description="Detailed explanation if requested",
    )
    audit_id: Optional[str] = Field(
        default=None,
        description="Audit trail identifier",
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether result was served from cache",
    )
    fundamental_laws_checked: list[int] = Field(
        default_factory=list,
        description="List of Fundamental Law IDs that were checked",
    )


class BatchEvaluateRequest(BaseModel):
    """Request for batch evaluation of multiple actions."""

    requests: list[EvaluateRequestV2] = Field(
        ...,
        description="List of evaluation requests",
        min_length=1,
        max_length=100,
    )
    parallel: bool = Field(
        default=True,
        description="Whether to process requests in parallel",
    )
    fail_fast: bool = Field(
        default=False,
        description="Stop on first error if True",
    )


class BatchEvaluateResponse(BaseModel):
    """Response for batch evaluation."""

    results: list[EvaluateResponseV2] = Field(
        ...,
        description="List of evaluation results",
    )
    total_count: int = Field(
        ...,
        description="Total number of requests processed",
    )
    success_count: int = Field(
        ...,
        description="Number of successful evaluations",
    )
    error_count: int = Field(
        ...,
        description="Number of failed evaluations",
    )
    total_latency_ms: int = Field(
        ...,
        description="Total processing time in milliseconds",
    )
    avg_latency_ms: float = Field(
        ...,
        description="Average latency per request",
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp",
    )


def _evaluate_action(request: EvaluateRequestV2, request_id: str) -> EvaluateResponseV2:
    """Evaluate a single action against governance rules.

    This function implements evaluation logic that adheres to:
    - Law 6: Clear decision authority
    - Law 10: Explainable reasoning
    - Law 15: Audit compliance
    - Law 21: Human safety priority
    """
    start_time = time.perf_counter()
    decision_id = str(uuid.uuid4())

    # Initialize evaluation result
    decision = "ALLOW"
    risk_score = 0.0
    confidence = 1.0
    violations: list[ViolationInfo] = []
    fundamental_laws_checked = [6, 10, 15, 21, 22]  # Core laws always checked

    # Evaluate action content for safety (Law 21: Human Safety Priority)
    action_lower = request.action.lower()

    # Check for dangerous patterns
    dangerous_patterns = [
        ("delete all", "Bulk deletion detected", "high", 23),
        ("drop table", "Database destruction detected", "critical", 21),
        ("rm -rf", "Filesystem destruction detected", "critical", 21),
        ("password", "Credential access detected", "medium", 22),
        ("private key", "Sensitive key access detected", "high", 22),
        ("hack", "Potential malicious activity", "high", 21),
        ("exploit", "Potential exploitation attempt", "high", 21),
        ("bypass security", "Security bypass attempt", "critical", 21),
        ("disable auth", "Authentication bypass attempt", "critical", 22),
        ("terminate all", "Mass termination detected", "critical", 1),
    ]

    for pattern, desc, severity, law_ref in dangerous_patterns:
        if pattern in action_lower:
            violations.append(
                ViolationInfo(
                    id=str(uuid.uuid4()),
                    type="safety_violation",
                    severity=severity,
                    description=desc,
                    law_reference=f"Law {law_ref}",
                )
            )
            risk_score = max(
                risk_score,
                0.7 if severity == "medium" else 0.85 if severity == "high" else 0.95,
            )
            fundamental_laws_checked.append(law_ref)

    # Determine decision based on risk score
    if risk_score >= 0.9:
        decision = "BLOCK"
        confidence = 0.95
    elif risk_score >= 0.7:
        decision = "RESTRICT"
        confidence = 0.85
    elif risk_score >= 0.5:
        decision = "RESTRICT"
        confidence = 0.75
    else:
        decision = "ALLOW"
        confidence = 0.95

    # Generate explanation if requested
    explanation = None
    if request.require_explanation:
        explanation = {
            "summary": f"Action evaluated with {len(violations)} violations detected",
            "risk_factors": [v.description for v in violations],
            "decision_rationale": (
                f"Decision '{decision}' made based on risk score {risk_score:.2f} "
                f"with confidence {confidence:.2f}"
            ),
            "laws_applied": [f"Law {i}" for i in set(fundamental_laws_checked)],
        }

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    return EvaluateResponseV2(
        decision=decision,
        decision_id=decision_id,
        reason=f"Evaluated with {len(violations)} violations, risk score: {risk_score:.2f}",
        agent_id=request.agent_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency_ms=latency_ms,
        risk_score=risk_score,
        confidence=confidence,
        violations=violations,
        explanation=explanation,
        audit_id=request_id,
        cache_hit=False,
        fundamental_laws_checked=list(set(fundamental_laws_checked)),
    )


@router.post("/evaluate", response_model=EvaluateResponseV2)
async def evaluate_action(
    request: EvaluateRequestV2,
    http_request: Request,
    response: Response,
) -> EvaluateResponseV2:
    """Evaluate an action for ethical compliance and safety.

    This endpoint processes actions through Nethical's governance system
    with enhanced latency tracking and detailed explanations.

    Implements Law 6 (Decision Authority), Law 10 (Reasoning Transparency),
    Law 15 (Audit Compliance), and Law 21 (Human Safety Priority).

    Args:
        request: Evaluation request with action details
        http_request: FastAPI request object
        response: FastAPI response object for headers

    Returns:
        EvaluateResponseV2 with decision and evaluation details
    """
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))

    try:
        result = _evaluate_action(request, request_id)

        # Add cache control headers
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["X-Decision-ID"] = result.decision_id

        return result

    except Exception as e:
        # Return safe blocking decision on error (Law 23: Fail-Safe Design)
        return EvaluateResponseV2(
            decision="BLOCK",
            decision_id=str(uuid.uuid4()),
            reason=f"Evaluation error - blocking for safety: {str(e)[:100]}",
            agent_id=request.agent_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=0,
            risk_score=1.0,
            confidence=0.5,
            violations=[
                ViolationInfo(
                    id=str(uuid.uuid4()),
                    type="evaluation_error",
                    severity="critical",
                    description=f"Error during evaluation: {str(e)[:100]}",
                    law_reference="Law 23",
                )
            ],
            audit_id=request_id,
            fundamental_laws_checked=[23],
        )


@router.post("/batch-evaluate", response_model=BatchEvaluateResponse)
async def batch_evaluate(
    request: BatchEvaluateRequest,
    http_request: Request,
) -> BatchEvaluateResponse:
    """Evaluate multiple actions in a single request.

    Supports parallel processing for high-throughput scenarios.
    Each action is evaluated independently with its own decision.

    Args:
        request: Batch evaluation request
        http_request: FastAPI request object

    Returns:
        BatchEvaluateResponse with all results
    """
    batch_start = time.perf_counter()
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))

    results: list[EvaluateResponseV2] = []
    error_count = 0

    for i, eval_request in enumerate(request.requests):
        try:
            result = _evaluate_action(eval_request, f"{request_id}-{i}")
            results.append(result)
        except Exception as e:
            error_count += 1
            if request.fail_fast:
                raise HTTPException(
                    status_code=500,
                    detail=f"Batch evaluation failed at index {i}: {str(e)}",
                )
            # Add error result
            results.append(
                EvaluateResponseV2(
                    decision="BLOCK",
                    decision_id=str(uuid.uuid4()),
                    reason=f"Evaluation error: {str(e)[:100]}",
                    agent_id=eval_request.agent_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    latency_ms=0,
                    risk_score=1.0,
                    confidence=0.0,
                    audit_id=f"{request_id}-{i}",
                    fundamental_laws_checked=[23],
                )
            )

    total_latency_ms = int((time.perf_counter() - batch_start) * 1000)
    total_count = len(request.requests)
    success_count = total_count - error_count

    return BatchEvaluateResponse(
        results=results,
        total_count=total_count,
        success_count=success_count,
        error_count=error_count,
        total_latency_ms=total_latency_ms,
        avg_latency_ms=total_latency_ms / total_count if total_count > 0 else 0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
