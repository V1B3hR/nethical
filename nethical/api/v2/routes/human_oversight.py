"""Human Oversight API routes for EU AI Act Article 14 compliance.

This module implements human oversight mechanisms required by EU AI Act
Article 14 for high-risk AI systems.

Provides:
- GET /oversight/status - Get current oversight status
- POST /oversight/override - Override an AI decision
- POST /oversight/suspend - Suspend an agent
- POST /oversight/emergency-stop - Emergency stop all operations
- GET /oversight/pending-reviews - Get pending human reviews
- POST /oversight/review - Submit human review decision

Adheres to:
- Law 13: Human Authority - Human decision override
- Law 14: Appeal Rights - Right to appeal AI decisions
- Law 21: Correction Rights - Right to correction

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

router = APIRouter(prefix="/oversight", tags=["Human Oversight"])


# Enums
class OversightMode(str, Enum):
    """Human oversight operation modes per EU AI Act Article 14."""
    MONITORING = "monitoring"  # Passive monitoring
    VERIFICATION = "verification"  # Human verifies each decision
    APPROVAL = "approval"  # Human approves before execution
    DISABLED = "disabled"  # AI operates autonomously


class ReviewPriority(str, Enum):
    """Priority levels for human review requests."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReviewStatus(str, Enum):
    """Status of a human review."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    EXPIRED = "expired"


class OverrideReason(str, Enum):
    """Reasons for overriding AI decisions."""
    FALSE_POSITIVE = "false_positive"
    CONTEXT_NOT_CONSIDERED = "context_not_considered"
    POLICY_EXCEPTION = "policy_exception"
    TESTING = "testing"
    OTHER = "other"


# Request/Response Models
class OversightStatusResponse(BaseModel):
    """Current human oversight status."""
    
    mode: OversightMode = Field(
        ...,
        description="Current oversight operation mode",
    )
    active_reviewers: int = Field(
        default=0,
        description="Number of active human reviewers",
    )
    pending_reviews: int = Field(
        default=0,
        description="Number of decisions awaiting human review",
    )
    average_review_time_seconds: float = Field(
        default=0.0,
        description="Average time for human review completion",
    )
    last_override: Optional[str] = Field(
        default=None,
        description="Timestamp of last manual override",
    )
    emergency_stop_active: bool = Field(
        default=False,
        description="Whether emergency stop is currently active",
    )
    ai_autonomy_level: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Current AI autonomy level (0.0 = full human control, 1.0 = full autonomy)",
    )
    eu_ai_act_compliant: bool = Field(
        default=True,
        description="Whether current configuration is EU AI Act compliant",
    )


class OverrideRequest(BaseModel):
    """Request to override an AI decision."""
    
    decision_id: str = Field(
        ...,
        description="ID of the decision to override",
    )
    override_decision: str = Field(
        ...,
        description="New decision: ALLOW, RESTRICT, BLOCK, or TERMINATE",
    )
    reason: OverrideReason = Field(
        ...,
        description="Reason for the override",
    )
    justification: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Detailed justification for the override",
    )
    apply_retroactively: bool = Field(
        default=False,
        description="Whether to apply override to already-executed actions",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "decision_id": "dec-123e4567-e89b-12d3-a456-426614174000",
                    "override_decision": "ALLOW",
                    "reason": "false_positive",
                    "justification": "The flagged action was a false positive. The user was performing legitimate testing.",
                    "apply_retroactively": False,
                }
            ]
        }
    }


class OverrideResponse(BaseModel):
    """Response confirming a decision override."""
    
    override_id: str = Field(
        ...,
        description="Unique identifier for this override",
    )
    decision_id: str = Field(
        ...,
        description="ID of the overridden decision",
    )
    original_decision: str = Field(
        ...,
        description="Original AI decision",
    )
    new_decision: str = Field(
        ...,
        description="New human-overridden decision",
    )
    reviewer_id: str = Field(
        ...,
        description="ID of the human reviewer",
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of the override",
    )
    audit_logged: bool = Field(
        default=True,
        description="Whether the override was logged for audit",
    )
    fundamental_law_13_applied: bool = Field(
        default=True,
        description="Confirms Law 13 (Human Authority) was applied",
    )


class SuspendAgentRequest(BaseModel):
    """Request to suspend an AI agent."""
    
    agent_id: str = Field(
        ...,
        description="ID of the agent to suspend",
    )
    duration_hours: Optional[int] = Field(
        default=None,
        description="Duration of suspension in hours (None = indefinite)",
    )
    reason: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Reason for suspension",
    )


class SuspendAgentResponse(BaseModel):
    """Response confirming agent suspension."""
    
    suspension_id: str = Field(
        ...,
        description="Unique identifier for this suspension",
    )
    agent_id: str = Field(
        ...,
        description="ID of the suspended agent",
    )
    suspended_at: str = Field(
        ...,
        description="ISO 8601 timestamp of suspension",
    )
    expires_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp when suspension expires",
    )
    active: bool = Field(
        default=True,
        description="Whether the suspension is currently active",
    )


class EmergencyStopRequest(BaseModel):
    """Request to trigger emergency stop."""
    
    scope: str = Field(
        default="all",
        description="Scope: 'all', 'agent', 'domain', or 'region'",
    )
    target_id: Optional[str] = Field(
        default=None,
        description="Target ID for scoped emergency stops",
    )
    reason: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Reason for emergency stop",
    )
    require_manual_restart: bool = Field(
        default=True,
        description="Whether manual intervention is required to restart",
    )


class EmergencyStopResponse(BaseModel):
    """Response confirming emergency stop activation."""
    
    stop_id: str = Field(
        ...,
        description="Unique identifier for this emergency stop",
    )
    scope: str = Field(
        ...,
        description="Scope of the emergency stop",
    )
    activated_at: str = Field(
        ...,
        description="ISO 8601 timestamp of activation",
    )
    affected_agents: int = Field(
        default=0,
        description="Number of agents affected",
    )
    require_manual_restart: bool = Field(
        default=True,
        description="Whether manual restart is required",
    )
    fundamental_law_23_applied: bool = Field(
        default=True,
        description="Confirms Law 23 (Fail-Safe Design) was applied",
    )


class PendingReview(BaseModel):
    """A decision pending human review."""
    
    review_id: str = Field(
        ...,
        description="Unique identifier for this review request",
    )
    decision_id: str = Field(
        ...,
        description="ID of the decision awaiting review",
    )
    agent_id: str = Field(
        ...,
        description="Agent that triggered the review",
    )
    action_summary: str = Field(
        ...,
        description="Summary of the action under review",
    )
    ai_recommendation: str = Field(
        ...,
        description="AI's recommended decision",
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="AI-computed risk score",
    )
    priority: ReviewPriority = Field(
        default=ReviewPriority.MEDIUM,
        description="Review priority level",
    )
    status: ReviewStatus = Field(
        default=ReviewStatus.PENDING,
        description="Current review status",
    )
    created_at: str = Field(
        ...,
        description="ISO 8601 timestamp when review was created",
    )
    expires_at: str = Field(
        ...,
        description="ISO 8601 timestamp when review expires",
    )


class PendingReviewsResponse(BaseModel):
    """Response containing pending reviews."""
    
    reviews: list[PendingReview] = Field(
        default_factory=list,
        description="List of pending reviews",
    )
    total_count: int = Field(
        default=0,
        description="Total number of pending reviews",
    )
    page: int = Field(
        default=1,
        description="Current page number",
    )
    page_size: int = Field(
        default=20,
        description="Number of items per page",
    )


class ReviewDecisionRequest(BaseModel):
    """Request to submit a human review decision."""
    
    review_id: str = Field(
        ...,
        description="ID of the review to complete",
    )
    decision: str = Field(
        ...,
        description="Human decision: APPROVE, REJECT, or MODIFY",
    )
    modified_decision: Optional[str] = Field(
        default=None,
        description="If MODIFY, the new decision to apply",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Reviewer notes",
    )


class ReviewDecisionResponse(BaseModel):
    """Response confirming review decision."""
    
    review_id: str = Field(
        ...,
        description="ID of the completed review",
    )
    decision_id: str = Field(
        ...,
        description="ID of the original decision",
    )
    reviewer_id: str = Field(
        ...,
        description="ID of the reviewer",
    )
    decision: str = Field(
        ...,
        description="Final decision applied",
    )
    reviewed_at: str = Field(
        ...,
        description="ISO 8601 timestamp of review",
    )
    review_duration_seconds: float = Field(
        default=0.0,
        description="Time taken to review",
    )


# Mock storage
_oversight_state = {
    "mode": OversightMode.MONITORING,
    "emergency_stop": False,
    "pending_reviews": [],
    "overrides": [],
    "suspensions": [],
}


@router.get("/status", response_model=OversightStatusResponse)
async def get_oversight_status() -> OversightStatusResponse:
    """Get current human oversight status.
    
    Returns the current state of human oversight mechanisms,
    including active reviewers, pending reviews, and AI autonomy level.
    
    This endpoint supports EU AI Act Article 14 compliance by providing
    transparency into human oversight capabilities.
    
    Returns:
        OversightStatusResponse with current oversight status
    """
    return OversightStatusResponse(
        mode=_oversight_state.get("mode", OversightMode.MONITORING),
        active_reviewers=3,
        pending_reviews=len(_oversight_state.get("pending_reviews", [])),
        average_review_time_seconds=45.2,
        last_override=_oversight_state.get("last_override"),
        emergency_stop_active=_oversight_state.get("emergency_stop", False),
        ai_autonomy_level=0.75,
        eu_ai_act_compliant=True,
    )


@router.post("/override", response_model=OverrideResponse)
async def override_decision(
    request: OverrideRequest,
    response: Response,
) -> OverrideResponse:
    """Override an AI decision with human judgment.
    
    Implements EU AI Act Article 14.3(b) - ability to discard AI output
    and substitute human decision.
    
    Also implements Fundamental Law 13 (Human Authority) - humans can
    override AI decisions.
    
    Args:
        request: Override request with justification
        response: FastAPI response object
        
    Returns:
        OverrideResponse confirming the override
    """
    override_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Record override
    override_record = {
        "override_id": override_id,
        "decision_id": request.decision_id,
        "new_decision": request.override_decision,
        "reason": request.reason.value,
        "justification": request.justification,
        "timestamp": timestamp,
    }
    _oversight_state.setdefault("overrides", []).append(override_record)
    _oversight_state["last_override"] = timestamp
    
    # Set response headers
    response.headers["X-Override-ID"] = override_id
    response.headers["X-Law-13-Applied"] = "true"
    
    return OverrideResponse(
        override_id=override_id,
        decision_id=request.decision_id,
        original_decision="BLOCK",  # Would be fetched from storage
        new_decision=request.override_decision,
        reviewer_id="current-user-id",  # Would be from auth
        timestamp=timestamp,
        audit_logged=True,
        fundamental_law_13_applied=True,
    )


@router.post("/suspend", response_model=SuspendAgentResponse)
async def suspend_agent(
    request: SuspendAgentRequest,
    response: Response,
) -> SuspendAgentResponse:
    """Suspend an AI agent.
    
    Implements EU AI Act Article 14.3(a) - ability to stop AI operation
    for specific agents.
    
    Args:
        request: Suspension request
        response: FastAPI response object
        
    Returns:
        SuspendAgentResponse confirming suspension
    """
    suspension_id = str(uuid.uuid4())
    suspended_at = datetime.now(timezone.utc)
    
    expires_at = None
    if request.duration_hours:
        from datetime import timedelta
        expires_at = (suspended_at + timedelta(hours=request.duration_hours)).isoformat()
    
    # Record suspension
    suspension_record = {
        "suspension_id": suspension_id,
        "agent_id": request.agent_id,
        "suspended_at": suspended_at.isoformat(),
        "expires_at": expires_at,
        "reason": request.reason,
    }
    _oversight_state.setdefault("suspensions", []).append(suspension_record)
    
    response.headers["X-Suspension-ID"] = suspension_id
    
    return SuspendAgentResponse(
        suspension_id=suspension_id,
        agent_id=request.agent_id,
        suspended_at=suspended_at.isoformat(),
        expires_at=expires_at,
        active=True,
    )


@router.post("/emergency-stop", response_model=EmergencyStopResponse)
async def emergency_stop(
    request: EmergencyStopRequest,
    response: Response,
) -> EmergencyStopResponse:
    """Trigger emergency stop.
    
    Implements EU AI Act Article 14.3(a) - ability to stop AI operation.
    
    Also implements Fundamental Law 23 (Fail-Safe Design) - system
    can be safely stopped at any time.
    
    Args:
        request: Emergency stop request
        response: FastAPI response object
        
    Returns:
        EmergencyStopResponse confirming emergency stop
    """
    stop_id = str(uuid.uuid4())
    activated_at = datetime.now(timezone.utc).isoformat()
    
    # Activate emergency stop
    _oversight_state["emergency_stop"] = True
    
    response.headers["X-Emergency-Stop-ID"] = stop_id
    response.headers["X-Law-23-Applied"] = "true"
    response.status_code = 200
    
    return EmergencyStopResponse(
        stop_id=stop_id,
        scope=request.scope,
        activated_at=activated_at,
        affected_agents=100,  # Would be computed
        require_manual_restart=request.require_manual_restart,
        fundamental_law_23_applied=True,
    )


@router.get("/pending-reviews", response_model=PendingReviewsResponse)
async def get_pending_reviews(
    priority: Optional[ReviewPriority] = Query(
        default=None,
        description="Filter by priority level",
    ),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> PendingReviewsResponse:
    """Get pending human reviews.
    
    Lists AI decisions that require human review before execution
    or final determination.
    
    Implements EU AI Act Article 14.4 - proportionate human oversight
    for high-risk decisions.
    
    Args:
        priority: Optional priority filter
        page: Page number
        page_size: Items per page
        
    Returns:
        PendingReviewsResponse with list of pending reviews
    """
    # Generate sample pending reviews
    reviews = [
        PendingReview(
            review_id=str(uuid.uuid4()),
            decision_id=f"dec-{i:04d}",
            agent_id=f"agent-{i % 5:03d}",
            action_summary=f"Action requiring review #{i}",
            ai_recommendation="BLOCK" if i % 3 == 0 else "RESTRICT",
            risk_score=0.5 + (i % 5) * 0.1,
            priority=ReviewPriority.HIGH if i % 4 == 0 else ReviewPriority.MEDIUM,
            status=ReviewStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at=datetime.now(timezone.utc).isoformat(),
        )
        for i in range(1, 6)
    ]
    
    # Apply priority filter
    if priority:
        reviews = [r for r in reviews if r.priority == priority]
    
    return PendingReviewsResponse(
        reviews=reviews,
        total_count=len(reviews),
        page=page,
        page_size=page_size,
    )


@router.post("/review", response_model=ReviewDecisionResponse)
async def submit_review(
    request: ReviewDecisionRequest,
    response: Response,
) -> ReviewDecisionResponse:
    """Submit a human review decision.
    
    Completes a pending human review by submitting the human's decision.
    
    Implements EU AI Act Article 14 - effective human oversight.
    Also implements Fundamental Law 14 (Appeal Rights).
    
    Args:
        request: Review decision
        response: FastAPI response object
        
    Returns:
        ReviewDecisionResponse confirming the review
        
    Note:
        In production, reviewer_id would come from authenticated session,
        decision_id from the review record, and duration from actual timestamps.
    """
    reviewed_at = datetime.now(timezone.utc)
    
    final_decision = request.modified_decision or request.decision
    
    # In production, these would come from the actual review record
    # For now, we use UUID-based generation and placeholder values
    decision_id = str(uuid.uuid4())
    
    response.headers["X-Review-Complete"] = "true"
    response.headers["X-Law-14-Applied"] = "true"
    
    return ReviewDecisionResponse(
        review_id=request.review_id,
        decision_id=decision_id,
        reviewer_id="authenticated-user-placeholder",  # TODO: Get from auth context
        decision=final_decision,
        reviewed_at=reviewed_at.isoformat(),
        review_duration_seconds=0.0,  # TODO: Calculate from review start time
    )


@router.post("/set-mode")
async def set_oversight_mode(
    mode: OversightMode,
    response: Response,
) -> dict[str, Any]:
    """Set the human oversight operation mode.
    
    Changes the level of human oversight applied to AI decisions.
    
    Args:
        mode: New oversight mode
        response: FastAPI response object
        
    Returns:
        Confirmation of mode change
    """
    previous_mode = _oversight_state.get("mode", OversightMode.MONITORING)
    _oversight_state["mode"] = mode
    
    return {
        "previous_mode": previous_mode.value,
        "new_mode": mode.value,
        "changed_at": datetime.now(timezone.utc).isoformat(),
        "eu_ai_act_compliant": mode != OversightMode.DISABLED,
    }


__all__ = [
    "router",
    "OversightMode",
    "ReviewPriority",
    "ReviewStatus",
]
