"""Appeals routes for API v2.

Provides appeal submission and tracking for contested decisions.

Implements:
- POST /appeals - Submit an appeal
- GET /appeals/{id} - Get appeal status
- GET /appeals - List appeals

Adheres to:
- Law 7: Override Rights - Humans can contest decisions
- Law 14: Error Acknowledgment - System acknowledges potential errors
- Law 19: Collaborative Problem-Solving - Appeals enable collaboration
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()

# In-memory appeals store (would be database in production)
_appeals_store: dict[str, dict[str, Any]] = {}


class AppealSubmission(BaseModel):
    """Request to submit an appeal."""

    decision_id: str = Field(
        ...,
        description="ID of the decision being appealed",
    )
    appellant_id: str = Field(
        ...,
        description="ID of the entity submitting the appeal",
    )
    reason: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Reason for the appeal",
    )
    evidence: Optional[dict[str, Any]] = Field(
        default=None,
        description="Supporting evidence for the appeal",
    )
    requested_outcome: str = Field(
        default="reconsider",
        description="Requested outcome: reconsider, override, escalate",
    )
    priority: str = Field(
        default="normal",
        description="Priority: low, normal, high, urgent",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "decision_id": "dec-12345",
                    "appellant_id": "user-67890",
                    "reason": "The decision was based on incomplete context. The action was a test query, not a production operation.",
                    "evidence": {"test_mode": True, "environment": "staging"},
                    "requested_outcome": "reconsider",
                    "priority": "normal",
                }
            ]
        }
    }


class AppealRecord(BaseModel):
    """Complete appeal record."""

    appeal_id: str = Field(..., description="Unique appeal identifier")
    decision_id: str = Field(..., description="Related decision ID")
    appellant_id: str = Field(..., description="Appellant identifier")
    reason: str = Field(..., description="Appeal reason")
    evidence: Optional[dict[str, Any]] = Field(None, description="Supporting evidence")
    requested_outcome: str = Field(..., description="Requested outcome")
    priority: str = Field(..., description="Appeal priority")
    status: str = Field(
        ...,
        description="Status: pending, under_review, approved, denied, escalated",
    )
    resolution: Optional[str] = Field(None, description="Resolution if decided")
    resolution_reason: Optional[str] = Field(None, description="Reason for resolution")
    reviewer_id: Optional[str] = Field(None, description="Reviewer ID if assigned")
    fundamental_laws: list[int] = Field(
        default_factory=lambda: [7, 14, 19],
        description="Relevant Fundamental Laws",
    )
    created_at: str = Field(..., description="Submission timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    resolved_at: Optional[str] = Field(None, description="Resolution timestamp")


class AppealListResponse(BaseModel):
    """Paginated list of appeals."""

    appeals: list[AppealRecord] = Field(..., description="List of appeals")
    total_count: int = Field(..., description="Total number of appeals")
    pending_count: int = Field(..., description="Number of pending appeals")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")
    has_next: bool = Field(..., description="Whether more pages exist")
    timestamp: str = Field(..., description="Response timestamp")


class AppealResolution(BaseModel):
    """Request to resolve an appeal."""

    resolution: str = Field(
        ...,
        description="Resolution: approved, denied",
    )
    resolution_reason: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Reason for the resolution",
    )
    reviewer_id: str = Field(..., description="ID of the reviewer")


@router.post("/appeals", response_model=AppealRecord, status_code=201)
async def submit_appeal(submission: AppealSubmission) -> AppealRecord:
    """Submit an appeal for a decision.

    Implements Law 7 (Override Rights) by allowing humans
    to contest automated decisions.

    Appeals enter a review queue and are processed according
    to their priority level.

    Args:
        submission: Appeal submission details

    Returns:
        Created appeal record
    """
    appeal_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    appeal_data = {
        "appeal_id": appeal_id,
        "decision_id": submission.decision_id,
        "appellant_id": submission.appellant_id,
        "reason": submission.reason,
        "evidence": submission.evidence,
        "requested_outcome": submission.requested_outcome,
        "priority": submission.priority,
        "status": "pending",
        "resolution": None,
        "resolution_reason": None,
        "reviewer_id": None,
        "fundamental_laws": [7, 14, 19],
        "created_at": now,
        "updated_at": now,
        "resolved_at": None,
    }

    _appeals_store[appeal_id] = appeal_data

    return AppealRecord(**appeal_data)


@router.get("/appeals/{appeal_id}", response_model=AppealRecord)
async def get_appeal(appeal_id: str) -> AppealRecord:
    """Get the status of an appeal.

    Args:
        appeal_id: Appeal identifier

    Returns:
        Appeal record

    Raises:
        HTTPException: If appeal not found
    """
    appeal = _appeals_store.get(appeal_id)

    if not appeal:
        raise HTTPException(
            status_code=404,
            detail=f"Appeal {appeal_id} not found",
        )

    return AppealRecord(**appeal)


@router.get("/appeals", response_model=AppealListResponse)
async def list_appeals(
    status: Optional[str] = Query(None, description="Filter by status"),
    appellant_id: Optional[str] = Query(None, description="Filter by appellant"),
    decision_id: Optional[str] = Query(None, description="Filter by decision"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> AppealListResponse:
    """List appeals with optional filtering.

    Args:
        status: Optional status filter
        appellant_id: Optional appellant filter
        decision_id: Optional decision filter
        page: Page number
        page_size: Items per page

    Returns:
        Paginated list of appeals
    """
    all_appeals = list(_appeals_store.values())

    if status:
        all_appeals = [a for a in all_appeals if a.get("status") == status]

    if appellant_id:
        all_appeals = [a for a in all_appeals if a.get("appellant_id") == appellant_id]

    if decision_id:
        all_appeals = [a for a in all_appeals if a.get("decision_id") == decision_id]

    # Sort by created_at (most recent first)
    all_appeals.sort(key=lambda a: a.get("created_at", ""), reverse=True)

    # Count pending
    pending_count = sum(
        1 for a in _appeals_store.values() if a.get("status") == "pending"
    )

    # Paginate
    total_count = len(all_appeals)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_appeals = all_appeals[start_idx:end_idx]

    records = [AppealRecord(**a) for a in page_appeals]

    return AppealListResponse(
        appeals=records,
        total_count=total_count,
        pending_count=pending_count,
        page=page,
        page_size=page_size,
        has_next=end_idx < total_count,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/appeals/{appeal_id}/resolve", response_model=AppealRecord)
async def resolve_appeal(
    appeal_id: str,
    resolution: AppealResolution,
) -> AppealRecord:
    """Resolve an appeal.

    Implements Law 14 (Error Acknowledgment) by processing
    appeals and potentially reversing decisions.

    Args:
        appeal_id: Appeal identifier
        resolution: Resolution details

    Returns:
        Updated appeal record

    Raises:
        HTTPException: If appeal not found or already resolved
    """
    appeal = _appeals_store.get(appeal_id)

    if not appeal:
        raise HTTPException(
            status_code=404,
            detail=f"Appeal {appeal_id} not found",
        )

    if appeal.get("status") not in ["pending", "under_review"]:
        raise HTTPException(
            status_code=400,
            detail=f"Appeal {appeal_id} is already resolved",
        )

    now = datetime.now(timezone.utc).isoformat()

    appeal["status"] = resolution.resolution
    appeal["resolution"] = resolution.resolution
    appeal["resolution_reason"] = resolution.resolution_reason
    appeal["reviewer_id"] = resolution.reviewer_id
    appeal["updated_at"] = now
    appeal["resolved_at"] = now

    _appeals_store[appeal_id] = appeal

    return AppealRecord(**appeal)
