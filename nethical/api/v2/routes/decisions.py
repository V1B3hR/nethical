"""Decision routes for API v2.

Provides decision lookup and history functionality.

Implements:
- GET /decisions/{id} - Lookup a specific decision
- GET /decisions - List recent decisions (with pagination)

Adheres to:
- Law 10: Reasoning Transparency - Decisions are explainable
- Law 15: Audit Compliance - Decisions are retrievable
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()

# In-memory decision store (would be database in production)
_decision_store: dict[str, dict[str, Any]] = {}


class DecisionRecord(BaseModel):
    """Complete record of a governance decision."""
    
    decision_id: str = Field(..., description="Unique decision identifier")
    decision: str = Field(..., description="ALLOW, RESTRICT, BLOCK, or TERMINATE")
    agent_id: str = Field(..., description="Agent that requested the action")
    action_summary: str = Field(..., description="Summary of the evaluated action")
    action_type: str = Field(..., description="Type of action")
    risk_score: float = Field(..., description="Risk score (0.0-1.0)")
    confidence: float = Field(..., description="Decision confidence")
    reasoning: str = Field(..., description="Explanation for the decision")
    violations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of violations detected",
    )
    fundamental_laws: list[int] = Field(
        default_factory=list,
        description="Fundamental Laws that were applied",
    )
    timestamp: str = Field(..., description="When the decision was made")
    latency_ms: int = Field(..., description="Evaluation latency")
    audit_id: Optional[str] = Field(default=None, description="Audit trail ID")


class DecisionListResponse(BaseModel):
    """Paginated list of decisions."""
    
    decisions: list[DecisionRecord] = Field(..., description="List of decisions")
    total_count: int = Field(..., description="Total number of decisions")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    has_next: bool = Field(..., description="Whether more pages exist")
    timestamp: str = Field(..., description="Response timestamp")


def _store_decision(decision_id: str, decision_data: dict[str, Any]) -> None:
    """Store a decision for later retrieval (Law 15: Audit Compliance)."""
    _decision_store[decision_id] = {
        **decision_data,
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


def _get_decision(decision_id: str) -> Optional[dict[str, Any]]:
    """Retrieve a stored decision."""
    return _decision_store.get(decision_id)


@router.get("/decisions/{decision_id}", response_model=DecisionRecord)
async def get_decision(decision_id: str) -> DecisionRecord:
    """Retrieve a specific decision by ID.
    
    This endpoint supports Law 10 (Reasoning Transparency) by providing
    access to historical decisions and their reasoning.
    
    Args:
        decision_id: Unique identifier of the decision
        
    Returns:
        DecisionRecord with full decision details
        
    Raises:
        HTTPException: If decision not found
    """
    decision = _get_decision(decision_id)
    
    if not decision:
        raise HTTPException(
            status_code=404,
            detail=f"Decision {decision_id} not found",
        )
    
    return DecisionRecord(
        decision_id=decision.get("decision_id", decision_id),
        decision=decision.get("decision", "UNKNOWN"),
        agent_id=decision.get("agent_id", "unknown"),
        action_summary=decision.get("action_summary", ""),
        action_type=decision.get("action_type", "unknown"),
        risk_score=decision.get("risk_score", 0.0),
        confidence=decision.get("confidence", 0.0),
        reasoning=decision.get("reasoning", ""),
        violations=decision.get("violations", []),
        fundamental_laws=decision.get("fundamental_laws", []),
        timestamp=decision.get("timestamp", datetime.now(timezone.utc).isoformat()),
        latency_ms=decision.get("latency_ms", 0),
        audit_id=decision.get("audit_id"),
    )


@router.get("/decisions", response_model=DecisionListResponse)
async def list_decisions(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    decision: Optional[str] = Query(None, description="Filter by decision type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> DecisionListResponse:
    """List recent decisions with optional filtering.
    
    Supports pagination and filtering by agent ID and decision type.
    Implements Law 15 (Audit Compliance) for decision history access.
    
    Args:
        agent_id: Optional filter by agent
        decision: Optional filter by decision type
        page: Page number (1-indexed)
        page_size: Number of items per page
        
    Returns:
        Paginated list of decisions
    """
    # Filter decisions
    all_decisions = list(_decision_store.values())
    
    if agent_id:
        all_decisions = [d for d in all_decisions if d.get("agent_id") == agent_id]
    
    if decision:
        all_decisions = [d for d in all_decisions if d.get("decision") == decision.upper()]
    
    # Sort by timestamp (most recent first)
    all_decisions.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
    
    # Paginate
    total_count = len(all_decisions)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_decisions = all_decisions[start_idx:end_idx]
    
    # Convert to records
    records = [
        DecisionRecord(
            decision_id=d.get("decision_id", "unknown"),
            decision=d.get("decision", "UNKNOWN"),
            agent_id=d.get("agent_id", "unknown"),
            action_summary=d.get("action_summary", ""),
            action_type=d.get("action_type", "unknown"),
            risk_score=d.get("risk_score", 0.0),
            confidence=d.get("confidence", 0.0),
            reasoning=d.get("reasoning", ""),
            violations=d.get("violations", []),
            fundamental_laws=d.get("fundamental_laws", []),
            timestamp=d.get("timestamp", datetime.now(timezone.utc).isoformat()),
            latency_ms=d.get("latency_ms", 0),
            audit_id=d.get("audit_id"),
        )
        for d in page_decisions
    ]
    
    return DecisionListResponse(
        decisions=records,
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next=end_idx < total_count,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
