"""Audit log routes for API v1.

Provides read-only access to audit logs with Merkle tree verification.

Endpoints:
- GET /api/v1/audit/logs - Get audit logs with pagination and filtering
- GET /api/v1/audit/logs/{id} - Get single audit log details
- GET /api/v1/audit/merkle-tree - Get Merkle tree structure
- POST /api/v1/audit/verify - Verify Merkle proof for a log entry
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from nethical.api.rbac import User, get_current_user, require_auditor_or_admin
from nethical.database import AuditLog, get_db

router = APIRouter(prefix="/audit", tags=["Audit Logs"])


class AuditLogResponse(BaseModel):
    """Audit log response model."""
    
    id: int
    log_id: str
    event_type: str
    agent_id: Optional[str]
    action: Optional[str]
    outcome: Optional[str]
    threat_type: Optional[str]
    threat_level: Optional[str]
    risk_score: Optional[float]
    details: dict[str, Any]
    merkle_hash: Optional[str]
    previous_hash: Optional[str]
    timestamp: str
    verified: bool


class AuditLogListResponse(BaseModel):
    """Paginated list of audit logs."""
    
    logs: list[AuditLogResponse]
    total: int
    page: int
    per_page: int
    pages: int


class MerkleNode(BaseModel):
    """Merkle tree node."""
    
    hash: str
    left: Optional[str] = None
    right: Optional[str] = None
    log_id: Optional[str] = None


class MerkleTreeResponse(BaseModel):
    """Merkle tree structure response."""
    
    root_hash: str
    total_logs: int
    tree_height: int
    nodes: list[MerkleNode]
    generated_at: str


class VerifyMerkleProofRequest(BaseModel):
    """Request to verify Merkle proof."""
    
    log_id: str = Field(..., description="Log ID to verify")
    merkle_path: list[tuple[str, str]] = Field(
        ...,
        description="Merkle path: list of (hash, direction) tuples where direction is 'left' or 'right'"
    )


class VerifyMerkleProofResponse(BaseModel):
    """Merkle proof verification response."""
    
    log_id: str
    verified: bool
    root_hash: str
    computed_root: str
    message: str


@router.get("/logs", response_model=AuditLogListResponse)
async def get_audit_logs(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auditor_or_admin)],
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    threat_level: Optional[str] = Query(None, description="Filter by threat level"),
    from_date: Optional[str] = Query(None, description="Start date (ISO 8601)"),
    to_date: Optional[str] = Query(None, description="End date (ISO 8601)"),
) -> AuditLogListResponse:
    """Get audit logs with pagination and filtering.
    
    **Required role:** auditor or admin
    
    Args:
        db: Database session
        current_user: Current authenticated user
        page: Page number (1-indexed)
        per_page: Items per page (max 100)
        agent_id: Filter by agent ID
        event_type: Filter by event type
        threat_level: Filter by threat level (low, medium, high, critical)
        from_date: Start date filter (ISO 8601 format)
        to_date: End date filter (ISO 8601 format)
        
    Returns:
        Paginated list of audit logs
    """
    query = db.query(AuditLog)
    
    # Apply filters
    if agent_id:
        query = query.filter(AuditLog.agent_id == agent_id)
    if event_type:
        query = query.filter(AuditLog.event_type == event_type)
    if threat_level:
        query = query.filter(AuditLog.threat_level == threat_level)
    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
            query = query.filter(AuditLog.timestamp >= from_dt)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid from_date format. Use ISO 8601 format."
            )
    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
            query = query.filter(AuditLog.timestamp <= to_dt)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid to_date format. Use ISO 8601 format."
            )
    
    # Get total count
    total = query.count()
    
    # Apply pagination and ordering (most recent first)
    logs = query.order_by(AuditLog.timestamp.desc()).offset((page - 1) * per_page).limit(per_page).all()
    
    return AuditLogListResponse(
        logs=[AuditLogResponse(**log.to_dict()) for log in logs],
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page
    )


@router.get("/logs/{log_id}", response_model=AuditLogResponse)
async def get_audit_log(
    log_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auditor_or_admin)],
) -> AuditLogResponse:
    """Get single audit log details.
    
    **Required role:** auditor or admin
    
    Args:
        log_id: Log identifier
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Audit log details
        
    Raises:
        HTTPException: 404 if log not found
    """
    log = db.query(AuditLog).filter(AuditLog.log_id == log_id).first()
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit log '{log_id}' not found"
        )
    
    return AuditLogResponse(**log.to_dict())


def build_merkle_tree(logs: list[AuditLog]) -> tuple[str, list[MerkleNode]]:
    """Build Merkle tree from audit logs.
    
    Args:
        logs: List of audit logs
        
    Returns:
        Tuple of (root_hash, nodes)
    """
    if not logs:
        return ("", [])
    
    nodes = []
    
    # Create leaf nodes
    leaf_hashes = []
    for log in logs:
        leaf_hash = hashlib.sha256(
            f"{log.log_id}:{log.timestamp}:{log.event_type}".encode()
        ).hexdigest()
        leaf_hashes.append(leaf_hash)
        nodes.append(MerkleNode(hash=leaf_hash, log_id=log.log_id))
    
    # Build tree level by level
    current_level = leaf_hashes
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else left
            
            parent_hash = hashlib.sha256(f"{left}{right}".encode()).hexdigest()
            next_level.append(parent_hash)
            nodes.append(MerkleNode(hash=parent_hash, left=left, right=right))
        
        current_level = next_level
    
    root_hash = current_level[0] if current_level else ""
    return (root_hash, nodes)


@router.get("/merkle-tree", response_model=MerkleTreeResponse)
async def get_merkle_tree(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auditor_or_admin)],
    limit: int = Query(100, ge=1, le=1000, description="Number of recent logs to include"),
) -> MerkleTreeResponse:
    """Get Merkle tree structure for audit logs.
    
    **Required role:** auditor or admin
    
    Args:
        db: Database session
        current_user: Current authenticated user
        limit: Number of recent logs to include in tree (max 1000)
        
    Returns:
        Merkle tree structure
    """
    # Get recent logs
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
    
    # Build Merkle tree
    root_hash, nodes = build_merkle_tree(logs)
    
    # Calculate tree height
    tree_height = 0
    if logs:
        import math
        tree_height = math.ceil(math.log2(len(logs))) + 1
    
    return MerkleTreeResponse(
        root_hash=root_hash,
        total_logs=len(logs),
        tree_height=tree_height,
        nodes=nodes[:100],  # Return first 100 nodes for performance
        generated_at=datetime.now(timezone.utc).isoformat()
    )


@router.post("/verify", response_model=VerifyMerkleProofResponse)
async def verify_merkle_proof(
    request: VerifyMerkleProofRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_auditor_or_admin)],
) -> VerifyMerkleProofResponse:
    """Verify Merkle proof for a log entry.
    
    **Required role:** auditor or admin
    
    Args:
        request: Verification request with log ID and Merkle path
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Verification result
        
    Raises:
        HTTPException: 404 if log not found
    """
    # Get the log
    log = db.query(AuditLog).filter(AuditLog.log_id == request.log_id).first()
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit log '{request.log_id}' not found"
        )
    
    # Compute leaf hash
    current_hash = hashlib.sha256(
        f"{log.log_id}:{log.timestamp}:{log.event_type}".encode()
    ).hexdigest()
    
    # Traverse Merkle path
    for sibling_hash, direction in request.merkle_path:
        if direction == "left":
            current_hash = hashlib.sha256(f"{sibling_hash}{current_hash}".encode()).hexdigest()
        else:
            current_hash = hashlib.sha256(f"{current_hash}{sibling_hash}".encode()).hexdigest()
    
    # Get current root hash
    recent_logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(100).all()
    root_hash, _ = build_merkle_tree(recent_logs)
    
    # Verify
    verified = current_hash == root_hash
    
    return VerifyMerkleProofResponse(
        log_id=request.log_id,
        verified=verified,
        root_hash=root_hash,
        computed_root=current_hash,
        message="Merkle proof verified successfully" if verified else "Merkle proof verification failed"
    )
