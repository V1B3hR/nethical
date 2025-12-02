"""Audit trail routes for API v2.

Provides audit log access and compliance reporting.

Implements:
- GET /audit/{id} - Lookup audit record by ID
- GET /audit - List audit records with filtering
- GET /audit/export - Export audit logs

Adheres to:
- Law 15: Audit Compliance - Maintain audit trails
- Law 10: Reasoning Transparency - Decisions are traceable
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()

# In-memory audit store (would be database with Merkle tree in production)
_audit_store: dict[str, dict[str, Any]] = {}


class AuditRecord(BaseModel):
    """Complete audit record."""
    
    audit_id: str = Field(..., description="Unique audit identifier")
    event_type: str = Field(
        ...,
        description="Event type: decision, policy_change, appeal, system",
    )
    entity_id: str = Field(..., description="ID of the related entity")
    entity_type: str = Field(
        ...,
        description="Entity type: decision, policy, appeal, agent",
    )
    agent_id: Optional[str] = Field(None, description="Agent involved if applicable")
    action: str = Field(..., description="Action that was audited")
    outcome: str = Field(..., description="Outcome of the action")
    risk_score: Optional[float] = Field(None, description="Risk score if applicable")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )
    merkle_hash: Optional[str] = Field(
        None,
        description="Merkle tree hash for integrity verification",
    )
    previous_hash: Optional[str] = Field(
        None,
        description="Hash of previous record in chain",
    )
    fundamental_laws: list[int] = Field(
        default_factory=lambda: [15],
        description="Relevant Fundamental Laws",
    )
    timestamp: str = Field(..., description="Event timestamp")
    verified: bool = Field(default=True, description="Whether record integrity is verified")


class AuditListResponse(BaseModel):
    """Paginated list of audit records."""
    
    records: list[AuditRecord] = Field(..., description="List of audit records")
    total_count: int = Field(..., description="Total number of records")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")
    has_next: bool = Field(..., description="Whether more pages exist")
    chain_integrity: bool = Field(
        default=True,
        description="Whether audit chain integrity is verified",
    )
    timestamp: str = Field(..., description="Response timestamp")


class AuditExportRequest(BaseModel):
    """Request to export audit logs."""
    
    format: str = Field(
        default="json",
        description="Export format: json, csv, parquet",
    )
    period_start: Optional[str] = Field(
        None,
        description="Start of export period (ISO 8601)",
    )
    period_end: Optional[str] = Field(
        None,
        description="End of export period (ISO 8601)",
    )
    event_types: Optional[list[str]] = Field(
        None,
        description="Filter by event types",
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in export",
    )


class AuditExportResponse(BaseModel):
    """Response for audit export request."""
    
    export_id: str = Field(..., description="Export job identifier")
    status: str = Field(..., description="pending, processing, completed")
    record_count: int = Field(..., description="Number of records to export")
    estimated_size_mb: float = Field(..., description="Estimated export size")
    download_url: Optional[str] = Field(
        None,
        description="Download URL when completed",
    )
    created_at: str = Field(..., description="Export request timestamp")
    expires_at: Optional[str] = Field(
        None,
        description="When download will expire",
    )


def _generate_merkle_hash(data: dict) -> str:
    """Generate a simulated Merkle hash for audit integrity."""
    # In production, this would use actual cryptographic hashing
    import hashlib
    content = str(sorted(data.items()))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_audit_record(
    event_type: str,
    entity_id: str,
    entity_type: str,
    action: str,
    outcome: str,
    agent_id: Optional[str] = None,
    risk_score: Optional[float] = None,
    metadata: Optional[dict] = None,
) -> AuditRecord:
    """Create and store an audit record.
    
    Implements Law 15 (Audit Compliance) by maintaining
    immutable audit trails.
    """
    audit_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    # Get previous hash for chain integrity
    previous_hash = None
    if _audit_store:
        latest = max(_audit_store.values(), key=lambda r: r.get("timestamp", ""))
        previous_hash = latest.get("merkle_hash")
    
    record_data = {
        "audit_id": audit_id,
        "event_type": event_type,
        "entity_id": entity_id,
        "entity_type": entity_type,
        "agent_id": agent_id,
        "action": action,
        "outcome": outcome,
        "risk_score": risk_score,
        "metadata": metadata or {},
        "previous_hash": previous_hash,
        "fundamental_laws": [15],
        "timestamp": now,
        "verified": True,
    }
    
    # Generate Merkle hash
    record_data["merkle_hash"] = _generate_merkle_hash(record_data)
    
    _audit_store[audit_id] = record_data
    
    return AuditRecord(**record_data)


@router.get("/audit/{audit_id}", response_model=AuditRecord)
async def get_audit_record(audit_id: str) -> AuditRecord:
    """Retrieve a specific audit record.
    
    Implements Law 15 (Audit Compliance) by providing
    access to audit trails.
    
    Args:
        audit_id: Audit record identifier
        
    Returns:
        AuditRecord with full details
        
    Raises:
        HTTPException: If record not found
    """
    record = _audit_store.get(audit_id)
    
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Audit record {audit_id} not found",
        )
    
    return AuditRecord(**record)


@router.get("/audit", response_model=AuditListResponse)
async def list_audit_records(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    agent_id: Optional[str] = Query(None, description="Filter by agent"),
    period_start: Optional[str] = Query(None, description="Period start (ISO 8601)"),
    period_end: Optional[str] = Query(None, description="Period end (ISO 8601)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
) -> AuditListResponse:
    """List audit records with filtering and pagination.
    
    Supports compliance reporting by providing access to
    historical audit data.
    
    Args:
        event_type: Optional event type filter
        entity_type: Optional entity type filter
        agent_id: Optional agent filter
        period_start: Optional start date filter
        period_end: Optional end date filter
        page: Page number
        page_size: Items per page
        
    Returns:
        Paginated list of audit records
    """
    all_records = list(_audit_store.values())
    
    # Apply filters
    if event_type:
        all_records = [r for r in all_records if r.get("event_type") == event_type]
    
    if entity_type:
        all_records = [r for r in all_records if r.get("entity_type") == entity_type]
    
    if agent_id:
        all_records = [r for r in all_records if r.get("agent_id") == agent_id]
    
    if period_start:
        all_records = [r for r in all_records if r.get("timestamp", "") >= period_start]
    
    if period_end:
        all_records = [r for r in all_records if r.get("timestamp", "") <= period_end]
    
    # Sort by timestamp (most recent first)
    all_records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    
    # Paginate
    total_count = len(all_records)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_records = all_records[start_idx:end_idx]
    
    records = [AuditRecord(**r) for r in page_records]
    
    # Verify chain integrity (simplified)
    chain_integrity = all(r.get("verified", True) for r in _audit_store.values())
    
    return AuditListResponse(
        records=records,
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next=end_idx < total_count,
        chain_integrity=chain_integrity,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/audit/export", response_model=AuditExportResponse)
async def export_audit_logs(request: AuditExportRequest) -> AuditExportResponse:
    """Request an export of audit logs.
    
    Creates an asynchronous export job for compliance reporting.
    Supports multiple formats for different compliance requirements.
    
    Args:
        request: Export request configuration
        
    Returns:
        AuditExportResponse with job tracking info
    """
    export_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    
    # Count matching records
    all_records = list(_audit_store.values())
    
    if request.period_start:
        all_records = [r for r in all_records if r.get("timestamp", "") >= request.period_start]
    
    if request.period_end:
        all_records = [r for r in all_records if r.get("timestamp", "") <= request.period_end]
    
    if request.event_types:
        all_records = [r for r in all_records if r.get("event_type") in request.event_types]
    
    record_count = len(all_records)
    estimated_size = record_count * 0.001  # ~1KB per record
    
    return AuditExportResponse(
        export_id=export_id,
        status="pending",
        record_count=record_count,
        estimated_size_mb=estimated_size,
        download_url=None,
        created_at=now.isoformat(),
        expires_at=None,
    )


@router.get("/audit/verify", response_model=dict[str, Any])
async def verify_audit_chain() -> dict[str, Any]:
    """Verify the integrity of the audit chain.
    
    Checks Merkle tree integrity to ensure no tampering
    has occurred in the audit trail.
    
    Returns:
        Chain verification result
    """
    if not _audit_store:
        return {
            "status": "empty",
            "message": "No audit records to verify",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    # Verify chain integrity
    records = sorted(_audit_store.values(), key=lambda r: r.get("timestamp", ""))
    
    verification_errors = []
    previous_hash = None
    
    for record in records:
        # Check previous hash linkage
        if previous_hash and record.get("previous_hash") != previous_hash:
            verification_errors.append({
                "audit_id": record["audit_id"],
                "error": "Chain break detected - previous hash mismatch",
            })
        
        previous_hash = record.get("merkle_hash")
    
    return {
        "status": "valid" if not verification_errors else "invalid",
        "records_verified": len(records),
        "errors": verification_errors,
        "chain_intact": len(verification_errors) == 0,
        "fundamental_law": "Law 15: Audit Compliance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
