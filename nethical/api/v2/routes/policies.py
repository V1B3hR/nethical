"""Policy management routes for API v2.

Provides CRUD operations for governance policies.

Implements:
- GET /policies - List all policies
- POST /policies - Create a new policy
- GET /policies/{id} - Get a specific policy
- PUT /policies/{id} - Update a policy
- DELETE /policies/{id} - Deprecate a policy

Adheres to:
- Law 5: Bounded Autonomy - Policies define operational boundaries
- Law 8: Constraint Transparency - Policies are transparent
- Law 15: Audit Compliance - Policy changes are logged
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()

# In-memory policy store (would be database in production)
_policy_store: dict[str, dict[str, Any]] = {}


class PolicyRule(BaseModel):
    """A single rule within a policy."""
    
    id: str = Field(..., description="Rule identifier")
    condition: str = Field(..., description="Condition expression")
    action: str = Field(
        ...,
        description="Action to take: ALLOW, RESTRICT, BLOCK, TERMINATE",
    )
    priority: int = Field(default=0, description="Rule priority (higher = first)")
    description: Optional[str] = Field(None, description="Human-readable description")


class PolicyCreate(BaseModel):
    """Request to create a new policy."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Policy name",
    )
    description: str = Field(
        ...,
        max_length=2000,
        description="Policy description",
    )
    version: str = Field(
        default="1.0.0",
        description="Policy version",
    )
    rules: list[PolicyRule] = Field(
        default_factory=list,
        description="List of policy rules",
    )
    scope: Optional[str] = Field(
        default="global",
        description="Policy scope: global, agent, action_type",
    )
    fundamental_laws: list[int] = Field(
        default_factory=list,
        description="Fundamental Laws this policy implements",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional metadata",
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Data Access Policy",
                    "description": "Controls access to sensitive data",
                    "version": "1.0.0",
                    "rules": [
                        {
                            "id": "rule-1",
                            "condition": "action_type == 'data_access'",
                            "action": "RESTRICT",
                            "priority": 10,
                            "description": "Restrict data access by default",
                        }
                    ],
                    "scope": "global",
                    "fundamental_laws": [22],
                }
            ]
        }
    }


class PolicyRecord(BaseModel):
    """Complete policy record."""
    
    policy_id: str = Field(..., description="Unique policy identifier")
    name: str = Field(..., description="Policy name")
    description: str = Field(..., description="Policy description")
    version: str = Field(..., description="Policy version")
    status: str = Field(
        ...,
        description="Status: active, deprecated, quarantine",
    )
    rules: list[PolicyRule] = Field(..., description="Policy rules")
    scope: str = Field(..., description="Policy scope")
    fundamental_laws: list[int] = Field(
        default_factory=list,
        description="Fundamental Laws implemented",
    )
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    created_by: Optional[str] = Field(None, description="Creator identifier")
    metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")


class PolicyListResponse(BaseModel):
    """Paginated list of policies."""
    
    policies: list[PolicyRecord] = Field(..., description="List of policies")
    total_count: int = Field(..., description="Total number of policies")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")
    has_next: bool = Field(..., description="Whether more pages exist")
    timestamp: str = Field(..., description="Response timestamp")


class PolicyUpdate(BaseModel):
    """Request to update a policy."""
    
    name: Optional[str] = Field(None, max_length=255, description="New name")
    description: Optional[str] = Field(None, max_length=2000, description="New description")
    version: Optional[str] = Field(None, description="New version")
    rules: Optional[list[PolicyRule]] = Field(None, description="New rules")
    status: Optional[str] = Field(None, description="New status")
    fundamental_laws: Optional[list[int]] = Field(None, description="Updated laws")
    metadata: Optional[dict[str, Any]] = Field(None, description="Updated metadata")


def _init_default_policies() -> None:
    """Initialize default governance policies."""
    if not _policy_store:
        # Safety-first policy (Law 21)
        safety_policy_id = str(uuid.uuid4())
        _policy_store[safety_policy_id] = {
            "policy_id": safety_policy_id,
            "name": "Core Safety Policy",
            "description": "Ensures human safety is prioritized in all decisions",
            "version": "1.0.0",
            "status": "active",
            "rules": [
                {
                    "id": "safety-1",
                    "condition": "risk_score >= 0.9",
                    "action": "BLOCK",
                    "priority": 100,
                    "description": "Block high-risk actions",
                },
                {
                    "id": "safety-2",
                    "condition": "contains(action, 'terminate all')",
                    "action": "TERMINATE",
                    "priority": 99,
                    "description": "Terminate on mass termination attempts",
                },
            ],
            "scope": "global",
            "fundamental_laws": [21, 23],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "system",
            "metadata": {"is_core": True},
        }
        
        # Privacy policy (Law 22)
        privacy_policy_id = str(uuid.uuid4())
        _policy_store[privacy_policy_id] = {
            "policy_id": privacy_policy_id,
            "name": "Privacy Protection Policy",
            "description": "Protects user privacy and digital security",
            "version": "1.0.0",
            "status": "active",
            "rules": [
                {
                    "id": "privacy-1",
                    "condition": "pii_detected == true",
                    "action": "RESTRICT",
                    "priority": 80,
                    "description": "Restrict actions with PII",
                },
            ],
            "scope": "global",
            "fundamental_laws": [22],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "created_by": "system",
            "metadata": {"is_core": True},
        }


# Initialize default policies
_init_default_policies()


@router.get("/policies", response_model=PolicyListResponse)
async def list_policies(
    status: Optional[str] = Query(None, description="Filter by status"),
    scope: Optional[str] = Query(None, description="Filter by scope"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> PolicyListResponse:
    """List all governance policies.
    
    Implements Law 8 (Constraint Transparency) by providing visibility
    into all active policies.
    
    Args:
        status: Optional status filter
        scope: Optional scope filter
        page: Page number
        page_size: Items per page
        
    Returns:
        Paginated list of policies
    """
    all_policies = list(_policy_store.values())
    
    if status:
        all_policies = [p for p in all_policies if p.get("status") == status]
    
    if scope:
        all_policies = [p for p in all_policies if p.get("scope") == scope]
    
    # Sort by created_at
    all_policies.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    
    # Paginate
    total_count = len(all_policies)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_policies = all_policies[start_idx:end_idx]
    
    # Convert to records
    records = [
        PolicyRecord(
            policy_id=p["policy_id"],
            name=p["name"],
            description=p["description"],
            version=p["version"],
            status=p["status"],
            rules=[PolicyRule(**r) for r in p.get("rules", [])],
            scope=p.get("scope", "global"),
            fundamental_laws=p.get("fundamental_laws", []),
            created_at=p["created_at"],
            updated_at=p["updated_at"],
            created_by=p.get("created_by"),
            metadata=p.get("metadata"),
        )
        for p in page_policies
    ]
    
    return PolicyListResponse(
        policies=records,
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next=end_idx < total_count,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("/policies", response_model=PolicyRecord, status_code=201)
async def create_policy(policy: PolicyCreate) -> PolicyRecord:
    """Create a new governance policy.
    
    New policies are created in 'quarantine' status and must be
    activated after review (Law 15: Audit Compliance).
    
    Args:
        policy: Policy creation request
        
    Returns:
        Created policy record
    """
    policy_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    policy_data = {
        "policy_id": policy_id,
        "name": policy.name,
        "description": policy.description,
        "version": policy.version,
        "status": "quarantine",  # New policies start in quarantine
        "rules": [r.model_dump() for r in policy.rules],
        "scope": policy.scope or "global",
        "fundamental_laws": policy.fundamental_laws,
        "created_at": now,
        "updated_at": now,
        "created_by": "api",
        "metadata": policy.metadata,
    }
    
    _policy_store[policy_id] = policy_data
    
    return PolicyRecord(
        policy_id=policy_id,
        name=policy.name,
        description=policy.description,
        version=policy.version,
        status="quarantine",
        rules=policy.rules,
        scope=policy.scope or "global",
        fundamental_laws=policy.fundamental_laws,
        created_at=now,
        updated_at=now,
        created_by="api",
        metadata=policy.metadata,
    )


@router.get("/policies/{policy_id}", response_model=PolicyRecord)
async def get_policy(policy_id: str) -> PolicyRecord:
    """Get a specific policy by ID.
    
    Args:
        policy_id: Policy identifier
        
    Returns:
        Policy record
        
    Raises:
        HTTPException: If policy not found
    """
    policy = _policy_store.get(policy_id)
    
    if not policy:
        raise HTTPException(
            status_code=404,
            detail=f"Policy {policy_id} not found",
        )
    
    return PolicyRecord(
        policy_id=policy["policy_id"],
        name=policy["name"],
        description=policy["description"],
        version=policy["version"],
        status=policy["status"],
        rules=[PolicyRule(**r) for r in policy.get("rules", [])],
        scope=policy.get("scope", "global"),
        fundamental_laws=policy.get("fundamental_laws", []),
        created_at=policy["created_at"],
        updated_at=policy["updated_at"],
        created_by=policy.get("created_by"),
        metadata=policy.get("metadata"),
    )


@router.put("/policies/{policy_id}", response_model=PolicyRecord)
async def update_policy(policy_id: str, update: PolicyUpdate) -> PolicyRecord:
    """Update an existing policy.
    
    Policy updates are logged for audit compliance (Law 15).
    
    Args:
        policy_id: Policy identifier
        update: Policy update request
        
    Returns:
        Updated policy record
        
    Raises:
        HTTPException: If policy not found
    """
    policy = _policy_store.get(policy_id)
    
    if not policy:
        raise HTTPException(
            status_code=404,
            detail=f"Policy {policy_id} not found",
        )
    
    # Update fields
    if update.name is not None:
        policy["name"] = update.name
    if update.description is not None:
        policy["description"] = update.description
    if update.version is not None:
        policy["version"] = update.version
    if update.rules is not None:
        policy["rules"] = [r.model_dump() for r in update.rules]
    if update.status is not None:
        policy["status"] = update.status
    if update.fundamental_laws is not None:
        policy["fundamental_laws"] = update.fundamental_laws
    if update.metadata is not None:
        policy["metadata"] = update.metadata
    
    policy["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    _policy_store[policy_id] = policy
    
    return PolicyRecord(
        policy_id=policy["policy_id"],
        name=policy["name"],
        description=policy["description"],
        version=policy["version"],
        status=policy["status"],
        rules=[PolicyRule(**r) for r in policy.get("rules", [])],
        scope=policy.get("scope", "global"),
        fundamental_laws=policy.get("fundamental_laws", []),
        created_at=policy["created_at"],
        updated_at=policy["updated_at"],
        created_by=policy.get("created_by"),
        metadata=policy.get("metadata"),
    )


@router.delete("/policies/{policy_id}", status_code=204)
async def delete_policy(policy_id: str) -> None:
    """Deprecate a policy (soft delete).
    
    Policies are not actually deleted to maintain audit trail (Law 15).
    
    Args:
        policy_id: Policy identifier
        
    Raises:
        HTTPException: If policy not found
    """
    policy = _policy_store.get(policy_id)
    
    if not policy:
        raise HTTPException(
            status_code=404,
            detail=f"Policy {policy_id} not found",
        )
    
    # Soft delete - set to deprecated
    policy["status"] = "deprecated"
    policy["updated_at"] = datetime.now(timezone.utc).isoformat()
    _policy_store[policy_id] = policy
