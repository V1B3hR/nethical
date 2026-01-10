"""Policy management routes for API v1.

Provides CRUD operations for governance policies.

Endpoints:
- POST /api/v1/policies - Create new policy
- GET /api/v1/policies - List all policies with pagination
- GET /api/v1/policies/{id} - Get policy details
- PATCH /api/v1/policies/{id} - Update policy
- DELETE /api/v1/policies/{id} - Delete policy
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from nethical.api.rbac import User, get_current_user, require_admin
from nethical.database import Policy, get_db

router = APIRouter(prefix="/policies", tags=["Policy Management"])


class PolicyRule(BaseModel):
    """A single rule within a policy."""
    
    id: str = Field(..., description="Rule identifier")
    condition: str = Field(..., description="Condition expression")
    action: str = Field(..., description="Action: ALLOW, RESTRICT, BLOCK, TERMINATE")
    priority: int = Field(default=0, description="Rule priority (higher = first)")
    description: Optional[str] = Field(None, description="Rule description")


class PolicyCreate(BaseModel):
    """Request to create a new policy."""
    
    policy_id: str = Field(..., min_length=1, max_length=255, description="Unique policy identifier")
    name: str = Field(..., min_length=1, max_length=255, description="Policy name")
    description: Optional[str] = Field(None, description="Policy description")
    version: str = Field(default="1.0.0", description="Policy version")
    policy_type: str = Field(default="governance", description="Policy type")
    priority: int = Field(default=100, description="Policy priority")
    status: str = Field(default="active", description="Policy status")
    rules: list[PolicyRule] = Field(default_factory=list, description="Policy rules")
    scope: str = Field(default="global", description="Policy scope")
    fundamental_laws: list[int] = Field(default_factory=list, description="Fundamental Laws")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "policy_id": "policy-data-access-001",
                    "name": "Data Access Policy",
                    "description": "Controls access to sensitive data",
                    "version": "1.0.0",
                    "policy_type": "governance",
                    "priority": 100,
                    "status": "active",
                    "rules": [
                        {
                            "id": "rule-1",
                            "condition": "action_type == 'data_access'",
                            "action": "RESTRICT",
                            "priority": 10,
                            "description": "Restrict data access by default"
                        }
                    ],
                    "scope": "global",
                    "fundamental_laws": [22],
                    "metadata": {
                        "department": "compliance",
                        "version_notes": "Initial version"
                    }
                }
            ]
        }
    }


class PolicyUpdate(BaseModel):
    """Request to update a policy."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Policy name")
    description: Optional[str] = Field(None, description="Policy description")
    version: Optional[str] = Field(None, description="Policy version")
    policy_type: Optional[str] = Field(None, description="Policy type")
    priority: Optional[int] = Field(None, description="Policy priority")
    status: Optional[str] = Field(None, description="Policy status")
    rules: Optional[list[PolicyRule]] = Field(None, description="Policy rules")
    scope: Optional[str] = Field(None, description="Policy scope")
    fundamental_laws: Optional[list[int]] = Field(None, description="Fundamental Laws")
    metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")


class PolicyResponse(BaseModel):
    """Policy response model."""
    
    id: int
    policy_id: str
    name: str
    description: Optional[str]
    version: str
    policy_type: str
    priority: int
    status: str
    rules: list[dict[str, Any]]
    scope: str
    fundamental_laws: list[int]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
    activated_at: Optional[str]
    deprecated_at: Optional[str]
    created_by: Optional[str]


class PolicyListResponse(BaseModel):
    """Paginated list of policies."""
    
    policies: list[PolicyResponse]
    total: int
    page: int
    per_page: int
    pages: int


@router.post("", response_model=PolicyResponse, status_code=status.HTTP_201_CREATED)
async def create_policy(
    policy: PolicyCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_admin)],
) -> PolicyResponse:
    """Create a new policy.
    
    **Required role:** admin
    
    Args:
        policy: Policy creation data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Created policy
        
    Raises:
        HTTPException: 409 if policy_id already exists
        HTTPException: 422 if validation fails
    """
    # Check if policy_id already exists
    existing = db.query(Policy).filter(Policy.policy_id == policy.policy_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Policy with policy_id '{policy.policy_id}' already exists"
        )
    
    # Validate rules
    for rule in policy.rules:
        if rule.action not in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid action '{rule.action}' in rule '{rule.id}'. Must be ALLOW, RESTRICT, BLOCK, or TERMINATE"
            )
    
    # Create new policy
    db_policy = Policy(
        policy_id=policy.policy_id,
        name=policy.name,
        description=policy.description,
        version=policy.version,
        policy_type=policy.policy_type,
        priority=policy.priority,
        status=policy.status,
        rules=[rule.model_dump() for rule in policy.rules],
        scope=policy.scope,
        fundamental_laws=policy.fundamental_laws,
        metadata=policy.metadata,
        created_by=current_user.username,
        activated_at=datetime.now(timezone.utc) if policy.status == "active" else None,
    )
    
    db.add(db_policy)
    db.commit()
    db.refresh(db_policy)
    
    return PolicyResponse(**db_policy.to_dict())


@router.get("", response_model=PolicyListResponse)
async def list_policies(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    policy_type: Optional[str] = Query(None, description="Filter by policy type"),
    scope: Optional[str] = Query(None, description="Filter by scope"),
) -> PolicyListResponse:
    """List all policies with pagination and filtering.
    
    **Required role:** Any authenticated user
    
    Args:
        db: Database session
        current_user: Current authenticated user
        page: Page number (1-indexed)
        per_page: Items per page (max 100)
        status: Filter by status
        policy_type: Filter by policy type
        scope: Filter by scope
        
    Returns:
        Paginated list of policies
    """
    query = db.query(Policy)
    
    # Apply filters
    if status:
        query = query.filter(Policy.status == status)
    if policy_type:
        query = query.filter(Policy.policy_type == policy_type)
    if scope:
        query = query.filter(Policy.scope == scope)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    policies = query.order_by(Policy.priority.desc()).offset((page - 1) * per_page).limit(per_page).all()
    
    return PolicyListResponse(
        policies=[PolicyResponse(**policy.to_dict()) for policy in policies],
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page
    )


@router.get("/{policy_id}", response_model=PolicyResponse)
async def get_policy(
    policy_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> PolicyResponse:
    """Get policy details by ID.
    
    **Required role:** Any authenticated user
    
    Args:
        policy_id: Policy identifier
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Policy details
        
    Raises:
        HTTPException: 404 if policy not found
    """
    policy = db.query(Policy).filter(Policy.policy_id == policy_id).first()
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy '{policy_id}' not found"
        )
    
    return PolicyResponse(**policy.to_dict())


@router.patch("/{policy_id}", response_model=PolicyResponse)
async def update_policy(
    policy_id: str,
    policy_update: PolicyUpdate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_admin)],
) -> PolicyResponse:
    """Update policy configuration.
    
    **Required role:** admin
    
    Args:
        policy_id: Policy identifier
        policy_update: Policy update data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Updated policy
        
    Raises:
        HTTPException: 404 if policy not found
        HTTPException: 422 if validation fails
    """
    policy = db.query(Policy).filter(Policy.policy_id == policy_id).first()
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy '{policy_id}' not found"
        )
    
    # Validate rules if provided
    if policy_update.rules:
        for rule in policy_update.rules:
            if rule.action not in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid action '{rule.action}' in rule '{rule.id}'"
                )
    
    # Update fields
    update_data = policy_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if field == "rules" and value is not None:
            setattr(policy, field, [rule.model_dump() if hasattr(rule, "model_dump") else rule for rule in value])
        else:
            setattr(policy, field, value)
    
    # Update activated_at if status changes to active
    if policy_update.status == "active" and policy.activated_at is None:
        policy.activated_at = datetime.now(timezone.utc)
    
    # Update deprecated_at if status changes to deprecated
    if policy_update.status == "deprecated" and policy.deprecated_at is None:
        policy.deprecated_at = datetime.now(timezone.utc)
    
    policy.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(policy)
    
    return PolicyResponse(**policy.to_dict())


@router.delete("/{policy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_policy(
    policy_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_admin)],
) -> None:
    """Delete a policy.
    
    **Required role:** admin
    
    Args:
        policy_id: Policy identifier
        db: Database session
        current_user: Current authenticated user
        
    Raises:
        HTTPException: 404 if policy not found
    """
    policy = db.query(Policy).filter(Policy.policy_id == policy_id).first()
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy '{policy_id}' not found"
        )
    
    db.delete(policy)
    db.commit()
