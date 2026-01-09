"""Agent management routes for API v1.

Provides CRUD operations for AI agent configuration.

Endpoints:
- POST /api/v1/agents - Create new agent
- GET /api/v1/agents - List all agents with pagination
- GET /api/v1/agents/{id} - Get agent details
- PATCH /api/v1/agents/{id} - Update agent configuration
- DELETE /api/v1/agents/{id} - Delete agent
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from nethical.api.rbac import User, get_current_user, require_admin
from nethical.database import Agent, get_db

router = APIRouter(prefix="/agents", tags=["Agent Management"])


class AgentCreate(BaseModel):
    """Request to create a new agent."""
    
    agent_id: str = Field(..., min_length=1, max_length=255, description="Unique agent identifier")
    name: str = Field(..., min_length=1, max_length=255, description="Agent name")
    agent_type: str = Field(default="general", description="Agent type (general, specialized, etc.)")
    description: Optional[str] = Field(None, description="Agent description")
    trust_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Trust level (0.0-1.0)")
    status: str = Field(default="active", description="Agent status")
    configuration: dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    region_id: Optional[str] = Field(None, description="Region identifier")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "agent_id": "agent-gpt4-001",
                    "name": "GPT-4 Assistant",
                    "agent_type": "llm",
                    "description": "GPT-4 powered AI assistant",
                    "trust_level": 0.9,
                    "status": "active",
                    "configuration": {
                        "model": "gpt-4",
                        "temperature": 0.7,
                        "max_tokens": 2000
                    },
                    "metadata": {
                        "department": "engineering",
                        "environment": "production"
                    }
                }
            ]
        }
    }


class AgentUpdate(BaseModel):
    """Request to update an agent."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Agent name")
    agent_type: Optional[str] = Field(None, description="Agent type")
    description: Optional[str] = Field(None, description="Agent description")
    trust_level: Optional[float] = Field(None, ge=0.0, le=1.0, description="Trust level (0.0-1.0)")
    status: Optional[str] = Field(None, description="Agent status")
    configuration: Optional[dict[str, Any]] = Field(None, description="Agent configuration")
    metadata: Optional[dict[str, Any]] = Field(None, description="Additional metadata")
    region_id: Optional[str] = Field(None, description="Region identifier")


class AgentResponse(BaseModel):
    """Agent response model."""
    
    id: int
    agent_id: str
    name: str
    agent_type: str
    description: Optional[str]
    trust_level: float
    status: str
    configuration: dict[str, Any]
    metadata: dict[str, Any]
    region_id: Optional[str]
    created_at: str
    updated_at: str
    created_by: Optional[str]


class AgentListResponse(BaseModel):
    """Paginated list of agents."""
    
    agents: list[AgentResponse]
    total: int
    page: int
    per_page: int
    pages: int


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent: AgentCreate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_admin)],
) -> AgentResponse:
    """Create a new agent.
    
    **Required role:** admin
    
    Args:
        agent: Agent creation data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Created agent
        
    Raises:
        HTTPException: 409 if agent_id already exists
    """
    # Check if agent_id already exists
    existing = db.query(Agent).filter(Agent.agent_id == agent.agent_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent with agent_id '{agent.agent_id}' already exists"
        )
    
    # Create new agent
    db_agent = Agent(
        agent_id=agent.agent_id,
        name=agent.name,
        agent_type=agent.agent_type,
        description=agent.description,
        trust_level=agent.trust_level,
        status=agent.status,
        configuration=agent.configuration,
        metadata=agent.metadata,
        region_id=agent.region_id,
        created_by=current_user.username,
    )
    
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)
    
    return AgentResponse(**db_agent.to_dict())


@router.get("", response_model=AgentListResponse)
async def list_agents(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
) -> AgentListResponse:
    """List all agents with pagination and filtering.
    
    **Required role:** Any authenticated user
    
    Args:
        db: Database session
        current_user: Current authenticated user
        page: Page number (1-indexed)
        per_page: Items per page (max 100)
        status: Filter by status
        agent_type: Filter by agent type
        
    Returns:
        Paginated list of agents
    """
    query = db.query(Agent)
    
    # Apply filters
    if status:
        query = query.filter(Agent.status == status)
    if agent_type:
        query = query.filter(Agent.agent_type == agent_type)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    agents = query.offset((page - 1) * per_page).limit(per_page).all()
    
    return AgentListResponse(
        agents=[AgentResponse(**agent.to_dict()) for agent in agents],
        total=total,
        page=page,
        per_page=per_page,
        pages=(total + per_page - 1) // per_page
    )


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> AgentResponse:
    """Get agent details by ID.
    
    **Required role:** Any authenticated user
    
    Args:
        agent_id: Agent identifier
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Agent details
        
    Raises:
        HTTPException: 404 if agent not found
    """
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found"
        )
    
    return AgentResponse(**agent.to_dict())


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    agent_update: AgentUpdate,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_admin)],
) -> AgentResponse:
    """Update agent configuration.
    
    **Required role:** admin
    
    Args:
        agent_id: Agent identifier
        agent_update: Agent update data
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Updated agent
        
    Raises:
        HTTPException: 404 if agent not found
    """
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found"
        )
    
    # Update fields
    update_data = agent_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(agent, field, value)
    
    agent.updated_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(agent)
    
    return AgentResponse(**agent.to_dict())


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(require_admin)],
) -> None:
    """Delete an agent.
    
    **Required role:** admin
    
    Args:
        agent_id: Agent identifier
        db: Database session
        current_user: Current authenticated user
        
    Raises:
        HTTPException: 404 if agent not found
    """
    agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{agent_id}' not found"
        )
    
    db.delete(agent)
    db.commit()
