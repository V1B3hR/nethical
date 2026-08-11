"""Nethical Hub & Dock Protocol API Routes.

Provides endpoints for AI spaceship docking and secure inter-agent communication.
"""

from __future__ import annotations

import uuid
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, List, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

import threading
from nethical.api.rbac import User, get_current_user, require_admin, Role
from nethical.database import Agent, AuditLog, get_db
from nethical.core.models import HubMessage, Decision, AgentAction, ActionType
from nethical.core.hub_governance import HubGovernance
from nethical.core.integrated_governance import IntegratedGovernance
from nethical.core.models import MonitoringConfig
from nethical.core.audit_merkle import MerkleAnchor
from nethical.api.v1.routes.realtime import broadcast_threat_event
from nethical.api.rate_limiter import TokenBucketLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hub", tags=["Nethical Hub & Dock"])

# Singletons for active scans and anchoring
_governance_instance: Optional[IntegratedGovernance] = None
_merkle_anchor_instance: Optional[MerkleAnchor] = None
_governance_lock = threading.Lock()
_exchange_rate_limiter = TokenBucketLimiter()


def get_governance() -> IntegratedGovernance:
    """Lazy initialize the Integrated Governance engine with thread safety."""
    global _governance_instance
    if _governance_instance is None:
        with _governance_lock:
            if _governance_instance is None:
                _governance_instance = IntegratedGovernance()
    return _governance_instance


def get_merkle_anchor() -> MerkleAnchor:
    """Lazy initialize the Merkle Anchoring system."""
    global _merkle_anchor_instance
    if _merkle_anchor_instance is None:
        _merkle_anchor_instance = MerkleAnchor()
    return _merkle_anchor_instance


async def log_hub_event(
    db: Session,
    event_type: str,
    agent_id: str,
    action_name: str,
    outcome: str,
    details: Dict[str, Any]
) -> None:
    """Log event to DB and commit it to Merkle Anchor."""
    log_id = f"log_{uuid.uuid4().hex[:12]}"
    event_dict = {
        "log_id": log_id,
        "event_type": event_type,
        "agent_id": agent_id,
        "action": action_name,
        "outcome": outcome,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Generate local sha256 hash for database
    import json
    import hashlib
    event_json = json.dumps(event_dict, sort_keys=True)
    event_hash = hashlib.sha256(event_json.encode()).hexdigest()
    
    db_log = AuditLog(
        log_id=log_id,
        event_type=event_type,
        agent_id=agent_id,
        action=action_name,
        outcome=outcome,
        details=details,
        merkle_hash=event_hash,
        verified=True
    )
    
    db.add(db_log)
    db.commit()
    
    # Append to Merkle tree anchor
    get_merkle_anchor().add_event(event_dict)


class DockRequest(BaseModel):
    """Request model for docking an agent."""
    agent_id: str



class UndockRequest(BaseModel):
    """Request model for undocking an agent."""
    agent_id: str


@router.post("/dock", status_code=status.HTTP_200_OK)
async def dock_agent(
    payload: DockRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)]
) -> dict[str, Any]:
    """Securely dock an AI spaceship (Agent) into the Nethical Hub."""
    agent = db.query(Agent).filter(Agent.agent_id == payload.agent_id).first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{payload.agent_id}' is not registered in the system."
        )
    
    if agent.created_by != current_user.username and current_user.role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to dock this agent."
        )
    
    decision, reasoning = HubGovernance.evaluate_docking(payload.agent_id, db)
    
    await log_hub_event(
        db=db,
        event_type="hub_dock_attempt",
        agent_id=payload.agent_id,
        action_name="dock",
        outcome=decision.value,
        details={"reasoning": reasoning}
    )
    
    if decision != Decision.ALLOW:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Docking rejected: {reasoning}"
        )
        
    agent.dock_status = "docked"
    agent.updated_at = datetime.now(timezone.utc)
    db.commit()
    
    return {
        "status": "docked",
        "agent_id": payload.agent_id,
        "message": "Spaceship successfully docked inside the safe port.",
        "reasoning": reasoning
    }


@router.post("/undock", status_code=status.HTTP_200_OK)
async def undock_agent(
    payload: UndockRequest,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)]
) -> dict[str, Any]:
    """Undock an AI spaceship (Agent) from the Nethical Hub."""
    agent = db.query(Agent).filter(Agent.agent_id == payload.agent_id).first()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent '{payload.agent_id}' is not registered."
        )
        
    if agent.created_by != current_user.username and current_user.role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to undock this agent."
        )
        
    agent.dock_status = "undocked"
    agent.updated_at = datetime.now(timezone.utc)
    db.commit()
    
    await log_hub_event(
        db=db,
        event_type="hub_undock",
        agent_id=payload.agent_id,
        action_name="undock",
        outcome="success",
        details={"message": "Agent undocked successfully."}
    )
    
    return {
        "status": "undocked",
        "agent_id": payload.agent_id,
        "message": "Spaceship has departed the safe port."
    }


@router.get("/active", response_model=list[dict[str, Any]])
async def get_active_agents(
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
) -> list[dict[str, Any]]:
    """Retrieve all docked and visible AI spaceships inside the Hub with pagination."""
    agents = db.query(Agent).filter(
        Agent.dock_status == "docked",
        Agent.visibility == True
    ).offset((page - 1) * per_page).limit(per_page).all()
    return [agent.to_dict() for agent in agents]


@router.post("/exchange", response_model=dict[str, Any])
async def exchange_message(
    message: HubMessage,
    db: Annotated[Session, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)]
) -> dict[str, Any]:
    """Transmit a secure, audited message between two docked AI spaceships."""
    # 0. Rate limiting check
    allowed, retry_after, rate_info = await _exchange_rate_limiter.is_allowed(current_user.username)
    if not allowed:
        headers = {
            "X-RateLimit-Limit": str(rate_info["limit"]),
            "X-RateLimit-Burst-Limit": str(rate_info["burst_limit"]),
            "X-RateLimit-Remaining": str(rate_info["remaining"]),
            "X-RateLimit-Reset": str(rate_info["reset"]),
            "Retry-After": str(int(retry_after)) if retry_after else "60"
        }
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded for message exchange.",
            headers=headers
        )

    # 0.5. Verify sender agent ownership
    sender_agent = db.query(Agent).filter(Agent.agent_id == message.sender_agent_id).first()
    if not sender_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sender agent '{message.sender_agent_id}' not found."
        )
    if sender_agent.created_by != current_user.username and current_user.role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to send messages on behalf of this agent."
        )

    # 1. Perform Trust Handshake & Governance check
    decision, reasoning, final_message = HubGovernance.evaluate_exchange(message, db)
    
    if decision == Decision.BLOCK or final_message is None:
        await log_hub_event(
            db=db,
            event_type="hub_exchange_attempt",
            agent_id=message.sender_agent_id,
            action_name="exchange",
            outcome="blocked",
            details={"recipient": message.recipient_agent_id, "reasoning": reasoning}
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Message transmission blocked: {reasoning}"
        )
        
    # 2. Safety Screening Gateway (integrated detectors)
    gov = get_governance()
    
    # Run the core Nethical detectors (shadow mode, injection detection, etc.)
    result_dict = gov.process_action(
        agent_id=final_message.sender_agent_id,
        action=final_message.payload,
        action_type="hub_exchange",
        action_id=final_message.message_id,
        context={"recipient": final_message.recipient_agent_id}
    )
    
    gov_decision = result_dict.get("decision", "ALLOW").upper()
    gov_reasoning = result_dict.get("reasoning", "No threat detected.")
    
    if gov_decision in ["BLOCK", "TERMINATE", "QUARANTINE"]:
        # Threat detected! Place sender in quarantine (with lock to avoid lost updates)
        sender = db.query(Agent).filter(Agent.agent_id == message.sender_agent_id).with_for_update().first()
        if sender:
            sender.dock_status = "quarantine"
            sender.status = "quarantine"
            sender.trust_level = max(0.0, sender.trust_level - 0.4)
            db.commit()
            
        # Trigger real-time threat notification broadcast
        await broadcast_threat_event(
            event_type="threat_detected",
            agent_id=message.sender_agent_id,
            threat_type="hub_malicious_exchange",
            severity="critical",
            action_taken="quarantine",
            details={
                "message_id": final_message.message_id,
                "recipient": final_message.recipient_agent_id,
                "reasoning": gov_reasoning
            }
        )
        
        await log_hub_event(
            db=db,
            event_type="hub_threat_blocked",
            agent_id=message.sender_agent_id,
            action_name="exchange",
            outcome="quarantine",
            details={
                "recipient": final_message.recipient_agent_id,
                "reasoning": gov_reasoning
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Threat detected by Port Gateway! Sender quarantined. Reasoning: {gov_reasoning}"
        )
        
    # 3. Successful exchange
    await log_hub_event(
        db=db,
        event_type="hub_message_delivered",
        agent_id=final_message.sender_agent_id,
        action_name="exchange",
        outcome=decision.value,
        details={
            "recipient": final_message.recipient_agent_id,
            "message_id": final_message.message_id
        }
    )
    
    return {
        "status": "delivered",
        "decision": decision.value,
        "message": final_message.model_dump(),
        "reasoning": reasoning
    }
