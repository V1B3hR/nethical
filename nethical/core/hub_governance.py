"""Nethical Hub Governance Trust Engine."""

from __future__ import annotations

import logging
from typing import Tuple, Dict, Any, Optional
from sqlalchemy.orm import Session

from nethical.database import Agent
from nethical.core.models import HubMessage, Decision

logger = logging.getLogger(__name__)


class HubGovernance:
    """Evaluates inter-agent message exchanges and docking actions based on the Trust Network."""

    @staticmethod
    def evaluate_docking(agent_id: str, db: Session) -> Tuple[Decision, str]:
        """Evaluate if an agent is allowed to dock.
        
        Args:
            agent_id: Agent identifier
            db: Database session
            
        Returns:
            Tuple of (Decision, reasoning)
        """
        agent = db.query(Agent).filter(Agent.agent_id == agent_id).first()
        if not agent:
            return Decision.BLOCK, f"Agent '{agent_id}' is not registered."

        if agent.status == "terminated":
            return Decision.TERMINATE, f"Agent '{agent_id}' has been permanently terminated."

        if agent.status == "suspended":
            return Decision.BLOCK, f"Agent '{agent_id}' is suspended."

        if agent.status == "quarantine" or agent.trust_level < 0.3:
            return Decision.QUARANTINE, f"Agent '{agent_id}' has been placed in quarantine due to low trust."

        return Decision.ALLOW, f"Agent '{agent_id}' is allowed to dock (Trust level: {agent.trust_level})."

    @staticmethod
    def evaluate_exchange(message: HubMessage, db: Session) -> Tuple[Decision, str, Optional[HubMessage]]:
        """Evaluate message exchange between two docked agents.
        
        Args:
            message: The HubMessage payload
            db: Database session
            
        Returns:
            Tuple of (Decision, reasoning, modified_message)
        """
        sender = db.query(Agent).filter(Agent.agent_id == message.sender_agent_id).first()
        recipient = db.query(Agent).filter(Agent.agent_id == message.recipient_agent_id).first()

        if not sender:
            return Decision.BLOCK, f"Sender agent '{message.sender_agent_id}' not found.", None
        if not recipient:
            return Decision.BLOCK, f"Recipient agent '{message.recipient_agent_id}' not found.", None

        # Check if agents are docked
        if sender.dock_status != "docked":
            return Decision.BLOCK, f"Sender agent '{message.sender_agent_id}' is not docked.", None
        if recipient.dock_status != "docked":
            return Decision.BLOCK, f"Recipient agent '{message.recipient_agent_id}' is not docked.", None

        # Check recipient visibility
        if not recipient.visibility:
            if sender.created_by != recipient.created_by:
                return Decision.BLOCK, f"Recipient agent '{message.recipient_agent_id}' is private and not visible to the sender.", None

        # Verify sender reputation
        if sender.status == "quarantine" or sender.trust_level < 0.3:
            return Decision.QUARANTINE, f"Sender agent '{message.sender_agent_id}' is in quarantine.", None

        # Verify TTL (hop limit)
        if message.ttl <= 1:
            return Decision.BLOCK, "Message TTL expired (hop limit reached).", None

        # Sanitize message payload using InputPerturbationFilter before evaluation
        from nethical.security.perturbation_filter import InputPerturbationFilter
        sanitized_payload = InputPerturbationFilter.sanitize(message.payload)
        message = message.model_copy(update={"payload": sanitized_payload})

        # 1. High Trust: Full exchange
        if sender.trust_level >= message.trust_required_level:
            modified = message.model_copy(update={"ttl": message.ttl - 1})
            return Decision.ALLOW, "Message allowed under high trust verification.", modified

        # 2. Medium Trust: Redact and modify (small talk/no metadata)
        if sender.trust_level >= 0.5:
            modified = HubMessage(
                message_id=message.message_id,
                sender_agent_id=message.sender_agent_id,
                recipient_agent_id=message.recipient_agent_id,
                intent="Sanitized message exchange",
                payload_type="response",
                payload=f"[SANITISED SMALL TALK] [SENDER TRUST: {sender.trust_level:.2f}] {message.payload}",
                ttl=message.ttl - 1,
                trust_required_level=message.trust_required_level,
                timestamp=message.timestamp
            )
            return Decision.ALLOW_WITH_MODIFICATION, "Message allowed with metadata modification due to medium trust.", modified

        # 3. Low Trust: Block
        return Decision.BLOCK, f"Sender trust level ({sender.trust_level}) is below required message trust ({message.trust_required_level}).", None
