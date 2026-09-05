"""Human-in-the-Loop (HITL) Case Management & Escalation Engine (Faza 5).

Wdraża wymóg Artykułu 14 EU AI Act (Human Oversight) oraz standardy UK FCA:
- Kolejkowanie decyzji RESTRICT do zatwierdzenia przez operatora ludzkiego
- Priorytetyzacja eskalacji (URGENT, HIGH, STANDARD) i kontrola limitów czasu (SLA)
- Bezpieczny audyt orzeczeń operatora z pieczęcią w Merkle-DAG
- Zabezpieczenie przed samowolnym przekroczeniem granic autonomii przez agentów
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import MerkleLedger

logger = logging.getLogger("nethical.gateway.hitl")


class HITLResolution(BaseModel):
    """Formalne orzeczenie operatora ludzkiego dotyczące eskalowanego wywołania."""

    decision: str = Field(..., description="APPROVE, REJECT, OVERRIDE_ALLOW, TERMINATE_AGENT")
    reviewer_id: str = Field(..., description="Identyfikator audytora lub oficera compliance")
    reviewer_notes: str = Field(..., description="Uzasadnienie orzeczenia ludzkiego")
    modified_arguments: Optional[Dict[str, Any]] = Field(default=None, description="Skorygowane bezpieczne argumenty narzędzia")
    receipt_id: Optional[str] = Field(default=None, description="Kwit Merkle-DAG pieczętujący decyzję ludzką")
    resolved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HITLTicket(BaseModel):
    """Bilet sprawy w kolejce nadzoru ludzkiego (Human-in-the-Loop)."""

    ticket_id: str = Field(default_factory=lambda: f"hitl_{uuid.uuid4().hex[:10]}")
    agent_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reasons: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    priority: str = Field(default="STANDARD", description="URGENT, HIGH, STANDARD")
    status: str = Field(default="PENDING", description="PENDING, RESOLVED, EXPIRED")
    timeout_seconds: int = 300
    context: Dict[str, Any] = Field(default_factory=dict)
    enqueued_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolution: Optional[HITLResolution] = None


class HITLQueueManager:
    """Zarządca kolejki spraw i orzecznictwa Human-in-the-Loop."""

    def __init__(self, ledger: Optional[MerkleLedger] = None) -> None:
        self.ledger = ledger or MerkleLedger()
        self.tickets: Dict[str, HITLTicket] = {}
        self.total_enqueued = 0
        self.total_approved = 0
        self.total_rejected = 0

    def enqueue_ticket(
        self,
        agent_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        reasons: Optional[List[str]] = None,
        violations: Optional[List[str]] = None,
        priority: str = "STANDARD",
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 300,
    ) -> HITLTicket:
        """Tworzy i rejestruje nowy bilet w kolejce eskalacji operatorskiej."""
        ticket = HITLTicket(
            agent_id=agent_id,
            tool_name=tool_name,
            arguments=arguments,
            reasons=reasons or [],
            violations=violations or [],
            priority=priority,
            context=context or {},
            timeout_seconds=timeout_seconds,
        )
        self.tickets[ticket.ticket_id] = ticket
        self.total_enqueued += 1
        logger.info(
            "Wprowadzono bilet HITL [%s] priorytet: %s dla agenta '%s' (narzędzie: '%s')",
            ticket.ticket_id,
            priority,
            agent_id,
            tool_name,
        )
        return ticket

    def resolve_ticket(
        self,
        ticket_id: str,
        reviewer_id: str,
        decision: str,
        notes: str,
        modified_arguments: Optional[Dict[str, Any]] = None,
    ) -> HITLTicket:
        """Zatwierdza lub odrzuca bilet przez uprawnionego człowieka, pieczętując wynik w MerkleLedger."""
        if ticket_id not in self.tickets:
            raise KeyError(f"Bilet HITL {ticket_id} nie istnieje w kolejce.")

        ticket = self.tickets[ticket_id]
        if ticket.status == "RESOLVED":
            raise ValueError(f"Bilet {ticket_id} został już uprzednio rozwiązany.")

        valid_decisions = ["APPROVE", "REJECT", "OVERRIDE_ALLOW", "TERMINATE_AGENT"]
        if decision not in valid_decisions:
            raise ValueError(f"Nieprawidłowa decyzja HITL: '{decision}'. Dozwolone: {valid_decisions}")

        # Pieczętowanie orzeczenia ludzkiego w rejestrze Merkle-DAG (niezaprzeczalność)
        hitl_payload = {
            "type": "HITL_HUMAN_OVERSIGHT_RESOLUTION",
            "ticket_id": ticket.ticket_id,
            "agent_id": ticket.agent_id,
            "tool_name": ticket.tool_name,
            "decision": decision,
            "reviewer_id": reviewer_id,
            "reviewer_notes": notes,
            "original_arguments": ticket.arguments,
            "modified_arguments": modified_arguments,
        }
        receipt = self.ledger.append_decision(
            decision_data=hitl_payload,
            ambassador_notes=f"HITL Verified by Reviewer {reviewer_id} ({decision})",
        )

        resolution = HITLResolution(
            decision=decision,
            reviewer_id=reviewer_id,
            reviewer_notes=notes,
            modified_arguments=modified_arguments,
            receipt_id=receipt.receipt_id,
        )

        ticket.status = "RESOLVED"
        ticket.resolution = resolution

        if decision in ["APPROVE", "OVERRIDE_ALLOW"]:
            self.total_approved += 1
        else:
            self.total_rejected += 1

        logger.info("Rozwiązano bilet HITL [%s]: %s przez audytora '%s'", ticket_id, decision, reviewer_id)
        return ticket

    def get_pending_tickets(self, priority: Optional[str] = None) -> List[HITLTicket]:
        """Zwraca listę oczekujących biletów, opcjonalnie przefiltrowanych wg priorytetu."""
        pending = [t for t in self.tickets.values() if t.status == "PENDING"]
        if priority:
            pending = [t for t in pending if t.priority == priority]
        # Sortowanie wg priorytetu: URGENT > HIGH > STANDARD
        priority_order = {"URGENT": 0, "HIGH": 1, "STANDARD": 2}
        return sorted(pending, key=lambda t: priority_order.get(t.priority, 3))

    def get_ticket(self, ticket_id: str) -> Optional[HITLTicket]:
        """Pobiera bilet o wskazanym ID."""
        return self.tickets.get(ticket_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Zwraca metryki wydajnościowe kolejki nadzoru ludzkiego."""
        pending_count = sum(1 for t in self.tickets.values() if t.status == "PENDING")
        resolved_count = sum(1 for t in self.tickets.values() if t.status == "RESOLVED")
        return {
            "total_enqueued": self.total_enqueued,
            "pending_count": pending_count,
            "resolved_count": resolved_count,
            "total_approved": self.total_approved,
            "total_rejected": self.total_rejected,
            "approval_rate": round(self.total_approved / max(1, resolved_count), 2),
        }
