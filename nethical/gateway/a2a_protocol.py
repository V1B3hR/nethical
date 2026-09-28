# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Agent-to-Agent (A2A) Contractual Governance Protocol - Faza 4 Roadmapy.

Zapewnia bezpieczną, obustronnie uwierzytelnioną i kryptograficznie wiążącą współpracę
pomiędzy autonomicznymi agentami AI (Multi-Agent Swarms, interakcje A2A).
"""

import hashlib
import hmac
import logging
import os
import secrets
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import canonical_json_bytes

logger = logging.getLogger("nethical.gateway.a2a_protocol")


class A2ACapabilityBoundary(BaseModel):
    """Zbiór reguł i ograniczeń dla danej sesji współpracy międzyagentowej."""

    allowed_tools: List[str] = Field(default_factory=list, description="Biała lista dozwolonych narzędzi")
    max_budget_units: float = Field(default=100.0, description="Maksymalny limit zasobów / tokenów na sesję")
    disallowed_patterns: List[str] = Field(
        default_factory=lambda: ["drop table", "rm -rf", "override", "format c:", "grant all"],
        description="Zakazane frazy i polecenia niszczące",
    )
    pii_clearance_level: str = Field(default="ANONYMIZED", description="Poziom dostępu do PII: NONE, ANONYMIZED, FULL")
    require_ambassador_signoff: bool = Field(default=False, description="Wymóg kontrasygnaty Ambasadora Błyskawicy")


class A2ASessionContract(BaseModel):
    """Kryptograficzny kontrakt sesji A2A podpisany przez obie strony."""

    session_id: str
    initiator_agent_id: str
    target_agent_id: str
    boundaries: A2ACapabilityBoundary
    nonce: str
    created_at: str
    expires_at: str
    initiator_signature: str
    target_signature: str
    merkle_receipt_id: Optional[str] = None
    spent_budget_units: float = 0.0
    active: bool = True


class A2AHandshakeManager:
    """Zarządca negocjacji kontraktów oraz weryfikacji operacji w protokole A2A."""

    def __init__(
        self,
        default_ttl_seconds: int = 3600,
        cluster_secret: Optional[str] = None,
    ) -> None:
        self.default_ttl = default_ttl_seconds
        raw_secret = (
            cluster_secret
            or os.getenv("NETHICAL_A2A_SECRET")
            or os.getenv("NETHICAL_SECRET_KEY")
            or "nethical_default_sovereign_a2a_cluster_key_2026"
        )
        self.cluster_secret = raw_secret.encode("utf-8")
        self.active_sessions: Dict[str, A2ASessionContract] = {}

    def _sign_proposal(self, initiator_id: str, proposal_payload: Dict[str, Any]) -> str:
        """Wylicza kryptograficzny HMAC-SHA256 dla propozycji kontraktu."""
        msg = f"INIT_SIGN::{initiator_id}::".encode("utf-8") + canonical_json_bytes(proposal_payload)
        return hmac.new(self.cluster_secret, msg, hashlib.sha256).hexdigest()

    def _sign_accept(self, target_id: str, init_sig: str, proposal_payload: Dict[str, Any]) -> str:
        """Wylicza kryptograficzną kontrasygnatę HMAC-SHA256 dla akceptacji kontraktu."""
        msg = f"TARGET_ACCEPT::{target_id}::{init_sig}::".encode("utf-8") + canonical_json_bytes(proposal_payload)
        return hmac.new(self.cluster_secret, msg, hashlib.sha256).hexdigest()

    def propose_handshake(
        self,
        initiator_id: str,
        target_id: str,
        boundaries: Optional[A2ACapabilityBoundary] = None,
        ttl_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Inicjator tworzy propozycję kontraktu partnerskiego (A2A Proposal)."""
        session_id = f"A2A-{uuid.uuid4().hex[:12].upper()}"
        nonce = secrets.token_hex(16)
        ttl = ttl_seconds or self.default_ttl
        now = datetime.now(timezone.utc)
        expires = (now + timedelta(seconds=ttl)).isoformat()

        bound = boundaries or A2ACapabilityBoundary()

        proposal_payload = {
            "session_id": session_id,
            "initiator_agent_id": initiator_id,
            "target_agent_id": target_id,
            "boundaries": bound.model_dump(),
            "nonce": nonce,
            "created_at": now.isoformat(),
            "expires_at": expires,
        }

        # Podpis kryptograficzny HMAC-SHA256 inicjatora
        init_sig = self._sign_proposal(initiator_id, proposal_payload)

        return {
            "proposal": proposal_payload,
            "initiator_signature": init_sig,
        }

    def accept_handshake(
        self,
        handshake_offer: Dict[str, Any],
        target_id: str,
        merkle_receipt_id: Optional[str] = None,
    ) -> A2ASessionContract:
        """Strona docelowa akceptuje i kontrasygnuje kontrakt partnerski po kryptograficznej weryfikacji."""
        proposal = handshake_offer["proposal"]
        init_sig = handshake_offer["initiator_signature"]

        # Weryfikacja kryptograficzna podpisu inicjatora (HMAC Constant-Time)
        expected_sig = self._sign_proposal(proposal["initiator_agent_id"], proposal)
        if not hmac.compare_digest(init_sig, expected_sig):
            raise PermissionError(
                f"Sfałszowany lub unieważniony podpis inicjatora propozycji A2A (Agent: '{proposal.get('initiator_agent_id')}')!"
            )

        if proposal["target_agent_id"] != target_id:
            raise PermissionError(
                f"Niezgodność odbiorcy oferty: oczekiwano {proposal['target_agent_id']}, otrzymano {target_id}"
            )

        # Kontrasygnata HMAC odbiorcy
        target_sig = self._sign_accept(target_id, init_sig, proposal)

        contract = A2ASessionContract(
            session_id=proposal["session_id"],
            initiator_agent_id=proposal["initiator_agent_id"],
            target_agent_id=target_id,
            boundaries=A2ACapabilityBoundary(**proposal["boundaries"]),
            nonce=proposal["nonce"],
            created_at=proposal["created_at"],
            expires_at=proposal["expires_at"],
            initiator_signature=init_sig,
            target_signature=target_sig,
            merkle_receipt_id=merkle_receipt_id,
            spent_budget_units=0.0,
            active=True,
        )

        self.active_sessions[contract.session_id] = contract
        logger.info(
            "Zawarto pomyślnie kontrakt A2A: %s pomiędzy [%s] a [%s]",
            contract.session_id,
            contract.initiator_agent_id,
            contract.target_agent_id,
        )
        return contract

    def validate_tool_execution(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        cost_units: float = 1.0,
    ) -> Tuple[bool, Optional[str]]:
        """Waliduje wywołanie narzędziowe w ramach aktywnej sesji kontraktowej."""
        contract = self.active_sessions.get(session_id)
        if not contract:
            return False, f"Sesja kontraktowa A2A '{session_id}' nie istnieje."

        if not contract.active:
            return False, f"Sesja A2A '{session_id}' została zakończona."

        # Sprawdzenie terminu ważności
        exp_time = datetime.fromisoformat(contract.expires_at)
        if datetime.now(timezone.utc) > exp_time:
            contract.active = False
            return False, f"Sesja A2A '{session_id}' wygasła o {contract.expires_at}."

        # Sprawdzenie białej listy narzędzi (jeśli zdefiniowano)
        if contract.boundaries.allowed_tools and tool_name not in contract.boundaries.allowed_tools:
            return False, f"Narzędzie '{tool_name}' nie znajduje się na białej liście kontraktu: {contract.boundaries.allowed_tools}"

        # Sprawdzenie budżetu
        if contract.spent_budget_units + cost_units > contract.boundaries.max_budget_units:
            return False, f"Przekroczenie budżetu sesji ({contract.spent_budget_units + cost_units} > {contract.boundaries.max_budget_units})"

        # Sprawdzenie wzorców zakazanych
        raw_text = " ".join(f"{k}={v}" for k, v in arguments.items()).lower()
        for dis in contract.boundaries.disallowed_patterns:
            if dis in raw_text:
                return False, f"Wykryto niedozwolony wzorzec kontraktowy [{dis}] w argumentach!"

        # Zaktualizowanie zużycia budżetu
        contract.spent_budget_units += cost_units
        return True, None

    def terminate_session(self, session_id: str) -> bool:
        """Natychmiastowe zamknięcie sesji (Killswitch kontraktowy)."""
        contract = self.active_sessions.get(session_id)
        if contract:
            contract.active = False
            logger.warning("Sesja kontraktowa A2A %s została wymuszenie przerwana.", session_id)
            return True
        return False
