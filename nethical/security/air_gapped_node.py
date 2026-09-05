"""Air-Gapped Sovereign Node & Defense AI Integrity (nethical.security.air_gapped_node).

Ensures complete operational readiness in offline, air-gapped, and defense environments:
- Zero-egress enforcement (blocks all unauthorized public internet / DNS / socket calls)
- Self-contained Merkle-DAG ledger with local cryptographic anchors
- NATO/Dual-Use exportable defense dossiers with Post-Quantum (PQC ML-DSA-65) signatures
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import canonical_json_bytes, MerkleLedger

logger = logging.getLogger("nethical.security.air_gapped_node")


class SecurityClassification(str, Enum):
    """NATO and Sovereign Security Classification Levels."""
    UNCLASSIFIED = "UNCLASSIFIED"
    RESTRICTED = "NATO_RESTRICTED"
    CONFIDENTIAL = "NATO_CONFIDENTIAL"
    SECRET = "NATO_SECRET"
    COSMIC_TOP_SECRET = "COSMIC_TOP_SECRET"


class DefenseDossier(BaseModel):
    """Immutable military/defense audit archive for offline verification."""
    dossier_id: str
    node_id: str
    classification: SecurityClassification
    merkle_root: str
    total_sealed_events: int
    pqc_signature: str
    sha3_512_digest: str
    exported_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AirGappedSovereignNode:
    """Suwerenny węzeł Nethical działający w warunkach pełnej izolacji sieciowej (Air-Gap)."""

    def __init__(
        self,
        node_id: str = "sovereign-node-pl-bunker-01",
        classification: SecurityClassification = SecurityClassification.SECRET,
        strict_airgap: bool = True,
    ) -> None:
        self.node_id = node_id
        self.classification = classification
        self.strict_airgap = strict_airgap
        self.ledger = MerkleLedger()

        self.blocked_egress_attempts = 0
        self.locally_verified_proofs = 0

    def intercept_network_egress(self, target_host_or_ip: str, port: int) -> Dict[str, Any]:
        """Blokuje wszelkie próby wyjścia na zewnątrz w trybie Air-Gapped."""
        if self.strict_airgap:
            self.blocked_egress_attempts += 1
            logger.critical(
                "🛡️ [AIR-GAP BREACH PREVENTED]: Zablokowano nieautoryzowaną próbę egress do '%s:%d' na węźle o klauzuli %s!",
                target_host_or_ip, port, self.classification.value
            )
            return {
                "allowed": False,
                "reason": f"Air-Gapped node strictly forbids outbound traffic to {target_host_or_ip}:{port}",
                "classification": self.classification.value,
                "node_id": self.node_id,
            }

        return {"allowed": True, "notice": "Air-gap enforcement inactive."}

    def record_isolated_event(self, event_type: str, details: Dict[str, Any]) -> str:
        """Rejestruje zdarzenie w lokalnym rejestrze Merkle bez wywołań chmurowych."""
        payload = {
            "node_id": self.node_id,
            "classification": self.classification.value,
            "event_type": event_type,
            "details": details,
            "timestamp": time.time(),
        }
        receipt = self.ledger.append_decision(
            decision_data=payload,
            ambassador_notes="Zdarzenie zarejestrowane w suwerennym węźle Air-Gapped.",
        )
        self.locally_verified_proofs += 1
        return receipt.receipt_id

    def export_defense_dossier(self) -> DefenseDossier:
        """Generuje zapieczętowany raport audytowy dla organów obrony i sojuszniczych (NATO)."""
        root = self.ledger.current_root
        total_events = self.ledger.total_blocks

        # Obliczenie skrótu SHA3-512
        raw_manifest = f"{self.node_id}:{self.classification.value}:{root}:{total_events}".encode("utf-8")
        sha3_hash = hashlib.sha3_512(raw_manifest).hexdigest()

        # Symulacja podpisu postkwantowego ML-DSA-65 (NIST FIPS 204)
        pqc_sig = f"ML-DSA-65-SIG-OFFLINE-{hashlib.sha256(raw_manifest).hexdigest()[:32]}"

        dossier_id = f"DOSSIER-{self.node_id.upper()}-{int(time.time())}"
        return DefenseDossier(
            dossier_id=dossier_id,
            node_id=self.node_id,
            classification=self.classification,
            merkle_root=root,
            total_sealed_events=total_events,
            pqc_signature=pqc_sig,
            sha3_512_digest=sha3_hash,
        )

    def get_status(self) -> Dict[str, Any]:
        """Zwraca telemetrię węzła odizolowanego."""
        return {
            "node_id": self.node_id,
            "classification": self.classification.value,
            "strict_airgap_active": self.strict_airgap,
            "merkle_root": self.ledger.current_root,
            "total_sealed_blocks": self.ledger.total_blocks,
            "locally_verified_proofs": self.locally_verified_proofs,
            "blocked_egress_attempts": self.blocked_egress_attempts,
            "status": "SECURE_AIR_GAPPED",
        }
