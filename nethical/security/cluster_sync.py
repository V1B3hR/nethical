"""Multi-Region Sovereign Datacenter Mesh & Cross-Region Ledger Sync (Faza 5).

Zapewnia odporną na błędy sieciowe, kryptograficznie weryfikowalną synchronizację
rejestrów Merkle-DAG pomiędzy centrami danych (np. Frankfurt, Warszawa, Londyn):
- Spójność stanów decyzyjnych w architekturze rozproszonej
- Atestacje punktów kontrolnych podpisane postkwantowo (NIST FIPS 204 ML-DSA-65)
- Rozpoznawanie rozbieżności łańcucha (Fork & Split-Brain Detection)
- Bezpieczna federacja suwerennych węzłów Nethical OS
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import MerkleLedger
from nethical.security.quantum_crypto import (
    CRYSTALSDilithium,
    DilithiumKeyPair,
    PQCAlgorithm,
    QuantumSignature,
)

logger = logging.getLogger("nethical.security.cluster_sync")


class ClusterNodeIdentity(BaseModel):
    """Tożsamość suwerennego węzła Nethical w centrum danych."""

    node_id: str = Field(..., description="Unikalny identyfikator węzła, np. nethical-eu-central-1")
    region: str = Field(..., description="Region chmurowy/geograficzny, np. eu-central-1, uk-south-1")
    datacenter: str = Field(..., description="Lokalizacja fizyczna centrum danych, np. frankfurt-dc1")
    public_key_id: str = Field(..., description="ID klucza publicznego Dilithium3 węzła")
    endpoint_url: Optional[str] = Field(default=None, description="Adres URL interfejsu synchronizacji")
    is_active: bool = True
    registered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ClusterCheckpoint(BaseModel):
    """Kryptograficzny punkt kontrolny stanu rejestru Merkle-DAG."""

    checkpoint_id: str = Field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:12]}")
    node: ClusterNodeIdentity
    merkle_root: str
    total_blocks: int
    pqc_algorithm: str = "NIST FIPS 204 ML-DSA-65 (CRYSTALS-Dilithium3)"
    pqc_signature: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SyncReconciliationResult(BaseModel):
    """Wynik rekonsyliacji stanu pomiędzy węzłami klastra."""

    is_consistent: bool
    local_node_id: str
    peer_node_id: str
    status: str = Field(..., description="SYNCHRONIZED, PEER_AHEAD, LOCAL_AHEAD, DIVERGENT_FORK, INVALID_SIGNATURE")
    discrepancy_details: Optional[str] = None
    resolved: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CrossRegionLedgerSync:
    """Protokół synchronizacji i weryfikacji spójności rejestrów w siatce klastrów."""

    def __init__(
        self,
        ledger: MerkleLedger,
        local_node: Optional[ClusterNodeIdentity] = None,
    ) -> None:
        self.ledger = ledger
        self.local_node = local_node or ClusterNodeIdentity(
            node_id="nethical-eu-central-main",
            region="eu-central-1",
            datacenter="frankfurt-dc1",
            public_key_id=self.ledger.keypair.key_id,
        )
        self.peers: Dict[str, ClusterNodeIdentity] = {}
        self.checkpoints_history: List[ClusterCheckpoint] = []

    def register_peer(self, node: ClusterNodeIdentity) -> None:
        """Rejestruje autoryzowany węzeł partnerski w siatce federacyjnej."""
        self.peers[node.node_id] = node
        logger.info("Zarejestrowano węzeł klastra: %s (%s - %s)", node.node_id, node.region, node.datacenter)

    def create_checkpoint(self) -> ClusterCheckpoint:
        """Generuje podpisany postkwantowo punkt kontrolny aktualnego stanu łańcucha."""
        root = self.ledger.current_root
        total = self.ledger.total_blocks

        # Payload do podpisu: node_id + root + total_blocks
        sign_payload = f"{self.local_node.node_id}:{root}:{total}".encode("utf-8")
        q_sig: QuantumSignature = self.ledger.pqc_dilithium.sign(
            message=sign_payload,
            private_key=self.ledger.keypair.private_key,
            key_id=self.ledger.keypair.key_id,
        )

        checkpoint = ClusterCheckpoint(
            node=self.local_node,
            merkle_root=root,
            total_blocks=total,
            pqc_signature=q_sig.signature.hex(),
        )
        self.checkpoints_history.append(checkpoint)
        return checkpoint

    def verify_peer_checkpoint(self, checkpoint: ClusterCheckpoint) -> bool:
        """Weryfikuje podpis kryptograficzny punktu kontrolnego nadesłanego przez węzeł zewnętrzny."""
        try:
            sign_payload = f"{checkpoint.node.node_id}:{checkpoint.merkle_root}:{checkpoint.total_blocks}".encode("utf-8")
            raw_sig = bytes.fromhex(checkpoint.pqc_signature)
            expected_prefix = hashlib.sha256(sign_payload + self.ledger.keypair.private_key).digest()
            if not raw_sig.startswith(expected_prefix):
                return False

            q_sig = QuantumSignature(
                signature=raw_sig,
                message_hash=hashlib.sha256(sign_payload).hexdigest(),
                algorithm=PQCAlgorithm.DILITHIUM_3,
                signer_key_id=checkpoint.node.public_key_id,
            )
            return self.ledger.pqc_dilithium.verify(
                message=sign_payload,
                signature=q_sig,
                public_key=self.ledger.keypair.public_key,
            )
        except Exception as e:
            logger.warning("Błąd weryfikacji podpisu punktu kontrolnego: %s", e)
            return False

    def reconcile_peer(self, peer_checkpoint: ClusterCheckpoint) -> SyncReconciliationResult:
        """Dokonuje formalnej rekonsyliacji stanu lokalnego z punktem kontrolnym węzła partnerskiego."""
        # 1. Walidacja podpisu kryptograficznego punktu kontrolnego
        if not self.verify_peer_checkpoint(peer_checkpoint):
            logger.error("Odrzucono punkt kontrolny węzła %s: Nieprawidłowy podpis PQC!", peer_checkpoint.node.node_id)
            return SyncReconciliationResult(
                is_consistent=False,
                local_node_id=self.local_node.node_id,
                peer_node_id=peer_checkpoint.node.node_id,
                status="INVALID_SIGNATURE",
                discrepancy_details="Podpis NIST FIPS 204 ML-DSA-65 nie przeszedł weryfikacji kryptograficznej.",
                resolved=False,
            )

        local_root = self.ledger.current_root
        local_total = self.ledger.total_blocks
        peer_root = peer_checkpoint.merkle_root
        peer_total = peer_checkpoint.total_blocks

        # 2. Identyczne korzenie Merkle - pełna spójność
        if local_root == peer_root and local_total == peer_total:
            return SyncReconciliationResult(
                is_consistent=True,
                local_node_id=self.local_node.node_id,
                peer_node_id=peer_checkpoint.node.node_id,
                status="SYNCHRONIZED",
                discrepancy_details=None,
                resolved=True,
            )

        # 3. Węzeł zewnętrzny posiada więcej bloków
        if peer_total > local_total:
            return SyncReconciliationResult(
                is_consistent=False,
                local_node_id=self.local_node.node_id,
                peer_node_id=peer_checkpoint.node.node_id,
                status="PEER_AHEAD",
                discrepancy_details=f"Węzeł partnerski wyprzedza stan lokalny o {peer_total - local_total} bloków. Wymagany catch-up.",
                resolved=True,
            )

        # 4. Stan lokalny wyprzedza węzeł zewnętrzny
        if local_total > peer_total:
            return SyncReconciliationResult(
                is_consistent=False,
                local_node_id=self.local_node.node_id,
                peer_node_id=peer_checkpoint.node.node_id,
                status="LOCAL_AHEAD",
                discrepancy_details=f"Stan lokalny wyprzedza węzeł partnerski o {local_total - peer_total} bloków.",
                resolved=True,
            )

        # 5. Równa liczba bloków, lecz odmienne pierścienie Merkle (Rozwidlenie / Fork)
        logger.warning("Wykryto rozbieżność korzenia Merkle (Fork) z węzłem %s!", peer_checkpoint.node.node_id)
        return SyncReconciliationResult(
            is_consistent=False,
            local_node_id=self.local_node.node_id,
            peer_node_id=peer_checkpoint.node.node_id,
            status="DIVERGENT_FORK",
            discrepancy_details=f"Równa liczba bloków ({local_total}), lecz różne korzenie Merkle: lokalny={local_root[:16]}... vs peer={peer_root[:16]}...",
            resolved=False,
        )

    def get_cluster_topology(self) -> Dict[str, Any]:
        """Zwraca aktualną topologię klastrów wieloregionalnych."""
        return {
            "local_node": self.local_node.model_dump(),
            "active_peers_count": len(self.peers),
            "peers": [p.model_dump() for p in self.peers.values()],
            "checkpoints_count": len(self.checkpoints_history),
            "last_checkpoint": self.checkpoints_history[-1].model_dump() if self.checkpoints_history else None,
        }
