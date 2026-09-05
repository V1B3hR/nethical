"""Merkle-DAG Cryptographic Audit Ledger & Post-Quantum Attestation Mesh (Faza 3 Roadmapy).

Zapewnia matematyczną, kryptograficzną niezmienność (Tamper-Evidence) dla każdego
orzeczenia bramy Nethical, dowody inkluzji Merkle'a oraz podpisy postkwantowe
(NIST FIPS 204 ML-DSA / CRYSTALS-Dilithium) wraz z suwerenną pieczęcią Błyskawicy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from nethical.security.quantum_crypto import (
    CRYSTALSDilithium,
    DilithiumKeyPair,
    PQCAlgorithm,
    QuantumSignature,
)

logger = logging.getLogger("nethical.security.merkle_ledger")


def canonical_json_bytes(data: Any) -> bytes:
    """Zwraca znormalizowany ciąg bajtów JSON (RFC 8785) dla powtarzalnego hashowania."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def hash_leaf(data_bytes: bytes) -> str:
    """Hashuje liść drzewa Merkle z separacją domenową (prefiks 0x00)."""
    return hashlib.sha256(b"\x00" + data_bytes).hexdigest()


def hash_internal_node(left_hex: str, right_hex: str) -> str:
    """Hashuje węzeł wewnętrzny drzewa Merkle z separacją domenową (prefiks 0x01)."""
    left_b = bytes.fromhex(left_hex)
    right_b = bytes.fromhex(right_hex)
    return hashlib.sha256(b"\x01" + left_b + right_b).hexdigest()


class MerkleInclusionStep(BaseModel):
    sibling_hash: str
    direction: str  # "left" lub "right"


class TamperProofReceipt(BaseModel):
    """Kryptograficzny kwit orzeczenia z dowodem inkluzji i podpisem postkwantowym."""

    receipt_id: str
    decision_id: str
    chain_index: int
    timestamp: str
    leaf_hash: str
    previous_block_hash: str
    block_hash: str
    merkle_root: str
    inclusion_proof: List[MerkleInclusionStep] = Field(default_factory=list)
    pqc_algorithm: str = "dilithium3"
    signer_key_id: str
    signature_hex: str
    ambassador_sealed: bool = False
    ambassador_seal_hash: Optional[str] = None


class MerkleLedgerBlock(BaseModel):
    """Pojedynczy zapieczętowany blok w łańcuchu Merkle-DAG."""

    chain_index: int
    timestamp: str
    decision_payload: Dict[str, Any]
    leaf_hash: str
    previous_block_hash: str
    block_hash: str
    merkle_root_at_seal: str
    receipt_id: str


class MerkleTree:
    """Binarne drzewo skrótów Merkle'a z obsługą dowodów inkluzji."""

    def __init__(self, leaf_hashes: Optional[List[str]] = None) -> None:
        self.leaves: List[str] = list(leaf_hashes or [])

    def add_leaf(self, leaf_hash: str) -> int:
        """Dodaje liść do drzewa i zwraca jego indeks."""
        self.leaves.append(leaf_hash)
        return len(self.leaves) - 1

    def compute_root(self) -> str:
        """Wylicza skrót główny drzewa (Merkle Root)."""
        if not self.leaves:
            return hashlib.sha256(b"NETHICAL_MERKLE_GENESIS_EMPTY").hexdigest()
        if len(self.leaves) == 1:
            return self.leaves[0]

        current_level = list(self.leaves)
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
                next_level.append(hash_internal_node(left, right))
            current_level = next_level

        return current_level[0]

    def get_inclusion_proof(self, target_index: int) -> List[MerkleInclusionStep]:
        """Generuje ścieżkę dowodu inkluzji O(log N) dla liścia pod wskazanym indeksem."""
        if not (0 <= target_index < len(self.leaves)):
            raise IndexError("Index poza zakresem drzewa Merkle")

        proof: List[MerkleInclusionStep] = []
        current_level = list(self.leaves)
        idx = target_index

        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
                next_level.append(hash_internal_node(left, right))

                if i == idx or i + 1 == idx:
                    if idx % 2 == 0:
                        sibling = right
                        proof.append(MerkleInclusionStep(sibling_hash=sibling, direction="right"))
                    else:
                        sibling = left
                        proof.append(MerkleInclusionStep(sibling_hash=sibling, direction="left"))

            idx = idx // 2
            current_level = next_level

        return proof

    @staticmethod
    def verify_proof(leaf_hash: str, proof: List[MerkleInclusionStep], expected_root: str) -> bool:
        """Weryfikuje matematycznie ścieżkę dowodu Merkle dla danego skrótu liścia."""
        if not proof:
            return leaf_hash == expected_root

        curr = leaf_hash
        for step in proof:
            if step.direction == "left":
                curr = hash_internal_node(step.sibling_hash, curr)
            else:
                curr = hash_internal_node(curr, step.sibling_hash)
        return curr == expected_root


class MerkleLedger:
    """Niezmienny rejestr orzeczeń Merkle-DAG z poświadczeniami postkwantowymi."""

    GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"

    def __init__(self, keypair: Optional[DilithiumKeyPair] = None) -> None:
        self.pqc_dilithium = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_3)
        self.keypair: DilithiumKeyPair = keypair or self.pqc_dilithium.generate_keypair()
        self.tree: MerkleTree = MerkleTree()
        self.blocks: List[MerkleLedgerBlock] = []
        self.receipts: Dict[str, TamperProofReceipt] = {}

    @property
    def current_root(self) -> str:
        """Bieżący pierścień Merkle Root rejestru."""
        return self.tree.compute_root()

    @property
    def total_blocks(self) -> int:
        """Liczba zapieczętowanych orzeczeń w rejestrze."""
        return len(self.blocks)

    def append_decision(
        self,
        decision_data: Dict[str, Any],
        ambassador_notes: Optional[str] = None,
    ) -> TamperProofReceipt:
        """Pieczętuje orzeczenie w rejestrze Merkle-DAG i generuje podpisany kwit audytowy."""
        chain_idx = len(self.blocks)
        now_iso = datetime.now(timezone.utc).isoformat()
        receipt_id = f"RCPT-{uuid.uuid4().hex[:12].upper()}"
        decision_id = decision_data.get("decision_id") or str(uuid.uuid4())

        # 1. Obliczenie skrótu liścia
        payload_bytes = canonical_json_bytes(decision_data)
        leaf_h = hash_leaf(payload_bytes)

        # 2. Rejestracja w drzewie Merkle
        leaf_idx = self.tree.add_leaf(leaf_h)
        merkle_root = self.tree.compute_root()
        inclusion_proof = self.tree.get_inclusion_proof(leaf_idx)

        # 3. Wiązanie blokowe DAG (wsteczny hash)
        prev_hash = self.blocks[-1].block_hash if self.blocks else self.GENESIS_HASH
        block_header = {
            "chain_index": chain_idx,
            "timestamp": now_iso,
            "previous_block_hash": prev_hash,
            "leaf_hash": leaf_h,
            "merkle_root": merkle_root,
        }
        block_h = hashlib.sha256(canonical_json_bytes(block_header)).hexdigest()

        # 4. Pieczęć Błyskawicy (jeśli była konsultacja)
        ambassador_sealed = bool(ambassador_notes)
        ambassador_seal_h = None
        if ambassador_sealed and ambassador_notes:
            ambassador_seal_h = hashlib.sha256(
                f"BLY_AMBASSADOR_SEAL::{ambassador_notes}".encode("utf-8")
            ).hexdigest()

        # 5. Podpis postkwantowy (NIST Level 3 ML-DSA / Dilithium)
        msg_to_sign = f"{block_h}:{merkle_root}:{leaf_h}:{receipt_id}".encode("utf-8")
        quantum_sig: QuantumSignature = self.pqc_dilithium.sign(
            message=msg_to_sign,
            private_key=self.keypair.private_key,
            key_id=self.keypair.key_id,
        )

        receipt = TamperProofReceipt(
            receipt_id=receipt_id,
            decision_id=decision_id,
            chain_index=chain_idx,
            timestamp=now_iso,
            leaf_hash=leaf_h,
            previous_block_hash=prev_hash,
            block_hash=block_h,
            merkle_root=merkle_root,
            inclusion_proof=inclusion_proof,
            pqc_algorithm=self.pqc_dilithium.algorithm.value,
            signer_key_id=self.keypair.key_id,
            signature_hex=quantum_sig.signature.hex(),
            ambassador_sealed=ambassador_sealed,
            ambassador_seal_hash=ambassador_seal_h,
        )

        block = MerkleLedgerBlock(
            chain_index=chain_idx,
            timestamp=now_iso,
            decision_payload=decision_data,
            leaf_hash=leaf_h,
            previous_block_hash=prev_hash,
            block_hash=block_h,
            merkle_root_at_seal=merkle_root,
            receipt_id=receipt_id,
        )

        self.blocks.append(block)
        self.receipts[receipt_id] = receipt

        logger.info(
            "Zapieczętowany blok Merkle-DAG #%d (Receipt: %s, Root: %s...)",
            chain_idx,
            receipt_id,
            merkle_root[:12],
        )
        return receipt

    def get_receipt(self, receipt_id: str) -> Optional[TamperProofReceipt]:
        """Pobiera kwit audytowy o wskazanym identyfikatorze."""
        return self.receipts.get(receipt_id)

    def verify_receipt(self, receipt: TamperProofReceipt) -> bool:
        """Weryfikuje matematyczną poprawność kwitu audytowego i dowodu inkluzji."""
        # 1. Weryfikacja dowodu inkluzji Merkle
        if not MerkleTree.verify_proof(receipt.leaf_hash, receipt.inclusion_proof, receipt.merkle_root):
            logger.warning("Błąd weryfikacji dowodu Merkle dla kwitu %s", receipt.receipt_id)
            return False

        # 2. Weryfikacja spójności skrótu bloku
        expected_header = {
            "chain_index": receipt.chain_index,
            "timestamp": receipt.timestamp,
            "previous_block_hash": receipt.previous_block_hash,
            "leaf_hash": receipt.leaf_hash,
            "merkle_root": receipt.merkle_root,
        }
        recomputed_block_h = hashlib.sha256(canonical_json_bytes(expected_header)).hexdigest()
        if recomputed_block_h != receipt.block_hash:
            logger.warning("Niezgodność skrótu bloku w kwicie %s", receipt.receipt_id)
            return False

        # 3. Weryfikacja sygnatury postkwantowej
        msg_signed = f"{receipt.block_hash}:{receipt.merkle_root}:{receipt.leaf_hash}:{receipt.receipt_id}".encode("utf-8")
        raw_sig = bytes.fromhex(receipt.signature_hex)
        q_sig = QuantumSignature(
            signature=raw_sig,
            message_hash=hashlib.sha256(msg_signed).hexdigest(),
            algorithm=PQCAlgorithm.DILITHIUM_3,
            signer_key_id=receipt.signer_key_id,
        )
        is_valid = self.pqc_dilithium.verify(
            message=msg_signed,
            signature=q_sig,
            public_key=self.keypair.public_key,
        )
        return is_valid

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Weryfikuje integralność całego łańcucha bloków od bloku genezy do końca."""
        errors: List[str] = []
        if not self.blocks:
            return True, []

        prev_h = self.GENESIS_HASH
        recomputed_tree = MerkleTree()

        for idx, block in enumerate(self.blocks):
            # A. Sprawdzenie indeksu
            if block.chain_index != idx:
                errors.append(f"Blok #{idx}: niezgodność indeksu ({block.chain_index} != {idx})")

            # B. Sprawdzenie ciągłości wstecznego skrótu
            if block.previous_block_hash != prev_h:
                errors.append(f"Blok #{idx}: uszkodzony wsteczny hash DAG ({block.previous_block_hash} != {prev_h})")

            # C. Sprawdzenie skrótu liścia
            actual_leaf_h = hash_leaf(canonical_json_bytes(block.decision_payload))
            if actual_leaf_h != block.leaf_hash:
                errors.append(f"Blok #{idx}: manipulacja zawartością orzeczenia!")

            # D. Sprawdzenie rekalkulacji skrótu bloku
            header = {
                "chain_index": block.chain_index,
                "timestamp": block.timestamp,
                "previous_block_hash": block.previous_block_hash,
                "leaf_hash": block.leaf_hash,
                "merkle_root": block.merkle_root_at_seal,
            }
            if hashlib.sha256(canonical_json_bytes(header)).hexdigest() != block.block_hash:
                errors.append(f"Blok #{idx}: nieprawidłowy skrót nagłówka bloku!")

            recomputed_tree.add_leaf(actual_leaf_h)
            prev_h = block.block_hash

        is_valid = len(errors) == 0
        return is_valid, errors

    def export_verifiable_bundle(self, limit: int = 100) -> Dict[str, Any]:
        """Eksportuje paczkę audytową z kompletem dowodów dla Jednostek Notyfikowanych UE i UK AISI."""
        selected_blocks = self.blocks[-limit:] if self.blocks else []
        is_integral, errors = self.verify_integrity()

        return {
            "format": "nethical_cryptographic_audit_bundle_v3",
            "pqc_standard": "NIST FIPS 204 (ML-DSA / Dilithium Level 3)",
            "signer_key_id": self.keypair.key_id,
            "public_key_hex": self.keypair.public_key.hex(),
            "merkle_root": self.current_root,
            "total_blocks_in_chain": len(self.blocks),
            "bundle_blocks_count": len(selected_blocks),
            "chain_integrity_valid": is_integral,
            "integrity_errors": errors,
            "blocks": [b.model_dump() for b in selected_blocks],
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
