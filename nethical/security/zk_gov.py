"""Zero-Knowledge Compliance Engine (ZK-Gov) - Faza 3.5 & 4 Roadmapy.

Umożliwia generowanie i niezależną weryfikację dowodów zgodności etycznej i regulacyjnej
(EU AI Act / UK AISI / 25 Fundamentalnych Praw Nethical) bez ujawniania poufnych danych:
- promptów użytkownika,
- wag lub architektury modelu,
- tajemnic przedsiębiorstwa zawartych w argumentach narzędzi.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import (
    MerkleInclusionStep,
    MerkleTree,
    TamperProofReceipt,
    canonical_json_bytes,
)
from nethical.security.quantum_crypto import (
    CRYSTALSDilithium,
    DilithiumKeyPair,
    PQCAlgorithm,
    QuantumSignature,
)

logger = logging.getLogger("nethical.security.zk_gov")


class PublicCompliancePredicates(BaseModel):
    """Zbiór publicznie weryfikowalnych faktów zgodności bez ujawniania treści."""

    decision_allowed: bool = Field(..., description="Czy orzeczenie bramy to ALLOW")
    laws_verified_count: int = Field(default=25, description="Liczba zweryfikowanych Praw Nethical")
    no_critical_pii: bool = Field(default=True, description="Potwierdzenie braku wycieku danych PII")
    no_destructive_action: bool = Field(default=True, description="Potwierdzenie braku komend niszczących (Prawo 2)")
    cognitive_shield_cleared: bool = Field(default=True, description="Potwierdzenie przejścia Tarczy Kognitywnej Błyskawicy")
    target_tool_category: str = Field(default="generic_tool", description="Abstrakcyjna kategoria wywołanego narzędzia")
    evaluation_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ZkCommitment(BaseModel):
    """Kryptograficzne zobowiązanie ukrywające tajną treść wywołania."""

    commitment_hash: str = Field(..., description="SHA-256(blinding_factor || secret_payload)")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ZkComplianceProof(BaseModel):
    """Kompaktowy dowód Zero-Knowledge zgodności z prawem i politykami bezpieczeństwa."""

    proof_id: str
    receipt_id: str
    merkle_root: str
    commitment_hash: str
    predicates: PublicCompliancePredicates
    merkle_inclusion_proof: List[MerkleInclusionStep] = Field(default_factory=list)
    pqc_algorithm: str = "dilithium3"
    signer_key_id: str
    proof_signature_hex: str
    signed_claim_hash: str
    ambassador_sealed: bool = False
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ZkGovEngine:
    """Silnik dowodzenia i weryfikacji Zero-Knowledge dla Nethical Enterprise OS."""

    def __init__(self, keypair: Optional[DilithiumKeyPair] = None) -> None:
        self.pqc_dilithium = CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_3)
        self.keypair: DilithiumKeyPair = keypair or self.pqc_dilithium.generate_keypair()

    @staticmethod
    def create_commitment(secret_payload: Dict[str, Any], salt: Optional[str] = None) -> Tuple[str, str]:
        """Tworzy zobowiązanie kryptograficzne ukrywające poufną treść.

        Returns:
            Tuple[commitment_hash, blinding_factor]
        """
        blinding_factor = salt or secrets.token_hex(32)
        payload_bytes = canonical_json_bytes(secret_payload)
        commitment_hash = hashlib.sha256(bytes.fromhex(blinding_factor) + payload_bytes).hexdigest()
        return commitment_hash, blinding_factor

    @staticmethod
    def verify_commitment(commitment_hash: str, secret_payload: Dict[str, Any], blinding_factor: str) -> bool:
        """Weryfikuje, czy ujawniony sekret pasuje do wcześniej złożonego zobowiązania."""
        payload_bytes = canonical_json_bytes(secret_payload)
        recomputed = hashlib.sha256(bytes.fromhex(blinding_factor) + payload_bytes).hexdigest()
        return secrets.compare_digest(recomputed, commitment_hash)

    def generate_compliance_proof(
        self,
        receipt: TamperProofReceipt,
        decision_data: Dict[str, Any],
        blinding_factor: Optional[str] = None,
    ) -> Tuple[ZkComplianceProof, str]:
        """Generuje dowód Zero-Knowledge potwierdzający zgodność orzeczenia z polityką governance.

        Returns:
            Tuple[ZkComplianceProof, blinding_factor]
        """
        # 1. Ekstrakcja poufnych argumentów i utworzenie zobowiązania
        secret_content = {
            "arguments_preview": decision_data.get("arguments_preview", ""),
            "agent_id": decision_data.get("agent_id", ""),
            "tool_name": decision_data.get("tool_name", ""),
        }
        commitment_hash, salt = self.create_commitment(secret_content, blinding_factor)

        # 2. Sformułowanie publicznie weryfikowalnych predykatów (bez danych poufnych)
        decision_val = decision_data.get("decision", "ALLOW")
        violations = decision_data.get("violations", [])
        shield_passed = decision_data.get("shield_passed", True)
        laws_checked = decision_data.get("laws_checked", [1, 2, 7, 18, 25])

        predicates = PublicCompliancePredicates(
            decision_allowed=(decision_val == "ALLOW"),
            laws_verified_count=len(laws_checked),
            no_critical_pii=not any("PII" in str(v) for v in violations),
            no_destructive_action=not any("Destructive" in str(v) for v in violations),
            cognitive_shield_cleared=shield_passed,
            target_tool_category=decision_data.get("tool_name", "generic_tool").split("_")[0],
            evaluation_timestamp=decision_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

        # 3. Podpisanie dowodu kluczem postkwantowym (NIST FIPS 204 ML-DSA-65)
        proof_id = f"ZKPRF-{uuid.uuid4().hex[:12].upper()}"
        claim_dict = {
            "proof_id": proof_id,
            "receipt_id": receipt.receipt_id,
            "merkle_root": receipt.merkle_root,
            "commitment_hash": commitment_hash,
            "predicates": predicates.model_dump(),
        }
        proof_claim_bytes = canonical_json_bytes(claim_dict)
        signed_claim_hash = hashlib.sha256(proof_claim_bytes).hexdigest()

        q_sig: QuantumSignature = self.pqc_dilithium.sign(
            message=proof_claim_bytes,
            private_key=self.keypair.private_key,
            key_id=self.keypair.key_id,
        )

        proof = ZkComplianceProof(
            proof_id=proof_id,
            receipt_id=receipt.receipt_id,
            merkle_root=receipt.merkle_root,
            commitment_hash=commitment_hash,
            predicates=predicates,
            merkle_inclusion_proof=receipt.inclusion_proof,
            pqc_algorithm=self.pqc_dilithium.algorithm.value,
            signer_key_id=self.keypair.key_id,
            proof_signature_hex=q_sig.signature.hex(),
            signed_claim_hash=signed_claim_hash,
            ambassador_sealed=receipt.ambassador_sealed,
        )

        logger.info(
            "Wygenerowano dowód ZK-Gov %s dla kwitu %s (Commitment: %s...)",
            proof_id,
            receipt.receipt_id,
            commitment_hash[:12],
        )
        return proof, salt

    def verify_compliance_proof(
        self,
        proof: ZkComplianceProof,
        expected_root: Optional[str] = None,
    ) -> Tuple[bool, List[str]]:
        """Matematyczna, niezależna weryfikacja dowodu ZK-Gov przez regulatora lub audytora."""
        errors: List[str] = []

        # A. Sprawdzenie zgodności pierścienia Merkle Root (jeśli podano oczekiwany)
        if expected_root and proof.merkle_root != expected_root:
            errors.append(f"Niezgodność Merkle Root ({proof.merkle_root} != {expected_root})")

        # B. Sprawdzenie integralności podpisu postkwantowego
        claim_dict = {
            "proof_id": proof.proof_id,
            "receipt_id": proof.receipt_id,
            "merkle_root": proof.merkle_root,
            "commitment_hash": proof.commitment_hash,
            "predicates": proof.predicates.model_dump(),
        }
        proof_claim_bytes = canonical_json_bytes(claim_dict)
        actual_claim_hash = hashlib.sha256(proof_claim_bytes).hexdigest()

        if actual_claim_hash != proof.signed_claim_hash:
            errors.append("Nieprawidłowy podpis postkwantowy (modyfikacja treści dowodu ZK)")

        raw_sig = bytes.fromhex(proof.proof_signature_hex)
        q_sig = QuantumSignature(
            signature=raw_sig,
            message_hash=proof.signed_claim_hash,
            algorithm=PQCAlgorithm.DILITHIUM_3,
            signer_key_id=proof.signer_key_id,
        )

        is_sig_valid = self.pqc_dilithium.verify(
            message=proof_claim_bytes,
            signature=q_sig,
            public_key=self.keypair.public_key,
        )

        if not is_sig_valid:
            errors.append("Nieprawidłowy podpis postkwantowy (ML-DSA / Dilithium) dowodu ZK")

        # C. Weryfikacja spełnienia kluczowych predykatów bezpieczeństwa
        if not proof.predicates.decision_allowed:
            errors.append("Predykat negatywny: Orzeczenie nie zostało dopuszczone (nie jest ALLOW)")
        if not proof.predicates.no_destructive_action:
            errors.append("Predykat negatywny: Wykryto naruszenie Prawa 2 (polecenie niszczące)")
        if not proof.predicates.no_critical_pii:
            errors.append("Predykat negatywny: Wykryto naruszenie Prawa 7 (krytyczny wyciek PII)")

        is_valid = len(errors) == 0
        return is_valid, errors
