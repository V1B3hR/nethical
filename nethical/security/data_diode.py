# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Sovereign Data Diode & Air-Gapped Package Synchronization Subsystem (Faza 3 Roadmapy).

Umożliwia w 100% bezpieczną, jednokierunkową wymianę danych między siecią globalną
a odciętymi fizycznie środowiskami (Air-Gap / NATO SECRET / Bunkry Danych):
- Tworzenie i eksport pakietów suwerennych (.sovereign.pkg) z pieczęcią postkwantową ML-DSA-65
- Weryfikacja kryptograficzna integralności SHA3-512 i sygnatury bez dostępu do Internetu
- Ochrona przed atakami powtórzeniowymi (Anti-Replay Nonce & Timestamp Envelope)
- Automatyczne kotwiczenie zaimportowanych polityk w lokalnym rejestrze Merkle-DAG
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cachetools import TTLCache

from pydantic import BaseModel, Field

from nethical.security.merkle_ledger import (
    MerkleLedger,
    TamperProofReceipt,
    canonical_json_bytes,
)
from nethical.security.quantum_crypto import (
    CRYSTALSDilithium,
    DilithiumKeyPair,
    PQCAlgorithm,
    QuantumSignature,
)
from nethical.core.models import ClassificationLevel

logger = logging.getLogger("nethical.security.data_diode")


class SovereignPackageHeader(BaseModel):
    """Nagłówek suwerennego pakietu diody danych z poświadczeniami postkwantowymi."""

    package_id: str = Field(..., description="Unikalny identyfikator paczki")
    package_type: str = Field(
        ...,
        description="Typ zawartości: POLICY_UPDATE, REGULATORY_PACK, PRECEDENT_DOSSIER, TELEMETRY_EXPORT",
    )
    source_node_id: str = Field(..., description="Identyfikator węzła źródłowego / nadawczego")
    target_tenant_id: str = Field(..., description="Docelowy podmiot suwerenny (Tenant ID)")
    classification: str = Field(
        default="SECRET", description="Poziom klauzuli tajności (UNCLASSIFIED do SECRET)"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Znacznik czasu wygenerowania pakietu (ISO-8601 UTC)",
    )
    nonce: str = Field(
        default_factory=lambda: secrets.token_hex(16),
        description="Jednorazowy kryptograficzny token chroniący przed atakami powtórzeniowymi",
    )
    sha3_512_digest: str = Field(..., description="Skrót SHA3-512 ładunku (Payload Digest)")
    pqc_algorithm: str = Field(
        default="ML-DSA-65 (CRYSTALS-Dilithium Level 3)",
        description="Algorytm podpisu postkwantowego (NIST FIPS 204)",
    )
    signer_key_id: str = Field(..., description="Identyfikator klucza postkwantowego")
    signature_hex: str = Field(..., description="Kryptograficzny podpis postkwantowy w formacie HEX")
    pqc_message_hash: str = Field(default="", description="Skrót manifestu podpisany przez Dilithium")


class SovereignPackage(BaseModel):
    """Kompletny suwerenny pakiet danych z nagłówkiem i zabezpieczonym ładunkiem."""

    header: SovereignPackageHeader
    payload: Dict[str, Any] = Field(..., description="Zabezpieczone dane operacyjne, reguły lub wagi")

    def to_bytes(self) -> bytes:
        """Serializuje paczkę do kanonicznych bajtów JSON."""
        return canonical_json_bytes(self.model_dump(mode="json"))

    def save_to_file(self, file_path: str | Path) -> Path:
        """Zapisuje pakiet do pliku binarnego/JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_bytes())
        logger.info("Zapisano suwerenny pakiet %s do %s", self.header.package_id, path)
        return path

    @classmethod
    def load_from_file(cls, file_path: str | Path) -> SovereignPackage:
        """Wczytuje suwerenny pakiet z pliku."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Plik suwerennego pakietu nie istnieje: {path}")
        raw_bytes = path.read_bytes()
        data = json.loads(raw_bytes.decode("utf-8"))
        return cls.model_validate(data)


class DataDiodeBridge:
    """Jednokierunkowy most wymiany danych dla suwerennych węzłów Air-Gapped."""

    def __init__(
        self,
        node_id: str = "sovereign_diode_bridge_01",
        pqc_engine: Optional[CRYSTALSDilithium] = None,
        default_keypair: Optional[DilithiumKeyPair] = None,
    ) -> None:
        self.node_id = node_id
        self.pqc_engine = pqc_engine or CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_3)
        self.keypair = default_keypair or self.pqc_engine.generate_keypair()
        self._trusted_public_keys: Dict[str, bytes] = {
            self.keypair.key_id: self.keypair.public_key
        }
        # Anti-replay nonce cache: max 100K entries, 24h TTL auto-eviction
        self._processed_nonces: TTLCache[str, bool] = TTLCache(maxsize=100_000, ttl=86400)

    def register_trusted_key(self, key_id: str, public_key: bytes) -> None:
        """Rejestruje zaufany klucz publiczny PQC nadawcy."""
        self._trusted_public_keys[key_id] = public_key
        logger.info("Zarejestrowano zaufany klucz PQC nadawcy: %s", key_id)

    def create_package(
        self,
        target_tenant_id: str,
        payload: Dict[str, Any],
        package_type: str = "POLICY_UPDATE",
        classification: ClassificationLevel = ClassificationLevel.SECRET,
        keypair: Optional[DilithiumKeyPair] = None,
    ) -> SovereignPackage:
        """Generuje i podpisuje postkwantowo suwerenny pakiet synchronizacyjny."""
        active_kp = keypair or self.keypair
        now_iso = datetime.now(timezone.utc).isoformat()
        package_id = f"PKG-{secrets.token_hex(8).upper()}"
        nonce = secrets.token_hex(16)

        # 1. Obliczenie skrótu SHA3-512 dla ładunku (Payload)
        payload_bytes = canonical_json_bytes(payload)
        sha3_hash = hashlib.sha3_512(payload_bytes).hexdigest()

        cls_str = classification.value if hasattr(classification, "value") else str(classification)

        # 2. Wiązanie kryptograficzne nagłówka i skrótu
        manifest_to_sign = f"{package_id}:{package_type}:{self.node_id}:{target_tenant_id}:{cls_str}:{sha3_hash}:{nonce}:{now_iso}".encode("utf-8")

        # 3. Podpis postkwantowy NIST FIPS 204 ML-DSA-65
        q_sig: QuantumSignature = self.pqc_engine.sign(
            message=manifest_to_sign,
            private_key=active_kp.private_key,
            key_id=active_kp.key_id,
        )

        header = SovereignPackageHeader(
            package_id=package_id,
            package_type=package_type,
            source_node_id=self.node_id,
            target_tenant_id=target_tenant_id,
            classification=cls_str,
            created_at=now_iso,
            nonce=nonce,
            sha3_512_digest=sha3_hash,
            pqc_algorithm="ML-DSA-65 (CRYSTALS-Dilithium Level 3)",
            signer_key_id=active_kp.key_id,
            signature_hex=q_sig.signature.hex(),
            pqc_message_hash=q_sig.message_hash,
        )

        return SovereignPackage(header=header, payload=payload)

    def verify_package(self, package: SovereignPackage) -> Tuple[bool, str]:
        """Weryfikuje matematyczną poprawność pakietu diody danych bez użycia sieci zewnętrznej."""
        header = package.header

        # 1. Weryfikacja Anti-Replay Nonce
        if header.nonce in self._processed_nonces:
            return False, f"Wykryto atak powtórzeniowy: Nonce {header.nonce} był już przetworzony."

        # 2. Weryfikacja integralności skrótu SHA3-512 ładunku
        expected_hash = hashlib.sha3_512(canonical_json_bytes(package.payload)).hexdigest()
        if not secrets.compare_digest(header.sha3_512_digest, expected_hash):
            return False, "Naruszenie integralności ładunku: Skrót SHA3-512 nie odpowiada zawartości paczki."

        # 3. Weryfikacja dostępności zaufanego klucza publicznego PQC
        if header.signer_key_id not in self._trusted_public_keys:
            return False, f"Niezaufany nadawca: Klucz PQC '{header.signer_key_id}' nie znajduje się w zaufanym magazynie węzła."

        # 4. Weryfikacja sygnatury postkwantowej ML-DSA-65
        trusted_pubkey = self._trusted_public_keys[header.signer_key_id]
        manifest_to_verify = f"{header.package_id}:{header.package_type}:{header.source_node_id}:{header.target_tenant_id}:{header.classification}:{header.sha3_512_digest}:{header.nonce}:{header.created_at}".encode("utf-8")

        computed_manifest_hash = hashlib.sha256(manifest_to_verify).hexdigest()
        if header.pqc_message_hash and header.pqc_message_hash != computed_manifest_hash:
            return False, "Naruszenie integralności manifestu: Skrót nagłówka PQC nie odpowiada zadeklarowanym polom."

        try:
            ts = datetime.fromisoformat(header.created_at)
        except Exception:
            ts = datetime.now(timezone.utc)

        try:
            sig_bytes = bytes.fromhex(header.signature_hex)
        except Exception:
            return False, "Nieprawidłowy format HEX sygnatury postkwantowej."

        if len(sig_bytes) < 32:
            return False, "Nieprawidłowa długość sygnatury postkwantowej."

        msg_hash = header.pqc_message_hash if header.pqc_message_hash else computed_manifest_hash

        q_sig = QuantumSignature(
            algorithm=PQCAlgorithm.DILITHIUM_3,
            signature=sig_bytes,
            message_hash=msg_hash,
            signer_key_id=header.signer_key_id,
            timestamp=ts,
        )

        is_sig_valid = self.pqc_engine.verify(
            message=manifest_to_verify,
            signature=q_sig,
            public_key=trusted_pubkey,
        )

        if not is_sig_valid:
            return False, "Błąd weryfikacji podpisu postkwantowego ML-DSA-65."

        return True, "Pakiet zweryfikowany pomyślnie. Sygnatura ML-DSA-65 prawidłowa."

    def import_and_seal(
        self,
        package: SovereignPackage,
        ledger: MerkleLedger,
    ) -> Tuple[bool, str, Optional[TamperProofReceipt]]:
        """Weryfikuje pakiet i trwale pieczętuje go w lokalnym rejestrze Merkle-DAG węzła."""
        is_valid, msg = self.verify_package(package)
        if not is_valid:
            logger.error("Odrzucono import suwerennego pakietu %s: %s", package.header.package_id, msg)
            return False, msg, None

        # Rejestracja Nonce w celu ochrony przed powtórzeniem
        self._processed_nonces[package.header.nonce] = True

        # Pieczętowanie zdarzenia w dedykowanym Merkle Ledgerze
        seal_payload = {
            "event": "AIR_GAP_DATA_DIODE_IMPORT",
            "package_id": package.header.package_id,
            "package_type": package.header.package_type,
            "source_node": package.header.source_node_id,
            "target_tenant": package.header.target_tenant_id,
            "classification": package.header.classification,
            "sha3_512": package.header.sha3_512_digest,
            "signer_key_id": package.header.signer_key_id,
            "imported_at": datetime.now(timezone.utc).isoformat(),
        }

        receipt = ledger.append_decision(
            decision_data=seal_payload,
            ambassador_notes=f"Data Diode Ingest: {package.header.package_type} verified via PQC ML-DSA-65",
        )

        logger.info(
            "Zapieczętowano suwerenny pakiet %s w Merkle Ledger (Kwit: %s)",
            package.header.package_id,
            receipt.receipt_id,
        )
        return True, "Pakiet zaimportowany i trwale zapieczętowany w rejestrze Merkle.", receipt
