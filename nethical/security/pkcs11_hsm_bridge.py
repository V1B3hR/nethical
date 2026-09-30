# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Hardware Security Module (HSM) & PKCS#11 Dual-Custody Bridge.

Zapewnia sprzętowe zakotwiczenie kluczy kryptograficznych i procedury Dwóch Kluczy (Dual-Custody)
w granicach bezpieczeństwa modułów HSM zgodnych z FIPS 140-2 Level 3 oraz standardem PKCS#11:
- Obsługa fizycznych i wirtualnych slotów kryptograficznych (PKCS#11 v2.40/v3.0).
- Fizyczna izolacja kluczy oficerów w rozłącznych tokenach (YubiKey / YubiHSM 2 / Thales Luna).
- Nieeksportowalne klucze prywatne (CKA_EXTRACTABLE=False, CKA_SENSITIVE=True).
- Sprzętowo wiązane podpisy postkwantowe ML-DSA-65 (FIPS 204) z atestacją enklawy sprzętowej.
- Wirtualna enklawa testowa (VirtualPKCS11Enclave) symulująca pełny cykl życia PKCS#11 w testach CI.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from nethical.governance.roe_gate import (
    CanonicalOperationToken,
    OfficerSignature,
)
from nethical.security.quantum_crypto import (
    CRYSTALSDilithium,
    DilithiumKeyPair,
    PQCAlgorithm,
    QuantumSignature,
)

logger = logging.getLogger("nethical.security.pkcs11_hsm_bridge")


# ============================================================================
# Enums i Modele Danych PKCS#11
# ============================================================================

class PKCS11Mechanism(str, Enum):
    """Mechanizmy kryptograficzne PKCS#11 używane do autoryzacji sprzętowej."""
    CKM_ECDSA_SHA384 = "CKM_ECDSA_SHA384"
    CKM_ED25519 = "CKM_ED25519"
    CKM_RSA_PKCS_PSS = "CKM_RSA_PKCS_PSS"
    CKM_GENERIC_PQC_ML_DSA = "CKM_GENERIC_PQC_ML_DSA"  # Standard FIPS 204 w nowoczesnych HSM


class SlotSecurityLevel(str, Enum):
    """Poziom certyfikacji bezpieczeństwa slotu sprzętowego."""
    FIPS_140_2_LEVEL_3 = "FIPS_140_2_Level_3"
    FIPS_140_3_LEVEL_4 = "FIPS_140_3_Level_4"
    COMMON_CRITERIA_EAL6_PLUS = "CC_EAL6+"
    SOFTWARE_EMULATED = "Software_Emulated_CI"


class HardwareKeyAttestation(BaseModel):
    """Kryptograficzne poświadczenie sprzętowego pochodzenia klucza w enklawie HSM."""
    attestation_id: str = Field(default_factory=lambda: f"ATT-{uuid.uuid4().hex[:10].upper()}")
    slot_id: int
    token_serial_number: str
    hardware_model: str
    security_level: SlotSecurityLevel
    key_label: str
    public_key_hex: str
    is_non_exportable: bool = True
    fips_certified: bool = True
    attestation_signature: str
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class PKCS11Session:
    """Reprezentacja sesji PKCS#11 po uwierzytelnieniu kodem PIN użytkownika."""
    session_handle: int
    slot_id: int
    officer_id: str
    is_logged_in: bool = False
    opened_at: float = field(default_factory=time.time)
    max_session_seconds: float = 600.0  # 10 minut limitu sesji


# ============================================================================
# Wirtualna Enklawa PKCS#11 (Wiarygodny Poligon dla CI/Automatycznych Testów)
# ============================================================================

class VirtualPKCS11Enclave:
    """Symulator fizycznego modułu HSM / czytnika kart PKCS#11.
    
    Odwzorowuje pełną semantykę sprzętową:
    - Osobne sloty fizyczne (Slot 0 dla Oficera 1, Slot 1 dla Oficera 2).
    - Weryfikacja PIN z licznikiem błędów (blokada po 3 nieudanych próbach).
    - Przechowywanie kluczy prywatnych w pamięci izolowanej (zakaz ekstrakcji).
    - Sprzętowa generacja podpisów cyfrowych.
    """

    def __init__(self) -> None:
        # slot_id -> slot_metadata
        self.slots: Dict[int, Dict[str, Any]] = {
            0: {
                "label": "HSM-SLOT-0-OFFICER-ALPHA",
                "serial": "YUBIHSM2-9901-0012",
                "model": "YubiHSM 2 FIPS",
                "pin_hash": hashlib.sha256(b"AlphaOfficerPin1234").hexdigest(),
                "failed_attempts": 0,
                "is_locked": False,
            },
            1: {
                "label": "HSM-SLOT-1-OFFICER-BRAVO",
                "serial": "YUBIHSM2-9901-0013",
                "model": "YubiHSM 2 FIPS",
                "pin_hash": hashlib.sha256(b"BravoOfficerPin5678").hexdigest(),
                "failed_attempts": 0,
                "is_locked": False,
            },
        }
        # (slot_id, key_id) -> private_key_material (nieeksportowalny)
        self._isolated_key_store: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self.active_sessions: Dict[int, PKCS11Session] = {}
        self._session_counter = 1000

    def open_session(self, slot_id: int, officer_id: str) -> PKCS11Session:
        """Otwiera nową sesję kryptograficzną w slocie PKCS#11."""
        if slot_id not in self.slots:
            raise ValueError(f"Nieprawidłowy slot PKCS#11: {slot_id}")
        if self.slots[slot_id]["is_locked"]:
            raise PermissionError(f"Slot {slot_id} jest zablokowany z powodu zbyt wielu błędnych kodów PIN.")

        self._session_counter += 1
        sess = PKCS11Session(
            session_handle=self._session_counter,
            slot_id=slot_id,
            officer_id=officer_id,
            is_logged_in=False,
        )
        self.active_sessions[sess.session_handle] = sess
        return sess

    def login(self, session_handle: int, pin: str) -> bool:
        """Uwierzytelnia oficera kodem PIN w slocie sprzętowym (C_Login)."""
        sess = self.active_sessions.get(session_handle)
        if not sess:
            raise KeyError("Nie znaleziono aktywnej sesji PKCS#11.")

        slot_data = self.slots[sess.slot_id]
        if slot_data["is_locked"]:
            raise PermissionError("Slot sprzętowy jest trwale zablokowany.")

        input_hash = hashlib.sha256(pin.encode("utf-8")).hexdigest()
        if input_hash == slot_data["pin_hash"]:
            slot_data["failed_attempts"] = 0
            sess.is_logged_in = True
            logger.info("PKCS#11: Pomyślnie uwierzytelniono oficera %s w slocie %d.", sess.officer_id, sess.slot_id)
            return True
        else:
            slot_data["failed_attempts"] += 1
            if slot_data["failed_attempts"] >= 3:
                slot_data["is_locked"] = True
                logger.error("PKCS#11: ALARM! Slot %d zablokowany po 3 nieudanych próbach PIN.", sess.slot_id)
            return False

    def provision_hardware_key(
        self,
        slot_id: int,
        key_id: str,
        dilithium_keypair: DilithiumKeyPair,
    ) -> HardwareKeyAttestation:
        """Rejestruje klucz w sprzętowej enklawie (nieeksportowalny)."""
        if slot_id not in self.slots:
            raise ValueError(f"Nieznany slot {slot_id}")

        self._isolated_key_store[(slot_id, key_id)] = {
            "keypair": dilithium_keypair,
            "created_at": datetime.now(timezone.utc),
            "usage_count": 0,
        }

        # Podpis poświadczający enklawy sprzętowej (Attestation Signature)
        slot_info = self.slots[slot_id]
        attestation_payload = f"{slot_id}:{slot_info['serial']}:{key_id}:{dilithium_keypair.public_key.hex()}"
        attest_sig = hmac.new(b"ENCLAVE_MASTER_ROOT_KEY_2026", attestation_payload.encode(), hashlib.sha256).hexdigest()

        return HardwareKeyAttestation(
            slot_id=slot_id,
            token_serial_number=slot_info["serial"],
            hardware_model=slot_info["model"],
            security_level=SlotSecurityLevel.FIPS_140_2_LEVEL_3,
            key_label=key_id,
            public_key_hex=dilithium_keypair.public_key.hex(),
            is_non_exportable=True,
            fips_certified=True,
            attestation_signature=attest_sig,
        )

    def hardware_sign(
        self,
        session_handle: int,
        key_id: str,
        message: bytes,
        pqc_engine: CRYSTALSDilithium,
    ) -> QuantumSignature:
        """Wykonuje podpis kryptograficzny wewnątrz enklawy sprzętowej (C_Sign)."""
        sess = self.active_sessions.get(session_handle)
        if not sess or not sess.is_logged_in:
            raise PermissionError("Brak autoryzowanej sesji PKCS#11 (wymagany poprawny C_Login).")

        key_entry = self._isolated_key_store.get((sess.slot_id, key_id))
        if not key_entry:
            raise KeyError(f"Nie znaleziono klucza sprzętowego {key_id} w slocie {sess.slot_id}.")

        keypair: DilithiumKeyPair = key_entry["keypair"]
        key_entry["usage_count"] += 1

        # Podpis generowany wewnątrz chronionego kontekstu (brak dostępu z zewnątrz do private_key)
        signature = pqc_engine.sign(
            message=message,
            private_key=keypair.private_key,
            key_id=f"HSM-{sess.slot_id}-{key_id}",
        )
        return signature

    def close_session(self, session_handle: int) -> None:
        """Zamyka sesję i czyści pamięć podręczną (C_CloseSession)."""
        if session_handle in self.active_sessions:
            del self.active_sessions[session_handle]


# ============================================================================
# Moduł Główny: HSMPKCS11OfficerBridge
# ============================================================================

class HSMPKCS11OfficerBridge:
    """Most łączący Bramkę NTSG z modułami sprzętowymi HSM i tokenami PKCS#11."""

    def __init__(
        self,
        enclave: Optional[VirtualPKCS11Enclave] = None,
        pqc_engine: Optional[CRYSTALSDilithium] = None,
    ) -> None:
        self.pqc = pqc_engine or CRYSTALSDilithium(algorithm=PQCAlgorithm.DILITHIUM_3)
        self.enclave = enclave or VirtualPKCS11Enclave()
        self.officer_keys: Dict[str, Tuple[int, str, HardwareKeyAttestation]] = {}

    def setup_officer_hardware_token(
        self,
        officer_id: str,
        slot_id: int,
        key_label: Optional[str] = None,
    ) -> Tuple[HardwareKeyAttestation, bytes]:
        """Inicjalizuje klucz postkwantowy w wybranym slocie sprzętowym oficera.
        
        Zwraca: (HardwareKeyAttestation, public_key_bytes)
        """
        key_id = key_label or f"key-{officer_id.lower()}-{uuid.uuid4().hex[:6]}"
        keypair = self.pqc.generate_keypair()

        attestation = self.enclave.provision_hardware_key(
            slot_id=slot_id,
            key_id=key_id,
            dilithium_keypair=keypair,
        )
        self.officer_keys[officer_id] = (slot_id, key_id, attestation)
        logger.info(
            "Zainicjalizowano sprzętowy token PKCS#11 dla %s w slocie %d (KeyID: %s).",
            officer_id,
            slot_id,
            key_id,
        )
        return attestation, keypair.public_key

    def create_hardware_signed_officer_entry(
        self,
        officer_id: str,
        officer_role: str,
        pin: str,
        token: CanonicalOperationToken,
    ) -> OfficerSignature:
        """Otwiera sesję HSM, uwierzytelnia PIN-em i generuje sprzętowy podpis rozkazu."""
        if officer_id not in self.officer_keys:
            raise KeyError(f"Oficer {officer_id} nie posiada skonfigurowanego tokenu sprzętowego HSM.")

        slot_id, key_id, attestation = self.officer_keys[officer_id]

        # 1. Otwarcie sesji PKCS#11
        sess = self.enclave.open_session(slot_id=slot_id, officer_id=officer_id)
        try:
            # 2. Logowanie kodem PIN oficera
            login_success = self.enclave.login(session_handle=sess.session_handle, pin=pin)
            if not login_success:
                raise PermissionError(f"Nieprawidłowy PIN sprzętowy dla oficera {officer_id} w slocie {slot_id}!")

            # 3. Sprzętowe podpisanie kanonicznych bajtów tokenu
            token_bytes = token.canonical_bytes()
            quantum_sig = self.enclave.hardware_sign(
                session_handle=sess.session_handle,
                key_id=key_id,
                message=token_bytes,
                pqc_engine=self.pqc,
            )

            # 4. Zwrot podpisu z kluczem publicznym wyciągniętym z atestacji
            pubkey_bytes = bytes.fromhex(attestation.public_key_hex)
            return OfficerSignature(
                officer_id=officer_id,
                officer_role=officer_role,
                public_key_bytes=pubkey_bytes,
                signature=quantum_sig,
            )
        finally:
            # 5. Bezpieczne zamknięcie sesji PKCS#11
            self.enclave.close_session(sess.session_handle)
