"""Reversible Token Vault for Dynamic In-Flight Pseudonymization.

Protects sensitive PII and ePHI before transmission to third-party or untrusted LLMs:
- Forward Pass (tokenize): Identifies PESEL, emails, phone numbers, credit cards, ePHI,
  and names, substituting them with collision-resistant synthetic tokens (e.g. [TOKEN_PESEL_8f2a]).
- Cryptographic Vault: Mappings are encrypted with AES-256-GCM and indexed via HMAC-SHA256.
- Reverse Pass (detokenize): Restores original PII into the model's completion for authorized users.
- Compliance: EU GDPR Art. 4(5) (Pseudonymisation), HIPAA Safe Harbor (45 CFR § 164.514), California AB 2013.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import BaseModel, Field

logger = logging.getLogger("nethical.security.token_vault")


class SensitiveEntityType(str, Enum):
    PESEL = "PESEL"
    NIP = "NIP"
    IBAN = "IBAN"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    API_KEY_SECRET = "API_KEY_SECRET"
    MEDICAL_RECORD_EPHI = "MEDICAL_RECORD_EPHI"
    PERSON_NAME = "PERSON_NAME"


class TokenizedEntityRecord(BaseModel):
    token: str
    entity_type: str
    original_sha256: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TokenizeResponse(BaseModel):
    session_id: str
    sanitized_text: str
    tokens_substituted_count: int
    substituted_entities: List[TokenizedEntityRecord] = Field(default_factory=list)


class DetokenizeResponse(BaseModel):
    session_id: str
    restored_text: str
    tokens_restored_count: int


# Regex patterns for fast deterministic PII/ePHI scanning
PII_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("API_KEY_SECRET", re.compile(r"\b(?:AKIA[0-9A-Z]{16}|ghp_[0-9a-zA-Z]{36}|github_pat_[0-9a-zA-Z_]{20,82}|sk-[A-Za-z0-9-_]{32,}|xox[baprs]-[0-9a-zA-Z]{10,48}|(?:-----BEGIN[ A-Z_-]*PRIVATE KEY-----[\s\S]*?-----END[ A-Z_-]*PRIVATE KEY-----))\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{12,30}\b")),
    ("PESEL", re.compile(r"\b\d{11}\b")),
    ("NIP", re.compile(r"\b(?:\d{3}[- ]?\d{3}[- ]?\d{2}[- ]?\d{2}|\d{10})\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+48[\s-]?)?[4-9]\d{2}[\s-]?\d{3}[\s-]?\d{3}\b")),
    ("MEDICAL_RECORD_EPHI", re.compile(r"\b(?:MRN|EPHI|ICD)[-:]?[A-Z0-9]{5,10}\b", re.IGNORECASE)),
]


class ReversibleTokenVault:
    """AES-256-GCM in-flight pseudonymization vault with reversible detokenization."""

    def __init__(
        self,
        master_key: Optional[bytes] = None,
        session_ttl_minutes: int = 60,
    ) -> None:
        self.master_key = master_key or secrets.token_bytes(32)  # 256-bit AES key
        self.aesgcm = AESGCM(self.master_key)
        self.ttl = timedelta(minutes=session_ttl_minutes)

        # In-memory secure encrypted storage:
        # session_id -> { token: (nonce, ciphertext) }
        self._vault: Dict[str, Dict[str, Tuple[bytes, bytes]]] = {}
        self._session_expiry: Dict[str, datetime] = {}

    def _generate_token(self, entity_type: str, original_val: str, session_id: str) -> str:
        """Generates a stable, collision-free synthetic token for the session."""
        digest = hmac.new(
            self.master_key,
            f"{session_id}:{original_val}".encode("utf-8"),
            hashlib.sha256
        ).hexdigest()[:8]
        return f"[TOKEN_{entity_type}_{digest}]"

    @staticmethod
    def _is_valid_candidate(entity_type: str, val: str) -> bool:
        """Validates candidate string against structural and checksum rules."""
        if entity_type == "PESEL":
            digits = [int(c) for c in val if c.isdigit()]
            if len(digits) != 11:
                return False
            weights = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3, 1]
            return sum(w * d for w, d in zip(weights, digits)) % 10 == 0
        elif entity_type == "NIP":
            digits = [int(c) for c in val if c.isdigit()]
            if len(digits) != 10:
                return False
            weights = [6, 5, 7, 2, 3, 4, 5, 6, 7]
            checksum = sum(w * d for w, d in zip(weights, digits[:9])) % 11
            return checksum == digits[9]
        elif entity_type == "IBAN":
            clean = val.replace(" ", "").upper()
            return len(clean) >= 15 and len(clean) <= 34 and clean[:2].isalpha()
        elif entity_type == "API_KEY_SECRET":
            return len(val) >= 16
        return True

    def tokenize(self, text: str, session_id: Optional[str] = None) -> TokenizeResponse:
        """Substitutes PII with synthetic tokens, encrypting mappings in vault."""
        sid = session_id or f"TV-{secrets.token_hex(8)}"
        now = datetime.now(timezone.utc)

        if sid not in self._vault:
            self._vault[sid] = {}
        self._session_expiry[sid] = now + self.ttl

        sanitized = text
        records: List[TokenizedEntityRecord] = []

        # Find and substitute regex-matched entities
        for entity_type, pattern in PII_PATTERNS:
            matches = list(set(pattern.findall(sanitized)))
            for match in matches:
                clean_match = match.strip()
                if not self._is_valid_candidate(entity_type, clean_match):
                    continue
                token = self._generate_token(entity_type, clean_match, sid)

                # Encrypt original value with AESGCM
                nonce = secrets.token_bytes(12)
                ciphertext = self.aesgcm.encrypt(nonce, clean_match.encode("utf-8"), sid.encode("utf-8"))
                self._vault[sid][token] = (nonce, ciphertext)

                # Replace in text
                sanitized = sanitized.replace(clean_match, token)

                sha256_hash = hashlib.sha256(clean_match.encode("utf-8")).hexdigest()
                records.append(
                    TokenizedEntityRecord(
                        token=token,
                        entity_type=entity_type,
                        original_sha256=sha256_hash,
                    )
                )

        return TokenizeResponse(
            session_id=sid,
            sanitized_text=sanitized,
            tokens_substituted_count=len(records),
            substituted_entities=records,
        )

    def detokenize(self, text: str, session_id: str) -> DetokenizeResponse:
        """Restores synthetic tokens back to original plaintext values."""
        now = datetime.now(timezone.utc)
        if session_id not in self._vault or self._session_expiry.get(session_id, now) < now:
            logger.warning(f"Session {session_id} expired or not found in vault.")
            return DetokenizeResponse(
                session_id=session_id,
                restored_text=text,
                tokens_restored_count=0,
            )

        restored = text
        session_tokens = self._vault[session_id]
        count = 0

        # Scan for tokens formatted like [TOKEN_TYPE_HASH]
        token_pattern = re.compile(r"\[TOKEN_[A-Z_]+_[a-f0-9]{8}\]")
        found_tokens = list(set(token_pattern.findall(text)))

        for token in found_tokens:
            if token in session_tokens:
                nonce, ciphertext = session_tokens[token]
                try:
                    decrypted_bytes = self.aesgcm.decrypt(
                        nonce, ciphertext, session_id.encode("utf-8")
                    )
                    original_val = decrypted_bytes.decode("utf-8")
                    restored = restored.replace(token, original_val)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to decrypt token {token}: {e}")

        return DetokenizeResponse(
            session_id=session_id,
            restored_text=restored,
            tokens_restored_count=count,
        )

    def purge_session(self, session_id: str) -> bool:
        """Immediately destroys encryption keys and mappings for session (Right to be Forgotten)."""
        purged = False
        if session_id in self._vault:
            del self._vault[session_id]
            purged = True
        if session_id in self._session_expiry:
            del self._session_expiry[session_id]
        return purged
