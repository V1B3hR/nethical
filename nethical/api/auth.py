"""
Authentication manager for Nethical API.

Provides API key authentication with:
    - Permissive mode when no keys configured
    - Constant-time comparison for key validation
    - Privacy-preserving pseudonym for logging / rate limiting (PBKDF2 derivation)
    - Runtime key reload capability

IMPORTANT:
    We are NOT storing passwords; API keys are secrets used for request auth.
    CodeQL may flag direct SHA256 hashing of secrets as weak for *password hashing*.
    To avoid that and strengthen pseudonym derivation, we use PBKDF2-HMAC with a
    configurable iteration count and salt. The derived value is ONLY used as a
    pseudonymous identifier (not reversible without the key + salt).

Environment Variables:
    NETHICAL_API_KEYS                Comma-separated API keys (optional)
    NETHICAL_API_KEY_HASH_SALT       Hex/base64/utf-8 salt (optional; random if unset)
    NETHICAL_API_KEY_HASH_ITERATIONS PBKDF2 iteration count (default: 150000)

Public Methods:
    is_permissive() -> bool
    validate_key(api_key: str) -> bool
    extract_identity(api_key: Optional[str], client_ip: str) -> str
    reload_keys(new_keys: Optional[Iterable[str]] = None) -> None
    get_stats() -> dict
    list_keys_masked() -> list[str]
"""

import os
import hmac
import hashlib
import secrets
import logging
from typing import Optional, Set, Iterable

logger = logging.getLogger(__name__)


class AuthManager:
    def __init__(self) -> None:
        self._api_keys: Optional[Set[str]] = None
        self._permissive_mode: bool = True
        self._last_reload_ts: Optional[float] = None

        # Pseudonym derivation config
        self._iterations = self._load_iterations()
        self._salt = self._load_salt()

        self._load_from_env(initial=True)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _now(self) -> float:
        import time

        return time.time()

    def _load_iterations(self) -> int:
        raw = os.getenv("NETHICAL_API_KEY_HASH_ITERATIONS", "").strip()
        if not raw:
            return 150_000
        try:
            val = int(raw)
            if val < 50_000:
                logger.warning(
                    "Iteration count %d below recommended minimum (50k); using 50k instead.",
                    val,
                )
                return 50_000
            return val
        except ValueError:
            logger.warning("Invalid iteration count '%s'; falling back to 150000.", raw)
            return 150_000

    def _load_salt(self) -> bytes:
        salt_env = os.getenv("NETHICAL_API_KEY_HASH_SALT", "").strip()
        if not salt_env:
            # Generate random salt if not provided
            salt = secrets.token_bytes(32)
            logger.info("Generated random API key hash salt (not persisted).")
            return salt
        # Attempt decode: try hex first, then base64, else raw utf-8
        for decoder in (bytes.fromhex, self._b64decode):
            try:
                decoded = decoder(salt_env)
                if decoded:
                    logger.info("Loaded API key hash salt from environment.")
                    return decoded
            except Exception:
                continue
        # Fallback: raw utf-8
        logger.info("Using raw UTF-8 environment value as salt.")
        return salt_env.encode("utf-8")

    def _b64decode(self, s: str) -> bytes:
        import base64

        return base64.b64decode(s)

    def _load_from_env(self, initial: bool = False) -> None:
        keys_env = os.getenv("NETHICAL_API_KEYS", "").strip()
        if keys_env:
            self._api_keys = {k.strip() for k in keys_env.split(",") if k.strip()}
            self._permissive_mode = False
            self._last_reload_ts = self._now()
            logger.info(
                "Authentication %s with %d API keys (iterations=%d).",
                "initialized" if initial else "reloaded",
                len(self._api_keys),
                self._iterations,
            )
        else:
            self._api_keys = None
            self._permissive_mode = True
            self._last_reload_ts = self._now()
            logger.warning(
                "No API keys configured; PERMISSIVE MODE enabled (rate limiting still applies)."
            )

    # ------------------------------------------------------------------ #
    # Key validation & pseudonym derivation
    # ------------------------------------------------------------------ #
    def is_permissive(self) -> bool:
        return self._permissive_mode

    def validate_key(self, api_key: str) -> bool:
        if self._permissive_mode:
            return True
        if not api_key or not self._api_keys:
            return False
        for stored in self._api_keys:
            if hmac.compare_digest(stored, api_key):
                return True
        return False

    def _derive_pseudonym(self, api_key: str) -> str:
        """
        Derive a pseudonym for logging/rate limiting using PBKDF2-HMAC.
        Output truncated for brevity; not a password hash replacement.
        """
        dk = hashlib.pbkdf2_hmac(
            "sha256",  # underlying PRF
            api_key.encode("utf-8"),
            self._salt,
            self._iterations,
            dklen=32,
        )
        # Represent as hex; truncate to first 32 chars for log readability
        return dk.hex()[:32]

    def extract_identity(self, api_key: Optional[str], client_ip: str) -> str:
        if api_key and (self._permissive_mode or self.validate_key(api_key)):
            pseudonym = self._derive_pseudonym(api_key)
            return f"key:{pseudonym}"
        return f"ip:{client_ip}"

    # ------------------------------------------------------------------ #
    # Runtime management
    # ------------------------------------------------------------------ #
    def reload_keys(self, new_keys: Optional[Iterable[str]] = None) -> None:
        """
        Reload keys from provided iterable or environment.
        Salt & iteration count remain unchanged unless env vars changed and
        a process restart occurs (keeping runtime complexity low).
        """
        if new_keys is not None:
            new_set = {k.strip() for k in new_keys if k and k.strip()}
            if new_set:
                self._api_keys = new_set
                self._permissive_mode = False
                self._last_reload_ts = self._now()
                logger.info(
                    "Authentication reloaded with %d provided API keys.", len(new_set)
                )
            else:
                self._api_keys = None
                self._permissive_mode = True
                self._last_reload_ts = self._now()
                logger.warning("Provided new_keys empty; PERMISSIVE MODE re-enabled.")
        else:
            self._load_from_env(initial=False)

    def list_keys_masked(self) -> list[str]:
        if not self._api_keys:
            return []
        return [f"{k[:4]}…(len={len(k)})" for k in self._api_keys]

    def get_stats(self) -> dict:
        return {
            "permissive_mode": self._permissive_mode,
            "configured_keys": len(self._api_keys) if self._api_keys else 0,
            "last_reload_ts": self._last_reload_ts,
            "masked_keys": self.list_keys_masked(),
            "kdf_iterations": self._iterations,
            "salt_length": len(self._salt),
        }
