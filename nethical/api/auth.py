"""
Authentication manager for Nethical API.

Provides simple API key authentication with environment-based configuration.
Designed for single-instance deployments with minimal overhead, with optional
runtime key reload capability.

Environment:
    NETHICAL_API_KEYS   Comma-separated API keys (optional; if unset => permissive mode)

Features:
    - Permissive mode when no keys configured
    - Constant-time comparison for API key validation
    - Hashed identity (SHA256) used for rate limiting & logs (privacy)
    - Runtime key reload via reload_keys()
"""

import os
import hmac
import hashlib
import logging
from typing import Optional, Set, Iterable

logger = logging.getLogger(__name__)


class AuthManager:
    """
    Simple API key authentication manager.

    If no keys configured (NETHICAL_API_KEYS empty or unset) the manager runs in
    permissive mode: all requests considered authenticated (rate limiting still applies).

    Methods:
        is_permissive() -> bool
        validate_key(api_key: str) -> bool
        extract_identity(api_key: Optional[str], client_ip: str) -> str
        reload_keys(new_keys: Optional[Iterable[str]] = None) -> None
        get_stats() -> dict
        list_keys_masked() -> list[str]
    """

    def __init__(self) -> None:
        self._api_keys: Optional[Set[str]] = None
        self._permissive_mode: bool = True
        self._last_reload_ts: Optional[float] = None
        self._load_from_env(initial=True)

    # --------------------------------------------------------------------- #
    # Internal loading helpers
    # --------------------------------------------------------------------- #
    def _load_from_env(self, initial: bool = False) -> None:
        keys_env = os.getenv("NETHICAL_API_KEYS", "").strip()
        if keys_env:
            self._api_keys = {k.strip() for k in keys_env.split(",") if k.strip()}
            self._permissive_mode = False
            self._last_reload_ts = self._now()
            logger.info(
                f"Authentication {'initialized' if initial else 'reloaded'} "
                f"with {len(self._api_keys)} configured API keys"
            )
        else:
            self._api_keys = None
            self._permissive_mode = True
            self._last_reload_ts = self._now()
            logger.warning(
                "No API keys configured (NETHICAL_API_KEYS not set). "
                "Running in PERMISSIVE MODE - requests authenticated only by IP-based rate limiting."
            )

    def _now(self) -> float:
        import time
        return time.time()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def is_permissive(self) -> bool:
        """Return True if running in permissive (no auth required) mode."""
        return self._permissive_mode

    def validate_key(self, api_key: str) -> bool:
        """
        Validate an API key using constant-time comparison.

        Returns True if:
          - permissive mode OR
          - key matches one of configured keys
        """
        if self._permissive_mode:
            return True
        if not api_key or not self._api_keys:
            return False

        # Constant-time comparison against each stored key
        for stored in self._api_keys:
            if hmac.compare_digest(stored, api_key):
                return True
        return False

    def extract_identity(self, api_key: Optional[str], client_ip: str) -> str:
        """
        Determine identity string used for rate limiting/logging.

        Prefers API key (hashed) when valid; falls back to IP address.
        """
        if api_key and (self._permissive_mode or self.validate_key(api_key)):
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            return f"key:{key_hash}"
        return f"ip:{client_ip}"

    def reload_keys(self, new_keys: Optional[Iterable[str]] = None) -> None:
        """
        Reload API keys at runtime.

        If new_keys provided, uses them directly; otherwise reloads from environment.
        """
        if new_keys is not None:
            new_set = {k.strip() for k in new_keys if k and k.strip()}
            if new_set:
                self._api_keys = new_set
                self._permissive_mode = False
                self._last_reload_ts = self._now()
                logger.info(f"Authentication reloaded with {len(self._api_keys)} provided API keys")
            else:
                self._api_keys = None
                self._permissive_mode = True
                self._last_reload_ts = self._now()
                logger.warning(
                    "Provided new_keys empty; switching to PERMISSIVE MODE."
                )
        else:
            self._load_from_env(initial=False)

    def list_keys_masked(self) -> list[str]:
        """
        Return a masked list of configured keys for debugging (first 4 chars + length).
        """
        if not self._api_keys:
            return []
        masked = []
        for k in self._api_keys:
            masked.append(f"{k[:4]}…(len={len(k)})")
        return masked

    def get_stats(self) -> dict:
        """Return auth subsystem statistics."""
        return {
            "permissive_mode": self._permissive_mode,
            "configured_keys": len(self._api_keys) if self._api_keys else 0,
            "last_reload_ts": self._last_reload_ts,
            "keys_masked": self.list_keys_masked(),
        }
