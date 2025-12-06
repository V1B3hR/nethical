"""
JWT-based Authentication System for Nethical

This module provides:
- JWT token generation and validation (using PyJWT library)
- API key management
- Authentication middleware
- Token refresh mechanism
- Multi-factor authentication support (stub for future implementation)

Security Notes:
- JWT tokens are signed using HS256 with HMAC-SHA256
- Revoked tokens are stored in-memory; use Redis/database for production persistence
- Secret key should be provided explicitly; auto-generated keys are not persisted
"""

from __future__ import annotations

import hashlib
from argon2 import PasswordHasher
import hmac
import logging
import secrets
import warnings
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    import jwt
except ImportError as e:
    raise ImportError(
        "PyJWT is required for nethical.security.auth. "
        "Please install it using 'pip install pyjwt'."
    ) from e

__all__ = [
    "AuthenticationError",
    "TokenExpiredError",
    "InvalidTokenError",
    "AuthManager",
    "TokenPayload",
    "APIKey",
    "get_auth_manager",
    "set_auth_manager",
]

log = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Base exception for authentication errors"""


class TokenExpiredError(AuthenticationError):
    """Raised when a token has expired"""


class InvalidTokenError(AuthenticationError):
    """Raised when a token is invalid"""


class TokenType(str, Enum):
    """Types of authentication tokens"""

    ACCESS = "access"
    REFRESH = "refresh"


@dataclass
class TokenPayload:
    """JWT token payload"""

    user_id: str
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    jti: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    scope: Optional[str] = None

    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub": self.user_id,
            "type": self.token_type.value,
            "iat": int(self.issued_at.timestamp()),
            "exp": int(self.expires_at.timestamp()),
            "jti": self.jti,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenPayload":
        return cls(
            user_id=data["sub"],
            token_type=TokenType(data["type"]),
            issued_at=datetime.fromtimestamp(data["iat"], tz=timezone.utc),
            expires_at=datetime.fromtimestamp(data["exp"], tz=timezone.utc),
            jti=data["jti"],
            scope=data.get("scope"),
        )


@dataclass
class APIKey:
    """API Key for service-to-service authentication"""

    key_id: str
    user_id: str
    key_hash: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    enabled: bool = True

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def is_valid(self) -> bool:
        return self.enabled and not self.is_expired()


class AuthManager:
    def __init__(
        self,
        secret_key: Optional[str] = None,
        access_token_expiry: timedelta = timedelta(hours=1),
        refresh_token_expiry: timedelta = timedelta(days=7),
        revocation_store: Optional[Callable[[str], None]] = None,
        revocation_checker: Optional[Callable[[str], bool]] = None,
    ):
        if secret_key is None:
            self.secret_key = secrets.token_urlsafe(32)
            warnings.warn(
                "AuthManager initialized without explicit secret_key. "
                "Auto-generated key will be lost on restart, invalidating all tokens. "
                "For production, always provide an explicit secret_key.",
                UserWarning,
                stacklevel=2,
            )
            log.warning(
                "AuthManager: No secret_key provided. Auto-generated key will "
                "not persist across restarts. Provide explicit key for production."
            )
        else:
            self.secret_key = secret_key

        self.access_token_expiry = access_token_expiry
        self.refresh_token_expiry = refresh_token_expiry

        self.api_keys: Dict[str, APIKey] = {}
        self._revoked_tokens: set[str] = set()  # JTI of revoked tokens (in-memory)

        self._revocation_store = revocation_store
        self._revocation_checker = revocation_checker

        if not revocation_store:
            log.warning(
                "AuthManager: Token revocation storage is in-memory only. "
                "Revoked tokens will be lost on restart. For production, "
                "provide revocation_store and revocation_checker callbacks."
            )

        log.info("AuthManager initialized")

    @property
    def revoked_tokens(self) -> set[str]:
        return self._revoked_tokens

    def _is_token_revoked(self, jti: str) -> bool:
        if jti in self._revoked_tokens:
            return True
        if self._revocation_checker:
            return self._revocation_checker(jti)
        return False

    def _store_revocation(self, jti: str) -> None:
        self._revoked_tokens.add(jti)
        if self._revocation_store:
            self._revocation_store(jti)

    def _encode_token(self, payload: TokenPayload) -> str:
        return jwt.encode(
            payload.to_dict(),
            self.secret_key,
            algorithm="HS256",
        )

    def _decode_token(self, token: str) -> TokenPayload:
        try:
            payload_data = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"],
                options={
                    "require": ["sub", "type", "iat", "exp", "jti"],
                },
            )
            payload = TokenPayload.from_dict(payload_data)
            if self._is_token_revoked(payload.jti):
                raise InvalidTokenError("Token has been revoked")
            return payload
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {e}")
        except (ValueError, KeyError) as e:
            raise InvalidTokenError(f"Failed to decode token: {e}")

    def create_access_token(
        self, user_id: str, scope: Optional[str] = None
    ) -> Tuple[str, TokenPayload]:
        now = datetime.now(timezone.utc)
        payload = TokenPayload(
            user_id=user_id,
            token_type=TokenType.ACCESS,
            issued_at=now,
            expires_at=now + self.access_token_expiry,
            scope=scope,
        )
        token = self._encode_token(payload)
        log.info(f"Created access token for user {user_id}")
        return token, payload

    def create_refresh_token(self, user_id: str) -> Tuple[str, TokenPayload]:
        now = datetime.now(timezone.utc)
        payload = TokenPayload(
            user_id=user_id,
            token_type=TokenType.REFRESH,
            issued_at=now,
            expires_at=now + self.refresh_token_expiry,
        )
        token = self._encode_token(payload)
        log.info(f"Created refresh token for user {user_id}")
        return token, payload

    def verify_token(self, token: str) -> TokenPayload:
        return self._decode_token(token)

    def revoke_token(self, token: str) -> None:
        try:
            payload = self._decode_token(token)
            self._store_revocation(payload.jti)
            log.info(f"Revoked token {payload.jti} for user {payload.user_id}")
        except (TokenExpiredError, InvalidTokenError):
            log.warning("Attempted to revoke invalid/expired token")

    def refresh_access_token(self, refresh_token: str) -> Tuple[str, TokenPayload]:
        payload = self.verify_token(refresh_token)
        if payload.token_type != TokenType.REFRESH:
            raise InvalidTokenError("Token is not a refresh token")
        return self.create_access_token(payload.user_id, payload.scope)

    def create_api_key(
        self, user_id: str, name: str, expires_at: Optional[datetime] = None
    ) -> Tuple[str, APIKey]:
        key_id = secrets.token_urlsafe(8)
        key_secret = secrets.token_urlsafe(32)
        api_key_string = f"{key_id}.{key_secret}"
        ph = PasswordHasher()
        key_hash = ph.hash(api_key_string)
        api_key = APIKey(
            key_id=key_id,
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
        )
        self.api_keys[key_id] = api_key
        log.info(f"Created API key '{name}' (ID: {key_id}) for user {user_id}")
        return api_key_string, api_key

    def verify_api_key(self, api_key_string: str) -> APIKey:
        try:
            parts = api_key_string.split(".")
            if len(parts) != 2:
                raise InvalidTokenError("Invalid API key format")
            key_id, _ = parts
            api_key = self.api_keys.get(key_id)
            if not api_key:
                raise InvalidTokenError("API key not found")
            ph = PasswordHasher()
            try:
                ph.verify(api_key.key_hash, api_key_string)
            except Exception:
                raise InvalidTokenError("Invalid API key")
            if not api_key.is_valid():
                raise InvalidTokenError("API key is disabled or expired")
            api_key.last_used_at = datetime.now(timezone.utc)
            log.info(f"Verified API key for user {api_key.user_id}")
            return api_key
        except (ValueError, AttributeError) as e:
            raise InvalidTokenError(f"Failed to verify API key: {e}")

    def revoke_api_key(self, key_id: str) -> None:
        api_key = self.api_keys.get(key_id)
        if api_key:
            api_key.enabled = False
            log.info(f"Revoked API key {key_id}")
        else:
            log.warning(f"Attempted to revoke non-existent API key {key_id}")

    def list_api_keys(self, user_id: Optional[str] = None) -> list[APIKey]:
        keys = list(self.api_keys.values())
        if user_id:
            keys = [k for k in keys if k.user_id == user_id]
        return keys

    def cleanup_expired_tokens(self) -> int:
        count = len(self.revoked_tokens)
        self.revoked_tokens.clear()
        return count


_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def set_auth_manager(manager: AuthManager) -> None:
    global _auth_manager
    _auth_manager = manager


def authenticate_request(
    authorization_header: Optional[str] = None, api_key_header: Optional[str] = None
) -> str:
    auth_manager = get_auth_manager()
    if authorization_header:
        if not authorization_header.startswith("Bearer "):
            raise AuthenticationError("Invalid authorization header format")
        token = authorization_header[7:]
        try:
            payload = auth_manager.verify_token(token)
            return payload.user_id
        except (TokenExpiredError, InvalidTokenError) as e:
            raise AuthenticationError(f"Token authentication failed: {e}")
    if api_key_header:
        try:
            api_key = auth_manager.verify_api_key(api_key_header)
            return api_key.user_id
        except InvalidTokenError as e:
            raise AuthenticationError(f"API key authentication failed: {e}")
    raise AuthenticationError("No authentication credentials provided")
