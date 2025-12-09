"""
Authentication and authorization module

This module provides JWT-based authentication with support for: 
- Access and refresh tokens
- Token revocation
- API key management
- Secure secret key handling from environment

Security Features:
- JWT tokens with configurable expiry
- Token revocation support (in-memory with optional external storage)
- API key hashing (bcrypt)
- Automatic secret key generation with warnings for ephemeral keys
- Environment-driven secret key for production deployments
"""

from __future__ import annotations

import hashlib
import bcrypt
import logging
import os
import secrets
import warnings
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass

import jwt

__all__ = [
    "TokenType",
    "TokenPayload",
    "APIKey",
    "AuthManager",
    "TokenExpiredError",
    "InvalidTokenError",
]

log = logging.getLogger(__name__)


class TokenExpiredError(Exception):
    """Raised when a token has expired"""


class InvalidTokenError(Exception):
    """Raised when a token is invalid"""


class TokenType(str, Enum):
    """Token type enumeration"""

    ACCESS = "access"
    REFRESH = "refresh"


@dataclass
class TokenPayload:
    """JWT token payload"""

    user_id: str
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    jti: str  # JWT ID for revocation tracking
    scope: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JWT encoding"""
        return {
            "sub": self.user_id,
            "type": self.token_type.value,
            "iat": int(self.issued_at.timestamp()),
            "exp": int(self. expires_at.timestamp()),
            "jti": self.jti,
            "scope": self. scope,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TokenPayload:
        """Create from JWT payload dictionary"""
        return cls(
            user_id=data["sub"],
            token_type=TokenType(data["type"]),
            issued_at=datetime.fromtimestamp(data["iat"], tz=timezone.utc),
            expires_at=datetime.fromtimestamp(data["exp"], tz=timezone.utc),
            jti=data["jti"],
            scope=data. get("scope"),
        )

    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class APIKey: 
    """API Key metadata"""

    key_id: str
    key_hash: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    enabled: bool = True

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.now(timezone. utc) > self.expires_at

    def is_valid(self) -> bool:
        return self.enabled and not self.is_expired()


class AuthManager:
    """
    Authentication Manager with secure JWT handling
    
    Security Best Practices:
    - Always provide secret_key via environment variable (JWT_SECRET) in production
    - Never commit secret keys to version control
    - Rotate secret keys periodically
    - Use external revocation storage for distributed deployments
    """
    
    # Insecure literal secret that should NEVER be used
    _INSECURE_SECRET = "secret"
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        access_token_expiry: timedelta = timedelta(hours=1),
        refresh_token_expiry: timedelta = timedelta(days=7),
        revocation_store: Optional[Callable[[str], None]] = None,
        revocation_checker: Optional[Callable[[str], bool]] = None,
    ):
        """
        Initialize AuthManager with secure secret key handling
        
        Args:
            secret_key: JWT signing secret.  If None, reads from JWT_SECRET env var. 
                       Falls back to auto-generated ephemeral key with warning.
            access_token_expiry: Access token lifetime (default: 1 hour)
            refresh_token_expiry:  Refresh token lifetime (default: 7 days)
            revocation_store: Optional callback to persist revoked token JTIs
            revocation_checker: Optional callback to check if JTI is revoked
            
        Raises:
            ValueError: If secret_key is the insecure literal "secret"
        """
        # Determine secret key with security checks
        if secret_key is None:
            # Try environment variable first (production best practice)
            secret_key = os.environ.get("JWT_SECRET")
            
            if secret_key is None:
                # Generate ephemeral key with strong warning
                secret_key = secrets.token_urlsafe(32)
                warnings.warn(
                    "AuthManager initialized without explicit secret_key or JWT_SECRET environment variable. "
                    "Auto-generated key will be lost on restart, invalidating all tokens.  "
                    "For production, set JWT_SECRET environment variable.",
                    UserWarning,
                    stacklevel=2,
                )
                log.warning(
                    "AuthManager:  No secret_key provided and JWT_SECRET not set. "
                    "Auto-generated ephemeral key will not persist across restarts. "
                    "Set JWT_SECRET environment variable for production."
                )
        
        # Block insecure literal secret
        if secret_key == self._INSECURE_SECRET:
            raise ValueError(
                f"Refusing to use insecure literal secret '{self._INSECURE_SECRET}'. "
                "Set JWT_SECRET environment variable or provide a cryptographically secure secret_key."
            )
        
        # Additional length check for security
        if len(secret_key) < 16:
            raise ValueError(
                f"secret_key too short ({len(secret_key)} chars). "
                "Use at least 16 characters for cryptographic security.  "
                "Recommended:  32+ characters or set JWT_SECRET environment variable."
            )
        
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
                "Revoked tokens will be lost on restart.  For production, "
                "provide revocation_store and revocation_checker callbacks."
            )

        log.info("AuthManager initialized with secure secret key")

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
            payload. to_dict(),
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
                    "require":  ["sub", "type", "iat", "exp", "jti"],
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
            token_type=TokenType. ACCESS,
            issued_at=now,
            expires_at=now + self.access_token_expiry,
            scope=scope,
            jti=secrets.token_urlsafe(16),
        )
        token = self._encode_token(payload)
        log.info(f"Created access token for user {user_id}")
        return token, payload

    def create_refresh_token(self, user_id: str) -> Tuple[str, TokenPayload]: 
        now = datetime.now(timezone.utc)
        payload = TokenPayload(
            user_id=user_id,
            token_type=TokenType. REFRESH,
            issued_at=now,
            expires_at=now + self.refresh_token_expiry,
            jti=secrets.token_urlsafe(16),
        )
        token = self._encode_token(payload)
        log.info(f"Created refresh token for user {user_id}")
        return token, payload

    def verify_token(self, token: str) -> TokenPayload:
        return self._decode_token(token)

    def revoke_token(self, token: str) -> None:
        payload = self._decode_token(token)
        self._store_revocation(payload. jti)
        log.info(f"Revoked token {payload.jti} for user {payload.user_id}")

    def create_api_key(
        self,
        key_id: str,
        name: str,
        expires_at: Optional[datetime] = None,
    ) -> Tuple[str, APIKey]:
        """Create a new API key (returns unhashed key once)"""
        raw_key = secrets.token_urlsafe(32)
        key_hash = bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime. now(timezone.utc),
            expires_at=expires_at,
        )

        self.api_keys[key_id] = api_key
        log.info(f"Created API key {key_id}")
        return raw_key, api_key

    def verify_api_key(self, raw_key: str) -> Optional[str]:
        """Verify API key and return key_id if valid"""

        for key_id, api_key in self.api_keys.items():
            if (
                api_key.is_valid() and
                bcrypt.checkpw(raw_key.encode(), api_key.key_hash.encode())
            ):
                api_key.last_used_at = datetime.now(timezone.utc)
                log.info(f"API key {key_id} verified")
                return key_id

        log.warning("Invalid API key attempt")
        return None

    def revoke_api_key(self, key_id: str) -> None:
        """Revoke an API key"""
        if key_id in self.api_keys:
            self.api_keys[key_id].enabled = False
            log. info(f"Revoked API key {key_id}")
        else:
            log.warning(f"Attempted to revoke non-existent API key {key_id}")
