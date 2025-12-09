# -*- coding: utf-8 -*-
"""
nethical.security.auth

Rewritten AuthManager with secure JWT secret handling:
- Reads secret from constructor parameter, or JWT_SECRET env var.
- Rejects obviously insecure literal 'secret'.
- Generates ephemeral secret only when explicitly allowed; logs a warning.
- Encodes/decodes tokens with HS256 and required claims.
- Raises explicit errors for misconfiguration and invalid tokens.
"""

from __future__ import annotations

import os
import logging
import secrets
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, Optional, Tuple, Any

import jwt  # PyJWT

log = logging.getLogger(__name__)

# Exceptions
class TokenExpiredError(Exception):
    pass

class InvalidTokenError(Exception):
    pass

class ConfigurationError(Exception):
    pass

# Token types
class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"

@dataclass
class TokenPayload:
    sub: str
    type: TokenType
    iat: int
    exp: int
    jti: str
    scope: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # convert TokenType to value for serialization
        d["type"] = self.type.value if isinstance(self.type, TokenType) else self.type
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TokenPayload":
        try:
            return TokenPayload(
                sub=data["sub"],
                type=TokenType(data["type"]),
                iat=int(data["iat"]),
                exp=int(data["exp"]),
                jti=str(data["jti"]),
                scope=data.get("scope"),
            )
        except Exception as e:
            raise InvalidTokenError(f"Malformed token payload: {e}")

class AuthManager:
    """
    AuthManager handles JWT access/refresh token creation and verification.

    Usage:
      auth = AuthManager(secret_key=None)  # will read from JWT_SECRET env var if present
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        access_token_expiry: timedelta = timedelta(minutes=15),
        refresh_token_expiry: timedelta = timedelta(days=7),
        allow_ephemeral_secret: bool = False,
    ) -> None:
        """
        Initialize the AuthManager.

        - secret_key: explicit HMAC secret. If None, we attempt to read JWT_SECRET env var.
        - allow_ephemeral_secret: when True, an ephemeral secret will be generated if none provided.
          If False and no secret is provided, a ConfigurationError is raised.
        """
        provided = secret_key or os.environ.get("JWT_SECRET")

        if provided:
            if provided.strip().lower() == "secret":
                # Treat the literal "secret" as a misconfiguration — refuse to run with it.
                raise ConfigurationError(
                    "Insecure JWT secret value 'secret' detected. Provide a strong secret via "
                    "the 'secret_key' parameter or JWT_SECRET environment variable."
                )
            self.secret_key = provided
        else:
            if allow_ephemeral_secret:
                # Generate ephemeral secret but warn loudly. Tokens will not persist across restarts.
                self.secret_key = secrets.token_hex(32)
                log.warning(
                    "AuthManager: No JWT secret provided; generated ephemeral secret for runtime only. "
                    "Set JWT_SECRET in environment for persistent, secure operation."
                )
            else:
                raise ConfigurationError(
                    "AuthManager requires a JWT secret. Set the 'secret_key' argument or "
                    "the JWT_SECRET environment variable."
                )

        self.access_token_expiry = access_token_expiry
        self.refresh_token_expiry = refresh_token_expiry

        # In-memory revocation store (JTI strings)
        self._revoked_jtis: set[str] = set()

        log.info("AuthManager initialized (HS256)")

    def _now_ts(self) -> int:
        return int(datetime.now(timezone.utc).timestamp())

    def _encode_token(self, payload: TokenPayload) -> str:
        token_dict = payload.to_dict()
        encoded = jwt.encode(token_dict, self.secret_key, algorithm="HS256")
        # In PyJWT v2+, jwt.encode returns a str
        return encoded

    def _decode_token(self, token: str) -> TokenPayload:
        try:
            payload_data = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"],
                options={"require": ["sub", "type", "iat", "exp", "jti"]},
            )
            payload = TokenPayload.from_dict(payload_data)
            if payload.jti in self._revoked_jtis:
                raise InvalidTokenError("Token has been revoked")
            # Check expiration manually (PyJWT decodes into ints)
            now_ts = self._now_ts()
            if payload.exp < now_ts:
                raise TokenExpiredError("Token has expired")
            return payload
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {e}")

    def create_access_token(self, user_id: str, scope: Optional[str] = None) -> Tuple[str, TokenPayload]:
        now = datetime.now(timezone.utc)
        iat = int(now.timestamp())
        exp = int((now + self.access_token_expiry).timestamp())
        jti = secrets.token_hex(16)
        payload = TokenPayload(sub=user_id, type=TokenType.ACCESS, iat=iat, exp=exp, jti=jti, scope=scope)
        token = self._encode_token(payload)
        log.info(f"Created access token for user {user_id}")
        return token, payload

    def create_refresh_token(self, user_id: str) -> Tuple[str, TokenPayload]:
        now = datetime.now(timezone.utc)
        iat = int(now.timestamp())
        exp = int((now + self.refresh_token_expiry).timestamp())
        jti = secrets.token_hex(16)
        payload = TokenPayload(sub=user_id, type=TokenType.REFRESH, iat=iat, exp=exp, jti=jti)
        token = self._encode_token(payload)
        log.info(f"Created refresh token for user {user_id}")
        return token, payload

    def verify_token(self, token: str) -> TokenPayload:
        return self._decode_token(token)

    def revoke_token(self, jti: str) -> None:
        self._revoked_jtis.add(jti)
        log.info(f"Revoked token jti={jti}")

    @property
    def revoked_jtis(self) -> set:
        return set(self._revoked_jtis)
