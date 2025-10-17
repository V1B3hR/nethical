"""
JWT-based Authentication System for Nethical

This module provides:
- JWT token generation and validation
- API key management
- Authentication middleware
- Token refresh mechanism
- Multi-factor authentication support (stub for future implementation)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

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
    pass


class TokenExpiredError(AuthenticationError):
    """Raised when a token has expired"""
    pass


class InvalidTokenError(AuthenticationError):
    """Raised when a token is invalid"""
    pass


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
    jti: str = field(default_factory=lambda: secrets.token_urlsafe(16))  # JWT ID
    scope: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.now(timezone.utc) > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JWT encoding"""
        return {
            "sub": self.user_id,
            "type": self.token_type.value,
            "iat": int(self.issued_at.timestamp()),
            "exp": int(self.expires_at.timestamp()),
            "jti": self.jti,
            "scope": self.scope,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TokenPayload:
        """Create from dictionary (JWT decoding)"""
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
    key_hash: str  # SHA-256 hash of the actual key
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    enabled: bool = True
    
    def is_expired(self) -> bool:
        """Check if API key is expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if API key is valid"""
        return self.enabled and not self.is_expired()


class AuthManager:
    """
    Authentication Manager
    
    Handles JWT tokens and API keys for authentication.
    Note: This is a basic implementation. For production, consider using
    a proper JWT library like PyJWT and a secure secret management system.
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        access_token_expiry: timedelta = timedelta(hours=1),
        refresh_token_expiry: timedelta = timedelta(days=7),
    ):
        """
        Initialize Authentication Manager
        
        Args:
            secret_key: Secret key for signing tokens (generated if not provided)
            access_token_expiry: Expiry time for access tokens
            refresh_token_expiry: Expiry time for refresh tokens
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.access_token_expiry = access_token_expiry
        self.refresh_token_expiry = refresh_token_expiry
        
        # Storage for API keys and revoked tokens
        self.api_keys: Dict[str, APIKey] = {}
        self.revoked_tokens: set[str] = set()  # JTI of revoked tokens
        
        log.info("AuthManager initialized")
    
    def _encode_token(self, payload: TokenPayload) -> str:
        """
        Encode a JWT token (simplified implementation)
        
        Note: This is a basic implementation for demonstration.
        In production, use a proper JWT library like PyJWT.
        """
        # Create token parts
        header = {
            "alg": "HS256",
            "typ": "JWT"
        }
        
        # Base64 encode (URL-safe)
        import base64
        
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header, separators=(",", ":")).encode()
        ).decode().rstrip("=")
        
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload.to_dict(), separators=(",", ":")).encode()
        ).decode().rstrip("=")
        
        # Create signature
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")
        
        return f"{message}.{signature_b64}"
    
    def _decode_token(self, token: str) -> TokenPayload:
        """
        Decode and verify a JWT token (simplified implementation)
        
        Note: This is a basic implementation for demonstration.
        In production, use a proper JWT library like PyJWT.
        """
        import base64
        
        try:
            # Split token
            parts = token.split(".")
            if len(parts) != 3:
                raise InvalidTokenError("Invalid token format")
            
            header_b64, payload_b64, signature_b64 = parts
            
            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
            
            # Add padding if needed
            padding = 4 - len(signature_b64) % 4
            if padding != 4:
                signature_b64 += "=" * padding
            
            actual_signature = base64.urlsafe_b64decode(signature_b64)
            
            if not hmac.compare_digest(expected_signature, actual_signature):
                raise InvalidTokenError("Invalid signature")
            
            # Decode payload
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            
            payload_json = base64.urlsafe_b64decode(payload_b64).decode()
            payload_data = json.loads(payload_json)
            
            payload = TokenPayload.from_dict(payload_data)
            
            # Check if token is revoked
            if payload.jti in self.revoked_tokens:
                raise InvalidTokenError("Token has been revoked")
            
            # Check expiry
            if payload.is_expired():
                raise TokenExpiredError(f"Token expired at {payload.expires_at}")
            
            return payload
            
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            raise InvalidTokenError(f"Failed to decode token: {e}")
    
    def create_access_token(
        self,
        user_id: str,
        scope: Optional[str] = None
    ) -> Tuple[str, TokenPayload]:
        """
        Create an access token for a user
        
        Args:
            user_id: User identifier
            scope: Optional scope for the token
            
        Returns:
            Tuple of (token_string, token_payload)
        """
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
    
    def create_refresh_token(
        self,
        user_id: str
    ) -> Tuple[str, TokenPayload]:
        """
        Create a refresh token for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (token_string, token_payload)
        """
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
        """
        Verify and decode a token
        
        Args:
            token: JWT token string
            
        Returns:
            TokenPayload if valid
            
        Raises:
            TokenExpiredError: If token is expired
            InvalidTokenError: If token is invalid
        """
        return self._decode_token(token)
    
    def revoke_token(self, token: str) -> None:
        """
        Revoke a token
        
        Args:
            token: JWT token string to revoke
        """
        try:
            payload = self._decode_token(token)
            self.revoked_tokens.add(payload.jti)
            log.info(f"Revoked token {payload.jti} for user {payload.user_id}")
        except (TokenExpiredError, InvalidTokenError):
            # Already invalid, just log
            log.warning("Attempted to revoke invalid/expired token")
    
    def refresh_access_token(self, refresh_token: str) -> Tuple[str, TokenPayload]:
        """
        Create a new access token from a refresh token
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Tuple of (new_access_token, token_payload)
            
        Raises:
            InvalidTokenError: If refresh token is invalid or not a refresh token
        """
        payload = self.verify_token(refresh_token)
        
        if payload.token_type != TokenType.REFRESH:
            raise InvalidTokenError("Token is not a refresh token")
        
        return self.create_access_token(payload.user_id, payload.scope)
    
    def create_api_key(
        self,
        user_id: str,
        name: str,
        expires_at: Optional[datetime] = None
    ) -> Tuple[str, APIKey]:
        """
        Create an API key for a user
        
        Args:
            user_id: User identifier
            name: Human-readable name for the key
            expires_at: Optional expiration datetime
            
        Returns:
            Tuple of (api_key_string, api_key_object)
        """
        # Generate random API key
        key_id = secrets.token_urlsafe(8)
        key_secret = secrets.token_urlsafe(32)
        api_key_string = f"{key_id}.{key_secret}"
        
        # Hash the key for storage
        # Note: SHA256 is acceptable for high-entropy API keys (not passwords)
        # API keys are 32-byte random tokens with ~256 bits of entropy
        # For user passwords, use bcrypt/scrypt/argon2 instead
        key_hash = hashlib.sha256(api_key_string.encode()).hexdigest()
        
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
        """
        Verify an API key
        
        Args:
            api_key_string: API key string
            
        Returns:
            APIKey object if valid
            
        Raises:
            InvalidTokenError: If API key is invalid
        """
        try:
            # Parse key
            parts = api_key_string.split(".")
            if len(parts) != 2:
                raise InvalidTokenError("Invalid API key format")
            
            key_id, _ = parts
            
            # Look up key
            api_key = self.api_keys.get(key_id)
            if not api_key:
                raise InvalidTokenError("API key not found")
            
            # Verify hash
            key_hash = hashlib.sha256(api_key_string.encode()).hexdigest()
            if not hmac.compare_digest(api_key.key_hash, key_hash):
                raise InvalidTokenError("Invalid API key")
            
            # Check validity
            if not api_key.is_valid():
                raise InvalidTokenError("API key is disabled or expired")
            
            # Update last used
            api_key.last_used_at = datetime.now(timezone.utc)
            
            log.info(f"Verified API key (ID: {key_id}) for user {api_key.user_id}")
            return api_key
            
        except (ValueError, AttributeError) as e:
            raise InvalidTokenError(f"Failed to verify API key: {e}")
    
    def revoke_api_key(self, key_id: str) -> None:
        """
        Revoke an API key
        
        Args:
            key_id: API key identifier
        """
        api_key = self.api_keys.get(key_id)
        if api_key:
            api_key.enabled = False
            log.info(f"Revoked API key {key_id}")
        else:
            log.warning(f"Attempted to revoke non-existent API key {key_id}")
    
    def list_api_keys(self, user_id: Optional[str] = None) -> list[APIKey]:
        """
        List API keys, optionally filtered by user
        
        Args:
            user_id: Optional user identifier to filter by
            
        Returns:
            List of APIKey objects
        """
        keys = list(self.api_keys.values())
        if user_id:
            keys = [k for k in keys if k.user_id == user_id]
        return keys
    
    def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired tokens from revoked list
        
        Returns:
            Number of tokens cleaned up
        """
        # This is a simplified cleanup
        # In production, you'd track token expiry times
        # For now, we just clear the set periodically
        count = len(self.revoked_tokens)
        self.revoked_tokens.clear()
        return count


# Global auth manager instance
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get or create the global auth manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def set_auth_manager(manager: AuthManager) -> None:
    """Set the global auth manager instance"""
    global _auth_manager
    _auth_manager = manager


def authenticate_request(
    authorization_header: Optional[str] = None,
    api_key_header: Optional[str] = None
) -> str:
    """
    Authenticate a request using either JWT token or API key
    
    Args:
        authorization_header: Authorization header value (Bearer token)
        api_key_header: API key header value
        
    Returns:
        User ID if authenticated
        
    Raises:
        AuthenticationError: If authentication fails
    """
    auth_manager = get_auth_manager()
    
    # Try JWT token first
    if authorization_header:
        if not authorization_header.startswith("Bearer "):
            raise AuthenticationError("Invalid authorization header format")
        
        token = authorization_header[7:]  # Remove "Bearer " prefix
        try:
            payload = auth_manager.verify_token(token)
            return payload.user_id
        except (TokenExpiredError, InvalidTokenError) as e:
            raise AuthenticationError(f"Token authentication failed: {e}")
    
    # Try API key
    if api_key_header:
        try:
            api_key = auth_manager.verify_api_key(api_key_header)
            return api_key.user_id
        except InvalidTokenError as e:
            raise AuthenticationError(f"API key authentication failed: {e}")
    
    raise AuthenticationError("No authentication credentials provided")
