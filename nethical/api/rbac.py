# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Role-Based Access Control (RBAC) module.

Provides JWT authentication and authorization with role-based access control.

Roles:
- admin: Full access to all operations
- auditor: Read-only access to logs and audit data
- operator: Can evaluate risk, but cannot modify configuration
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Annotated, Any, Callable, Optional

try:
    import bcrypt
except ImportError:
    bcrypt = None

try:
    import jwt
except ImportError:
    jwt = None

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

__all__ = [
    "Role",
    "ADMIN_ROLES",
    "AUDITOR_ROLES",
    "OPERATOR_ROLES",
    "TokenData",
    "User",
    "create_access_token",
    "get_current_user",
    "require_role",
    "require_admin",
    "require_auditor_or_admin",
    "verify_password",
    "get_password_hash",
    "revoke_token",
    "is_token_revoked",
    "_initialize_secret_key",
]

# HTTP Bearer security scheme
security = HTTPBearer()

INSECURE_SECRET_KEYS = {
    "development-secret-key-change-in-production",
    "secret",
    "changeme",
    "change-me",
    "default",
    "",
}


def _initialize_secret_key() -> str:
    """Initialize JWT secret key with strict boundary defense for production/staging.

    In production/staging environments, execution halts immediately with RuntimeError
    if NETHICAL_SECRET_KEY is omitted or set to an insecure default.
    In development environments, an ephemeral high-entropy key is generated if unset.
    """
    env = os.getenv("NETHICAL_ENV", os.getenv("ENVIRONMENT", "development")).lower()
    raw_secret = os.getenv("NETHICAL_SECRET_KEY")

    if env in ("production", "staging", "prod"):
        if not raw_secret or raw_secret in INSECURE_SECRET_KEYS:
            raise RuntimeError(
                "CRITICAL SECURITY DEFENSE: NETHICAL_SECRET_KEY must be set to a cryptographically secure "
                "value in production/staging environments. Startup aborted to prevent unauthorized access."
            )
        return raw_secret

    if not raw_secret or raw_secret in INSECURE_SECRET_KEYS:
        logger.warning(
            "SECURITY WARNING: NETHICAL_SECRET_KEY is unset or using a default key in non-production. "
            "Generating ephemeral high-entropy runtime secret."
        )
        return secrets.token_hex(32)

    return raw_secret


# JWT configuration
SECRET_KEY = _initialize_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


class Role(str, Enum):
    """User roles for RBAC supporting both canonical UserRole and legacy aliases."""
    
    # Legacy short names
    ADMIN = "admin"
    AUDITOR = "auditor"
    OPERATOR = "operator"

    # Canonical UserRole names
    GLOBAL_ADMIN = "global_admin"
    SECURITY_OFFICER = "security_officer"
    COMPLIANCE_AUDITOR = "compliance_auditor"
    HITL_REVIEWER = "hitl_reviewer"
    AGENT_OPERATOR = "agent_operator"

    @classmethod
    def normalize(cls, role_val: Any) -> "Role":
        """Normalize legacy or canonical role string into Role enum."""
        if isinstance(role_val, Role):
            return role_val
        val = str(role_val or "").strip().lower()
        mapping = {
            "admin": cls.ADMIN,
            "global_admin": cls.GLOBAL_ADMIN,
            "auditor": cls.AUDITOR,
            "compliance_auditor": cls.COMPLIANCE_AUDITOR,
            "security_officer": cls.SECURITY_OFFICER,
            "operator": cls.OPERATOR,
            "agent_operator": cls.AGENT_OPERATOR,
            "hitl_reviewer": cls.HITL_REVIEWER,
        }
        return mapping.get(val, cls.OPERATOR)


# Role hierarchy sets for access checking
ADMIN_ROLES = {Role.ADMIN, Role.GLOBAL_ADMIN, Role.SECURITY_OFFICER}
AUDITOR_ROLES = {Role.AUDITOR, Role.COMPLIANCE_AUDITOR, *ADMIN_ROLES}
OPERATOR_ROLES = {Role.OPERATOR, Role.AGENT_OPERATOR, Role.HITL_REVIEWER, *AUDITOR_ROLES}


class TokenData(BaseModel):
    """Token payload data."""
    
    username: str
    role: Role
    tenant_id: str = "default_tenant"
    scopes: list[str] = []


class User(BaseModel):
    """User model for authentication."""
    
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    role: Role
    tenant_id: str = "default_tenant"
    is_active: bool = True



def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password (bcrypt)
        
    Returns:
        True if password matches
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8') if isinstance(hashed_password, str) else hashed_password
        )
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Hash password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def create_access_token(data: dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token.
    
    Args:
        data: Token payload data
        expires_delta: Token expiration time
        
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


import hashlib

_in_memory_revoked: set[str] = set()


def is_token_revoked(identifier: str) -> bool:
    """Check if token hash or JTI is revoked in cache or database."""
    if identifier in _in_memory_revoked:
        return True
    try:
        from nethical.database import RevokedToken, SessionLocal
        if SessionLocal is not None and RevokedToken is not None:
            with SessionLocal() as db:
                entry = db.query(RevokedToken).filter(RevokedToken.jti == identifier).first()
                if entry:
                    _in_memory_revoked.add(identifier)
                    return True
    except Exception:
        pass
    return False


def revoke_token(token: str, reason: Optional[str] = None) -> bool:
    """Revoke token both in-memory and persistently in the database."""
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    _in_memory_revoked.add(token_hash)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
        jti = payload.get("jti", token_hash)
        _in_memory_revoked.add(jti)
        exp_ts = payload.get("exp")
        expires_at = datetime.fromtimestamp(exp_ts, tz=timezone.utc) if exp_ts else datetime.now(timezone.utc) + timedelta(hours=24)
        user_id = str(payload.get("user_id") or payload.get("sub") or "")

        from nethical.database import RevokedToken, SessionLocal
        if SessionLocal is not None and RevokedToken is not None:
            with SessionLocal() as db:
                existing = db.query(RevokedToken).filter(RevokedToken.jti == jti).first()
                if not existing:
                    rec = RevokedToken(
                        jti=jti,
                        token_type=payload.get("token_type", "access"),
                        user_id=user_id,
                        expires_at=expires_at,
                        reason=reason or "logout",
                    )
                    db.add(rec)
                    db.commit()
        return True
    except Exception as e:
        logger.warning(f"Error persisting revoked token: {e}")
        return True


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> User:
    """Get current user from JWT token with revocation checking.
    
    Args:
        credentials: HTTP bearer credentials
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If token is invalid, expired, or revoked
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        
        # Check revocation before heavy decoding
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if is_token_revoked(token_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        jti = payload.get("jti")
        if jti and is_token_revoked(jti):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )

        username: str = payload.get("sub")
        raw_role = payload.get("role")
        
        if username is None or raw_role is None:
            raise credentials_exception
        
        role = Role.normalize(raw_role)
        tenant_id = payload.get("tenant_id", "default_tenant")
        
        token_data = TokenData(
            username=username,
            role=role,
            tenant_id=tenant_id,
            scopes=payload.get("scopes", [])
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except (jwt.InvalidTokenError, ValueError):
        raise credentials_exception
    
    user = User(
        id=payload.get("user_id", 0),
        username=token_data.username,
        email=payload.get("email", f"{token_data.username}@example.com"),
        full_name=payload.get("full_name"),
        role=token_data.role,
        tenant_id=token_data.tenant_id,
        is_active=True
    )
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user


def require_role(*allowed_roles: Role) -> Callable:
    """Decorator to require specific role(s) for endpoint access."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("current_user")
            if user is None:
                for arg in args:
                    if isinstance(arg, User):
                        user = arg
                        break
            
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check if user role matches or is contained in allowed roles
            allowed_set = set(allowed_roles)
            if user.role not in allowed_set:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required role: {', '.join(r.value for r in allowed_roles)}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_admin(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Require admin role (supports legacy admin and canonical global_admin/security_officer)."""
    if current_user.role not in ADMIN_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_auditor_or_admin(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Require auditor or admin role (supports legacy and canonical roles)."""
    if current_user.role not in AUDITOR_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Auditor or admin access required"
        )
    return current_user

