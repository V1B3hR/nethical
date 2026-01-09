"""Role-Based Access Control (RBAC) module.

Provides JWT authentication and authorization with role-based access control.

Roles:
- admin: Full access to all operations
- auditor: Read-only access to logs and audit data
- operator: Can evaluate risk, but cannot modify configuration
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Annotated, Any, Callable, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel

__all__ = [
    "Role",
    "TokenData",
    "create_access_token",
    "get_current_user",
    "require_role",
    "verify_password",
    "get_password_hash",
]

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.getenv("NETHICAL_SECRET_KEY", "development-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# HTTP Bearer security scheme
security = HTTPBearer()


class Role(str, Enum):
    """User roles for RBAC."""
    
    ADMIN = "admin"
    AUDITOR = "auditor"
    OPERATOR = "operator"


class TokenData(BaseModel):
    """Token payload data."""
    
    username: str
    role: Role
    scopes: list[str] = []


class User(BaseModel):
    """User model for authentication."""
    
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    role: Role
    is_active: bool = True


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


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


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> User:
    """Get current user from JWT token.
    
    Args:
        credentials: HTTP bearer credentials
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        
        if username is None or role is None:
            raise credentials_exception
        
        token_data = TokenData(
            username=username,
            role=Role(role),
            scopes=payload.get("scopes", [])
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (jwt.InvalidTokenError, ValueError):
        raise credentials_exception
    
    # In production, fetch user from database
    # For now, return user from token data
    user = User(
        id=payload.get("user_id", 0),
        username=token_data.username,
        email=payload.get("email", f"{token_data.username}@example.com"),
        full_name=payload.get("full_name"),
        role=token_data.role,
        is_active=True
    )
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user


def require_role(*allowed_roles: Role) -> Callable:
    """Decorator to require specific role(s) for endpoint access.
    
    Args:
        *allowed_roles: Roles that are allowed to access the endpoint
        
    Returns:
        Decorated function
        
    Example:
        @app.get("/admin")
        @require_role(Role.ADMIN)
        async def admin_endpoint(user: User = Depends(get_current_user)):
            return {"message": "Admin access granted"}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from kwargs (injected by FastAPI)
            user = kwargs.get("current_user")
            
            if user is None:
                # Try to get from args if not in kwargs
                for arg in args:
                    if isinstance(arg, User):
                        user = arg
                        break
            
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if user.role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required role: {', '.join(r.value for r in allowed_roles)}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Dependency for role checking in FastAPI
def require_admin(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Require admin role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_auditor_or_admin(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Require auditor or admin role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user if auditor or admin
        
    Raises:
        HTTPException: If user is not auditor or admin
    """
    if current_user.role not in [Role.ADMIN, Role.AUDITOR]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Auditor or admin access required"
        )
    return current_user
