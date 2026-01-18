"""Authentication routes for API v1.

Provides login and token management endpoints.

Endpoints:
- POST /api/v1/auth/login - Login and get access token
- POST /api/v1/auth/refresh - Refresh access token
"""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from nethical.api.rbac import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_password_hash,
    verify_password,
)
from nethical.database import User, get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    """Login request."""
    
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Token response."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    """Login and get access token.
    
    Args:
        credentials: Username and password
        db: Database session
        
    Returns:
        Access token and user information
        
    Raises:
        HTTPException: 401 if credentials are invalid
    """
    # Get user from database
    user = db.query(User).filter(User.username == credentials.username).first()
    
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Create access token
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "scopes": [],
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        }
    )


@router.post("/register", response_model=dict, include_in_schema=False)
async def register(
    credentials: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
    email: str = "user@example.com",
    full_name: str = "User",
    role: str = "operator",
) -> dict:
    """Register a new user (development only, hidden from schema).
    
    Args:
        credentials: Username and password
        db: Database session
        email: Email address
        full_name: Full name
        role: User role
        
    Returns:
        Created user information
        
    Raises:
        HTTPException: 409 if username already exists
    """
    # Check if user exists
    existing = db.query(User).filter(User.username == credentials.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists"
        )
    
    # Create user
    user = User(
        username=credentials.username,
        email=email,
        full_name=full_name,
        hashed_password=get_password_hash(credentials.password),
        role=role,
        is_active=True,
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user.to_dict()
