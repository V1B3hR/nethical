# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Authentication routes for API v1.

Provides login, MFA management, token refresh, and logout/revocation endpoints.

Endpoints:
- POST /api/v1/auth/login - Login and get access token (with MFA challenge support)
- POST /api/v1/auth/mfa/setup - Initialize TOTP MFA setup
- POST /api/v1/auth/mfa/verify - Verify and activate TOTP MFA
- POST /api/v1/auth/mfa/disable - Disable MFA
- POST /api/v1/auth/logout - Revoke active access token
- POST /api/v1/auth/register - Register a new user
"""

from __future__ import annotations

import secrets
from datetime import timedelta, timezone
from typing import Annotated, Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from nethical.api.rbac import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALGORITHM,
    SECRET_KEY,
    User as RBACUser,
    create_access_token,
    get_current_user,
    get_password_hash,
    revoke_token,
    security,
    verify_password,
)
from nethical.database import User, get_db
from nethical.security.mfa import MFAManager, MFASetup, MFAMethod

router = APIRouter(prefix="/auth", tags=["Authentication"])
_mfa_manager = MFAManager()


class LoginRequest(BaseModel):
    """Login request."""
    
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    totp_code: Optional[str] = Field(None, description="6-digit TOTP code if MFA is enabled")
    mfa_token: Optional[str] = Field(None, description="Temporary MFA session token")


class TokenResponse(BaseModel):
    """Token response."""
    
    access_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    user: Optional[dict] = None
    mfa_required: bool = False
    mfa_token: Optional[str] = None


class MFASetupResponse(BaseModel):
    """MFA Setup response containing TOTP secret, QR code URI, and backup codes."""
    
    totp_secret: str
    provisioning_uri: str
    backup_codes: list[str]


class MFAVerifyRequest(BaseModel):
    """MFA verification and activation request."""
    
    totp_code: str = Field(..., description="6-digit TOTP code from authenticator app")
    totp_secret: Optional[str] = Field(None, description="TOTP secret received during setup")


class MFADisableRequest(BaseModel):
    """Request to disable MFA."""
    
    password: str = Field(..., description="Current user password for confirmation")


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    """Login and get access token with seamless MFA challenge handling.
    
    Args:
        credentials: Username, password, and optional TOTP code / MFA token
        db: Database session
        
    Returns:
        Access token and user info, or MFA challenge if 2FA is active
    """
    user: Optional[User] = None

    # Scenario A: Resolving via temporary mfa_token
    if credentials.mfa_token and credentials.totp_code:
        try:
            mfa_payload = jwt.decode(credentials.mfa_token, SECRET_KEY, algorithms=[ALGORITHM])
            if mfa_payload.get("token_type") != "mfa_pending":
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid MFA token")
            username = mfa_payload.get("sub")
            user = db.query(User).filter(User.username == username).first()
        except jwt.PyJWTError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="MFA token expired or invalid")
    else:
        # Scenario B: Standard username + password validation
        user = db.query(User).filter(User.username == credentials.username).first()
        if not user or not verify_password(credentials.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Check if MFA is required for this account
    if user.mfa_enabled:
        if not credentials.totp_code:
            # Generate short-lived MFA challenge token (valid 5 minutes)
            mfa_token = create_access_token(
                data={
                    "sub": user.username,
                    "user_id": user.id,
                    "token_type": "mfa_pending",
                },
                expires_delta=timedelta(minutes=5),
            )
            return TokenResponse(
                access_token=None,
                token_type="mfa_challenge",
                expires_in=300,
                user={"username": user.username},
                mfa_required=True,
                mfa_token=mfa_token,
            )

        # Validate TOTP code
        _mfa_manager.user_mfa[user.username] = MFASetup(
            user_id=user.username,
            enabled=True,
            totp_secret=user.mfa_secret,
            backup_codes=user.mfa_backup_codes or [],
            methods=[MFAMethod.TOTP],
        )
        is_valid = _mfa_manager.verify_totp(user.username, credentials.totp_code)
        if not is_valid and user.mfa_backup_codes:
            is_valid = _mfa_manager.verify_backup_code(user.username, credentials.totp_code)

        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid MFA code",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Issue Full JWT Access Token
    access_token = create_access_token(
        data={
            "sub": user.username,
            "user_id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "tenant_id": getattr(user, "tenant_id", "default_tenant"),
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
            "tenant_id": getattr(user, "tenant_id", "default_tenant"),
            "mfa_enabled": user.mfa_enabled,
        }
    )


@router.post("/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(
    current_user: Annotated[RBACUser, Depends(get_current_user)],
) -> MFASetupResponse:
    """Initialize TOTP MFA setup for current authenticated user."""
    totp_secret, provisioning_uri, backup_codes = _mfa_manager.setup_totp(
        user_id=current_user.username,
        issuer="Nethical AI Governance",
    )
    return MFASetupResponse(
        totp_secret=totp_secret,
        provisioning_uri=provisioning_uri,
        backup_codes=backup_codes,
    )


@router.post("/mfa/verify", response_model=dict)
async def verify_and_enable_mfa(
    request: MFAVerifyRequest,
    current_user: Annotated[RBACUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Verify first TOTP code and activate MFA on the user account."""
    user = db.query(User).filter(User.username == current_user.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    secret = request.totp_secret or getattr(user, "mfa_secret", None)
    if not secret:
        # Check in MFAManager
        if current_user.username in _mfa_manager.user_mfa:
            secret = _mfa_manager.user_mfa[current_user.username].totp_secret

    if not secret:
        raise HTTPException(status_code=400, detail="MFA setup has not been initiated. Call /mfa/setup first.")

    # Configure MFAManager for verification
    _mfa_manager.user_mfa[user.username] = MFASetup(
        user_id=user.username,
        enabled=True,
        totp_secret=secret,
        methods=[MFAMethod.TOTP],
    )
    
    if not _mfa_manager.verify_totp(user.username, request.totp_code):
        raise HTTPException(status_code=400, detail="Invalid verification code")

    # Persist MFA activation in DB
    user.mfa_enabled = True
    user.mfa_secret = secret
    if current_user.username in _mfa_manager.user_mfa:
        user.mfa_backup_codes = _mfa_manager.user_mfa[current_user.username].backup_codes

    db.commit()
    db.refresh(user)

    return {
        "status": "mfa_enabled",
        "message": "Two-factor authentication successfully verified and enabled.",
        "mfa_enabled": True,
    }


@router.post("/mfa/disable", response_model=dict)
async def disable_mfa(
    request: MFADisableRequest,
    current_user: Annotated[RBACUser, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    """Disable MFA on user account after verifying current password."""
    user = db.query(User).filter(User.username == current_user.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect password")

    user.mfa_enabled = False
    user.mfa_secret = None
    user.mfa_backup_codes = []
    _mfa_manager.disable_mfa(user.username)

    db.commit()
    db.refresh(user)

    return {
        "status": "mfa_disabled",
        "message": "Two-factor authentication has been disabled.",
        "mfa_enabled": False,
    }


@router.post("/logout", response_model=dict)
async def logout(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> dict:
    """Logout and revoke the provided JWT token permanently."""
    token = credentials.credentials
    revoke_token(token, reason="user_logout")
    return {
        "status": "logged_out",
        "message": "Access token has been successfully revoked and blacklisted.",
    }


@router.post("/register", response_model=dict, include_in_schema=False)
async def register(
    credentials: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
    email: str = "user@example.com",
    full_name: str = "User",
    role: str = "operator",
    tenant_id: str = "default_tenant",
) -> dict:
    """Register a new user (development only, hidden from schema)."""
    existing = db.query(User).filter(User.username == credentials.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists"
        )
    
    user = User(
        username=credentials.username,
        email=email,
        full_name=full_name,
        hashed_password=get_password_hash(credentials.password),
        role=role,
        tenant_id=tenant_id,
        is_active=True,
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user.to_dict()

