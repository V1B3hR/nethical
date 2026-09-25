# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Tests for API v1 Multi-Factor Authentication (MFA) and Token Revocation."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from nethical.api.v1.app import create_v1_app
from nethical.database import Base, get_db, User
from nethical.api.rbac import get_password_hash
from nethical.security.mfa import MFAManager

TEST_DB_URL = "sqlite:///./test_mfa_auth.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
async def mfa_client():
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    app = create_v1_app()
    app.dependency_overrides[get_db] = override_get_db

    # Create baseline user
    db = TestSessionLocal()
    user = User(
        username="mfa_admin",
        email="mfa_admin@nethical.local",
        full_name="MFA Test Admin",
        hashed_password=get_password_hash("SecretP@ssword123!"),
        role="admin",
        tenant_id="gov_pl_cyber",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.close()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    Base.metadata.drop_all(bind=test_engine)


@pytest.mark.asyncio
async def test_standard_login_without_mfa(mfa_client):
    """Test standard password authentication when MFA is disabled."""
    response = await mfa_client.post(
        "/auth/login",
        json={"username": "mfa_admin", "password": "SecretP@ssword123!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["access_token"] is not None
    assert data["mfa_required"] is False
    assert data["user"]["username"] == "mfa_admin"


@pytest.mark.asyncio
async def test_mfa_setup_and_verify_lifecycle(mfa_client):
    """Test setting up TOTP, generating secret, and verifying to activate MFA."""
    # 1. Login to get initial auth token
    login_res = await mfa_client.post(
        "/auth/login",
        json={"username": "mfa_admin", "password": "SecretP@ssword123!"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Call MFA setup endpoint
    setup_res = await mfa_client.post("/auth/mfa/setup", headers=headers)
    assert setup_res.status_code == 200
    setup_data = setup_res.json()
    assert "totp_secret" in setup_data
    assert "provisioning_uri" in setup_data
    assert len(setup_data["backup_codes"]) > 0

    secret = setup_data["totp_secret"]

    # 3. Generate valid TOTP code
    import pyotp
    totp = pyotp.TOTP(secret)
    valid_code = totp.now()

    # 4. Verify & Enable MFA
    verify_res = await mfa_client.post(
        "/auth/mfa/verify",
        headers=headers,
        json={"totp_code": valid_code, "totp_secret": secret},
    )
    assert verify_res.status_code == 200
    assert verify_res.json()["status"] == "mfa_enabled"

    # 5. Now try logging in with password only -> expect MFA challenge
    challenge_res = await mfa_client.post(
        "/auth/login",
        json={"username": "mfa_admin", "password": "SecretP@ssword123!"},
    )
    assert challenge_res.status_code == 200
    challenge_data = challenge_res.json()
    assert challenge_data["mfa_required"] is True
    assert challenge_data["access_token"] is None
    assert challenge_data["mfa_token"] is not None

    mfa_session_token = challenge_data["mfa_token"]

    # 6. Complete login using mfa_token + TOTP code
    current_code = totp.now()
    final_login_res = await mfa_client.post(
        "/auth/login",
        json={
            "username": "mfa_admin",
            "password": "SecretP@ssword123!",
            "mfa_token": mfa_session_token,
            "totp_code": current_code,
        },
    )
    assert final_login_res.status_code == 200
    final_data = final_login_res.json()
    assert final_data["access_token"] is not None
    assert final_data["user"]["mfa_enabled"] is True


@pytest.mark.asyncio
async def test_token_revocation_on_logout(mfa_client):
    """Test that calling /logout permanently revokes the access token."""
    # 1. Login
    login_res = await mfa_client.post(
        "/auth/login",
        json={"username": "mfa_admin", "password": "SecretP@ssword123!"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Access protected endpoint before logout
    agents_res = await mfa_client.get("/agents", headers=headers)
    assert agents_res.status_code == 200

    # 3. Call logout to revoke token
    logout_res = await mfa_client.post("/auth/logout", headers=headers)
    assert logout_res.status_code == 200
    assert logout_res.json()["status"] == "logged_out"

    # 4. Attempt to access protected endpoint with the revoked token -> must fail with 401
    revoked_access_res = await mfa_client.get("/agents", headers=headers)
    assert revoked_access_res.status_code == 401
    assert "revoked" in revoked_access_res.json()["detail"].lower()

