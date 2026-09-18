# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Unit tests for Nethical Multi-Tenancy, RBAC, and Sovereign Air-Gapped Authentication."""

import pytest
import time
from nethical.core.models import (
    ActionType,
    AgentAction,
    ClassificationLevel,
    TenantConfig,
    UserIdentity,
    UserRole,
)
from nethical.auth.tenant_manager import TenantManager
from nethical.auth.rbac import (
    RBACManager,
    SovereignAuthToken,
    PERM_AUDIT_EXPORT,
    PERM_AUDIT_VERIFY,
    PERM_KILL_SWITCH,
    PERM_MANAGE_TENANTS,
    PERM_MANAGE_USERS,
    PERM_REVIEW_HITL,
)


def test_tenant_seeding_and_isolation() -> None:
    """Verifies that default sovereign and enterprise tenants are seeded and ledgers are strictly isolated."""
    tm = TenantManager()
    tenants = tm.list_tenants()
    tenant_ids = [t.tenant_id for t in tenants]

    assert "default_tenant" in tenant_ids
    assert "gov_pl_cyber" in tenant_ids
    assert "enterprise_fin_eu" in tenant_ids
    assert "defense_airgap" in tenant_ids

    # Verify ledgers are distinct instances
    gov_ledger = tm.get_tenant_ledger("gov_pl_cyber")
    fin_ledger = tm.get_tenant_ledger("enterprise_fin_eu")
    assert gov_ledger is not fin_ledger

    # Add transaction to Gov ledger
    action_gov = AgentAction(
        agent_id="gov_sentinel",
        action_type=ActionType.FUNCTION_CALL,
        parameters={"interface": "ksc_edge_0"},
        tenant_id="gov_pl_cyber",
    )
    gov_receipt = gov_ledger.append_decision(action_gov.model_dump())
    assert gov_receipt.receipt_id is not None
    assert gov_ledger.total_blocks > 0
    assert fin_ledger.total_blocks == 0  # Strict cryptographic boundary


def test_tenant_creation_and_custom_policies() -> None:
    """Verifies dynamic creation of a sovereign tenant with custom regulatory framework."""
    tm = TenantManager()
    new_tenant = tm.create_tenant(
        name="Singapurski Węzeł AI FinTech",
        jurisdiction="SG",
        classification=ClassificationLevel.CONFIDENTIAL,
        allowed_frameworks=["MAS_FEAT", "ISO_42001"],
        metadata={"sovereign_enclave": "intel_sgx_v2"},
    )
    assert new_tenant.tenant_id is not None
    assert new_tenant.jurisdiction == "SG"
    assert new_tenant.classification_level == ClassificationLevel.CONFIDENTIAL
    assert "MAS_FEAT" in new_tenant.allowed_frameworks

    retrieved = tm.get_tenant(new_tenant.tenant_id)
    assert retrieved is not None
    assert retrieved.name == "Singapurski Węzeł AI FinTech"

    # Ledger created automatically
    ledger = tm.get_tenant_ledger(new_tenant.tenant_id)
    assert ledger is not None


def test_sovereign_auth_token_lifecycle() -> None:
    """Verifies generation, verification, and tamper detection of sovereign HMAC-SHA256 tokens."""
    token_service = SovereignAuthToken(secret_key=b"test-secret-key-32-bytes-minimum!!")
    user = UserIdentity(
        username="secops_lead",
        role=UserRole.SECURITY_OFFICER,
        tenant_id="gov_pl_cyber",
    )

    # 1. Valid token
    token_str = token_service.generate_token(user, expires_in_seconds=3600)
    assert isinstance(token_str, str)
    assert len(token_str.split(".")) == 3

    payload = token_service.verify_token(token_str)
    assert payload is not None
    assert payload["username"] == "secops_lead"
    assert payload["role"] == UserRole.SECURITY_OFFICER.value
    assert payload["tenant_id"] == "gov_pl_cyber"

    # 2. Tampered token
    parts = token_str.split(".")
    tampered_token = f"{parts[0]}.{parts[1]}.tamperedSignatureHere"
    assert token_service.verify_token(tampered_token) is None

    # 3. Expired token
    expired_token = token_service.generate_token(user, expires_in_seconds=-10)
    assert token_service.verify_token(expired_token) is None


def test_production_rbac_unseeded_fails_secure() -> None:
    """Verifies that in default production mode, RBACManager starts unseeded with zero accounts."""
    rbac = RBACManager(auto_seed_dev=False)
    assert rbac.is_bootstrapped() is False
    assert len(rbac.list_users()) == 0
    # Authentication fails for any user
    assert rbac.authenticate("admin", "nethical_admin_sovereign_password") is None
    assert rbac.authenticate_api_key("sk-nethical-admin-global-mesh-token") is None


def test_production_rbac_bootstrap_admin() -> None:
    """Verifies production bootstrapping of initial administrator with high-entropy credentials."""
    rbac = RBACManager(auto_seed_dev=False)
    user, pwd, api_key = rbac.bootstrap_admin(username="enterprise_ciso", tenant_id="corp_hq")
    assert rbac.is_bootstrapped() is True
    assert user.username == "enterprise_ciso"
    assert user.role == UserRole.GLOBAL_ADMIN
    assert len(pwd) >= 20  # High entropy generated password
    assert api_key.startswith("sk-sovereign-")

    # Authenticate with bootstrapped credentials
    auth_result = rbac.authenticate("enterprise_ciso", pwd)
    assert auth_result is not None
    assert rbac.authenticate_api_key(api_key) is not None

    # Cannot bootstrap twice
    with pytest.raises(RuntimeError, match="already bootstrapped"):
        rbac.bootstrap_admin(username="attacker")


def test_rbac_authentication_and_permissions() -> None:
    """Verifies RBAC authentication, password verification, and permission boundaries."""
    rbac = RBACManager(auto_seed_dev=True)

    # 1. Successful authentication with seeded credentials
    auth_result = rbac.authenticate("admin", "nethical_admin_sovereign_password")
    assert auth_result is not None
    admin_user, admin_token = auth_result
    assert admin_user.role == UserRole.GLOBAL_ADMIN
    assert admin_user.tenant_id == "default_tenant"

    # 2. Failed authentication with invalid password
    failed_result = rbac.authenticate("admin", "wrong_password")
    assert failed_result is None

    # 3. Non-existent user
    assert rbac.authenticate("ghost_user", "password") is None

    # 4. Role Permissions
    assert rbac.has_permission(admin_user, PERM_KILL_SWITCH) is True
    assert rbac.has_permission(admin_user, PERM_MANAGE_TENANTS) is True
    assert rbac.has_permission(admin_user, PERM_MANAGE_USERS) is True

    # Auditor role permissions
    auditor_auth = rbac.authenticate("auditor_anna", "auditor_sovereign_password")
    assert auditor_auth is not None
    auditor_user, _ = auditor_auth
    assert auditor_user.role == UserRole.COMPLIANCE_AUDITOR
    assert rbac.has_permission(auditor_user, PERM_AUDIT_VERIFY) is True
    assert rbac.has_permission(auditor_user, PERM_AUDIT_EXPORT) is True
    assert rbac.has_permission(auditor_user, PERM_KILL_SWITCH) is False
    assert rbac.has_permission(auditor_user, PERM_MANAGE_USERS) is False

    # HITL Reviewer role permissions
    hitl_auth = rbac.authenticate("reviewer_jan", "reviewer_sovereign_password")
    assert hitl_auth is not None
    hitl_user, _ = hitl_auth
    assert hitl_user.role == UserRole.HITL_REVIEWER
    assert rbac.has_permission(hitl_user, PERM_REVIEW_HITL) is True
    assert rbac.has_permission(hitl_user, PERM_KILL_SWITCH) is True
    assert rbac.has_permission(hitl_user, PERM_MANAGE_TENANTS) is False


def test_rbac_tenant_access_control() -> None:
    """Verifies multi-tenant access boundaries: non-admins cannot access foreign tenant workspaces."""
    rbac = RBACManager(auto_seed_dev=True)

    admin = rbac.get_user("admin")
    secops = rbac.get_user("secops_lead")
    operator = rbac.get_user("operator_kris")

    assert admin is not None
    assert secops is not None
    assert operator is not None

    # Global admin can access any tenant
    assert rbac.can_access_tenant(admin, "gov_pl_cyber") is True
    assert rbac.can_access_tenant(admin, "enterprise_fin_eu") is True
    assert rbac.can_access_tenant(admin, "defense_airgap") is True

    # Secops belongs to gov_pl_cyber only
    assert rbac.can_access_tenant(secops, "gov_pl_cyber") is True
    assert rbac.can_access_tenant(secops, "enterprise_fin_eu") is False
    assert rbac.can_access_tenant(secops, "defense_airgap") is False

    # Operator belongs to defense_airgap only
    assert rbac.can_access_tenant(operator, "defense_airgap") is True
    assert rbac.can_access_tenant(operator, "gov_pl_cyber") is False


def test_api_key_authentication() -> None:
    """Verifies authentication via pre-shared sovereign API keys."""
    rbac = RBACManager(auto_seed_dev=True)
    user = rbac.authenticate_api_key("sk-nethical-admin-global-mesh-token")
    assert user is not None
    assert user.username == "admin"
    assert user.role == UserRole.GLOBAL_ADMIN

    assert rbac.authenticate_api_key("invalid-token-key") is None


def test_api_endpoints_auth_and_multitenancy() -> None:
    """Verifies FastAPI endpoints for bootstrap, login, profile, multi-tenancy, and isolated ledger queries."""
    from fastapi.testclient import TestClient
    from nethical.api import app, rbac_manager_instance

    with TestClient(app) as client:
        # 0. Test Bootstrap endpoint if not already bootstrapped
        if not rbac_manager_instance.is_bootstrapped():
            boot_resp = client.post(
                "/api/v1/auth/bootstrap",
                json={"username": "admin", "password": "nethical_admin_sovereign_password"},
            )
            assert boot_resp.status_code == 200
            assert boot_resp.json()["status"] == "BOOTSTRAP_SUCCESS"

            # Subsequent bootstrap attempts must be forbidden (HTTP 403)
            second_boot = client.post(
                "/api/v1/auth/bootstrap",
                json={"username": "attacker", "password": "hacked"},
            )
            assert second_boot.status_code == 403

        # 0b. Verify auth status endpoint
        status_resp = client.get("/api/v1/auth/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["bootstrapped"] is True

        # 1. Login endpoint with valid sovereign admin credentials
        login_resp = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "nethical_admin_sovereign_password"},
        )
        assert login_resp.status_code == 200
        login_data = login_resp.json()
        assert "access_token" in login_data
        assert login_data["user"]["username"] == "admin"
        assert login_data["user"]["role"] == "global_admin"
        token = login_data["access_token"]

    # 2. Login with bad credentials
    bad_login = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "wrong_password"},
    )
    assert bad_login.status_code == 401

    # 3. GET /api/v1/auth/me with Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    me_resp = client.get("/api/v1/auth/me", headers=headers)
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["username"] == "admin"
    assert "security:kill_switch" in me_data["permissions"]

    # 4. GET /api/v1/tenants
    tenants_resp = client.get("/api/v1/tenants", headers=headers)
    assert tenants_resp.status_code == 200
    tenants_list = tenants_resp.json()
    t_ids = [t["tenant_id"] for t in tenants_list]
    assert "default_tenant" in t_ids
    assert "gov_pl_cyber" in t_ids
    assert "enterprise_fin_eu" in t_ids

    # 5. POST /api/v1/tenants (create new workspace)
    new_tenant_resp = client.post(
        "/api/v1/tenants",
        headers=headers,
        json={
            "name": "Krajowa Sieć Kardiologiczna AI",
            "jurisdiction": "PL",
            "classification": "confidential",
            "allowed_frameworks": ["HEALTHCARE_MED", "EU_AI_ACT"],
        },
    )
    assert new_tenant_resp.status_code == 200
    assert new_tenant_resp.json()["name"] == "Krajowa Sieć Kardiologiczna AI"

    # 6. GET /api/v1/tenants/{tenant_id}/ledger
    ledger_resp = client.get("/api/v1/tenants/gov_pl_cyber/ledger")
    assert ledger_resp.status_code == 200
    ledger_data = ledger_resp.json()
    assert ledger_data["tenant_id"] == "gov_pl_cyber"
    assert ledger_data["pqc_algorithm"] == "ML-DSA-65 (CRYSTALS-Dilithium Level 3)"
    assert ledger_data["integrity_valid"] is True

    # 7. GET /api/v1/portal/stats includes multitenancy
    stats_resp = client.get("/api/v1/portal/stats")
    assert stats_resp.status_code == 200
    stats_data = stats_resp.json()
    assert "multitenancy" in stats_data
    assert stats_data["multitenancy"]["active_tenants_count"] >= 4

    # 8. GET /portal serves the multi-tenancy UI and offline typography
    portal_resp = client.get("/portal")
    assert portal_resp.status_code == 200
    assert "tenantSelect" in portal_resp.text
    assert "AIR-GAP READY" in portal_resp.text
    assert "authModal" in portal_resp.text
    assert "fonts.googleapis.com" not in portal_resp.text  # Zero external CDN calls in air-gap


