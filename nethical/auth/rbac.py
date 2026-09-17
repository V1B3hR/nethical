# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Role-Based Access Control (RBAC) and Sovereign Air-Gapped Authentication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.models import UserIdentity, UserRole

logger = logging.getLogger("nethical.auth.rbac")

# Default secret key for token signing in local/air-gapped mode.
# In production, this can be overridden via NETHICAL_AUTH_SECRET environment variable.
_DEFAULT_SIGNING_SECRET = os.environ.get(
    "NETHICAL_AUTH_SECRET", "nethical-sovereign-auth-secret-key-airgap-production-grade"
).encode("utf-8")

# Permissions catalog
PERM_READ_TELEMETRY = "telemetry:read"
PERM_EXECUTE_ACTION = "action:execute"
PERM_REVIEW_HITL = "hitl:review"
PERM_KILL_SWITCH = "security:kill_switch"
PERM_MANAGE_POLICIES = "policy:manage"
PERM_AUDIT_VERIFY = "audit:verify"
PERM_AUDIT_EXPORT = "audit:export"
PERM_MANAGE_TENANTS = "tenant:manage"
PERM_MANAGE_USERS = "user:manage"
PERM_PQC_ROTATE = "pqc:rotate"

# Role to Permissions Mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[str]] = {
    UserRole.GLOBAL_ADMIN: {
        PERM_READ_TELEMETRY,
        PERM_EXECUTE_ACTION,
        PERM_REVIEW_HITL,
        PERM_KILL_SWITCH,
        PERM_MANAGE_POLICIES,
        PERM_AUDIT_VERIFY,
        PERM_AUDIT_EXPORT,
        PERM_MANAGE_TENANTS,
        PERM_MANAGE_USERS,
        PERM_PQC_ROTATE,
    },
    UserRole.SECURITY_OFFICER: {
        PERM_READ_TELEMETRY,
        PERM_REVIEW_HITL,
        PERM_KILL_SWITCH,
        PERM_MANAGE_POLICIES,
        PERM_AUDIT_VERIFY,
        PERM_AUDIT_EXPORT,
        PERM_PQC_ROTATE,
    },
    UserRole.COMPLIANCE_AUDITOR: {
        PERM_READ_TELEMETRY,
        PERM_AUDIT_VERIFY,
        PERM_AUDIT_EXPORT,
    },
    UserRole.HITL_REVIEWER: {
        PERM_READ_TELEMETRY,
        PERM_REVIEW_HITL,
        PERM_KILL_SWITCH,
    },
    UserRole.AGENT_OPERATOR: {
        PERM_READ_TELEMETRY,
        PERM_EXECUTE_ACTION,
    },
}


def _b64url_encode(data: bytes) -> str:
    """Base64 URL-safe encoding without trailing padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    """Base64 URL-safe decoding with added padding if necessary."""
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data.encode("utf-8"))


class SovereignAuthToken:
    """Zero-external-dependency, air-gapped cryptographic token generator and validator.
    
    Generates HMAC-SHA256 signed web tokens compatible with standard Bearer authorization
    without requiring external Identity Providers or network connectivity.
    """

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        self.secret_key = secret_key or _DEFAULT_SIGNING_SECRET

    def generate_token(
        self,
        user: UserIdentity,
        expires_in_seconds: int = 86400,  # 24 hours
        extra_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Issues a signed sovereign token for a given user."""
        header = {"alg": "HS256", "typ": "NETHICAL-SOVEREIGN"}
        now = int(time.time())
        role_val = user.role.value if hasattr(user.role, "value") else str(user.role)
        payload: Dict[str, Any] = {
            "sub": user.user_id,
            "username": user.username,
            "role": role_val,
            "tenant_id": user.tenant_id,
            "iat": now,
            "exp": now + expires_in_seconds,
            "jti": secrets.token_hex(12),
        }
        if extra_claims:
            payload.update(extra_claims)

        header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        message = f"{header_b64}.{payload_b64}".encode("utf-8")

        signature = hmac.new(self.secret_key, message, hashlib.sha256).digest()
        sig_b64 = _b64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{sig_b64}"

    def verify_token(self, token_str: str) -> Optional[Dict[str, Any]]:
        """Verifies a sovereign token signature and expiration.
        
        Returns payload dict if valid, or None if invalid/expired.
        """
        try:
            parts = token_str.strip().split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, sig_b64 = parts
            message = f"{header_b64}.{payload_b64}".encode("utf-8")
            expected_sig = hmac.new(self.secret_key, message, hashlib.sha256).digest()
            actual_sig = _b64url_decode(sig_b64)

            if not hmac.compare_digest(expected_sig, actual_sig):
                logger.warning("Token signature mismatch.")
                return None

            payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
            now = int(time.time())

            if payload.get("exp") and now > payload["exp"]:
                logger.warning(f"Token expired for user: {payload.get('username')}")
                return None

            return payload
        except Exception as e:
            logger.debug(f"Failed to verify token: {e}")
            return None


class RBACManager:
    """Manages role-based access control, credentials hashing, and authorization checks."""

    def __init__(self, token_service: Optional[SovereignAuthToken] = None) -> None:
        self.token_service = token_service or SovereignAuthToken()
        self._users: Dict[str, UserIdentity] = {}
        self._credentials: Dict[str, Tuple[str, str]] = {}  # username -> (salt_hex, hash_hex)
        self._api_keys: Dict[str, UserIdentity] = {}  # api_key -> UserIdentity

        self._seed_default_users()

    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
        """Derives a PBKDF2-HMAC-SHA256 key from password and salt."""
        if salt is None:
            salt = secrets.token_bytes(16)
        pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return salt.hex(), pwd_hash.hex()

    def verify_password(self, password: str, salt_hex: str, hash_hex: str) -> bool:
        """Verifies password against PBKDF2 salt and hash using constant-time comparison."""
        salt = bytes.fromhex(salt_hex)
        expected_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()
        return hmac.compare_digest(expected_hash, hash_hex)

    def register_user(
        self,
        username: str,
        password: str,
        role: UserRole,
        tenant_id: str = "default_tenant",
        full_name: Optional[str] = None,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> UserIdentity:
        """Registers a user account with hashed password and role."""
        salt_hex, hash_hex = self.hash_password(password)
        user = UserIdentity(
            username=username,
            role=role,
            tenant_id=tenant_id,
            full_name=full_name,
            email=email,
        )
        self._users[username] = user
        self._credentials[username] = (salt_hex, hash_hex)

        if api_key:
            self._api_keys[api_key] = user

        logger.info(f"Registered user '{username}' with role '{role.value}' for tenant '{tenant_id}'")
        return user

    def authenticate(self, username: str, password: str) -> Optional[Tuple[UserIdentity, str]]:
        """Authenticates username + password and issues a sovereign token."""
        if username not in self._users or username not in self._credentials:
            return None

        salt_hex, hash_hex = self._credentials[username]
        if not self.verify_password(password, salt_hex, hash_hex):
            return None

        user = self._users[username]
        token = self.token_service.generate_token(user)
        return user, token

    def authenticate_token(self, token_str: str) -> Optional[UserIdentity]:
        """Validates a sovereign token and returns the corresponding UserIdentity."""
        payload = self.token_service.verify_token(token_str)
        if not payload:
            return None

        username = payload.get("username")
        if username and username in self._users:
            return self._users[username]

        # Ephemeral or stateless user reconstruction from token
        try:
            return UserIdentity(
                user_id=payload.get("sub", "unknown"),
                username=payload.get("username", "anonymous"),
                role=UserRole(payload.get("role", UserRole.AGENT_OPERATOR.value)),
                tenant_id=payload.get("tenant_id", "default_tenant"),
            )
        except Exception:
            return None

    def authenticate_api_key(self, api_key: str) -> Optional[UserIdentity]:
        """Authenticates request via pre-shared API Key."""
        return self._api_keys.get(api_key)

    def has_permission(self, user: UserIdentity, permission: str) -> bool:
        """Checks if a user's role grants a specific permission."""
        role_enum = UserRole(user.role) if isinstance(user.role, str) else user.role
        allowed_perms = ROLE_PERMISSIONS.get(role_enum, set())
        return permission in allowed_perms

    def can_access_tenant(self, user: UserIdentity, target_tenant_id: str) -> bool:
        """Checks if a user can access a specific tenant workspace."""
        role_val = user.role.value if hasattr(user.role, "value") else str(user.role)
        if role_val == UserRole.GLOBAL_ADMIN.value:
            return True
        return user.tenant_id == target_tenant_id

    def get_user(self, username: str) -> Optional[UserIdentity]:
        """Retrieves a user by username."""
        return self._users.get(username)

    def list_users(self, tenant_id: Optional[str] = None) -> List[UserIdentity]:
        """Lists registered users, optionally filtered by tenant."""
        if tenant_id:
            return [u for u in self._users.values() if u.tenant_id == tenant_id]
        return list(self._users.values())

    def _seed_default_users(self) -> None:
        """Seeds default operational accounts for immediate use."""
        # 1. Global System Administrator
        self.register_user(
            username="admin",
            password="nethical_admin_sovereign_password",
            role=UserRole.GLOBAL_ADMIN,
            tenant_id="default_tenant",
            full_name="Global Sovereign Administrator",
            email="admin@sovereign.local",
            api_key="sk-nethical-admin-global-mesh-token",
        )

        # 2. Polish Government & Infrastructure Security Officer
        self.register_user(
            username="secops_lead",
            password="secops_sovereign_password",
            role=UserRole.SECURITY_OFFICER,
            tenant_id="gov_pl_cyber",
            full_name="Oficer Bezpieczeństwa Cybernetycznego",
            email="secops@csirt.gov.pl",
            api_key="sk-nethical-gov-secops-token",
        )

        # 3. Independent Compliance Auditor (EU AI Act & ISO 42001)
        self.register_user(
            username="auditor_anna",
            password="auditor_sovereign_password",
            role=UserRole.COMPLIANCE_AUDITOR,
            tenant_id="enterprise_fin_eu",
            full_name="dr Anna Kowalska (Lead Compliance Auditor)",
            email="auditor.anna@compliance-eu.org",
            api_key="sk-nethical-auditor-eu-token",
        )

        # 4. Human-in-the-Loop Reviewer
        self.register_user(
            username="reviewer_jan",
            password="reviewer_sovereign_password",
            role=UserRole.HITL_REVIEWER,
            tenant_id="default_tenant",
            full_name="Jan Nowak (HITL Oversight Officer)",
            email="hitl.jan@sovereign.local",
            api_key="sk-nethical-hitl-reviewer-token",
        )

        # 5. Autonomous Agent Fleet Operator
        self.register_user(
            username="operator_kris",
            password="operator_sovereign_password",
            role=UserRole.AGENT_OPERATOR,
            tenant_id="defense_airgap",
            full_name="Krzysztof Wiśniewski (Tactical Fleet Operator)",
            email="operator@tactical.mil.local",
            api_key="sk-nethical-operator-mil-token",
        )
