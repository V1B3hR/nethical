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
import sys
import time
from typing import Any

from ..core.models import UserIdentity, UserRole

logger = logging.getLogger("nethical.auth.rbac")


def resolve_signing_secret(
    explicit_secret: bytes | None = None,
    allow_ephemeral: bool = True,
) -> bytes:
    """Resolves the cryptographic token signing secret key securely.

    Order of precedence:
    1. Explicit secret passed via parameter (min 16 bytes)
    2. NETHICAL_AUTH_SECRET environment variable
    3. File path specified in NETHICAL_SECRET_FILE environment variable
    4. Ephemeral cryptographically random 256-bit key (dev/test mode or allow_ephemeral=True)

    Raises:
        ValueError: If no secret is configured and ephemeral generation is not permitted.
    """
    if explicit_secret is not None:
        if len(explicit_secret) < 16:
            raise ValueError("Signing secret must be at least 16 bytes long.")
        return explicit_secret

    env_secret = os.environ.get("NETHICAL_AUTH_SECRET")
    if env_secret:
        secret_bytes = env_secret.encode("utf-8")
        if len(secret_bytes) < 16:
            raise ValueError("NETHICAL_AUTH_SECRET must be at least 16 characters long.")
        return secret_bytes

    secret_file = os.environ.get("NETHICAL_SECRET_FILE")
    if secret_file and os.path.exists(secret_file):
        with open(secret_file, "rb") as f:
            secret_bytes = f.read().strip()
            if len(secret_bytes) >= 16:
                return secret_bytes

    # Ephemeral mode: permitted in dev, test, or when explicitly allowed
    is_test_or_dev = (
        "pytest" in sys.modules
        or os.environ.get("NETHICAL_ENV", "").lower() in ("development", "dev", "test")
        or os.environ.get("NETHICAL_ALLOW_DEV_DEFAULTS", "0").lower() in ("1", "true", "yes")
    )
    if allow_ephemeral or is_test_or_dev:
        generated = secrets.token_bytes(32)
        logger.warning(
            "⚠️ NETHICAL_AUTH_SECRET not configured! Generated ephemeral 256-bit runtime key. "
            "Tokens will invalidate upon process restart. Set NETHICAL_AUTH_SECRET for production."
        )
        return generated

    raise ValueError(
        "CRITICAL SECURITY CONFIGURATION ERROR: NETHICAL_AUTH_SECRET environment variable "
        "or NETHICAL_SECRET_FILE must be configured for production deployments. "
        "Hardcoded default credentials are strictly prohibited."
    )

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
ROLE_PERMISSIONS: dict[UserRole, set[str]] = {
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

    def __init__(
        self,
        secret_key: bytes | None = None,
        allow_ephemeral: bool = True,
    ) -> None:
        self.secret_key = resolve_signing_secret(secret_key, allow_ephemeral=allow_ephemeral)

    def generate_token(
        self,
        user: UserIdentity,
        expires_in_seconds: int = 86400,  # 24 hours
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        """Issues a signed sovereign token for a given user."""
        header = {"alg": "HS256", "typ": "NETHICAL-SOVEREIGN"}
        now = int(time.time())
        role_val = user.role.value if hasattr(user.role, "value") else str(user.role)
        payload: dict[str, Any] = {
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
        message = f"{header_b64}.{payload_b64}".encode()

        signature = hmac.new(self.secret_key, message, hashlib.sha256).digest()
        sig_b64 = _b64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{sig_b64}"

    def verify_token(self, token_str: str) -> dict[str, Any] | None:
        """Verifies a sovereign token signature and expiration.

        Returns payload dict if valid, or None if invalid/expired.
        """
        try:
            parts = token_str.strip().split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, sig_b64 = parts
            message = f"{header_b64}.{payload_b64}".encode()
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

    def __init__(
        self,
        token_service: SovereignAuthToken | None = None,
        auto_seed_dev: bool | None = None,
    ) -> None:
        self.token_service = token_service or SovereignAuthToken()
        self._users: dict[str, UserIdentity] = {}
        self._credentials: dict[str, tuple[str, str]] = {}  # username -> (salt_hex, hash_hex)
        self._api_keys: dict[str, UserIdentity] = {}  # api_key -> UserIdentity

        # In production, users are NEVER seeded with hardcoded credentials.
        # Development accounts are seeded ONLY when explicitly requested via auto_seed_dev=True
        # or when NETHICAL_ALLOW_DEV_DEFAULTS=1 is explicitly set in the environment.
        should_seed = auto_seed_dev if auto_seed_dev is not None else (
            os.environ.get("NETHICAL_ALLOW_DEV_DEFAULTS", "0").lower() in ("1", "true", "yes")
        )
        if should_seed:
            self.seed_development_users()

    @staticmethod
    def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
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
        full_name: str | None = None,
        email: str | None = None,
        api_key: str | None = None,
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

    def authenticate(self, username: str, password: str) -> tuple[UserIdentity, str] | None:
        """Authenticates username + password and issues a sovereign token."""
        if username not in self._users or username not in self._credentials:
            return None

        salt_hex, hash_hex = self._credentials[username]
        if not self.verify_password(password, salt_hex, hash_hex):
            return None

        user = self._users[username]
        token = self.token_service.generate_token(user)
        return user, token

    def authenticate_token(self, token_str: str) -> UserIdentity | None:
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

    def authenticate_api_key(self, api_key: str) -> UserIdentity | None:
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

    def get_user(self, username: str) -> UserIdentity | None:
        """Retrieves a user by username."""
        return self._users.get(username)

    def list_users(self, tenant_id: str | None = None) -> list[UserIdentity]:
        """Lists registered users, optionally filtered by tenant."""
        if tenant_id:
            return [u for u in self._users.values() if u.tenant_id == tenant_id]
        return list(self._users.values())

    def is_bootstrapped(self) -> bool:
        """Checks if the system has been bootstrapped with at least one administrator or user account."""
        return len(self._users) > 0

    def bootstrap_admin(
        self,
        username: str = "admin",
        password: str | None = None,
        email: str | None = None,
        full_name: str | None = None,
        tenant_id: str = "default_tenant",
    ) -> tuple[UserIdentity, str, str]:
        """Securely bootstraps the initial Global Administrator account for production deployment.

        If password is not provided, a high-entropy cryptographically secure password
        is generated using secrets.token_urlsafe(20).
        A high-entropy API key is always generated.

        Returns:
            Tuple[UserIdentity, str, str]: (UserIdentity, cleartext_password, api_key)

        Raises:
            RuntimeError: If the system is already bootstrapped with users.
        """
        if self.is_bootstrapped():
            raise RuntimeError(
                "System is already bootstrapped. Cannot run bootstrap_admin when accounts exist."
            )

        generated_password = password or secrets.token_urlsafe(20)
        generated_api_key = f"sk-sovereign-{secrets.token_hex(24)}"

        user = self.register_user(
            username=username,
            password=generated_password,
            role=UserRole.GLOBAL_ADMIN,
            tenant_id=tenant_id,
            full_name=full_name or "Global Sovereign Administrator",
            email=email or f"{username}@{tenant_id}.local",
            api_key=generated_api_key,
        )
        logger.info(f"✅ System successfully bootstrapped with Global Administrator: '{username}'")
        return user, generated_password, generated_api_key

    def seed_development_users(self) -> None:
        """Seeds development test accounts with a prominent security warning.

        WARNING: This method must NEVER be invoked in a production environment.
        """
        logger.warning(
            "⚠️ INSECURE CONFIGURATION WARNING: Development test accounts are being seeded! "
            "Do NOT enable NETHICAL_ALLOW_DEV_DEFAULTS in production environments."
        )
        # 1. Global System Administrator (DEV ONLY)
        self.register_user(
            username="admin",
            password="nethical_admin_sovereign_password",
            role=UserRole.GLOBAL_ADMIN,
            tenant_id="default_tenant",
            full_name="Global Sovereign Administrator (DEV ONLY)",
            email="admin@sovereign.local",
            api_key="sk-nethical-admin-global-mesh-token",
        )

        # 2. Polish Government & Infrastructure Security Officer (DEV ONLY)
        self.register_user(
            username="secops_lead",
            password="secops_sovereign_password",
            role=UserRole.SECURITY_OFFICER,
            tenant_id="gov_pl_cyber",
            full_name="Oficer Bezpieczeństwa Cybernetycznego (DEV ONLY)",
            email="secops@csirt.gov.pl",
            api_key="sk-nethical-gov-secops-token",
        )

        # 3. Independent Compliance Auditor (EU AI Act & ISO 42001) (DEV ONLY)
        self.register_user(
            username="auditor_anna",
            password="auditor_sovereign_password",
            role=UserRole.COMPLIANCE_AUDITOR,
            tenant_id="enterprise_fin_eu",
            full_name="dr Anna Kowalska (DEV ONLY)",
            email="auditor.anna@compliance-eu.org",
            api_key="sk-nethical-auditor-eu-token",
        )

        # 4. Human-in-the-Loop Reviewer (DEV ONLY)
        self.register_user(
            username="reviewer_jan",
            password="reviewer_sovereign_password",
            role=UserRole.HITL_REVIEWER,
            tenant_id="default_tenant",
            full_name="Jan Nowak (DEV ONLY)",
            email="hitl.jan@sovereign.local",
            api_key="sk-nethical-hitl-reviewer-token",
        )

        # 5. Autonomous Agent Fleet Operator (DEV ONLY)
        self.register_user(
            username="operator_kris",
            password="operator_sovereign_password",
            role=UserRole.AGENT_OPERATOR,
            tenant_id="defense_airgap",
            full_name="Krzysztof Wiśniewski (DEV ONLY)",
            email="operator@tactical.mil.local",
            api_key="sk-nethical-operator-mil-token",
        )

    def _seed_default_users(self) -> None:
        """Deprecated alias for seed_development_users (kept for test compatibility)."""
        self.seed_development_users()
