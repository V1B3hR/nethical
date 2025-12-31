"""
Role-Based Access Control (RBAC) System for Nethical

This module provides a comprehensive RBAC system with:
- Role hierarchy: admin, operator, auditor, viewer
- Decorator-based access control
- Permission management
- Audit logging for access decisions
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Set, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps

__all__ = [
    "Role",
    "Permission",
    "RBACManager",
    "require_role",
    "require_permission",
    "AccessDeniedError",
]

log = logging.getLogger(__name__)


class Role(str, Enum):
    """Role hierarchy in ascending order of privilege"""

    VIEWER = "viewer"
    AUDITOR = "auditor"
    OPERATOR = "operator"
    ADMIN = "admin"


class Permission(str, Enum):
    """Fine-grained permissions for system operations"""

    # Read permissions
    READ_POLICIES = "read_policies"
    READ_ACTIONS = "read_actions"
    READ_VIOLATIONS = "read_violations"
    READ_METRICS = "read_metrics"
    READ_AUDIT_LOGS = "read_audit_logs"

    # Write permissions
    WRITE_POLICIES = "write_policies"
    WRITE_ACTIONS = "write_actions"
    EXECUTE_ACTIONS = "execute_actions"

    # Management permissions
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_QUARANTINE = "manage_quarantine"
    MANAGE_SYSTEM = "manage_system"

    # Administrative permissions
    ADMIN_OVERRIDE = "admin_override"
    SYSTEM_CONFIG = "system_config"

    # Custom permissions
    MODIFY_CODE = "modify_code"  # <-- Added the new permission


@dataclass
class AccessDecision:
    """Result of an access control decision"""

    allowed: bool
    user_id: str
    role: Role
    required_role: Optional[Role] = None
    required_permission: Optional[Permission] = None
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AccessDeniedError(PermissionError):
    """Raised when access is denied by RBAC"""

    def __init__(self, decision: AccessDecision):
        self.decision = decision
        super().__init__(f"Access denied: {decision.reason}")


class RBACManager:
    """
    Role-Based Access Control Manager

    Manages roles, permissions, and access control decisions.
    """

    # Role hierarchy: higher roles inherit permissions from lower roles
    ROLE_HIERARCHY: Dict[Role, int] = {
        Role.VIEWER: 0,
        Role.AUDITOR: 1,
        Role.OPERATOR: 2,
        Role.ADMIN: 3,
    }

    # Role to permissions mapping
    ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
        Role.VIEWER: {
            Permission.READ_POLICIES,
            Permission.READ_ACTIONS,
            Permission.READ_METRICS,
        },
        Role.AUDITOR: {
            Permission.READ_POLICIES,
            Permission.READ_ACTIONS,
            Permission.READ_VIOLATIONS,
            Permission.READ_METRICS,
            Permission.READ_AUDIT_LOGS,
        },
        Role.OPERATOR: {
            Permission.READ_POLICIES,
            Permission.READ_ACTIONS,
            Permission.READ_VIOLATIONS,
            Permission.READ_METRICS,
            Permission.WRITE_ACTIONS,
            Permission.EXECUTE_ACTIONS,
            Permission.MANAGE_QUARANTINE,
        },
        Role.ADMIN: {
            Permission.READ_POLICIES,
            Permission.READ_ACTIONS,
            Permission.READ_VIOLATIONS,
            Permission.READ_METRICS,
            Permission.READ_AUDIT_LOGS,
            Permission.WRITE_POLICIES,
            Permission.WRITE_ACTIONS,
            Permission.EXECUTE_ACTIONS,
            Permission.MANAGE_USERS,
            Permission.MANAGE_ROLES,
            Permission.MANAGE_QUARANTINE,
            Permission.MANAGE_SYSTEM,
            Permission.ADMIN_OVERRIDE,
            Permission.SYSTEM_CONFIG,
            Permission.MODIFY_CODE,  # <-- Make sure admins have MODIFY_CODE by default
        },
    }

    def __init__(self, audit_logger: Optional[Callable[[AccessDecision], None]] = None):
        """
        Initialize RBAC Manager

        Args:
            audit_logger: Optional callback for logging access decisions
        """
        self.user_roles: Dict[str, Role] = {}
        self.user_custom_permissions: Dict[str, Set[Permission]] = {}
        self.audit_logger = audit_logger or self._default_audit_logger
        self.access_history: List[AccessDecision] = []

    def _default_audit_logger(self, decision: AccessDecision) -> None:
        """Default audit logger that logs to standard logging"""
        if decision.allowed:
            log.info(
                f"Access granted: user={decision.user_id}, role={decision.role}, "
                f"required={decision.required_role or decision.required_permission}"
            )
        else:
            log.warning(
                f"Access denied: user={decision.user_id}, role={decision.role}, "
                f"reason={decision.reason}"
            )

    def assign_role(self, user_id: str, role: Role) -> None:
        """Assign a role to a user"""
        self.user_roles[user_id] = role
        log.info(f"Assigned role {role} to user {user_id}")

    def revoke_role(self, user_id: str) -> None:
        """Revoke a user's role"""
        if user_id in self.user_roles:
            role = self.user_roles.pop(user_id)
            log.info(f"Revoked role {role} from user {user_id}")

    def get_role(self, user_id: str) -> Optional[Role]:
        """Get a user's role"""
        return self.user_roles.get(user_id)

    def grant_permission(self, user_id: str, permission: Permission) -> None:
        """Grant a custom permission to a user"""
        if user_id not in self.user_custom_permissions:
            self.user_custom_permissions[user_id] = set()
        self.user_custom_permissions[user_id].add(permission)
        log.info(f"Granted permission {permission} to user {user_id}")

    def revoke_permission(self, user_id: str, permission: Permission) -> None:
        """Revoke a custom permission from a user"""
        if user_id in self.user_custom_permissions:
            self.user_custom_permissions[user_id].discard(permission)
            log.info(f"Revoked permission {permission} from user {user_id}")

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user (role-based + custom)"""
        role = self.get_role(user_id)
        permissions = set()

        if role:
            # Get permissions for this role
            permissions.update(self.ROLE_PERMISSIONS.get(role, set()))

        # Add custom permissions
        permissions.update(self.user_custom_permissions.get(user_id, set()))

        return permissions

    def has_role(self, user_id: str, required_role: Role) -> bool:
        """Check if user has at least the required role level"""
        user_role = self.get_role(user_id)
        if not user_role:
            return False

        user_level = self.ROLE_HIERARCHY.get(user_role, -1)
        required_level = self.ROLE_HIERARCHY.get(required_role, 999)

        return user_level >= required_level

    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has a specific permission"""
        return permission in self.get_user_permissions(user_id)

    def check_role(
        self, user_id: str, required_role: Role, raise_on_deny: bool = True
    ) -> AccessDecision:
        """
        Check if user has required role

        Args:
            user_id: User identifier
            required_role: Required role level
            raise_on_deny: If True, raise AccessDeniedError on denial

        Returns:
            AccessDecision with result

        Raises:
            AccessDeniedError: If access denied and raise_on_deny is True
        """
        user_role = self.get_role(user_id)

        if not user_role:
            decision = AccessDecision(
                allowed=False,
                user_id=user_id,
                role=Role.VIEWER,  # Default
                required_role=required_role,
                reason=f"User has no assigned role",
            )
        elif self.has_role(user_id, required_role):
            decision = AccessDecision(
                allowed=True,
                user_id=user_id,
                role=user_role,
                required_role=required_role,
                reason=f"User has sufficient role: {user_role}",
            )
        else:
            decision = AccessDecision(
                allowed=False,
                user_id=user_id,
                role=user_role,
                required_role=required_role,
                reason=f"User role {user_role} insufficient for {required_role}",
            )

        self.access_history.append(decision)
        self.audit_logger(decision)

        if not decision.allowed and raise_on_deny:
            raise AccessDeniedError(decision)

        return decision

    def check_permission(
        self, user_id: str, permission: Permission, raise_on_deny: bool = True
    ) -> AccessDecision:
        """
        Check if user has required permission

        Args:
            user_id: User identifier
            permission: Required permission
            raise_on_deny: If True, raise AccessDeniedError on denial

        Returns:
            AccessDecision with result

        Raises:
            AccessDeniedError: If access denied and raise_on_deny is True
        """
        user_role = self.get_role(user_id)

        if not user_role:
            decision = AccessDecision(
                allowed=False,
                user_id=user_id,
                role=Role.VIEWER,  # Default
                required_permission=permission,
                reason=f"User has no assigned role",
            )
        elif self.has_permission(user_id, permission):
            decision = AccessDecision(
                allowed=True,
                user_id=user_id,
                role=user_role,
                required_permission=permission,
                reason=f"User has required permission: {permission}",
            )
        else:
            decision = AccessDecision(
                allowed=False,
                user_id=user_id,
                role=user_role,
                required_permission=permission,
                reason=f"User lacks permission: {permission}",
            )

        self.access_history.append(decision)
        self.audit_logger(decision)

        if not decision.allowed and raise_on_deny:
            raise AccessDeniedError(decision)

        return decision

    def get_access_history(
        self, user_id: Optional[str] = None, limit: int = 100
    ) -> List[AccessDecision]:
        """Get access history, optionally filtered by user"""
        history = self.access_history
        if user_id:
            history = [d for d in history if d.user_id == user_id]
        return history[-limit:]

    def list_users(self) -> Dict[str, Dict[str, Any]]:
        """List all users with their roles and permissions"""
        users = {}
        for user_id, role in self.user_roles.items():
            users[user_id] = {
                "role": role.value,
                "permissions": [p.value for p in self.get_user_permissions(user_id)],
                "custom_permissions": [
                    p.value for p in self.user_custom_permissions.get(user_id, set())
                ],
            }
        return users


# Global RBAC manager instance
_rbac_manager: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get or create the global RBAC manager instance"""
    global _rbac_manager
    if _rbac_manager is None:
        _rbac_manager = RBACManager()
    return _rbac_manager


def set_rbac_manager(manager: RBACManager) -> None:
    """Set the global RBAC manager instance"""
    global _rbac_manager
    _rbac_manager = manager


def require_role(role: Role) -> Callable:
    """
    Decorator to require a specific role for a function

    Usage:
        @require_role(Role.ADMIN)
        def delete_user(user_id: str, current_user: str):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user_id from kwargs or args
            user_id = kwargs.get("current_user") or kwargs.get("user_id")
            if not user_id and args:
                # Try to find user_id in args (common patterns)
                for arg in args:
                    if isinstance(arg, str) and len(arg) > 0:
                        user_id = arg
                        break

            if not user_id:
                raise ValueError("Cannot determine user_id for RBAC check")

            manager = get_rbac_manager()
            manager.check_role(user_id, role, raise_on_deny=True)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_permission(permission: Permission) -> Callable:
    """
    Decorator to require a specific permission for a function

    Usage:
        @require_permission(Permission.WRITE_POLICIES)
        def update_policy(policy_id: str, current_user: str):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract user_id from kwargs or args
            user_id = kwargs.get("current_user") or kwargs.get("user_id")
            if not user_id and args:
                # Try to find user_id in args (common patterns)
                for arg in args:
                    if isinstance(arg, str) and len(arg) > 0:
                        user_id = arg
                        break

            if not user_id:
                raise ValueError("Cannot determine user_id for RBAC check")

            manager = get_rbac_manager()
            manager.check_permission(user_id, permission, raise_on_deny=True)

            return func(*args, **kwargs)

        return wrapper

    return decorator
