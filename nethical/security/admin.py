"""
Admin Interface for Role and User Management

This module provides an administrative interface for managing users,
roles, and permissions in the Nethical system.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .auth import get_auth_manager, AuthManager, APIKey
from nethical.core.rbac import (
    get_rbac_manager,
    RBACManager,
    Role,
    Permission,
    AccessDeniedError,
)

__all__ = [
    "AdminInterface",
    "UserInfo",
]

log = logging.getLogger(__name__)


class UserInfo:
    """User information including role and permissions"""
    
    def __init__(
        self,
        user_id: str,
        role: Optional[Role] = None,
        permissions: Optional[List[Permission]] = None,
        custom_permissions: Optional[List[Permission]] = None,
        api_keys: Optional[List[APIKey]] = None
    ):
        self.user_id = user_id
        self.role = role
        self.permissions = permissions or []
        self.custom_permissions = custom_permissions or []
        self.api_keys = api_keys or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "user_id": self.user_id,
            "role": self.role.value if self.role else None,
            "permissions": [p.value for p in self.permissions],
            "custom_permissions": [p.value for p in self.custom_permissions],
            "api_keys": [
                {
                    "key_id": k.key_id,
                    "name": k.name,
                    "created_at": k.created_at.isoformat(),
                    "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                    "enabled": k.enabled,
                }
                for k in self.api_keys
            ],
        }


class AdminInterface:
    """
    Administrative Interface for User and Role Management
    
    Provides methods to manage users, roles, permissions, and API keys.
    All operations are logged for audit purposes.
    """
    
    def __init__(
        self,
        auth_manager: Optional[AuthManager] = None,
        rbac_manager: Optional[RBACManager] = None,
        admin_user_id: Optional[str] = None
    ):
        """
        Initialize admin interface
        
        Args:
            auth_manager: Authentication manager (uses global if not provided)
            rbac_manager: RBAC manager (uses global if not provided)
            admin_user_id: User ID of the admin (for audit logging)
        """
        self.auth = auth_manager or get_auth_manager()
        self.rbac = rbac_manager or get_rbac_manager()
        self.admin_user_id = admin_user_id or "system"
        log.info(f"AdminInterface initialized by {self.admin_user_id}")
    
    def _check_admin(self) -> None:
        """Verify that the current user has admin privileges"""
        if self.admin_user_id != "system":
            try:
                self.rbac.check_role(self.admin_user_id, Role.ADMIN, raise_on_deny=True)
            except AccessDeniedError:
                log.error(f"Non-admin user {self.admin_user_id} attempted admin operation")
                raise
    
    # User Management
    
    def create_user(
        self,
        user_id: str,
        role: Role,
        create_api_key: bool = False,
        api_key_name: Optional[str] = None
    ) -> UserInfo:
        """
        Create a new user with specified role
        
        Args:
            user_id: Unique user identifier
            role: Role to assign to user
            create_api_key: If True, create an API key for the user
            api_key_name: Name for the API key (if created)
            
        Returns:
            UserInfo object with user details
        """
        self._check_admin()
        
        # Assign role
        self.rbac.assign_role(user_id, role)
        log.info(f"Admin {self.admin_user_id} created user {user_id} with role {role}")
        
        # Create API key if requested
        api_keys = []
        if create_api_key:
            key_name = api_key_name or f"{user_id}_default_key"
            key_string, api_key = self.auth.create_api_key(user_id, key_name)
            api_keys.append(api_key)
            log.info(f"Created API key '{key_name}' for user {user_id}")
        
        return self.get_user(user_id)
    
    def get_user(self, user_id: str) -> UserInfo:
        """
        Get user information
        
        Args:
            user_id: User identifier
            
        Returns:
            UserInfo object
        """
        role = self.rbac.get_role(user_id)
        permissions = list(self.rbac.get_user_permissions(user_id))
        custom_permissions = list(self.rbac.user_custom_permissions.get(user_id, set()))
        api_keys = self.auth.list_api_keys(user_id)
        
        return UserInfo(
            user_id=user_id,
            role=role,
            permissions=permissions,
            custom_permissions=custom_permissions,
            api_keys=api_keys
        )
    
    def list_users(self) -> List[UserInfo]:
        """
        List all users
        
        Returns:
            List of UserInfo objects
        """
        users = []
        for user_id in self.rbac.user_roles.keys():
            users.append(self.get_user(user_id))
        return users
    
    def delete_user(self, user_id: str) -> None:
        """
        Delete a user and revoke all their access
        
        Args:
            user_id: User identifier
        """
        self._check_admin()
        
        # Revoke all API keys
        for api_key in self.auth.list_api_keys(user_id):
            self.auth.revoke_api_key(api_key.key_id)
        
        # Revoke role
        self.rbac.revoke_role(user_id)
        
        # Remove custom permissions
        if user_id in self.rbac.user_custom_permissions:
            del self.rbac.user_custom_permissions[user_id]
        
        log.info(f"Admin {self.admin_user_id} deleted user {user_id}")
    
    # Role Management
    
    def assign_role(self, user_id: str, role: Role) -> None:
        """
        Assign or change a user's role
        
        Args:
            user_id: User identifier
            role: New role for the user
        """
        self._check_admin()
        old_role = self.rbac.get_role(user_id)
        self.rbac.assign_role(user_id, role)
        log.info(
            f"Admin {self.admin_user_id} changed role for {user_id} "
            f"from {old_role} to {role}"
        )
    
    def revoke_role(self, user_id: str) -> None:
        """
        Revoke a user's role
        
        Args:
            user_id: User identifier
        """
        self._check_admin()
        role = self.rbac.get_role(user_id)
        self.rbac.revoke_role(user_id)
        log.info(f"Admin {self.admin_user_id} revoked role {role} from {user_id}")
    
    # Permission Management
    
    def grant_permission(self, user_id: str, permission: Permission) -> None:
        """
        Grant a custom permission to a user
        
        Args:
            user_id: User identifier
            permission: Permission to grant
        """
        self._check_admin()
        self.rbac.grant_permission(user_id, permission)
        log.info(
            f"Admin {self.admin_user_id} granted permission {permission} "
            f"to user {user_id}"
        )
    
    def revoke_permission(self, user_id: str, permission: Permission) -> None:
        """
        Revoke a custom permission from a user
        
        Args:
            user_id: User identifier
            permission: Permission to revoke
        """
        self._check_admin()
        self.rbac.revoke_permission(user_id, permission)
        log.info(
            f"Admin {self.admin_user_id} revoked permission {permission} "
            f"from user {user_id}"
        )
    
    # API Key Management
    
    def create_api_key(
        self,
        user_id: str,
        key_name: str,
        expires_in_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """
        Create an API key for a user
        
        Args:
            user_id: User identifier
            key_name: Name for the API key
            expires_in_days: Optional expiration in days
            
        Returns:
            Tuple of (key_string, api_key_object)
            Note: key_string is only available at creation time
        """
        self._check_admin()
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        key_string, api_key = self.auth.create_api_key(user_id, key_name, expires_at)
        log.info(
            f"Admin {self.admin_user_id} created API key '{key_name}' "
            f"for user {user_id}"
        )
        
        return key_string, api_key
    
    def revoke_api_key(self, key_id: str) -> None:
        """
        Revoke an API key
        
        Args:
            key_id: API key identifier
        """
        self._check_admin()
        self.auth.revoke_api_key(key_id)
        log.info(f"Admin {self.admin_user_id} revoked API key {key_id}")
    
    def list_api_keys(self, user_id: Optional[str] = None) -> List[APIKey]:
        """
        List API keys, optionally filtered by user
        
        Args:
            user_id: Optional user identifier to filter by
            
        Returns:
            List of APIKey objects
        """
        return self.auth.list_api_keys(user_id)
    
    # Token Management
    
    def create_tokens_for_user(
        self,
        user_id: str,
        scope: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create access and refresh tokens for a user
        
        Args:
            user_id: User identifier
            scope: Optional scope for the tokens
            
        Returns:
            Dictionary with 'access_token' and 'refresh_token'
        """
        self._check_admin()
        
        access_token, _ = self.auth.create_access_token(user_id, scope)
        refresh_token, _ = self.auth.create_refresh_token(user_id)
        
        log.info(f"Admin {self.admin_user_id} created tokens for user {user_id}")
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
        }
    
    # Audit and Reporting
    
    def get_access_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get access decision history
        
        Args:
            user_id: Optional user to filter by
            limit: Maximum number of records
            
        Returns:
            List of access decisions
        """
        history = self.rbac.get_access_history(user_id, limit)
        return [
            {
                "user_id": d.user_id,
                "allowed": d.allowed,
                "role": d.role.value if d.role else None,
                "required_role": d.required_role.value if d.required_role else None,
                "required_permission": d.required_permission.value if d.required_permission else None,
                "reason": d.reason,
                "timestamp": d.timestamp.isoformat(),
            }
            for d in history
        ]
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the system state
        
        Returns:
            Dictionary with system statistics
        """
        users = self.list_users()
        
        role_counts = {}
        for role in Role:
            role_counts[role.value] = sum(1 for u in users if u.role == role)
        
        total_api_keys = sum(len(u.api_keys) for u in users)
        active_api_keys = sum(
            len([k for k in u.api_keys if k.enabled])
            for u in users
        )
        
        return {
            "total_users": len(users),
            "users_by_role": role_counts,
            "total_api_keys": total_api_keys,
            "active_api_keys": active_api_keys,
            "access_history_size": len(self.rbac.access_history),
        }
