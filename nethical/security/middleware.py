"""
API Authentication and Authorization Middleware for Nethical

This module provides middleware components that combine authentication
and RBAC authorization for API protection.
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, Any, Dict
from functools import wraps

from .auth import authenticate_request, AuthenticationError
from nethical.core.rbac import (
    get_rbac_manager,
    Role,
    Permission,
    AccessDeniedError,
)

__all__ = [
    "AuthMiddleware",
    "require_auth",
    "require_auth_and_role",
    "require_auth_and_permission",
]

log = logging.getLogger(__name__)


class AuthMiddleware:
    """
    Authentication and Authorization Middleware
    
    Provides methods to authenticate requests and check authorization.
    Can be used as a base class for API framework-specific middleware.
    """
    
    def __init__(self):
        """Initialize the middleware"""
        log.info("AuthMiddleware initialized")
    
    def authenticate(
        self,
        authorization_header: Optional[str] = None,
        api_key_header: Optional[str] = None
    ) -> str:
        """
        Authenticate a request and return the user ID
        
        Args:
            authorization_header: Authorization header value (Bearer token)
            api_key_header: API key header value
            
        Returns:
            User ID if authenticated
            
        Raises:
            AuthenticationError: If authentication fails
        """
        return authenticate_request(authorization_header, api_key_header)
    
    def check_role(self, user_id: str, required_role: Role) -> None:
        """
        Check if user has required role
        
        Args:
            user_id: User identifier
            required_role: Required role level
            
        Raises:
            AccessDeniedError: If user lacks required role
        """
        rbac = get_rbac_manager()
        rbac.check_role(user_id, required_role, raise_on_deny=True)
    
    def check_permission(self, user_id: str, permission: Permission) -> None:
        """
        Check if user has required permission
        
        Args:
            user_id: User identifier
            permission: Required permission
            
        Raises:
            AccessDeniedError: If user lacks required permission
        """
        rbac = get_rbac_manager()
        rbac.check_permission(user_id, permission, raise_on_deny=True)
    
    def process_request(
        self,
        authorization_header: Optional[str] = None,
        api_key_header: Optional[str] = None,
        required_role: Optional[Role] = None,
        required_permission: Optional[Permission] = None
    ) -> str:
        """
        Process a request with authentication and optional authorization
        
        Args:
            authorization_header: Authorization header value
            api_key_header: API key header value
            required_role: Optional required role
            required_permission: Optional required permission
            
        Returns:
            User ID if authenticated and authorized
            
        Raises:
            AuthenticationError: If authentication fails
            AccessDeniedError: If authorization fails
        """
        # Authenticate
        user_id = self.authenticate(authorization_header, api_key_header)
        
        # Check role if required
        if required_role:
            self.check_role(user_id, required_role)
        
        # Check permission if required
        if required_permission:
            self.check_permission(user_id, required_permission)
        
        return user_id


def require_auth(func: Callable) -> Callable:
    """
    Decorator to require authentication for a function
    
    The function must accept kwargs with 'authorization_header' or 'api_key_header',
    and will receive 'current_user' in kwargs after successful authentication.
    
    Usage:
        @require_auth
        def get_metrics(current_user: str = None, **kwargs):
            return f"Metrics for {current_user}"
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        middleware = AuthMiddleware()
        
        # Extract auth headers from kwargs
        auth_header = kwargs.pop("authorization_header", None)
        api_key_header = kwargs.pop("api_key_header", None)
        
        try:
            user_id = middleware.authenticate(auth_header, api_key_header)
            kwargs["current_user"] = user_id
            return func(*args, **kwargs)
        except AuthenticationError as e:
            log.warning(f"Authentication failed: {e}")
            raise
    
    return wrapper


def require_auth_and_role(role: Role) -> Callable:
    """
    Decorator to require authentication and a specific role for a function
    
    The function must accept kwargs with 'authorization_header' or 'api_key_header',
    and will receive 'current_user' in kwargs after successful authentication.
    
    Usage:
        @require_auth_and_role(Role.ADMIN)
        def delete_policy(policy_id: str, current_user: str = None):
            return f"Policy {policy_id} deleted by {current_user}"
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            middleware = AuthMiddleware()
            
            # Extract auth headers from kwargs
            auth_header = kwargs.pop("authorization_header", None)
            api_key_header = kwargs.pop("api_key_header", None)
            
            try:
                user_id = middleware.process_request(
                    authorization_header=auth_header,
                    api_key_header=api_key_header,
                    required_role=role
                )
                kwargs["current_user"] = user_id
                return func(*args, **kwargs)
            except (AuthenticationError, AccessDeniedError) as e:
                log.warning(f"Access denied: {e}")
                raise
        
        return wrapper
    return decorator


def require_auth_and_permission(permission: Permission) -> Callable:
    """
    Decorator to require authentication and a specific permission for a function
    
    The function must accept kwargs with 'authorization_header' or 'api_key_header',
    and will receive 'current_user' in kwargs after successful authentication.
    
    Usage:
        @require_auth_and_permission(Permission.WRITE_POLICIES)
        def update_policy(policy_id: str, data: dict, current_user: str = None):
            return f"Policy {policy_id} updated by {current_user}"
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            middleware = AuthMiddleware()
            
            # Extract auth headers from kwargs
            auth_header = kwargs.pop("authorization_header", None)
            api_key_header = kwargs.pop("api_key_header", None)
            
            try:
                user_id = middleware.process_request(
                    authorization_header=auth_header,
                    api_key_header=api_key_header,
                    required_permission=permission
                )
                kwargs["current_user"] = user_id
                return func(*args, **kwargs)
            except (AuthenticationError, AccessDeniedError) as e:
                log.warning(f"Access denied: {e}")
                raise
        
        return wrapper
    return decorator
