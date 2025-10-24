"""
Integration tests for Auth + RBAC middleware and admin interface
"""

import pytest
from datetime import datetime, timedelta

from nethical.security.auth import (
    AuthManager,
    set_auth_manager,
)
from nethical.security.middleware import (
    AuthMiddleware,
    require_auth,
    require_auth_and_role,
    require_auth_and_permission,
)
from nethical.security.admin import (
    AdminInterface,
)
from nethical.core.rbac import (
    RBACManager,
    Role,
    Permission,
    AccessDeniedError,
    set_rbac_manager,
)


class TestAuthMiddleware:
    """Test cases for authentication middleware"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth = AuthManager(secret_key="test-secret")
        self.rbac = RBACManager()
        set_auth_manager(self.auth)
        set_rbac_manager(self.rbac)
        
        # Create test users
        self.rbac.assign_role("admin_user", Role.ADMIN)
        self.rbac.assign_role("operator_user", Role.OPERATOR)
        self.rbac.assign_role("viewer_user", Role.VIEWER)
        
        self.middleware = AuthMiddleware()
    
    def test_authenticate_with_jwt(self):
        """Test middleware authentication with JWT token"""
        token, _ = self.auth.create_access_token("admin_user")
        auth_header = f"Bearer {token}"
        
        user_id = self.middleware.authenticate(authorization_header=auth_header)
        assert user_id == "admin_user"
    
    def test_authenticate_with_api_key(self):
        """Test middleware authentication with API key"""
        key_string, _ = self.auth.create_api_key("operator_user", "Test Key")
        
        user_id = self.middleware.authenticate(api_key_header=key_string)
        assert user_id == "operator_user"
    
    def test_check_role_success(self):
        """Test role checking in middleware"""
        self.middleware.check_role("admin_user", Role.OPERATOR)
        # Should not raise
    
    def test_check_role_failure(self):
        """Test role checking failure in middleware"""
        with pytest.raises(AccessDeniedError):
            self.middleware.check_role("viewer_user", Role.ADMIN)
    
    def test_check_permission_success(self):
        """Test permission checking in middleware"""
        self.middleware.check_permission("operator_user", Permission.EXECUTE_ACTIONS)
        # Should not raise
    
    def test_check_permission_failure(self):
        """Test permission checking failure in middleware"""
        with pytest.raises(AccessDeniedError):
            self.middleware.check_permission("viewer_user", Permission.EXECUTE_ACTIONS)
    
    def test_process_request_full_flow(self):
        """Test complete request processing flow"""
        token, _ = self.auth.create_access_token("admin_user")
        auth_header = f"Bearer {token}"
        
        user_id = self.middleware.process_request(
            authorization_header=auth_header,
            required_role=Role.OPERATOR,
            required_permission=Permission.READ_METRICS
        )
        
        assert user_id == "admin_user"


class TestAuthDecorators:
    """Test cases for authentication decorators"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth = AuthManager(secret_key="test-secret")
        self.rbac = RBACManager()
        set_auth_manager(self.auth)
        set_rbac_manager(self.rbac)
        
        self.rbac.assign_role("admin_user", Role.ADMIN)
        self.rbac.assign_role("operator_user", Role.OPERATOR)
        self.rbac.assign_role("viewer_user", Role.VIEWER)
    
    def test_require_auth_decorator(self):
        """Test require_auth decorator"""
        @require_auth
        def protected_function(current_user: str = None):
            return f"Hello {current_user}"
        
        token, _ = self.auth.create_access_token("admin_user")
        
        result = protected_function(authorization_header=f"Bearer {token}")
        assert result == "Hello admin_user"
    
    def test_require_auth_decorator_no_auth(self):
        """Test require_auth decorator without authentication"""
        @require_auth
        def protected_function(current_user: str = None):
            return f"Hello {current_user}"
        
        from nethical.security.auth import AuthenticationError
        with pytest.raises(AuthenticationError):
            protected_function()
    
    def test_require_auth_and_role_decorator(self):
        """Test require_auth_and_role decorator"""
        @require_auth_and_role(Role.OPERATOR)
        def admin_function(data: str, current_user: str = None):
            return f"{current_user} processed {data}"
        
        token, _ = self.auth.create_access_token("operator_user")
        
        result = admin_function("test_data", authorization_header=f"Bearer {token}")
        assert "operator_user processed test_data" in result
    
    def test_require_auth_and_role_decorator_insufficient_role(self):
        """Test require_auth_and_role decorator with insufficient role"""
        @require_auth_and_role(Role.ADMIN)
        def admin_function(current_user: str = None):
            return "success"
        
        token, _ = self.auth.create_access_token("viewer_user")
        
        with pytest.raises(AccessDeniedError):
            admin_function(authorization_header=f"Bearer {token}")
    
    def test_require_auth_and_permission_decorator(self):
        """Test require_auth_and_permission decorator"""
        @require_auth_and_permission(Permission.EXECUTE_ACTIONS)
        def execute_action(action: str, current_user: str = None):
            return f"{current_user} executed {action}"
        
        token, _ = self.auth.create_access_token("operator_user")
        
        result = execute_action("test_action", authorization_header=f"Bearer {token}")
        assert "operator_user executed test_action" in result
    
    def test_require_auth_and_permission_decorator_no_permission(self):
        """Test require_auth_and_permission decorator without permission"""
        @require_auth_and_permission(Permission.ADMIN_OVERRIDE)
        def admin_action(current_user: str = None):
            return "success"
        
        token, _ = self.auth.create_access_token("viewer_user")
        
        with pytest.raises(AccessDeniedError):
            admin_action(authorization_header=f"Bearer {token}")


class TestAdminInterface:
    """Test cases for admin interface"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth = AuthManager(secret_key="test-secret")
        self.rbac = RBACManager()
        set_auth_manager(self.auth)
        set_rbac_manager(self.rbac)
        
        # Create admin user
        self.rbac.assign_role("admin", Role.ADMIN)
        self.admin = AdminInterface(admin_user_id="admin")
    
    def test_create_user(self):
        """Test user creation"""
        user_info = self.admin.create_user(
            "new_user",
            Role.OPERATOR,
            create_api_key=True,
            api_key_name="default_key"
        )
        
        assert user_info.user_id == "new_user"
        assert user_info.role == Role.OPERATOR
        assert len(user_info.api_keys) == 1
        assert user_info.api_keys[0].name == "default_key"
    
    def test_get_user(self):
        """Test getting user information"""
        self.rbac.assign_role("test_user", Role.VIEWER)
        
        user_info = self.admin.get_user("test_user")
        
        assert user_info.user_id == "test_user"
        assert user_info.role == Role.VIEWER
        assert Permission.READ_METRICS in user_info.permissions
    
    def test_list_users(self):
        """Test listing all users"""
        self.rbac.assign_role("user1", Role.OPERATOR)
        self.rbac.assign_role("user2", Role.VIEWER)
        
        users = self.admin.list_users()
        
        # Should include admin + user1 + user2
        assert len(users) >= 3
        user_ids = [u.user_id for u in users]
        assert "admin" in user_ids
        assert "user1" in user_ids
        assert "user2" in user_ids
    
    def test_delete_user(self):
        """Test user deletion"""
        # Create user with API key
        user_info = self.admin.create_user(
            "temp_user",
            Role.VIEWER,
            create_api_key=True
        )
        
        # Verify user exists
        assert self.rbac.get_role("temp_user") == Role.VIEWER
        
        # Delete user
        self.admin.delete_user("temp_user")
        
        # Verify user is deleted
        assert self.rbac.get_role("temp_user") is None
    
    def test_assign_role(self):
        """Test role assignment"""
        self.rbac.assign_role("test_user", Role.VIEWER)
        
        self.admin.assign_role("test_user", Role.OPERATOR)
        
        assert self.rbac.get_role("test_user") == Role.OPERATOR
    
    def test_grant_permission(self):
        """Test granting custom permission"""
        self.rbac.assign_role("test_user", Role.VIEWER)
        
        self.admin.grant_permission("test_user", Permission.EXECUTE_ACTIONS)
        
        user_info = self.admin.get_user("test_user")
        assert Permission.EXECUTE_ACTIONS in user_info.permissions
    
    def test_revoke_permission(self):
        """Test revoking custom permission"""
        self.rbac.assign_role("test_user", Role.VIEWER)
        self.admin.grant_permission("test_user", Permission.EXECUTE_ACTIONS)
        
        self.admin.revoke_permission("test_user", Permission.EXECUTE_ACTIONS)
        
        user_info = self.admin.get_user("test_user")
        # Should still have viewer permissions but not EXECUTE_ACTIONS
        assert Permission.READ_METRICS in user_info.permissions
        # Note: Custom permissions are separate from role permissions
        assert Permission.EXECUTE_ACTIONS not in user_info.custom_permissions
    
    def test_create_api_key(self):
        """Test API key creation"""
        self.rbac.assign_role("test_user", Role.OPERATOR)
        
        key_string, api_key = self.admin.create_api_key(
            "test_user",
            "Test Key",
            expires_in_days=30
        )
        
        assert isinstance(key_string, str)
        assert api_key.user_id == "test_user"
        assert api_key.name == "Test Key"
        assert api_key.expires_at is not None
    
    def test_revoke_api_key(self):
        """Test API key revocation"""
        self.rbac.assign_role("test_user", Role.OPERATOR)
        key_string, api_key = self.admin.create_api_key("test_user", "Test Key")
        
        # Verify key works
        user_id = self.auth.verify_api_key(key_string).user_id
        assert user_id == "test_user"
        
        # Revoke key
        self.admin.revoke_api_key(api_key.key_id)
        
        # Verify key no longer works
        from nethical.security.auth import InvalidTokenError
        with pytest.raises(InvalidTokenError):
            self.auth.verify_api_key(key_string)
    
    def test_create_tokens_for_user(self):
        """Test token creation for user"""
        self.rbac.assign_role("test_user", Role.OPERATOR)
        
        tokens = self.admin.create_tokens_for_user("test_user", scope="read write")
        
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        
        # Verify tokens work
        access_payload = self.auth.verify_token(tokens["access_token"])
        assert access_payload.user_id == "test_user"
        assert access_payload.scope == "read write"
    
    def test_get_access_history(self):
        """Test access history retrieval"""
        self.rbac.assign_role("test_user", Role.OPERATOR)
        
        # Generate some access checks
        self.rbac.check_role("test_user", Role.VIEWER, raise_on_deny=False)
        self.rbac.check_permission("test_user", Permission.READ_METRICS, raise_on_deny=False)
        
        history = self.admin.get_access_history("test_user")
        
        assert len(history) >= 2
        assert all(h["user_id"] == "test_user" for h in history)
    
    def test_get_system_summary(self):
        """Test system summary"""
        self.rbac.assign_role("user1", Role.ADMIN)
        self.rbac.assign_role("user2", Role.OPERATOR)
        self.rbac.assign_role("user3", Role.VIEWER)
        
        summary = self.admin.get_system_summary()
        
        assert "total_users" in summary
        assert "users_by_role" in summary
        assert summary["total_users"] >= 4  # admin + user1 + user2 + user3
        assert summary["users_by_role"][Role.ADMIN.value] >= 2  # admin + user1
    
    def test_non_admin_access_denied(self):
        """Test that non-admin users cannot use admin interface"""
        self.rbac.assign_role("viewer", Role.VIEWER)
        non_admin = AdminInterface(admin_user_id="viewer")
        
        with pytest.raises(AccessDeniedError):
            non_admin.create_user("new_user", Role.OPERATOR)


class TestIntegratedAuthFlow:
    """Integration tests for complete authentication flows"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.auth = AuthManager(secret_key="test-secret")
        self.rbac = RBACManager()
        set_auth_manager(self.auth)
        set_rbac_manager(self.rbac)
        
        # Create system admin
        self.rbac.assign_role("system_admin", Role.ADMIN)
        self.admin = AdminInterface(admin_user_id="system_admin")
    
    def test_complete_user_lifecycle(self):
        """Test complete user lifecycle from creation to deletion"""
        # 1. Create user
        user_info = self.admin.create_user(
            "lifecycle_user",
            Role.OPERATOR,
            create_api_key=True,
            api_key_name="initial_key"
        )
        assert user_info.role == Role.OPERATOR
        
        # 2. Create tokens for user
        tokens = self.admin.create_tokens_for_user("lifecycle_user")
        
        # 3. Use token to authenticate
        middleware = AuthMiddleware()
        user_id = middleware.authenticate(
            authorization_header=f"Bearer {tokens['access_token']}"
        )
        assert user_id == "lifecycle_user"
        
        # 4. Check permissions
        middleware.check_permission(user_id, Permission.EXECUTE_ACTIONS)
        
        # 5. Grant additional permission
        self.admin.grant_permission(user_id, Permission.READ_AUDIT_LOGS)
        
        # 6. Verify new permission
        user_info = self.admin.get_user(user_id)
        assert Permission.READ_AUDIT_LOGS in user_info.permissions
        
        # 7. Delete user
        self.admin.delete_user(user_id)
        
        # 8. Verify user is gone
        assert self.rbac.get_role(user_id) is None
    
    def test_api_protection_workflow(self):
        """Test protecting API endpoints with decorators"""
        # Create test users
        self.admin.create_user("api_admin", Role.ADMIN)
        self.admin.create_user("api_operator", Role.OPERATOR)
        self.admin.create_user("api_viewer", Role.VIEWER)
        
        # Define protected API endpoints
        @require_auth_and_role(Role.ADMIN)
        def delete_resource(resource_id: str, current_user: str = None):
            return f"Resource {resource_id} deleted by {current_user}"
        
        @require_auth_and_role(Role.OPERATOR)
        def update_resource(resource_id: str, data: dict, current_user: str = None):
            return f"Resource {resource_id} updated by {current_user}"
        
        @require_auth
        def read_resource(resource_id: str, current_user: str = None):
            return f"Resource {resource_id} read by {current_user}"
        
        # Get tokens for users
        admin_token = self.admin.create_tokens_for_user("api_admin")["access_token"]
        operator_token = self.admin.create_tokens_for_user("api_operator")["access_token"]
        viewer_token = self.admin.create_tokens_for_user("api_viewer")["access_token"]
        
        # Test admin can do everything
        result = delete_resource("res1", authorization_header=f"Bearer {admin_token}")
        assert "deleted by api_admin" in result
        
        result = update_resource("res1", {}, authorization_header=f"Bearer {admin_token}")
        assert "updated by api_admin" in result
        
        # Test operator can update but not delete
        result = update_resource("res1", {}, authorization_header=f"Bearer {operator_token}")
        assert "updated by api_operator" in result
        
        with pytest.raises(AccessDeniedError):
            delete_resource("res1", authorization_header=f"Bearer {operator_token}")
        
        # Test viewer can only read
        result = read_resource("res1", authorization_header=f"Bearer {viewer_token}")
        assert "read by api_viewer" in result
        
        with pytest.raises(AccessDeniedError):
            update_resource("res1", {}, authorization_header=f"Bearer {viewer_token}")
        
        with pytest.raises(AccessDeniedError):
            delete_resource("res1", authorization_header=f"Bearer {viewer_token}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
