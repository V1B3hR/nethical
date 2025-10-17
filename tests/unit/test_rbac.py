"""
Tests for RBAC (Role-Based Access Control) System
"""

import pytest
from datetime import datetime, timezone

from nethical.core.rbac import (
    Role,
    Permission,
    RBACManager,
    AccessDeniedError,
    require_role,
    require_permission,
    get_rbac_manager,
    set_rbac_manager,
)


class TestRBACManager:
    """Test cases for RBACManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.rbac = RBACManager()
        # Reset global instance
        set_rbac_manager(self.rbac)
    
    def test_initialization(self):
        """Test RBAC manager initialization"""
        assert isinstance(self.rbac, RBACManager)
        assert len(self.rbac.user_roles) == 0
        assert len(self.rbac.user_custom_permissions) == 0
    
    def test_assign_role(self):
        """Test role assignment"""
        self.rbac.assign_role("user1", Role.ADMIN)
        assert self.rbac.get_role("user1") == Role.ADMIN
        
        self.rbac.assign_role("user2", Role.VIEWER)
        assert self.rbac.get_role("user2") == Role.VIEWER
    
    def test_revoke_role(self):
        """Test role revocation"""
        self.rbac.assign_role("user1", Role.ADMIN)
        self.rbac.revoke_role("user1")
        assert self.rbac.get_role("user1") is None
    
    def test_role_hierarchy(self):
        """Test role hierarchy levels"""
        # Admin should have highest level
        assert self.rbac.ROLE_HIERARCHY[Role.ADMIN] > self.rbac.ROLE_HIERARCHY[Role.OPERATOR]
        assert self.rbac.ROLE_HIERARCHY[Role.OPERATOR] > self.rbac.ROLE_HIERARCHY[Role.AUDITOR]
        assert self.rbac.ROLE_HIERARCHY[Role.AUDITOR] > self.rbac.ROLE_HIERARCHY[Role.VIEWER]
    
    def test_has_role(self):
        """Test role checking with hierarchy"""
        self.rbac.assign_role("admin", Role.ADMIN)
        self.rbac.assign_role("operator", Role.OPERATOR)
        self.rbac.assign_role("viewer", Role.VIEWER)
        
        # Admin has all roles
        assert self.rbac.has_role("admin", Role.ADMIN)
        assert self.rbac.has_role("admin", Role.OPERATOR)
        assert self.rbac.has_role("admin", Role.VIEWER)
        
        # Operator has operator and below
        assert not self.rbac.has_role("operator", Role.ADMIN)
        assert self.rbac.has_role("operator", Role.OPERATOR)
        assert self.rbac.has_role("operator", Role.VIEWER)
        
        # Viewer only has viewer
        assert not self.rbac.has_role("viewer", Role.ADMIN)
        assert not self.rbac.has_role("viewer", Role.OPERATOR)
        assert self.rbac.has_role("viewer", Role.VIEWER)
    
    def test_role_permissions(self):
        """Test role-based permissions"""
        self.rbac.assign_role("admin", Role.ADMIN)
        self.rbac.assign_role("operator", Role.OPERATOR)
        self.rbac.assign_role("viewer", Role.VIEWER)
        
        # Admin has all permissions
        admin_perms = self.rbac.get_user_permissions("admin")
        assert Permission.ADMIN_OVERRIDE in admin_perms
        assert Permission.MANAGE_USERS in admin_perms
        
        # Operator has execution permissions but not admin
        operator_perms = self.rbac.get_user_permissions("operator")
        assert Permission.EXECUTE_ACTIONS in operator_perms
        assert Permission.ADMIN_OVERRIDE not in operator_perms
        
        # Viewer only has read permissions
        viewer_perms = self.rbac.get_user_permissions("viewer")
        assert Permission.READ_METRICS in viewer_perms
        assert Permission.EXECUTE_ACTIONS not in viewer_perms
    
    def test_custom_permissions(self):
        """Test custom permission grants"""
        self.rbac.assign_role("user1", Role.VIEWER)
        
        # Grant custom permission
        self.rbac.grant_permission("user1", Permission.EXECUTE_ACTIONS)
        
        perms = self.rbac.get_user_permissions("user1")
        assert Permission.EXECUTE_ACTIONS in perms
        
        # Revoke custom permission
        self.rbac.revoke_permission("user1", Permission.EXECUTE_ACTIONS)
        perms = self.rbac.get_user_permissions("user1")
        assert Permission.EXECUTE_ACTIONS not in perms
    
    def test_has_permission(self):
        """Test permission checking"""
        self.rbac.assign_role("operator", Role.OPERATOR)
        
        assert self.rbac.has_permission("operator", Permission.EXECUTE_ACTIONS)
        assert not self.rbac.has_permission("operator", Permission.ADMIN_OVERRIDE)
    
    def test_check_role_success(self):
        """Test successful role check"""
        self.rbac.assign_role("admin", Role.ADMIN)
        
        decision = self.rbac.check_role("admin", Role.OPERATOR, raise_on_deny=False)
        assert decision.allowed
        assert decision.user_id == "admin"
        assert decision.role == Role.ADMIN
    
    def test_check_role_denied(self):
        """Test denied role check"""
        self.rbac.assign_role("viewer", Role.VIEWER)
        
        decision = self.rbac.check_role("viewer", Role.ADMIN, raise_on_deny=False)
        assert not decision.allowed
        assert decision.user_id == "viewer"
        assert decision.role == Role.VIEWER
    
    def test_check_role_raises(self):
        """Test that denied role check raises exception"""
        self.rbac.assign_role("viewer", Role.VIEWER)
        
        with pytest.raises(AccessDeniedError) as exc_info:
            self.rbac.check_role("viewer", Role.ADMIN, raise_on_deny=True)
        
        assert exc_info.value.decision.user_id == "viewer"
    
    def test_check_permission_success(self):
        """Test successful permission check"""
        self.rbac.assign_role("operator", Role.OPERATOR)
        
        decision = self.rbac.check_permission(
            "operator", Permission.EXECUTE_ACTIONS, raise_on_deny=False
        )
        assert decision.allowed
    
    def test_check_permission_denied(self):
        """Test denied permission check"""
        self.rbac.assign_role("viewer", Role.VIEWER)
        
        decision = self.rbac.check_permission(
            "viewer", Permission.ADMIN_OVERRIDE, raise_on_deny=False
        )
        assert not decision.allowed
    
    def test_access_history(self):
        """Test access decision history tracking"""
        self.rbac.assign_role("user1", Role.ADMIN)
        
        self.rbac.check_role("user1", Role.OPERATOR, raise_on_deny=False)
        self.rbac.check_permission("user1", Permission.READ_METRICS, raise_on_deny=False)
        
        history = self.rbac.get_access_history()
        assert len(history) >= 2
        assert all(isinstance(d.timestamp, datetime) for d in history)
    
    def test_list_users(self):
        """Test listing users with their roles"""
        self.rbac.assign_role("admin", Role.ADMIN)
        self.rbac.assign_role("viewer", Role.VIEWER)
        self.rbac.grant_permission("viewer", Permission.EXECUTE_ACTIONS)
        
        users = self.rbac.list_users()
        assert "admin" in users
        assert "viewer" in users
        assert users["admin"]["role"] == Role.ADMIN.value
        assert Permission.EXECUTE_ACTIONS.value in users["viewer"]["custom_permissions"]
    
    def test_no_role_assigned(self):
        """Test behavior when user has no role"""
        decision = self.rbac.check_role("unknown_user", Role.VIEWER, raise_on_deny=False)
        assert not decision.allowed
        assert "no assigned role" in decision.reason.lower()


class TestRBACDecorators:
    """Test cases for RBAC decorators"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.rbac = RBACManager()
        set_rbac_manager(self.rbac)
        self.rbac.assign_role("admin", Role.ADMIN)
        self.rbac.assign_role("operator", Role.OPERATOR)
        self.rbac.assign_role("viewer", Role.VIEWER)
    
    def test_require_role_decorator(self):
        """Test require_role decorator"""
        @require_role(Role.ADMIN)
        def admin_function(current_user: str):
            return f"Success: {current_user}"
        
        # Should succeed for admin
        result = admin_function(current_user="admin")
        assert result == "Success: admin"
        
        # Should fail for viewer
        with pytest.raises(AccessDeniedError):
            admin_function(current_user="viewer")
    
    def test_require_permission_decorator(self):
        """Test require_permission decorator"""
        @require_permission(Permission.EXECUTE_ACTIONS)
        def execute_action(current_user: str):
            return f"Executed by {current_user}"
        
        # Should succeed for operator
        result = execute_action(current_user="operator")
        assert result == "Executed by operator"
        
        # Should fail for viewer
        with pytest.raises(AccessDeniedError):
            execute_action(current_user="viewer")
    
    def test_decorator_with_args(self):
        """Test decorator with function arguments"""
        @require_role(Role.OPERATOR)
        def process_data(data: str, current_user: str):
            return f"{current_user} processed {data}"
        
        result = process_data("test_data", current_user="operator")
        assert "operator processed test_data" in result


class TestRBACIntegration:
    """Integration tests for RBAC system"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.rbac = RBACManager()
        set_rbac_manager(self.rbac)
    
    def test_complete_workflow(self):
        """Test complete RBAC workflow"""
        # 1. Assign roles to users
        self.rbac.assign_role("alice", Role.ADMIN)
        self.rbac.assign_role("bob", Role.OPERATOR)
        self.rbac.assign_role("charlie", Role.VIEWER)
        
        # 2. Alice (admin) can do everything
        assert self.rbac.has_role("alice", Role.ADMIN)
        assert self.rbac.has_permission("alice", Permission.ADMIN_OVERRIDE)
        assert self.rbac.has_permission("alice", Permission.EXECUTE_ACTIONS)
        
        # 3. Bob (operator) can execute but not manage
        assert self.rbac.has_role("bob", Role.OPERATOR)
        assert self.rbac.has_permission("bob", Permission.EXECUTE_ACTIONS)
        assert not self.rbac.has_permission("bob", Permission.MANAGE_USERS)
        
        # 4. Charlie (viewer) can only read
        assert self.rbac.has_role("charlie", Role.VIEWER)
        assert self.rbac.has_permission("charlie", Permission.READ_METRICS)
        assert not self.rbac.has_permission("charlie", Permission.EXECUTE_ACTIONS)
        
        # 5. Grant Charlie custom permission
        self.rbac.grant_permission("charlie", Permission.READ_AUDIT_LOGS)
        assert self.rbac.has_permission("charlie", Permission.READ_AUDIT_LOGS)
        
        # 6. Check access history (need to perform access checks to generate history)
        self.rbac.check_role("alice", Role.ADMIN, raise_on_deny=False)
        self.rbac.check_permission("bob", Permission.EXECUTE_ACTIONS, raise_on_deny=False)
        history = self.rbac.get_access_history()
        assert len(history) > 0
    
    def test_role_promotion(self):
        """Test promoting user to higher role"""
        self.rbac.assign_role("user", Role.VIEWER)
        assert not self.rbac.has_permission("user", Permission.EXECUTE_ACTIONS)
        
        # Promote to operator
        self.rbac.assign_role("user", Role.OPERATOR)
        assert self.rbac.has_permission("user", Permission.EXECUTE_ACTIONS)
    
    def test_audit_logging(self):
        """Test that access decisions are logged"""
        logged_decisions = []
        
        def custom_logger(decision):
            logged_decisions.append(decision)
        
        rbac = RBACManager(audit_logger=custom_logger)
        rbac.assign_role("user", Role.VIEWER)
        
        rbac.check_role("user", Role.VIEWER, raise_on_deny=False)
        
        assert len(logged_decisions) > 0
        assert logged_decisions[0].user_id == "user"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
