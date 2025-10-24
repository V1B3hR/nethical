"""
Security and Access Control Demo for Nethical

This example demonstrates the complete RBAC and authentication system including:
- User creation and management
- Role-based access control
- API endpoint protection
- Admin interface usage
- Token and API key management
"""

from nethical.security import (
    AuthManager,
    AdminInterface,
    AuthMiddleware,
    require_auth,
    require_auth_and_role,
    require_auth_and_permission,
    set_auth_manager,
)
from nethical.core import (
    Role,
    Permission,
    RBACManager,
    set_rbac_manager,
    AccessDeniedError,
)


def print_section(title: str) -> None:
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_basic_auth():
    """Demonstrate basic authentication"""
    print_section("1. Basic Authentication")
    
    # Initialize auth manager
    auth = AuthManager(secret_key="demo-secret-key")
    set_auth_manager(auth)
    
    # Create access and refresh tokens
    access_token, access_payload = auth.create_access_token("alice", scope="read write")
    refresh_token, _ = auth.create_refresh_token("alice")
    
    print(f"✓ Created access token for 'alice'")
    print(f"  Token expires at: {access_payload.expires_at}")
    print(f"  Scope: {access_payload.scope}")
    
    # Verify token
    verified = auth.verify_token(access_token)
    print(f"✓ Token verified for user: {verified.user_id}")
    
    # Create API key
    api_key_string, api_key = auth.create_api_key(
        "alice",
        "Alice's API Key",
        expires_at=None
    )
    print(f"✓ Created API key: {api_key.key_id}")
    print(f"  Key name: {api_key.name}")


def demo_rbac_system():
    """Demonstrate RBAC system"""
    print_section("2. Role-Based Access Control")
    
    # Initialize RBAC manager
    rbac = RBACManager()
    set_rbac_manager(rbac)
    
    # Assign roles to users
    rbac.assign_role("alice", Role.ADMIN)
    rbac.assign_role("bob", Role.OPERATOR)
    rbac.assign_role("charlie", Role.AUDITOR)
    rbac.assign_role("diana", Role.VIEWER)
    
    print("✓ Assigned roles:")
    print(f"  alice: {Role.ADMIN.value}")
    print(f"  bob: {Role.OPERATOR.value}")
    print(f"  charlie: {Role.AUDITOR.value}")
    print(f"  diana: {Role.VIEWER.value}")
    
    # Check permissions
    print("\n✓ Permission checks:")
    print(f"  alice can ADMIN_OVERRIDE: {rbac.has_permission('alice', Permission.ADMIN_OVERRIDE)}")
    print(f"  bob can EXECUTE_ACTIONS: {rbac.has_permission('bob', Permission.EXECUTE_ACTIONS)}")
    print(f"  charlie can READ_AUDIT_LOGS: {rbac.has_permission('charlie', Permission.READ_AUDIT_LOGS)}")
    print(f"  diana can READ_METRICS: {rbac.has_permission('diana', Permission.READ_METRICS)}")
    print(f"  diana can EXECUTE_ACTIONS: {rbac.has_permission('diana', Permission.EXECUTE_ACTIONS)}")
    
    # Grant custom permission
    rbac.grant_permission("diana", Permission.READ_AUDIT_LOGS)
    print(f"\n✓ Granted custom permission READ_AUDIT_LOGS to diana")
    print(f"  diana now has: {len(rbac.get_user_permissions('diana'))} permissions")


def demo_admin_interface():
    """Demonstrate admin interface"""
    print_section("3. Admin Interface")
    
    # Create admin interface
    admin = AdminInterface(admin_user_id="system")
    
    # Create a new user
    user_info = admin.create_user(
        "new_operator",
        Role.OPERATOR,
        create_api_key=True,
        api_key_name="Operator Key"
    )
    print(f"✓ Created user: {user_info.user_id}")
    print(f"  Role: {user_info.role.value}")
    print(f"  Permissions: {len(user_info.permissions)}")
    print(f"  API keys: {len(user_info.api_keys)}")
    
    # Create tokens for user
    tokens = admin.create_tokens_for_user("new_operator")
    print(f"\n✓ Created tokens for new_operator")
    print(f"  Access token length: {len(tokens['access_token'])}")
    print(f"  Refresh token length: {len(tokens['refresh_token'])}")
    
    # Grant additional permission
    admin.grant_permission("new_operator", Permission.READ_AUDIT_LOGS)
    print(f"\n✓ Granted additional permission to new_operator")
    
    # Get system summary
    summary = admin.get_system_summary()
    print(f"\n✓ System summary:")
    print(f"  Total users: {summary['total_users']}")
    print(f"  Users by role: {summary['users_by_role']}")
    print(f"  Active API keys: {summary['active_api_keys']}")


def demo_api_protection():
    """Demonstrate API endpoint protection"""
    print_section("4. API Endpoint Protection")
    
    # Initialize managers
    auth = AuthManager(secret_key="demo-secret")
    rbac = RBACManager()
    set_auth_manager(auth)
    set_rbac_manager(rbac)
    
    # Set up users
    rbac.assign_role("admin", Role.ADMIN)
    rbac.assign_role("operator", Role.OPERATOR)
    rbac.assign_role("viewer", Role.VIEWER)
    
    # Define protected API functions
    @require_auth
    def get_metrics(current_user: str = None):
        return f"Metrics accessed by {current_user}"
    
    @require_auth_and_role(Role.OPERATOR)
    def execute_action(action_name: str, current_user: str = None):
        return f"{current_user} executed action: {action_name}"
    
    @require_auth_and_role(Role.ADMIN)
    def delete_policy(policy_id: str, current_user: str = None):
        return f"{current_user} deleted policy: {policy_id}"
    
    @require_auth_and_permission(Permission.MANAGE_QUARANTINE)
    def manage_quarantine(agent_id: str, current_user: str = None):
        return f"{current_user} managed quarantine for agent: {agent_id}"
    
    # Test with different users
    admin_token, _ = auth.create_access_token("admin")
    operator_token, _ = auth.create_access_token("operator")
    viewer_token, _ = auth.create_access_token("viewer")
    
    print("✓ Testing API access with different roles:\n")
    
    # Viewer can read metrics
    try:
        result = get_metrics(authorization_header=f"Bearer {viewer_token}")
        print(f"  [viewer] get_metrics: ✓ {result}")
    except Exception as e:
        print(f"  [viewer] get_metrics: ✗ {type(e).__name__}")
    
    # Viewer cannot execute actions
    try:
        result = execute_action("test", authorization_header=f"Bearer {viewer_token}")
        print(f"  [viewer] execute_action: ✓ {result}")
    except AccessDeniedError:
        print(f"  [viewer] execute_action: ✗ Access denied (expected)")
    
    # Operator can execute actions
    try:
        result = execute_action("test", authorization_header=f"Bearer {operator_token}")
        print(f"  [operator] execute_action: ✓ {result}")
    except Exception as e:
        print(f"  [operator] execute_action: ✗ {type(e).__name__}")
    
    # Operator can manage quarantine
    try:
        result = manage_quarantine("agent123", authorization_header=f"Bearer {operator_token}")
        print(f"  [operator] manage_quarantine: ✓ {result}")
    except Exception as e:
        print(f"  [operator] manage_quarantine: ✗ {type(e).__name__}")
    
    # Operator cannot delete policies
    try:
        result = delete_policy("policy1", authorization_header=f"Bearer {operator_token}")
        print(f"  [operator] delete_policy: ✓ {result}")
    except AccessDeniedError:
        print(f"  [operator] delete_policy: ✗ Access denied (expected)")
    
    # Admin can do everything
    try:
        result = delete_policy("policy1", authorization_header=f"Bearer {admin_token}")
        print(f"  [admin] delete_policy: ✓ {result}")
    except Exception as e:
        print(f"  [admin] delete_policy: ✗ {type(e).__name__}")


def demo_middleware():
    """Demonstrate middleware usage"""
    print_section("5. Middleware Usage")
    
    # Initialize managers
    auth = AuthManager(secret_key="demo-secret")
    rbac = RBACManager()
    set_auth_manager(auth)
    set_rbac_manager(rbac)
    
    rbac.assign_role("admin", Role.ADMIN)
    
    # Create middleware
    middleware = AuthMiddleware()
    
    # Create token
    token, _ = auth.create_access_token("admin")
    
    # Process request with authentication only
    user_id = middleware.process_request(
        authorization_header=f"Bearer {token}"
    )
    print(f"✓ Authenticated user: {user_id}")
    
    # Process request with role check
    user_id = middleware.process_request(
        authorization_header=f"Bearer {token}",
        required_role=Role.OPERATOR
    )
    print(f"✓ Authenticated and authorized (role): {user_id}")
    
    # Process request with permission check
    user_id = middleware.process_request(
        authorization_header=f"Bearer {token}",
        required_permission=Permission.ADMIN_OVERRIDE
    )
    print(f"✓ Authenticated and authorized (permission): {user_id}")


def demo_audit_trail():
    """Demonstrate audit trail"""
    print_section("6. Audit Trail")
    
    # Initialize managers
    auth = AuthManager(secret_key="demo-secret")
    rbac = RBACManager()
    admin = AdminInterface(admin_user_id="system")
    set_auth_manager(auth)
    set_rbac_manager(rbac)
    
    # Create users
    rbac.assign_role("alice", Role.ADMIN)
    rbac.assign_role("bob", Role.VIEWER)
    
    # Perform various access checks
    rbac.check_role("alice", Role.ADMIN, raise_on_deny=False)
    rbac.check_permission("alice", Permission.ADMIN_OVERRIDE, raise_on_deny=False)
    rbac.check_role("bob", Role.ADMIN, raise_on_deny=False)  # Will be denied
    rbac.check_permission("bob", Permission.READ_METRICS, raise_on_deny=False)
    
    # Get audit history
    history = admin.get_access_history(limit=10)
    
    print(f"✓ Access history (last {len(history)} entries):\n")
    for entry in history[-5:]:  # Show last 5
        status = "✓ ALLOWED" if entry["allowed"] else "✗ DENIED"
        print(f"  {status}")
        print(f"    User: {entry['user_id']}")
        print(f"    Role: {entry['role']}")
        if entry['required_role']:
            print(f"    Required role: {entry['required_role']}")
        if entry['required_permission']:
            print(f"    Required permission: {entry['required_permission']}")
        print(f"    Reason: {entry['reason']}")
        print()


def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("  NETHICAL SECURITY & ACCESS CONTROL DEMO")
    print("="*60)
    print("\nThis demo showcases the complete RBAC and authentication system.")
    
    try:
        demo_basic_auth()
        demo_rbac_system()
        demo_admin_interface()
        demo_api_protection()
        demo_middleware()
        demo_audit_trail()
        
        print_section("Demo Complete")
        print("\n✓ All components demonstrated successfully!")
        print("\nKey Features:")
        print("  • JWT and API key authentication")
        print("  • 4-tier role hierarchy (admin, operator, auditor, viewer)")
        print("  • 16+ fine-grained permissions")
        print("  • Decorator-based API protection")
        print("  • Admin interface for user management")
        print("  • Comprehensive audit logging")
        print()
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
