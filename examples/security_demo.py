"""
Example: Using RBAC and Authentication in Nethical

This example demonstrates how to use the new RBAC and authentication
features implemented in Phase 1.
"""

from datetime import datetime, timedelta, timezone

from nethical.core.rbac import (
    RBACManager,
    Role,
    Permission,
    require_role,
    require_permission,
    get_rbac_manager,
    set_rbac_manager,
)
from nethical.security.auth import (
    AuthManager,
    authenticate_request,
    AuthenticationError,
    get_auth_manager,
    set_auth_manager,
)

# Global instances for demo consistency
_demo_rbac = None
_demo_auth = None


def get_demo_rbac():
    """Get demo RBAC manager"""
    global _demo_rbac
    if _demo_rbac is None:
        _demo_rbac = RBACManager()
        set_rbac_manager(_demo_rbac)
    return _demo_rbac


def get_demo_auth():
    """Get demo auth manager"""
    global _demo_auth
    if _demo_auth is None:
        _demo_auth = AuthManager(secret_key="demo-secret-key")
        set_auth_manager(_demo_auth)
    return _demo_auth


def example_rbac_basic():
    """Example 1: Basic RBAC usage"""
    print("=" * 60)
    print("Example 1: Basic RBAC Usage")
    print("=" * 60)

    # Create RBAC manager
    rbac = get_demo_rbac()

    # Assign roles to users
    rbac.assign_role("alice", Role.ADMIN)
    rbac.assign_role("bob", Role.OPERATOR)
    rbac.assign_role("charlie", Role.VIEWER)

    print("\n1. Role Assignments:")
    print(f"   Alice: {rbac.get_role('alice')}")
    print(f"   Bob: {rbac.get_role('bob')}")
    print(f"   Charlie: {rbac.get_role('charlie')}")

    # Check permissions
    print("\n2. Permission Checks:")
    print(
        f"   Can Alice manage users? {rbac.has_permission('alice', Permission.MANAGE_USERS)}"
    )
    print(
        f"   Can Bob execute actions? {rbac.has_permission('bob', Permission.EXECUTE_ACTIONS)}"
    )
    print(
        f"   Can Charlie execute actions? {rbac.has_permission('charlie', Permission.EXECUTE_ACTIONS)}"
    )

    # Check role hierarchy
    print("\n3. Role Hierarchy:")
    print(
        f"   Does Alice have operator privileges? {rbac.has_role('alice', Role.OPERATOR)}"
    )
    print(f"   Does Bob have admin privileges? {rbac.has_role('bob', Role.ADMIN)}")

    print()


def example_rbac_decorators():
    """Example 2: Using RBAC decorators"""
    print("=" * 60)
    print("Example 2: Using RBAC Decorators")
    print("=" * 60)

    # Create RBAC manager
    rbac = get_demo_rbac()
    rbac.assign_role("admin_user", Role.ADMIN)
    rbac.assign_role("regular_user", Role.VIEWER)

    # Define protected functions
    @require_role(Role.ADMIN)
    def delete_policy(policy_id: str, current_user: str):
        return f"Policy {policy_id} deleted by {current_user}"

    @require_permission(Permission.EXECUTE_ACTIONS)
    def execute_action(action_id: str, current_user: str):
        return f"Action {action_id} executed by {current_user}"

    print("\n1. Admin-only function:")
    try:
        result = delete_policy("policy_123", current_user="admin_user")
        print(f"   ✅ Success: {result}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    try:
        result = delete_policy("policy_123", current_user="regular_user")
        print(f"   ✅ Success: {result}")
    except Exception as e:
        print(f"   ❌ Expected failure: Access denied for regular_user")

    print()


def example_authentication():
    """Example 3: JWT Authentication"""
    print("=" * 60)
    print("Example 3: JWT Authentication")
    print("=" * 60)

    # Create auth manager
    auth = get_demo_auth()

    # Create tokens for a user
    access_token, access_payload = auth.create_access_token(
        "user123", scope="read write"
    )
    refresh_token, refresh_payload = auth.create_refresh_token("user123")

    print("\n1. Token Creation:")
    print(f"   Access token created (expires in {auth.access_token_expiry})")
    print(f"   Token type: {access_payload.token_type}")
    print(f"   Token scope: {access_payload.scope}")
    print(f"   Token ID: {access_payload.jti}")

    # Verify token
    print("\n2. Token Verification:")
    try:
        verified = auth.verify_token(access_token)
        print(f"   ✅ Token valid for user: {verified.user_id}")
        print(f"   Expires at: {verified.expires_at}")
    except Exception as e:
        print(f"   ❌ Token invalid: {e}")

    # Use refresh token
    print("\n3. Token Refresh:")
    new_access_token, new_payload = auth.refresh_access_token(refresh_token)
    print(f"   ✅ New access token created")
    print(f"   New token ID: {new_payload.jti}")

    print()


def example_api_keys():
    """Example 4: API Key Management"""
    print("=" * 60)
    print("Example 4: API Key Management")
    print("=" * 60)

    # Create auth manager
    auth = get_demo_auth()

    # Create API key
    expires_at = datetime.now(timezone.utc) + timedelta(days=90)
    api_key_string, api_key = auth.create_api_key(
        user_id="service_account", name="Production API Key", expires_at=expires_at
    )

    print("\n1. API Key Creation:")
    print(f"   Key ID: {api_key.key_id}")
    print(f"   User: {api_key.user_id}")
    print(f"   Name: {api_key.name}")
    print(f"   Expires: {api_key.expires_at}")
    print(
        f"   API key has been generated. (Value is not shown for security reasons)"
    )  # FIXED

    # Verify API key
    print("\n2. API Key Verification:")
    try:
        verified = auth.verify_api_key(api_key_string)
        print(f"   ✅ API key valid for user: {verified.user_id}")
        print(f"   Last used: {verified.last_used_at}")
    except Exception as e:
        print(f"   ❌ API key invalid: {e}")

    # List all API keys
    print("\n3. List API Keys:")
    keys = auth.list_api_keys(user_id="service_account")
    print(f"   Found {len(keys)} keys for service_account")
    for key in keys:
        print(f"   - {key.name} (ID: {key.key_id}, Enabled: {key.enabled})")

    print()


def example_request_authentication():
    """Example 5: Authenticating HTTP Requests"""
    print("=" * 60)
    print("Example 5: Authenticating HTTP Requests")
    print("=" * 60)

    # Create auth manager
    auth = get_demo_auth()

    # Create token
    token, _ = auth.create_access_token("web_user")

    # Simulate HTTP request authentication
    print("\n1. JWT Bearer Token:")
    try:
        user_id = authenticate_request(authorization_header=f"Bearer {token}")
        print(f"   ✅ Authenticated as: {user_id}")
    except AuthenticationError as e:
        print(f"   ❌ Authentication failed: {e}")

    # Create API key
    api_key_string, _ = auth.create_api_key("api_client", "Client API Key")

    print("\n2. API Key:")
    try:
        user_id = authenticate_request(api_key_header=api_key_string)
        print(f"   ✅ Authenticated as: {user_id}")
    except AuthenticationError as e:
        print(f"   ❌ Authentication failed: {e}")

    print("\n3. No Credentials:")
    try:
        user_id = authenticate_request()
        print(f"   ✅ Authenticated as: {user_id}")
    except AuthenticationError as e:
        print(f"   ❌ Expected failure: {e}")

    print()


def example_integrated():
    """Example 6: Integrated RBAC + Auth"""
    print("=" * 60)
    print("Example 6: Integrated RBAC + Authentication")
    print("=" * 60)

    # Setup
    rbac = get_demo_rbac()
    auth = get_demo_auth()

    # Create user with role
    user_id = "integrated_user"
    rbac.assign_role(user_id, Role.OPERATOR)

    # Create token for user
    token, _ = auth.create_access_token(user_id)

    print("\n1. User Setup:")
    print(f"   User ID: {user_id}")
    print(f"   Role: {rbac.get_role(user_id)}")
    print(f"   Token created: ✅")

    # Simulate authenticated API call with permission check
    @require_permission(Permission.EXECUTE_ACTIONS)
    def protected_operation(data: str, current_user: str):
        return f"Operation successful: {data} by {current_user}"

    print("\n2. Protected API Call:")
    try:
        # Authenticate request
        authenticated_user = authenticate_request(
            authorization_header=f"Bearer {token}"
        )
        print(f"   ✅ Authentication: {authenticated_user}")

        # Execute protected operation
        result = protected_operation("test_data", current_user=authenticated_user)
        print(f"   ✅ Authorization: {result}")

    except (AuthenticationError, PermissionError) as e:
        print(f"   ❌ Failed: {e}")

    print()


def main():
    """Run all examples"""
    print("\n")
    print("*" * 60)
    print("* Nethical Phase 1 Security Features Demo")
    print("*" * 60)
    print()

    example_rbac_basic()
    example_rbac_decorators()
    example_authentication()
    example_api_keys()
    example_request_authentication()
    example_integrated()

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
