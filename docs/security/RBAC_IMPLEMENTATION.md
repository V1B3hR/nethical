# Phase 1: RBAC and Access Control - Implementation Guide

## Overview

This guide documents the implementation of Role-Based Access Control (RBAC) and authentication middleware for the Nethical system, completing Phase 1 of the security enhancements.

## Components

### 1. Authentication System (`nethical/security/auth.py`)

The authentication system provides JWT token and API key management:

- **JWT Tokens**: HS256-signed access and refresh tokens
- **API Keys**: SHA-256 hashed keys for service-to-service authentication
- **Token Management**: Creation, verification, revocation, and refresh
- **API Key Management**: Create, verify, revoke, list keys with expiration

**Key Classes:**
- `AuthManager`: Central authentication manager
- `TokenPayload`: JWT token payload representation
- `APIKey`: API key representation

**Example:**
```python
from nethical.security import AuthManager, set_auth_manager

# Initialize
auth = AuthManager(secret_key="your-secret-key")
set_auth_manager(auth)

# Create tokens
access_token, payload = auth.create_access_token("user123", scope="read write")
refresh_token, _ = auth.create_refresh_token("user123")

# Verify token
payload = auth.verify_token(access_token)

# Create API key
api_key_string, api_key = auth.create_api_key("user123", "My API Key")
```

### 2. RBAC System (`nethical/core/rbac.py`)

The RBAC system implements a 4-tier role hierarchy with fine-grained permissions:

**Role Hierarchy:**
1. **Admin**: Full system control (highest privileges)
2. **Operator**: Execute actions and manage quarantine
3. **Auditor**: Read-only access to logs and violations
4. **Viewer**: Basic read access to metrics and policies (lowest privileges)

**Permissions (16+ types):**
- Read: `READ_POLICIES`, `READ_ACTIONS`, `READ_VIOLATIONS`, `READ_METRICS`, `READ_AUDIT_LOGS`
- Write: `WRITE_POLICIES`, `WRITE_ACTIONS`, `EXECUTE_ACTIONS`
- Management: `MANAGE_USERS`, `MANAGE_ROLES`, `MANAGE_QUARANTINE`, `MANAGE_SYSTEM`
- Administrative: `ADMIN_OVERRIDE`, `SYSTEM_CONFIG`

**Key Classes:**
- `RBACManager`: Central RBAC manager
- `Role`: Role enumeration
- `Permission`: Permission enumeration
- `AccessDecision`: Result of access control check

**Decorators:**
- `@require_role(role)`: Require specific role
- `@require_permission(permission)`: Require specific permission

**Example:**
```python
from nethical.core import RBACManager, Role, Permission, require_role

# Initialize
rbac = RBACManager()

# Assign roles
rbac.assign_role("alice", Role.ADMIN)
rbac.assign_role("bob", Role.OPERATOR)

# Check permissions
if rbac.has_permission("bob", Permission.EXECUTE_ACTIONS):
    print("Bob can execute actions")

# Use decorators
@require_role(Role.ADMIN)
def delete_user(user_id: str, current_user: str):
    return f"User {user_id} deleted by {current_user}"
```

### 3. API Middleware (`nethical/security/middleware.py`)

The middleware combines authentication and authorization for API protection:

**Key Classes:**
- `AuthMiddleware`: Framework-agnostic middleware

**Decorators:**
- `@require_auth`: Require authentication only
- `@require_auth_and_role(role)`: Require auth + specific role
- `@require_auth_and_permission(permission)`: Require auth + specific permission

**Example:**
```python
from nethical.security import require_auth_and_role
from nethical.core import Role

@require_auth_and_role(Role.OPERATOR)
def execute_action(action_name: str, current_user: str = None):
    return f"{current_user} executed {action_name}"

# Call with authentication
token = "..."  # JWT token
result = execute_action("test", authorization_header=f"Bearer {token}")
```

### 4. Admin Interface (`nethical/security/admin.py`)

The admin interface provides user and role management functionality:

**Key Classes:**
- `AdminInterface`: Administrative operations
- `UserInfo`: User information representation

**Features:**
- User lifecycle management (create, read, update, delete)
- Role assignment and revocation
- Custom permission grants
- API key management
- Token creation for users
- Access history and system reporting
- Admin-only operations with access control

**Example:**
```python
from nethical.security import AdminInterface
from nethical.core import Role

# Initialize (requires admin privileges)
admin = AdminInterface(admin_user_id="admin_user")

# Create user
user_info = admin.create_user(
    "new_user",
    Role.OPERATOR,
    create_api_key=True,
    api_key_name="Default Key"
)

# Create tokens for user
tokens = admin.create_tokens_for_user("new_user")

# Grant additional permission
admin.grant_permission("new_user", Permission.READ_AUDIT_LOGS)

# Get system summary
summary = admin.get_system_summary()
print(f"Total users: {summary['total_users']}")
```

## Integration Patterns

### Pattern 1: Protecting API Endpoints

```python
from nethical.security import require_auth_and_role
from nethical.core import Role, Permission

# Role-based protection
@require_auth_and_role(Role.ADMIN)
def delete_policy(policy_id: str, current_user: str = None):
    # Only admins can delete policies
    return f"Policy {policy_id} deleted by {current_user}"

# Permission-based protection
@require_auth_and_permission(Permission.MANAGE_QUARANTINE)
def manage_quarantine(agent_id: str, current_user: str = None):
    # Anyone with MANAGE_QUARANTINE permission can call this
    return f"Quarantine managed by {current_user}"
```

### Pattern 2: Manual Authorization Checks

```python
from nethical.security import AuthMiddleware
from nethical.core import Role, Permission

middleware = AuthMiddleware()

def handle_request(auth_header: str):
    # Authenticate
    user_id = middleware.authenticate(authorization_header=auth_header)
    
    # Check authorization
    middleware.check_role(user_id, Role.OPERATOR)
    middleware.check_permission(user_id, Permission.EXECUTE_ACTIONS)
    
    # Process request
    return f"Request processed by {user_id}"
```

### Pattern 3: Admin Operations

```python
from nethical.security import AdminInterface
from nethical.core import Role

# Initialize with admin user
admin = AdminInterface(admin_user_id="system_admin")

# Complete user setup workflow
user_info = admin.create_user("service_account", Role.OPERATOR)
tokens = admin.create_tokens_for_user("service_account")
api_key, _ = admin.create_api_key("service_account", "Service Key")

# Grant additional permissions as needed
admin.grant_permission("service_account", Permission.READ_AUDIT_LOGS)
```

## Security Considerations

### 1. Token Security
- Store tokens securely (e.g., HTTP-only cookies, secure storage)
- Use HTTPS for all token transmission
- Implement token rotation and refresh mechanisms
- Set appropriate expiration times (default: 1 hour access, 7 days refresh)

### 2. API Key Security
- API keys are hashed with SHA-256 before storage
- Keys should be treated as secrets and transmitted securely
- Implement key rotation policies
- Set expiration dates for keys when possible

### 3. Role Assignment
- Follow principle of least privilege
- Regularly audit role assignments
- Use custom permissions sparingly and document them
- Implement approval workflows for role changes

### 4. Audit Logging
- All access decisions are logged automatically
- Review audit logs regularly
- Set up alerts for suspicious patterns
- Retain logs according to compliance requirements

## Testing

The implementation includes comprehensive test coverage:

- **Unit Tests**: 52 tests for Auth and RBAC systems
  - `tests/unit/test_auth.py`: Authentication system tests
  - `tests/unit/test_rbac.py`: RBAC system tests

- **Integration Tests**: 28 tests for middleware and admin interface
  - `tests/integration/test_auth_rbac_integration.py`: End-to-end tests

**Run tests:**
```bash
# Run all auth and RBAC tests
pytest tests/unit/test_auth.py tests/unit/test_rbac.py tests/integration/test_auth_rbac_integration.py -v

# Run with coverage
pytest tests/unit/test_auth.py tests/unit/test_rbac.py tests/integration/test_auth_rbac_integration.py --cov=nethical.security --cov=nethical.core.rbac
```

## Example Usage

See `examples/security_rbac_demo.py` for a complete demonstration of all features:

```bash
python examples/security_rbac_demo.py
```

The demo showcases:
1. Basic authentication (tokens and API keys)
2. Role-based access control
3. Admin interface operations
4. API endpoint protection
5. Middleware usage
6. Audit trail

## Performance Considerations

- Token verification is fast (HMAC comparison)
- RBAC checks are in-memory operations
- Audit logs are stored in memory by default (implement persistent storage for production)
- Consider caching permission lookups for high-frequency checks

## Migration Guide

If you have existing authentication/authorization:

1. **Initialize managers:**
   ```python
   from nethical.security import AuthManager, set_auth_manager
   from nethical.core import RBACManager, set_rbac_manager
   
   auth = AuthManager(secret_key="your-secret")
   rbac = RBACManager()
   set_auth_manager(auth)
   set_rbac_manager(rbac)
   ```

2. **Migrate users:**
   ```python
   from nethical.security import AdminInterface
   
   admin = AdminInterface(admin_user_id="system")
   for user in existing_users:
       admin.create_user(user.id, user.role)
   ```

3. **Update API endpoints:**
   ```python
   # Before
   def my_api_function(user_id: str):
       # manual auth check
       pass
   
   # After
   from nethical.security import require_auth_and_role
   from nethical.core import Role
   
   @require_auth_and_role(Role.OPERATOR)
   def my_api_function(current_user: str = None):
       # auth handled by decorator
       pass
   ```

## Compliance

This implementation supports compliance with:

- **NIST AI RMF**: Access control and audit logging
- **OWASP Top 10**: Authentication and authorization best practices
- **GDPR/CCPA**: Audit trails and access controls
- **SOC 2**: User management and access logging

See `docs/security/threat_model.md` for detailed security analysis.

## Future Enhancements

Potential improvements for future phases:

1. **Multi-factor Authentication (MFA)**: Add TOTP/SMS verification
2. **OAuth2/OIDC**: Support external identity providers
3. **Session Management**: Implement session tracking and timeout
4. **Permission Inheritance**: Support permission inheritance beyond roles
5. **Dynamic Permissions**: Runtime permission computation
6. **Audit Log Persistence**: Database backend for audit logs
7. **Role Templates**: Predefined role configurations
8. **Compliance Reports**: Automated compliance reporting

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: `docs/security/`
- Examples: `examples/security_rbac_demo.py`
