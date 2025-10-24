# RBAC Quick Start Guide

This guide gets you started with Nethical's RBAC and authentication system in 5 minutes.

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Initialize the System

```python
from nethical.security import AuthManager, set_auth_manager
from nethical.core import RBACManager, set_rbac_manager

# Initialize managers
auth = AuthManager(secret_key="your-secret-key")
rbac = RBACManager()

# Set global instances
set_auth_manager(auth)
set_rbac_manager(rbac)
```

### 2. Create Users and Assign Roles

```python
from nethical.security import AdminInterface
from nethical.core import Role

# Create admin interface
admin = AdminInterface(admin_user_id="system")

# Create users with different roles
admin.create_user("admin_user", Role.ADMIN, create_api_key=True)
admin.create_user("operator_user", Role.OPERATOR, create_api_key=True)
admin.create_user("viewer_user", Role.VIEWER, create_api_key=True)
```

### 3. Protect Your API Endpoints

```python
from nethical.security import require_auth_and_role, require_auth_and_permission
from nethical.core import Role, Permission

# Require admin role
@require_auth_and_role(Role.ADMIN)
def delete_policy(policy_id: str, current_user: str = None):
    return f"Policy {policy_id} deleted by {current_user}"

# Require specific permission
@require_auth_and_permission(Permission.EXECUTE_ACTIONS)
def execute_action(action: str, current_user: str = None):
    return f"{current_user} executed {action}"

# Just authentication
@require_auth
def get_metrics(current_user: str = None):
    return f"Metrics for {current_user}"
```

### 4. Use the Protected Endpoints

```python
# Get a token for a user
tokens = admin.create_tokens_for_user("operator_user")
access_token = tokens["access_token"]

# Call protected endpoint
result = execute_action(
    "test_action",
    authorization_header=f"Bearer {access_token}"
)
```

## Role Hierarchy

From highest to lowest privilege:

1. **Admin** - Full system control
2. **Operator** - Execute actions, manage quarantine
3. **Auditor** - Read-only access to logs and violations
4. **Viewer** - Basic read access to metrics and policies

## Common Tasks

### Create an API Key

```python
key_string, api_key = admin.create_api_key(
    user_id="service_account",
    key_name="Production Key",
    expires_in_days=90
)
# Save key_string securely - it won't be shown again!
```

### Grant Custom Permission

```python
from nethical.core import Permission

admin.grant_permission("viewer_user", Permission.READ_AUDIT_LOGS)
```

### Check User Permissions

```python
user_info = admin.get_user("operator_user")
print(f"Role: {user_info.role}")
print(f"Permissions: {len(user_info.permissions)}")
```

### View Audit Trail

```python
history = admin.get_access_history(user_id="operator_user", limit=10)
for entry in history:
    print(f"{entry['timestamp']}: {entry['reason']}")
```

## Running the Demo

```bash
python examples/security_rbac_demo.py
```

## Testing

```bash
# Run all tests
pytest tests/unit/test_auth.py tests/unit/test_rbac.py tests/integration/test_auth_rbac_integration.py -v

# Quick test
python -m pytest tests/integration/test_auth_rbac_integration.py::TestIntegratedAuthFlow::test_api_protection_workflow -v
```

## Documentation

- Full guide: `docs/security/RBAC_IMPLEMENTATION.md`
- Threat model: `docs/security/threat_model.md`
- Examples: `examples/security_rbac_demo.py`

## Next Steps

1. Review the full implementation guide
2. Run the demo to see all features
3. Integrate with your API framework
4. Set up audit log persistence
5. Configure production secret keys

For more information, see the [full implementation guide](RBAC_IMPLEMENTATION.md).
