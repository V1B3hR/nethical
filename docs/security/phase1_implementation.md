# Phase 1: Security and Governance Implementation

## Overview

This document describes the Phase 1 security and governance enhancements implemented in the Nethical system.

## 1. Role-Based Access Control (RBAC)

### Location
`nethical/core/rbac.py`

### Role Hierarchy

The system implements a four-tier role hierarchy:

1. **Viewer** (Level 0) - Read-only access
   - Read policies, actions, and metrics
   - No write or execute permissions

2. **Auditor** (Level 1) - Extended read access
   - All Viewer permissions
   - Read violations and audit logs
   - Cannot modify system state

3. **Operator** (Level 2) - Action execution
   - All Auditor permissions
   - Execute actions
   - Manage quarantine
   - No user or system management

4. **Admin** (Level 3) - Full control
   - All Operator permissions
   - Manage users and roles
   - System configuration
   - Admin override capabilities

### Permissions

Fine-grained permissions include:
- Read: `READ_POLICIES`, `READ_ACTIONS`, `READ_VIOLATIONS`, `READ_METRICS`, `READ_AUDIT_LOGS`
- Write: `WRITE_POLICIES`, `WRITE_ACTIONS`, `EXECUTE_ACTIONS`
- Management: `MANAGE_USERS`, `MANAGE_ROLES`, `MANAGE_QUARANTINE`, `MANAGE_SYSTEM`
- Admin: `ADMIN_OVERRIDE`, `SYSTEM_CONFIG`

### Usage Examples

```python
from nethical.core.rbac import RBACManager, Role, Permission, require_role

# Initialize RBAC manager
rbac = RBACManager()

# Assign roles
rbac.assign_role("alice", Role.ADMIN)
rbac.assign_role("bob", Role.OPERATOR)
rbac.assign_role("charlie", Role.VIEWER)

# Check permissions
if rbac.has_permission("bob", Permission.EXECUTE_ACTIONS):
    print("Bob can execute actions")

# Use decorators
@require_role(Role.ADMIN)
def delete_user(user_id: str, current_user: str):
    # Only admins can call this function
    pass

@require_permission(Permission.WRITE_POLICIES)
def update_policy(policy_id: str, current_user: str):
    # Only users with WRITE_POLICIES can call this
    pass
```

### Features

- **Hierarchical Inheritance**: Higher roles automatically inherit permissions from lower roles
- **Custom Permissions**: Grant specific permissions to users beyond their role
- **Audit Trail**: All access decisions are logged for compliance
- **Decorator Support**: Easy integration with `@require_role` and `@require_permission`

## 2. JWT Authentication System

### Location
`nethical/security/auth.py`

### Token Types

1. **Access Tokens**
   - Short-lived (1 hour default)
   - Used for API authentication
   - HS256-signed JWT
   - Can be scoped

2. **Refresh Tokens**
   - Long-lived (7 days default)
   - Used to obtain new access tokens
   - Reduces credential transmission

### API Keys

- Service-to-service authentication
- SHA-256 hashed storage
- Optional expiration
- Per-user management
- Enable/disable functionality

### Usage Examples

```python
from nethical.security.auth import AuthManager, authenticate_request

# Initialize auth manager
auth = AuthManager(secret_key="your-secret-key")

# Create tokens
access_token, payload = auth.create_access_token("user123")
refresh_token, _ = auth.create_refresh_token("user123")

# Verify token
payload = auth.verify_token(access_token)
print(f"User: {payload.user_id}")

# Refresh access token
new_access_token, _ = auth.refresh_access_token(refresh_token)

# Create API key
api_key_string, api_key = auth.create_api_key(
    user_id="service_account",
    name="Production API Key",
    expires_at=datetime.now(timezone.utc) + timedelta(days=90)
)

# Authenticate request
user_id = authenticate_request(
    authorization_header=f"Bearer {access_token}"
)
```

### Security Features

- **Token Revocation**: Tokens can be revoked before expiration
- **Signature Verification**: HMAC-SHA256 signature validation
- **Expiration Checking**: Automatic expiry validation
- **API Key Hashing**: Keys are hashed with SHA-256
- **Last Used Tracking**: API keys track last usage

## 3. Supply Chain Security

### Location
`.github/dependabot.yml`

### Features

- **Automated Dependency Updates**
  - Weekly schedule (Monday 06:00 UTC)
  - Separate PRs for different update types
  - Grouped minor/patch updates

- **Multi-Ecosystem Support**
  - Python (pip)
  - GitHub Actions
  - Docker

- **Security Focus**
  - Major updates only for security issues
  - Automatic labeling and assignment
  - Reviewer notification

### Configuration

```yaml
# Dependency scanning for Python
- package-ecosystem: "pip"
  schedule:
    interval: "weekly"
  groups:
    production-dependencies:
      dependency-type: "production"
      update-types: ["minor", "patch"]
```

### Pinned Dependencies

Core dependencies are now pinned in `requirements.txt`:
- pydantic==2.12.3
- numpy==2.3.4
- pandas==2.3.3
- dataclasses-json==0.6.7

## 4. Threat Model Automation

### Location
`.github/workflows/threat-model.yml`

### Features

- **STRIDE Validation**
  - Automatic checking of all STRIDE categories
  - Security controls mapping
  - Coverage metrics

- **Code-to-Controls Mapping**
  - Links threat model controls to implementation
  - Identifies missing implementations
  - Validates control coverage

- **PR Integration**
  - Comments on PRs with security status
  - Coverage percentage reporting
  - Threshold checking (60% minimum)

- **Scheduled Validation**
  - Weekly validation runs
  - Ensures threat model stays current

### Workflow Triggers

- Pull requests to main/develop
- Code changes in security-sensitive areas
- Weekly scheduled runs
- Manual workflow dispatch

### Metrics

The workflow generates:
- Total security controls count
- Complete/Partial/Missing status
- Coverage percentage
- Implementation mapping

## Testing

All new features have comprehensive test coverage:

- **RBAC Tests**: `tests/unit/test_rbac.py` (22 tests)
  - Role management
  - Permission checking
  - Decorator functionality
  - Integration scenarios

- **Auth Tests**: `tests/unit/test_auth.py` (30 tests)
  - Token creation/verification
  - API key management
  - Expiration handling
  - Integration flows

Run tests:
```bash
pytest tests/unit/test_rbac.py tests/unit/test_auth.py -v
```

## Integration with Existing Systems

### RBAC Integration

```python
from nethical.core import (
    IntegratedGovernance,
    get_rbac_manager,
    Role
)

# Initialize systems
governance = IntegratedGovernance()
rbac = get_rbac_manager()

# Assign roles to agents
rbac.assign_role("agent_001", Role.OPERATOR)

# Check permissions before action
@require_permission(Permission.EXECUTE_ACTIONS)
def execute_agent_action(action, current_user):
    result = governance.process_action(action)
    return result
```

### Auth Integration

```python
from nethical.security import (
    authenticate_request,
    AuthenticationError
)

def api_endpoint(authorization_header):
    try:
        user_id = authenticate_request(
            authorization_header=authorization_header
        )
        # Proceed with authenticated user
        return {"status": "success", "user": user_id}
    except AuthenticationError as e:
        return {"status": "error", "message": str(e)}
```

## Security Best Practices

1. **Secret Management**
   - Use environment variables for auth secret keys
   - Rotate secrets regularly
   - Never commit secrets to version control

2. **Token Expiration**
   - Keep access tokens short-lived (≤1 hour)
   - Use refresh tokens for long sessions
   - Implement token rotation

3. **Role Assignment**
   - Follow principle of least privilege
   - Regular access reviews
   - Document role assignments

4. **API Key Management**
   - Set expiration dates
   - Revoke unused keys
   - Monitor last-used timestamps

5. **Audit Logging**
   - Review access decisions regularly
   - Monitor for suspicious patterns
   - Archive logs for compliance

## Compliance

These implementations support:
- **SOC 2**: Access control, authentication, audit logging
- **ISO 27001**: Information security management
- **GDPR**: Access controls for personal data
- **HIPAA**: Authentication and authorization for healthcare data

## Future Enhancements

Potential future additions:
- SSO/SAML integration
- Multi-factor authentication (MFA)
- OAuth 2.0 support
- Hardware security module (HSM) integration
- Federated identity management
- Advanced threat detection
- Real-time security analytics

## References

- Threat Model: `docs/security/threat_model.md`
- RBAC Implementation: `nethical/core/rbac.py`
- Auth Implementation: `nethical/security/auth.py`
- Security Workflow: `.github/workflows/threat-model.yml`
- Dependabot Config: `.github/dependabot.yml`
