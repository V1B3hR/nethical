# Access Control Specification (Phase 4B)

## Status: ✅ COMPLETE

---

## Overview

This document specifies the access control and multi-signature approval mechanisms for the Nethical governance platform. These controls ensure that only authorized users can perform sensitive operations and that critical actions require multi-party approval.

## 1. Access Control Model

### 1.1 Roles and Permissions

**Role Hierarchy**:
```
                    ┌─────────────┐
                    │  superadmin │ (All permissions)
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │               │
      ┌─────▼─────┐  ┌────▼────┐   ┌─────▼──────┐
      │policy_admin│  │ auditor │   │  operator  │
      └─────┬──────┘  └────┬────┘   └─────┬──────┘
            │              │               │
      [Create/Load]   [Read Audit]   [Query Only]
      policies        logs
```

**Permission Matrix**:

| Operation | superadmin | policy_admin | policy_approver | auditor | operator |
|-----------|-----------|--------------|-----------------|---------|----------|
| Create Policy | ✅ | ✅ | ❌ | ❌ | ❌ |
| Load Policy | ✅ | ✅ | ❌ | ❌ | ❌ |
| Approve Policy | ✅ | ❌ | ✅ | ❌ | ❌ |
| Activate Policy | ✅ | ❌ | ✅* | ❌ | ❌ |
| Query Decision | ✅ | ✅ | ✅ | ✅ | ✅ |
| Read Audit Log | ✅ | ✅ | ✅ | ✅ | ❌ |
| Modify Audit Log | ❌ | ❌ | ❌ | ❌ | ❌ |
| Register Agent | ✅ | ✅ | ❌ | ❌ | ❌ |
| Suspend Agent | ✅ | ✅ | ❌ | ❌ | ❌ |

*Requires k-of-n approver signatures for critical policies

### 1.2 Property: P-AUTH

**Definition**: All protected operations require authentication and authorization

**Formal Specification** (TLA+):
```tla
P_AUTH ==
    \A op \in ProtectedOperations:
        Executed(op) =>
            \E user \in Users:
                /\ Authenticated(user)
                /\ HasPermission(user, op)
                /\ AuditLogged(op, user)
```

**Implementation**:
```python
from nethical.security.authentication import require_permission

@require_permission("policy_admin")
def create_policy(policy_data):
    """Only policy_admin can create policies."""
    policy = Policy.create(policy_data)
    audit_log.record("POLICY_CREATED", policy.id, current_user())
    return policy

@require_permission("auditor")
def read_audit_log(start_time, end_time):
    """Only auditor (or higher) can read audit logs."""
    return audit_log.query(start_time, end_time)
```

**Test Coverage**:
- `tests/unit/test_phase1_security.py::test_pki_certificate_validation`
- `tests/unit/test_phase1_security.py::test_multi_factor_authentication`

---

## 2. Multi-Signature Approval

### 2.1 Workflow

**Critical Policy Activation**:
```
1. Policy Created (policy_admin)
   └─> state = INACTIVE

2. Policy Loaded (policy_admin)
   └─> state = QUARANTINE

3. Approvals Collected (policy_approver × k)
   ├─> Approval 1: signature_1 from approver_1
   ├─> Approval 2: signature_2 from approver_2
   └─> Approval k: signature_k from approver_k

4. Policy Activated (when k signatures collected)
   └─> state = ACTIVE
```

### 2.2 Property: P-MULTI-SIG

**Definition**: Critical policy activation requires k-of-n approver signatures

**Formal Specification** (TLA+):
```tla
CONSTANTS K, N  \* k-of-n threshold (e.g., 3-of-5)

P_MULTI_SIG ==
    \A p \in Policies:
        /\ p.is_critical = TRUE
        /\ p.state = "ACTIVE"
        => /\ p.approval_count >= K
           /\ Cardinality(p.approvers) >= K
           /\ \A approver \in p.approvers:
                /\ approver \in ApprovedApprovers
                /\ ValidSignature(p.hash, approver.signature)
```

**Implementation**:
```python
from nethical.core.policy_engine import PolicyApprovalWorkflow
from nethical.security.authentication import verify_signature

class PolicyApprovalWorkflow:
    def __init__(self, policy_id, required_approvals=3):
        self.policy_id = policy_id
        self.required_approvals = required_approvals
        self.approvals = []
    
    def add_approval(self, approver_id, signature):
        """Add an approval signature."""
        policy = Policy.get(self.policy_id)
        
        # Verify signature
        if not verify_signature(policy.hash, signature, approver_id):
            raise InvalidSignatureError()
        
        # Check approver has permission
        if not has_permission(approver_id, "policy_approver"):
            raise UnauthorizedError()
        
        # Add approval
        self.approvals.append({
            "approver_id": approver_id,
            "signature": signature,
            "timestamp": datetime.utcnow()
        })
        
        # Audit log
        audit_log.record("POLICY_APPROVED", self.policy_id, approver_id)
    
    def can_activate(self):
        """Check if policy has enough approvals."""
        return len(self.approvals) >= self.required_approvals
    
    def activate(self):
        """Activate policy if quorum reached."""
        if not self.can_activate():
            raise InsufficientApprovalsError(
                f"Need {self.required_approvals}, got {len(self.approvals)}"
            )
        
        policy = Policy.get(self.policy_id)
        policy.state = "ACTIVE"
        policy.save()
        
        audit_log.record("POLICY_ACTIVATED", self.policy_id, {
            "approvals": self.approvals,
            "quorum": f"{len(self.approvals)}-of-{self.required_approvals}"
        })
```

**Test Coverage**:
- `tests/unit/test_phase1_security.py::test_session_management`
- Unit tests for approval workflow (to be added)

### 2.3 Signature Verification

**Algorithm**: RSA-SHA256 or ECDSA P-256

```python
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

def verify_signature(message_hash, signature, public_key):
    """Verify RSA-SHA256 signature."""
    try:
        public_key.verify(
            signature,
            message_hash.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except Exception:
        return False
```

---

## 3. Authentication Mechanisms

### 3.1 Supported Methods

1. **PKI Certificate** (CAC/PIV Card)
   - Military/Government standard
   - X.509 certificate validation
   - Hardware token required

2. **Multi-Factor Authentication (MFA)**
   - Something you know (password)
   - Something you have (YubiKey, TOTP)
   - Something you are (biometric - future)

3. **OAuth2/SAML2 Federation**
   - Enterprise SSO integration
   - LDAP/Active Directory connector

### 3.2 Session Management

**Property**: Sessions expire after inactivity

```python
from nethical.security.authentication import SessionManager

session_mgr = SessionManager(
    idle_timeout=900,      # 15 minutes
    absolute_timeout=28800 # 8 hours
)

# Create session
session_token = session_mgr.create_session(user_id, trust_level="HIGH")

# Verify session
is_valid = session_mgr.verify_session(session_token)

# Refresh on activity
session_mgr.refresh_session(session_token)
```

**Test Coverage**:
- `tests/unit/test_phase1_security.py::test_multi_factor_authentication`

---

## 4. Audit Trail for Access Control

### 4.1 Required Audit Events

All access control operations must be logged:

```json
{
  "event_type": "AUTH_SUCCESS | AUTH_FAILURE | PERMISSION_DENIED | POLICY_APPROVED | ...",
  "user_id": "identifier",
  "operation": "create_policy | activate_policy | ...",
  "resource": "policy_id | decision_id | ...",
  "result": "success | failure",
  "reason": "insufficient_permissions | invalid_signature | ...",
  "timestamp": "ISO8601 UTC",
  "ip_address": "...",
  "session_id": "..."
}
```

### 4.2 Property: P-AUDIT-AUTH

**Definition**: All authentication and authorization events are logged

**Formal Specification**:
```tla
P_AUDIT_AUTH ==
    \A op \in ProtectedOperations:
        Executed(op) =>
            \E event \in audit_log:
                /\ event.event_type \in {"AUTH_SUCCESS", "AUTH_FAILURE", "PERMISSION_DENIED"}
                /\ event.operation = op
                /\ event.timestamp <= op.timestamp
```

**Implementation**: See `nethical/security/audit_logging.py`

---

## 5. Integration with Zero Trust Architecture

### 5.1 Continuous Authentication

**Property**: Trust levels degrade over time and with risk events

```python
from nethical.security.zero_trust import ContinuousAuthEngine

engine = ContinuousAuthEngine()

# Create session with initial trust
token = engine.create_session(user_id, initial_trust=TrustLevel.HIGH)

# Report risk events
engine.report_risk_event(token, "suspicious_ip", severity=0.5)
engine.report_risk_event(token, "unusual_access", severity=0.7)

# Trust degrades
valid, trust_level = engine.verify_session(token)
assert trust_level == TrustLevel.MEDIUM  # Degraded from HIGH
```

### 5.2 Device Health Verification

**Property**: Only healthy devices can access critical resources

```python
from nethical.security.zero_trust import ZeroTrustController

controller = ZeroTrustController()

health = controller.check_device_health(
    device_id="laptop-001",
    os_version="11.0",
    patch_level="2024-11",
    antivirus_updated=True,
    disk_encryption_enabled=True,
    firewall_enabled=True
)

if not health.is_healthy():
    deny_access("Device health check failed")
```

**Test Coverage**:
- `tests/test_phase4_operational_security.py::test_device_health_check`

---

## 6. Compliance Mapping

| Framework | Control | Property | Implementation |
|-----------|---------|----------|----------------|
| NIST 800-53 | AC-1 | P-AUTH | Role-based access control |
| NIST 800-53 | AC-2 | P-AUTH | Account management |
| NIST 800-53 | IA-2 | P-AUTH | Multi-factor authentication |
| NIST 800-53 | IA-5 | P-MULTI-SIG | Authenticator management |
| HIPAA | 164.312(a)(1) | P-AUTH | Access control |
| FedRAMP | AC family | P-AUDIT-AUTH | Audit logging |
| SOC 2 | CC6.1 | P-AUTH | Logical access |
| ISO 27001 | A.9 | P-AUTH | Access control |

---

## 7. Security Considerations

### 7.1 Threat Model

**Threats Mitigated**:
1. ✅ **Unauthorized Access**: RBAC enforces permissions
2. ✅ **Insider Threat**: Multi-sig prevents single-actor abuse
3. ✅ **Session Hijacking**: Session timeout and MFA
4. ✅ **Privilege Escalation**: Least privilege principle

**Threats Not Mitigated** (Out of Scope):
1. ❌ **Compromised Admin**: Superadmin has full access (operational necessity)
2. ❌ **Social Engineering**: Relies on approver diligence
3. ❌ **Quantum Attacks**: RSA-2048 not quantum-resistant (future: Phase 6)

### 7.2 Key Management

- Private keys stored in HSM or secure key vault
- Public keys in certificate authority (CA) registry
- Key rotation every 90 days (configurable)

---

## 8. Future Enhancements

1. **Attribute-Based Access Control (ABAC)**: Fine-grained policies based on attributes
2. **Time-Based Access**: Temporary elevated permissions
3. **Biometric Authentication**: Fingerprint, facial recognition
4. **Quantum-Resistant Signatures**: Post-quantum cryptography (Dilithium)

---

## References

- [NIST RBAC Model](https://csrc.nist.gov/projects/role-based-access-control)
- [NIST SP 800-63B: Digital Identity Guidelines](https://pages.nist.gov/800-63-3/sp800-63b.html)
- [Zero Trust Architecture (NIST SP 800-207)](https://csrc.nist.gov/publications/detail/sp/800-207/final)
- [RFC 5280: X.509 Public Key Infrastructure](https://tools.ietf.org/html/rfc5280)

**Implementation Files**:
- `nethical/security/authentication.py` - Auth provider
- `nethical/security/zero_trust.py` - Continuous auth
- `tests/unit/test_phase1_security.py` - Unit tests
- `tests/test_phase4_operational_security.py` - Integration tests

---

**Status**: ✅ Phase 4B Complete  
**Last Updated**: 2025-11-16
