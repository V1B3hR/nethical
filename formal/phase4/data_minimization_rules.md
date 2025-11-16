# Data Minimization and Tenant Isolation (Phase 4C)

## Status: ✅ COMPLETE

---

## Overview

This document specifies data minimization and tenant isolation mechanisms for the Nethical governance platform. These controls ensure that:

1. Only necessary context data is accessed during policy evaluation (P-DATA-MIN)
2. Tenants are cryptographically isolated from each other (P-TENANT-ISO)
3. Sensitive data is protected throughout its lifecycle (P-PII-PROTECT)

---

## 1. Data Minimization (P-DATA-MIN)

### 1.1 Principle

**Definition**: Policy evaluation must access only the minimum context fields required to make a decision.

**Rationale**:
- Reduces privacy risk (GDPR Article 5(1)(c))
- Limits attack surface
- Improves audit trail clarity
- Supports compliance (HIPAA minimum necessary standard)

### 1.2 Context Field Whitelisting

**Formal Specification** (TLA+):
```tla
CONSTANTS AllowedFields  \* Set of whitelisted field names

P_DATA_MIN ==
    \A d \in Decisions:
        d.state = "DECIDED" =>
            \A field \in d.accessed_fields:
                field \in AllowedFields
```

**Implementation**:
```python
from nethical.core.context_validator import ContextValidator

class ContextValidator:
    """Enforces data minimization by filtering context fields."""
    
    def __init__(self, allowed_fields):
        self.allowed_fields = set(allowed_fields)
    
    def filter_context(self, context):
        """Remove non-whitelisted fields."""
        filtered = {}
        removed = []
        
        for field, value in context.items():
            if field in self.allowed_fields:
                filtered[field] = value
            else:
                removed.append(field)
                audit_log.record("CONTEXT_FIELD_FILTERED", field)
        
        if removed:
            logger.warning(f"Filtered non-whitelisted fields: {removed}")
        
        return filtered
    
    def validate_policy_context_usage(self, policy):
        """Verify policy only references allowed fields."""
        referenced_fields = policy.extract_field_references()
        
        for field in referenced_fields:
            if field not in self.allowed_fields:
                raise PolicyValidationError(
                    f"Policy references non-whitelisted field: {field}"
                )
        
        return True
```

**Usage Example**:
```python
# Define allowed fields for healthcare scenario
validator = ContextValidator(allowed_fields=[
    "patient_id",        # Required: identify patient
    "provider_id",       # Required: identify provider
    "action_type",       # Required: what action (read, write, delete)
    "resource_type",     # Required: what resource (record, image, prescription)
    "timestamp",         # Required: when
    "urgency_level",     # Optional: for emergency override rules
])

# User provides context (may include extra fields)
raw_context = {
    "patient_id": "P12345",
    "provider_id": "D67890",
    "action_type": "read_record",
    "resource_type": "medical_history",
    "timestamp": "2024-11-16T10:00:00Z",
    "patient_ssn": "123-45-6789",      # FILTERED: not in whitelist
    "patient_address": "123 Main St",  # FILTERED: not in whitelist
}

# Filter context before policy evaluation
clean_context = validator.filter_context(raw_context)
# clean_context = {
#     "patient_id": "P12345",
#     "provider_id": "D67890",
#     "action_type": "read_record",
#     "resource_type": "medical_history",
#     "timestamp": "2024-11-16T10:00:00Z",
# }

# Evaluate with minimal context
decision = policy_engine.evaluate(action, clean_context)
```

### 1.3 Policy Declaration

**Policy Metadata**:
```yaml
policy_id: "hipaa_minimum_necessary"
version: 1
required_fields:
  - patient_id
  - provider_id
  - action_type
  - resource_type
optional_fields:
  - urgency_level
forbidden_fields:
  - patient_ssn
  - patient_address
  - date_of_birth
```

**Validation**: Before activating, verify policy only uses declared fields

### 1.4 Compliance Mapping

| Regulation | Requirement | Implementation |
|------------|-------------|----------------|
| **GDPR Article 5(1)(c)** | Data minimization | Context field whitelisting |
| **HIPAA §164.502(b)** | Minimum necessary | Only required fields accessed |
| **CCPA §1798.100(c)** | Collection limitation | Filtered context |
| **NIST 800-53 SC-28** | Protection of information at rest | Encrypted context storage |

---

## 2. Tenant Isolation (P-TENANT-ISO)

### 2.1 Principle

**Definition**: Tenant A's data and policies cannot influence Tenant B's decisions.

**Threat Model**:
- **Cross-Tenant Leakage**: Tenant A's context accidentally included in Tenant B's evaluation
- **Policy Poisoning**: Tenant A's policy affects Tenant B's decisions
- **Audit Log Contamination**: Tenant A can read Tenant B's audit events

### 2.2 Network Segmentation

**Zero Trust Architecture**: Each tenant has a dedicated network segment

```python
from nethical.security.zero_trust import NetworkSegment, PolicyEnforcer, TrustLevel

# Define tenant-specific segments
segment_tenant_a = NetworkSegment(
    segment_id="tenant_a",
    name="Tenant A Production",
    allowed_services=["tenant_a_api", "tenant_a_db"],
    allowed_protocols=["https"],
    min_trust_level=TrustLevel.HIGH,
    require_mfa=True
)

segment_tenant_b = NetworkSegment(
    segment_id="tenant_b",
    name="Tenant B Production",
    allowed_services=["tenant_b_api", "tenant_b_db"],
    allowed_protocols=["https"],
    min_trust_level=TrustLevel.HIGH,
    require_mfa=True
)

# Enforce isolation
enforcer = PolicyEnforcer([segment_tenant_a, segment_tenant_b])

# Attempt cross-tenant access (should be blocked)
allowed, reason = enforcer.prevent_lateral_movement(
    source_segment="tenant_a",
    target_segment="tenant_b",
    user_id="user_from_tenant_a"
)

assert not allowed, "Cross-tenant access must be blocked"
# reason = "Lateral movement from tenant_a to tenant_b is not allowed"
```

### 2.3 Formal Specification

**TLA+ Definition**:
```tla
CONSTANTS Tenants  \* Set of tenant identifiers

\* Each decision is associated with exactly one tenant
DecisionTenant == [DecisionIds -> Tenants]

\* Each policy is associated with exactly one tenant
PolicyTenant == [PolicyIds -> Tenants]

P_TENANT_ISO ==
    \A d \in DecisionIds, p \in PolicyIds:
        /\ UsesPolicy(d, p)
        => DecisionTenant[d] = PolicyTenant[p]
    /\ \A d1, d2 \in DecisionIds:
        /\ DecisionTenant[d1] # DecisionTenant[d2]
        => /\ decisions[d1].context \intersect decisions[d2].context = {}
           /\ \A p \in PolicyIds:
                UsesPolicy(d1, p) => ~UsesPolicy(d2, p)
```

**Invariant**: No policy or context is shared across tenants

### 2.4 Implementation

**Tenant-Scoped Policy Engine**:
```python
from nethical.core.policy_engine import PolicyEngine

class TenantScopedPolicyEngine:
    """Policy engine with strict tenant isolation."""
    
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id
        self.engine = PolicyEngine()
        self._load_tenant_policies(tenant_id)
    
    def _load_tenant_policies(self, tenant_id):
        """Load only policies belonging to this tenant."""
        policies = Policy.query(tenant_id=tenant_id).all()
        for policy in policies:
            self.engine.load_policy(policy)
    
    def evaluate(self, action, context):
        """Evaluate using only tenant-scoped policies."""
        # Enforce tenant_id in context
        if context.get("tenant_id") != self.tenant_id:
            raise TenantMismatchError(
                f"Context tenant_id {context.get('tenant_id')} "
                f"does not match engine tenant_id {self.tenant_id}"
            )
        
        # Add implicit tenant filter
        context["_tenant_id"] = self.tenant_id
        
        # Evaluate with isolated policy set
        result = self.engine.evaluate(action, context)
        
        # Audit with tenant_id
        audit_log.record("DECISION_MADE", result.decision_id, {
            "tenant_id": self.tenant_id,
            "action": action,
            "result": result.verdict
        })
        
        return result
```

**Tenant-Scoped Audit Log**:
```python
from nethical.security.audit_logging import EnhancedAuditLogger

class TenantScopedAuditLogger:
    """Audit logger with tenant isolation."""
    
    def __init__(self, tenant_id):
        self.tenant_id = tenant_id
        self.logger = EnhancedAuditLogger()
    
    def log_event(self, event_type, entity_id, action, **kwargs):
        """Log event with tenant_id tag."""
        return self.logger.log_event(
            event_type=event_type,
            user_id=entity_id,
            action=action,
            tenant_id=self.tenant_id,
            **kwargs
        )
    
    def query_events(self, start_time, end_time):
        """Query only this tenant's events."""
        all_events = self.logger.query_events(start_time, end_time)
        
        # Filter by tenant_id
        tenant_events = [
            e for e in all_events
            if e.get("tenant_id") == self.tenant_id
        ]
        
        return tenant_events
```

### 2.5 Verification

**Test Cross-Tenant Leakage**:
```python
def test_tenant_isolation():
    # Create two tenants
    engine_a = TenantScopedPolicyEngine(tenant_id="tenant_a")
    engine_b = TenantScopedPolicyEngine(tenant_id="tenant_b")
    
    # Tenant A's context
    context_a = {
        "tenant_id": "tenant_a",
        "user_id": "user_a",
        "action": "read_data"
    }
    
    # Tenant B's context
    context_b = {
        "tenant_id": "tenant_b",
        "user_id": "user_b",
        "action": "read_data"
    }
    
    # Evaluate separately
    result_a = engine_a.evaluate("read", context_a)
    result_b = engine_b.evaluate("read", context_b)
    
    # Verify isolation: results use different policy sets
    assert result_a.policy_ids != result_b.policy_ids
    
    # Verify audit isolation
    audit_a = TenantScopedAuditLogger("tenant_a")
    audit_b = TenantScopedAuditLogger("tenant_b")
    
    events_a = audit_a.query_events(start, end)
    events_b = audit_b.query_events(start, end)
    
    # No overlap in audit logs
    assert not any(e in events_b for e in events_a)
```

**Test Coverage**:
- `tests/test_phase4_operational_security.py::test_lateral_movement_prevention`

---

## 3. Sensitive Data Protection (P-PII-PROTECT)

### 3.1 Principle

**Definition**: Personally Identifiable Information (PII) and Protected Health Information (PHI) must be:
1. Encrypted at rest and in transit
2. Minimally accessed (see P-DATA-MIN)
3. Audited on every access
4. Retained only as long as necessary

### 3.2 Data Classification

| Class | Examples | Encryption | Access Control | Audit |
|-------|----------|-----------|----------------|-------|
| **Public** | Policy IDs, timestamps | Optional | Read-only | Minimal |
| **Internal** | Decision IDs, agent IDs | Yes | RBAC | Standard |
| **Confidential** | User IDs, IP addresses | Yes | RBAC + MFA | Enhanced |
| **Restricted** | SSN, PHI, financial data | Yes | RBAC + Multi-sig | Full |

### 3.3 Encryption

**At Rest**: AES-256-GCM
```python
from nethical.security.encryption import EncryptionProvider

encryptor = EncryptionProvider(
    algorithm="AES-256-GCM",
    key_management="HSM"
)

# Encrypt sensitive context field
encrypted_ssn = encryptor.encrypt(
    plaintext="123-45-6789",
    context={"field": "patient_ssn", "tenant_id": "tenant_a"}
)

# Store encrypted value
context_encrypted = {
    "patient_id": "P12345",  # Not sensitive: plaintext
    "patient_ssn_encrypted": encrypted_ssn,  # Sensitive: encrypted
}
```

**In Transit**: TLS 1.3 with mutual authentication
```python
from nethical.security.zero_trust import ServiceMeshConfig

config = ServiceMeshConfig(
    service_name="policy_engine",
    enable_mtls=True,
    tls_version="1.3",
    cipher_suites=["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"]
)
```

### 3.4 Audit Logging for PII Access

**Event Structure**:
```json
{
  "event_type": "PII_ACCESSED",
  "user_id": "doctor_123",
  "action": "read_patient_record",
  "resource": "patient_P12345",
  "pii_fields": ["patient_ssn", "date_of_birth"],
  "justification": "Treatment - annual checkup",
  "timestamp": "2024-11-16T10:30:00Z",
  "ip_address": "192.168.1.100",
  "approved_by": "supervisor_456",
  "retention_policy": "7_years"
}
```

**Compliance**: HIPAA §164.308(a)(1)(ii)(D) - Information system activity review

---

## 4. Service Mesh and Micro-Segmentation

### 4.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Service Mesh (Istio/Linkerd)           │
│  ┌──────────┐  mTLS  ┌──────────┐  mTLS  ┌──────────┐      │
│  │ Policy   │ <----> │ Decision │ <----> │ Audit    │      │
│  │ Engine   │        │ Engine   │        │ Logger   │      │
│  │ (Tenant  │        │ (Tenant  │        │ (Tenant  │      │
│  │  A)      │        │  A)      │        │  A)      │      │
│  └──────────┘        └──────────┘        └──────────┘      │
│       ↑                                                      │
│       │ Network Policy: DENY lateral movement               │
│       ↓                                                      │
│  ┌──────────┐        ┌──────────┐        ┌──────────┐      │
│  │ Policy   │        │ Decision │        │ Audit    │      │
│  │ Engine   │        │ Engine   │        │ Logger   │      │
│  │ (Tenant  │        │ (Tenant  │        │ (Tenant  │      │
│  │  B)      │        │  B)      │        │  B)      │      │
│  └──────────┘        └──────────┘        └──────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Network Policies

**Deny Lateral Movement**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-tenant-a-to-b
  namespace: nethical
spec:
  podSelector:
    matchLabels:
      tenant: tenant_a
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          tenant: tenant_a  # Only allow within same tenant
```

**Allow Audit Logging (All Tenants)**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-audit-logging
  namespace: nethical
spec:
  podSelector:
    matchLabels:
      app: audit-logger
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tenant: tenant_a
    - podSelector:
        matchLabels:
          tenant: tenant_b
```

---

## 5. Compliance Mapping

| Framework | Control | Property | Implementation |
|-----------|---------|----------|----------------|
| **GDPR** | Article 5(1)(c) | P-DATA-MIN | Context field whitelisting |
| **GDPR** | Article 25 | P-DATA-MIN | Privacy by design |
| **GDPR** | Article 32 | P-PII-PROTECT | Encryption + access control |
| **HIPAA** | §164.502(b) | P-DATA-MIN | Minimum necessary standard |
| **HIPAA** | §164.312(a)(2) | P-PII-PROTECT | Encryption + audit |
| **NIST 800-53** | SC-7 | P-TENANT-ISO | Boundary protection |
| **NIST 800-53** | SC-28 | P-PII-PROTECT | Encryption at rest |
| **SOC 2** | CC6.7 | P-TENANT-ISO | Logical access - segregation |
| **ISO 27001** | A.13.1 | P-TENANT-ISO | Network security management |

---

## 6. Testing

### 6.1 Test Coverage

| Test | Status | File |
|------|--------|------|
| Context field filtering | ✅ | `test_context_validator.py` |
| Policy validation (field usage) | ✅ | `test_policy_engine.py` |
| Tenant isolation (policy) | ✅ | `test_tenant_isolation.py` |
| Tenant isolation (audit) | ✅ | `test_tenant_audit.py` |
| Lateral movement prevention | ✅ | `test_phase4_operational_security.py` |
| PII encryption | ✅ | `test_phase1_security.py` |
| Service mesh mTLS | ✅ | `test_phase4_operational_security.py` |

---

## 7. Performance Considerations

| Operation | Overhead | Mitigation |
|-----------|----------|-----------|
| Context filtering | ~1 ms | In-memory whitelist lookup |
| Tenant check | ~0.5 ms | Cached tenant_id comparison |
| Encryption | ~5 ms | Hardware acceleration (AES-NI) |
| mTLS handshake | ~50 ms | Connection pooling + session resumption |

---

## 8. Future Enhancements

1. **Attribute-Based Access Control**: Dynamic policies based on context attributes
2. **Homomorphic Encryption**: Compute on encrypted data
3. **Differential Privacy**: Add noise to aggregate queries
4. **Federated Learning**: Train ML models without centralizing data

---

## References

- [GDPR Article 5: Principles](https://gdpr-info.eu/art-5-gdpr/)
- [HIPAA Minimum Necessary Standard](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/minimum-necessary-requirement/index.html)
- [NIST SP 800-207: Zero Trust Architecture](https://csrc.nist.gov/publications/detail/sp/800-207/final)
- [Istio Security](https://istio.io/latest/docs/concepts/security/)

**Implementation Files**:
- `nethical/core/context_validator.py` - Context filtering
- `nethical/security/zero_trust.py` - Network segmentation
- `nethical/security/encryption.py` - PII encryption
- `tests/test_phase4_operational_security.py` - Integration tests

---

**Status**: ✅ Phase 4C Complete  
**Last Updated**: 2025-11-16
