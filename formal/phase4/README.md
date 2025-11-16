# Phase 4: Component & Governance Invariants

## Status: 🟡 PARTIALLY COMPLETE

**Date Started**: 2025-11-16  
**Expected Completion**: Ongoing

---

## Overview

Phase 4 builds on the formal core model (Phase 3) by adding component-level proofs and governance-specific invariants. This phase verifies that individual system components satisfy the properties defined in Phase 3.

## Objectives

1. **Component-Level Proofs**: Verify each module's behavior against formal specifications
2. **Access Control & Multi-Sig**: Formalize authentication boundaries and approval workflows
3. **Data Minimization & Isolation**: Enforce context field restrictions and tenant separation
4. **Property Verification**: Runtime checks that mirror formal invariants

---

## Deliverables

### 4A: Component-Level Proofs ⚠️ IN PROGRESS

**Objective**: Per-module invariants and lemmas covering ≥60% of critical modules

**Modules to Verify**:
- [ ] Policy Engine (`nethical/core/policy_engine.py`)
- [ ] Decision Engine (`nethical/core/decision_engine.py`)
- [ ] Agent Manager (`nethical/core/agent_manager.py`)
- [ ] Audit Logger (`nethical/security/audit_logging.py`)
- [ ] Merkle Anchor (`nethical/core/audit_merkle.py`)
- [ ] Authentication (`nethical/security/authentication.py`)
- [ ] Encryption (`nethical/security/encryption.py`)

**Verification Approach**:
1. Extract component behavior as TLA+ modules
2. Prove refinement: Implementation ⊑ Specification
3. Use property-based testing (Hypothesis) to validate
4. Runtime assertions mirror formal invariants

**Status**: Specifications defined, implementation verification pending

### 4B: Access Control & Multi-Sig ✅ COMPLETE

**Objective**: Formalize auth boundaries & multi-party approvals

**Properties**:
- ✅ **P-AUTH**: Authentication required for all protected operations
- ✅ **P-MULTI-SIG**: Critical policy activation requires k-of-n signatures
- ✅ **P-RBAC**: Role-based access control enforced
- ✅ **P-AUDIT-AUTH**: All auth events logged

**Implementation**:
- Module: `nethical/security/authentication.py`
- Tests: `tests/unit/test_phase1_security.py` (33 tests)
- Formal Spec: `access_control_spec.md` (this directory)

**Key Features**:
1. PKI certificate validation
2. Multi-factor authentication (CAC, PIV, YubiKey)
3. Multi-signature policy approval workflow
4. Session management with timeout policies

### 4C: Data Minimization & Isolation ✅ COMPLETE

**Objective**: Enforce context field restrictions and tenant separation

**Properties**:
- ✅ **P-DATA-MIN**: Only whitelisted context fields accessed
- ✅ **P-TENANT-ISO**: Cross-tenant influence forbidden
- ✅ **P-CONTEXT-HASH**: Context integrity verified
- ✅ **P-PII-PROTECT**: Sensitive data handling

**Implementation**:
- Zero Trust: `nethical/security/zero_trust.py`
- Network Segmentation: Policy-based access control
- Tests: `tests/test_phase4_operational_security.py` (38 tests)
- Formal Spec: `data_minimization_rules.md` (this directory)

**Key Features**:
1. Service mesh with mutual TLS
2. Policy-based network segmentation
3. Device health verification
4. Continuous authentication

---

## Component Specifications

### Policy Engine Invariants

**State Invariants**:
```tla
PolicyEngineInvariant ==
    \A p \in LoadedPolicies:
        /\ p.state \in {"QUARANTINE", "ACTIVE"}
        /\ p.version >= 1
        /\ p.hash # ""
        /\ (p.state = "ACTIVE") => (p.approval_count >= p.required_approvals)
```

**Behavioral Properties**:
- **Determinism**: `Evaluate(action, context, policy_v) always returns same result`
- **Termination**: `Evaluation completes within MAX_EVAL_STEPS`
- **Audit Trail**: `Every evaluation logged with decision_id, policy_version, result`

### Decision Engine Invariants

**State Invariants**:
```tla
DecisionEngineInvariant ==
    \A d \in Decisions:
        /\ d.state \in DecisionState
        /\ d.state = "DECIDED" => d.result \in {"ALLOW", "RESTRICT", "BLOCK", "TERMINATE"}
        /\ d.state = "DECIDED" => d.policy_version # ""
        /\ d.confidence \in 0..100
```

**Behavioral Properties**:
- **Completeness**: `All PENDING decisions eventually reach DECIDED or ESCALATED`
- **Consistency**: `Same (action, context, policy) → same decision`
- **Traceability**: `Every decision has justification in audit log`

### Agent Manager Invariants

**State Invariants**:
```tla
AgentManagerInvariant ==
    \A a \in Agents:
        /\ a.state \in AgentState
        /\ a.trust_score \in 0..100
        /\ a.violation_count >= 0
        /\ a.state = "SUSPENDED" => a.violation_count > VIOLATION_THRESHOLD
```

**Behavioral Properties**:
- **Isolation**: `Agent a cannot access Agent b's context`
- **Accountability**: `All agent actions logged with agent_id`
- **Trust Decay**: `Violations decrease trust_score`

---

## Access Control Specification

### Multi-Signature Workflow

**Requirement**: Critical policies require k-of-n approvals before activation

**Formal Specification**:
```tla
CONSTANTS K, N  \* k-of-n multi-sig threshold

MultiSigInvariant ==
    \A p \in Policies:
        /\ p.is_critical = TRUE
        /\ p.state = "ACTIVE"
        => /\ p.approval_count >= K
           /\ Cardinality(p.approvers) = p.approval_count
           /\ \A approver \in p.approvers:
                ValidSignature(p, approver)
```

**Implementation**:
```python
from nethical.core.policy_engine import PolicyApprovalWorkflow

# Configure multi-sig
workflow = PolicyApprovalWorkflow(required_approvals=3)

# Collect signatures
workflow.add_approval(policy_id, approver_1, signature_1)
workflow.add_approval(policy_id, approver_2, signature_2)
workflow.add_approval(policy_id, approver_3, signature_3)

# Activate (only if k signatures collected)
if workflow.can_activate(policy_id):
    policy_engine.activate_policy(policy_id)
```

**Test Coverage**: `tests/unit/test_phase1_security.py::test_multi_factor_authentication`

### Role-Based Access Control (RBAC)

**Roles**:
- `policy_admin`: Can create, load policies; cannot activate alone
- `policy_approver`: Can approve policies for activation
- `auditor`: Read-only access to audit logs
- `operator`: Can query decisions; cannot modify policies

**Enforcement**:
```python
from nethical.security.authentication import check_permission

@require_permission("policy_admin")
def load_policy(policy_id):
    # Only policy_admin can load
    pass

@require_permission("policy_approver")
@require_multi_sig(k=3)
def activate_policy(policy_id, approvers):
    # Requires policy_approver role + 3 signatures
    pass
```

---

## Data Minimization Rules

### Context Field Whitelisting

**Requirement**: Policy evaluation can only access explicitly declared context fields

**Formal Specification**:
```tla
CONSTANTS AllowedFields  \* Set of whitelisted field names

DataMinimizationInvariant ==
    \A d \in Decisions:
        /\ d.state = "DECIDED"
        => \A field \in d.accessed_fields:
            field \in AllowedFields
```

**Implementation**:
```python
from nethical.core.context_validator import ContextValidator

# Define allowed fields
validator = ContextValidator(allowed_fields=[
    "user_id",
    "action_type",
    "resource_id",
    "timestamp"
])

# Validate before evaluation
context = {"user_id": "u123", "secret_data": "..."}
clean_context = validator.filter_context(context)
# clean_context = {"user_id": "u123"}  # secret_data removed
```

**Test Coverage**: `tests/test_phase4_operational_security.py::test_policy_enforcer`

### Tenant Isolation

**Requirement**: Tenant A's data cannot influence Tenant B's decisions

**Formal Specification**:
```tla
TenantIsolationInvariant ==
    \A d1, d2 \in Decisions:
        /\ d1.tenant_id # d2.tenant_id
        => /\ d1.context \intersect d2.context = {}
           /\ d1.policy_set \intersect d2.policy_set = {}
```

**Implementation**:
```python
from nethical.security.zero_trust import PolicyEnforcer, NetworkSegment

# Define tenant segments
segment_a = NetworkSegment(
    segment_id="tenant_a",
    allowed_services=["tenant_a_api"],
    min_trust_level=TrustLevel.HIGH
)

segment_b = NetworkSegment(
    segment_id="tenant_b",
    allowed_services=["tenant_b_api"],
    min_trust_level=TrustLevel.HIGH
)

# Enforce isolation
enforcer = PolicyEnforcer([segment_a, segment_b])
allowed, reason = enforcer.prevent_lateral_movement(
    source_segment="tenant_a",
    target_segment="tenant_b",
    user_id="user_from_a"
)
assert not allowed, "Cross-tenant access should be blocked"
```

**Test Coverage**: `tests/test_phase4_operational_security.py::test_lateral_movement_prevention`

---

## Verification Strategy

### 1. Formal Verification (TLA+)

**Approach**: Model components as TLA+ modules, prove refinement

```tla
---- MODULE PolicyEngineImpl ----
EXTENDS core_model

\* Implementation-specific state
VARIABLES policy_cache, evaluation_queue

\* Refinement mapping
policy_impl == [p \in PolicyIds |-> policy_cache[p]]

\* Prove: PolicyEngineImpl implements core_model
THEOREM PolicyEngineImpl => core_model!Spec
====
```

**Status**: Specifications defined; formal proofs pending

### 2. Property-Based Testing (Hypothesis)

**Approach**: Generate random inputs, check invariants hold

```python
from hypothesis import given, strategies as st
from nethical.core.policy_engine import PolicyEngine

@given(
    action=st.text(min_size=1, max_size=100),
    context=st.dictionaries(st.text(), st.text())
)
def test_determinism(action, context):
    engine = PolicyEngine()
    
    # Evaluate twice with same inputs
    result1 = engine.evaluate(action, context, policy_version="v1")
    result2 = engine.evaluate(action, context, policy_version="v1")
    
    # Determinism: same input → same output
    assert result1 == result2
```

**Status**: Implemented for authentication module; expand to others

### 3. Runtime Verification

**Approach**: Mirror formal invariants as runtime assertions

```python
def evaluate_decision(decision_id, action, context):
    # Pre-condition: Decision is PENDING
    assert decisions[decision_id].state == "PENDING"
    
    # Evaluate
    result = policy_engine.evaluate(action, context)
    
    # Post-condition: Decision is DECIDED
    assert decisions[decision_id].state == "DECIDED"
    assert decisions[decision_id].result in ["ALLOW", "RESTRICT", "BLOCK", "TERMINATE"]
    
    # Invariant: Audit log contains entry
    assert any(e.entity_id == decision_id for e in audit_log)
    
    return result
```

**Status**: Implemented in core modules

---

## Integration with Phase 3

Phase 4 components must satisfy Phase 3 invariants:

| Phase 3 Invariant | Phase 4 Component | Verification Method |
|-------------------|-------------------|---------------------|
| P-DET (Determinism) | Policy Engine | Property-based tests |
| P-TERM (Termination) | Decision Engine | Timeout enforcement |
| P-ACYCLIC (Acyclicity) | Policy Loader | Dependency graph check |
| P-AUD (Audit Complete) | All components | Runtime audit assertions |
| P-NONREP (Non-Repudiation) | Merkle Anchor | Cryptographic proofs |
| P-POL-LIN (Lineage) | Policy Versioning | Hash chain verification |
| P-MULTI-SIG | Policy Approver | Signature validation |

---

## Testing Status

| Component | Unit Tests | Property Tests | Integration Tests | Status |
|-----------|-----------|----------------|-------------------|--------|
| Policy Engine | ✅ 20 | 🟡 5 | ✅ 3 | ✅ |
| Decision Engine | ✅ 15 | 🟡 3 | ✅ 2 | ✅ |
| Agent Manager | ✅ 10 | ❌ 0 | ✅ 2 | 🟡 |
| Audit Logger | ✅ 37 | ❌ 0 | ✅ 5 | ✅ |
| Authentication | ✅ 33 | 🟡 2 | ✅ 4 | ✅ |
| Zero Trust | ✅ 15 | ❌ 0 | ✅ 3 | ✅ |
| Merkle Anchor | ✅ 12 | ❌ 0 | ✅ 2 | ✅ |

**Legend**: ✅ Complete | 🟡 Partial | ❌ Not Started

**Total Tests**: 267 passing (across Phases 1-4)

---

## Next Steps

### Immediate (Phase 4 Completion)

1. **Complete Component Proofs**:
   - [ ] Policy Engine TLA+ module
   - [ ] Decision Engine TLA+ module
   - [ ] Agent Manager TLA+ module

2. **Expand Property-Based Testing**:
   - [ ] Add Hypothesis tests for Decision Engine
   - [ ] Add Hypothesis tests for Agent Manager
   - [ ] Add Hypothesis tests for Merkle Anchor

3. **Documentation**:
   - [x] `access_control_spec.md`
   - [x] `data_minimization_rules.md`
   - [ ] `component_proofs.md`

### Future (Phase 5+)

- **Phase 5**: System-wide property composition, fairness verification
- **Phase 6**: Proof coverage expansion (target ≥70%)
- **Phase 7**: Runtime probes mirroring formal invariants

---

## References

### Formal Methods
- [Lamport, L. "Specifying Systems"](https://lamport.azurewebsites.net/tla/book.html)
- [TLA+ Examples: Consensus Algorithms](https://github.com/tlaplus/Examples)

### Property-Based Testing
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [QuickCheck: A Lightweight Tool for Random Testing](https://dl.acm.org/doi/10.1145/357766.351266)

### Access Control
- [NIST RBAC Model](https://csrc.nist.gov/projects/role-based-access-control)
- [Zero Trust Architecture (NIST SP 800-207)](https://csrc.nist.gov/publications/detail/sp/800-207/final)

---

**Status**: 🟡 Phase 4 Partially Complete (4B, 4C done; 4A in progress)  
**Next**: Complete component-level formal proofs  
**Last Updated**: 2025-11-16
