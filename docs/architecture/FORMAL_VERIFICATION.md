# Formal Verification in Nethical

## Overview

Nethical employs formal verification techniques to ensure the correctness, safety, and compliance of its AI governance framework. This document describes the verification methodologies, tools, and processes used.

## Table of Contents

1. [TLA+ Specifications](#tla-specifications)
2. [Z3 SMT Verification](#z3-smt-verification)
3. [Runtime Verification](#runtime-verification)
4. [Verification Coverage](#verification-coverage)
5. [Running Verification](#running-verification)
6. [External Audit Requirements](#external-audit-requirements)

---

## TLA+ Specifications

### What is TLA+?

TLA+ (Temporal Logic of Actions) is a formal specification language used to design, model, and verify concurrent and distributed systems. Nethical uses TLA+ to specify and verify critical system properties.

### Specifications

#### 1. GovernanceStateMachine.tla

**Purpose**: Verifies state transitions in the governance decision model (ALLOW → RESTRICT → BLOCK → TERMINATE).

**Key Properties Verified:**
- **Terminality**: Once an agent is TERMINATED, it remains TERMINATED
- **Risk Alignment**: High risk scores (≥90) force TERMINATE state
- **Valid Transitions**: Only valid state transitions are allowed (no BLOCK → ALLOW directly)
- **Risk Score Bounds**: Risk scores stay within [0, 100]

**Safety Properties:**
```tla
TerminalityProperty == 
    [](∀ agent: terminated → []terminated)

RiskAlignmentProperty ==
    [](∀ agent: risk ≥ 90 → state = TERMINATE)
```

#### 2. PolicyConsistency.tla

**Purpose**: Ensures policies don't conflict and produce deterministic decisions.

**Key Properties Verified:**
- **Determinism**: No two policies with the same priority contradict each other
- **Completeness**: Every action type has an effective decision
- **Priority Respect**: Higher priority policies override lower ones
- **Version Monotonicity**: Policy versions never decrease

**Safety Properties:**
```tla
DeterminismProperty == [](NoPoliciesConflict)

CompletenessProperty == 
    [](∀ actionType: ∃ decision: effective(actionType) = decision)
```

#### 3. AuditIntegrity.tla

**Purpose**: Verifies Merkle tree-based audit log properties.

**Key Properties Verified:**
- **Append-Only**: Audit log only grows, never shrinks
- **Immutability**: Existing entries never change
- **Sequential IDs**: Entry IDs are sequential (1, 2, 3, ...)
- **Monotonic Timestamps**: Timestamps never decrease
- **Anchored Immutability**: Anchored entries cannot be modified

**Safety Properties:**
```tla
ImmutabilityProperty ==
    [](∀ i ∈ log: log'[i] = log[i])

ChainIntegrityProperty ==
    [](∀ i > 1: log[i].id = log[i-1].id + 1)
```

#### 4. FundamentalLaws.tla

**Purpose**: Verifies the 25 Fundamental Laws are never violated simultaneously.

**Key Properties Verified:**
- **Critical Law Emergency**: Violation of critical laws (e.g., Law 21: Human Safety) triggers emergency stop
- **System State Alignment**: System state correctly reflects violation count
- **Violation Bounds**: Total violations bounded in non-critical states
- **Law 21 Safety**: Human safety law violations always result in emergency stop

**Safety Properties:**
```tla
Law21SafetyProperty ==
    [](∀ agent: violates(Law 21) → EMERGENCY_STOP)

CriticalLawEmergencyProperty ==
    [](∀ agent: violates(CriticalLaw) → systemState = EMERGENCY_STOP)
```

### Running TLA+ Model Checking

To verify the TLA+ specifications, use the TLC model checker:

```bash
# Install TLA+ Toolbox
wget https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/TLAToolbox-1.8.0-linux.gtk.x86_64.zip
unzip TLAToolbox-1.8.0-linux.gtk.x86_64.zip

# Run model checker on a specification
cd formal/tla
java -cp tla2tools.jar tlc2.TLC GovernanceStateMachine.tla

# Or use the TLA+ Toolbox IDE for interactive verification
```

**Configuration Parameters:**
```tla
CONSTANTS:
    Agents = {a1, a2, a3}
    MaxRiskScore = 100
    BlockThreshold = 70
    TerminateThreshold = 90

CONSTRAINTS:
    StateConstraint == len(history) <= 50
```

---

## Z3 SMT Verification

### What is Z3?

Z3 is a high-performance SMT (Satisfiability Modulo Theories) solver used for formal verification. Nethical uses Z3 to verify policy consistency and decision correctness.

### PolicyVerifier

Located in `formal/z3/policy_verifier.py`, the PolicyVerifier provides:

**Verification Methods:**

1. **verify_policy_non_contradiction(policies)**
   - Verifies no two policies contradict each other
   - Checks priority-based conflict resolution

2. **verify_policy_completeness(policies, actionTypes)**
   - Ensures all action types have a defined policy
   - Validates default fallback policies

3. **verify_decision_determinism(policies, action)**
   - Proves decisions are deterministic
   - No ambiguity for any given input

4. **verify_law_compliance(policy, laws)**
   - Checks policy doesn't violate fundamental laws
   - Validates alignment with Law 21 (Human Safety)

### Example Usage

```python
from nethical.formal.z3 import PolicyVerifier

verifier = PolicyVerifier()

# Define policies
policies = [
    {"id": "p1", "priority": 10, "action": "DATA_ACCESS", "decision": "ALLOW"},
    {"id": "p2", "priority": 5, "action": "DATA_ACCESS", "decision": "BLOCK"},
]

# Verify non-contradiction
result = verifier.verify_policy_non_contradiction(policies)
print(f"Result: {result.result}")  # VALID (p1 overrides p2)

# Verify completeness
action_types = ["DATA_ACCESS", "EXECUTE_CODE", "MODIFY_SYSTEM"]
result = verifier.verify_policy_completeness(policies, action_types)
print(f"Result: {result.result}")  # INVALID (missing coverage)
```

### Running Z3 Verification

```bash
# Install Z3
pip install z3-solver

# Run verification tests
pytest tests/formal/test_z3_verification.py -v
```

---

## Runtime Verification

### RuntimeMonitor

Located in `nethical/core/verification/runtime_monitor.py`, the RuntimeMonitor provides real-time verification during system operation.

### Features

1. **Invariant Checking**
   - Risk score bounded [0, 100]
   - No critical violations in ALLOW decisions
   - BLOCK decisions have justification
   - Terminated agents cannot make new decisions

2. **Temporal Property Monitoring**
   - "BLOCK always followed by audit log"
   - "TERMINATE triggers emergency protocol"
   - Pattern matching on event sequences

3. **Contract Assertions**
   - Precondition checks before operations
   - Postcondition checks after operations
   - Decorator-based contract enforcement

4. **Violation Recovery**
   - Automatic remediation attempts
   - Emergency stop on critical violations
   - Violation statistics and reporting

### Example Usage

```python
from nethical.core.verification.runtime_monitor import RuntimeMonitor, StateInvariant, ViolationSeverity

# Create monitor
monitor = RuntimeMonitor(
    enable_remediation=True,
    max_violations=100,
    emergency_stop_on_critical=True
)

# Add custom invariant
monitor.add_invariant(
    StateInvariant(
        name="max_concurrent_requests",
        predicate=lambda ctx: ctx.get("concurrent_requests", 0) <= 1000,
        severity=ViolationSeverity.WARNING,
        description="Concurrent requests exceed threshold"
    )
)

# Check invariants
context = {
    "risk_score": 95,
    "decision": "ALLOW",
    "has_critical_violations": True,  # VIOLATION!
}

violations = monitor.check_all(context)
print(f"Violations: {len(violations)}")  # 1 critical violation

# Get statistics
stats = monitor.get_statistics()
print(f"Emergency stop: {stats['emergency_stop']}")  # True
```

### Decorators

```python
from nethical.core.verification.runtime_monitor import requires, ensures

@requires(lambda ctx: ctx["args"][0] > 0, "Input must be positive")
def process_value(value):
    return value * 2

@ensures(lambda result: result >= 0, "Result must be non-negative")
def calculate_risk(agent):
    # ... calculation
    return risk_score
```

---

## Verification Coverage

### Current Status

| Component | Verification Method | Coverage | Status |
|-----------|-------------------|----------|--------|
| **State Machine** | TLA+ | 100% | ✅ Complete |
| **Policy Consistency** | TLA+ + Z3 | 100% | ✅ Complete |
| **Audit Integrity** | TLA+ | 100% | ✅ Complete |
| **Fundamental Laws** | TLA+ | 100% | ✅ Complete |
| **Runtime Invariants** | Runtime Monitor | 85% | ✅ Active |
| **Contract Assertions** | Runtime Monitor | 70% | 🔄 In Progress |
| **Attack Detection** | Unit Tests | 80% | 🔄 In Progress |

### What is Verified

✅ **Formally Verified:**
- State transition correctness
- Policy non-contradiction
- Audit log immutability
- Fundamental law compliance
- Risk score bounds
- Deterministic decisions

✅ **Runtime Verified:**
- Invariant violations
- Temporal property compliance
- Pre/post condition contracts
- Critical safety checks

### What Requires External Audit

🔍 **Requires Third-Party Audit:**
- Cryptographic implementations (Merkle trees, signatures)
- ML model fairness and bias detection
- Privacy-preserving algorithms (differential privacy)
- Regulatory compliance (GDPR, EU AI Act)
- Security vulnerability assessments

---

## Running Verification

### Full Verification Suite

```bash
# Run all verification tests
make verify

# Or manually:

# 1. TLA+ Model Checking (requires TLA+ Toolbox)
cd formal/tla
./run_tlc.sh

# 2. Z3 Verification
pytest tests/formal/test_z3_verification.py -v

# 3. Runtime Monitor Tests
pytest tests/core/verification/test_runtime_monitor.py -v

# 4. Integration Tests
pytest tests/integration/test_formal_verification.py -v
```

### Continuous Integration

Verification runs automatically in CI/CD pipeline:

```yaml
# .github/workflows/verification.yml
- name: Run Formal Verification
  run: |
    pip install z3-solver
    pytest tests/formal/ -v --tb=short
    
- name: Runtime Verification Tests
  run: |
    pytest tests/core/verification/ -v --tb=short
```

### Performance Impact

- **TLA+ Model Checking**: Offline, no runtime impact
- **Z3 Verification**: Policy load time (+5-10ms per policy set)
- **Runtime Monitor**: Per-decision overhead (~0.1-0.5ms)

**Recommended Configuration:**
- Production: Enable runtime monitor with critical checks only
- Testing: Enable full verification suite
- Development: Use TLA+ for design validation

---

## External Audit Requirements

### What Needs External Audit

1. **Cryptographic Primitives**
   - Merkle tree implementation
   - Digital signature schemes
   - Hash function security
   - **Auditor**: Cryptography expert or firm (e.g., Trail of Bits, NCC Group)

2. **Privacy Guarantees**
   - Differential privacy implementation
   - K-anonymity algorithms
   - Data minimization effectiveness
   - **Auditor**: Privacy researcher or consultant

3. **Regulatory Compliance**
   - GDPR Article 22 compliance
   - EU AI Act conformance assessment
   - ISO 27001 certification
   - **Auditor**: Legal and compliance firm

4. **Security Vulnerabilities**
   - Penetration testing
   - Vulnerability scanning
   - Threat modeling
   - **Auditor**: Security firm (e.g., Cure53, Bishop Fox)

5. **ML Model Fairness**
   - Bias detection accuracy
   - Fairness metric validation
   - Disparate impact analysis
   - **Auditor**: AI ethics researcher or firm

### Audit Certification Roadmap

**Phase 1 (Current):**
- ✅ Formal specifications complete
- ✅ Runtime verification active
- ✅ Test coverage >80%

**Phase 2 (3-6 months):**
- 🔄 External cryptography audit
- 🔄 Security penetration test
- 🔄 Privacy implementation review

**Phase 3 (6-12 months):**
- ⏳ GDPR compliance audit
- ⏳ EU AI Act conformance assessment
- ⏳ ISO 27001 certification

**Phase 4 (12+ months):**
- ⏳ ML fairness audit
- ⏳ Full regulatory certification
- ⏳ Third-party verification publication

---

## References

1. **TLA+**
   - [TLA+ Home](https://lamport.azurewebsites.net/tla/tla.html)
   - [TLC Model Checker](https://github.com/tlaplus/tlaplus)

2. **Z3 SMT Solver**
   - [Z3 GitHub](https://github.com/Z3Prover/z3)
   - [Z3 Tutorial](https://rise4fun.com/z3/tutorial)

3. **Runtime Verification**
   - [RV 2023 Conference](https://rv2023.cs.brown.edu/)
   - [Runtime Verification Inc.](https://runtimeverification.com/)

4. **Formal Methods in Practice**
   - Amazon Web Services: [How AWS Uses Formal Methods](https://cacm.acm.org/magazines/2015/4/184701-how-amazon-web-services-uses-formal-methods/fulltext)
   - Microsoft: [Formal Verification at Microsoft](https://www.microsoft.com/en-us/research/project/formal-verification/)

---

## Contact

For questions about formal verification:
- GitHub Issues: [Report verification issues](https://github.com/V1B3hR/nethical/issues)
- Discussions: [Ask questions](https://github.com/V1B3hR/nethical/discussions)

For external audit inquiries:
- Email: security@nethical.ai (if configured)
- See [SECURITY.md](../SECURITY.md) for responsible disclosure

---

**Last Updated**: 2025-12-11
**Version**: 1.0.0
