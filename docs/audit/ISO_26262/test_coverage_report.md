# Test Coverage Report - MC/DC Analysis

## Document Information

| Field | Value |
|-------|-------|
| Document ID | TC-001 |
| Version | 1.0 |
| ASIL Classification | D |
| Date | 2025-12-03 |
| Author | Nethical Safety Team |
| Status | Template |

## 1. Overview

This document provides the test coverage analysis for Nethical's AI Governance System, demonstrating compliance with ISO 26262 ASIL-D requirements for Modified Condition/Decision Coverage (MC/DC).

## 2. ASIL-D Coverage Requirements

Per ISO 26262-6 Table 9:

| Test Method | ASIL A | ASIL B | ASIL C | ASIL D |
|-------------|--------|--------|--------|--------|
| Statement Coverage | HR | HR | HR | HR |
| Branch Coverage | R | HR | HR | HR |
| MC/DC | R | R | HR | HR |

**Legend:** HR = Highly Recommended, R = Recommended

## 3. Coverage Targets

### Safety-Critical Modules

| Module | Statement | Branch | MC/DC | Status |
|--------|-----------|--------|-------|--------|
| `nethical/edge/local_governor.py` | 100% | 100% | 100% | Target |
| `nethical/edge/safe_defaults.py` | 100% | 100% | 100% | Target |
| `nethical/edge/offline_fallback.py` | 100% | 100% | 100% | Target |
| `nethical/edge/fast_detector.py` | 100% | 100% | 100% | Target |
| `nethical/core/risk_engine.py` | 100% | 100% | 100% | Target |

### Supporting Modules

| Module | Statement | Branch | MC/DC | Status |
|--------|-----------|--------|-------|--------|
| `nethical/cache/l1_memory.py` | 95%+ | 95%+ | N/A | Target |
| `nethical/cache/cache_hierarchy.py` | 95%+ | 95%+ | N/A | Target |
| `nethical/streaming/policy_subscriber.py` | 90%+ | 90%+ | N/A | Target |

## 4. MC/DC Methodology

### 4.1 Definition

Modified Condition/Decision Coverage (MC/DC) requires that:
1. Every entry and exit point is invoked
2. Every decision takes every possible outcome
3. Each condition in a decision independently affects the decision's outcome

### 4.2 Example Analysis

**Code Under Test:**
```python
def should_block_action(self, risk_score: float, has_violation: bool, 
                         is_critical: bool) -> bool:
    return risk_score > 0.8 or (has_violation and is_critical)
```

**MC/DC Test Cases:**

| TC | risk_score > 0.8 | has_violation | is_critical | Result | Covers |
|----|------------------|---------------|-------------|--------|--------|
| 1 | False | False | False | False | Baseline |
| 2 | **True** | False | False | **True** | C1 independence |
| 3 | False | **True** | True | **True** | C2 independence |
| 4 | False | True | **False** | False | C3 independence |

**Independence Pairs:**
- Condition 1 (risk_score > 0.8): TC1 ↔ TC2
- Condition 2 (has_violation): TC1 ↔ TC3 (with is_critical=True held)
- Condition 3 (is_critical): TC3 ↔ TC4

## 5. Tool Configuration

### 5.1 Coverage Tools

| Tool | Purpose | Configuration |
|------|---------|---------------|
| coverage.py | Statement/Branch | `--branch` flag |
| pytest-cov | Integration | `.coveragerc` |
| MutPy | Mutation testing | Custom config |
| Custom MC/DC | MC/DC analysis | `tests/mcdc/` |

### 5.2 Coverage Configuration

```ini
# .coveragerc
[run]
branch = True
source = nethical/edge,nethical/core

[report]
fail_under = 100
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:

[html]
directory = coverage_html
```

## 6. Safety-Critical Function Coverage

### 6.1 EdgeGovernor.evaluate()

**Function Signature:**
```python
async def evaluate(self, action: Action, context: Context) -> Decision
```

**Decision Points:**
1. Cache hit check
2. Policy match
3. Risk threshold comparison
4. Fundamental law check
5. Safe default trigger

**Coverage Status:**

| Decision Point | Conditions | Test Cases | MC/DC % |
|----------------|------------|------------|---------|
| DP1: Cache | 2 | 4 | 100% |
| DP2: Policy | 3 | 7 | 100% |
| DP3: Risk | 2 | 4 | 100% |
| DP4: Laws | 3 | 7 | 100% |
| DP5: Default | 2 | 4 | 100% |
| **Total** | **12** | **26** | **100%** |

### 6.2 SafeDefaults.get_default()

**Function Signature:**
```python
def get_default(self, action_type: str, severity: Severity) -> Decision
```

**Decision Points:**
1. Action type classification
2. Severity level
3. Domain-specific rules

**Coverage Status:**

| Decision Point | Conditions | Test Cases | MC/DC % |
|----------------|------------|------------|---------|
| DP1: Action | 5 | 11 | 100% |
| DP2: Severity | 4 | 9 | 100% |
| DP3: Domain | 3 | 7 | 100% |
| **Total** | **12** | **27** | **100%** |

### 6.3 OfflineFallback.handle_network_loss()

**Function Signature:**
```python
def handle_network_loss(self, reason: str, duration_ms: int) -> FallbackState
```

**Coverage Status:**

| Decision Point | Conditions | Test Cases | MC/DC % |
|----------------|------------|------------|---------|
| DP1: Duration | 3 | 7 | 100% |
| DP2: Reason | 4 | 9 | 100% |
| DP3: State | 2 | 4 | 100% |
| **Total** | **9** | **20** | **100%** |

## 7. Test Suite Structure

```
tests/
├── mcdc/
│   ├── __init__.py
│   ├── test_edge_governor_mcdc.py
│   ├── test_safe_defaults_mcdc.py
│   ├── test_offline_fallback_mcdc.py
│   └── test_risk_engine_mcdc.py
├── edge/
│   ├── test_local_governor.py
│   ├── test_safe_defaults.py
│   └── test_offline_fallback.py
└── integration/
    ├── test_edge_integration.py
    └── test_failover.py
```

## 8. Mutation Testing

### 8.1 Purpose

Mutation testing validates test suite effectiveness by introducing deliberate faults (mutants) and verifying tests detect them.

### 8.2 Mutation Operators

| Operator | Description | Example |
|----------|-------------|---------|
| AOR | Arithmetic operator replacement | `+` → `-` |
| ROR | Relational operator replacement | `>` → `>=` |
| LCR | Logical connector replacement | `and` → `or` |
| SDL | Statement deletion | Remove statement |
| UOI | Unary operator insertion | `x` → `-x` |

### 8.3 Target Mutation Score

| Module | Target | Current |
|--------|--------|---------|
| Safety-critical | ≥ 95% | TBD |
| Supporting | ≥ 90% | TBD |

## 9. Continuous Integration

### 9.1 Coverage Gates

```yaml
# .github/workflows/safety-coverage.yml
safety-coverage:
  runs-on: ubuntu-latest
  steps:
    - name: Run safety-critical tests
      run: |
        pytest tests/edge/ tests/mcdc/ \
          --cov=nethical.edge \
          --cov=nethical.core \
          --cov-branch \
          --cov-fail-under=100
          
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        flags: safety-critical
```

### 9.2 Reporting

Coverage reports are:
1. Generated on every commit
2. Stored as CI artifacts
3. Trended over time
4. Linked to requirements

## 10. Traceability

### 10.1 Requirements to Tests

| Requirement | Test Cases | Coverage |
|-------------|------------|----------|
| SWR-001 | TC-001..TC-005 | 100% |
| SWR-002 | TC-006..TC-012 | 100% |
| SWR-003 | TC-013..TC-020 | 100% |

### 10.2 Traceability Matrix Location

`docs/certification/ISO_26262/traceability_matrix.xlsx`

## 11. Deviations

### 11.1 Accepted Deviations

| ID | Code Location | Reason | Approval |
|----|---------------|--------|----------|
| DEV-001 | `__repr__` methods | Debug only | Safety Manager |
| DEV-002 | Type hints | Static analysis | Safety Manager |

### 11.2 Deviation Approval Process

1. Developer identifies untestable code
2. Safety justification written
3. Safety Manager reviews
4. Documented in deviation log

## 12. Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Test Lead | | | |
| Safety Manager | | | |
| Quality Assurance | | | |

---

**Classification:** ISO 26262 ASIL-D Development  
**Retention Period:** Life of product + 15 years
