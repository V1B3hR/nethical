# Misuse Testing Suite

## Overview

This directory contains adversarial test cases designed to actively attempt to violate the system's negative properties (P-NO-*). These tests complement positive testing by explicitly trying to break security controls and cause forbidden behaviors.

## Test Categories

### 1. Authentication & Authorization Tests (`test_auth_misuse.py`)
- Password spraying and brute force
- JWT token forgery
- Session fixation
- Privilege escalation attempts
- Multi-signature bypass

### 2. Data Integrity Tests (`test_integrity_misuse.py`)
- Audit log backdating
- Merkle tree forgery
- Log injection attacks
- Replay attacks
- Nonce prediction

### 3. Tenant Isolation Tests (`test_isolation_misuse.py`)
- Cross-tenant data access
- SQL injection
- Cache key collision
- Network segmentation bypass
- Metadata leakage

### 4. Denial of Service Tests (`test_dos_misuse.py`)
- Request floods
- Resource exhaustion
- ReDoS (Regular Expression DoS)
- ZIP bombs
- Slowloris attacks

### 5. Policy Manipulation Tests (`test_policy_misuse.py`)
- Policy injection
- Policy tampering
- Signature stripping
- Rollback attacks
- Logic bombs

### 6. Concurrency Tests (`test_concurrency_misuse.py`)
- TOCTOU (Time-of-Check-Time-of-Use)
- Race conditions
- Deadlock scenarios
- Resource contention

### 7. Fuzzing Infrastructure (`test_fuzzing.py`)
- Input mutation
- Grammar-based fuzzing
- Coverage-guided fuzzing
- Crash detection

### 8. Boundary Condition Tests (`test_boundary_misuse.py`)
- Integer overflow/underflow
- Buffer overflows
- Off-by-one errors
- Null pointer dereferences
- Type confusion

## Running Tests

### Run All Misuse Tests
```bash
pytest tests/misuse/ -v
```

### Run Specific Category
```bash
pytest tests/misuse/test_auth_misuse.py -v
```

### Run with Coverage
```bash
pytest tests/misuse/ --cov=nethical --cov-report=html
```

### Run Only Critical Tests
```bash
pytest tests/misuse/ -m critical
```

## Test Markers

- `@pytest.mark.critical` - Tests for critical vulnerabilities
- `@pytest.mark.high` - Tests for high-severity issues
- `@pytest.mark.medium` - Tests for medium-severity issues
- `@pytest.mark.low` - Tests for low-severity issues
- `@pytest.mark.slow` - Tests that take >1 second
- `@pytest.mark.requires_redis` - Tests requiring Redis
- `@pytest.mark.requires_db` - Tests requiring database

## Expected Behavior

**ALL MISUSE TESTS SHOULD FAIL** (i.e., attacks should be blocked).

A test passes when:
- The attack is properly rejected (exception raised)
- Security controls detect and block the malicious action
- Audit logs record the attempted violation

A test fails when:
- The attack succeeds (vulnerability found!)
- Security controls can be bypassed
- No audit log entry is created

## Continuous Integration

These tests run in CI/CD on every commit. Any passing misuse test (i.e., successful attack) will fail the build and block deployment.

## Test Data

Use synthetic test data only. Never use production data or real credentials.

Test fixtures in `conftest.py` provide:
- Mock authentication tokens
- Synthetic policy data
- Fake audit logs
- Test tenant configurations

## Security Advisory

Results from these tests may reveal vulnerabilities. Treat test results as confidential and follow responsible disclosure practices.

## Contributing

When adding new misuse tests:
1. Document the attack vector
2. Reference the negative property being tested (P-NO-*)
3. Include steps to reproduce
4. Add appropriate markers (@pytest.mark.critical, etc.)
5. Ensure test is hermetic (no external dependencies)

## Success Criteria

- ✅ 100+ adversarial test cases
- ✅ All negative properties (P-NO-*) have ≥10 test cases each
- ✅ 100% of misuse tests pass (attacks blocked)
- ✅ Code coverage >90% for security-critical paths
- ✅ Fuzzing runs for 24 hours with 0 crashes
