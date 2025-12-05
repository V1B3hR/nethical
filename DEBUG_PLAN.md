# Nethical Debug Plan

**Project:** Nethical AI Governance Framework  
**Version:** 2.3.0  
**Purpose:** Comprehensive debugging guide for all components  
**Last Updated:** 2025-12-05  
**Status:** Active

---

## Table of Contents

1. [Debug Philosophy & Principles](#1-debug-philosophy--principles)
2. [Environment Setup](#2-environment-setup)
3. [Module-by-Module Debug Guide](#3-module-by-module-debug-guide)
4. [Layered Debugging Approach](#4-layered-debugging-approach)
5. [Performance Debugging](#5-performance-debugging)
6. [Test Suite Debugging](#6-test-suite-debugging)
7. [Security-Sensitive Debugging](#7-security-sensitive-debugging)
8. [Distributed System Debugging](#8-distributed-system-debugging)
9. [Chaos Engineering Debug Scenarios](#9-chaos-engineering-debug-scenarios)
10. [Compliance Debugging](#10-compliance-debugging)
11. [Common Issues & Solutions](#11-common-issues--solutions)
12. [Debug Tools Reference](#12-debug-tools-reference)
13. [Emergency Debug Procedures](#13-emergency-debug-procedures)
14. [Debug Logging Configuration](#14-debug-logging-configuration)
15. [Appendices](#15-appendices)

---

## 1. Debug Philosophy & Principles

### 1.1 Safety-First Debugging

Nethical is a **safety-critical AI governance system**. All debugging activities must adhere to these core principles:

| Principle | Description |
|-----------|-------------|
| **Non-Intrusive** | Debugging should not compromise the integrity of governance decisions |
| **Audit Trail Preservation** | All debug activities must be logged and traceable |
| **Fail-Safe Default** | When in doubt, prefer BLOCK over ALLOW decisions during debugging |
| **Isolation** | Debug activities should be isolated from production workloads |
| **Reproducibility** | All issues should be reproducible in development/staging environments |

### 1.2 The 25 Fundamental Laws Awareness

When debugging governance decisions, always consider the relevant [Fundamental Laws](FUNDAMENTAL_LAWS.md):

- **Law 15 (Audit Compliance)**: Maintain appropriate logs during debugging
- **Law 21 (Human Safety Priority)**: Never disable safety checks in production
- **Law 23 (Fail-Safe Design)**: Ensure fail-safe modes remain operational

### 1.3 Production vs Development Modes

```yaml
# Development Mode (config/example_config.yaml)
debug:
  enabled: true
  verbose_logging: true
  trace_decisions: true
  mock_external_services: true
  safety_checks: enabled  # NEVER disable in production

# Production Mode
debug:
  enabled: false
  verbose_logging: false
  trace_decisions: false  # Enable temporarily with audit approval
  mock_external_services: false
  safety_checks: enabled
```

### 1.4 Debug Decision Matrix

```
Issue Detected
    |
    v
Safety Critical? --YES--> Enable Safe Mode --> Debug with Audit Trail
    |                                                    |
   NO                                                    v
    |                                              Apply Fix
    v                                                    |
Production Environment? --YES--> Replicate in Staging   |
    |                                  |                 |
   NO                                  v                 |
    |                            Debug Directly <--------+
    v                                  |
Debug Directly                         v
    |                            Verify Fix
    v                                  |
Apply Fix                              v
    |                         Deploy with Monitoring
    v
Verify Fix
    |
    v
Deploy with Monitoring
```

---

## 2. Environment Setup

### 2.1 Development Environment Configuration

#### Prerequisites

```bash
# Required Python version
python --version  # >= 3.10

# Clone and setup
git clone https://github.com/V1B3hR/nethical.git
cd nethical

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install with development dependencies
pip install -e ".[test]"
pip install -r requirements-dev.txt
```

#### Debug Dependencies

```bash
# Install debug-specific tools
pip install debugpy          # VS Code debugging
pip install ipdb             # Enhanced interactive debugger
pip install memory_profiler  # Memory profiling
pip install py-spy           # Sampling profiler
pip install line_profiler    # Line-by-line profiling
```

### 2.2 Environment Variables for Debug Modes

```bash
# Core Debug Settings
export NETHICAL_DEBUG=true
export NETHICAL_LOG_LEVEL=DEBUG
export NETHICAL_TRACE_DECISIONS=true
export NETHICAL_TRACE_POLICIES=true

# Performance Profiling
export NETHICAL_PROFILE_ENABLED=true
export NETHICAL_PROFILE_OUTPUT=/tmp/nethical_profile

# Component-Specific Debug
export NETHICAL_DEBUG_GOVERNANCE=true
export NETHICAL_DEBUG_SECURITY=true
export NETHICAL_DEBUG_DETECTORS=true
export NETHICAL_DEBUG_COMPLIANCE=true
export NETHICAL_DEBUG_CACHE=true

# Test Environment
export NETHICAL_TEST_MODE=true
export NETHICAL_MOCK_EXTERNAL=true
```

### 2.3 Docker-Based Debugging Setup

```yaml
# docker-compose.debug.yml
version: '3.8'

services:
  nethical-debug:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - NETHICAL_DEBUG=true
      - NETHICAL_LOG_LEVEL=DEBUG
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
    ports:
      - "8000:8000"
      - "5678:5678"  # debugpy port
    volumes:
      - ./nethical:/app/nethical
      - ./tests:/app/tests
      - ./logs:/app/logs
    command: >
      python -m debugpy --listen 0.0.0.0:5678 --wait-for-client
      -m uvicorn nethical.api:app --host 0.0.0.0 --reload

  redis-debug:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --loglevel verbose
```

```bash
# Launch debug environment
docker-compose -f docker-compose.debug.yml up
```

### 2.4 IDE Configuration

#### VS Code (.vscode/launch.json)

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Nethical API",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["nethical.api:app", "--reload", "--port", "8000"],
      "env": {
        "NETHICAL_DEBUG": "true",
        "NETHICAL_LOG_LEVEL": "DEBUG"
      },
      "justMyCode": false
    },
    {
      "name": "Debug Current Test",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["${file}", "-v", "-s"],
      "justMyCode": false
    },
    {
      "name": "Attach to Docker",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "/app"
        }
      ]
    }
  ]
}
```

---

## 3. Module-by-Module Debug Guide

### 3.1 Core Governance Engine (nethical/core/)

#### Common Issues

| Symptom | Likely Cause | Debug Approach |
|---------|--------------|----------------|
| Decisions always BLOCK | Overly restrictive policy | Check policy threshold in config |
| Decisions always ALLOW | Missing detectors | Verify detector registration |
| Slow decision latency | Complex policy evaluation | Profile policy engine |
| Inconsistent decisions | Race conditions | Check thread safety |

#### Debug Commands

```python
# Debug governance decision flow
from nethical.core import IntegratedGovernance

gov = IntegratedGovernance(debug=True)

# Enable verbose tracing
gov.enable_trace(
    trace_policies=True,
    trace_detectors=True,
    trace_judges=True
)

# Process action with full trace
result = gov.process_action(
    agent_id="debug-agent",
    action="test_action",
    context={"debug": True}
)

# Print decision trace
print(gov.get_decision_trace())
```

#### Log Locations

```
logs/governance/decisions.log       # All governance decisions
logs/governance/policy_eval.log     # Policy evaluation details
logs/governance/trace.log           # Full execution traces (debug mode)
```

#### Key Breakpoints

```python
# nethical/core/governance.py
# Line ~150: process_action() entry point
# Line ~250: policy evaluation loop
# Line ~350: final decision aggregation

# nethical/core/integrated_governance.py
# Line ~100: IntegratedGovernance.evaluate()
# Line ~200: detector coordination
# Line ~300: judge invocation
```

#### Health Check

```bash
# Governance health endpoint
curl http://localhost:8000/health/governance

# Expected response
# {
#   "status": "healthy",
#   "components": {
#     "policy_engine": "ok",
#     "detectors": "ok",
#     "judges": "ok",
#     "risk_engine": "ok"
#   },
#   "active_policies": 15,
#   "registered_detectors": 12
# }
```

### 3.2 Governance Laws (nethical/governance/)

#### Common Issues

| Symptom | Likely Cause | Debug Approach |
|---------|--------------|----------------|
| Law not evaluated | Law not registered | Check FundamentalLawsRegistry |
| Wrong compliance score | Weight misconfiguration | Verify category weights |
| Missing law violations | Detection threshold too high | Lower sensitivity in config |

#### Debug Commands

```python
from nethical.governance import FundamentalLawsRegistry

registry = FundamentalLawsRegistry()

# List all registered laws
for law in registry.get_all_laws():
    print(f"Law {law.number}: {law.title} - Active: {law.active}")

# Test specific law evaluation
from nethical.judges import LawJudge

judge = LawJudge(debug=True)
result = judge.evaluate(
    action="data_access",
    context={"purpose": "analytics"},
    laws_to_check=[9, 18, 22]  # Transparency, Non-Deception, Digital Security
)
print(result.violations)
print(result.compliance_score)
```

### 3.3 Security Module (nethical/security/)

#### Common Issues

| Symptom | Likely Cause | Debug Approach |
|---------|--------------|----------------|
| Authentication failures | Token expiration | Check JWT configuration |
| TLS handshake errors | Certificate issues | Verify cert chain |
| Rate limiting false positives | Threshold too low | Adjust rate limits |

#### Debug Commands

```python
from nethical.security import SecurityManager, ZeroTrustValidator

# Debug authentication
security = SecurityManager(debug=True)
security.debug_auth_flow(
    token="<jwt_token>",
    show_claims=True,
    verify_signature=True
)

# Debug zero-trust validation
ztv = ZeroTrustValidator(debug=True)
result = ztv.validate_request(
    source_ip="192.168.1.100",
    user_agent="test-agent/1.0",
    request_path="/api/v1/evaluate"
)
print(result.trust_score)
print(result.validation_details)
```

#### Log Locations

```
logs/security/auth.log          # Authentication events
logs/security/access.log        # Access control decisions
logs/security/threats.log       # Threat detection alerts
logs/security/rate_limit.log    # Rate limiting events
```

### 3.4 Detectors (nethical/detectors/)

#### Detector Types

| Detector | Purpose | Debug Focus |
|----------|---------|-------------|
| SafetyDetector | Safety constraint violations | Threshold tuning |
| EthicalDetector | Ethical violations | Pattern matching |
| ManipulationDetector | Manipulation attempts | ML model accuracy |
| DarkPatternDetector | Dark pattern detection | Pattern database |
| CognitiveWarfareDetector | Cognitive attacks | Signal detection |
| SystemLimitsDetector | Resource abuse | Limit configuration |

#### Debug Commands

```python
from nethical.detectors import DetectorSuite

suite = DetectorSuite(debug=True)

# Test all detectors against a sample
result = suite.analyze(
    text="Test input for detection",
    context={"source": "debug"},
    verbose=True
)

for detection in result.detections:
    print(f"Detector: {detection.detector}")
    print(f"Type: {detection.type}")
    print(f"Confidence: {detection.confidence}")
    print(f"Details: {detection.details}")
    print("---")

# Debug specific detector
from nethical.detectors import SafetyDetector

detector = SafetyDetector(debug=True)
detector.set_threshold(0.5)  # Lower threshold for testing
result = detector.detect(
    "potentially unsafe content",
    return_features=True
)
```

### 3.5 Compliance Module (nethical/compliance/)

#### Compliance Frameworks

| Framework | Module | Debug Focus |
|-----------|--------|-------------|
| GDPR | gdpr_validator.py | Consent tracking, data rights |
| EU AI Act | eu_ai_act_validator.py | Risk classification |
| ISO 27001 | iso27001_validator.py | Control mapping |
| HIPAA | hipaa_validator.py | PHI handling |

#### Debug Commands

```python
from nethical.compliance import ComplianceSuite

suite = ComplianceSuite(debug=True)

# Run compliance check with detailed output
result = suite.validate(
    action="process_user_data",
    context={
        "data_type": "personal",
        "purpose": "analytics",
        "region": "EU",
        "consent_obtained": True
    },
    frameworks=["GDPR", "EU_AI_ACT"]
)

for framework, status in result.items():
    print(f"\n{framework}:")
    print(f"  Compliant: {status.compliant}")
    print(f"  Score: {status.score}")
    print(f"  Violations: {status.violations}")
    print(f"  Recommendations: {status.recommendations}")
```

### 3.6 Edge Deployment (nethical/edge/)

#### Common Issues

| Symptom | Likely Cause | Debug Approach |
|---------|--------------|----------------|
| High latency (>10ms) | Model not optimized | Profile inference |
| Offline mode failures | Stale policy cache | Check sync status |
| Memory pressure | Model too large | Enable model quantization |

#### Debug Commands

```bash
# Edge device diagnostics
nethical edge diagnose --verbose

# Check policy sync status
nethical edge sync-status

# Force policy refresh
nethical edge sync --force

# Profile edge performance
nethical edge benchmark --iterations 1000
```

```python
from nethical.edge import EdgeRuntime

runtime = EdgeRuntime(debug=True)

# Check runtime status
print(runtime.status())

# Profile decision latency
import time
times = []
for _ in range(100):
    start = time.perf_counter()
    runtime.evaluate({"action": "test"})
    times.append((time.perf_counter() - start) * 1000)

print(f"p50: {sorted(times)[50]:.2f}ms")
print(f"p99: {sorted(times)[99]:.2f}ms")
```

### 3.7 Policy Engine (nethical/policy/)

#### Debug Commands

```python
from nethical.policy import PolicyEngine

engine = PolicyEngine(debug=True)

# Load and validate policies
engine.load_policies("policies/")
errors = engine.validate_all()
for error in errors:
    print(f"Policy {error.policy_id}: {error.message}")

# Test policy evaluation
result = engine.evaluate(
    action="execute_code",
    context={"code": "print('hello')"},
    trace=True
)

# Print evaluation trace
for step in result.trace:
    print(f"Step: {step.policy_name}")
    print(f"  Matched: {step.matched}")
    print(f"  Decision: {step.decision}")
    print(f"  Reason: {step.reason}")
```

### 3.8 Storage Layer (nethical/storage/)

#### Common Issues

| Symptom | Likely Cause | Debug Approach |
|---------|--------------|----------------|
| Slow writes | Index issues | Check database indexes |
| Merkle verification failures | Data corruption | Run integrity check |
| Storage growth | Retention policy | Review cleanup jobs |

#### Debug Commands

```python
from nethical.storage import AuditStorage

storage = AuditStorage(debug=True)

# Verify Merkle tree integrity
result = storage.verify_merkle_tree()
print(f"Verified: {result.verified}")
print(f"Entries checked: {result.entries_checked}")
print(f"Corrupted entries: {result.corrupted}")

# Check storage statistics
stats = storage.get_statistics()
print(f"Total entries: {stats.total_entries}")
print(f"Storage size: {stats.size_bytes / 1024 / 1024:.2f} MB")
print(f"Oldest entry: {stats.oldest_entry}")
```

### 3.9 Cache Layer (nethical/cache/)

#### Debug Commands

```python
from nethical.cache import CacheManager

cache = CacheManager(debug=True)

# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats.hit_rate:.2%}")
print(f"Miss rate: {stats.miss_rate:.2%}")
print(f"Evictions: {stats.evictions}")
print(f"Memory usage: {stats.memory_bytes / 1024 / 1024:.2f} MB")

# Debug specific cache key
cache.debug_key("policy:agent-001")

# Clear cache (development only)
# CRITICAL: Never use in production - add environment check
# if os.getenv('NETHICAL_ENV') != 'production':
#     cache.clear()
cache.clear()  # WARNING: Verify non-production environment first
```

### 3.10 API Layer (nethical/api/)

#### Debug Commands

```bash
# Test API endpoints
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -H "X-Debug: true" \
  -d '{"agent_id": "test", "action": "test_action"}'

# Get detailed error information
curl http://localhost:8000/api/v1/debug/last-error

# API metrics
curl http://localhost:8000/metrics
```

---

## 4. Layered Debugging Approach

### 4.1 Layer Overview

```
+-------------------------------------------------------------+
| Layer 1: API/Interface Layer                                |
|   - Request validation, authentication, rate limiting       |
+-------------------------------------------------------------+
| Layer 2: Governance Engine                                  |
|   - Decision orchestration, policy coordination             |
+-------------------------------------------------------------+
| Layer 3: Policy Evaluation                                  |
|   - Policy matching, rule evaluation, condition checking    |
+-------------------------------------------------------------+
| Layer 4: Detection Layer                                    |
|   - Safety detectors, ethical detectors, anomaly detection  |
+-------------------------------------------------------------+
| Layer 5: Security Layer                                     |
|   - Authentication, authorization, encryption               |
+-------------------------------------------------------------+
| Layer 6: Storage/Audit Layer                                |
|   - Merkle anchoring, audit logs, data persistence          |
+-------------------------------------------------------------+
| Layer 7: Integration Layer                                  |
|   - LangChain, Claude, Gemini, Grok integrations            |
+-------------------------------------------------------------+
```

### 4.2 Layer 1: API/Interface Layer Debugging

```python
# Enable API debug middleware
from nethical.middleware import DebugMiddleware

app.add_middleware(
    DebugMiddleware,
    log_requests=True,
    log_responses=True,
    log_headers=True,
    redact_sensitive=True
)
```

### 4.3 Layer 2: Governance Engine Debugging

```python
from nethical.core import GovernanceEngine

engine = GovernanceEngine()
engine.enable_debug_mode()

result = engine.process(action_request)
timeline = engine.get_execution_timeline()
for event in timeline:
    print(f"{event.timestamp} | {event.component} | {event.duration_ms}ms")
```

---

## 5. Performance Debugging

### 5.1 Latency Profiling

#### Performance Targets (from docs/ops/SLOs.md)

| Metric | Target | Edge Target |
|--------|--------|-------------|
| p50 | <50ms | <5ms |
| p95 | <200ms | <15ms |
| p99 | <500ms | <25ms |

#### Profiling Commands

```python
from nethical.performanceprofiling import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.trace("governance_decision"):
    result = governance.process_action(action_request)

report = profiler.get_report()
print(report.to_markdown())
```

#### Using py-spy (Sampling Profiler)

```bash
py-spy top --pid $(pgrep -f "nethical")
py-spy record -o profile.svg --pid $(pgrep -f "nethical")
```

### 5.2 Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    pass
```

### 5.3 Throughput Analysis

```bash
wrk -t4 -c100 -d30s http://localhost:8000/api/v1/evaluate

hey -n 10000 -c 100 -m POST \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"load-test","action":"test"}' \
  http://localhost:8000/api/v1/evaluate
```

---

## 6. Test Suite Debugging

### 6.1 Test Suite Overview

**Total Tests:** 497  
**Pass Rate:** ~88%+  
**See:** [tests/TEST_STATUS.md](tests/TEST_STATUS.md)

#### Test Categories

| Category | Location | Purpose |
|----------|----------|---------|
| Phase Tests | tests/test_phase*.py | Feature phase validation |
| Unit Tests | tests/unit/ | Component isolation |
| Adversarial | tests/adversarial/ | Attack simulation |
| Performance | tests/performance/ | Latency/throughput |
| Security | tests/security/ | Security validation |
| Compliance | tests/validation/ | Compliance verification |
| Chaos | tests/chaos/ | Resilience testing |

### 6.2 Running Individual Test Categories

```bash
python -m pytest tests/test_phase3.py -v
python -m pytest tests/test_phase3.py::TestRiskEngine -v
python -m pytest tests/ -k "governance" -v
python -m pytest tests/ -k "not marketplace" -v
```

### 6.3 Debugging Failing Tests

```bash
python -m pytest tests/test_failing.py -v -s --tb=long
python -m pytest tests/ -x
python -m pytest tests/ --showlocals
python -m pytest tests/test_failing.py --pdb
```

### 6.4 Coverage Analysis

```bash
python -m pytest tests/ --cov=nethical --cov-report=html
coverage report --show-missing
```

---

## 7. Security-Sensitive Debugging

### 7.1 Safe Debugging of Security Components

> **WARNING**: Security debugging requires extra caution. Never log sensitive data in production.

```python
from nethical.security import SecurityDebugger

debugger = SecurityDebugger(
    redact_secrets=True,
    redact_tokens=True,
    redact_pii=True
)

debugger.debug_credentials(
    credential_id="cred-001",
    show_type=True,
    show_expiry=True,
    show_value=False  # CRITICAL: NEVER set to True in production
)
```

### 7.2 Audit Log Integrity Verification

```python
from nethical.storage import AuditVerifier

verifier = AuditVerifier()

result = verifier.verify_all()
print(f"Total entries: {result.total}")
print(f"Verified: {result.verified}")
print(f"Corrupted: {result.corrupted}")
```

### 7.3 HSM Debugging Considerations

```python
from nethical.security.hsm import HSMDebugger

hsm_debug = HSMDebugger(
    hsm_slot=0,
    readonly=True  # CRITICAL: Always use readonly mode
)

print(hsm_debug.get_status())
hsm_debug.verify_key_access("signing-key-001")
```

---

## 8. Distributed System Debugging

### 8.1 Multi-Region Debugging

```bash
nethical cluster status --all-regions
nethical cluster debug --region eu-west-1
nethical cluster diff-config --regions us-east-1,eu-west-1
```

### 8.2 CRDT Synchronization Debugging

```python
from nethical.sync import CRDTDebugger

debugger = CRDTDebugger()

status = debugger.get_sync_status()
for region, state in status.items():
    print(f"{region}: {state.vector_clock}")
```

### 8.3 Redis Cluster Debugging

```bash
redis-cli -c cluster info
redis-cli -c cluster nodes
```

---

## 9. Chaos Engineering Debug Scenarios

### 9.1 Network Chaos Debugging

```python
from nethical.chaos import NetworkChaos

chaos = NetworkChaos()

with chaos.inject_latency(delay_ms=100, jitter_ms=20):
    result = governance.process_action(...)
    print(f"Latency: {result.latency_ms}ms")
```

### 9.2 Resource Exhaustion Debugging

```python
from nethical.chaos import ResourceChaos

chaos = ResourceChaos()

with chaos.memory_pressure(usage_percent=85):
    result = governance.process_action(...)
```

### 9.3 Dependency Failure Debugging

```python
from nethical.chaos import DependencyChaos

chaos = DependencyChaos()

with chaos.fail_dependency("postgresql"):
    result = governance.process_action(...)
    assert result.source == "cache"
```

---

## 10. Compliance Debugging

### 10.1 GDPR Compliance Verification

```python
from nethical.compliance import GDPRDebugger

debugger = GDPRDebugger()

rights_check = debugger.verify_data_subject_rights()
print(f"Access (Art. 15): {rights_check.access}")
print(f"Erasure (Art. 17): {rights_check.erasure}")
```

### 10.2 EU AI Act Compliance Checks

```python
from nethical.compliance import EUAIActDebugger

debugger = EUAIActDebugger()

classification = debugger.verify_risk_classification()
print(f"System classification: {classification.level}")
```

### 10.3 ISO 27001 Audit Debugging

```python
from nethical.compliance import ISO27001Debugger

debugger = ISO27001Debugger()

controls = debugger.verify_controls()
for control_id, status in controls.items():
    print(f"{control_id}: {status.status}")
```

---

## 11. Common Issues & Solutions

### 11.1 Categorized Troubleshooting Guide

#### Governance Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| All decisions BLOCK | Every action blocked | Check policy thresholds |
| All decisions ALLOW | Nothing blocked | Verify detectors enabled |
| Inconsistent decisions | Different results | Check for race conditions |
| Slow decisions | High latency | Profile policy evaluation |
| Missing audit logs | Logs not appearing | Check storage connection |

#### Security Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Auth failures | 401 responses | Check JWT configuration |
| Rate limiting | 429 responses | Adjust rate limits |
| TLS errors | Handshake failures | Verify certificate chain |

### 11.2 Error Code Reference

| Code | Description | Debug Action |
|------|-------------|--------------|
| E001 | Policy evaluation failed | Check policy syntax |
| E002 | Detector init error | Verify config |
| E003 | Storage write failed | Check DB connectivity |
| E004 | Auth failed | Verify credentials |
| E005 | Rate limit exceeded | Check rate config |
| E006 | Merkle verification failed | Run integrity check |

---

## 12. Debug Tools Reference

### 12.1 Built-in Debug Utilities

- nethical debug status - System status
- nethical debug trace - Trace request
- nethical debug policies - List policies
- nethical debug detectors - List detectors
- nethical debug health-check - Health check

### 12.2 External Tools

Prometheus metrics, Grafana dashboards in dashboards/ directory.

---

## 13. Emergency Debug Procedures

### 13.1 Production Incident Debugging

CRITICAL: Follow incident response procedures first.

### 13.2 Kill Switch

See config/kill_switch.yaml

### 13.3 Recovery

Use nethical recovery commands.

---

## 14. Debug Logging Configuration

### 14.1 Log Levels

Set via NETHICAL_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR

---

## 15. Appendices

### 15.1 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NETHICAL_DEBUG | Debug mode | false |
| NETHICAL_LOG_LEVEL | Log level | INFO |
| NETHICAL_TRACE_DECISIONS | Trace decisions | false |
| NETHICAL_TEST_MODE | Test mode | false |
| NETHICAL_SAFE_MODE | Safe mode | false |

### 15.2 Escalation

| Level | Contact | Response |
|-------|---------|----------|
| P1 Critical | See internal wiki for security contacts | 15 min |
| P2 High | See internal wiki for on-call contacts | 1 hour |
| P3 Medium | See internal wiki for support contacts | 4 hours |
| P4 Low | GitHub Issues | 24 hours |

### 15.3 Related Docs

- README.md - Project overview
- ARCHITECTURE.md - System architecture
- SECURITY.md - Security policy
- PRIVACY.md - Privacy policy
- FUNDAMENTAL_LAWS.md - 25 Laws
- tests/TEST_STATUS.md - Test status
- docs/ops/SLOs.md - SLOs
- docs/debugging/ - Debug guides

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-05  
**Maintainer:** Nethical Core Team  
**Review Cycle:** Quarterly
