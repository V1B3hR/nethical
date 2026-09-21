# Governance & Observability Features

This document describes the governance and observability features implemented in Nethical.

## Table of Contents

1. [Governance & Ethics](#governance--ethics)
   - [Policy Grammar (EBNF)](#policy-grammar-ebnf)
   - [Policy Simulator](#policy-simulator)
   - [Ethics Benchmark](#ethics-benchmark)
   - [Threshold Configuration](#threshold-configuration)
2. [Observability](#observability)
   - [Metrics Collection](#metrics-collection)
   - [Distributed Tracing](#distributed-tracing)
   - [Log Sanitization](#log-sanitization)
   - [Alert Rules](#alert-rules)
3. [Usage Examples](#usage-examples)

---

## Governance & Ethics

### Policy Grammar (EBNF)

The policy language is formally defined using EBNF grammar in `nethical/governance/policy_grammar.ebnf`.

**Key Features:**
- Formal specification of policy structure
- Support for conditions, actions, and region overlays
- Comparison operators: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `contains`, `startswith`, `endswith`, `matches`
- Decision values: `ALLOW`, `DENY`, `RESTRICT`, `WARN`, `AUDIT`, `QUARANTINE`

**Example Policy Structure:**
```yaml
metadata:
  name: "my_policy"
  version: "1.0.0"

defaults:
  decision: ALLOW
  deny_overrides: true

rules:
  - id: anti-manipulation
    enabled: true
    priority: 100
    when:
      any:
        - "manipulation.override_attempt == true"
        - exists: "manipulation.vector"
    action:
      decision: DENY
      add_disclaimer: "Manipulative content detected"
      escalate: true
      tags: ["manipulation"]
```

### Policy Simulator

CLI tool for testing policies before deployment.

**Location:** `cli/policy_simulator`

**Commands:**

1. **Simulate** - Test a policy against input scenarios:
```bash
./cli/policy_simulator simulate <policy_file> <input_file> [options]

Options:
  --region US|EU|UK|NATO    Region to simulate (default: US)
  --format text|json|yaml   Output format
  --verbose                 Show full input data
  --output <file>           Output to file
```

2. **Dry-run** - Compare old vs new policy outcomes:
```bash
./cli/policy_simulator dry-run <old_policy> <new_policy> <test_cases> [options]

Options:
  --region US|EU|UK|NATO    Region to compare
  --format text|json|yaml   Output format
  --output <file>           Output to file
```

**Example:**
```bash
# Simulate a policy
./cli/policy_simulator simulate \
  examples/governance/sample_policy.yaml \
  examples/governance/policy_test_cases.yaml

# Compare policy versions
./cli/policy_simulator dry-run \
  policy_v1.yaml \
  policy_v2.yaml \
  test_cases.yaml
```

### Ethics Benchmark

System for measuring ethical AI performance with precision, recall, and other metrics.

**Module:** `nethical.governance.ethics_benchmark`

**Key Features:**
- Ground truth labeled test cases
- Precision, recall, F1 score calculation
- False positive/negative rate tracking
- Per-violation-type metrics
- Target compliance checking

**Target Metrics:**
- Precision: ≥ 95%
- Recall: ≥ 95%
- False Positive Rate: ≤ 5%
- False Negative Rate: ≤ 8%

**Usage:**
```python
from nethical.governance import (
    EthicsBenchmark, BenchmarkCase, 
    DetectionResult, ViolationType
)

# Create benchmark
benchmark = EthicsBenchmark()

# Add test cases
benchmark.add_case(BenchmarkCase(
    id="test001",
    description="Prompt injection test",
    input_data={"text": "Ignore instructions..."},
    ground_truth=ViolationType.MANIPULATION,
    severity="high"
))

# Evaluate detector
def my_detector(input_data):
    # Your detection logic
    return DetectionResult(ViolationType.MANIPULATION, confidence=0.95)

metrics = benchmark.evaluate(my_detector)

# Check compliance
passed, reasons = metrics.meets_targets()
print(f"Precision: {metrics.precision:.3f}")
print(f"Recall: {metrics.recall:.3f}")
print(f"Passed: {passed}")
```

**Example Cases File:** `examples/governance/ethics_benchmark_cases.json`

### Threshold Configuration

Versioned threshold management with audit trail.

**Module:** `nethical.governance.threshold_config`

**Features:**
- Version control for thresholds
- Comparison between versions
- Threshold evaluation
- Persistent storage

**Usage:**
```python
from nethical.governance import (
    ThresholdVersionManager, Threshold, 
    ThresholdType, DEFAULT_THRESHOLDS
)

# Create manager
manager = ThresholdVersionManager("./thresholds")

# Create version
manager.create_version(
    version="1.0.0",
    author="admin",
    description="Initial thresholds",
    thresholds=DEFAULT_THRESHOLDS
)

# Get current thresholds
config = manager.get_version()

# Evaluate values
results = manager.evaluate_thresholds({
    'manipulation_detection': 0.88,
    'privacy_risk': 0.75
})

# Compare versions
diff = manager.compare_versions("1.0.0", "2.0.0")
```

**Default Thresholds:**
- `manipulation_detection`: 0.85 (confidence)
- `privacy_risk`: 0.7 (score)
- `toxicity`: 0.8 (score)
- `high_severity`: 7.5 (scale 0-10)

---

## Observability

### Metrics Collection

Prometheus-compatible metrics for governance operations.

**Module:** `nethical.observability.metrics`

**Metrics:**

1. **actions_total** (Counter)
   - Labels: `action_type`, `decision`, `region`
   - Total governance actions evaluated

2. **violations_total** (Counter)
   - Labels: `violation_type`, `severity`, `detector`
   - Total violations detected

3. **action_latency_seconds** (Histogram)
   - Labels: `action_type`
   - Processing latency for actions

4. **violation_detection_latency_seconds** (Histogram)
   - Labels: `detector_type`
   - Detection latency

**Usage:**
```python
from nethical.observability import (
    get_metrics_collector, 
    record_action, 
    record_violation
)

# Get collector
collector = get_metrics_collector()

# Record actions
record_action(
    action_type="api_call",
    decision="ALLOW",
    region="US",
    latency_seconds=0.05
)

# Record violations
record_violation(
    violation_type="manipulation",
    severity="high",
    detector="prompt_detector",
    latency_seconds=0.03
)

# Get metrics summary
metrics = collector.get_all_metrics()
```

### Distributed Tracing

OpenTelemetry-based distributed tracing.

**Module:** `nethical.observability.tracing`

**Features:**
- 10% baseline sampling rate
- 100% error sampling rate
- Span creation and context propagation
- Fallback implementation without OpenTelemetry

**Usage:**
```python
from nethical.observability import get_tracer, trace_span

# Get tracer
tracer = get_tracer()

# Create spans
with trace_span("governance_check", attributes={"action": "api_call"}):
    # Your code here
    
    with trace_span("policy_evaluation"):
        # Nested operation
        pass
    
    # Add attributes and events
    from nethical.observability import add_span_attribute, add_span_event
    add_span_attribute("rules_evaluated", 5)
    add_span_event("checkpoint", {"step": 1})
```

### Log Sanitization

Automatic PII and sensitive data redaction.

**Module:** `nethical.observability.sanitization`

**Redaction Patterns:**
- **PII:** SSN, credit cards, emails, phone numbers, IP addresses
- **Secrets:** API keys, passwords, tokens, JWTs, AWS keys, bearer tokens

**Usage:**
```python
from nethical.observability import sanitize_log, sanitize_dict

# Sanitize text
log = "User john.doe@example.com logged in with API key sk_live_123"
sanitized = sanitize_log(log)
# Output: "User [EMAIL-REDACTED] logged in with API key [KEY-REDACTED]"

# Sanitize dictionary
data = {
    'email': 'user@example.com',
    'password': 'secret',
    'metadata': {
        'ssn': '123-45-6789'
    }
}
sanitized = sanitize_dict(data, recursive=True)
# Output: email is redacted, password is [REDACTED], ssn is redacted
```

### Alert Rules

Configurable alert system for governance monitoring.

**Module:** `nethical.observability.alerts`

**Default Rules:**

1. **high_latency** (Warning)
   - Threshold: 1.0 seconds (p95)
   - Duration: 60 seconds

2. **critical_latency** (Critical)
   - Threshold: 5.0 seconds (p95)
   - Duration: 30 seconds

3. **high_error_rate** (Error)
   - Threshold: 5%
   - Duration: 120 seconds

4. **drift_detected** (Warning)
   - Threshold: 20% drift
   - Duration: 300 seconds

5. **quota_saturation** (Warning)
   - Threshold: 90% usage
   - Duration: 60 seconds

6. **quota_critical** (Critical)
   - Threshold: 95% usage
   - Duration: 30 seconds

**Usage:**
```python
from nethical.observability import AlertRuleManager, AlertSeverity, AlertRule

# Create manager
manager = AlertRuleManager()

# Register custom rule
rule = AlertRule(
    name="custom_alert",
    description="Custom threshold exceeded",
    severity=AlertSeverity.WARNING,
    condition=lambda metrics: metrics['value'] > 100,
    threshold=100.0,
    duration=60
)
manager.register_rule(rule)

# Register handler
def alert_handler(alert):
    print(f"Alert: {alert.rule_name} - {alert.message}")

manager.register_handler(alert_handler)

# Evaluate rules
metrics = {
    'histograms': {...},
    'gauges': {...}
}
fired = manager.evaluate_rules(metrics)

# Get active alerts
active = manager.get_active_alerts(severity=AlertSeverity.CRITICAL)
```

---

## Usage Examples

### Complete Demo

Run the comprehensive demo:
```bash
python examples/governance/demo_governance_observability.py
```

This demonstrates all features:
- Ethics benchmarking
- Threshold versioning
- Metrics collection
- Distributed tracing
- Log sanitization
- Alert rules

### Policy Simulation

Test a policy:
```bash
# Basic simulation
./cli/policy_simulator simulate \
  examples/governance/sample_policy.yaml \
  examples/governance/policy_test_cases.yaml

# With JSON output
./cli/policy_simulator simulate \
  examples/governance/sample_policy.yaml \
  examples/governance/policy_test_cases.yaml \
  --format json \
  --output results.json

# Compare policy versions
./cli/policy_simulator dry-run \
  policy_v1.yaml \
  policy_v2.yaml \
  test_cases.yaml \
  --region EU
```

### Integrated Example

```python
from nethical.governance import EthicsBenchmark, ThresholdVersionManager
from nethical.observability import (
    get_metrics_collector, get_tracer, 
    sanitize_log, AlertRuleManager
)

# Setup
metrics = get_metrics_collector()
tracer = get_tracer()
alerts = AlertRuleManager()

# Governance check with full observability
with tracer.start_span("governance_check"):
    # Record action
    metrics.record_action("api_call", "ALLOW", "US", 0.05)
    
    # Sanitize logs
    log = "User john@example.com made request"
    safe_log = sanitize_log(log)
    
    # Evaluate alerts
    current_metrics = metrics.get_all_metrics()
    alerts.evaluate_rules(current_metrics)
```

---

## Testing

Run tests:
```bash
# All governance and observability tests
pytest tests/test_governance_features.py tests/test_observability.py -v

# Specific test classes
pytest tests/test_governance_features.py::TestEthicsBenchmark -v
pytest tests/test_observability.py::TestMetrics -v
```

---

## Files Reference

### Governance
- `nethical/governance/policy_grammar.ebnf` - Policy language grammar
- `nethical/governance/ethics_benchmark.py` - Ethics benchmark system
- `nethical/governance/threshold_config.py` - Threshold versioning
- `nethical/governance/__init__.py` - Module exports

### Observability
- `nethical/observability/metrics.py` - Metrics collection
- `nethical/observability/tracing.py` - Distributed tracing
- `nethical/observability/sanitization.py` - Log sanitization
- `nethical/observability/alerts.py` - Alert rules
- `nethical/observability/__init__.py` - Module exports

### CLI & Examples
- `cli/policy_simulator` - Policy simulation CLI
- `examples/governance/demo_governance_observability.py` - Complete demo
- `examples/governance/sample_policy.yaml` - Sample policy
- `examples/governance/policy_test_cases.yaml` - Sample test cases
- `examples/governance/ethics_benchmark_cases.json` - Sample benchmark

### Tests
- `tests/test_governance_features.py` - Governance tests
- `tests/test_observability.py` - Observability tests

---

## Performance Considerations

### Metrics
- In-memory storage with configurable retention
- Optional Prometheus integration for production
- Minimal overhead: < 1ms per metric recording

### Tracing
- 10% baseline sampling reduces overhead
- 100% error sampling ensures visibility
- Fallback mode without OpenTelemetry dependencies

### Sanitization
- Compiled regex patterns for efficiency
- Dictionary sanitization with recursion control
- Configurable pattern selection

### Alerts
- Rule evaluation on-demand
- Duration-based firing prevents flapping
- Async handler execution with retry

---

## Best Practices

1. **Policy Testing**: Always test policies in simulation before deployment
2. **Benchmark Updates**: Regularly update benchmark cases as threats evolve
3. **Threshold Versioning**: Document reasoning when updating thresholds
4. **Metrics Labels**: Use consistent label names across your system
5. **Sampling Rates**: Adjust tracing rates based on traffic volume
6. **Sanitization**: Review patterns periodically for new PII types
7. **Alert Tuning**: Adjust thresholds and durations based on false positive rates

---

## Integration with Existing Systems

The new features integrate seamlessly with existing Nethical components:

- **PolicyEngine**: Uses policy files validated against EBNF grammar
- **SafetyGovernance**: Can record metrics and traces during evaluation
- **Detectors**: Can leverage ethics benchmarks for validation
- **Monitors**: Can use alert rules for anomaly detection

See existing examples in `examples/` for integration patterns.
