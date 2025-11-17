# Runtime Probes Specification

## Overview

Runtime probes provide continuous validation that the Nethical system maintains its formal guarantees in production environments. Each probe mirrors a formal invariant or governance property defined in Phases 3-6.

## Probe Architecture

### Base Probe Infrastructure

All probes inherit from `BaseProbe` which provides:
- Standardized check execution with error handling
- Check history tracking (last 1000 results)
- Consecutive failure tracking for alerting
- Aggregated metrics computation
- Status reporting (HEALTHY, WARNING, CRITICAL, UNKNOWN)

### Probe Types

1. **Invariant Probes**: Monitor formal invariants (P-DET, P-TERM, P-ACYCLIC, P-AUD, P-NONREP)
2. **Governance Probes**: Monitor governance properties (P-MULTI-SIG, P-POL-LIN, P-DATA-MIN, P-TENANT-ISO)
3. **Performance Probes**: Monitor system performance (latency, throughput, resources)

## Invariant Monitoring Probes

### P-DET: Determinism Probe

**Purpose**: Validates that identical inputs produce identical outputs.

**Implementation**: `probes.invariant_probes.DeterminismProbe`

**Check Frequency**: Every 5 minutes (300 seconds)

**Methodology**:
1. Maintains a set of test cases (policy + context pairs)
2. Evaluates each test case twice with identical inputs
3. Compares hashes of results
4. Reports violations if hashes differ

**Metrics**:
- `test_cases_checked`: Number of test cases validated
- `violations_count`: Number of non-deterministic results
- `determinism_rate`: Percentage of deterministic evaluations

**Thresholds**:
- **HEALTHY**: No violations
- **CRITICAL**: Any violations detected

**Usage**:
```python
from probes import DeterminismProbe

probe = DeterminismProbe(
    evaluation_service=eval_service,
    check_interval_seconds=300,
    sample_size=10
)

# Add test cases
probe.add_test_case(
    policy_id="pol_001",
    context={"action": "read", "resource": "document"}
)

# Run check
result = probe.run()
print(f"Status: {result.status}, Violations: {len(result.violations)}")
```

### P-TERM: Termination Probe

**Purpose**: Validates that all policy evaluations complete within bounded time.

**Implementation**: `probes.invariant_probes.TerminationProbe`

**Check Frequency**: Every 1 minute (60 seconds)

**Methodology**:
1. Records evaluation duration for each policy evaluation
2. Checks for timeouts (default: 5000ms max)
3. Identifies incomplete evaluations
4. Tracks average and maximum durations

**Metrics**:
- `evaluations_checked`: Number of evaluations analyzed
- `timeout_violations`: Evaluations exceeding timeout
- `incomplete_evaluations`: Evaluations that didn't complete
- `avg_duration_ms`: Average evaluation time
- `max_duration_ms`: Maximum evaluation time
- `termination_rate`: Percentage of completed evaluations

**Thresholds**:
- **HEALTHY**: All evaluations complete, none exceed timeout
- **WARNING**: Some timeouts but <5% of total
- **CRITICAL**: >5% timeouts or any incomplete evaluations

**Usage**:
```python
from probes import TerminationProbe

probe = TerminationProbe(
    max_evaluation_time_ms=5000,
    check_interval_seconds=60
)

# Record evaluations
probe.record_evaluation(
    policy_id="pol_001",
    duration_ms=234.5,
    completed=True
)

# Run check
result = probe.run()
```

### P-ACYCLIC: Acyclicity Probe

**Purpose**: Validates that policy dependencies form a directed acyclic graph (DAG).

**Implementation**: `probes.invariant_probes.AcyclicityProbe`

**Check Frequency**: Every 5 minutes (300 seconds)

**Methodology**:
1. Maintains policy dependency graph
2. Performs depth-first search to detect cycles
3. Calculates maximum dependency depth
4. Reports any cycles found

**Metrics**:
- `policies_count`: Number of policies in graph
- `cycles_found`: Number of cycles detected
- `max_depth`: Maximum dependency depth

**Thresholds**:
- **HEALTHY**: No cycles, depth within bounds
- **WARNING**: Depth exceeds maximum
- **CRITICAL**: Cycles detected

**Usage**:
```python
from probes import AcyclicityProbe

policy_graph = {
    "pol_001": ["pol_002", "pol_003"],
    "pol_002": [],
    "pol_003": ["pol_004"],
    "pol_004": []
}

probe = AcyclicityProbe(
    policy_graph=policy_graph,
    max_depth=10,
    check_interval_seconds=300
)

result = probe.run()
```

### P-AUD: Audit Completeness Probe

**Purpose**: Validates that all decisions are fully audited.

**Implementation**: `probes.invariant_probes.AuditCompletenessProbe`

**Check Frequency**: Every 1 minute (60 seconds)

**Methodology**:
1. Queries recent audit entries
2. Validates presence of required fields
3. Checks audit log monotonicity (always growing)
4. Identifies incomplete entries

**Required Fields**:
- `timestamp`
- `policy_id`
- `decision`
- `context`
- `agent_id`

**Metrics**:
- `total_audit_entries`: Total audit entries
- `incomplete_entries`: Entries missing required fields
- `completeness_rate`: Percentage of complete entries

**Thresholds**:
- **HEALTHY**: All entries complete, log is monotonic
- **WARNING**: Some incomplete entries
- **CRITICAL**: Log not monotonic (integrity violation)

### P-NONREP: Non-repudiation Probe

**Purpose**: Validates that audit logs are cryptographically signed and tamper-evident.

**Implementation**: `probes.invariant_probes.NonRepudiationProbe`

**Check Frequency**: Every 5 minutes (300 seconds)

**Methodology**:
1. Retrieves current Merkle root
2. Verifies root signature
3. Checks for valid root succession
4. Samples audit entries to verify signatures

**Metrics**:
- `merkle_root`: Current Merkle root (truncated)
- `signature_valid`: Boolean signature validity
- `sample_size`: Number of entries verified
- `invalid_signatures`: Count of invalid signatures

**Thresholds**:
- **HEALTHY**: All signatures valid, no tampering
- **CRITICAL**: Any signature failures or tampering detected

## Governance Property Probes

### P-MULTI-SIG: Multi-Signature Probe

**Purpose**: Validates policy activations require multiple authorized signatures.

**Implementation**: `probes.governance_probes.MultiSigProbe`

**Check Frequency**: Every 5 minutes (300 seconds)

**Methodology**:
1. Retrieves recent policy changes
2. Validates signature count meets minimum
3. Verifies all signers are authorized
4. Reports violations

**Metrics**:
- `changes_checked`: Number of policy changes
- `insufficient_signatures`: Changes with too few signatures
- `unauthorized_signatures`: Signatures from unauthorized parties
- `compliance_rate`: Percentage compliant changes

**Thresholds**:
- **HEALTHY**: All changes properly signed
- **CRITICAL**: Any insufficient or unauthorized signatures

### P-POL-LIN: Policy Lineage Probe

**Purpose**: Validates policy version history forms unbroken hash chain.

**Implementation**: `probes.governance_probes.PolicyLineageProbe`

**Check Frequency**: Every 5 minutes (300 seconds)

**Methodology**:
1. Retrieves all active policies
2. Gets version history for each
3. Verifies hash chain integrity
4. Checks for missing versions

**Metrics**:
- `policies_checked`: Number of policies verified
- `broken_chains`: Policies with broken hash chains
- `missing_versions`: Policies with version gaps
- `integrity_rate`: Percentage with intact lineage

**Thresholds**:
- **HEALTHY**: All lineages intact
- **WARNING**: Missing versions detected
- **CRITICAL**: Broken hash chains

### P-DATA-MIN: Data Minimization Probe

**Purpose**: Validates only required context fields are accessed.

**Implementation**: `probes.governance_probes.DataMinimizationProbe`

**Check Frequency**: Every 1 minute (60 seconds)

**Methodology**:
1. Records field accesses during evaluations
2. Compares against whitelist
3. Identifies unauthorized accesses
4. Tracks violation rate

**Metrics**:
- `accesses_checked`: Number of accesses verified
- `unauthorized_count`: Unauthorized field accesses
- `compliance_rate`: Percentage compliant accesses
- `unauthorized_fields`: List of fields accessed inappropriately

**Thresholds**:
- **HEALTHY**: No violations
- **WARNING**: <5% violations
- **CRITICAL**: >5% violations

### P-TENANT-ISO: Tenant Isolation Probe

**Purpose**: Validates tenant data and decisions are properly isolated.

**Implementation**: `probes.governance_probes.TenantIsolationProbe`

**Check Frequency**: Every 1 minute (60 seconds)

**Methodology**:
1. Records tenant accesses
2. Identifies cross-tenant accesses
3. Tracks affected tenant pairs
4. Reports violations

**Metrics**:
- `accesses_checked`: Number of accesses verified
- `cross_tenant_count`: Cross-tenant accesses
- `isolation_rate`: Percentage isolated accesses
- `affected_tenant_pairs`: Number of tenant pairs involved

**Thresholds**:
- **HEALTHY**: No cross-tenant accesses
- **WARNING**: <1% violations
- **CRITICAL**: >1% violations (stricter than data minimization)

## Performance Probes

### Latency Probe

**Purpose**: Tracks request latency and validates against SLO targets.

**Implementation**: `probes.performance_probes.LatencyProbe`

**Check Frequency**: Every 1 minute (60 seconds)

**SLO Targets**:
- P95: 100ms
- P99: 500ms

**Metrics**:
- `p50_ms`, `p95_ms`, `p99_ms`: Latency percentiles
- `avg_ms`: Average latency
- `max_ms`: Maximum latency
- `slo_compliance`: Boolean SLO compliance

### Throughput Probe

**Purpose**: Tracks request throughput and validates capacity.

**Implementation**: `probes.performance_probes.ThroughputProbe`

**Check Frequency**: Every 1 minute (60 seconds)

**Target**: 1000 RPS

**Metrics**:
- `current_rps`: Current requests per second
- `target_rps`: Target capacity
- `capacity_utilization_percent`: Utilization percentage

### Resource Utilization Probe

**Purpose**: Tracks system resource usage.

**Implementation**: `probes.performance_probes.ResourceUtilizationProbe`

**Check Frequency**: Every 1 minute (60 seconds)

**Thresholds**:
- CPU: 80%
- Memory: 85%
- Disk: 90%

**Metrics**:
- `cpu_percent`: CPU utilization
- `memory_percent`: Memory utilization
- `disk_percent`: Disk utilization
- Various detailed resource metrics

## Anomaly Detection

**Implementation**: `probes.anomaly_detector.AnomalyDetector`

The anomaly detector analyzes probe results to identify:
- Statistical anomalies (z-score based)
- Trends (increasing/decreasing patterns)
- Correlations across probes

**Configuration**:
- `sensitivity`: Standard deviations for threshold (default: 2.0)
- `lookback_window`: Historical samples to analyze (default: 100)

## Alert System

**Implementation**: `probes.anomaly_detector.AlertSystem`

The alert system manages:
- Alert creation and deduplication
- Severity routing (INFO, WARNING, CRITICAL)
- Escalation policies (default: 1 hour)
- Alert acknowledgment and resolution

**Features**:
- Automatic deduplication of similar alerts
- Configurable alert handlers per severity
- Escalation for unacknowledged alerts
- Alert history tracking

## Integration

### With Monitoring Systems

Probes can export metrics to:
- **Prometheus**: Via custom exporters
- **Grafana**: For visualization
- **CloudWatch**: For AWS deployments
- **Datadog**: For comprehensive monitoring

### With Dashboard

Probes feed real-time data to the governance dashboard:
```python
from probes import DeterminismProbe, TerminationProbe
from dashboards import GovernanceDashboard

# Create probes
det_probe = DeterminismProbe(eval_service)
term_probe = TerminationProbe()

# Create dashboard
dashboard = GovernanceDashboard()

# Run probes and update dashboard
det_result = det_probe.run()
dashboard.update_probe_result("P-DET-Determinism", det_result)

term_result = term_probe.run()
dashboard.update_probe_result("P-TERM-Termination", term_result)

# Get dashboard metrics
metrics = dashboard.get_metrics(sections=["invariant_violations"])
```

## Deployment

### Production Deployment

1. Deploy probes as separate services/processes
2. Configure check intervals based on criticality
3. Set up alerting and escalation
4. Integrate with existing monitoring
5. Enable metric exporters

### Example Deployment (Kubernetes)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nethical-probes
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: probes
        image: nethical/probes:latest
        env:
        - name: CHECK_INTERVAL
          value: "60"
        - name: ALERT_THRESHOLD
          value: "3"
```

## Maintenance

### Health Checks

Probes themselves should be monitored:
- Check probe availability
- Monitor probe execution time
- Track probe failure rates

### Configuration Updates

Probe configuration can be updated without restart:
- Adjust check intervals
- Modify thresholds
- Update test cases
- Add/remove probes

## References

- [Governance Dashboard Guide](./governance_dashboard.md)
- [SLO Definitions](./slo_definitions.md)
- [Operational Runbook](./runbook.md)
- Phase 3 Formal Specifications: `/formal/phase3/`
- Phase 4 Governance Invariants: `/formal/phase4/`
