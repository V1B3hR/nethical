# Operational Runbook

## Overview

This runbook provides step-by-step procedures for operating, monitoring, and troubleshooting the Nethical governance platform's Phase 7 operational reliability and observability systems.

## Table of Contents

1. [System Startup and Shutdown](#system-startup-and-shutdown)
2. [Probe Management](#probe-management)
3. [Dashboard Operations](#dashboard-operations)
4. [Alert Response](#alert-response)
5. [Incident Response](#incident-response)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance Procedures](#maintenance-procedures)
8. [Emergency Procedures](#emergency-procedures)

---

## System Startup and Shutdown

### Starting Runtime Probes

**Procedure**:

1. **Verify Prerequisites**:
   ```bash
   # Check Python version (3.8+)
   python --version
   
   # Verify dependencies installed
   pip list | grep nethical
   ```

2. **Start Probe Service**:
   ```python
   from probes import (
       DeterminismProbe,
       TerminationProbe,
       AcyclicityProbe,
       AuditCompletenessProbe,
       NonRepudiationProbe,
   )
   from probes import AnomalyDetector, AlertSystem
   
   # Initialize probes
   probes = {
       'determinism': DeterminismProbe(eval_service),
       'termination': TerminationProbe(),
       'acyclicity': AcyclicityProbe(policy_graph),
       'audit': AuditCompletenessProbe(audit_service),
       'non_repudiation': NonRepudiationProbe(audit_service),
   }
   
   # Initialize monitoring
   anomaly_detector = AnomalyDetector()
   alert_system = AlertSystem()
   
   # Start probe loop
   for name, probe in probes.items():
       result = probe.run()
       anomaly = anomaly_detector.analyze(result)
       if anomaly:
           alert_system.create_alert(...)
   ```

3. **Verify Probes Running**:
   ```python
   # Check probe status
   for name, probe in probes.items():
       metrics = probe.get_metrics()
       print(f"{name}: {metrics['last_check_time']}")
   ```

**Expected Output**:
```
determinism: 2025-11-17T05:12:00Z
termination: 2025-11-17T05:12:00Z
acyclicity: 2025-11-17T05:12:00Z
audit: 2025-11-17T05:12:00Z
non_repudiation: 2025-11-17T05:12:00Z
```

### Starting Dashboard Service

**Procedure**:

1. **Initialize Dashboard**:
   ```python
   from dashboards import GovernanceDashboard
   
   dashboard = GovernanceDashboard(
       config_path="dashboards/governance.json",
       cache_ttl_seconds=60
   )
   ```

2. **Start API Server** (if using REST API):
   ```python
   from flask import Flask, jsonify
   
   app = Flask(__name__)
   
   @app.route('/api/dashboard/metrics')
   def get_metrics():
       metrics = dashboard.get_metrics()
       return jsonify(metrics.to_dict())
   
   app.run(host='0.0.0.0', port=8080)
   ```

3. **Verify Dashboard Responding**:
   ```bash
   curl http://localhost:8080/api/dashboard/metrics
   ```

### Graceful Shutdown

**Procedure**:

1. **Stop Accepting New Requests**:
   ```python
   # Set maintenance mode flag
   maintenance_mode = True
   ```

2. **Wait for In-Flight Requests**:
   ```python
   # Allow current requests to complete (max 30s)
   time.sleep(30)
   ```

3. **Flush Metrics**:
   ```python
   # Export final metrics
   dashboard.export_metrics(format="json")
   
   # Save probe state
   for name, probe in probes.items():
       probe_state = probe.get_metrics()
       save_state(name, probe_state)
   ```

4. **Shutdown Services**:
   ```python
   # Close connections
   dashboard = None
   probes.clear()
   ```

---

## Probe Management

### Adding a New Test Case (Determinism Probe)

**Procedure**:

1. **Define Test Case**:
   ```python
   test_case = {
       "policy_id": "pol_new_001",
       "context": {
           "action": "write",
           "resource": "sensitive_data",
           "user_role": "admin"
       }
   }
   ```

2. **Add to Probe**:
   ```python
   determinism_probe.add_test_case(
       policy_id=test_case["policy_id"],
       context=test_case["context"]
   )
   ```

3. **Verify Addition**:
   ```python
   result = determinism_probe.run()
   print(f"Test cases: {result.metrics['test_cases_checked']}")
   ```

### Updating Policy Graph (Acyclicity Probe)

**Procedure**:

1. **Build New Graph**:
   ```python
   new_policy_graph = {
       "pol_001": ["pol_002"],
       "pol_002": ["pol_003"],
       "pol_003": [],
       "pol_004": ["pol_002"]
   }
   ```

2. **Update Probe**:
   ```python
   acyclicity_probe.update_graph(new_policy_graph)
   ```

3. **Immediate Check**:
   ```python
   result = acyclicity_probe.run()
   if result.status == ProbeStatus.CRITICAL:
       print(f"Cycles detected: {result.violations}")
   ```

### Recording Evaluation Metrics

**Termination Probe**:
```python
# Record each policy evaluation
termination_probe.record_evaluation(
    policy_id="pol_001",
    duration_ms=125.5,
    completed=True
)
```

**Data Minimization Probe**:
```python
# Record field accesses
data_min_probe.record_access(
    fields_accessed={"action_type", "resource_type", "agent_id"},
    context="policy_evaluation"
)
```

**Tenant Isolation Probe**:
```python
# Record tenant accesses
tenant_iso_probe.record_access(
    tenant_id="tenant_123",
    resource_tenant_id="tenant_123",
    access_type="read"
)
```

### Adjusting Probe Thresholds

**Procedure**:

1. **Review Current Thresholds**:
   ```python
   print(f"Latency P95 target: {latency_probe.p95_target_ms}ms")
   print(f"CPU threshold: {resource_probe.cpu_threshold}%")
   ```

2. **Update Threshold**:
   ```python
   # Adjust based on observed behavior
   latency_probe.p95_target_ms = 150.0  # Relax from 100ms to 150ms
   resource_probe.cpu_threshold = 85.0   # Tighten from 80% to 85%
   ```

3. **Document Change**:
   ```python
   # Log threshold change with rationale
   log_change("latency_probe.p95_target_ms", 100, 150,
               "Increased due to new feature complexity")
   ```

---

## Dashboard Operations

### Querying Dashboard Metrics

**Get All Metrics**:
```python
metrics = dashboard.get_metrics()
print(f"Timestamp: {metrics.timestamp}")
print(f"Fairness: {metrics.fairness['summary']['overall_status']}")
print(f"Query latency: {metrics.slo_compliance['query_latency_seconds']}s")
```

**Get Specific Sections**:
```python
# Only fairness and appeals
metrics = dashboard.get_metrics(sections=["fairness", "appeals"])

# Only invariant violations
metrics = dashboard.get_metrics(sections=["invariant_violations"])
```

### Recording Dashboard Data

**Record Fairness Decisions**:
```python
# For each decision
dashboard.fairness_collector.record_decision(
    decision="allow",  # or "deny"
    protected_group="female",  # or None
    context={"age": 30}
)
```

**Record Policy Version**:
```python
dashboard.lineage_tracker.record_policy_version(
    policy_id="pol_001",
    version=3,
    content="policy content here",
    parent_hash="previous_content_hash",
    signatures=[
        {"signer_id": "admin1", "signature": "sig1"},
        {"signer_id": "admin2", "signature": "sig2"}
    ],
    author="admin1"
)
```

**Record Appeal**:
```python
# File appeal
dashboard.appeals_collector.record_appeal(
    appeal_id="app_001",
    decision_id="dec_123"
)

# Later, resolve appeal
dashboard.appeals_collector.resolve_appeal(
    appeal_id="app_001",
    outcome="overturned"  # or "upheld", "modified", "withdrawn"
)
```

### Exporting Dashboard Data

**JSON Export**:
```python
json_data = dashboard.export_metrics(format="json")
with open("metrics_export.json", "w") as f:
    f.write(json_data)
```

**CSV Export**:
```python
csv_data = dashboard.export_metrics(format="csv")
with open("metrics_export.csv", "w") as f:
    f.write(csv_data)
```

**Filtered Export**:
```python
# Export only specific sections
json_data = dashboard.export_metrics(
    format="json",
    sections=["fairness", "audit_log"]
)
```

### Cache Management

**Clear Cache**:
```python
# Force fresh computation
metrics = dashboard.get_metrics(use_cache=False)
```

**Check Cache Status**:
```python
# View cache timestamps
for key, timestamp in dashboard._cache_timestamps.items():
    age = (datetime.utcnow() - timestamp).total_seconds()
    print(f"{key}: {age}s old")
```

---

## Alert Response

### Checking Active Alerts

**Procedure**:

1. **List All Active Alerts**:
   ```python
   active_alerts = alert_system.get_active_alerts()
   for alert in active_alerts:
       print(f"{alert.alert_id}: {alert.severity.value} - {alert.message}")
   ```

2. **Filter by Severity**:
   ```python
   critical_alerts = alert_system.get_active_alerts(
       severity=AlertSeverity.CRITICAL
   )
   ```

### Acknowledging Alerts

**Procedure**:

1. **Acknowledge Alert**:
   ```python
   success = alert_system.acknowledge_alert("ALT-000001")
   if success:
       print("Alert acknowledged")
   ```

2. **Verify Acknowledgment**:
   ```python
   active = alert_system.get_active_alerts()
   unacked = [a for a in active if not a.acknowledged]
   print(f"Unacknowledged alerts: {len(unacked)}")
   ```

### Resolving Alerts

**Procedure**:

1. **Resolve Issue** (perform fix)

2. **Resolve Alert**:
   ```python
   success = alert_system.resolve_alert("ALT-000001")
   if success:
       print("Alert resolved")
   ```

3. **Verify Resolution**:
   ```python
   active = alert_system.get_active_alerts()
   # Alert should no longer appear
   ```

### Alert Escalation

**Check for Escalation**:
```python
to_escalate = alert_system.check_escalation()
for alert in to_escalate:
    print(f"Escalating {alert.alert_id} - unacknowledged for >1h")
    # Notify next tier
    notify_escalation(alert)
```

**Automatic Escalation** (in monitoring loop):
```python
import time

while True:
    # Check for escalation every 5 minutes
    to_escalate = alert_system.check_escalation()
    for alert in to_escalate:
        escalate_alert(alert)
    
    time.sleep(300)  # 5 minutes
```

---

## Incident Response

### P0: Critical Incident

**Indicators**:
- Security breach detected
- Complete system outage
- Data loss or corruption
- Determinism violations
- Tenant isolation breach

**Response Procedure**:

1. **Immediate Actions** (0-5 minutes):
   ```bash
   # Page on-call engineer
   pagerduty trigger --severity critical --summary "P0: [Description]"
   
   # Create incident
   incident_id=$(create_incident "P0: [Description]")
   
   # Start incident channel
   slack incident start $incident_id
   ```

2. **Assessment** (5-15 minutes):
   - Determine scope and impact
   - Identify affected customers
   - Gather initial data
   
   ```python
   # Get system status
   metrics = dashboard.get_metrics()
   probe_status = {name: probe.get_metrics() 
                   for name, probe in probes.items()}
   
   # Check violations
   violations = []
   for result in probe_status.values():
       if result.get('critical_count', 0) > 0:
           violations.append(result)
   ```

3. **Containment** (15-30 minutes):
   - Stop further damage
   - Isolate affected components
   - Preserve evidence
   
   ```python
   # Enter maintenance mode
   maintenance_mode = True
   
   # Capture state
   export_all_metrics()
   export_all_logs()
   ```

4. **Resolution** (Target: <4 hours):
   - Implement fix
   - Verify resolution
   - Monitor for regression

5. **Post-Incident**:
   - Conduct root cause analysis
   - Update runbook
   - Implement preventive measures

### P1: High Severity Incident

**Indicators**:
- Major feature degradation
- Multiple probe failures
- SLO breach
- High alert volume

**Response Procedure**: Similar to P0 but with extended timeframes (1h response, 8h resolution)

---

## Troubleshooting

### Probe Not Reporting

**Symptoms**:
- No recent check timestamp
- Probe shows UNKNOWN status
- Missing metrics

**Diagnosis**:
```python
# Check probe status
metrics = probe.get_metrics()
print(f"Last check: {metrics.get('last_check_time')}")
print(f"Total checks: {metrics.get('total_checks')}")
print(f"Errors: {metrics.get('critical_count')}")

# Check recent results
history = probe.get_history(limit=10)
for result in history:
    print(f"{result.timestamp}: {result.status} - {result.message}")
```

**Resolution**:
1. Verify probe dependencies
2. Check error logs
3. Restart probe if necessary
4. Review configuration

### Dashboard Query Timeout

**Symptoms**:
- Queries exceed 5s SLO
- Slow API responses
- Cache misses

**Diagnosis**:
```python
# Check query performance
import time

start = time.time()
metrics = dashboard.get_metrics(use_cache=False)
elapsed = time.time() - start
print(f"Query time: {elapsed:.2f}s")

# Check cache hit rate
cache_age = {k: (datetime.utcnow() - v).total_seconds() 
             for k, v in dashboard._cache_timestamps.items()}
print(f"Cache ages: {cache_age}")
```

**Resolution**:
1. Enable caching (if disabled)
2. Reduce cache TTL if data is stale
3. Query specific sections only
4. Optimize slow collectors

### Fairness Metric Anomalies

**Symptoms**:
- Unexpected fairness violations
- Statistical Parity out of bounds
- Disparate Impact ratio abnormal

**Diagnosis**:
```python
# Get detailed fairness metrics
fairness = dashboard.fairness_collector

sp = fairness.get_statistical_parity()
di = fairness.get_disparate_impact()
eo = fairness.get_equal_opportunity()

print(f"SP Difference: {sp['difference']} (threshold: 0.10)")
print(f"Protected rate: {sp['protected_rate']}")
print(f"Unprotected rate: {sp['unprotected_rate']}")
print(f"Sample size: {sp['sample_size']}")

print(f"\nDI Ratio: {di['ratio']} (threshold: 0.80-1.25)")
print(f"Protected count: {di['protected_count']}")
print(f"Unprotected count: {di['unprotected_count']}")
```

**Resolution**:
1. Verify sufficient sample size
2. Check protected group labels
3. Review recent policy changes
4. Investigate biased decisions
5. Consider bias mitigation

### Alert Storm

**Symptoms**:
- High volume of alerts
- Multiple probe failures
- System instability

**Diagnosis**:
```python
# Get alert metrics
alert_metrics = alert_system.get_metrics()
print(f"Active alerts: {alert_metrics['active_alerts']}")
print(f"Critical: {alert_metrics['critical_alerts']}")
print(f"Unacknowledged: {alert_metrics['unacknowledged_alerts']}")

# Check alert patterns
alerts = alert_system.get_active_alerts()
by_probe = {}
for alert in alerts:
    probe = alert.probe_name
    by_probe[probe] = by_probe.get(probe, 0) + 1

print(f"Alerts by probe: {by_probe}")
```

**Resolution**:
1. Identify root cause (single probe vs. systemic)
2. Acknowledge alerts
3. Fix underlying issue
4. Adjust alert thresholds if needed
5. Review false positive rate

---

## Maintenance Procedures

### Scheduled Maintenance

**Pre-Maintenance**:
1. Announce maintenance window (7 days notice)
2. Notify stakeholders
3. Backup current state
4. Verify rollback procedure

**During Maintenance**:
1. Enable maintenance mode
2. Stop probe collection
3. Perform maintenance tasks
4. Test changes
5. Resume probe collection
6. Disable maintenance mode

**Post-Maintenance**:
1. Verify all probes reporting
2. Check dashboard functionality
3. Monitor for anomalies
4. Document changes

### Probe Configuration Updates

**Procedure**:
1. Review change in non-production
2. Update configuration file
3. Apply via CI/CD pipeline
4. Verify change applied
5. Monitor for issues

### Dashboard Configuration Updates

**Procedure**:
1. Edit `dashboards/governance.json`
2. Validate JSON schema
3. Deploy configuration
4. Reload dashboard
5. Verify changes

---

## Emergency Procedures

### Emergency Probe Shutdown

**When**: Probe causing system instability

**Procedure**:
```python
# Stop specific probe
probe.reset()
probe = None

# Or stop all probes
for probe in probes.values():
    probe.reset()
probes.clear()
```

### Emergency Threshold Adjustment

**When**: Flood of false positive alerts

**Procedure**:
```python
# Temporarily relax thresholds
latency_probe.p95_target_ms *= 2  # Double threshold
resource_probe.cpu_threshold += 10  # Add 10%

# Document emergency change
log_emergency_change("Relaxed thresholds due to alert storm")
```

### Data Export for Forensics

**When**: Security incident or audit

**Procedure**:
```python
# Export all current state
timestamp = datetime.utcnow().isoformat()

# Export probe states
probe_states = {name: probe.get_history(limit=1000)
                for name, probe in probes.items()}

# Export dashboard data
dashboard_export = dashboard.export_metrics(format="json")

# Export alerts
alert_export = [alert.to_dict() 
                for alert in alert_system._alerts]

# Save forensics bundle
forensics = {
    "timestamp": timestamp,
    "probes": probe_states,
    "dashboard": dashboard_export,
    "alerts": alert_export
}

with open(f"forensics_{timestamp}.json", "w") as f:
    json.dump(forensics, f, indent=2)
```

---

## References

- [Runtime Probes Specification](./runtime_probes.md)
- [Governance Dashboard Guide](./governance_dashboard.md)
- [SLO Definitions](./slo_definitions.md)
- [Incident Response Plan](../../security/incident_response.md)
- [On-Call Procedures](./oncall.md)
