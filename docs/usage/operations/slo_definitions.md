# SLO/SLA Definitions

## Overview

This document defines Service Level Objectives (SLOs) and Service Level Agreements (SLAs) for the Nethical governance platform, focusing on operational reliability and observability metrics.

## Service Level Objectives (SLOs)

### 1. Governance Dashboard Query Latency

**Objective**: Dashboard queries return results within acceptable time limits

**Metrics**:
- **Target**: P95 < 5 seconds
- **Measurement**: Time from query initiation to response delivery
- **Evaluation Period**: 5-minute rolling window

**Implementation**:
```python
# Measured in dashboard.py
elapsed = (datetime.utcnow() - start_time).total_seconds()
slo_met = elapsed < 5.0
```

**Rationale**: Interactive dashboards require responsive queries to be effective for operators and stakeholders.

**Error Budget**: 5% of queries may exceed 5s (allowing 5% of requests to be slower)

**Consequences of Breach**:
- **Minor** (5-10% over target): Investigate performance issues
- **Moderate** (10-20% over target): Implement caching improvements
- **Severe** (>20% over target): Emergency optimization required

---

### 2. Runtime Probe Availability

**Objective**: Runtime probes are operational and collecting data

**Metrics**:
- **Target**: ≥99.9% availability
- **Measurement**: Percentage of successful probe executions
- **Evaluation Period**: 30-day rolling window

**Calculation**:
```
Availability = (Successful Checks / Total Expected Checks) × 100%
```

**Allowed Downtime**:
- Per month: 43.2 minutes
- Per week: 10.1 minutes
- Per day: 1.4 minutes

**Implementation**:
```python
# Track probe execution success
total_checks = probe.get_metrics()['total_checks']
healthy_checks = probe.get_metrics()['healthy_count']
availability = (healthy_checks / total_checks) * 100
```

**Consequences of Breach**:
- **Minor** (99.5-99.9%): Monitor for patterns
- **Moderate** (99.0-99.5%): Review probe reliability
- **Severe** (<99.0%): Immediate investigation required

---

### 3. Alert False Positive Rate

**Objective**: Minimize operator fatigue from false alarms

**Metrics**:
- **Target**: <5% false positive rate
- **Measurement**: Percentage of alerts marked as false positives
- **Evaluation Period**: 7-day rolling window

**Calculation**:
```
FP_Rate = (False Positive Alerts / Total Alerts) × 100%
```

**Implementation**:
```python
# From alert_system.py
alert_metrics = alert_system.get_metrics()
fp_rate = (false_positives / total_alerts) * 100
```

**Thresholds**:
- **Healthy**: <5% FP rate
- **Warning**: 5-10% FP rate
- **Critical**: >10% FP rate

**Consequences of Breach**:
- **Minor** (5-7%): Review alert thresholds
- **Moderate** (7-10%): Tune anomaly detection
- **Severe** (>10%): Major alert configuration overhaul

---

### 4. Audit Log Completeness

**Objective**: All decisions are fully audited

**Metrics**:
- **Target**: 100% completeness
- **Measurement**: Percentage of decisions with complete audit entries
- **Evaluation Period**: 1-day rolling window

**Calculation**:
```
Completeness = (Audited Decisions / Total Decisions) × 100%
```

**Implementation**:
```python
# From AuditCompletenessProbe
completeness_rate = audited_decisions / total_decisions
```

**Tolerance**: Zero tolerance for incomplete auditing

**Consequences of Breach**:
- **Any breach**: Critical incident, immediate investigation
- Potential compliance violation
- Requires root cause analysis

---

### 5. Policy Evaluation Termination

**Objective**: All policy evaluations complete within bounded time

**Metrics**:
- **Target**: 100% of evaluations complete within 5 seconds
- **Measurement**: Percentage of evaluations within timeout
- **Evaluation Period**: 1-hour rolling window

**Calculation**:
```
Termination_Rate = (Completed_Within_Timeout / Total_Evaluations) × 100%
```

**Implementation**:
```python
# From TerminationProbe
termination_rate = (total - incomplete) / total
```

**Thresholds**:
- **Healthy**: 100% within timeout
- **Warning**: 95-100% within timeout
- **Critical**: <95% within timeout

**Consequences of Breach**:
- **Minor** (95-99%): Monitor for trends
- **Moderate** (90-95%): Optimize slow policies
- **Severe** (<90%): Emergency policy review

---

### 6. Determinism Compliance

**Objective**: All evaluations are deterministic (reproducible)

**Metrics**:
- **Target**: 100% deterministic
- **Measurement**: Percentage of evaluations producing identical results
- **Evaluation Period**: Continuous

**Calculation**:
```
Determinism_Rate = (Deterministic_Evaluations / Total_Tests) × 100%
```

**Implementation**:
```python
# From DeterminismProbe
determinism_rate = (total - violations) / total
```

**Tolerance**: Zero tolerance for non-determinism

**Consequences of Breach**:
- **Any breach**: Critical incident
- Appeals process may be compromised
- Requires immediate investigation

---

### 7. Multi-Signature Compliance

**Objective**: All policy changes properly authorized

**Metrics**:
- **Target**: 100% compliant
- **Measurement**: Percentage of changes with required signatures
- **Evaluation Period**: Continuous

**Calculation**:
```
Compliance_Rate = (Properly_Signed / Total_Changes) × 100%
```

**Implementation**:
```python
# From MultiSigProbe
compliance_rate = properly_signed / total_changes
```

**Tolerance**: Zero tolerance for unauthorized changes

**Consequences of Breach**:
- **Any breach**: Security incident
- Governance violation
- Requires audit and remediation

---

### 8. Tenant Isolation Integrity

**Objective**: Prevent cross-tenant data leakage

**Metrics**:
- **Target**: 100% isolation
- **Measurement**: Percentage of accesses respecting tenant boundaries
- **Evaluation Period**: Continuous

**Calculation**:
```
Isolation_Rate = (Isolated_Accesses / Total_Accesses) × 100%
```

**Implementation**:
```python
# From TenantIsolationProbe
isolation_rate = 1 - (cross_tenant_count / total_accesses)
```

**Tolerance**: <1% cross-tenant access (emergency scenarios only)

**Consequences of Breach**:
- **Minor** (<1%): Investigate legitimate exceptions
- **Moderate** (1-5%): Security review required
- **Severe** (>5%): Critical security incident

---

### 9. Appeals Resolution Time

**Objective**: Timely resolution of appeals

**Metrics**:
- **Target**: Median < 72 hours
- **Measurement**: Time from filing to resolution
- **Evaluation Period**: 30-day rolling window

**Percentiles**:
- **P50 (Median)**: < 72 hours
- **P95**: < 7 days
- **P99**: < 14 days

**Implementation**:
```python
# From AppealsMetricsCollector
resolution_metrics = appeals_collector.get_resolution_metrics()
median_hours = resolution_metrics['median_hours']
```

**Consequences of Breach**:
- **Minor** (72-96h median): Review staffing
- **Moderate** (96-120h median): Process improvements needed
- **Severe** (>120h median): Escalation to management

---

### 10. Fairness Metric Stability

**Objective**: Maintain fair decision-making across groups

**Metrics**:
- **Statistical Parity**: |difference| ≤ 0.10
- **Disparate Impact**: 0.80 ≤ ratio ≤ 1.25
- **Equal Opportunity**: |difference| ≤ 0.10
- **Evaluation Period**: 24-hour rolling window

**Implementation**:
```python
# From FairnessMetricsCollector
sp = fairness_collector.get_statistical_parity()
di = fairness_collector.get_disparate_impact()
```

**Consequences of Breach**:
- **Minor** (Warning thresholds): Monitor and review
- **Moderate** (Sustained warnings): Bias mitigation required
- **Severe** (Critical thresholds): Immediate policy review

---

## Service Level Agreements (SLAs)

### External SLAs (Customer-Facing)

#### 1. System Availability SLA

**Commitment**: 99.9% uptime (excluding planned maintenance)

**Measurement**: 
- Monthly uptime percentage
- Excludes scheduled maintenance windows
- Includes all core governance functions

**Downtime Credits**:
- 99.0-99.9%: 10% service credit
- 95.0-99.0%: 25% service credit
- <95.0%: 50% service credit

**Exclusions**:
- Scheduled maintenance (with 7-day notice)
- Customer-caused incidents
- Force majeure events

---

#### 2. Dashboard Response Time SLA

**Commitment**: 95% of queries return within 5 seconds

**Measurement**: P95 latency over monthly period

**Consequences**:
- Breach: 5% service credit
- Two consecutive months: 10% service credit
- Three consecutive months: Contract review

---

#### 3. Audit Data Retention SLA

**Commitment**: Maintain audit logs for minimum 7 years (2555 days)

**Measurement**: Oldest audit entry age

**Consequences**:
- Data loss: Critical incident, regulatory reporting
- Service credit: 100% for affected period
- Potential regulatory penalties

---

### Internal SLAs (Operational)

#### 1. Incident Response SLA

**Severity P0 (Critical)**:
- **Response Time**: 15 minutes
- **Resolution Time**: 4 hours
- **Examples**: Security breach, data loss, complete outage

**Severity P1 (High)**:
- **Response Time**: 1 hour
- **Resolution Time**: 8 hours
- **Examples**: Major feature broken, significant performance degradation

**Severity P2 (Medium)**:
- **Response Time**: 4 hours
- **Resolution Time**: 2 business days
- **Examples**: Minor feature broken, moderate performance issues

**Severity P3 (Low)**:
- **Response Time**: 1 business day
- **Resolution Time**: 1 week
- **Examples**: Cosmetic issues, enhancement requests

---

#### 2. Alert Acknowledgment SLA

**Target**: All critical alerts acknowledged within 15 minutes

**Escalation**:
- 15 minutes: Primary on-call
- 30 minutes: Secondary on-call
- 45 minutes: Engineering manager
- 60 minutes: VP Engineering

---

#### 3. Probe Configuration Update SLA

**Target**: Configuration changes applied within 5 minutes

**Measurement**: Time from configuration commit to deployment

**Implementation**: Automated CI/CD pipeline

---

## Monitoring and Reporting

### Real-Time Monitoring

All SLOs are monitored in real-time via:
- Governance Dashboard
- Prometheus metrics
- Grafana dashboards
- Alert system

### Reporting Schedule

**Daily**:
- SLO compliance summary
- Critical violations report
- Performance trends

**Weekly**:
- Detailed SLO analysis
- Trend identification
- Improvement recommendations

**Monthly**:
- SLA compliance report
- Error budget usage
- Customer-facing metrics

### Error Budgets

Each SLO has an associated error budget:

```
Error Budget = (1 - SLO Target) × Total Requests
```

**Example** (Dashboard Latency):
```
SLO Target: 95% < 5s
Error Budget: 5% of queries may exceed 5s

Monthly queries: 1,000,000
Allowed slow queries: 50,000
```

**Error Budget Policy**:
- **>50% remaining**: Continue normal operations
- **25-50% remaining**: Review and optimize
- **<25% remaining**: Freeze non-critical deployments
- **Exhausted**: Emergency response, feature freeze

---

## Compliance and Auditing

### Regulatory Requirements

SLOs support compliance with:
- **GDPR**: Data retention, right to explanation
- **CCPA**: Data access, audit requirements
- **SOC 2**: Availability, security monitoring
- **HIPAA**: Audit logging, integrity controls
- **FedRAMP**: Continuous monitoring, incident response

### Audit Trail

All SLO measurements are:
- Logged with timestamps
- Stored for 7 years
- Available for external audit
- Tamper-evident (Merkle tree)

---

## References

- [Runtime Probes Specification](./runtime_probes.md)
- [Governance Dashboard Guide](./governance_dashboard.md)
- [Operational Runbook](./runbook.md)
- [Incident Response Plan](../../security/incident_response.md)
