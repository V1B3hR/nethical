# Governance Metrics Dashboard Configuration Guide

## Overview

The Nethical Governance Dashboard provides real-time visibility into system governance, fairness, policy lineage, appeals processing, audit compliance, and runtime invariant violations.

## Architecture

### Components

1. **Dashboard Core** (`dashboards.dashboard.GovernanceDashboard`)
   - Aggregates metrics from all subsystems
   - Provides caching for <5s latency SLO
   - Exports metrics in multiple formats

2. **Fairness Metrics Collector** (`dashboards.fairness_metrics.FairnessMetricsCollector`)
   - Computes Statistical Parity, Disparate Impact, Equal Opportunity
   - Tracks metrics by protected attribute
   - Validates against thresholds

3. **Policy Lineage Tracker** (`dashboards.policy_lineage_tracker.PolicyLineageTracker`)
   - Monitors hash chain integrity
   - Tracks version history
   - Validates multi-signature compliance

4. **Appeals Metrics Collector** (`dashboards.appeals_metrics.AppealsMetricsCollector`)
   - Tracks appeal volume and resolution times
   - Monitors SLO compliance (72-hour target)
   - Analyzes outcome distribution

## Configuration

### Dashboard Configuration File

The dashboard is configured via `dashboards/governance.json`:

```json
{
  "metrics": {
    "fairness": {
      "statistical_parity": {
        "threshold": 0.10,
        "visualization": {
          "type": "line_chart",
          "refresh_interval_seconds": 300
        }
      }
    }
  },
  "slo_definitions": {
    "governance_dashboard_latency": {
      "target": "< 5s",
      "target_ms": 5000
    }
  }
}
```

### Key Configuration Sections

#### 1. Fairness Metrics

Configures fairness monitoring for protected attributes:

- **Statistical Parity**: Threshold 0.10 (10% difference)
- **Disparate Impact**: Ratio 0.80-1.25
- **Equal Opportunity**: Threshold 0.10
- **Protected Attributes**: age, gender, race, ethnicity, disability, national_origin, religion

#### 2. Policy Lineage

Configures policy version tracking:

- **Chain Integrity**: 100% target
- **Multi-Sig Compliance**: 100% target (minimum 2 signatures)
- **Version Tracking**: All changes logged

#### 3. Appeals Processing

Configures appeals monitoring:

- **Resolution Time SLO**: 72 hours (median)
- **Volume Tracking**: Daily rates
- **Outcome Distribution**: upheld, overturned, modified, withdrawn

#### 4. Audit Log

Configures audit compliance:

- **Completeness**: 100% target
- **Integrity**: Merkle tree validation
- **Retention**: 2555 days (7 years)

#### 5. Invariant Violations

Configures real-time violation tracking for:
- P-DET (Determinism)
- P-TERM (Termination)
- P-ACYCLIC (Acyclicity)
- P-AUD (Audit Completeness)
- P-NONREP (Non-repudiation)

#### 6. SLO Definitions

Defines service level objectives:

- **Dashboard Latency**: <5s (P95)
- **Probe Availability**: ≥99.9%
- **Alert False Positive Rate**: <5%
- **Audit Completeness**: 100%

## Dashboard Layout

The dashboard is organized into 6 sections:

### 1. Overview Section

**Purpose**: High-level system health

**Panels**:
- System health indicator
- Active alerts summary
- SLO compliance status

### 2. Fairness Section

**Purpose**: Monitor bias and discrimination metrics

**Panels**:
- Statistical Parity trend line
- Disparate Impact gauge
- Equal Opportunity bar chart
- Protected attribute analysis heatmap

**Key Metrics**:
- Difference in approval rates by group
- Ratio of approval rates
- True positive rate differences
- Status: healthy, warning, critical

### 3. Policy Lineage Section

**Purpose**: Track policy evolution and integrity

**Panels**:
- Chain integrity status indicator
- Version history timeline
- Multi-signature compliance pie chart
- Recent policy changes list

**Key Metrics**:
- Total policies and versions
- Verified vs broken chains
- Properly signed changes
- Recent changes (24h)

### 4. Appeals Section

**Purpose**: Monitor appeals processing performance

**Panels**:
- Appeal volume line chart
- Resolution time histogram
- Outcome distribution pie chart
- Pending appeals list

**Key Metrics**:
- Total, pending, resolved appeals
- Median, P95, P99 resolution times
- SLO compliance rate
- Outcome percentages

### 5. Audit & Compliance Section

**Purpose**: Validate audit log integrity

**Panels**:
- Completeness gauge
- Integrity status indicator
- Retention metrics line chart
- Recent audit entries table

**Key Metrics**:
- Completeness rate (target: 100%)
- Merkle root validity
- Signature validity
- Storage metrics

### 6. Runtime Invariants Section

**Purpose**: Real-time invariant violation tracking

**Panels**:
- Determinism violations alert stream
- Termination violations alert stream
- Acyclicity violations alert stream
- Audit violations alert stream
- Non-repudiation violations alert stream

**Key Metrics**:
- Violation counts by type
- Recent violations
- Probe status (healthy/warning/critical)

## Usage

### Basic Usage

```python
from dashboards import GovernanceDashboard

# Initialize dashboard
dashboard = GovernanceDashboard(
    config_path="dashboards/governance.json",
    cache_ttl_seconds=60
)

# Get all metrics
metrics = dashboard.get_metrics()

print(f"Fairness Status: {metrics.fairness['summary']['overall_status']}")
print(f"Appeals Pending: {metrics.appeals['volume']['pending_appeals']}")
print(f"SLO Compliance: {metrics.slo_compliance}")
```

### Query Specific Sections

```python
# Get only fairness metrics
metrics = dashboard.get_metrics(sections=["fairness"])

# Get fairness and appeals
metrics = dashboard.get_metrics(sections=["fairness", "appeals"])
```

### Disable Caching

```python
# Force fresh computation (slower but current)
metrics = dashboard.get_metrics(use_cache=False)
```

### Recording Data

#### Record Fairness Decisions

```python
# Record decisions for fairness analysis
dashboard.fairness_collector.record_decision(
    decision="allow",
    protected_group="female",  # or None for unprotected
    context={"age": 30, "location": "US"}
)
```

#### Record Policy Changes

```python
# Record policy version
dashboard.lineage_tracker.record_policy_version(
    policy_id="pol_001",
    version=2,
    content="<policy content>",
    parent_hash="abc123...",
    signatures=[
        {"signer_id": "user1", "signature": "sig1"},
        {"signer_id": "user2", "signature": "sig2"}
    ],
    author="admin"
)
```

#### Record Appeals

```python
# File appeal
dashboard.appeals_collector.record_appeal(
    appeal_id="app_001",
    decision_id="dec_123"
)

# Resolve appeal
dashboard.appeals_collector.resolve_appeal(
    appeal_id="app_001",
    outcome="overturned"
)
```

### Exporting Metrics

```python
# Export as JSON
json_data = dashboard.export_metrics(format="json")

# Export as CSV
csv_data = dashboard.export_metrics(format="csv")

# Export specific sections
json_data = dashboard.export_metrics(
    format="json",
    sections=["fairness", "appeals"]
)
```

## Integration

### With Runtime Probes

```python
from probes import DeterminismProbe, TerminationProbe
from dashboards import GovernanceDashboard

dashboard = GovernanceDashboard()

# Create and run probes
det_probe = DeterminismProbe(eval_service)
result = det_probe.run()

# Update dashboard with probe result
dashboard.update_probe_result(
    probe_name="P-DET-Determinism",
    result=result
)

# Get invariant violations
metrics = dashboard.get_metrics(sections=["invariant_violations"])
print(metrics.invariant_violations)
```

### With REST API

```python
from flask import Flask, jsonify

app = Flask(__name__)
dashboard = GovernanceDashboard()

@app.route('/api/dashboard/metrics')
def get_metrics():
    metrics = dashboard.get_metrics()
    return jsonify(metrics.to_dict())

@app.route('/api/dashboard/fairness')
def get_fairness():
    metrics = dashboard.get_metrics(sections=["fairness"])
    return jsonify(metrics.fairness)

@app.route('/api/dashboard/export/<format>')
def export_metrics(format):
    data = dashboard.export_metrics(format=format)
    return data
```

### With GraphQL

```python
import graphene

class FairnessMetrics(graphene.ObjectType):
    statistical_parity = graphene.Float()
    disparate_impact = graphene.Float()
    status = graphene.String()

class Query(graphene.ObjectType):
    fairness_metrics = graphene.Field(FairnessMetrics)
    
    def resolve_fairness_metrics(self, info):
        dashboard = GovernanceDashboard()
        metrics = dashboard.get_metrics(sections=["fairness"])
        sp = metrics.fairness['statistical_parity']
        return FairnessMetrics(
            statistical_parity=sp['difference'],
            disparate_impact=metrics.fairness['disparate_impact']['ratio'],
            status=sp['status']
        )

schema = graphene.Schema(query=Query)
```

## Visualization

### Prometheus Integration

```python
from prometheus_client import Gauge, Counter, Histogram

# Define metrics
fairness_sp_gauge = Gauge(
    'nethical_fairness_statistical_parity',
    'Statistical Parity Difference'
)

appeals_counter = Counter(
    'nethical_appeals_total',
    'Total Appeals Filed'
)

latency_histogram = Histogram(
    'nethical_dashboard_query_latency_seconds',
    'Dashboard Query Latency'
)

# Update metrics
def update_prometheus_metrics(dashboard):
    metrics = dashboard.get_metrics()
    
    sp = metrics.fairness['statistical_parity']
    fairness_sp_gauge.set(sp['difference'])
    
    appeals_counter.inc(metrics.appeals['volume']['total_appeals'])
    
    latency_histogram.observe(
        metrics.slo_compliance['query_latency_seconds']
    )
```

### Grafana Dashboard

Example Grafana dashboard JSON available in `/examples/grafana/governance_dashboard.json`.

Key panels:
- Fairness metrics time series
- Policy lineage integrity status
- Appeals resolution time percentiles
- Invariant violations alert list
- SLO compliance indicators

## Accessibility

The dashboard meets WCAG 2.1 AA standards:

### Features

- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: ARIA labels and semantic HTML
- **High Contrast Mode**: Support for high contrast themes
- **Text Alternatives**: Alt text for all visual elements
- **Color Contrast**: Minimum 4.5:1 ratio
- **Focus Indicators**: Clear focus indicators for all interactive elements

### Accessibility Information

```python
# Get accessibility info
accessibility = dashboard.get_accessibility_info()

print(f"WCAG Version: {accessibility['wcag_version']}")
print(f"Conformance Level: {accessibility['conformance_level']}")
print(f"Features: {accessibility['features']}")
```

## Performance

### SLO Compliance

The dashboard is designed to meet <5s latency SLO:

- **Caching**: 60-second cache TTL for expensive queries
- **Indexing**: Optimized data structures for fast lookups
- **Sampling**: Large datasets are sampled when appropriate
- **Async**: Non-blocking operations where possible

### Monitoring Dashboard Performance

```python
# Get dashboard performance metrics
metrics = dashboard.get_metrics()
latency = metrics.slo_compliance['query_latency_seconds']
slo_met = metrics.slo_compliance['latency_slo_met']

print(f"Query latency: {latency:.3f}s")
print(f"SLO met: {slo_met}")
```

### Optimization Tips

1. **Use Caching**: Default cache is enabled (60s TTL)
2. **Query Specific Sections**: Only request needed sections
3. **Batch Updates**: Record multiple data points before querying
4. **Monitor Cache Hit Rate**: Track cache effectiveness

## Notification Channels

Configure notification channels for critical alerts:

### Email Notifications

```json
{
  "type": "email",
  "severity_levels": ["critical", "warning"],
  "recipients": ["ops@example.com", "security@example.com"]
}
```

### Slack Integration

```json
{
  "type": "slack",
  "severity_levels": ["critical"],
  "webhook_url": "https://hooks.slack.com/...",
  "channel": "#governance-alerts"
}
```

### PagerDuty Integration

```json
{
  "type": "pagerduty",
  "severity_levels": ["critical"],
  "integration_key": "..."
}
```

## Troubleshooting

### Dashboard Latency Exceeds 5s

1. Check cache hit rate
2. Review data volume (consider sampling)
3. Optimize probe check intervals
4. Add database indexes

### Metrics Not Updating

1. Verify data is being recorded
2. Check cache TTL settings
3. Force cache refresh with `use_cache=False`
4. Review probe execution logs

### Missing Fairness Metrics

1. Ensure decisions are being recorded
2. Check protected group labels
3. Verify sufficient sample size
4. Review time window configuration

## References

- [Runtime Probes Specification](./runtime_probes.md)
- [SLO Definitions](./slo_definitions.md)
- [Operational Runbook](./runbook.md)
- [Governance Configuration](../../dashboards/governance.json)
