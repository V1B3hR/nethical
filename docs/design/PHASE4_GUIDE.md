# Phase 4 Guide: Integrity & Ethics Operationalization

This guide provides comprehensive documentation for Phase 4 of the Nethical safety governance system.

## Table of Contents

1. [Overview](#overview)
2. [Merkle Anchoring](#merkle-anchoring)
3. [Policy Diff Auditing](#policy-diff-auditing)
4. [Quarantine Mode](#quarantine-mode)
5. [Ethical Taxonomy](#ethical-taxonomy)
6. [SLA Monitoring](#sla-monitoring)
7. [Integration](#integration)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

---

## Overview

Phase 4 focuses on operational integrity, ethics, and performance guarantees:

- **Merkle Anchoring**: Immutable audit trails with cryptographic verification
- **Policy Diff Auditing**: Change management and risk assessment for policy updates
- **Quarantine Mode**: Rapid incident response with sub-second activation
- **Ethical Taxonomy**: Multi-dimensional ethical impact analysis
- **SLA Monitoring**: Performance guarantees with latency tracking

These components ensure governance decisions are auditable, ethical implications are understood, incidents can be contained rapidly, and performance meets business requirements.

### Key Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `MerkleAnchor` | Audit integrity | Tamper-proof logs, cryptographic verification |
| `PolicyDiffAuditor` | Change management | Risk scoring, recommendations, history tracking |
| `QuarantineManager` | Incident response | Sub-second activation, cohort isolation |
| `EthicalTaxonomy` | Ethical analysis | Multi-dimensional tagging, coverage tracking |
| `SLAMonitor` | Performance SLA | P95/P99 latency, compliance tracking |
| `Phase4IntegratedGovernance` | Unified interface | All Phase 4 features combined |

---

## Merkle Anchoring

Merkle Anchoring provides cryptographically verifiable audit trails that are tamper-proof and can detect any unauthorized modifications.

### Architecture

```
Events → Chunk → Merkle Tree → Root Hash → Verification
            ↓
       SHA-256 Hashing
            ↓
    Immutable Storage
```

### Core Features

#### 1. Event Recording

Record governance events with automatic hashing:

```python
from nethical.core import MerkleAnchor

anchor = MerkleAnchor(
    storage_path="./audit_logs",
    s3_bucket=None  # Optional S3 for cloud storage
)

# Add events to current chunk
for i in range(100):
    anchor.add_event({
        'timestamp': datetime.utcnow().isoformat(),
        'agent_id': f'agent_{i}',
        'action': f'action_{i}',
        'decision': 'allow',
        'risk_score': 0.3
    })

print(f"Events in current chunk: {len(anchor.current_chunk.events)}")
```

#### 2. Chunk Finalization

Finalize chunks and compute Merkle root:

```python
# Finalize when chunk reaches size limit or on schedule
merkle_root = anchor.finalize_chunk()

print(f"✓ Chunk finalized")
print(f"  Merkle Root: {merkle_root}")
print(f"  Total Events: {anchor.current_chunk_index * anchor.chunk_size}")
```

#### 3. Verification

Verify chunk integrity:

```python
# Get chunk ID
chunk_id = list(anchor.finalized_chunks.keys())[0]

# Verify integrity
is_valid = anchor.verify_chunk(chunk_id)

if is_valid:
    print("✅ Chunk integrity verified")
else:
    print("❌ ALERT: Chunk tampering detected!")
    # Trigger security incident
```

#### 4. Merkle Tree Construction

The system builds a Merkle tree from event hashes:

```
        Root Hash
       /         \
    H(AB)       H(CD)
    /  \        /  \
   H(A) H(B)  H(C) H(D)
   |    |     |    |
  Evt1 Evt2 Evt3 Evt4
```

Each level is computed as: `hash(left + right)`

#### 5. Audit Statistics

Get comprehensive audit statistics:

```python
stats = anchor.get_statistics()

print(f"Total Chunks: {stats['total_chunks']}")
print(f"Total Events: {stats['total_events']}")
print(f"Current Chunk Events: {stats['current_chunk_events']}")
print(f"Hash Algorithm: {stats['hash_algorithm']}")
print(f"Storage Path: {stats['storage_path']}")
```

#### 6. Cloud Storage Integration

Upload to S3 for compliance:

```python
# Initialize with S3 bucket
anchor = MerkleAnchor(
    storage_path="./audit_logs",
    s3_bucket="my-audit-bucket"
)

# Finalize chunk (automatically uploads to S3)
merkle_root = anchor.finalize_chunk()

# Root hash stored both locally and in cloud
```

#### 7. Chunk Retrieval

Access historical chunks:

```python
# List all finalized chunks
for chunk_id, chunk in anchor.finalized_chunks.items():
    print(f"Chunk {chunk_id}:")
    print(f"  Events: {len(chunk.events)}")
    print(f"  Merkle Root: {chunk.merkle_root}")
    print(f"  Finalized: {chunk.finalized_at}")
```

---

## Policy Diff Auditing

Policy Diff Auditing tracks policy changes, assesses risk, and provides recommendations for safe deployments.

### Architecture

```
Old Policy → Diff Analysis → Risk Assessment → Recommendations
New Policy ↗              ↓
                    Version History
```

### Core Features

#### 1. Policy Comparison

Compare policy versions and assess risk:

```python
from nethical.core import PolicyDiffAuditor

auditor = PolicyDiffAuditor(
    storage_path="./policy_history"
)

old_policy = {
    'threshold': 0.5,
    'rate_limit': 100,
    'features': {
        'ml_enabled': True,
        'strict_mode': False
    }
}

new_policy = {
    'threshold': 0.8,  # Modified
    'rate_limit': 150,  # Modified
    'features': {
        'ml_enabled': True,
        'strict_mode': True  # Modified
    },
    'new_feature': 'enabled'  # Added
}

# Compare policies
diff = auditor.compare_policies(old_policy, new_policy)

print(f"Risk Level: {diff.risk_level.value}")
print(f"Risk Score: {diff.risk_score:.3f}")
print(f"Changes: {diff.summary['added']} added, {diff.summary['modified']} modified, {diff.summary['removed']} removed")
```

#### 2. Risk Levels

Policy changes are classified by risk:

| Risk Level | Score Range | Meaning | Action |
|------------|-------------|---------|--------|
| `LOW` | 0.0 - 0.29 | Minimal impact | Deploy immediately |
| `MEDIUM` | 0.3 - 0.59 | Moderate impact | Review recommended |
| `HIGH` | 0.6 - 0.79 | Significant impact | Careful review required |
| `CRITICAL` | 0.8 - 1.0 | Major impact | Extended testing needed |

#### 3. Change Analysis

Detailed breakdown of changes:

```python
print("\nDetailed Changes:")
for change in diff.changes:
    print(f"  {change['type'].upper()}: {change['path']}")
    if change['type'] == 'modified':
        print(f"    Old: {change['old_value']}")
        print(f"    New: {change['new_value']}")
    elif change['type'] == 'added':
        print(f"    Value: {change['new_value']}")
```

#### 4. Recommendations

Get automated recommendations:

```python
print("\nRecommendations:")
for rec in diff.recommendations:
    print(f"  • {rec}")

# Example output:
# • Test in staging environment before production
# • Monitor closely for 24 hours after deployment
# • Prepare rollback plan
# • Document changes in release notes
```

#### 5. Version History

Track all policy changes:

```python
# Record policy deployment
auditor.record_deployment(
    policy=new_policy,
    version="2.1.0",
    deployed_by="admin@example.com",
    notes="Increased thresholds based on operational data"
)

# Get history
history = auditor.get_history(limit=10)

for entry in history:
    print(f"Version {entry['version']}:")
    print(f"  Deployed: {entry['deployed_at']}")
    print(f"  By: {entry['deployed_by']}")
    print(f"  Risk: {entry['risk_level']}")
```

#### 6. Rollback Support

Retrieve previous policy versions:

```python
# Get specific version
previous = auditor.get_version("2.0.0")

# Rollback to previous version
if incident_detected():
    rollback_policy = auditor.get_previous_version()
    deploy_policy(rollback_policy)
```

#### 7. Risk Scoring Algorithm

Risk is calculated based on:

- Number of changes (more changes = higher risk)
- Type of changes (modified > added > removed)
- Critical field changes (security, thresholds, limits)
- Magnitude of changes (larger deltas = higher risk)

```python
# High-risk changes automatically flagged:
# - Security settings disabled
# - Thresholds relaxed significantly
# - Rate limits increased drastically
# - Critical features removed
```

---

## Quarantine Mode

Quarantine Mode enables rapid isolation of compromised agents or cohorts to contain security incidents.

### Architecture

```
Threat Detection → Quarantine Activation → Agent/Cohort Isolation
                         ↓
                   <1 second response
                         ↓
                   Action Blocking
```

### Core Features

#### 1. Agent Registration

Register agents with cohort membership:

```python
from nethical.core import QuarantineManager, QuarantineReason

manager = QuarantineManager(
    storage_path="./quarantine_data"
)

# Register agents to cohorts
manager.register_agent_cohort('agent_1', 'production')
manager.register_agent_cohort('agent_2', 'production')
manager.register_agent_cohort('agent_3', 'staging')
```

#### 2. Quarantine Activation

Quarantine individual agents or entire cohorts:

```python
# Quarantine single agent
manager.quarantine_agent(
    agent_id='agent_1',
    reason=QuarantineReason.MALICIOUS_ACTIVITY,
    duration_hours=24,
    notes="Detected suspicious pattern"
)

# Quarantine entire cohort
manager.quarantine_cohort(
    cohort='production',
    reason=QuarantineReason.SECURITY_INCIDENT,
    duration_hours=2,
    notes="Coordinated attack detected"
)
```

#### 3. Performance Guarantee

Quarantine activation completes in <1 second:

```python
import time

# Measure activation time
start = time.time()
manager.quarantine_cohort('production', QuarantineReason.SECURITY_INCIDENT)
duration = time.time() - start

print(f"Activation time: {duration:.3f}s")
# Typical: 0.001 - 0.050 seconds

assert duration < 1.0, "SLA violation: Quarantine took >1 second"
```

#### 4. Action Blocking

Quarantined agents/cohorts are blocked from actions:

```python
# Check if agent is quarantined
is_quarantined = manager.is_agent_quarantined('agent_1')

if is_quarantined:
    status = manager.get_agent_status('agent_1')
    print(f"Agent quarantined: {status['reason']}")
    print(f"Expires: {status['expires_at']}")
    # Block action
    return "QUARANTINED"
else:
    # Allow action
    process_action(agent)
```

#### 5. Release Management

Release agents or cohorts from quarantine:

```python
# Release single agent
manager.release_agent('agent_1')

# Release entire cohort
manager.release_cohort('production')

# Verify release
assert not manager.is_agent_quarantined('agent_1')
```

#### 6. Quarantine Reasons

Structured reasons for audit trail:

```python
class QuarantineReason(Enum):
    SECURITY_INCIDENT = "security_incident"
    MALICIOUS_ACTIVITY = "malicious_activity"
    POLICY_VIOLATION = "policy_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    MANUAL_OVERRIDE = "manual_override"
```

#### 7. Statistics and Monitoring

Track quarantine activity:

```python
stats = manager.get_statistics()

print(f"Currently Quarantined: {stats['quarantined_agents']}")
print(f"Total Quarantines (all time): {stats['total_quarantines']}")
print(f"Avg Duration: {stats['avg_duration_hours']:.1f} hours")

# Monitor quarantine events
events = manager.get_quarantine_history(limit=10)
for event in events:
    print(f"{event['timestamp']}: {event['agent_id']} - {event['action']}")
```

---

## Ethical Taxonomy

Ethical Taxonomy provides multi-dimensional analysis of ethical implications for each violation type.

### Architecture

```
Violation → Taxonomy Lookup → Multi-Dimension Scoring → Primary Dimension
                ↓
           Coverage Tracking
```

### Core Features

#### 1. Taxonomy Configuration

Load ethical taxonomy from JSON:

```python
from nethical.core import EthicalTaxonomy

taxonomy = EthicalTaxonomy(
    taxonomy_path="ethics_taxonomy.json"
)

# Taxonomy defines dimensions:
# - Privacy (data protection, consent)
# - Manipulation (emotional, authority abuse)
# - Fairness (discrimination, bias)
# - Safety (harm, risk)
```

#### 2. Violation Tagging

Tag violations with ethical dimensions:

```python
# Tag a privacy violation
tags = taxonomy.tag_violation('unauthorized_data_access')

print(f"Primary Dimension: {tags['primary_dimension']}")
print(f"Dimension Scores:")
for dimension, score in tags['dimensions'].items():
    print(f"  {dimension}: {score:.2f}")

# Example output:
# Primary Dimension: privacy
# Dimension Scores:
#   privacy: 0.95
#   safety: 0.40
#   fairness: 0.20
#   manipulation: 0.10
```

#### 3. Multi-Dimensional Analysis

Understand full ethical impact:

```python
# A violation can have multiple ethical dimensions
tags = taxonomy.tag_violation('discriminatory_behavior')

# Primary: fairness (0.90)
# Secondary: privacy (0.30), safety (0.25)

# This helps identify violations with broad impact
```

#### 4. Coverage Tracking

Ensure taxonomy covers all violation types:

```python
# Track which violation types have been encountered
taxonomy.track_violation_type('unauthorized_data_access')
taxonomy.track_violation_type('emotional_manipulation')
taxonomy.track_violation_type('harmful_content')

# Get coverage report
coverage = taxonomy.get_coverage_report()

print(f"Coverage: {coverage['coverage_percentage']:.1f}%")
print(f"Covered: {coverage['covered_types']}/{coverage['total_types']}")
print(f"Missing: {', '.join(coverage['missing_types'])}")
```

#### 5. Taxonomy Structure

Example taxonomy structure:

```json
{
  "dimensions": ["privacy", "manipulation", "fairness", "safety"],
  "violations": {
    "unauthorized_data_access": {
      "privacy": 0.95,
      "manipulation": 0.10,
      "fairness": 0.20,
      "safety": 0.40
    },
    "emotional_manipulation": {
      "privacy": 0.15,
      "manipulation": 0.90,
      "fairness": 0.30,
      "safety": 0.35
    }
  }
}
```

#### 6. Custom Taxonomies

Create domain-specific taxonomies:

```python
# Healthcare taxonomy
healthcare_taxonomy = {
    "dimensions": ["privacy", "safety", "autonomy", "beneficence"],
    "violations": {
        "hipaa_violation": {
            "privacy": 0.98,
            "safety": 0.60,
            "autonomy": 0.30,
            "beneficence": 0.20
        }
    }
}

# Financial services taxonomy
finserv_taxonomy = {
    "dimensions": ["privacy", "fairness", "transparency", "accountability"],
    "violations": {
        "discriminatory_lending": {
            "privacy": 0.30,
            "fairness": 0.95,
            "transparency": 0.70,
            "accountability": 0.85
        }
    }
}
```

#### 7. Impact Scoring

Calculate overall ethical impact:

```python
tags = taxonomy.tag_violation('harmful_content')

# Weighted impact score
impact = sum(tags['dimensions'].values()) / len(tags['dimensions'])
print(f"Overall Ethical Impact: {impact:.2f}")

# Use for prioritization
if impact > 0.7:
    escalate_to_ethics_board()
```

---

## SLA Monitoring

SLA Monitoring tracks system performance and ensures latency targets are met.

### Architecture

```
Action Processing → Latency Measurement → Percentile Calculation → SLA Check
                          ↓
                    Time-Series Storage
                          ↓
                   Performance Reports
```

### Core Features

#### 1. Latency Tracking

Automatically track processing latency:

```python
from nethical.core import SLAMonitor

monitor = SLAMonitor(
    p95_target_ms=200.0,  # P95 latency target
    p99_target_ms=500.0   # P99 latency target
)

# Track action latency
for i in range(1000):
    start = time.time()
    
    # Process action
    process_governance_action()
    
    duration_ms = (time.time() - start) * 1000
    monitor.record_latency(duration_ms)
```

#### 2. Percentile Calculation

Get latency percentiles:

```python
metrics = monitor.get_metrics()

print(f"P50 (Median): {metrics['p50_latency_ms']:.2f}ms")
print(f"P95: {metrics['p95_latency_ms']:.2f}ms")
print(f"P99: {metrics['p99_latency_ms']:.2f}ms")
print(f"P99.9: {metrics['p999_latency_ms']:.2f}ms")
print(f"Max: {metrics['max_latency_ms']:.2f}ms")
```

#### 3. SLA Compliance

Check if SLA targets are met:

```python
status = monitor.get_sla_status()

print(f"SLA Met: {'✅' if status.sla_met else '❌'}")
print(f"P95: {status.p95_latency_ms:.2f}ms / {status.p95_target_ms}ms")

if status.sla_met:
    margin = ((status.p95_target_ms - status.p95_latency_ms) / status.p95_target_ms) * 100
    print(f"Margin: {margin:.1f}%")
else:
    violation = status.p95_latency_ms - status.p95_target_ms
    print(f"⚠️  SLA Violation: {violation:.2f}ms over target")
```

#### 4. Performance Reports

Generate comprehensive performance reports:

```python
report = monitor.generate_report()

print(f"Report Period: {report['start_time']} to {report['end_time']}")
print(f"Total Requests: {report['total_requests']}")
print(f"SLA Compliance: {report['sla_compliance_pct']:.1f}%")

print("\nLatency Distribution:")
for percentile, latency in report['latency_distribution'].items():
    print(f"  {percentile}: {latency:.2f}ms")

print("\nSLA Violations:")
print(f"  Count: {report['sla_violations']}")
print(f"  Rate: {report['violation_rate_pct']:.2f}%")
```

#### 5. Alerting Integration

Set up alerts for SLA violations:

```python
# Check SLA every minute
def check_sla():
    status = monitor.get_sla_status()
    
    if not status.sla_met:
        alert_operations(
            severity="high",
            message=f"SLA violation: P95={status.p95_latency_ms:.2f}ms (target={status.p95_target_ms}ms)"
        )
    
    if status.p95_latency_ms > status.p95_target_ms * 0.9:
        alert_operations(
            severity="warning",
            message=f"Approaching SLA limit: P95={status.p95_latency_ms:.2f}ms"
        )
```

#### 6. Historical Tracking

Track SLA over time:

```python
# Store hourly snapshots
history = monitor.get_hourly_snapshots(hours=24)

for timestamp, metrics in history.items():
    print(f"{timestamp}: P95={metrics['p95']:.2f}ms, SLA={'✅' if metrics['sla_met'] else '❌'}")

# Visualize trends
plot_sla_trends(history)
```

#### 7. Performance Targets

Configure targets based on requirements:

```python
# Strict SLA for critical systems
monitor_strict = SLAMonitor(
    p95_target_ms=100.0,
    p99_target_ms=200.0
)

# Relaxed SLA for non-critical systems
monitor_relaxed = SLAMonitor(
    p95_target_ms=500.0,
    p99_target_ms=1000.0
)
```

---

## Integration

### Phase 4 Integrated Governance

The `Phase4IntegratedGovernance` class combines all Phase 4 components:

```python
from nethical.core import Phase4IntegratedGovernance

governance = Phase4IntegratedGovernance(
    storage_dir="./phase4_data",
    enable_merkle_anchoring=True,
    enable_quarantine=True,
    enable_ethical_taxonomy=True,
    enable_sla_monitoring=True,
    s3_bucket=None,  # Optional S3 bucket
    taxonomy_path="ethics_taxonomy.json"
)

# Process action through all Phase 4 components
result = governance.process_action(
    agent_id="agent_123",
    action="test_action",
    cohort="production",
    violation_detected=True,
    violation_type="unauthorized_data_access"
)

# Result includes all Phase 4 data
print(f"Action Allowed: {result['action_allowed']}")
print(f"Quarantine Status: {result['quarantine_status']}")
print(f"Ethical Tags: {result['ethical_tags']}")
print(f"Audit Event ID: {result['merkle']['event_id']}")
print(f"Latency: {result['sla']['latency_ms']:.2f}ms")
```

### Migration to IntegratedGovernance

Phase4IntegratedGovernance is deprecated. Use IntegratedGovernance instead:

```python
# Old (deprecated)
from nethical.core import Phase4IntegratedGovernance
governance = Phase4IntegratedGovernance(storage_dir="./data")

# New (recommended)
from nethical.core import IntegratedGovernance
governance = IntegratedGovernance(
    storage_dir="./data",
    enable_merkle_anchoring=True,
    enable_quarantine=True,
    enable_ethical_taxonomy=True,
    enable_sla_monitoring=True
)
```

### Audit Trail Integration

Ensure all governance decisions are audited:

```python
# Every action is automatically logged
result = governance.process_action(
    agent_id="agent_123",
    action="sensitive_operation",
    cohort="production"
)

# Finalize audit chunk periodically
if time_to_finalize_chunk():
    merkle_root = governance.finalize_audit_chunk()
    store_merkle_root_externally(merkle_root)
```

### Incident Response Integration

Integrate with security incident response:

```python
def handle_security_incident(affected_agents=None, affected_cohort=None):
    """Rapid incident response."""
    
    # Quarantine affected entities
    if affected_cohort:
        governance.quarantine_cohort(
            cohort=affected_cohort,
            reason=QuarantineReason.SECURITY_INCIDENT,
            duration_hours=4
        )
    
    if affected_agents:
        for agent_id in affected_agents:
            governance.quarantine_agent(
                agent_id=agent_id,
                reason=QuarantineReason.MALICIOUS_ACTIVITY,
                duration_hours=24
            )
    
    # Alert security team
    alert_security_team(
        incident_type="quarantine_activated",
        details={
            'cohort': affected_cohort,
            'agents': affected_agents
        }
    )
```

---

## Best Practices

### 1. Merkle Anchoring

- **Chunk Size**: Choose appropriate chunk size (1000-10000 events)
  - Smaller: More frequent verification, higher overhead
  - Larger: Less overhead, longer to detect tampering

```python
# Production recommendation: 5000 events per chunk
anchor = MerkleAnchor(
    storage_path="./audit",
    chunk_size=5000
)
```

- **Verification Schedule**: Verify chunks regularly (hourly/daily)
- **External Storage**: Store Merkle roots in blockchain or external system
- **Backup**: Regular backups of finalized chunks

### 2. Policy Diff Auditing

- **Version Control**: Use semantic versioning for policies
- **Testing**: Test high-risk changes in staging first
- **Documentation**: Document rationale for all policy changes
- **Rollback Plan**: Always have a tested rollback procedure

```python
# Best practice workflow
def deploy_policy(new_policy, version):
    # Compare with current
    diff = auditor.compare_policies(current_policy, new_policy)
    
    # High risk requires approval
    if diff.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
        require_approval(diff, approvers=["security_lead", "engineering_lead"])
    
    # Test in staging
    if not test_in_staging(new_policy):
        return "Staging tests failed"
    
    # Deploy
    auditor.record_deployment(new_policy, version, deployed_by=current_user)
    
    # Monitor closely
    schedule_enhanced_monitoring(duration_hours=24)
```

### 3. Quarantine Mode

- **Duration**: Set reasonable quarantine durations
  - Critical: 2-4 hours
  - High: 12-24 hours
  - Investigation: 48-72 hours

- **Communication**: Notify affected users
- **Release Criteria**: Define clear release criteria
- **Testing**: Test quarantine activation regularly

```python
# Regular drills
def quarterly_quarantine_drill():
    """Test quarantine system."""
    test_cohort = "quarantine_drill"
    
    # Activate quarantine
    start = time.time()
    manager.quarantine_cohort(test_cohort, QuarantineReason.MANUAL_OVERRIDE)
    activation_time = time.time() - start
    
    # Verify
    assert activation_time < 1.0, "Quarantine SLA violation"
    assert manager.is_cohort_quarantined(test_cohort)
    
    # Release
    manager.release_cohort(test_cohort)
    
    # Report
    log_drill_results(activation_time)
```

### 4. Ethical Taxonomy

- **Completeness**: Ensure taxonomy covers all violation types
- **Updates**: Review and update taxonomy quarterly
- **Domain Specificity**: Customize for your domain (healthcare, finance, etc.)
- **Training**: Train team on ethical dimensions

```python
# Quarterly review
def review_taxonomy():
    coverage = taxonomy.get_coverage_report()
    
    if coverage['coverage_percentage'] < 90:
        print(f"⚠️  Low coverage: {coverage['coverage_percentage']:.1f}%")
        print(f"Missing: {coverage['missing_types']}")
        
        # Update taxonomy
        for violation_type in coverage['missing_types']:
            add_to_taxonomy(violation_type)
```

### 5. SLA Monitoring

- **Target Setting**: Set realistic targets based on requirements
  - User-facing: P95 < 200ms
  - Background: P95 < 500ms
  - Batch: P95 < 2000ms

- **Alerting**: Alert before SLA is breached (90% threshold)
- **Optimization**: Investigate P99 outliers
- **Capacity**: Plan for 2x expected load

```python
# Production configuration
monitor = SLAMonitor(
    p95_target_ms=200.0,
    p99_target_ms=500.0
)

# Alert at 90% of target
if monitor.get_sla_status().p95_latency_ms > 180.0:
    alert_engineering(
        severity="warning",
        message="Approaching SLA limit"
    )
```

### 6. Integration Best Practices

- **Initialization**: Initialize all components at startup
- **Error Handling**: Handle component failures gracefully
- **Monitoring**: Monitor all Phase 4 components
- **Testing**: Test end-to-end workflows

```python
# Robust initialization
def initialize_governance():
    try:
        governance = Phase4IntegratedGovernance(
            storage_dir="./data",
            enable_merkle_anchoring=True,
            enable_quarantine=True,
            enable_ethical_taxonomy=True,
            enable_sla_monitoring=True
        )
        
        # Verify all components
        assert governance.merkle_anchor is not None
        assert governance.quarantine_manager is not None
        assert governance.taxonomy is not None
        assert governance.sla_monitor is not None
        
        return governance
        
    except Exception as e:
        log_error(f"Governance initialization failed: {e}")
        # Fall back to safe mode
        return SafeModeGovernance()
```

---

## Examples

### Example 1: Complete Audit Trail

```python
from nethical.core import MerkleAnchor

anchor = MerkleAnchor(storage_path="./audit")

# Process 10,000 actions
for i in range(10000):
    anchor.add_event({
        'action_id': f'action_{i}',
        'agent_id': f'agent_{i % 100}',
        'timestamp': datetime.utcnow().isoformat(),
        'decision': 'allow' if i % 5 != 0 else 'block',
        'risk_score': 0.3 + (i % 7) * 0.1
    })
    
    # Finalize every 1000 events
    if (i + 1) % 1000 == 0:
        merkle_root = anchor.finalize_chunk()
        print(f"Chunk {(i+1)//1000}: {merkle_root}")

# Verify all chunks
for chunk_id in anchor.finalized_chunks.keys():
    is_valid = anchor.verify_chunk(chunk_id)
    print(f"Chunk {chunk_id}: {'✅ Valid' if is_valid else '❌ Invalid'}")
```

### Example 2: Policy Change Management

```python
from nethical.core import PolicyDiffAuditor

auditor = PolicyDiffAuditor(storage_path="./policy_history")

# Current production policy
current = {
    'threshold': 0.5,
    'rate_limit': 100,
    'security': {'enabled': True, 'level': 'standard'}
}

# Proposed changes
proposed = {
    'threshold': 0.7,
    'rate_limit': 150,
    'security': {'enabled': True, 'level': 'high'}
}

# Assess risk
diff = auditor.compare_policies(current, proposed)

print(f"Risk Assessment: {diff.risk_level.value.upper()}")
print(f"Risk Score: {diff.risk_score:.3f}")

# Approval workflow
if diff.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
    print("\n⚠️  High risk change requires approval")
    print("\nRecommendations:")
    for rec in diff.recommendations:
        print(f"  • {rec}")
    
    # Wait for approval
    approved = get_approval_from_team_leads()
    
    if approved:
        # Record deployment
        auditor.record_deployment(
            policy=proposed,
            version="2.1.0",
            deployed_by="admin@example.com"
        )
        print("\n✅ Policy deployed and recorded")
else:
    # Auto-deploy low/medium risk
    auditor.record_deployment(proposed, "2.1.0", "auto-deploy")
    print("\n✅ Low risk change auto-deployed")
```

### Example 3: Rapid Incident Response

```python
from nethical.core import QuarantineManager, QuarantineReason
import time

manager = QuarantineManager(storage_path="./quarantine")

# Register production agents
for i in range(100):
    manager.register_agent_cohort(f'agent_{i}', 'production')

# Detect coordinated attack
print("⚠️  ALERT: Coordinated attack detected!")
print("Activating quarantine...")

# Measure response time
start = time.time()
manager.quarantine_cohort(
    cohort='production',
    reason=QuarantineReason.SECURITY_INCIDENT,
    duration_hours=2,
    notes="Coordinated SQL injection attempts detected"
)
response_time = time.time() - start

print(f"✅ Quarantine activated in {response_time:.3f}s")

# Verify SLA
if response_time < 1.0:
    print("✅ SLA met (< 1 second)")
else:
    print("❌ SLA violation (≥ 1 second)")

# All agents now blocked
for i in range(5):
    is_blocked = manager.is_agent_quarantined(f'agent_{i}')
    print(f"agent_{i}: {'🚫 BLOCKED' if is_blocked else '✅ ALLOWED'}")

# After investigation and remediation
print("\n🔍 Investigation complete, releasing quarantine...")
manager.release_cohort('production')
print("✅ Production cohort released")
```

### Example 4: Ethical Impact Analysis

```python
from nethical.core import EthicalTaxonomy

taxonomy = EthicalTaxonomy(taxonomy_path="ethics_taxonomy.json")

violations = [
    'unauthorized_data_access',
    'emotional_manipulation',
    'discriminatory_behavior',
    'harmful_content'
]

print("Ethical Impact Analysis\n")

for violation_type in violations:
    tags = taxonomy.tag_violation(violation_type)
    
    print(f"{violation_type}:")
    print(f"  Primary: {tags['primary_dimension']}")
    
    # Sort dimensions by score
    sorted_dims = sorted(
        tags['dimensions'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    print("  Dimensions:")
    for dim, score in sorted_dims:
        bar = '█' * int(score * 20)
        print(f"    {dim:15s} {bar} {score:.2f}")
    
    # Overall impact
    impact = sum(tags['dimensions'].values()) / len(tags['dimensions'])
    print(f"  Overall Impact: {impact:.2f}\n")

# Coverage report
coverage = taxonomy.get_coverage_report()
print(f"Taxonomy Coverage: {coverage['coverage_percentage']:.1f}%")
```

### Example 5: SLA Compliance Monitoring

```python
from nethical.core import SLAMonitor
import time
import random

monitor = SLAMonitor(
    p95_target_ms=200.0,
    p99_target_ms=500.0
)

print("Processing 1000 actions...\n")

for i in range(1000):
    # Simulate action processing
    start = time.time()
    
    # Variable latency (most fast, some slow)
    if random.random() < 0.95:
        time.sleep(random.uniform(0.01, 0.15))  # Fast path
    else:
        time.sleep(random.uniform(0.2, 0.6))     # Slow path
    
    duration_ms = (time.time() - start) * 1000
    monitor.record_latency(duration_ms)
    
    # Check every 100 actions
    if (i + 1) % 100 == 0:
        status = monitor.get_sla_status()
        print(f"[{i+1}/1000] P95: {status.p95_latency_ms:.1f}ms, "
              f"SLA: {'✅' if status.sla_met else '❌'}")

# Final report
print("\n" + "="*50)
print("FINAL SLA REPORT")
print("="*50)

report = monitor.generate_report()
print(f"\nTotal Requests: {report['total_requests']}")
print(f"SLA Compliance: {report['sla_compliance_pct']:.1f}%")

print("\nLatency Percentiles:")
print(f"  P50: {report['latency_distribution']['p50']:.1f}ms")
print(f"  P95: {report['latency_distribution']['p95']:.1f}ms (target: 200ms)")
print(f"  P99: {report['latency_distribution']['p99']:.1f}ms (target: 500ms)")
print(f"  Max: {report['latency_distribution']['max']:.1f}ms")

if report['sla_compliance_pct'] >= 99.0:
    print("\n✅ Excellent SLA compliance!")
elif report['sla_compliance_pct'] >= 95.0:
    print("\n⚠️  Acceptable SLA compliance (some violations)")
else:
    print("\n❌ Poor SLA compliance - investigation required")
```

### Example 6: Complete Phase 4 Workflow

```python
from nethical.core import Phase4IntegratedGovernance

# Initialize
governance = Phase4IntegratedGovernance(
    storage_dir="./phase4_demo",
    enable_merkle_anchoring=True,
    enable_quarantine=True,
    enable_ethical_taxonomy=True,
    enable_sla_monitoring=True
)

print("Phase 4 Integrated Governance Demo\n")

# Process multiple actions
for i in range(20):
    result = governance.process_action(
        agent_id=f'agent_{i % 5}',
        action=f'action_{i}',
        cohort='production',
        violation_detected=(i % 4 == 0),
        violation_type='unauthorized_data_access' if (i % 4 == 0) else None
    )
    
    # Check results
    if not result['action_allowed']:
        print(f"❌ Action {i} BLOCKED: {result['reason']}")
    elif result.get('ethical_tags'):
        tags = result['ethical_tags']
        print(f"⚠️  Action {i}: Ethical concern ({tags['primary_dimension']})")

# Finalize audit chunk
merkle_root = governance.finalize_audit_chunk()
print(f"\n✅ Audit chunk finalized: {merkle_root}")

# Check SLA
sla_report = governance.get_sla_report()
print(f"\n📊 SLA Status: {'✅ COMPLIANT' if sla_report['sla_met'] else '❌ VIOLATED'}")
print(f"   P95 Latency: {sla_report['p95_latency_ms']:.2f}ms")

# Get system status
status = governance.get_system_status()
print(f"\n🔍 System Status:")
for component, info in status['components'].items():
    enabled = "✅" if info.get('enabled') else "❌"
    print(f"   {enabled} {component}")
```

---

## API Reference

### Phase4IntegratedGovernance

Main integration class combining all Phase 4 components.

**Methods:**

- `process_action(agent_id, action, cohort, violation_detected, ...)`: Process action through all components
- `finalize_audit_chunk()`: Finalize current audit chunk and return Merkle root
- `verify_audit_segment(chunk_id)`: Verify integrity of audit chunk
- `compare_policies(old_policy, new_policy)`: Compare policy versions
- `quarantine_cohort(cohort, reason, duration_hours)`: Quarantine entire cohort
- `release_cohort(cohort)`: Release cohort from quarantine
- `simulate_quarantine(cohort)`: Test quarantine activation time
- `get_ethical_coverage()`: Get taxonomy coverage statistics
- `get_sla_report()`: Get SLA compliance report
- `get_system_status()`: Get system-wide status

### See Also

- [Phase 3 Guide](PHASE3_GUIDE.md) - Advanced Governance Features
- [Phase 5-7 Guide](PHASE5-7_GUIDE.md) - ML & Anomaly Detection
- [Phase 8-9 Guide](PHASE89_GUIDE.md) - Human-in-the-Loop & Optimization
- [Main README](../../README.md) - Project overview

---

**Last Updated**: November 5, 2025  
**Version**: Phase 4 Complete  
**Status**: Production Ready
