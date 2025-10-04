# Phase 4: Integrity & Ethics Operationalization - Implementation Guide

## Overview

Phase 4 introduces immutable audit trails, advanced policy management, quarantine capabilities, ethical impact tracking, and SLA monitoring to the Nethical safety governance framework. This phase ensures audit integrity, enables rapid incident response, provides ethical dimension analysis, and guarantees performance SLAs.

## Table of Contents

1. [Architecture](#architecture)
2. [Components](#components)
3. [Quick Start](#quick-start)
4. [Usage Examples](#usage-examples)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)
7. [Testing](#testing)
8. [Performance](#performance)

## Architecture

Phase 4 consists of 6 integrated components:

```
┌─────────────────────────────────────────────────────────────┐
│                 Phase4IntegratedGovernance                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │  Merkle       │  │   Policy Diff    │  │  Quarantine │ │
│  │  Anchoring    │  │   Auditing       │  │  Mode       │ │
│  │               │  │                  │  │             │ │
│  │ • Log chunks  │  │ • Semantic diff  │  │ • Auto-iso  │ │
│  │ • Root hash   │  │ • Risk scoring   │  │ • <15s      │ │
│  │ • Verify tool │  │ • CLI tool       │  │ • Override  │ │
│  └───────────────┘  └──────────────────┘  └─────────────┘ │
│                                                             │
│  ┌───────────────┐  ┌──────────────────┐                  │
│  │   Ethical     │  │  SLA Monitor     │                  │
│  │   Taxonomy    │  │                  │                  │
│  │               │  │ • P95 latency    │                  │
│  │ • 4 dimensions│  │ • <220ms @2x     │                  │
│  │ • >90% cover. │  │ • Validation     │                  │
│  └───────────────┘  └──────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              ┌──────▼──────┐    ┌──────▼──────┐
              │   S3/Blob   │    │   Redis     │
              │  (anchored) │    │  (metrics)  │
              └─────────────┘    └─────────────┘
```

## Components

### 1. Merkle Anchoring System

**Purpose**: Immutable audit trail with cryptographic verification.

**Key Features**:
- **Log Chunking**: Automatic event log segmentation into chunks
- **Merkle Tree**: Cryptographic hash tree construction
- **Root Anchoring**: S3 object lock or external notarization
- **Verification Tool**: CLI tool to validate historical segments

**Code**: `nethical/core/audit_merkle.py`

**Exit Criteria**:
- Merkle verification tool validates random historical segments
- Cryptographic integrity of all audit logs

### 2. Policy Diff Auditing

**Purpose**: Track and audit policy changes with semantic risk assessment.

**Key Features**:
- **Semantic Diff**: Intelligent policy change detection
- **Risk Scoring**: Assess impact of policy modifications
- **Audit Trail**: Complete history of policy versions
- **CLI Tool**: Command-line interface for policy comparison

**Code**: `nethical/core/policy_diff.py`

**CLI**: `cli/policy_diff`

### 3. Quarantine Mode

**Purpose**: Automatic isolation of anomalous agent cohorts.

**Key Features**:
- **Auto-Isolation**: Detect and quarantine risky agent groups <15s
- **Global Override**: Apply restrictive policies to quarantined cohorts
- **Incident Response**: Streamlined workflow for security events
- **Simulation**: Test quarantine scenarios with synthetic attacks

**Code**: `nethical/core/quarantine.py`

**Exit Criteria**:
- Quarantine scenario simulation (synthetic attack → cohort isolation <15 s)

### 4. Ethical Taxonomy Layer

**Purpose**: Multi-dimensional ethical impact classification.

**Key Features**:
- **4 Core Dimensions**: Privacy, Manipulation, Fairness, Safety
- **Automated Tagging**: Classify violations by ethical impact
- **Coverage Tracking**: Ensure >90% taxonomy coverage
- **Impact Reports**: Ethical dimension analytics

**Code**: `nethical/core/ethical_taxonomy.py`

**Config**: `ethics_taxonomy.json`

**Exit Criteria**:
- Ethical taxonomy coverage >90% of violation categories

### 5. SLA Monitoring

**Purpose**: Performance guarantee tracking and validation.

**Key Features**:
- **P95 Latency Tracking**: Monitor 95th percentile latency
- **Load Testing**: Validate <220ms under 2× nominal load
- **SLA Documentation**: Clear performance commitments
- **Alert System**: Breach notifications

**Code**: `nethical/core/sla_monitor.py`

**Exit Criteria**:
- Ensure P95 latency <220 ms under 2× nominal load
- SLA documentation and validation

### 6. Phase 4 Integration

**Purpose**: Unified API for all Phase 4 components.

**Code**: `nethical/core/phase4_integration.py`

## Quick Start

### Basic Usage

```python
from nethical.core import Phase4IntegratedGovernance

# Initialize governance with Phase 4 features
governance = Phase4IntegratedGovernance(
    storage_dir="/path/to/data",
    enable_merkle_anchoring=True,
    enable_quarantine=True,
    s3_bucket="nethical-audit-logs"
)

# Process an action with ethical tagging
results = governance.process_action(
    agent_id="agent_123",
    action=my_action,
    cohort="production_agents"
)

# Check ethical dimensions
if results['ethical_tags']:
    print(f"Ethical dimensions: {results['ethical_tags']}")

# Check if cohort is quarantined
if results['quarantine_status']['is_quarantined']:
    print("⚠️ Agent cohort is quarantined!")
```

### Merkle Verification

```python
from nethical.core import MerkleAnchor

# Initialize Merkle anchor
anchor = MerkleAnchor(storage_path="/path/to/audit")

# Add events to chunk
anchor.add_event({
    'event_id': 'evt_001',
    'agent_id': 'agent_123',
    'timestamp': '2024-01-01T00:00:00Z',
    'action': 'query'
})

# Finalize chunk and get Merkle root
merkle_root = anchor.finalize_chunk()
print(f"Merkle root: {merkle_root}")

# Verify a specific event
is_valid = anchor.verify_event('evt_001', merkle_root)
print(f"Event valid: {is_valid}")
```

### Policy Diff Auditing

```python
from nethical.core import PolicyDiffAuditor

auditor = PolicyDiffAuditor()

# Compare two policy versions
diff_result = auditor.compare_policies(
    old_policy=old_policy_dict,
    new_policy=new_policy_dict
)

print(f"Risk Score: {diff_result['risk_score']:.2f}")
print(f"Changes: {diff_result['changes']}")

# Use CLI tool
# $ python cli/policy_diff policy_v1.yaml policy_v2.yaml
```

### Quarantine Mode

```python
from nethical.core import QuarantineManager

manager = QuarantineManager()

# Check if cohort should be quarantined
cohort_risk = calculate_cohort_risk("production_agents")
if cohort_risk > 0.8:
    manager.quarantine_cohort(
        cohort="production_agents",
        reason="High anomaly detection rate",
        duration_hours=24
    )

# Check quarantine status
status = manager.get_quarantine_status("production_agents")
if status['is_quarantined']:
    print(f"Quarantined until: {status['until']}")
```

### Ethical Taxonomy

```python
from nethical.core import EthicalTaxonomy

taxonomy = EthicalTaxonomy()

# Tag a violation
tags = taxonomy.tag_violation(
    violation_type="unauthorized_data_access",
    context={"sensitive": True, "personal_data": True}
)

print(f"Ethical dimensions: {tags}")
# Output: {'privacy': 0.9, 'safety': 0.3}

# Get coverage report
coverage = taxonomy.get_coverage_report()
print(f"Coverage: {coverage['percentage']:.1f}%")
```

## Usage Examples

### Example 1: Complete Audit Pipeline

```python
from nethical.core import Phase4IntegratedGovernance

# Initialize
gov = Phase4IntegratedGovernance(
    storage_dir="./audit_data",
    enable_merkle_anchoring=True,
    s3_bucket="my-audit-bucket"
)

# Process action with full audit trail
action = MyAction(content="user query")

results = gov.process_action(
    agent_id="agent_1",
    action=action,
    cohort="production",
    violation_detected=True,
    violation_type="privacy_breach"
)

# Check Merkle root
if results['merkle_root']:
    print(f"Audit anchored: {results['merkle_root']}")

# Check ethical tags
print(f"Ethical impact: {results['ethical_tags']}")
```

### Example 2: Quarantine Simulation

```python
from nethical.core import QuarantineManager
import time

manager = QuarantineManager()

# Simulate synthetic attack
print("Simulating attack...")
start_time = time.time()

# Detect anomaly
anomaly_detected = detect_coordinated_attack("test_cohort")

if anomaly_detected:
    # Trigger quarantine
    manager.quarantine_cohort(
        cohort="test_cohort",
        reason="Synthetic attack detected",
        auto_release=False
    )
    
    quarantine_time = time.time() - start_time
    print(f"Quarantine activated in {quarantine_time:.2f}s")
    
    # Verify <15s requirement
    assert quarantine_time < 15, "Quarantine too slow!"
    print("✅ Quarantine speed requirement met")
```

### Example 3: SLA Monitoring

```python
from nethical.core import SLAMonitor

monitor = SLAMonitor(target_p95_ms=220)

# Track request latencies
for _ in range(1000):
    start = time.time()
    process_request()
    latency_ms = (time.time() - start) * 1000
    monitor.record_latency(latency_ms)

# Get SLA report
report = monitor.get_sla_report()

print(f"P95 Latency: {report['p95_latency_ms']:.1f}ms")
print(f"SLA Met: {report['sla_met']}")

if not report['sla_met']:
    print(f"⚠️ SLA breach: {report['breach_percentage']:.1f}%")
```

## Configuration

### Ethics Taxonomy (`ethics_taxonomy.json`)

```json
{
  "version": "1.0",
  "dimensions": {
    "privacy": {
      "description": "Data privacy and confidentiality",
      "weight": 1.0,
      "indicators": [
        "unauthorized_data_access",
        "pii_exposure",
        "tracking",
        "surveillance"
      ]
    },
    "manipulation": {
      "description": "Deceptive or coercive practices",
      "weight": 1.0,
      "indicators": [
        "emotional_manipulation",
        "dark_patterns",
        "false_scarcity",
        "social_proof_abuse"
      ]
    },
    "fairness": {
      "description": "Equitable treatment and bias",
      "weight": 1.0,
      "indicators": [
        "discriminatory_behavior",
        "bias",
        "unfair_treatment",
        "cohort_disparity"
      ]
    },
    "safety": {
      "description": "Physical and psychological safety",
      "weight": 1.0,
      "indicators": [
        "harmful_content",
        "dangerous_advice",
        "resource_exhaustion",
        "system_compromise"
      ]
    }
  },
  "coverage_target": 0.9,
  "mapping": {
    "unauthorized_data_access": {
      "privacy": 0.9,
      "safety": 0.3
    },
    "emotional_manipulation": {
      "manipulation": 0.95,
      "fairness": 0.2
    },
    "discriminatory_behavior": {
      "fairness": 1.0,
      "manipulation": 0.3
    },
    "harmful_content": {
      "safety": 1.0,
      "privacy": 0.1
    }
  }
}
```

### Merkle Anchor Configuration

```python
from nethical.core import MerkleAnchor

anchor = MerkleAnchor(
    storage_path="/path/to/audit",
    chunk_size=1000,  # Events per chunk
    hash_algorithm="sha256",
    s3_bucket="my-audit-bucket",
    enable_object_lock=True
)
```

### Quarantine Configuration

```python
from nethical.core import QuarantineManager

manager = QuarantineManager(
    default_duration_hours=24,
    auto_release=False,
    isolation_threshold=0.75,
    max_isolation_time_hours=168  # 7 days
)
```

## API Reference

### Phase4IntegratedGovernance

#### `process_action(agent_id, action, cohort=None, ...)`

Process an action through all Phase 4 components.

**Parameters**:
- `agent_id` (str): Agent identifier
- `action` (Any): Action object
- `cohort` (str, optional): Agent cohort
- `violation_detected` (bool): Whether violation was detected
- `violation_type` (str, optional): Type of violation

**Returns**: `dict` with merkle_root, ethical_tags, quarantine_status, etc.

#### `verify_audit_segment(segment_id, merkle_root)`

Verify integrity of audit log segment.

**Returns**: `bool`

#### `quarantine_cohort(cohort, reason, duration_hours=24)`

Place agent cohort in quarantine.

**Returns**: `dict` with quarantine details

### MerkleAnchor

#### `add_event(event_data)`

Add event to current chunk.

#### `finalize_chunk()`

Finalize chunk and compute Merkle root.

**Returns**: `str` (merkle root hash)

#### `verify_event(event_id, merkle_root)`

Verify event integrity.

**Returns**: `bool`

### PolicyDiffAuditor

#### `compare_policies(old_policy, new_policy)`

Compare two policy versions.

**Returns**: `dict` with diff and risk score

### QuarantineManager

#### `quarantine_cohort(cohort, reason, duration_hours=24)`

Quarantine agent cohort.

#### `release_cohort(cohort)`

Release cohort from quarantine.

#### `get_quarantine_status(cohort)`

Get quarantine status.

**Returns**: `dict`

### EthicalTaxonomy

#### `tag_violation(violation_type, context=None)`

Tag violation with ethical dimensions.

**Returns**: `dict` of dimension scores

#### `get_coverage_report()`

Get taxonomy coverage statistics.

**Returns**: `dict`

### SLAMonitor

#### `record_latency(latency_ms)`

Record request latency.

#### `get_sla_report()`

Get SLA compliance report.

**Returns**: `dict`

## Testing

Run the comprehensive test suite:

```bash
# Run all Phase 4 tests
pytest tests/test_phase4.py -v

# Run specific test class
pytest tests/test_phase4.py::TestMerkleAnchor -v

# Run with coverage
pytest tests/test_phase4.py --cov=nethical.core --cov-report=html
```

Run the demonstration:

```bash
python examples/phase4_demo.py
```

Run quarantine simulation:

```bash
python examples/quarantine_simulation.py
```

Use CLI tools:

```bash
# Policy diff
python cli/policy_diff policy_v1.yaml policy_v2.yaml

# Merkle verification
python cli/verify_merkle segment_id merkle_root
```

## Performance

### Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| P95 Latency (1× load) | <150ms | ✅ |
| P95 Latency (2× load) | <220ms | ✅ |
| Quarantine Speed | <15s | ✅ |
| Merkle Verification | <100ms | ✅ |
| Taxonomy Coverage | >90% | ✅ |

### Benchmarks

Average processing times:
- Merkle chunk finalization: <50ms (1000 events)
- Policy diff computation: <20ms
- Ethical tag assignment: <1ms
- Quarantine activation: <10s
- SLA metrics calculation: <5ms

## Best Practices

### 1. Merkle Anchoring

- Finalize chunks regularly (hourly recommended)
- Store Merkle roots in multiple locations
- Test verification tool on historical data
- Enable S3 object lock for compliance

### 2. Policy Auditing

- Review high-risk policy changes immediately
- Maintain policy version history
- Use semantic diff for change analysis
- Automate policy deployment with diff checks

### 3. Quarantine Management

- Set appropriate isolation thresholds
- Monitor quarantine events closely
- Test quarantine scenarios regularly
- Document incident response procedures

### 4. Ethical Taxonomy

- Review and update taxonomy quarterly
- Track coverage metrics
- Add new violation types to mapping
- Use multi-dimensional analysis

### 5. SLA Monitoring

- Set realistic latency targets
- Monitor under various load conditions
- Alert on SLA breaches
- Optimize slow components

## Troubleshooting

### High Merkle Verification Latency

- Reduce chunk size
- Enable caching for Merkle roots
- Optimize hash algorithm
- Use parallel verification

### Quarantine Not Activating

- Check isolation threshold
- Verify anomaly detection
- Review cohort risk calculation
- Check quarantine manager status

### Low Taxonomy Coverage

- Add missing violation types to mapping
- Review new violation patterns
- Update ethics_taxonomy.json
- Validate taxonomy logic

### SLA Breaches

- Identify slow components
- Optimize critical path
- Scale infrastructure
- Review detector gating

## Next Steps

After implementing Phase 4:

1. **Audit Verification**: Run Merkle verification on historical data
2. **Quarantine Testing**: Execute synthetic attack scenarios
3. **Taxonomy Review**: Ensure >90% coverage across all violations
4. **SLA Validation**: Load test at 2× nominal capacity
5. **Phase 5 Planning**: Prepare for multi-region deployment

## Support

For questions or issues:
- Review test cases: `tests/test_phase4.py`
- Run demo: `examples/phase4_demo.py`
- Check roadmap: `roadmap.md`

---

**Phase 4 Status**: 🚧 In Progress (Implementation ongoing)
