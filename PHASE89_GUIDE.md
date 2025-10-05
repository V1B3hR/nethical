# Phase 8-9 Guide: Human-in-the-Loop & Continuous Optimization

This guide provides comprehensive documentation for Phases 8 and 9 of the Nethical safety governance system.

## Table of Contents

1. [Overview](#overview)
2. [Phase 8: Human-in-the-Loop Operations](#phase-8-human-in-the-loop-operations)
3. [Phase 9: Continuous Optimization](#phase-9-continuous-optimization)
4. [Integration](#integration)
5. [Best Practices](#best-practices)
6. [Examples](#examples)

---

## Overview

Phases 8 and 9 represent the final evolution of the Nethical governance system, introducing:

- **Phase 8**: Human review workflows with SLA tracking and structured feedback collection
- **Phase 9**: Multi-objective optimization with automated tuning and promotion gates

These phases enable continuous improvement through human-in-the-loop feedback and data-driven optimization.

### Key Components

| Component | Purpose | Phase |
|-----------|---------|-------|
| `EscalationQueue` | Manage cases requiring human review | 8 |
| `FeedbackTag` | Structured feedback categories | 8 |
| `SLAMetrics` | Track review performance | 8 |
| `MultiObjectiveOptimizer` | Optimize system parameters | 9 |
| `PromotionGate` | Validate configuration changes | 9 |
| `Phase89IntegratedGovernance` | Unified interface | 8-9 |

---

## Phase 8: Human-in-the-Loop Operations

### Architecture

Phase 8 implements a priority-based escalation queue with SLA tracking:

```
Action → Decision → Escalation Check → Queue (by priority)
                                            ↓
                                      Human Review
                                            ↓
                                    Structured Feedback
                                            ↓
                                   Continuous Improvement
```

### Core Features

#### 1. Escalation Queue

The `EscalationQueue` manages cases requiring human review:

```python
from nethical.core import EscalationQueue, ReviewPriority

queue = EscalationQueue(
    storage_path="./data/escalations.db",
    triage_sla_seconds=3600,      # 1 hour to start review
    resolution_sla_seconds=86400   # 24 hours to complete
)

# Add case to queue
case = queue.add_case(
    judgment_id="judg_123",
    action_id="act_456",
    agent_id="agent_789",
    decision="block",
    confidence=0.65,
    violations=[{"type": "safety", "severity": 4}],
    priority=ReviewPriority.HIGH
)
```

#### 2. Review Workflow

Human reviewers follow this workflow:

```python
# 1. Get next case from queue
case = queue.get_next_case(reviewer_id="alice")

# 2. Review case details
print(f"Decision: {case.decision}")
print(f"Confidence: {case.confidence}")
print(f"Violations: {case.violations}")

# 3. Submit feedback
from nethical.core import FeedbackTag

feedback = queue.submit_feedback(
    case_id=case.case_id,
    reviewer_id="alice",
    feedback_tags=[FeedbackTag.FALSE_POSITIVE],
    rationale="Content was actually safe, detector too aggressive",
    corrected_decision="allow",
    confidence=0.9
)
```

#### 3. Feedback Tags

Structured feedback tags enable systematic improvement:

| Tag | Purpose |
|-----|---------|
| `FALSE_POSITIVE` | Incorrect violation detection |
| `MISSED_VIOLATION` | Real violation that wasn't detected |
| `POLICY_GAP` | Policy doesn't cover this scenario |
| `CORRECT_DECISION` | Decision was appropriate |
| `NEEDS_CLARIFICATION` | Case needs more context |
| `EDGE_CASE` | Unusual scenario for documentation |

#### 4. SLA Tracking

Monitor review performance:

```python
metrics = queue.get_sla_metrics()

print(f"Total Cases: {metrics.total_cases}")
print(f"Pending: {metrics.pending_cases}")
print(f"Completed: {metrics.completed_cases}")
print(f"Median Triage Time: {metrics.median_triage_time_seconds}s")
print(f"Median Resolution Time: {metrics.median_resolution_time_seconds}s")
print(f"SLA Breaches: {metrics.sla_breaches}")
```

#### 5. Feedback Summary

Aggregate feedback for continuous improvement:

```python
summary = queue.get_feedback_summary()

print(f"False Positive Rate: {summary['false_positive_rate']:.1%}")
print(f"Missed Violation Rate: {summary['missed_violation_rate']:.1%}")
print(f"Policy Gap Rate: {summary['policy_gap_rate']:.1%}")
print(f"Tag Counts: {summary['tag_counts']}")
```

### CLI Tool

Phase 8 includes a command-line interface for reviewers:

```bash
# List pending cases
python cli/review_queue list

# Get next case for review
python cli/review_queue next reviewer_alice

# Submit feedback
python cli/review_queue feedback esc_abc123 reviewer_alice \
    --tags false_positive \
    --rationale "Content was safe" \
    --corrected-decision allow

# View SLA metrics
python cli/review_queue stats

# View feedback summary
python cli/review_queue summary
```

### Priority Levels

Cases are prioritized for review:

| Priority | Description | Example Trigger |
|----------|-------------|-----------------|
| `EMERGENCY` | Immediate attention required | Severity 5 violations |
| `CRITICAL` | High urgency | Severity 4 violations |
| `HIGH` | Important review needed | BLOCK/TERMINATE decisions |
| `MEDIUM` | Standard review | Low confidence with violations |
| `LOW` | Optional review | Edge cases |

### Storage

All escalation data is persisted in SQLite:

- `escalation_cases` table: Case metadata and status
- `human_feedback` table: Reviewer feedback and labels
- Indexed for performance on status, priority, and timestamps

---

## Phase 9: Continuous Optimization

### Architecture

Phase 9 implements multi-objective optimization:

```
Baseline Config → Optimization Search → Candidate Configs
                                              ↓
                                       Evaluate Metrics
                                              ↓
                                       Promotion Gate
                                              ↓
                                    Production Deployment
```

### Core Features

#### 1. Multi-Objective Optimization

Optimize for multiple objectives simultaneously:

```python
from nethical.core import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer(storage_path="./data/optimization.db")

# Default objectives (from TrainTestPipeline.md):
# - detection_recall (weight: 0.4, maximize)
# - false_positive_rate (weight: 0.25, minimize)
# - decision_latency (weight: 0.15, minimize)
# - human_agreement (weight: 0.2, maximize)
#
# Fitness = 0.4*recall - 0.25*fp_rate - 0.15*latency + 0.2*agreement
```

#### 2. Configuration Management

Create and track configurations:

```python
config = optimizer.create_configuration(
    config_version="v2.0",
    classifier_threshold=0.6,
    confidence_threshold=0.75,
    gray_zone_lower=0.35,
    gray_zone_upper=0.65,
    escalation_confidence_threshold=0.85,
    escalation_violation_count=2
)

# Record performance metrics
metrics = optimizer.record_metrics(
    config_id=config.config_id,
    detection_recall=0.85,
    detection_precision=0.88,
    false_positive_rate=0.06,
    decision_latency_ms=10.5,
    human_agreement=0.89,
    total_cases=500
)

print(f"Fitness Score: {metrics.fitness_score:.4f}")
```

#### 3. Optimization Techniques

##### Grid Search

Exhaustive search over discrete parameter values:

```python
results = optimizer.grid_search(
    param_grid={
        'classifier_threshold': [0.4, 0.5, 0.6, 0.7],
        'gray_zone_lower': [0.3, 0.4, 0.5],
        'gray_zone_upper': [0.5, 0.6, 0.7]
    },
    evaluate_fn=my_evaluation_function,
    max_iterations=50
)
```

##### Random Search

Random sampling from parameter ranges:

```python
results = optimizer.random_search(
    param_ranges={
        'classifier_threshold': (0.3, 0.7),
        'confidence_threshold': (0.5, 0.9),
        'gray_zone_lower': (0.2, 0.5),
        'gray_zone_upper': (0.5, 0.8)
    },
    evaluate_fn=my_evaluation_function,
    n_iterations=100
)
```

##### Evolutionary Search

Genetic algorithm-inspired search:

```python
results = optimizer.evolutionary_search(
    base_config=baseline_config,
    evaluate_fn=my_evaluation_function,
    population_size=30,
    n_generations=15,
    mutation_rate=0.2
)
```

#### 4. Promotion Gate

Validate candidates before production deployment:

```python
from nethical.core import PromotionGate

gate = PromotionGate(
    min_recall_gain=0.03,          # +3% absolute recall improvement
    max_fp_increase=0.02,          # Max +2% FP rate increase
    max_latency_increase_ms=5.0,   # Max +5ms latency
    min_human_agreement=0.85,      # Min 85% agreement
    min_sample_size=100            # Min 100 cases evaluated
)

# Check if candidate passes
passed, reasons = optimizer.check_promotion_gate(
    candidate_id=candidate.config_id,
    baseline_id=baseline.config_id
)

if passed:
    optimizer.promote_configuration(candidate.config_id)
else:
    print("Gate failed:", reasons)
```

#### 5. Configuration Status

Configurations move through lifecycle states:

| Status | Description |
|--------|-------------|
| `CANDIDATE` | New configuration being evaluated |
| `SHADOW` | Running in shadow mode (passive) |
| `PRODUCTION` | Active in production |
| `DEPRECATED` | Replaced by newer version |

---

## Integration

### Phase89IntegratedGovernance

Unified interface for both phases:

```python
from nethical.core import Phase89IntegratedGovernance, FeedbackTag

governance = Phase89IntegratedGovernance(
    storage_dir="./data",
    triage_sla_seconds=3600,
    resolution_sla_seconds=86400,
    auto_escalate_on_block=True,
    auto_escalate_on_low_confidence=True,
    low_confidence_threshold=0.7
)

# Phase 8: Process with escalation
result = governance.process_with_escalation(
    judgment_id="judg_123",
    action_id="act_456",
    agent_id="agent_789",
    decision="block",
    confidence=0.65,
    violations=[{"type": "safety", "severity": 4}]
)

# Phase 8: Human review
if result['escalated']:
    case = governance.get_next_case(reviewer_id="alice")
    feedback = governance.submit_feedback(
        case_id=case.case_id,
        reviewer_id="alice",
        feedback_tags=[FeedbackTag.CORRECT_DECISION],
        rationale="Decision was appropriate"
    )

# Phase 9: Optimization
results = governance.optimize_configuration(
    technique="random_search",
    n_iterations=50
)

# Phase 9: Promotion
best_config, best_metrics = results[0]
passed, reasons = governance.check_promotion_gate(
    candidate_id=best_config.config_id,
    baseline_id=baseline.config_id
)

# Continuous improvement cycle
cycle_result = governance.continuous_improvement_cycle()
print(f"Needs Optimization: {cycle_result['needs_optimization']}")
print(f"Recommendations: {cycle_result['recommendations']}")
```

---

## Best Practices

### 1. Escalation Strategy

**Automatic Escalation Triggers:**
- BLOCK/TERMINATE decisions
- Confidence < threshold (default 0.7)
- Critical/Emergency violations (severity ≥ 4)
- Multiple high-confidence violations (≥ 3)

**Priority Assignment:**
```python
def determine_priority(decision, confidence, violations):
    # Emergency for critical violations
    if any(v['severity'] >= 5 for v in violations):
        return ReviewPriority.EMERGENCY
    
    # High for blocking decisions
    if decision in ['block', 'terminate']:
        return ReviewPriority.HIGH
    
    # Medium for low confidence
    if confidence < 0.7 and violations:
        return ReviewPriority.MEDIUM
    
    return ReviewPriority.LOW
```

### 2. Review Workflow

**Efficient Review Process:**
1. Set up dedicated review sessions
2. Use CLI tool for rapid case processing
3. Provide detailed rationales for feedback
4. Use corrected decisions to improve training data
5. Track metadata for edge cases

**Feedback Quality:**
- Always provide clear rationale
- Use specific feedback tags
- Include corrected decision when applicable
- Add metadata for complex cases
- Maintain high confidence in feedback (≥0.8)

### 3. SLA Management

**Setting SLAs:**
```python
# Critical cases: Fast triage
governance = Phase89IntegratedGovernance(
    triage_sla_seconds=1800,     # 30 minutes for critical
    resolution_sla_seconds=14400  # 4 hours for critical
)

# Standard cases: Reasonable targets
governance = Phase89IntegratedGovernance(
    triage_sla_seconds=3600,      # 1 hour
    resolution_sla_seconds=86400  # 24 hours
)
```

**Monitoring:**
- Track SLA breaches regularly
- Alert on high pending case counts (>50)
- Review P95 metrics for outliers
- Adjust SLAs based on actual performance

### 4. Optimization Strategy

**Start Simple:**
1. Establish baseline configuration
2. Run random search (20-50 iterations)
3. Identify promising regions
4. Use grid search to fine-tune
5. Validate with promotion gate

**Evaluation Function:**
```python
def evaluate_config(config):
    """Evaluate configuration with real data."""
    # Deploy in shadow mode
    # Collect metrics over evaluation period
    # Calculate performance metrics
    return optimizer.record_metrics(
        config_id=config.config_id,
        detection_recall=measured_recall,
        detection_precision=measured_precision,
        false_positive_rate=measured_fp_rate,
        decision_latency_ms=measured_latency,
        human_agreement=measured_agreement,
        total_cases=sample_size
    )
```

**Promotion Gate Tuning:**
- Adjust thresholds based on risk tolerance
- Consider business impact of changes
- Require sufficient sample size (≥100 cases)
- Monitor production after promotion

### 5. Continuous Improvement

**Weekly Cycle:**
```python
def weekly_improvement():
    # 1. Collect feedback
    summary = governance.get_feedback_summary()
    
    # 2. Identify issues
    needs_optimization = (
        summary['false_positive_rate'] > 0.1 or
        summary['missed_violation_rate'] > 0.05
    )
    
    # 3. Optimize if needed
    if needs_optimization:
        results = governance.optimize_configuration(
            technique="random_search",
            n_iterations=50
        )
        
        # 4. Validate best candidate
        best_config, best_metrics = results[0]
        passed, reasons = governance.check_promotion_gate(
            candidate_id=best_config.config_id,
            baseline_id=current_baseline.config_id
        )
        
        # 5. Deploy if passes gate
        if passed:
            governance.promote_configuration(best_config.config_id)
```

---

## Examples

### Example 1: Basic Escalation Workflow

```python
from nethical.core import Phase89IntegratedGovernance, FeedbackTag

# Initialize
gov = Phase89IntegratedGovernance(storage_dir="./data")

# Process action
result = gov.process_with_escalation(
    judgment_id="j1",
    action_id="a1",
    agent_id="agent1",
    decision="block",
    confidence=0.6,
    violations=[{"type": "safety", "severity": 4}]
)

print(f"Escalated: {result['escalated']}")

# Review
case = gov.get_next_case(reviewer_id="alice")
if case:
    feedback = gov.submit_feedback(
        case_id=case.case_id,
        reviewer_id="alice",
        feedback_tags=[FeedbackTag.CORRECT_DECISION],
        rationale="Decision appropriate"
    )
```

### Example 2: Optimization Campaign

```python
from nethical.core import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer()

# Create baseline
baseline = optimizer.create_configuration(
    config_version="baseline_v1",
    classifier_threshold=0.5
)
baseline_metrics = optimizer.record_metrics(
    config_id=baseline.config_id,
    detection_recall=0.80,
    detection_precision=0.85,
    false_positive_rate=0.10,
    decision_latency_ms=15.0,
    human_agreement=0.85,
    total_cases=1000
)

# Optimize
results = optimizer.random_search(
    param_ranges={
        'classifier_threshold': (0.4, 0.7),
        'gray_zone_lower': (0.3, 0.5),
        'gray_zone_upper': (0.5, 0.7)
    },
    evaluate_fn=lambda c: evaluate_in_shadow(c),
    n_iterations=30
)

# Check best
best_config, best_metrics = results[0]
passed, reasons = optimizer.check_promotion_gate(
    candidate_id=best_config.config_id,
    baseline_id=baseline.config_id
)

if passed:
    optimizer.promote_configuration(best_config.config_id)
    print("✓ Promoted to production")
```

### Example 3: Full Integration

See `examples/phase89_demo.py` for complete working example.

---

## Summary

Phases 8-9 complete the Nethical governance system with:

- **Phase 8**: Human-in-the-loop operations with escalation queue, SLA tracking, and structured feedback
- **Phase 9**: Continuous optimization with multi-objective search and promotion gates

Together, these phases enable:
- Systematic human review of uncertain cases
- Data-driven configuration optimization
- Continuous improvement feedback loops
- Production-safe deployment of optimizations

For more information, see:
- [README.md](README.md) - Main documentation
- [roadmap.md](roadmap.md) - Project roadmap
- [TrainTestPipeline.md](TrainTestPipeline.md) - Training and testing details
