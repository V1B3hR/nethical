# Phase 5-7 Guide: ML & Anomaly Detection

This guide provides comprehensive documentation for Phases 5, 6, and 7 of the Nethical safety governance system.

## Table of Contents

1. [Overview](#overview)
2. [Phase 5: ML Shadow Mode](#phase-5-ml-shadow-mode)
3. [Phase 6: ML Blended Risk](#phase-6-ml-blended-risk)
4. [Phase 7: Anomaly & Drift Detection](#phase-7-anomaly--drift-detection)
5. [Integration](#integration)
6. [Best Practices](#best-practices)
7. [Examples](#examples)

---

## Overview

Phases 5-7 introduce machine learning and anomaly detection capabilities to enhance governance with data-driven insights:

- **Phase 5: ML Shadow Mode**: Passive ML predictions for validation without affecting decisions
- **Phase 6: ML Blended Risk**: ML-assisted decision making in gray-zone cases
- **Phase 7: Anomaly & Drift Detection**: Behavioral monitoring and distribution drift detection

These phases enable the system to learn from data, assist with borderline decisions, and detect unusual patterns that rule-based systems might miss.

### Key Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| `MLShadowClassifier` | Passive ML validation | Agreement tracking, metrics collection |
| `MLBlendedRiskEngine` | Gray-zone assistance | Risk zone detection, weighted blending |
| `AnomalyDriftMonitor` | Behavior monitoring | Sequence anomalies, distribution drift (PSI, KL) |
| `Phase567IntegratedGovernance` | Unified interface | All Phase 5-7 features combined |

---

## Phase 5: ML Shadow Mode

ML Shadow Mode runs machine learning models in parallel with rule-based systems to validate and improve models without affecting production decisions.

### Architecture

```
Action → Rule-Based Decision (PRIMARY)
      ↘
        ML Shadow Prediction (PASSIVE)
             ↓
      Agreement Tracking
             ↓
      Model Improvement
```

### Core Features

#### 1. Shadow Classifier Setup

Initialize shadow classifier for passive monitoring:

```python
from nethical.core import MLShadowClassifier, MLModelType

classifier = MLShadowClassifier(
    model_type=MLModelType.HEURISTIC,  # or BASELINE, TRAINED
    shadow_score_threshold=0.1,  # Agreement threshold
    storage_path="./shadow_data"
)
```

#### 2. Model Types

Three model types available:

| Type | Description | Use Case |
|------|-------------|----------|
| `HEURISTIC` | Simple rule-based baseline | Initial deployment |
| `BASELINE` | Basic ML model | Early validation |
| `TRAINED` | Fully trained model | Production candidate |

#### 3. Shadow Predictions

Run shadow predictions alongside rule-based decisions:

```python
# Rule-based decision (primary)
rule_risk_score = 0.65
rule_classification = "warn"

# Shadow ML prediction (passive)
shadow_result = classifier.predict_shadow(
    action_id="action_123",
    action_type="response",
    features={
        'violation_count': 0.5,
        'severity_max': 0.7,
        'recency_score': 0.6
    },
    rule_risk_score=rule_risk_score,
    rule_classification=rule_classification
)

# Shadow result does NOT affect decision
print(f"Rule Decision: {rule_classification} (score: {rule_risk_score:.2f})")
print(f"ML Shadow: {shadow_result['ml_classification']} (score: {shadow_result['ml_risk_score']:.2f})")
print(f"Agreement: {shadow_result['classifications_agree']}")
```

#### 4. Agreement Tracking

Track how well ML predictions match rule-based decisions:

```python
# Process multiple actions
for i in range(1000):
    shadow_result = classifier.predict_shadow(
        action_id=f"action_{i}",
        action_type="response",
        features=get_features(i),
        rule_risk_score=get_rule_score(i),
        rule_classification=get_rule_classification(i)
    )

# Get agreement metrics
metrics = classifier.get_shadow_metrics()

print(f"Total Predictions: {metrics['total_predictions']}")
print(f"Score Agreement Rate: {metrics['score_agreement_rate']:.1%}")
print(f"Classification Agreement Rate: {metrics['classification_agreement_rate']:.1%}")
print(f"Avg Score Difference: {metrics['avg_score_difference']:.3f}")
```

#### 5. Performance Metrics

Comprehensive ML performance tracking:

```python
metrics = classifier.get_shadow_metrics()

# Classification metrics
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")

# Confusion matrix
confusion = metrics['confusion_matrix']
print(f"True Positives: {confusion['tp']}")
print(f"False Positives: {confusion['fp']}")
print(f"True Negatives: {confusion['tn']}")
print(f"False Negatives: {confusion['fn']}")
```

#### 6. Disagreement Analysis

Identify cases where ML and rules disagree:

```python
# Get disagreements
disagreements = classifier.get_disagreements(limit=10)

for disagreement in disagreements:
    print(f"Action: {disagreement['action_id']}")
    print(f"  Rule: {disagreement['rule_classification']} ({disagreement['rule_risk_score']:.2f})")
    print(f"  ML: {disagreement['ml_classification']} ({disagreement['ml_risk_score']:.2f})")
    print(f"  Features: {disagreement['features']}")
    print()

# Use disagreements to improve model
train_on_disagreements(disagreements)
```

#### 7. Model Promotion

Promote shadow model to primary when ready:

```python
# Check if model is ready for promotion
metrics = classifier.get_shadow_metrics()

if (metrics['classification_agreement_rate'] > 0.95 and
    metrics['f1_score'] > 0.90 and
    metrics['total_predictions'] > 10000):
    
    print("✅ Model ready for promotion!")
    print(f"  Agreement: {metrics['classification_agreement_rate']:.1%}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  Predictions: {metrics['total_predictions']}")
    
    # Promote to primary
    promote_model_to_primary(classifier)
else:
    print("⚠️  Model needs more validation")
```

---

## Phase 6: ML Blended Risk

ML Blended Risk uses machine learning to assist with gray-zone decisions where rule-based systems are uncertain.

### Architecture

```
Action → Risk Score → Risk Zone Classification
                           ↓
         Clear Zone: Use Rules Only
         Gray Zone: Blend Rules + ML
                           ↓
                  Final Decision
```

### Core Features

#### 1. Risk Zones

Three risk zones with different strategies:

| Zone | Score Range | Strategy | ML Usage |
|------|-------------|----------|----------|
| `CLEAR_ALLOW` | < 0.4 | Rules only | No ML |
| `GRAY_ZONE` | 0.4 - 0.6 | Blend rules + ML | ML assists |
| `CLEAR_DENY` | > 0.6 | Rules only | No ML |

```python
from nethical.core import MLBlendedRiskEngine

engine = MLBlendedRiskEngine(
    gray_zone_lower=0.4,  # Gray zone starts at 0.4
    gray_zone_upper=0.6,  # Gray zone ends at 0.6
    rule_weight=0.7,      # 70% weight to rules
    ml_weight=0.3,        # 30% weight to ML
    storage_path="./blended_data"
)
```

#### 2. Blended Risk Calculation

Compute blended risk in gray zone:

```python
# Clear allow zone (0.3)
result = engine.calculate_blended_risk(
    agent_id="agent_1",
    action_id="action_1",
    action_type="response",
    features={'indicator': 0.3},
    rule_risk_score=0.3,
    rule_classification="allow"
)

print(f"Zone: {result['risk_zone']}")  # CLEAR_ALLOW
print(f"ML Influenced: {result['ml_influenced']}")  # False
print(f"Final Score: {result['blended_risk_score']:.2f}")  # 0.3 (rules only)

# Gray zone (0.5)
result = engine.calculate_blended_risk(
    agent_id="agent_2",
    action_id="action_2",
    action_type="response",
    features={'indicator': 0.5},
    rule_risk_score=0.5,
    rule_classification="warn"
)

print(f"Zone: {result['risk_zone']}")  # GRAY_ZONE
print(f"ML Influenced: {result['ml_influenced']}")  # True
print(f"Final Score: {result['blended_risk_score']:.2f}")  # Weighted average
```

#### 3. Weighted Blending

Combine rule and ML scores with configurable weights:

```python
# Formula: blended = (rule * rule_weight) + (ml * ml_weight)

# Conservative (trust rules more)
engine = MLBlendedRiskEngine(
    rule_weight=0.8,  # 80% rules
    ml_weight=0.2     # 20% ML
)

# Balanced
engine = MLBlendedRiskEngine(
    rule_weight=0.5,  # 50% rules
    ml_weight=0.5     # 50% ML
)

# ML-heavy (trust ML more)
engine = MLBlendedRiskEngine(
    rule_weight=0.3,  # 30% rules
    ml_weight=0.7     # 70% ML
)
```

#### 4. Classification Changes

Track when ML changes the classification:

```python
result = engine.calculate_blended_risk(
    agent_id="agent_1",
    action_id="action_1",
    action_type="response",
    features={'indicator': 0.45},
    rule_risk_score=0.55,  # Borderline warn
    rule_classification="warn"
)

if result['classification_changed']:
    print(f"⚠️  ML Changed Classification!")
    print(f"  Original: {result['original_classification']}")
    print(f"  Blended: {result['blended_classification']}")
    print(f"  Reason: ML score ({result['ml_risk_score']:.2f}) "
          f"influenced final score ({result['blended_risk_score']:.2f})")
```

#### 5. Blending Metrics

Monitor ML influence on decisions:

```python
metrics = engine.get_blending_metrics()

print(f"Total Decisions: {metrics['total_decisions']}")
print(f"Gray Zone Decisions: {metrics['gray_zone_count']}")
print(f"ML Influenced: {metrics['ml_influenced_count']}")
print(f"ML Influence Rate: {metrics['ml_influence_rate']:.1%}")
print(f"Classification Changes: {metrics['classification_changes']}")
```

#### 6. Zone Statistics

Understand decision distribution across zones:

```python
stats = engine.get_zone_statistics()

print("Decision Distribution:")
for zone, count in stats['zone_counts'].items():
    pct = (count / stats['total_decisions']) * 100
    print(f"  {zone}: {count} ({pct:.1f}%)")

# Typical output:
# CLEAR_ALLOW: 6500 (65%)
# GRAY_ZONE: 2000 (20%)
# CLEAR_DENY: 1500 (15%)
```

#### 7. Dynamic Threshold Adjustment

Adjust gray zone boundaries based on performance:

```python
# If too many false positives in gray zone
if false_positive_rate > 0.1:
    # Widen gray zone to get more ML assistance
    engine.update_thresholds(
        gray_zone_lower=0.35,  # Was 0.4
        gray_zone_upper=0.65   # Was 0.6
    )

# If ML is performing well
if ml_accuracy > 0.95:
    # Increase ML weight
    engine.update_weights(
        rule_weight=0.5,  # Was 0.7
        ml_weight=0.5     # Was 0.3
    )
```

---

## Phase 7: Anomaly & Drift Detection

Anomaly & Drift Detection identifies unusual behavioral patterns and detects when the data distribution changes over time.

### Architecture

```
Actions → Sequence Analysis → Anomaly Detection
      ↘
        Baseline Distribution → Drift Detection (PSI, KL)
                                      ↓
                              Alerts & Reports
```

### Core Features

#### 1. Anomaly Monitor Setup

Initialize anomaly and drift monitoring:

```python
from nethical.core import AnomalyDriftMonitor

monitor = AnomalyDriftMonitor(
    sequence_n=3,           # Track 3-action sequences
    psi_threshold=0.2,      # PSI drift threshold
    anomaly_sequence_threshold=0.7,  # Sequence anomaly threshold
    anomaly_drift_threshold=0.3,     # Drift alert threshold
    storage_path="./anomaly_data"
)
```

#### 2. Sequence Anomaly Detection

Detect unusual action sequences:

```python
# Track agent actions
actions = [
    "login", "view_profile", "logout",  # Normal sequence
    "login", "view_profile", "logout",  # Normal sequence
    "login", "delete_data", "delete_data", "delete_data"  # Anomalous!
]

for i, action in enumerate(actions):
    result = monitor.track_action(
        agent_id="agent_1",
        action_id=f"action_{i}",
        action_type=action,
        features={'indicator': 0.5},
        risk_score=0.5,
        cohort="production"
    )
    
    if result['anomalies']:
        for anomaly in result['anomalies']:
            print(f"⚠️  Anomaly detected: {anomaly['type']}")
            print(f"   Description: {anomaly['description']}")
            print(f"   Severity: {anomaly['severity']}")
```

#### 3. Anomaly Types

Four types of anomalies detected:

| Type | Description | Detection Method |
|------|-------------|-----------------|
| `SEQUENCE` | Unusual action patterns | N-gram frequency analysis |
| `BEHAVIORAL` | Out-of-character actions | Per-agent behavior baseline |
| `FREQUENCY` | Abnormal action rate | Rate threshold checking |
| `DRIFT` | Distribution shift | PSI and KL divergence |

#### 4. Baseline Distribution

Set baseline for drift detection:

```python
# Collect baseline data (e.g., from first month)
baseline_scores = []
for action in baseline_actions:
    risk_score = calculate_risk(action)
    baseline_scores.append(risk_score)

# Set baseline
monitor.set_baseline_distribution(
    baseline_scores,
    cohort="production"
)

print(f"✓ Baseline set with {len(baseline_scores)} samples")
```

#### 5. Drift Detection (PSI & KL)

Detect distribution shifts using statistical measures:

```python
# Process current actions
current_scores = []
for action in current_actions:
    risk_score = calculate_risk(action)
    current_scores.append(risk_score)
    
    result = monitor.track_action(
        agent_id=action.agent_id,
        action_id=action.id,
        action_type=action.type,
        features=action.features,
        risk_score=risk_score,
        cohort="production"
    )

# Calculate drift
drift_report = monitor.calculate_drift(
    current_scores,
    cohort="production"
)

print(f"PSI (Population Stability Index): {drift_report['psi']:.3f}")
print(f"KL Divergence: {drift_report['kl_divergence']:.3f}")

# Interpret drift
if drift_report['psi'] < 0.1:
    print("✅ No significant drift")
elif drift_report['psi'] < 0.2:
    print("⚠️  Slight drift detected")
else:
    print("❌ Significant drift - model may need retraining!")
```

**PSI (Population Stability Index) Interpretation:**
- < 0.1: No significant change
- 0.1 - 0.2: Slight change, monitor
- > 0.2: Significant change, action needed

#### 6. Behavioral Baseline

Track per-agent normal behavior:

```python
# Establish baseline for agent
for action in agent_history[:100]:
    monitor.track_action(
        agent_id="agent_1",
        action_id=action.id,
        action_type=action.type,
        features=action.features,
        risk_score=action.risk_score,
        cohort="production"
    )

# New action significantly different from baseline
result = monitor.track_action(
    agent_id="agent_1",
    action_id="new_action",
    action_type="admin_command",  # Never seen before!
    features={'indicator': 0.9},  # Much higher than usual
    risk_score=0.9,
    cohort="production"
)

if result['anomalies']:
    print("⚠️  Behavioral anomaly: Agent acting out of character")
```

#### 7. Anomaly Statistics

Get comprehensive anomaly reports:

```python
stats = monitor.get_anomaly_statistics(cohort="production")

print(f"Total Actions: {stats['total_actions']}")
print(f"Anomalies Detected: {stats['anomaly_count']}")
print(f"Anomaly Rate: {stats['anomaly_rate']:.1%}")

print("\nBy Type:")
for anomaly_type, count in stats['by_type'].items():
    print(f"  {anomaly_type}: {count}")

print("\nBy Severity:")
for severity, count in stats['by_severity'].items():
    print(f"  {severity}: {count}")
```

#### 8. Drift History

Track drift over time:

```python
# Get drift history
history = monitor.get_drift_history(cohort="production", days=30)

print("Drift History (Last 30 Days):")
for entry in history:
    print(f"{entry['date']}: PSI={entry['psi']:.3f}, KL={entry['kl']:.3f}")
    if entry['psi'] > 0.2:
        print("  ⚠️  Significant drift!")

# Plot drift trend
plot_drift_over_time(history)
```

---

## Integration

### Phase 5-7 Integrated Governance

The `Phase567IntegratedGovernance` class combines all Phase 5-7 components:

```python
from nethical.core import Phase567IntegratedGovernance

governance = Phase567IntegratedGovernance(
    storage_dir="./phase567_data",
    # Phase 5: Shadow Mode
    enable_shadow_mode=True,
    shadow_model_type=MLModelType.BASELINE,
    shadow_score_threshold=0.1,
    # Phase 6: Blended Risk
    enable_ml_blending=True,
    gray_zone_lower=0.4,
    gray_zone_upper=0.6,
    rule_weight=0.7,
    ml_weight=0.3,
    # Phase 7: Anomaly Detection
    enable_anomaly_detection=True,
    sequence_n=3,
    psi_threshold=0.2,
    anomaly_sequence_threshold=0.7,
    anomaly_drift_threshold=0.3
)

# Process action through all Phase 5-7 components
result = governance.process_action(
    agent_id="agent_123",
    action_id="action_456",
    action_type="response",
    features={
        'violation_count': 0.5,
        'severity_max': 0.6,
        'recency_score': 0.4
    },
    rule_risk_score=0.55,
    rule_classification="warn",
    cohort="production"
)

# Result includes all Phase 5-7 data
print(f"Shadow ML Score: {result['shadow']['ml_risk_score']:.2f}")
print(f"Blended Score: {result['blended']['blended_risk_score']:.2f}")
print(f"ML Influenced: {result['blended']['ml_influenced']}")
print(f"Anomalies: {len(result['anomalies'])}")
```

### Migration to IntegratedGovernance

Phase567IntegratedGovernance is deprecated. Use IntegratedGovernance instead:

```python
# Old (deprecated)
from nethical.core import Phase567IntegratedGovernance
governance = Phase567IntegratedGovernance(storage_dir="./data")

# New (recommended)
from nethical.core import IntegratedGovernance
governance = IntegratedGovernance(
    storage_dir="./data",
    enable_shadow_mode=True,
    enable_ml_blending=True,
    enable_anomaly_detection=True
)
```

### Progressive Rollout

Safely roll out ML features:

```python
# Stage 1: Shadow mode only (no impact)
governance = IntegratedGovernance(
    enable_shadow_mode=True,
    enable_ml_blending=False,
    enable_anomaly_detection=False
)

# Monitor shadow metrics for 2 weeks
# If agreement > 95%, proceed to Stage 2

# Stage 2: Add anomaly detection
governance = IntegratedGovernance(
    enable_shadow_mode=True,
    enable_ml_blending=False,
    enable_anomaly_detection=True
)

# Monitor anomaly detection for 1 week
# Tune thresholds, verify low false positives

# Stage 3: Enable ML blending (full deployment)
governance = IntegratedGovernance(
    enable_shadow_mode=True,
    enable_ml_blending=True,
    enable_anomaly_detection=True,
    rule_weight=0.8,  # Start conservative
    ml_weight=0.2
)

# Gradually increase ML weight as confidence grows
```

### Baseline Establishment

Set baselines for all cohorts:

```python
# Collect baseline data (first month)
baseline_data = collect_baseline_data(days=30)

for cohort, scores in baseline_data.items():
    governance.set_baseline_distribution(scores, cohort=cohort)
    print(f"✓ Baseline set for {cohort}: {len(scores)} samples")
```

---

## Best Practices

### 1. Shadow Mode Deployment

**Initial Deployment:**
- Start with shadow mode to validate ML without risk
- Collect 10,000+ predictions before evaluating
- Target 95%+ agreement rate before promotion

```python
# Monitor shadow mode
def monitor_shadow_performance():
    metrics = classifier.get_shadow_metrics()
    
    if metrics['total_predictions'] < 10000:
        return "Need more data"
    
    if metrics['classification_agreement_rate'] < 0.95:
        return "Model needs improvement"
    
    if metrics['f1_score'] < 0.90:
        return "Model performance insufficient"
    
    return "Ready for promotion"
```

**Disagreement Analysis:**
- Regularly review disagreements
- Use to improve model
- Identify edge cases for human review

```python
# Weekly disagreement review
disagreements = classifier.get_disagreements(limit=100)
for d in high_severity_disagreements(disagreements):
    review_with_human_expert(d)
    update_training_data(d)
```

### 2. Blended Risk Configuration

**Weight Selection:**
- Start conservative (0.8 rules, 0.2 ML)
- Gradually increase ML weight as confidence grows
- Never exceed 0.5 ML weight without extensive validation

```python
# Progressive weight adjustment
if ml_performance_excellent():
    engine.update_weights(rule_weight=0.6, ml_weight=0.4)
    monitor_for_issues(days=7)
    
    if no_issues_detected():
        engine.update_weights(rule_weight=0.5, ml_weight=0.5)
```

**Gray Zone Sizing:**
- Too narrow: ML rarely used
- Too wide: ML overused, risky
- Optimal: 15-25% of decisions in gray zone

```python
# Check gray zone usage
stats = engine.get_zone_statistics()
gray_pct = (stats['gray_zone_count'] / stats['total_decisions']) * 100

if gray_pct < 15:
    # Widen gray zone
    engine.update_thresholds(0.35, 0.65)
elif gray_pct > 25:
    # Narrow gray zone
    engine.update_thresholds(0.45, 0.55)
```

### 3. Anomaly Detection Tuning

**Threshold Selection:**
- Start with conservative thresholds (low false positives)
- Gradually tighten to catch more anomalies
- Balance detection vs. false positive rate

```python
# Initial (conservative)
monitor = AnomalyDriftMonitor(
    anomaly_sequence_threshold=0.8,  # High threshold
    anomaly_drift_threshold=0.4
)

# After tuning (balanced)
monitor = AnomalyDriftMonitor(
    anomaly_sequence_threshold=0.7,  # Moderate
    anomaly_drift_threshold=0.3
)
```

**Baseline Refresh:**
- Refresh baselines quarterly or when drift detected
- Use representative data (multiple weeks)
- Avoid biased samples (holidays, incidents)

```python
# Quarterly baseline refresh
def refresh_baselines():
    for cohort in ['production', 'staging', 'development']:
        # Collect last 30 days, excluding incidents
        scores = collect_normal_period_data(cohort, days=30)
        monitor.set_baseline_distribution(scores, cohort=cohort)
        print(f"✓ Refreshed {cohort} baseline")
```

### 4. Drift Monitoring

**PSI Thresholds:**
- < 0.1: Normal variation, no action
- 0.1-0.2: Monitor closely, investigate trends
- > 0.2: Alert, likely model retraining needed

```python
def check_drift():
    drift = monitor.calculate_drift(recent_scores, cohort="production")
    
    if drift['psi'] > 0.2:
        alert_data_science_team(
            severity="high",
            message=f"Significant drift detected: PSI={drift['psi']:.3f}"
        )
        initiate_model_retraining()
    elif drift['psi'] > 0.1:
        alert_data_science_team(
            severity="medium",
            message=f"Moderate drift detected: PSI={drift['psi']:.3f}"
        )
```

**Drift Investigation:**
- Check for data quality issues
- Verify no system changes
- Look for real behavioral changes
- Consider seasonality

### 5. Performance Optimization

**Resource Management:**
- ML operations are more expensive than rules
- Use feature caching where possible
- Batch process when appropriate

```python
# Cache feature extraction
from functools import lru_cache

@lru_cache(maxsize=1000)
def extract_features(action_id):
    # Expensive feature extraction
    return compute_features(action_id)

# Batch processing for non-realtime
def batch_process_anomalies(actions):
    results = []
    for action in actions:
        result = monitor.track_action(**action)
        results.append(result)
    return results
```

### 6. Monitoring and Alerting

**Key Metrics to Monitor:**
- Shadow mode agreement rate
- ML influence rate in blending
- Anomaly detection rate
- Drift metrics (PSI, KL)
- Model performance (precision, recall, F1)

```python
def daily_health_check():
    """Run daily health checks on ML components."""
    
    # Shadow mode
    shadow_metrics = governance.get_shadow_metrics()
    if shadow_metrics['classification_agreement_rate'] < 0.90:
        alert("Shadow mode agreement below threshold")
    
    # Blending
    blend_metrics = governance.get_blending_metrics()
    if blend_metrics['ml_influence_rate'] > 0.5:
        alert("ML influence rate unusually high")
    
    # Anomaly detection
    anomaly_stats = governance.get_anomaly_statistics()
    if anomaly_stats['anomaly_rate'] > 0.1:
        alert("High anomaly rate detected")
    
    # Drift
    for cohort in ['production', 'staging']:
        drift = governance.calculate_drift(cohort=cohort)
        if drift['psi'] > 0.2:
            alert(f"Significant drift in {cohort}")
```

---

## Examples

### Example 1: Shadow Mode Validation

```python
from nethical.core import MLShadowClassifier, MLModelType

# Initialize shadow classifier
classifier = MLShadowClassifier(
    model_type=MLModelType.BASELINE,
    storage_path="./shadow_validation"
)

print("Starting shadow mode validation...\n")

# Simulate 1000 actions
for i in range(1000):
    # Get rule-based decision
    rule_score = random.uniform(0.0, 1.0)
    rule_class = "allow" if rule_score < 0.5 else "deny"
    
    # Get shadow prediction
    result = classifier.predict_shadow(
        action_id=f"action_{i}",
        action_type="response",
        features={'indicator': rule_score},
        rule_risk_score=rule_score,
        rule_classification=rule_class
    )
    
    # Check every 100 actions
    if (i + 1) % 100 == 0:
        metrics = classifier.get_shadow_metrics()
        print(f"[{i+1}/1000] Agreement: {metrics['classification_agreement_rate']:.1%}, "
              f"F1: {metrics['f1_score']:.3f}")

# Final evaluation
print("\n" + "="*50)
print("SHADOW MODE VALIDATION RESULTS")
print("="*50)

metrics = classifier.get_shadow_metrics()
print(f"\nTotal Predictions: {metrics['total_predictions']}")
print(f"Score Agreement: {metrics['score_agreement_rate']:.1%}")
print(f"Classification Agreement: {metrics['classification_agreement_rate']:.1%}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")

# Promotion decision
if (metrics['classification_agreement_rate'] > 0.95 and
    metrics['f1_score'] > 0.90):
    print("\n✅ Model ready for promotion to production!")
else:
    print("\n⚠️  Model needs more training")
    
    # Analyze disagreements
    disagreements = classifier.get_disagreements(limit=10)
    print(f"\nTop {len(disagreements)} disagreements:")
    for d in disagreements:
        print(f"  {d['action_id']}: Rule={d['rule_classification']}, "
              f"ML={d['ml_classification']}")
```

### Example 2: Gray Zone ML Blending

```python
from nethical.core import MLBlendedRiskEngine

engine = MLBlendedRiskEngine(
    gray_zone_lower=0.4,
    gray_zone_upper=0.6,
    rule_weight=0.7,
    ml_weight=0.3,
    storage_path="./blending_demo"
)

print("ML Blended Risk Demo\n")

# Test different risk scores
test_scores = [0.2, 0.4, 0.5, 0.6, 0.8]

for score in test_scores:
    result = engine.calculate_blended_risk(
        agent_id="test_agent",
        action_id=f"action_{int(score*100)}",
        action_type="response",
        features={'risk': score},
        rule_risk_score=score,
        rule_classification="warn" if score > 0.5 else "allow"
    )
    
    print(f"Rule Score: {score:.2f}")
    print(f"  Zone: {result['risk_zone']}")
    print(f"  Blended Score: {result['blended_risk_score']:.2f}")
    print(f"  ML Influenced: {'✅' if result['ml_influenced'] else '❌'}")
    if result['classification_changed']:
        print(f"  ⚠️  Classification Changed: {result['original_classification']} → "
              f"{result['blended_classification']}")
    print()

# Get statistics
print("="*50)
metrics = engine.get_blending_metrics()
print(f"Total Decisions: {metrics['total_decisions']}")
print(f"ML Influenced: {metrics['ml_influenced_count']} "
      f"({metrics['ml_influence_rate']:.1%})")

stats = engine.get_zone_statistics()
print("\nDecision Distribution:")
for zone, count in stats['zone_counts'].items():
    pct = (count / stats['total_decisions']) * 100 if stats['total_decisions'] > 0 else 0
    print(f"  {zone}: {count} ({pct:.1f}%)")
```

### Example 3: Anomaly Detection

```python
from nethical.core import AnomalyDriftMonitor

monitor = AnomalyDriftMonitor(
    sequence_n=3,
    anomaly_sequence_threshold=0.7,
    storage_path="./anomaly_demo"
)

print("Anomaly Detection Demo\n")

# Establish normal pattern
print("Training on normal behavior...")
normal_actions = ["login", "view", "edit", "save", "logout"] * 20
for i, action in enumerate(normal_actions):
    monitor.track_action(
        agent_id="normal_agent",
        action_id=f"normal_{i}",
        action_type=action,
        features={'indicator': 0.3},
        risk_score=0.3,
        cohort="production"
    )
print(f"✓ Trained on {len(normal_actions)} normal actions\n")

# Test anomalous sequence
print("Testing anomalous behavior...")
anomalous_actions = ["login", "delete", "delete", "delete", "delete"]
for i, action in enumerate(anomalous_actions):
    result = monitor.track_action(
        agent_id="suspicious_agent",
        action_id=f"suspicious_{i}",
        action_type=action,
        features={'indicator': 0.8},
        risk_score=0.8,
        cohort="production"
    )
    
    print(f"Action {i+1}: {action}")
    if result['anomalies']:
        for anomaly in result['anomalies']:
            print(f"  ⚠️  {anomaly['type']}: {anomaly['description']}")
            print(f"      Severity: {anomaly['severity']}, "
                  f"Confidence: {anomaly['confidence']:.2f}")
    else:
        print("  ✅ No anomalies")
    print()

# Get statistics
stats = monitor.get_anomaly_statistics(cohort="production")
print("="*50)
print(f"Anomaly Statistics:")
print(f"  Total Actions: {stats['total_actions']}")
print(f"  Anomalies: {stats['anomaly_count']}")
print(f"  Anomaly Rate: {stats['anomaly_rate']:.1%}")
```

### Example 4: Drift Detection

```python
from nethical.core import AnomalyDriftMonitor
import random

monitor = AnomalyDriftMonitor(
    psi_threshold=0.2,
    storage_path="./drift_demo"
)

print("Drift Detection Demo\n")

# Set baseline (normal distribution)
print("Establishing baseline distribution...")
baseline_scores = [random.gauss(0.3, 0.1) for _ in range(1000)]
baseline_scores = [max(0, min(1, s)) for s in baseline_scores]  # Clamp to [0, 1]
monitor.set_baseline_distribution(baseline_scores, cohort="production")
print(f"✓ Baseline: mean={sum(baseline_scores)/len(baseline_scores):.2f}\n")

# Test scenarios
scenarios = [
    ("No Drift", [random.gauss(0.3, 0.1) for _ in range(1000)]),
    ("Slight Drift", [random.gauss(0.35, 0.1) for _ in range(1000)]),
    ("Significant Drift", [random.gauss(0.5, 0.15) for _ in range(1000)])
]

for name, current_scores in scenarios:
    # Clamp scores
    current_scores = [max(0, min(1, s)) for s in current_scores]
    
    # Calculate drift
    drift = monitor.calculate_drift(current_scores, cohort="production")
    
    print(f"{name}:")
    print(f"  Current mean: {sum(current_scores)/len(current_scores):.2f}")
    print(f"  PSI: {drift['psi']:.3f}")
    print(f"  KL Divergence: {drift['kl_divergence']:.3f}")
    
    # Interpretation
    if drift['psi'] < 0.1:
        print("  ✅ No significant drift")
    elif drift['psi'] < 0.2:
        print("  ⚠️  Slight drift - monitor closely")
    else:
        print("  ❌ Significant drift - action needed!")
    print()
```

### Example 5: Complete Phase 5-7 Workflow

```python
from nethical.core import Phase567IntegratedGovernance, MLModelType

# Initialize
governance = Phase567IntegratedGovernance(
    storage_dir="./phase567_demo",
    enable_shadow_mode=True,
    enable_ml_blending=True,
    enable_anomaly_detection=True,
    gray_zone_lower=0.4,
    gray_zone_upper=0.6
)

print("Phase 5-7 Integrated Demo\n")

# Set baseline for drift detection
print("Setting baseline...")
baseline = [random.uniform(0.2, 0.4) for _ in range(1000)]
governance.set_baseline_distribution(baseline, cohort="production")
print("✓ Baseline set\n")

# Process various actions
test_cases = [
    ("Normal Low Risk", 0.3, "allow"),
    ("Gray Zone - Rules Lean Allow", 0.45, "warn"),
    ("Gray Zone - Rules Lean Deny", 0.55, "warn"),
    ("Clear High Risk", 0.8, "deny")
]

for name, rule_score, classification in test_cases:
    print(f"{name}:")
    
    result = governance.process_action(
        agent_id="test_agent",
        action_id=f"action_{int(rule_score*100)}",
        action_type="response",
        features={'indicator': rule_score},
        rule_risk_score=rule_score,
        rule_classification=classification,
        cohort="production"
    )
    
    # Shadow mode
    if 'shadow' in result:
        shadow = result['shadow']
        print(f"  Shadow ML: {shadow['ml_risk_score']:.2f} "
              f"(agree: {'✅' if shadow['scores_agree'] else '❌'})")
    
    # Blended risk
    if 'blended' in result:
        blended = result['blended']
        print(f"  Blended: {blended['blended_risk_score']:.2f} "
              f"(zone: {blended['risk_zone']})")
        if blended['ml_influenced']:
            print(f"    ML influenced decision")
    
    # Anomalies
    if result['anomalies']:
        print(f"  ⚠️  {len(result['anomalies'])} anomalies detected")
    else:
        print(f"  ✅ No anomalies")
    
    print()

# Get comprehensive report
print("="*50)
print("SYSTEM REPORT")
print("="*50)

# Shadow metrics
shadow_metrics = governance.get_shadow_metrics()
print(f"\nShadow Mode:")
print(f"  Predictions: {shadow_metrics.get('total_predictions', 0)}")
print(f"  Agreement: {shadow_metrics.get('classification_agreement_rate', 0):.1%}")

# Blending metrics
blend_metrics = governance.get_blending_metrics()
print(f"\nML Blending:")
print(f"  Total Decisions: {blend_metrics.get('total_decisions', 0)}")
print(f"  ML Influenced: {blend_metrics.get('ml_influenced_count', 0)}")

# Anomaly stats
anomaly_stats = governance.get_anomaly_statistics(cohort="production")
print(f"\nAnomaly Detection:")
print(f"  Total Actions: {anomaly_stats['total_actions']}")
print(f"  Anomaly Rate: {anomaly_stats['anomaly_rate']:.1%}")

# System status
status = governance.get_system_status()
print(f"\nSystem Status:")
for component, info in status['components'].items():
    enabled = "✅" if info.get('enabled') else "❌"
    print(f"  {enabled} {component}")
```

---

## API Reference

### Phase567IntegratedGovernance

Main integration class combining all Phase 5-7 components.

**Methods:**

- `process_action(agent_id, action_id, action_type, features, rule_risk_score, ...)`: Process action through all components
- `set_baseline_distribution(scores, cohort)`: Set baseline for drift detection
- `calculate_drift(current_scores, cohort)`: Calculate drift from baseline
- `get_shadow_metrics()`: Get shadow mode performance metrics
- `get_blending_metrics()`: Get ML blending statistics
- `get_anomaly_statistics(cohort)`: Get anomaly detection stats
- `get_system_status()`: Get system-wide status
- `export_report(output_path)`: Export comprehensive report

### See Also

- [Phase 3 Guide](PHASE3_GUIDE.md) - Advanced Governance Features
- [Phase 4 Guide](PHASE4_GUIDE.md) - Integrity & Ethics Operationalization
- [Phase 8-9 Guide](PHASE89_GUIDE.md) - Human-in-the-Loop & Optimization
- [Main README](../../README.md) - Project overview

---

**Last Updated**: November 5, 2025  
**Version**: Phase 5-7 Complete  
**Status**: Production Ready
