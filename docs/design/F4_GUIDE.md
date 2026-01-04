# F4: Thresholds, Tuning & Adaptivity Guide

## Overview

F4 introduces advanced threshold management capabilities to the Nethical governance system:

- **Bayesian Optimization**: Intelligent parameter tuning using probabilistic models
- **Adaptive Thresholds**: Real-time threshold adjustment based on feedback
- **A/B Testing Framework**: Statistical testing for threshold variants
- **Agent-Specific Profiles**: Customized thresholds for different agents

## Components

### 1. Bayesian Optimization

The Bayesian optimization implementation uses a simplified Gaussian Process approach to intelligently explore the parameter space.

```python
from nethical.core import Phase89IntegratedGovernance

governance = Phase89IntegratedGovernance(storage_dir="./data")

# Run Bayesian optimization
results = governance.optimize_configuration(
    technique="bayesian",
    param_ranges={
        'classifier_threshold': (0.4, 0.7),
        'confidence_threshold': (0.6, 0.9),
        'gray_zone_lower': (0.2, 0.5),
        'gray_zone_upper': (0.5, 0.8)
    },
    n_iterations=30,
    n_initial_random=5
)

# Get best configuration
best_config, best_metrics = results[0]
print(f"Best fitness: {best_metrics.fitness_score:.3f}")
print(f"Classifier threshold: {best_config.classifier_threshold:.3f}")
```

**How it works:**
1. **Initial Exploration**: Starts with random sampling to explore the space
2. **Adaptive Sampling**: Uses Gaussian perturbations around best points
3. **Exploitation-Exploration Balance**: Variance decreases over iterations
4. **Result Ranking**: Configurations sorted by fitness score

### 2. Adaptive Threshold Tuner

The `AdaptiveThresholdTuner` learns from outcome feedback and automatically adjusts thresholds.

```python
# Record outcomes for learning
result = governance.record_outcome(
    action_id="act_123",
    judgment_id="judg_456",
    predicted_outcome="block",
    actual_outcome="false_positive",  # Human correction
    confidence=0.85,
    human_feedback="User should have been allowed"
)

# Get updated thresholds
thresholds = result['updated_thresholds']
print(f"Classifier threshold: {thresholds['classifier_threshold']:.3f}")

# Get performance statistics
stats = governance.get_tuning_performance()
print(f"Accuracy: {stats['accuracy']:.1%}")
print(f"Precision: {stats['precision']:.1%}")
print(f"Recall: {stats['recall']:.1%}")
print(f"False positive rate: {stats['false_positive_rate']:.1%}")
```

**Outcome Types:**
- `"correct"` or `"true_positive"`: Correct blocking decision
- `"true_negative"`: Correct allow decision
- `"false_positive"`: Incorrectly blocked (should have been allowed)
- `"false_negative"`: Incorrectly allowed (should have been blocked)

**Learning Behavior:**
- High false positive rate → Increases classifier threshold
- High false negative rate → Decreases classifier threshold
- Low confidence errors → Increases confidence threshold

### 3. Agent-Specific Threshold Profiles

Different agents may require different thresholds based on their risk profiles.

```python
# Set custom thresholds for a high-risk agent
governance.set_agent_thresholds(
    agent_id="agent_financial_bot",
    thresholds={
        'classifier_threshold': 0.35,  # More sensitive
        'confidence_threshold': 0.85,   # Higher confidence required
        'gray_zone_lower': 0.3,
        'gray_zone_upper': 0.7
    }
)

# Set lenient thresholds for a low-risk agent
governance.set_agent_thresholds(
    agent_id="agent_help_desk",
    thresholds={
        'classifier_threshold': 0.65,  # Less sensitive
        'confidence_threshold': 0.70,   # Lower confidence OK
        'gray_zone_lower': 0.4,
        'gray_zone_upper': 0.6
    }
)

# Get agent-specific thresholds
thresholds = governance.get_adaptive_thresholds(agent_id="agent_financial_bot")
print(f"Financial bot threshold: {thresholds['classifier_threshold']:.2f}")
```

### 4. A/B Testing Framework

Test threshold variants with statistical rigor before full deployment.

```python
# Create baseline and candidate configurations
baseline_config = governance.create_configuration(
    config_version="baseline_v1.0",
    classifier_threshold=0.5,
    confidence_threshold=0.7
)

candidate_config = governance.create_configuration(
    config_version="candidate_v1.1",
    classifier_threshold=0.55,  # Slightly more conservative
    confidence_threshold=0.75
)

# Start A/B test with 10% traffic to candidate
control_id, treatment_id = governance.create_ab_test(
    baseline_config,
    candidate_config,
    traffic_split=0.1
)

# Record metrics for both variants (after collecting data)
baseline_metrics = PerformanceMetrics(
    config_id=baseline_config.config_id,
    detection_recall=0.82,
    false_positive_rate=0.08,
    human_agreement=0.86,
    total_cases=500
)

candidate_metrics = PerformanceMetrics(
    config_id=candidate_config.config_id,
    detection_recall=0.85,  # 3% improvement
    false_positive_rate=0.06,  # 2% reduction
    human_agreement=0.88,
    total_cases=50
)

governance.record_ab_metrics(control_id, baseline_metrics)
governance.record_ab_metrics(treatment_id, candidate_metrics)

# Check statistical significance
is_significant, p_value, interpretation = governance.check_ab_significance(
    control_id,
    treatment_id,
    metric="detection_recall"
)

print(interpretation)
# Output: "Treatment is +3.7% vs control. Significant (p=0.0234)"

if is_significant:
    # Gradually increase traffic to candidate
    new_traffic = governance.gradual_rollout(
        treatment_id,
        target_traffic=0.5,
        step_size=0.2
    )
    print(f"Traffic increased to {new_traffic:.0%}")
else:
    print("Not significant yet - continue collecting data")
```

**Gradual Rollout Strategy:**
1. Start with small traffic split (5-10%)
2. Monitor metrics closely
3. Check statistical significance periodically
4. Gradually increase traffic in steps (10-20%)
5. Full rollout only after validation

**Rollback Mechanism:**
```python
# Rollback if performance degrades
success = governance.rollback_variant(treatment_id)
if success:
    print("Rolled back to baseline configuration")
```

### 5. Complete Workflow Example

```python
from nethical.core import Phase89IntegratedGovernance, PerformanceMetrics

# Initialize governance with adaptive features
governance = Phase89IntegratedGovernance(
    storage_dir="./production_data",
    triage_sla_seconds=3600,
    resolution_sla_seconds=86400
)

# Step 1: Run Bayesian optimization to find good candidates
print("Running Bayesian optimization...")
results = governance.optimize_configuration(
    technique="bayesian",
    param_ranges={
        'classifier_threshold': (0.4, 0.7),
        'confidence_threshold': (0.6, 0.9)
    },
    n_iterations=30
)

# Step 2: Create A/B test with top candidate
baseline = governance.create_configuration(
    config_version="production_v2.0",
    classifier_threshold=0.5
)

best_candidate = results[0][0]  # Top result from optimization
control_id, treatment_id = governance.create_ab_test(
    baseline,
    best_candidate,
    traffic_split=0.1
)

# Step 3: Collect data and record outcomes
# (In production, this would be automatic)
for action_id, decision in process_actions():
    # Get human feedback on decision
    actual_outcome = get_human_feedback(action_id)
    
    # Record for adaptive learning
    governance.record_outcome(
        action_id=action_id,
        judgment_id=f"judg_{action_id}",
        predicted_outcome=decision,
        actual_outcome=actual_outcome,
        confidence=0.85
    )

# Step 4: Check A/B test results
summary = governance.get_ab_summary()
print(f"Control traffic: {summary['variants'][control_id]['traffic']:.1%}")
print(f"Treatment traffic: {summary['variants'][treatment_id]['traffic']:.1%}")

is_sig, p_value, interpretation = governance.check_ab_significance(
    control_id, treatment_id, metric="detection_recall"
)
print(interpretation)

# Step 5: Gradual rollout if successful
if is_sig and p_value < 0.05:
    for target in [0.2, 0.4, 0.6, 0.8, 1.0]:
        governance.gradual_rollout(treatment_id, target, step_size=0.2)
        print(f"Rolled out to {target:.0%}")
        # Monitor and validate at each step
        time.sleep(86400)  # Wait 24h between steps

# Step 6: Check adaptive tuning performance
stats = governance.get_tuning_performance()
print(f"\nAdaptive Tuning Performance:")
print(f"  Accuracy: {stats['accuracy']:.1%}")
print(f"  Precision: {stats['precision']:.1%}")
print(f"  Recall: {stats['recall']:.1%}")
print(f"  FP Rate: {stats['false_positive_rate']:.1%}")
```

## Best Practices

### 1. Bayesian Optimization

- **Initial Random Samples**: Use 5-10 random samples to explore space
- **Iterations**: 20-50 iterations typically sufficient
- **Parameter Ranges**: Keep ranges reasonable (e.g., 0.3-0.7 for thresholds)
- **Evaluation Function**: Use real metrics when possible, not simulated

### 2. Adaptive Tuning

- **Learning Rate**: Start with 0.01, adjust based on stability
- **Objectives**: Choose 2-3 primary objectives (e.g., maximize_recall, minimize_fp)
- **Feedback Loop**: Record outcomes consistently for all decisions
- **Monitoring**: Track performance stats regularly

### 3. A/B Testing

- **Sample Size**: Minimum 100 samples per variant for significance
- **Traffic Split**: Start small (5-10%) for new candidates
- **Significance Level**: Use 0.05 (95% confidence) as standard
- **Rollout Steps**: Increase by 10-20% per step, not larger jumps
- **Monitoring Window**: Wait 24-48 hours between rollout steps

### 4. Agent-Specific Profiles

- **Risk-Based Tuning**: Higher risk agents → lower thresholds (more sensitive)
- **Context-Aware**: Consider agent domain, user base, criticality
- **Regular Review**: Update profiles based on performance data
- **Default Fallback**: Always have global thresholds as fallback

## Performance Considerations

### Computational Cost

- **Bayesian Optimization**: O(n²) where n = iterations
- **Adaptive Tuning**: O(1) per outcome
- **A/B Testing**: O(1) per metric update
- **Statistical Tests**: O(1) computation

### Storage Requirements

- **Outcomes**: ~200 bytes per outcome record
- **Configurations**: ~500 bytes per configuration
- **A/B Tests**: ~1KB per test with metrics
- **Agent Profiles**: ~300 bytes per agent

### Scalability

- **Concurrent A/B Tests**: Support multiple simultaneous tests
- **Agent Profiles**: Efficiently handles 1000+ agent profiles
- **Outcome History**: Automatically prunes old records
- **Database**: SQLite for small-medium scale, consider PostgreSQL for large scale

## Troubleshooting

### Issue: Thresholds oscillating

**Cause**: Learning rate too high
**Solution**: Reduce learning_rate to 0.005 or 0.001

### Issue: A/B test not showing significance

**Cause**: Insufficient sample size or small effect
**Solution**: Collect more data or increase effect size

### Issue: Bayesian optimization not converging

**Cause**: Parameter ranges too wide or evaluation function issues
**Solution**: Narrow parameter ranges, validate evaluation function

### Issue: Agent-specific thresholds not applying

**Cause**: Agent ID mismatch
**Solution**: Verify agent_id matches exactly (case-sensitive)

## API Reference

See inline documentation in:
- `nethical.core.optimization.AdaptiveThresholdTuner`
- `nethical.core.optimization.ABTestingFramework`
- `nethical.core.phase89_integration.Phase89IntegratedGovernance`

## Migration Guide

If upgrading from Phase 8-9:

```python
# Old way
governance = Phase89IntegratedGovernance(storage_dir="./data")

# New way - same API, F4 features automatically available
governance = Phase89IntegratedGovernance(storage_dir="./data")

# New methods available:
# - record_outcome()
# - get_adaptive_thresholds()
# - set_agent_thresholds()
# - create_ab_test()
# - optimize_configuration(technique="bayesian")
```

No breaking changes - all Phase 8-9 APIs remain compatible.

## See Also

- [PHASE89_GUIDE.md](PHASE89_GUIDE.md) - Phase 8-9 fundamentals
- [roadmap.md](roadmap.md) - F4 implementation details
- [tests/test_f4_adaptive_tuning.py](tests/test_f4_adaptive_tuning.py) - Test examples
