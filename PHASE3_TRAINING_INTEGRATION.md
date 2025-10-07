# Phase3 Integration in Model Training

This document describes the Phase3 Integrated Governance integration in `train_any_model.py`.

## Overview

The `train_any_model.py` script now supports Phase3 Integrated Governance, which provides comprehensive ethical oversight during model training. This integration brings together five powerful Phase 3 components:

1. **Risk Engine** - Multi-factor risk scoring and tier management
2. **Correlation Engine** - Multi-agent pattern detection
3. **Fairness Sampler** - Stratified sampling across cohorts
4. **Ethical Drift Reporter** - Cohort-based drift analysis
5. **Performance Optimizer** - Risk-based detector gating

## Usage

### Basic Phase3 Training

Enable Phase3 integration with the `--enable-phase3` flag:

```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-phase3
```

### Advanced Configuration

Specify a custom storage directory for Phase3 components:

```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-phase3 \
    --phase3-storage-dir my_phase3_data \
    --cohort-id production_v2
```

### Backward Compatibility

The traditional drift tracking mode is still available and works independently:

```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-drift-tracking \
    --drift-report-dir my_drift_reports
```

## Command-Line Arguments

### Phase3-Specific Arguments

- `--enable-phase3`: Enable Phase3 Integrated Governance (default: False)
- `--phase3-storage-dir`: Storage directory for Phase3 components (default: "nethical_training_data")
- `--cohort-id`: Cohort identifier for tracking (default: "{model-type}_{timestamp}")

### Traditional Drift Tracking Arguments

- `--enable-drift-tracking`: Enable standalone drift tracking (default: False)
- `--drift-report-dir`: Directory for drift reports (default: "training_drift_reports")

## What Gets Tracked

### With Phase3 Integration

When Phase3 is enabled, the following metrics are tracked:

1. **Risk Scores**
   - Model accuracy converted to risk score (1.0 - accuracy)
   - Risk tier assignment (LOW, NORMAL, HIGH, ELEVATED)
   - Risk profile history for the model

2. **Action Tracking**
   - Training validation actions tracked through governance
   - Actions include model type, accuracy, and metadata
   - Correlation patterns detected across multiple training runs

3. **Violations**
   - Calibration errors (ECE > 0.08)
   - Low accuracy (< 0.85)
   - Severity levels: high (ECE > 0.15 or accuracy < 0.70), medium (otherwise)

4. **Performance Metrics**
   - CPU time tracking for detectors
   - Performance optimization suggestions
   - Target achievement status (30% CPU reduction)

5. **Drift Analysis**
   - Cohort-based drift detection
   - Risk distribution analysis
   - Violation statistics by type and severity

### Output Structure

Phase3 creates the following directory structure:

```
{phase3-storage-dir}/
├── drift_reports/
│   └── drift_*.json           # Drift reports
├── fairness_samples/
│   └── job_*.json             # Fairness sampling jobs
└── [other Phase3 component data]
```

## Example Output

### Phase3 System Status

```
[INFO] Phase3 System Status:
  ✓ risk_engine: {'active_profiles': 1, 'enabled': True}
  ✓ correlation_engine: {'tracked_agents': 1, 'enabled': True}
  ✓ fairness_sampler: {'active_jobs': 0, 'enabled': True}
  ✓ ethical_drift_reporter: {'tracked_cohorts': 1, 'enabled': True}
  ✓ performance_optimizer: {'enabled': True, 'meeting_target': False}
```

### Drift Report Structure

```json
{
  "report_id": "drift_20251007_120309",
  "start_time": "2025-09-30T12:03:09.123456",
  "end_time": "2025-10-07T12:03:09.123456",
  "cohorts": {
    "production_v1": {
      "cohort_id": "production_v1",
      "agent_count": 0,
      "action_count": 1,
      "violation_stats": {
        "total_count": 1,
        "by_severity": {"medium": 1},
        "by_type": {"low_accuracy": 1},
        "by_time": []
      },
      "avg_risk_score": 0.305,
      "risk_distribution": {
        "low": 0,
        "normal": 1,
        "high": 0
      }
    }
  },
  "drift_metrics": {
    "has_drift": false,
    "message": "Insufficient cohorts for drift analysis"
  },
  "recommendations": [
    "No significant ethical drift detected across cohorts"
  ],
  "generated_at": "2025-10-07T12:03:09.123456"
}
```

## Use Cases

### 1. Production Model Training

Track model quality and ethical compliance during production training:

```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 50 \
    --num-samples 10000 \
    --enable-phase3 \
    --cohort-id production_v3 \
    --phase3-storage-dir production_governance
```

### 2. Experimental Training Runs

Compare multiple training configurations:

```bash
# Configuration A
python training/train_any_model.py \
    --model-type heuristic \
    --epochs 10 \
    --num-samples 1000 \
    --seed 42 \
    --enable-phase3 \
    --cohort-id experiment_a

# Configuration B
python training/train_any_model.py \
    --model-type logistic \
    --epochs 10 \
    --num-samples 1000 \
    --seed 42 \
    --enable-phase3 \
    --cohort-id experiment_b
```

Then analyze drift between cohorts using the generated reports.

### 3. Multi-Model Training Pipeline

Track drift across different model types:

```bash
for model_type in heuristic logistic anomaly correlation; do
    python training/train_any_model.py \
        --model-type $model_type \
        --epochs 20 \
        --num-samples 5000 \
        --enable-phase3 \
        --cohort-id pipeline_${model_type}
done
```

## Benefits Over Traditional Drift Tracking

| Feature | Traditional Drift | Phase3 Integration |
|---------|-------------------|-------------------|
| Drift monitoring | ✓ | ✓ |
| Risk scoring | ✗ | ✓ |
| Correlation detection | ✗ | ✓ |
| Performance optimization | ✗ | ✓ |
| Fairness sampling | ✗ | ✓ |
| System-wide context | ✗ | ✓ |
| Adaptive detection | ✗ | ✓ |

## Testing

Run the Phase3 integration tests:

```bash
python tests/test_train_phase3_integration.py
```

Run the full test suite including traditional drift tracking:

```bash
python tests/test_train_drift_tracking.py
python tests/test_train_phase3_integration.py
```

## Examples

See the following examples for practical usage:

- `examples/train_with_phase3.py` - Comprehensive Phase3 training example
- `examples/train_with_drift_tracking.py` - Traditional drift tracking example

## Migration Guide

### From Traditional Drift Tracking to Phase3

**Before:**
```bash
python training/train_any_model.py \
    --model-type logistic \
    --enable-drift-tracking \
    --drift-report-dir my_reports
```

**After:**
```bash
python training/train_any_model.py \
    --model-type logistic \
    --enable-phase3 \
    --phase3-storage-dir my_phase3_data
```

The Phase3 drift reports will be stored in `my_phase3_data/drift_reports/`.

### Keeping Both Modes

You can still use traditional drift tracking alongside Phase3:

```bash
# Use Phase3 for production
python training/train_any_model.py \
    --model-type logistic \
    --enable-phase3 \
    --cohort-id production

# Use traditional for quick experiments
python training/train_any_model.py \
    --model-type heuristic \
    --enable-drift-tracking \
    --cohort-id experiment
```

## Troubleshooting

### Phase3 Not Available

If you see:
```
[WARN] Phase3IntegratedGovernance not available. Phase3 integration will be disabled.
```

Ensure the Phase3 components are properly installed:
```bash
pip install -e .
```

### Storage Directory Issues

If Phase3 data is not being saved, check:
1. The `--phase3-storage-dir` path is writable
2. No permission issues with the directory
3. Sufficient disk space available

### Drift Report Not Generated

If no drift report is generated:
1. Ensure Phase3 is enabled with `--enable-phase3`
2. Check that training completed successfully
3. Look for error messages in the output

## See Also

- [PHASE3_INTEGRATION.md](PHASE3_INTEGRATION.md) - Phase3 components overview
- [DRIFT_TRACKING_IMPLEMENTATION.md](DRIFT_TRACKING_IMPLEMENTATION.md) - Drift tracking details
- [examples/phase3_demo.py](examples/phase3_demo.py) - Phase3 component demonstrations
