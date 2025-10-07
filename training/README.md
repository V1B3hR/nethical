# Model Training Scripts

This directory contains scripts for training machine learning models for the Nethical framework.

## train_any_model.py

A plug-and-play training pipeline that supports multiple model types with optional audit logging and ethical drift tracking.

### Features

- **Multiple Model Types**: Support for heuristic, logistic, anomaly, correlation, and transformer models
- **Kaggle Dataset Integration**: Automatically downloads and processes datasets from Kaggle
- **Synthetic Data Fallback**: Generates synthetic data when real datasets are unavailable
- **Promotion Gate**: Validates models against quality criteria before promotion
- **Audit Logging**: Optional Merkle tree-based audit trail for training events
- **Ethical Drift Tracking**: Track and analyze model performance across training cohorts
- **Performance Optimization**: Track and optimize CPU usage during training phases

### Usage

Basic training:
```bash
python training/train_any_model.py --model-type heuristic --epochs 10 --num-samples 1000
```

Training with audit logging:
```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 20 \
    --num-samples 2000 \
    --enable-audit \
    --audit-path training_logs
```

Training with ethical drift tracking:
```bash
python training/train_any_model.py \
    --model-type heuristic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-drift-tracking \
    --drift-report-dir drift_reports \
    --cohort-id cohort_alpha
```

Training with performance optimization tracking:
```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 20 \
    --num-samples 2000 \
    --enable-performance-optimization \
    --performance-target-reduction 30.0
```

Training with all features enabled:
```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 20 \
    --num-samples 2000 \
    --enable-audit \
    --enable-drift-tracking \
    --enable-performance-optimization
```

### Command-Line Arguments

- `--model-type`: Type of model to train (required)
  - Options: `heuristic`, `logistic`, `simple_transformer`, `anomaly`, `correlation`
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--num-samples`: Number of samples to use (default: 10000)
- `--seed`: Random seed for reproducibility (default: 42)

#### Audit Logging Options

- `--enable-audit`: Enable Merkle audit logging
- `--audit-path`: Path for audit logs (default: `training_audit_logs`)

#### Drift Tracking Options

- `--enable-drift-tracking`: Enable ethical drift tracking
- `--drift-report-dir`: Directory for drift reports (default: `training_drift_reports`)
- `--cohort-id`: Cohort identifier for drift tracking (default: `{model-type}_{timestamp}`)

#### Performance Optimization Options

- `--enable-performance-optimization`: Enable performance optimization tracking
- `--performance-target-reduction`: Target CPU reduction percentage (default: 30.0)

### Ethical Drift Tracking

Ethical drift tracking monitors model performance across different training cohorts to detect:

- **Performance Drift**: Variations in accuracy, precision, and recall across cohorts
- **Calibration Issues**: Differences in expected calibration error (ECE)
- **Promotion Gate Failures**: Violations when models fail quality criteria

#### How It Works

1. Each training run is assigned to a cohort (specified via `--cohort-id`)
2. During training:
   - Model performance metrics are tracked as actions with risk scores
   - Promotion gate failures are recorded as violations
3. At the end of training, a drift report is generated showing:
   - Cohort profiles with action counts and violation statistics
   - Drift metrics comparing performance across cohorts
   - Recommendations for addressing detected drift

#### Example Use Cases

- **A/B Testing**: Compare different model architectures or hyperparameters
- **Temporal Drift**: Monitor how model performance changes over time
- **Fairness Auditing**: Ensure consistent performance across different data cohorts
- **Model Monitoring**: Track model degradation across training iterations

#### Drift Report Structure

Each drift report (saved as JSON) contains:

```json
{
  "report_id": "drift_YYYYMMDD_HHMMSS",
  "start_time": "ISO 8601 timestamp",
  "end_time": "ISO 8601 timestamp",
  "cohorts": {
    "cohort_id": {
      "action_count": 1,
      "violation_stats": {
        "total_count": 0,
        "by_type": {},
        "by_severity": {}
      },
      "avg_risk_score": 0.1,
      "risk_distribution": {"low": 1}
    }
  },
  "drift_metrics": {
    "has_drift": false,
    "message": "Analysis results"
  },
  "recommendations": ["List of recommendations"]
}
```

### Performance Optimization Tracking

Performance optimization tracking monitors CPU usage during different training phases to identify optimization opportunities.

#### Tracked Training Phases

The performance optimizer tracks the following training phases:

- **Data Loading** (FAST tier): Time spent loading and preparing datasets
- **Preprocessing** (STANDARD tier): Time spent preprocessing and transforming data
- **Training** (ADVANCED tier): Time spent training the model
- **Validation** (STANDARD tier): Time spent validating model performance

#### How It Works

1. When enabled, the performance optimizer tracks CPU time for each training phase
2. Metrics are collected including:
   - Total CPU time per phase
   - Average CPU time per invocation
   - Number of invocations
3. At the end of training, a performance report is generated with:
   - Detailed timing metrics for each phase
   - Optimization suggestions based on performance data
   - Target achievement status (default: 30% CPU reduction)

#### Performance Report Structure

Performance reports are saved to `training_performance_reports/` as JSON files:

```json
{
  "timestamp": "ISO 8601 timestamp",
  "action_metrics": {
    "total_actions": 0,
    "total_cpu_time_ms": 0.0,
    "avg_cpu_time_ms": 0.0
  },
  "detector_stats": {
    "detectors": {
      "data_loading": {
        "tier": "fast",
        "total_invocations": 1,
        "total_cpu_time_ms": 150.5,
        "avg_cpu_time_ms": 150.5
      },
      "preprocessing": {
        "tier": "standard",
        "total_invocations": 1,
        "total_cpu_time_ms": 50.2,
        "avg_cpu_time_ms": 50.2
      },
      "training": {
        "tier": "advanced",
        "total_invocations": 1,
        "total_cpu_time_ms": 1200.8,
        "avg_cpu_time_ms": 1200.8
      },
      "validation": {
        "tier": "standard",
        "total_invocations": 1,
        "total_cpu_time_ms": 80.3,
        "avg_cpu_time_ms": 80.3
      }
    }
  },
  "optimization": {
    "baseline_cpu_ms": null,
    "baseline_established": false,
    "current_cpu_reduction_pct": 0.0,
    "target_cpu_reduction_pct": 30.0,
    "meeting_target": false
  }
}
```

#### Example Use Cases

- **Bottleneck Identification**: Identify which training phases consume the most CPU time
- **Optimization Verification**: Measure the impact of optimization changes
- **Resource Planning**: Understand resource requirements for different model types
- **Performance Regression Detection**: Track performance degradation over time

### Promotion Gate Criteria

Models must meet the following criteria to be promoted to production:

- **Maximum ECE**: ≤ 0.08 (Expected Calibration Error)
- **Minimum Accuracy**: ≥ 0.85

Models that pass the promotion gate are saved to `models/current/`, while those that fail are saved to `models/candidates/`.

### Examples

See `examples/train_with_drift_tracking.py` for a complete example of using drift tracking to compare multiple model cohorts.

### Testing

Run the test suite to verify functionality:

```bash
# Test basic training
python tests/test_train_model_real_data.py

# Test audit logging
python tests/test_train_audit_logging.py

# Test drift tracking
python tests/test_train_drift_tracking.py

# Test performance optimization
python tests/test_train_performance_optimization.py
```
