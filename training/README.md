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
- **SLA Monitoring**: Track and validate performance SLAs during training

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

Training with SLA monitoring:
```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-sla-monitoring \
    --sla-report-dir sla_reports
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

#### SLA Monitoring Options

- `--enable-sla-monitoring`: Enable SLA performance monitoring
- `--sla-report-dir`: Directory for SLA reports (default: `training_sla_reports`)

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

### SLA Monitoring

SLA monitoring tracks performance metrics during training to ensure training operations meet service level agreements:

- **P95 Latency Tracking**: Monitor 95th percentile latency for training operations
- **P99 Latency Tracking**: Monitor 99th percentile latency for training operations
- **Average Latency Tracking**: Monitor average latency across all operations
- **Compliance Validation**: Verify that performance meets defined SLA targets

#### How It Works

1. When SLA monitoring is enabled, the training script tracks latency for key operations:
   - Data loading
   - Data preprocessing
   - Model training
   - Validation
2. At the end of training, an SLA report is generated with:
   - Current performance metrics (P50, P95, P99, average latency)
   - SLA compliance status for each target
   - Performance margins and breach information
3. An SLA documentation file is also generated with:
   - Performance targets and descriptions
   - Current performance status
   - Compliance guarantees

#### SLA Targets

Default SLA targets for training operations:

- **P95 Latency**: ≤ 220ms
- **P99 Latency**: ≤ 500ms
- **Average Latency**: ≤ 100ms

#### SLA Report Structure

Each SLA report (saved as JSON) contains:

```json
{
  "timestamp": "ISO 8601 timestamp",
  "overall_status": "compliant|warning|breach",
  "sla_met": true,
  "p95_latency_ms": 0.37,
  "p95_target_ms": 220.0,
  "metrics": {
    "p50_latency_ms": 0.074,
    "p95_latency_ms": 0.37,
    "p99_latency_ms": 0.37,
    "avg_latency_ms": 0.14,
    "sample_count": 4
  },
  "targets": {
    "p95_latency": {
      "target": 220.0,
      "actual": 0.37,
      "status": "compliant",
      "description": "95th percentile latency"
    }
  }
}
```

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

# Test SLA monitoring
python tests/test_train_sla_monitoring.py
```
