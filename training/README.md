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
- **Continuous Optimization**: Multi-objective optimization with advanced promotion gates

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

Training with continuous optimization:
```bash
# Create a baseline configuration
python training/train_any_model.py \
    --model-type logistic \
    --epochs 20 \
    --num-samples 2000 \
    --enable-optimization \
    --optimization-db data/optimization.db

# Train a candidate and compare against baseline
python training/train_any_model.py \
    --model-type logistic \
    --epochs 20 \
    --num-samples 2000 \
    --enable-optimization \
    --optimization-db data/optimization.db \
    --baseline-config-id cfg_abc123
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

#### Optimization Options

- `--enable-optimization`: Enable continuous optimization with advanced promotion gates
- `--optimization-db`: Path to optimization database (default: `data/optimization.db`)
- `--baseline-config-id`: Baseline configuration ID for promotion gate comparison

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

### Continuous Optimization

The continuous optimization feature integrates the multi-objective optimizer from Phase 9 to enable:

- **Advanced Promotion Gates**: Multi-objective criteria including recall gain, FP rate, latency, and human agreement
- **Configuration Tracking**: Persistent storage of model configurations and performance metrics
- **A/B Testing Support**: Compare candidate models against baseline configurations
- **Production Safety**: Only promote models that meet strict quality criteria

#### How It Works

1. **Create a Baseline**: Run training with `--enable-optimization` to create a baseline configuration
2. **Train Candidates**: Train new models with `--baseline-config-id` to compare against the baseline
3. **Promotion Gate**: The optimizer evaluates candidates using multi-objective criteria:
   - Minimum recall gain: +3% (absolute)
   - Maximum FP rate increase: +2% (absolute)
   - Maximum latency increase: +5ms
   - Minimum human agreement: 85%
   - Minimum sample size: 100 cases

#### Advanced Promotion Gate Criteria

When optimization is enabled, models are evaluated against more rigorous criteria than the simple promotion gate:

| Criterion | Threshold | Description |
|-----------|-----------|-------------|
| Recall Gain | +3% | Candidate must improve recall by at least 3% over baseline |
| FP Rate Increase | +2% | False positive rate cannot increase by more than 2% |
| Latency Increase | +5ms | Decision latency cannot increase by more than 5ms |
| Human Agreement | 85% | Model decisions must align with human judgments at least 85% of the time |
| Sample Size | 100 | Minimum number of validation cases required |

#### Optimization Workflow

```bash
# Step 1: Create a baseline configuration
python training/train_any_model.py \
    --model-type logistic \
    --epochs 20 \
    --num-samples 5000 \
    --seed 42 \
    --enable-optimization \
    --optimization-db data/optimization.db

# Output: Recorded metrics for configuration: cfg_abc123

# Step 2: Train a candidate model with different parameters
python training/train_any_model.py \
    --model-type logistic \
    --epochs 30 \
    --num-samples 5000 \
    --seed 43 \
    --enable-optimization \
    --optimization-db data/optimization.db \
    --baseline-config-id cfg_abc123

# Output: Promotion Gate Results with detailed comparison
```

#### Benefits

- **Data-Driven Decisions**: Objective comparison of model performance
- **Risk Mitigation**: Prevents degradation of production models
- **Continuous Improvement**: Systematic approach to model optimization
- **Audit Trail**: Complete history of configurations and their performance

### Promotion Gate Criteria

#### Simple Promotion Gate (Default)

When optimization is not enabled, models are evaluated using a simple threshold-based promotion gate:

- **Maximum ECE**: ≤ 0.08 (Expected Calibration Error)
- **Minimum Accuracy**: ≥ 0.85

#### Advanced Promotion Gate (With Optimization)

When `--enable-optimization` is used with `--baseline-config-id`, models are evaluated using multi-objective criteria:

- **Recall Gain**: Must improve by ≥ +3% over baseline
- **FP Rate Increase**: Must not increase by > +2% over baseline
- **Latency Increase**: Must not increase by > +5ms over baseline
- **Human Agreement**: Must be ≥ 85%
- **Sample Size**: Must have ≥ 100 validation cases

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

# Test optimization integration
python tests/test_train_optimization.py
```
