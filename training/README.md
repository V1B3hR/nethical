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
- **Phase 8-9 Governance Integration**: Human-in-the-loop review and continuous optimization with configuration management

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

Training with Phase 8-9 governance and configuration optimization:
```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 20 \
    --num-samples 2000 \
    --enable-phase89 \
    --optimize-config \
    --optimization-technique random_search \
    --optimization-iterations 50
```

Complete training pipeline with all features:
```bash
python training/train_any_model.py \
    --model-type heuristic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-audit \
    --enable-drift-tracking \
    --enable-phase89 \
    --optimize-config \
    --optimization-iterations 20
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

#### Phase 8-9 Governance Options

- `--enable-phase89`: Enable Phase 8-9 governance integration
- `--phase89-storage`: Storage directory for Phase89 data (default: `training_governance_data`)
- `--optimize-config`: Run configuration optimization using Phase89
- `--optimization-technique`: Optimization technique to use (default: `random_search`)
  - Options: `random_search`, `grid_search`, `evolutionary`
- `--optimization-iterations`: Number of optimization iterations (default: 20)

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

### Phase 8-9 Governance Integration

Phase 8-9 governance provides human-in-the-loop review and continuous optimization capabilities for model training:

#### Features

- **Configuration Management**: Track model hyperparameters and training configurations
- **Metrics Recording**: Record comprehensive performance metrics with fitness scoring
- **Promotion Gate Validation**: Validate candidate configurations against baseline using multi-criteria assessment
- **Configuration Optimization**: Automatically search for optimal configurations using:
  - **Random Search**: Efficient exploration of hyperparameter space
  - **Grid Search**: Systematic evaluation of discrete parameter combinations
  - **Evolutionary Search**: Genetic algorithm-based optimization

#### How It Works

1. **Configuration Creation**: When `--enable-phase89` is enabled, the training script creates a Phase89 configuration with:
   - Model type, epochs, batch size, and other training parameters
   - Default classifier and confidence thresholds
   - Gray zone boundaries for uncertainty handling

2. **Metrics Recording**: Training metrics are mapped to Phase89 format:
   - Detection recall/precision from model recall/precision
   - False positive rate (approximated from precision)
   - Decision latency (inference time)
   - Human agreement (from feedback, if available)
   - Fitness score (weighted combination of metrics)

3. **Configuration Optimization** (when `--optimize-config` is enabled):
   - Multiple candidate configurations are generated and evaluated
   - Each configuration is scored using the Phase89 fitness function
   - Top configurations are ranked by fitness score
   - Best candidate is checked against promotion gate criteria

4. **Promotion Gate**: Validates candidates using:
   - Recall improvement (≥3% gain required)
   - False positive rate stability (≤2% increase allowed)
   - Latency constraint (≤5ms increase allowed)
   - Human agreement threshold (≥85% required)

#### Example Output

```
[INFO] Recording configuration with Phase89 governance...
[INFO] Configuration ID: cfg_3b58929a13a6
[INFO] Metrics recorded. Fitness score: 0.5133

[INFO] Running configuration optimization using random_search...
[INFO] Optimization completed. Evaluated 20 configurations

[INFO] Top 3 Configurations:
  1. random_v15
     Fitness: 0.5368
     Recall: 0.949, Precision: 0.887
     FP Rate: 0.049
  2. random_v3
     Fitness: 0.5335
     Recall: 0.932, Precision: 0.810
     FP Rate: 0.073
  3. random_v7
     Fitness: 0.5291
     Recall: 0.918, Precision: 0.709
     FP Rate: 0.016

[INFO] Checking promotion gate for best configuration...
[INFO] Promotion gate: PASSED
  - ✓ Recall gain: 0.049 >= 0.03
  - ✓ FP increase: 0.012 <= 0.02
  - ✓ Latency increase: 2.1ms <= 5.0ms
  - ✓ Human agreement: 0.895 >= 0.85
[INFO] Configuration promoted successfully
```

#### Benefits

- **Automated Optimization**: Systematically search for better configurations without manual tuning
- **Promotion Safety**: Prevent regressions by validating against multiple criteria
- **Reproducibility**: All configurations and metrics are tracked in persistent storage
- **Integration**: Seamlessly combines with audit logging and drift tracking for comprehensive governance

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
```
