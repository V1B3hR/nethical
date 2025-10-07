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
- **ML Blended Risk Evaluation**: Evaluate models using blended rule-based and ML risk scoring

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

Training with ML blended risk evaluation:
```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-blended-risk \
    --gray-zone-lower 0.4 \
    --gray-zone-upper 0.6 \
    --rule-weight 0.7 \
    --ml-weight 0.3
```

Training with all features enabled:
```bash
python training/train_any_model.py \
    --model-type heuristic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-audit \
    --enable-drift-tracking \
    --enable-blended-risk
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

#### Blended Risk Evaluation Options

- `--enable-blended-risk`: Enable ML blended risk evaluation during validation
- `--blended-risk-path`: Path for blended risk logs (default: `training_blended_risk`)
- `--gray-zone-lower`: Lower bound of gray zone for risk blending (default: 0.4)
- `--gray-zone-upper`: Upper bound of gray zone for risk blending (default: 0.6)
- `--rule-weight`: Weight for rule-based risk in blending (default: 0.7)
- `--ml-weight`: Weight for ML risk in blending (default: 0.3)

### ML Blended Risk Evaluation

ML Blended Risk Evaluation combines rule-based and ML-based risk scoring to evaluate model predictions during validation. This helps assess how well the model would perform in a blended enforcement scenario.

#### How It Works

1. During validation, for each sample:
   - A rule-based risk score is computed (simulated from features)
   - An ML risk score is obtained from the model prediction
   - A blended risk score is computed: `blended = rule_weight * rule_score + ml_weight * ml_score`

2. Risk zones are determined based on rule scores:
   - **Clear Allow Zone**: rule_score < gray_zone_lower (default: < 0.4)
   - **Gray Zone**: gray_zone_lower ≤ rule_score ≤ gray_zone_upper (default: 0.4-0.6)
   - **Clear Deny Zone**: rule_score > gray_zone_upper (default: > 0.6)

3. ML influence is applied only in the gray zone, where decisions are uncertain

4. Metrics tracked include:
   - Zone distribution (how many decisions fall in each zone)
   - ML influence rate (percentage of gray zone decisions influenced by ML)
   - Classification changes (when blended score changes the decision)
   - False positive delta (FP increase/decrease with blending)
   - Detection improvement (true positive gains)

#### Gate Check Criteria

The blended risk evaluation includes a promotion gate check:

- **Maximum FP Delta**: ≤ 5% (blended mode shouldn't increase false positives significantly)
- **Detection Improvement**: > 0 (blended mode should improve true positive detection)
- **Minimum Gray Zone Samples**: ≥ 100 (sufficient data for reliable evaluation)

#### Example Output

```
[INFO] Blended Risk Metrics:
  Total decisions: 40
  Gray zone percentage: 20.00%
  ML influence rate: 100.00%
  Classification changes: 2

[INFO] Blended Risk Gate Check:
  FP delta: 1 (5.00%)
  Detection improvement: 3 (15.00%)
  Gate result: PASS - All gate checks passed
```

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
