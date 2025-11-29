# Model Training Scripts

This directory contains scripts for training machine learning models for the Nethical framework.

## train_any_model.py

A plug-and-play training pipeline that supports multiple model types with optional audit logging, ethical drift tracking, and governance validation.

### Features

- **Multiple Model Types**: Support for heuristic, logistic, anomaly, correlation, and transformer models
- **Train All Models**: Use `--model-type all` to train all model types in a single run
- **Kaggle Dataset Integration**: Automatically downloads and processes datasets from Kaggle
- **Synthetic Data Fallback**: Generates synthetic data when real datasets are unavailable
- **Promotion Gate**: Validates models against quality criteria before promotion
- **Audit Logging**: Optional Merkle tree-based audit trail for training events
- **Ethical Drift Tracking**: Track and analyze model performance across training cohorts
- **Governance Validation**: Real-time safety and ethical validation of training data and predictions

### Usage

Basic training:
```bash
python training/train_any_model.py --model-type heuristic --epochs 10 --num-samples 1000
```

**Training all model types at once:**
```bash
python training/train_any_model.py --model-type all --epochs 10 --num-samples 1000
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

Training all models with full monitoring:
```bash
python training/train_any_model.py \
    --model-type all \
    --epochs 20 \
    --num-samples 2000 \
    --enable-audit \
    --enable-governance \
    --enable-drift-tracking \
    --drift-report-dir drift_reports
```

Training with governance validation:
```bash
python training/train_any_model.py \
    --model-type heuristic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-governance
```

Training with audit logging, governance, and drift tracking:
```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 20 \
    --num-samples 2000 \
    --enable-audit \
    --enable-governance \
    --enable-drift-tracking \
    --drift-report-dir drift_reports \
    --cohort-id cohort_alpha
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

### Command-Line Arguments

- `--model-type`: Type of model to train (required)
  - Options: `heuristic`, `logistic`, `simple_transformer`, `anomaly`, `correlation`
  - Use `all` to train all model types sequentially
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--num-samples`: Number of samples to use (default: 10000)
- `--seed`: Random seed for reproducibility (default: 42)

#### Audit Logging Options

- `--enable-audit`: Enable Merkle audit logging
- `--audit-path`: Path for audit logs (default: `training_audit_logs`)

#### Governance Validation Options

- `--enable-governance`: Enable governance validation during training

#### Drift Tracking Options

- `--enable-drift-tracking`: Enable ethical drift tracking
- `--drift-report-dir`: Directory for drift reports (default: `training_drift_reports`)
- `--cohort-id`: Cohort identifier for drift tracking (default: `{model-type}_{timestamp}`)

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

### Governance Validation

Governance validation provides real-time safety and ethical checks during model training. When enabled with `--enable-governance`, the system validates:

#### Training Data Validation

- Checks first 100 training samples for safety violations
- Detects harmful content, toxic language, and malicious patterns
- Reports data samples that should be quarantined or blocked

#### Model Prediction Validation

- Validates first 50 model predictions during the validation phase
- Ensures model outputs don't contain safety violations
- Flags predictions that would be blocked in production

#### Detected Violation Types

The governance system checks for:

- **Ethical Violations**: Harmful content, bias, discrimination
- **Safety Violations**: Dangerous commands, unsafe domains, privilege escalation
- **Manipulation**: Social engineering, phishing, emotional leverage
- **Dark Patterns**: NLP manipulation, weaponized empathy, dependency creation
- **Privacy Issues**: PII exposure (emails, phone numbers, credit cards, SSN, IP addresses)
- **Security Issues**: Prompt injection, adversarial attacks, obfuscation
- **Cognitive Warfare**: Reality distortion, psychological manipulation
- **Toxic Content**: Offensive language, hate speech
- **Model Extraction**: Suspicious probing patterns

#### Governance Output

Example governance validation output:

```
[INFO] Governance validation enabled
[INFO] Running governance validation on training data samples...
[WARN] Governance found 5 problematic data samples
[INFO] Running governance validation on model predictions...
[INFO] Governance validation passed for 50 predictions

[INFO] Governance Validation Summary:
  Data samples validated: 100
  Data violations found: 5
  Predictions validated: 50
  Prediction violations found: 0
```

#### Integration with Audit Logging

When both `--enable-governance` and `--enable-audit` are used, governance metrics are included in the audit summary:

```json
{
  "merkle_root": "abc123...",
  "model_type": "heuristic",
  "promoted": true,
  "metrics": {...},
  "governance": {
    "enabled": true,
    "data_violations": 5,
    "prediction_violations": 0,
    "total_violations_detected": 5,
    "total_actions_blocked": 0
  }
}
```

This provides a complete audit trail showing both model performance and safety compliance.

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
