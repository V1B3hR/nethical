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
- **Integrated Governance**: Full governance system integration with all phases (3, 4, 5-7, 8-9)

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

Training with full IntegratedGovernance (all phases):
```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 20 \
    --num-samples 2000 \
    --enable-integrated-governance \
    --governance-storage-dir training_governance
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

#### Integrated Governance Options

- `--enable-integrated-governance`: Enable full IntegratedGovernance system (includes all governance phases)
- `--governance-storage-dir`: Storage directory for IntegratedGovernance data (default: `training_governance_data`)

**Note**: When `--enable-integrated-governance` is enabled, it supersedes individual `--enable-audit` and `--enable-drift-tracking` flags, as the IntegratedGovernance system includes all these features plus additional governance capabilities.

### Integrated Governance System

The Integrated Governance system provides a unified interface to all governance features across all phases (3, 4, 5-7, 8-9). When enabled, it automatically tracks and monitors training activities through a comprehensive governance framework.

#### What It Includes

**Phase 3: Risk & Correlation**
- Risk scoring for training events and model performance
- Correlation detection across training runs
- Fairness sampling and cohort management
- Ethical drift reporting with enhanced metrics
- Performance optimization tracking

**Phase 4: Audit & Taxonomy**
- Merkle tree-based immutable audit logs
- Policy change auditing
- Ethical taxonomy tagging for violations
- SLA monitoring for training processes

**Phase 5-7: ML-Assisted Governance**
- Shadow mode ML classifier for performance prediction
- Blended risk scoring combining rule-based and ML approaches
- Anomaly detection for training patterns
- Distribution drift monitoring

**Phase 8-9: Human Oversight & Optimization**
- Escalation queue for review cases
- Human feedback integration
- Multi-objective optimization
- Configuration management

#### How It Works

When `--enable-integrated-governance` is enabled:

1. **Initialization**: All governance components are initialized and configured
2. **Training Events**: Each training step is tracked via `process_action()`:
   - Data loading
   - Data splitting
   - Training completion
   - Validation metrics
   - Model saving
3. **ML Features**: Validation metrics are converted to ML features for shadow mode and blending
4. **Comprehensive Reporting**: At the end, a full system status report is generated showing activity across all phases

#### Benefits

- **Complete Audit Trail**: Every training event is logged with full traceability
- **Proactive Monitoring**: Automatic detection of anomalies and drift patterns
- **ML-Powered Insights**: Shadow models provide predictive insights on model quality
- **Centralized Management**: Single system manages all governance aspects
- **Production-Ready**: Same governance system used in training and production

#### Example Output

```
[INFO] IntegratedGovernance enabled. Data stored in: training_governance_data
[INFO] All governance phases (3, 4, 5-7, 8-9) are active.
...
[INFO] Governance System Status:
  Timestamp: 2025-10-07T12:02:22.839935
  Components enabled: 13/15

  Phase 3 (Risk & Correlation):
    Active risk profiles: 2
    Tracked agents: 2
    Meeting performance target: False

  Phase 4 (Audit & Taxonomy):
    Merkle events logged: 6

  Phase 5-7 (ML & Anomaly Detection):
    Shadow classifier enabled: True
    ML blending enabled: True
    Anomaly monitoring enabled: True

  Phase 8-9 (Human Oversight & Optimization):
    Pending escalation cases: 0
    Tracked configurations: 0

[INFO] Full governance data saved to: training_governance_data
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
