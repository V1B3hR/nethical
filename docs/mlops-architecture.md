# MLOps Architecture

## Overview

The Nethical MLOps pipeline provides automated training, monitoring, and deployment of machine learning models with built-in governance, audit trails, and drift detection. The architecture is designed for continuous model improvement while maintaining strict ethical and safety standards.

## Architecture Components

### 1. Training Pipeline (`.github/workflows/ml-training.yml`)

**Purpose**: Automated model training with quality gates and performance tracking

**Triggers**:
- Manual dispatch (with configurable parameters)
- Weekly schedule (Sunday 2 AM UTC)
- Code changes to training/, nethical/mlops/, or datasets/datasets

**Key Features**:
- Multi-model training (all model types in one run)
- Baseline comparison and regression detection
- Automatic promotion to production
- Comprehensive metadata generation
- Audit trail integration

**Workflow Steps**:
1. **Setup**: Install dependencies, create directories
2. **Training**: Execute train_any_model.py with full governance
3. **Performance Tracking**: Compare with baseline models
4. **Regression Detection**: Create GitHub issues if performance degrades
5. **Deployment**: Deploy promoted models to staging/production

**Outputs**:
- Trained models (artifacts)
- Performance metrics
- Audit logs
- Drift reports

### 2. Model Monitoring (`.github/workflows/model-monitoring.yml`)

**Purpose**: Continuous monitoring of production models for drift and degradation

**Schedule**: Every 6 hours

**Key Features**:
- Automated drift detection
- Performance comparison with baseline
- Alert generation via GitHub issues
- Auto-trigger retraining on drift

**Monitoring Metrics**:
- Prediction distribution drift
- Accuracy degradation
- Calibration error changes
- Positive rate shifts

**Alert Thresholds** (configurable in training-schedule.json):
- Drift score > 0.15: Trigger alert and retraining
- Accuracy drop > 0.05: Create high-priority issue

### 3. Dataset Validation (`.github/workflows/dataset-validation.yml`)

**Purpose**: Validate training data freshness and quality

**Schedule**: Weekly (Monday 8 AM UTC)

**Validation Checks**:
1. **Freshness**: Dataset age < 30 days (configurable)
2. **Quality**: Minimum sample count (10,000 default)
3. **Distribution**: Class balance and feature distributions
4. **Missing Values**: < 20% missing data

**Outputs**:
- Validation reports
- GitHub issues for stale or problematic datasets

### 4. Model Deployment (scripts/deploy_model.py)

**Purpose**: Safe deployment with validation gates

**Deployment Modes**:
- **Shadow**: Stage models without routing traffic
- **Canary**: Gradual rollout (10% traffic default)
- **Full**: Complete deployment (100% traffic)

**Validation Gates**:
1. Model file integrity check
2. Metadata validation
3. Performance threshold verification
4. Loadability test

**Safety Features**:
- Pre-deployment validation
- Automatic rollback on failures
- Audit trail for all deployments

## Model Lifecycle

```
┌─────────────────┐
│  Code Changes   │
│   or Schedule   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│   Training Pipeline     │
│  - Load/generate data   │
│  - Train models         │
│  - Run governance       │
│  - Generate audit logs  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Performance Tracking   │
│  - Compare baseline     │
│  - Detect regression    │
│  - Update history       │
└────────┬────────────────┘
         │
         ▼
    ┌───┴────┐
    │ Passed │
    └───┬────┘
        │ Yes
        ▼
┌─────────────────────────┐
│  Model Promotion        │
│  - Save to current/     │
│  - Export metadata      │
│  - Archive old models   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Deployment             │
│  - Shadow mode          │
│  - Validation tests     │
│  - Canary rollout       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Monitoring             │
│  - Every 6 hours        │
│  - Drift detection      │
│  - Alert on issues      │
└────────┬────────────────┘
         │
         ▼
    ┌───┴────┐
    │ Drift? │
    └───┬────┘
        │ Yes
        ▼
┌─────────────────────────┐
│  Auto-Retrain           │
│  (loops back to top)    │
└─────────────────────────┘
```

## Configuration

### Training Schedule Configuration

Location: `.github/workflows/config/training-schedule.json`

```json
{
  "default_schedule": "0 2 * * 0",  // Weekly Sunday 2 AM UTC
  "model_retention": {
    "candidates": "30d",              // Keep candidates 30 days
    "production": "365d"              // Keep production 1 year
  },
  "performance_thresholds": {
    "accuracy_min": 0.85,             // Minimum accuracy for promotion
    "ece_max": 0.08,                  // Maximum ECE for promotion
    "drift_alert_threshold": 0.15     // Drift score threshold
  },
  "datasets": {
    "refresh_interval_days": 30,      // Max dataset age
    "minimum_samples": 10000          // Min samples required
  }
}
```

### Customization Options

**Training Frequency**:
- Edit `default_schedule` using cron syntax
- Examples:
  - Daily: `"0 2 * * *"`
  - Twice weekly: `"0 2 * * 0,3"`
  - Monthly: `"0 2 1 * *"`

**Quality Thresholds**:
- Adjust `accuracy_min` and `ece_max` based on your model requirements
- Stricter thresholds (e.g., accuracy_min: 0.90) reduce false promotions
- Looser thresholds enable faster iteration

**Drift Sensitivity**:
- Increase `drift_alert_threshold` to reduce alert frequency
- Decrease to catch subtle performance degradation earlier

## Data Flow

### Training Data Flow

```
Kaggle Datasets → Download → CSV Files → Preprocessing → Training Split
                     ↓
              Synthetic Fallback
                     ↓
              Feature Extraction → Training → Validation → Metrics
                                                              ↓
                                                    Promotion Gate
                                                    /           \
                                              Passed          Failed
                                                /               \
                                        models/current/    models/candidates/
```

### Model Metadata Flow

```
Training Metrics → Model Card Generation → metadata/
                         ↓
    ┌────────────────────┼────────────────────┐
    │                    │                    │
Audit Logs      Governance Stats      Drift Reports
    │                    │                    │
    └────────────────────┴────────────────────┘
                         ↓
              Comprehensive Model Card
```

## Integration Points

### 1. Governance System

The training pipeline integrates with Nethical's governance system:

- **Data Validation**: First 100 training samples checked for safety violations
- **Prediction Validation**: First 50 model predictions validated
- **Metrics Tracking**: Governance stats included in audit logs

### 2. Audit System

Merkle tree-based audit logging:

- **Event Recording**: All training events recorded with timestamps
- **Immutable Trail**: Merkle root provides tamper-proof audit trail
- **Chunk Finalization**: Audit chunks finalized after each training run

### 3. Drift Tracking

Ethical drift reporter integration:

- **Cohort Tracking**: Each training run assigned to cohort
- **Violation Recording**: Promotion failures and quality issues tracked
- **Drift Analysis**: Multi-cohort comparison for temporal drift

### 4. Adversarial Training

Optional integration with adversarial generator:

- **Synthetic Threats**: Generates hard negative samples
- **Attack Vectors**: Includes various attack patterns
- **Robustness Testing**: Validates model resilience

## Security Considerations

### Workflow Security

- **Minimal Permissions**: Each job uses least-privilege permissions
- **Secret Management**: Kaggle credentials via GitHub secrets
- **Isolation**: Training runs in isolated GitHub Actions runners

### Model Security

- **Integrity Checks**: File hashes and validation before deployment
- **Audit Trails**: Complete lineage from data to deployed model
- **Access Control**: Production models in protected artifacts

### Data Security

- **Synthetic Fallback**: No external data required for testing
- **Governance Checks**: All data validated before training
- **PII Protection**: Governance detects and blocks PII in training data

## Performance Optimization

### Workflow Execution Time

**Target**: < 30 minutes for default configuration

**Optimization Strategies**:
- Parallel model training (future enhancement)
- Cached dependencies (pip cache)
- Incremental dataset downloads
- Efficient artifact storage

### Resource Usage

- **CPU**: 2 cores standard GitHub runner
- **Memory**: 7 GB available
- **Storage**: Artifacts retained per schedule (30-365 days)

### Cost Management

- **Artifact Retention**: Configurable retention periods
- **Scheduled Runs**: Balance frequency vs. cost
- **Manual Triggers**: On-demand training for urgent updates

## Monitoring and Observability

### Metrics Tracked

1. **Training Metrics**:
   - Accuracy, Precision, Recall, F1
   - Expected Calibration Error (ECE)
   - Brier score (when probabilities available)

2. **Performance Metrics**:
   - Training duration
   - Dataset size
   - Promotion success rate

3. **Governance Metrics**:
   - Data violations detected
   - Prediction violations detected
   - Actions blocked

4. **Drift Metrics**:
   - Prediction distribution changes
   - Accuracy degradation
   - Calibration shifts

### Alerting

**GitHub Issues Created For**:
- Model performance regression
- Drift detection above threshold
- Dataset staleness
- Data quality issues
- Deployment failures

**Labels Used**:
- `ml-ops`: All MLOps-related issues
- `regression`: Performance regression detected
- `drift-alert`: Model drift detected
- `data-quality`: Dataset validation failures
- `needs-investigation`: Requires human review
- `needs-attention`: Urgent action required

## Disaster Recovery

### Model Rollback

If a deployed model fails:

1. Access previous production model from `models/archived/`
2. Verify model integrity and metadata
3. Use deploy_model.py to redeploy previous version
4. Investigate failure cause in audit logs

### Data Recovery

If training data is corrupted:

1. Use synthetic data fallback (`--no-download`)
2. Re-download from Kaggle (if available)
3. Restore from backup datasets
4. Validate data quality before retraining

### Workflow Recovery

If a workflow fails:

1. Check GitHub Actions logs for error details
2. Verify configuration files are valid
3. Ensure required secrets are configured
4. Re-run workflow with manual trigger
5. Use `--force-retrain` to bypass caching

## Future Enhancements

### Planned Features

1. **Multi-Environment Deployment**:
   - Staging environment with approval gates
   - Production environment with manual approval
   - Blue-green deployment support

2. **Advanced Monitoring**:
   - Real-time inference monitoring
   - Custom metric dashboards
   - Anomaly detection on model behavior

3. **Automated Hyperparameter Tuning**:
   - Grid search integration
   - Bayesian optimization
   - Multi-objective optimization

4. **Model Ensembling**:
   - Automatic ensemble creation
   - Weighted voting strategies
   - Confidence-based selection

5. **Enhanced Security**:
   - Model watermarking
   - Adversarial robustness testing
   - Backdoor detection

## Troubleshooting Guide

See [Training README](../training/README.md#troubleshooting-automated-workflows) for detailed troubleshooting steps.

## Related Documentation

- [Training Pipeline Documentation](../training/README.md)
- [Model Deployment Guide](./model-deployment-guide.md)
- [Model Directory Structure](../models/README.md)
- [GitHub Actions Workflows](../.github/workflows/)
