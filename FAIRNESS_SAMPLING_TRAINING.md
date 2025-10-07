# Fairness Sampling in Training Pipeline

## Overview

The training pipeline (`training/train_any_model.py`) now includes integrated fairness sampling capabilities using the `FairnessSampler` module. This feature enables stratified sampling of training and validation data for fairness evaluation and bias detection during model training.

## Features

- **Stratified Sampling**: Proportionally samples from different cohorts (e.g., train, validation)
- **Configurable Sample Size**: Set target number of samples to collect
- **Custom Cohorts**: Define custom cohort names for different data segments
- **Audit Trail Integration**: Fairness sampling events are logged in Merkle audit trail when enabled
- **Persistent Storage**: Samples are saved to JSON files for later analysis

## Usage

### Basic Usage

Enable fairness sampling with default settings:

```bash
python training/train_any_model.py \
  --model-type logistic \
  --enable-fairness-sampling
```

### Custom Configuration

Configure fairness sampling parameters:

```bash
python training/train_any_model.py \
  --model-type logistic \
  --enable-fairness-sampling \
  --fairness-cohorts "train,validation,test" \
  --fairness-sample-size 100 \
  --fairness-storage-dir "./custom_fairness_samples"
```

### With Audit Logging and Drift Tracking

Combine fairness sampling with other monitoring features:

```bash
python training/train_any_model.py \
  --model-type logistic \
  --enable-audit \
  --audit-path "./training_audit" \
  --enable-drift-tracking \
  --drift-report-dir "./drift_reports" \
  --enable-fairness-sampling \
  --fairness-sample-size 50
```

## Command-Line Arguments

### Fairness Sampling Arguments

- `--enable-fairness-sampling`: Enable fairness sampling (default: False)
- `--fairness-cohorts`: Comma-separated list of cohorts (default: "train,validation")
- `--fairness-sample-size`: Target number of samples to collect (default: 100)
- `--fairness-storage-dir`: Directory for storing samples (default: "fairness_samples")

## Output

### Fairness Sample File

Samples are saved in JSON format with the following structure:

```json
{
  "job_id": "job_20251007_120124_3965",
  "strategy": "stratified",
  "target_sample_size": 30,
  "cohorts": ["train", "validation"],
  "start_time": "2025-10-07T12:01:24.147852",
  "end_time": "2025-10-07T12:01:24.154591",
  "samples": [
    {
      "sample_id": "job_20251007_120124_3965_0",
      "agent_id": "train_sample_79",
      "action_id": "train_action_79",
      "cohort": "train",
      "violation_type": null,
      "severity": null,
      "timestamp": "2025-10-07T12:01:24.153688",
      "metadata": {
        "label": 1,
        "features": {...}
      }
    },
    ...
  ],
  "coverage": {
    "train": 24,
    "validation": 6
  },
  "metadata": {
    "model_type": "logistic",
    "training_start": "2025-10-07T12:01:24.147827+00:00",
    "seed": 42
  }
}
```

### Console Output

When fairness sampling is enabled, you'll see output like:

```
[INFO] Fairness sampling enabled. Storage: fairness_samples
[INFO] Sampling job ID: job_20251007_120124_3965
[INFO] Cohorts: ['train', 'validation']
[INFO] Target sample size: 30

[INFO] Performing fairness sampling...
[INFO] Collected 30 fairness samples
[INFO] Coverage rate: 100.0%
[INFO]   train: 24 samples (80.0%)
[INFO]   validation: 6 samples (20.0%)

[INFO] Finalizing fairness sampling job...
[INFO] Fairness sampling job finalized: job_20251007_120124_3965
[INFO] Final coverage: 30 samples
[INFO] Fairness samples saved to: fairness_samples/job_20251007_120124_3965.json
```

## Implementation Details

### Sampling Strategy

The implementation uses **stratified sampling** to ensure proportional representation from each cohort:

1. **Population Data**: Training and validation datasets are converted to population data structure
2. **Proportional Sampling**: Samples are drawn proportionally based on cohort size
3. **Coverage Tracking**: Sample counts per cohort are tracked and reported

### Sample Metadata

Each sample includes:
- **Sample ID**: Unique identifier for the sample
- **Agent ID**: Training sample identifier (e.g., `train_sample_79`)
- **Action ID**: Action identifier (e.g., `train_action_79`)
- **Cohort**: Which cohort the sample belongs to
- **Timestamp**: When the sample was collected
- **Metadata**: Original label and features from the training data

### Integration with Audit Trail

When audit logging is enabled (`--enable-audit`), fairness sampling events are logged:

```json
{
  "event_type": "fairness_sampling",
  "job_id": "job_20251007_120124_3965",
  "samples_collected": 30,
  "coverage_stats": {...},
  "timestamp": "2025-10-07T12:01:24.153688+00:00"
}
```

## Use Cases

### Bias Detection

Sample from different cohorts to analyze potential biases in model behavior:

```bash
python training/train_any_model.py \
  --model-type logistic \
  --enable-fairness-sampling \
  --fairness-cohorts "high_risk,medium_risk,low_risk" \
  --fairness-sample-size 300
```

### Fairness Auditing

Collect samples for fairness audits and compliance reporting:

```bash
python training/train_any_model.py \
  --model-type anomaly \
  --enable-fairness-sampling \
  --enable-audit \
  --fairness-sample-size 1000
```

### Model Validation

Sample validation data for detailed model evaluation:

```bash
python training/train_any_model.py \
  --model-type correlation \
  --enable-fairness-sampling \
  --fairness-cohorts "validation" \
  --fairness-sample-size 200
```

## Best Practices

1. **Sample Size**: Choose sample size based on dataset size (typically 5-10% of total data)
2. **Cohort Definition**: Define cohorts that align with fairness metrics of interest
3. **Storage Management**: Regularly clean up old sample files to manage disk space
4. **Audit Integration**: Enable audit logging for compliance and traceability
5. **Review Coverage**: Check coverage statistics to ensure adequate representation

## Troubleshooting

### No Samples Collected

If `Collected 0 fairness samples` is reported:
- Verify cohort names match the population data structure
- Check that the dataset is not empty
- Ensure target sample size is not larger than available data

### Import Errors

If you see `FairnessSampler not available`:
- Verify the `nethical.core.fairness_sampler` module is installed
- Check Python path includes the project root
- Reinstall dependencies if needed

### File Permission Issues

If samples cannot be saved:
- Check write permissions for the storage directory
- Verify disk space is available
- Use absolute paths for storage directories

## Related Documentation

- [FairnessSampler Module](nethical/core/fairness_sampler.py)
- [Phase 3 Integration](nethical/core/phase3_integration.py)
- [Audit Logging](AUDIT_LOGGING_IMPLEMENTATION.md)
- [Drift Tracking](DRIFT_TRACKING_IMPLEMENTATION.md)
