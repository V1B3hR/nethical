# Quarantine Integration in Training Pipeline

## Overview

The training pipeline now includes integrated quarantine functionality for automated model risk management. This feature allows you to automatically quarantine model cohorts that fail the promotion gate, providing rapid incident response for low-quality or risky models.

## Features

- **Automatic Quarantine**: Models that fail the promotion gate can be automatically quarantined
- **Cohort Tracking**: Training runs are registered as cohorts in the quarantine system
- **Risk-Based Categorization**: Quarantine reasons are determined based on the type of failure:
  - `high_risk_score`: Both ECE and accuracy thresholds violated
  - `policy_violation`: Single metric threshold violated
- **Configurable Duration**: Failed models are quarantined for 48 hours by default
- **Audit Integration**: Quarantine events are logged to the audit trail when enabled
- **Fast Activation**: Quarantine activation completes in milliseconds (target: <15 seconds)

## Usage

### Enable Quarantine System

To enable the quarantine system without auto-quarantine:

```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-quarantine
```

This initializes the quarantine manager and tracks the training cohort, but won't automatically quarantine models that fail.

### Enable Auto-Quarantine on Failure

To automatically quarantine models that fail the promotion gate:

```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-quarantine \
    --quarantine-on-failure
```

### Combined with Audit Logging

Combine quarantine with audit logging for complete governance:

```bash
python training/train_any_model.py \
    --model-type anomaly \
    --num-samples 5000 \
    --enable-quarantine \
    --quarantine-on-failure \
    --enable-audit \
    --audit-path ./training_audit
```

### Combined with Drift Tracking

Enable all governance features together:

```bash
python training/train_any_model.py \
    --model-type correlation \
    --num-samples 10000 \
    --enable-quarantine \
    --quarantine-on-failure \
    --enable-audit \
    --enable-drift-tracking
```

## Command-Line Arguments

- `--enable-quarantine`: Enable the quarantine system for model management
- `--quarantine-on-failure`: Automatically quarantine models that fail the promotion gate (requires `--enable-quarantine`)

## Output

When quarantine is enabled, you'll see:

1. **Initialization Message**: Confirms quarantine system is enabled and shows the cohort ID
2. **Quarantine Action**: If a model fails and auto-quarantine is enabled, shows quarantine details
3. **Quarantine Summary**: At the end of training, shows the quarantine status and statistics

Example output:

```
[INFO] Quarantine system enabled for model cohort: logistic_20251007_120500
...
[INFO] Promotion result: FAIL

[INFO] Model failed promotion gate. Quarantining cohort: logistic_20251007_120500
[INFO] Cohort quarantined:
  - Status: active
  - Reason: high_risk_score
  - Expires at: 2025-10-09T12:05:00.990894
  - Activation time: 0.06ms

======================================================================
[INFO] Quarantine System Summary
======================================================================
Cohort ID: logistic_20251007_120500
Is Quarantined: YES
  Reason: high_risk_score
  Status: active
  Activated at: 2025-10-07T12:05:00.990906
  Expires at: 2025-10-09T12:05:00.990894
  Duration: 48.0 hours
  Activation time: 0.06ms

Quarantine Statistics:
  Active quarantines: 1
  Total quarantines: 1
  Avg activation time: 0.06ms
  Target activation time: 15.0s
======================================================================
```

## Promotion Gate Criteria

Models are evaluated against the following criteria:

- **ECE (Expected Calibration Error)**: Must be ≤ 0.08
- **Accuracy**: Must be ≥ 0.85

If a model fails either criterion with `--quarantine-on-failure` enabled, it will be quarantined.

## Quarantine Reasons

The system assigns quarantine reasons based on the failure type:

1. **high_risk_score**: Both ECE > 0.08 AND accuracy < 0.85
   - Indicates a model with both calibration and accuracy issues
   - Highest risk category

2. **policy_violation**: Single metric violation
   - ECE > 0.08 (calibration issue)
   - OR accuracy < 0.85 (accuracy issue)

## Quarantine Duration

Failed models are quarantined for **48 hours** by default. During this time:

- The cohort status is tracked
- Quarantine can be manually released if needed
- Audit events are logged (if audit logging is enabled)

## Integration with Other Features

### Merkle Audit Logging

When both quarantine and audit logging are enabled:

- Quarantine initialization is logged
- Quarantine events (activation, release) are logged
- Model promotion events are logged
- All events are included in the Merkle tree for immutability

### Ethical Drift Tracking

When combined with drift tracking:

- Quarantine events can be correlated with drift patterns
- Failed models contribute to drift analysis
- Helps identify systematic issues in model training

## Testing

Run the quarantine integration tests:

```bash
python tests/test_train_quarantine.py
```

This will test:
- Basic quarantine system initialization
- Auto-quarantine on failure
- Integration with audit logging
- Quarantine activation time
- Statistics reporting

## Technical Details

### Quarantine Manager

The quarantine system uses the `QuarantineManager` class from `nethical.core.quarantine`:

- Cohorts are identified by a combination of model type and timestamp
- Agents (models) are registered to cohorts
- Quarantine activation is designed for <15 second response time
- Statistics are tracked for monitoring

### Cohort IDs

Cohort IDs follow the format: `{model_type}_{timestamp}`

Example: `logistic_20251007_120500`

This ensures unique identification of each training run.

## Use Cases

1. **Automated Quality Control**: Prevent low-quality models from being deployed
2. **Risk Management**: Quarantine models with calibration or accuracy issues
3. **Incident Response**: Rapidly isolate problematic model versions
4. **Compliance**: Maintain audit trails of model quarantine actions
5. **Investigation**: Track patterns in model failures through quarantine statistics

## Future Enhancements

Potential future improvements:

- Configurable quarantine duration per model type
- Manual quarantine management CLI
- Quarantine notification system
- Integration with deployment pipelines
- Quarantine release workflows
- Model remediation tracking
