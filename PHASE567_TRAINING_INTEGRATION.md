# Phase567 Integration in Model Training

This document describes the Phase567 integrated governance feature in `train_any_model.py`.

## Overview

The Phase567 integration brings comprehensive ML governance to the model training pipeline, enabling:

1. **Phase 5: ML Shadow Mode** - Passive ML model validation without enforcement risk
2. **Phase 6: ML Blended Risk** - ML-assisted decision making in gray-zone scenarios
3. **Phase 7: Anomaly & Drift Detection** - Behavioral monitoring and anomaly detection

## Usage

### Basic Usage (All Components Enabled)

Enable all Phase567 components (shadow mode, blended risk, and anomaly detection):

```bash
python training/train_any_model.py \
    --model-type heuristic \
    --num-samples 1000 \
    --enable-phase567
```

### Selective Component Activation

Enable only specific components:

```bash
# Enable only shadow mode and anomaly detection
python training/train_any_model.py \
    --model-type logistic \
    --num-samples 1000 \
    --enable-phase567 \
    --enable-shadow-mode \
    --enable-anomaly-detection
```

```bash
# Enable all three components explicitly
python training/train_any_model.py \
    --model-type heuristic \
    --num-samples 1000 \
    --enable-phase567 \
    --enable-shadow-mode \
    --enable-ml-blending \
    --enable-anomaly-detection
```

### Custom Storage Directory

Specify a custom directory for Phase567 data:

```bash
python training/train_any_model.py \
    --model-type heuristic \
    --num-samples 1000 \
    --enable-phase567 \
    --phase567-storage-dir ./my_phase567_data
```

## Command-Line Arguments

- `--enable-phase567` - Enable Phase567 integrated governance
- `--phase567-storage-dir DIR` - Storage directory for Phase567 data (default: `training_phase567_data`)
- `--enable-shadow-mode` - Enable ML shadow mode (requires `--enable-phase567`)
- `--enable-ml-blending` - Enable ML blended risk (requires `--enable-phase567`)
- `--enable-anomaly-detection` - Enable anomaly detection (requires `--enable-phase567`)

## How It Works

When Phase567 is enabled, the training script:

1. **Initializes Phase567IntegratedGovernance** with the specified components
2. **Sets baseline distribution** using training data for anomaly detection
3. **Processes validation samples** through the Phase567 pipeline:
   - Each prediction is evaluated by the shadow classifier
   - Blended risk analysis is performed (if enabled)
   - Anomaly detection monitors for unusual patterns
4. **Collects and displays metrics** from all enabled components
5. **Generates a comprehensive report** in markdown format

## Output

### Console Output

During training, you'll see Phase567 metrics displayed:

```
[INFO] Processing validation samples through Phase567 governance...
[INFO] Set baseline distribution with 80 training samples
[INFO] Processed 20 validation samples through Phase567 governance

[INFO] Phase567 Shadow Mode Metrics:
  Total Predictions: 20
  Agreement Rate: 45.0%
  F1 Score: 0.737

[INFO] Phase567 Blended Risk Metrics:
  Total Decisions: 20
  ML Influence Rate: 0.0%
  Classification Change Rate: 0.0%

[INFO] Phase567 Anomaly Detection Statistics:
  Total Alerts: 11
  Tracked Agents: 1
```

### Generated Files

Phase567 generates the following files in the storage directory:

```
training_phase567_data/
├── shadow_logs/          # Shadow mode prediction logs
├── blended_logs/         # Blended risk decision logs
├── anomaly_logs/         # Anomaly detection logs
└── phase567_report_*.md  # Comprehensive markdown reports
```

### Report Format

The generated markdown report includes:

- **System Status** - Overview of all enabled components
- **Phase 5 Metrics** - Shadow mode performance (predictions, agreement rate, F1 score)
- **Phase 6 Metrics** - Blended risk decisions (influence rate, classification changes)
- **Phase 7 Metrics** - Anomaly detection statistics (alerts, tracked agents, severities)

## Example

See `examples/train_with_phase567.py` for a complete example demonstrating:

- Training with all components enabled
- Training with selective components
- Viewing generated reports

Run the example:

```bash
python examples/train_with_phase567.py
```

## Benefits

1. **Validation Without Risk** - Shadow mode validates ML predictions without affecting training
2. **Enhanced Decision Quality** - Blended risk improves decisions in uncertain scenarios
3. **Anomaly Detection** - Identifies unusual patterns during validation
4. **Comprehensive Reporting** - Detailed metrics for analysis and auditing
5. **Flexible Configuration** - Enable only the components you need

## Integration with Other Features

Phase567 can be combined with other training features:

```bash
# Combine with audit logging and drift tracking
python training/train_any_model.py \
    --model-type heuristic \
    --num-samples 1000 \
    --enable-phase567 \
    --enable-audit \
    --enable-drift-tracking
```

## Best Practices

1. **Start with all components enabled** to get a complete picture
2. **Use shadow mode** for low-risk validation of ML models
3. **Enable blended risk** when dealing with uncertain classifications
4. **Monitor anomaly detection** to identify training data quality issues
5. **Review generated reports** after each training run
6. **Compare reports across runs** to track model behavior evolution

## Troubleshooting

### "Phase567IntegratedGovernance not available"

Ensure the nethical package is properly installed:
```bash
pip install -e .
```

### High Anomaly Alert Rate

This is normal during initial training with limited validation samples. Increase `--num-samples` for more stable results.

### No ML Influence in Blended Risk

This indicates predictions are clearly in allow or deny zones, not the gray zone (0.4-0.6). This is expected behavior.

## Related Documentation

- Phase 5 (ML Shadow Mode): See `nethical/core/ml_shadow.py`
- Phase 6 (ML Blended Risk): See `nethical/core/ml_blended_risk.py`
- Phase 7 (Anomaly Detection): See `nethical/core/anomaly_detector.py`
- Integrated Governance: See `nethical/core/phase567_integration.py`
- Demo: See `examples/phase567_demo.py`
