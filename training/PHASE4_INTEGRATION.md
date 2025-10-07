# Phase 4 Integration in Model Training

This document describes the Phase 4 Integrated Governance integration in the `train_any_model.py` script.

## Overview

Phase 4 Integrated Governance provides comprehensive audit, ethical taxonomy, and SLA monitoring capabilities for model training. When enabled, it tracks all training lifecycle events, detects violations, and generates detailed governance reports.

## Features

### 1. Merkle Anchoring
- Creates an immutable audit trail of all training events
- Generates cryptographic Merkle root hashes for verification
- Stores events in tamper-proof audit logs

### 2. Ethical Taxonomy
- Multi-dimensional violation tagging
- Tracks model quality issues as ethical violations
- Categories include:
  - `low_model_accuracy`: When accuracy < 0.70
  - `poor_calibration`: When ECE > 0.15
  - `promotion_gate_ece_failure`: When ECE exceeds promotion threshold (0.08)
  - `promotion_gate_accuracy_failure`: When accuracy below promotion threshold (0.85)

### 3. SLA Monitoring
- Tracks latency of all training operations
- Reports P95, P99, average, and max latency
- Validates against SLA targets (default: 220ms P95)

### 4. Comprehensive Reporting
- Auto-generates markdown reports with full governance status
- Includes system status, SLA performance, and ethical coverage
- Saved to configurable storage directory

## Usage

### Basic Training with Phase 4

```bash
python training/train_any_model.py \
    --model-type heuristic \
    --epochs 10 \
    --num-samples 1000 \
    --enable-phase4
```

### With Custom Storage Location

```bash
python training/train_any_model.py \
    --model-type logistic \
    --epochs 20 \
    --num-samples 2000 \
    --enable-phase4 \
    --phase4-storage ./my_phase4_data
```

### Command-Line Arguments

- `--enable-phase4`: Enable Phase 4 Integrated Governance
- `--phase4-storage DIR`: Storage directory for Phase 4 data (default: `training_phase4_data`)

## What Gets Tracked

Phase 4 tracks the following training lifecycle events:

1. **Training Start**: Configuration and initialization
2. **Data Loaded**: Number of samples loaded
3. **Data Split**: Train/validation split information
4. **Training Completed**: Training duration and completion time
5. **Validation Metrics**: Model performance metrics with violation detection
6. **Model Saved**: Save location and promotion status with gate violation tracking

## Violation Detection

Phase 4 automatically detects and tags the following violations:

| Violation Type | Condition | Severity |
|---------------|-----------|----------|
| `low_model_accuracy` | accuracy < 0.70 | High |
| `poor_calibration` | ECE > 0.15 | High |
| `promotion_gate_ece_failure` | ECE > 0.08 (promotion threshold) | Medium |
| `promotion_gate_accuracy_failure` | accuracy < 0.85 (promotion threshold) | Medium |

## Output Files

When Phase 4 is enabled, the following files are generated:

```
training_phase4_data/
├── audit_logs/
│   └── chunk_YYYYMMDD_HHMMSS_NNNNNN.json  # Merkle audit chunks
└── training_phase4_report_MODEL_YYYYMMDD_HHMMSS.md  # Comprehensive report
```

## Example Output

```
[INFO] Phase 4 Integrated Governance enabled. Storage: training_phase4_data
[INFO] Loading 200 synthetic samples...
[INFO] Preprocessing data for model type: heuristic
[INFO] Train samples: 160, Validation samples: 40
[INFO] Training heuristic model for 5 epochs, batch size 32...
[INFO] Validation Metrics:
  precision: 0.5789
  recall: 1.0000
  accuracy: 0.6000
  f1: 0.7333
  ece: 0.0211
[INFO] Promotion Gate: ECE <= 0.08, Accuracy >= 0.85
[INFO] ECE: 0.021, Accuracy: 0.600
[INFO] Promotion result: FAIL

[INFO] Finalizing Phase 4 Integrated Governance...
[INFO] Phase 4 audit trail finalized. Merkle root: f526f12c...
[INFO] SLA Performance:
  Status: SLAStatus.COMPLIANT
  P95 Latency: 0.04ms
  SLA Met: ✅
[INFO] Ethical Taxonomy Coverage:
  Coverage: 0.0%
  Target: 90.0%
  Meets Target: ❌
[INFO] Phase 4 comprehensive report saved to: training_phase4_data/training_phase4_report_heuristic_YYYYMMDD_HHMMSS.md
```

## Backward Compatibility

Phase 4 integration maintains full backward compatibility:
- Existing `--enable-audit` option still works for standalone Merkle anchoring
- When Phase 4 is enabled, standalone audit is disabled (Phase 4 includes Merkle anchoring)
- All existing command-line arguments remain functional

## Comparison with Standalone Audit

| Feature | Standalone Audit | Phase 4 Integrated |
|---------|-----------------|-------------------|
| Merkle Anchoring | ✅ | ✅ |
| Ethical Taxonomy | ❌ | ✅ |
| Violation Detection | ❌ | ✅ |
| SLA Monitoring | ❌ | ✅ |
| Comprehensive Reports | ❌ | ✅ |
| Policy Diff Auditing | ❌ | ✅ (available) |

## Integration with Other Features

Phase 4 works alongside existing features:
- **Drift Tracking** (`--enable-drift-tracking`): Can be used simultaneously with Phase 4
- **Standalone Audit** (`--enable-audit`): Disabled when Phase 4 is active (Phase 4 includes better audit capabilities)

## Best Practices

1. **Always enable Phase 4 for production training runs** to ensure full governance and auditability
2. **Review Phase 4 reports** after training to understand model quality and potential violations
3. **Archive Phase 4 data** for compliance and audit purposes
4. **Monitor SLA metrics** to ensure training performance meets requirements
5. **Track ethical taxonomy coverage** to ensure comprehensive violation detection

## Troubleshooting

### Phase 4 Not Available
If you see: `[WARN] Phase4IntegratedGovernance not available`
- Ensure `nethical.core.phase4_integration` is properly installed
- Check that all Phase 4 dependencies are available

### Storage Directory Issues
- Ensure the user has write permissions to the Phase 4 storage directory
- The directory will be created automatically if it doesn't exist

### Ethical Taxonomy Coverage Low
- This is expected during training (coverage tracks violation types seen during runtime)
- Coverage will increase as different violation types are encountered
- Coverage metrics are more relevant for production agent monitoring

## See Also

- [Phase 4 Demo](../examples/phase4_demo.py): Comprehensive demo of Phase 4 features
- [Training README](README.md): General model training documentation
- [Phase 4 Integration Module](../nethical/core/phase4_integration.py): Core implementation
