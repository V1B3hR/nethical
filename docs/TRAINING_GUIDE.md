# End-to-End Real Data Training - Quick Reference

## Overview

The end-to-end training pipeline downloads, processes, and trains the BaselineMLClassifier using real-world security datasets from Kaggle.

## Quick Start

### Option 1: Using train_model.py (Recommended for specific datasets)

The `train_model.py` script now uses real-world data by default from these two datasets:
- https://www.kaggle.com/code/kmldas/data-ethics-in-data-science-analytics-ml-and-ai
- https://www.kaggle.com/datasets/xontoloyo/security-breachhh

```bash
# Train with real datasets (downloads and processes automatically)
python scripts/train_model.py

# Train and test (complete workflow)
python scripts/train_model.py all
python scripts/train_model.py --run-all
```

**Note:** If Kaggle API is not available, manually download the CSV files and place them in `data/external/`.

### Option 2: Using baseline_orchestrator.py (For all datasets)

```bash
# Full pipeline
python scripts/baseline_orchestrator.py

# Or step-by-step
python scripts/baseline_orchestrator.py --download      # Download datasets
python scripts/baseline_orchestrator.py --process-only  # Process CSV files
python scripts/baseline_orchestrator.py --train-only    # Train model
```

### Option 3: Using train_any_model.py (With Audit Logging and Governance)

The `training/train_any_model.py` script supports both Merkle audit logging and governance validation:

```bash
# Train with audit logging enabled
python training/train_any_model.py --model-type heuristic --epochs 10 --num-samples 1000 --enable-audit

# Train with governance validation enabled
python training/train_any_model.py --model-type logistic --epochs 20 --num-samples 2000 --enable-governance

# Train with both audit logging and governance validation
python training/train_any_model.py --model-type heuristic --epochs 10 --num-samples 1000 --enable-audit --enable-governance

# Customize audit log path
python training/train_any_model.py --model-type logistic --epochs 20 --enable-audit --audit-path custom_audit_logs

# Train without audit logging or governance (default)
python training/train_any_model.py --model-type heuristic --epochs 10 --num-samples 1000
```

#### Audit Logging Features

When `--enable-audit` is specified:
- Creates an immutable audit trail using Merkle trees
- Logs key training events: start, data loading, split, training completion, metrics, model save
- Generates a Merkle root hash for cryptographic verification
- Saves audit logs in structured JSON format
- Creates a summary file with Merkle root and metrics

The audit logs can be used for:
- Compliance and regulatory requirements
- Training reproducibility verification
- Detecting tampering with training records
- Tracking model lineage and provenance

#### Governance Validation Features

When `--enable-governance` is specified:
- Validates training data samples for safety violations (toxic content, harmful patterns, etc.)
- Checks model predictions during validation for safety issues
- Reports governance violations and decisions
- Includes governance metrics in the training summary and audit logs

The governance system detects:
- Ethical violations (harmful content, bias, discrimination)
- Safety violations (dangerous commands, unsafe domains)
- Manipulation patterns (social engineering, phishing, emotional leverage)
- Dark patterns (NLP manipulation, weaponized empathy)
- Privacy issues (PII exposure)
- Security issues (prompt injection, adversarial attacks)

Governance validation provides:
- Real-time safety checks during training
- Detailed violation reports with confidence scores
- Decision tracking (allow, block, quarantine, etc.)
- Integration with audit logging for comprehensive compliance

**Example Output:**
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

## Dataset Sources

All datasets are listed in `datasets/datasets`. See README.md for the complete list.

## Next Steps

1. Review validation metrics
2. Test model with scripts/test_model.py
3. Deploy to Phase 5 shadow mode
4. Monitor performance with Phase 7 anomaly detection

See scripts/README.md and README.md for detailed documentation.
