# Model Directory Structure

This directory contains trained machine learning models for the Nethical framework, organized by deployment status.

## Directory Layout

```
models/
├── current/          # Production models actively in use
├── candidates/       # Trained models awaiting promotion
├── archived/         # Historical models for reference
└── metadata/         # Model cards and performance logs
```

## Directory Descriptions

### `current/`
Contains models that have passed validation gates and are deployed to production. These models are used by the Nethical system for active threat detection and analysis.

**Retention:** 365 days (configurable via `.github/workflows/config/training-schedule.json`)

**Naming Convention:** `{model_type}_{timestamp}.pkl` or `{model_type}_{timestamp}.pt`

### `candidates/`
Contains newly trained models that have not yet been promoted to production. Models remain here until they:
- Pass performance comparison with baseline models
- Meet minimum quality thresholds
- Are manually or automatically promoted

**Retention:** 30 days (configurable)

**Naming Convention:** `{model_type}_{timestamp}.pkl` or `{model_type}_{timestamp}.pt`

### `archived/`
Historical models that have been replaced or deprecated. Useful for:
- Performance comparison
- Rollback scenarios
- Audit trails
- Research and analysis

**Retention:** Indefinite (manual cleanup)

### `metadata/`
Contains model cards, training logs, and performance metrics for each model:
- `{model_id}_card.json` - Comprehensive model metadata
- `{model_id}_metrics.json` - Performance metrics
- `{model_id}_training_log.txt` - Training execution logs

## Model Metadata Schema

Each model in `metadata/` has an associated JSON file with the following structure:

```json
{
  "model_version": "1.0.0",
  "model_type": "logistic",
  "training_timestamp": "2026-01-03T20:00:00Z",
  "metrics": {
    "accuracy": 0.92,
    "precision": 0.90,
    "recall": 0.88,
    "f1": 0.89,
    "ece": 0.06
  },
  "training_config": {
    "epochs": 30,
    "batch_size": 64,
    "num_samples": 20000,
    "seed": 42
  },
  "dataset_info": {
    "source": "synthetic",
    "samples": 20000,
    "features": ["violation_count", "severity_max", "recency_score"]
  },
  "governance_summary": {
    "enabled": true,
    "data_violations": 0,
    "prediction_violations": 0
  },
  "audit_merkle_root": "abc123...",
  "drift_report_id": "drift_20260103_200000"
}
```

## Promotion Workflow

1. **Training**: Models are trained via `training/train_any_model.py` and saved to `candidates/`
2. **Validation**: Automated or manual validation against quality gates
3. **Comparison**: Performance compared with current production models
4. **Promotion**: If passing all gates, model is copied/moved to `current/`
5. **Archival**: Replaced production models moved to `archived/`

## Accessing Models

Models are stored as Python pickle files (`.pkl`) or PyTorch checkpoint files (`.pt`). To load:

```python
import pickle
from pathlib import Path

# Load a production model
model_path = Path("models/current/logistic_20260103_200000.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load associated metadata
metadata_path = Path("models/metadata/logistic_20260103_200000_card.json")
with open(metadata_path, "r") as f:
    metadata = json.load(f)
```

## Cleanup and Retention

Automated cleanup is performed by the ML training workflow:
- Candidate models older than 30 days are automatically archived
- Production models older than 365 days are archived
- Archived models are never automatically deleted

Manual cleanup can be performed using:
```bash
# Archive old candidates
find models/candidates -name "*.pkl" -mtime +30 -exec mv {} models/archived/ \;

# Remove very old archives (optional, manual only)
find models/archived -name "*.pkl" -mtime +730 -delete
```

## Security Considerations

- Models contain trained parameters but no sensitive data
- Audit logs and Merkle roots ensure model provenance
- All model files are included in Git LFS (if configured)
- Access to production models should be restricted via file permissions

## See Also

- [Training Pipeline Documentation](../../training/README.md)
- [MLOps Architecture](../../docs/mlops-architecture.md)
- [Model Deployment Guide](../../docs/model-deployment-guide.md)
