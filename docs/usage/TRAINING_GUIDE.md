# End-to-End ML Classifier Training Guide

## Overview

This guide covers the training workflow for all three core MLops classifier modules in the Nethical framework:

1. **BaselineMLClassifier**: Lightweight heuristic-based classifier for ethical violation detection
2. **AnomalyMLClassifier**: N-gram based anomaly detector for action sequence analysis
3. **CorrelationMLClassifier**: Multi-agent correlation pattern detector

All classifiers can be trained using the unified `training/train_any_model.py` script with optional audit logging, governance validation, and ethical drift tracking.

## Classifier Modules

### 1. BaselineMLClassifier

**Use case**: Detecting ethical violations based on numeric features

**Location**: `nethical/mlops/baseline.py`

**Features analyzed**:
- `violation_count`: Number of violations detected
- `severity_max`: Maximum severity level
- `recency_score`: How recent the event is
- `frequency_score`: Frequency of similar events
- `context_risk`: Contextual risk factors

**Training command**:
```bash
# Heuristic mode (raw features)
python training/train_any_model.py --model-type heuristic --num-samples 5000

# Logistic mode (normalized features)
python training/train_any_model.py --model-type logistic --num-samples 5000

# With governance and audit
python training/train_any_model.py --model-type heuristic --enable-governance --enable-audit
```

**Example usage**:
```python
from nethical.mlops.baseline import BaselineMLClassifier

clf = BaselineMLClassifier.load('models/current/heuristic_model.json')
result = clf.predict({
    'violation_count': 0.7,
    'severity_max': 0.8,
    'recency_score': 0.5,
    'frequency_score': 0.4,
    'context_risk': 0.6
})
print(f"Label: {result['label']}, Score: {result['score']:.3f}")
```

### 2. AnomalyMLClassifier

**Use case**: Detecting anomalous patterns in action sequences

**Location**: `nethical/mlops/anomaly_classifier.py`

**Features analyzed**:
- N-gram patterns (trigrams by default)
- Action frequency distributions
- Sequence entropy
- Pattern diversity

**Training command**:
```bash
# Train anomaly detector
python training/train_any_model.py --model-type anomaly --num-samples 5000

# With governance and drift tracking
python training/train_any_model.py --model-type anomaly --enable-governance --enable-drift-tracking
```

**Example usage**:
```python
from nethical.mlops.anomaly_classifier import AnomalyMLClassifier

clf = AnomalyMLClassifier.load('models/current/anomaly_model.json')
result = clf.predict({'sequence': ['read', 'process', 'write']})
print(f"Anomalous: {result['label'] == 1}, Score: {result['score']:.3f}")
```

### 3. CorrelationMLClassifier

**Use case**: Detecting coordinated multi-agent attack patterns

**Location**: `nethical/mlops/correlation_classifier.py`

**Features analyzed**:
- `agent_count`: Number of unique agents involved
- `action_rate`: Rate of actions per time unit
- `entropy_variance`: Variance in action entropy
- `time_correlation`: Temporal correlation between agents
- `payload_similarity`: Similarity of payloads across agents

**Training command**:
```bash
# Train correlation detector
python training/train_any_model.py --model-type correlation --num-samples 5000

# With full observability
python training/train_any_model.py --model-type correlation --enable-audit --enable-governance --enable-drift-tracking
```

**Example usage**:
```python
from nethical.mlops.correlation_classifier import CorrelationMLClassifier

clf = CorrelationMLClassifier.load('models/current/correlation_model.json')
result = clf.predict({
    'agent_count': 8,
    'action_rate': 45,
    'entropy_variance': 0.6,
    'time_correlation': 0.8,
    'payload_similarity': 0.75
})
print(f"Pattern detected: {result['label'] == 1}, Score: {result['score']:.3f}")
```

## Quick Start

### Option 1: Using train_any_model.py (Recommended)

The `training/train_any_model.py` script is the recommended unified training interface:

```bash
# Train BaselineMLClassifier (heuristic mode)
python training/train_any_model.py --model-type heuristic --epochs 10 --num-samples 5000

# Train AnomalyMLClassifier
python training/train_any_model.py --model-type anomaly --epochs 10 --num-samples 5000

# Train CorrelationMLClassifier
python training/train_any_model.py --model-type correlation --epochs 10 --num-samples 5000

# With full observability features
python training/train_any_model.py \
    --model-type heuristic \
    --num-samples 5000 \
    --enable-audit \
    --enable-governance \
    --enable-drift-tracking \
    --cohort-id production_v1
```

### Option 2: Using baseline_orchestrator.py (For Kaggle datasets)

```bash
# Full pipeline with Kaggle datasets
python scripts/baseline_orchestrator.py

# Or step-by-step
python scripts/baseline_orchestrator.py --download      # Download datasets
python scripts/baseline_orchestrator.py --process-only  # Process CSV files
python scripts/baseline_orchestrator.py --train-only    # Train model
```

## Dataset Setup

### Kaggle Datasets

All datasets are listed in `datasets/datasets`. The training pipeline supports automatic download:

```bash
# Automatic download (requires Kaggle API)
python training/train_any_model.py --model-type heuristic --num-samples 10000

# Skip download (use synthetic data or existing files)
python training/train_any_model.py --model-type heuristic --num-samples 10000 --no-download
```

**Kaggle API Setup**:
1. Get credentials from https://www.kaggle.com/account
2. Save to `~/.kaggle/kaggle.json`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

**Manual Download**:
1. Download CSV files from Kaggle
2. Place in `data/external/` directory
3. Run training script

### Synthetic Data Fallback

When real datasets are unavailable, the training pipeline automatically generates synthetic data:

- **BaselineMLClassifier**: Random features with rule-based labels
- **AnomalyMLClassifier**: Normal vs. anomalous action sequences
- **CorrelationMLClassifier**: Normal vs. coordinated activity patterns

## Feature Preprocessing

### BaselineMLClassifier Modes

1. **heuristic** (`--model-type heuristic`): Raw numeric features, no scaling
2. **logistic** (`--model-type logistic`): Min-max normalization to [0,1]
3. **simple_transformer** (`--model-type simple_transformer`): Features + text tokenization

### AnomalyMLClassifier

Input: Action sequences as list of strings
```python
{'sequence': ['read', 'process', 'write', 'logout']}
```

### CorrelationMLClassifier

Input: Multi-agent activity metrics
```python
{
    'agent_count': 5,
    'action_rate': 25.0,
    'entropy_variance': 0.4,
    'time_correlation': 0.6,
    'payload_similarity': 0.5
}
```

## Audit Logging

Enable Merkle tree-based audit logging for compliance and reproducibility:

```bash
python training/train_any_model.py \
    --model-type heuristic \
    --enable-audit \
    --audit-path training_audit_logs
```

**Logged Events**:
- Training start (config, parameters)
- Data loading (sample count)
- Data split (train/validation sizes)
- Training completion (duration)
- Validation metrics (accuracy, precision, recall, F1)
- Model save (paths)
- Governance validations (if enabled)

**Audit Summary** (`training_summary.json`):
```json
{
    "merkle_root": "abc123...",
    "model_type": "heuristic",
    "promoted": true,
    "metrics": {"accuracy": 0.92, "precision": 0.90, ...},
    "governance": {"enabled": true, "data_violations": 0, ...}
}
```

## Governance Validation

Enable real-time safety and ethical validation during training:

```bash
python training/train_any_model.py \
    --model-type heuristic \
    --enable-governance \
    --gov-data-samples 100 \
    --gov-pred-samples 50
```

**Detected Violations**:
- Ethical violations (harmful content, bias)
- Safety violations (dangerous commands)
- Manipulation patterns (social engineering)
- Privacy issues (PII exposure)
- Security issues (prompt injection)

**Fail-Fast Mode**:
```bash
python training/train_any_model.py \
    --model-type heuristic \
    --enable-governance \
    --gov-fail-on-violations
```

## Model Selection & Promotion

Models are evaluated against promotion gate criteria:

- **Max ECE (Expected Calibration Error)**: ≤ 0.08
- **Min Accuracy**: ≥ 0.85

**Customize thresholds**:
```bash
python training/train_any_model.py \
    --model-type heuristic \
    --promotion-max-ece 0.10 \
    --promotion-min-accuracy 0.80
```

**Model paths**:
- Promoted: `models/current/<model_type>_model_*.json`
- Candidates: `models/candidates/<model_type>_model_*.json`

## Example Training Scripts

See `examples/training/` for complete examples:

- `train_baseline_classifier.py`: BaselineMLClassifier training demo
- `train_anomaly_detector.py`: AnomalyMLClassifier training demo
- `correlation_model_demo.py`: CorrelationMLClassifier training demo
- `demo_governance_training.py`: Training with governance validation
- `train_with_drift_tracking.py`: Training with ethical drift tracking
- `real_data_training_demo.py`: Real data training workflow

## Tests

Validate classifier functionality:

```bash
# Test all classifiers
python -m pytest tests/test_anomaly_classifier.py -v
python -m pytest tests/test_correlation_classifier.py -v

# Test training pipeline
python -m pytest tests/test_train_audit_logging.py -v
python -m pytest tests/test_train_governance.py -v
python -m pytest tests/test_train_drift_tracking.py -v
```

## Related Documentation

- **training/README.md**: Detailed training script documentation
- **scripts/README.md**: Script usage and workflow guide
- **docs/implementation/TRAIN_MODEL_REAL_DATA_SUMMARY.md**: Real data training implementation
- **docs/AUDIT_LOGGING_GUIDE.md**: Audit logging details
- **docs/implementation/GOVERNANCE_TRAINING_IMPLEMENTATION.md**: Governance integration

## Next Steps After Training

1. Review validation metrics in model JSON files
2. Test models with `scripts/test_model.py`
3. Deploy to Phase 5 shadow mode (passive inference)
4. Promote to Phase 6 blended enforcement (active decisions)
5. Monitor with Phase 7 anomaly detection
6. Track drift with ethical drift reporting
