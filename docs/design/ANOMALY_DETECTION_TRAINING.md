# Anomaly Detection Model Training

This guide explains how to train ML-based anomaly detection models for the Nethical framework.

## Overview

The `AnomalyMLClassifier` is a trainable ML model that learns to distinguish between normal and anomalous agent behavior patterns. It uses:

- **N-gram analysis**: Detects unusual action sequences
- **Action frequency analysis**: Spots rare or suspicious actions
- **Entropy analysis**: Identifies repetitive patterns
- **Pattern diversity analysis**: Profiles behavioral characteristics

## Quick Start

### 1. Train a Model

```bash
python training/train_any_model.py --model-type anomaly --num-samples 5000
```

### 2. Check Results

The training script will:
- Load or generate training data with normal and anomalous sequences
- Train the anomaly detection model
- Evaluate on validation set
- Save the model if it passes the promotion gate

Expected output:
```
[INFO] Loading 5000 anomaly detection samples...
[INFO] Train samples: 4000, Validation samples: 1000
[INFO] Training anomaly model for 10 epochs, batch size 32...
[INFO] Validation Metrics:
  precision: 1.0000
  recall: 1.0000
  accuracy: 1.0000
  f1: 1.0000
  ece: 0.0000
[INFO] Promotion Gate: ECE <= 0.08, Accuracy >= 0.85
[INFO] ECE: 0.000, Accuracy: 1.000
[INFO] Promotion result: PASS
[INFO] Model saved to models/current/anomaly_model_20241006_120000.json
```

### 3. Use the Trained Model

```python
from nethical.mlops.anomaly_classifier import AnomalyMLClassifier

# Load trained model
clf = AnomalyMLClassifier.load('models/current/anomaly_model_*.json')

# Predict on a new sequence
sequence = ['read', 'process', 'write']
result = clf.predict({'sequence': sequence})

print(f"Sequence: {sequence}")
print(f"Anomalous: {result['label'] == 1}")  # 0=normal, 1=anomalous
print(f"Score: {result['score']:.3f}")       # 0.0-1.0 anomaly score
print(f"Confidence: {result['confidence']:.3f}")  # 0.0-1.0 confidence
```

## Training Options

```bash
python training/train_any_model.py --model-type anomaly \
    --num-samples 10000 \
    --epochs 10 \
    --batch-size 32 \
    --seed 42
```

Options:
- `--model-type anomaly`: Required - specifies anomaly detection model
- `--num-samples N`: Number of training samples (default: 10000)
- `--epochs N`: Number of training epochs (default: 10)
- `--batch-size N`: Batch size (default: 32)
- `--seed N`: Random seed for reproducibility (default: 42)

## Training Data

The model expects data in this format:

```python
{
    'features': {
        'sequence': ['action1', 'action2', 'action3']  # List of actions
    },
    'label': 0  # 0 = normal, 1 = anomalous
}
```

### Example Patterns

**Normal sequences** (typical agent behavior):
- `['read', 'process', 'write']`
- `['fetch', 'transform', 'load']`
- `['query', 'filter', 'aggregate']`
- `['request', 'authenticate', 'respond']`

**Anomalous sequences** (suspicious behavior):
- `['delete', 'exfiltrate', 'cover_tracks']`
- `['escalate', 'access', 'exfiltrate']`
- `['scan', 'exploit', 'inject']`
- `['brute_force', 'access', 'modify']`

## Model Configuration

### N-gram Size

The model uses 3-grams by default. This can be customized:

```python
clf = AnomalyMLClassifier(n=3)  # Use 3-grams
clf = AnomalyMLClassifier(n=4)  # Use 4-grams
```

Larger n-grams capture longer patterns but require more training data.

### Anomaly Threshold

The default threshold is 0.3. Adjust for different sensitivity:

```python
clf = AnomalyMLClassifier(anomaly_threshold=0.3)  # Balanced (default)
clf = AnomalyMLClassifier(anomaly_threshold=0.2)  # More sensitive
clf = AnomalyMLClassifier(anomaly_threshold=0.5)  # More conservative
```

Lower thresholds detect more anomalies but may increase false positives.

## Promotion Gate

Models must pass these criteria to be deployed to production:

- **Expected Calibration Error (ECE)**: ≤ 0.08
- **Accuracy**: ≥ 0.85

Models that pass are saved to `models/current/`, otherwise to `models/candidates/`.

## Integration with Existing Anomaly Detector

The ML model complements the rule-based `AnomalyDriftMonitor` in `anomaly_detector.py`:

```python
from nethical.core.anomaly_detector import AnomalyDriftMonitor
from nethical.mlops.anomaly_classifier import AnomalyMLClassifier

# Initialize both detectors
monitor = AnomalyDriftMonitor(sequence_n=3)
ml_clf = AnomalyMLClassifier.load('models/current/anomaly_model.json')

# Use rule-based detector for real-time tracking
alert = monitor.record_action(agent_id="agent_001", 
                              action_type="delete",
                              risk_score=0.8)

# Use ML model for deeper analysis
sequence = ['delete', 'exfiltrate', 'cover_tracks']
ml_result = ml_clf.predict({'sequence': sequence})

if ml_result['label'] == 1:
    print(f"ML model detected anomaly with score: {ml_result['score']:.3f}")
```

## Performance Tips

1. **Training Data Size**: Use at least 1000 samples for good performance
2. **Balanced Data**: Aim for 60-80% normal, 20-40% anomalous samples
3. **Pattern Variety**: Include diverse normal patterns to reduce false positives
4. **Validation**: Always validate on held-out data before deployment

## Example Scripts

See these examples for more details:

- `examples/train_anomaly_detector.py` - Training demo
- `examples/phase7_demo.py` - Rule-based anomaly detection
- `training/train_any_model.py` - General training pipeline

## Troubleshooting

### Low Recall
If the model doesn't detect enough anomalies:
- Lower the `anomaly_threshold`
- Ensure anomalous patterns in training data are sufficiently different from normal

### High False Positives
If the model flags too many normal sequences:
- Increase the `anomaly_threshold`
- Include more diverse normal patterns in training data
- Increase training data size

### Poor Accuracy
If overall accuracy is low:
- Check that training data is properly labeled
- Ensure normal and anomalous patterns are distinguishable
- Increase training data size
- Try different n-gram sizes

## API Reference

### AnomalyMLClassifier

```python
class AnomalyMLClassifier:
    def __init__(self, n: int = 3, anomaly_threshold: float = 0.3):
        """Initialize classifier.
        
        Args:
            n: N-gram size (default: 3)
            anomaly_threshold: Classification threshold (default: 0.3)
        """
    
    def train(self, train_data: List[Dict], epochs: int = 10, batch_size: int = 32):
        """Train the model on labeled sequences."""
    
    def predict(self, features: Dict) -> Dict:
        """Predict anomaly for a sequence.
        
        Returns:
            {
                'label': 0 or 1,      # 0=normal, 1=anomalous
                'score': float,       # 0.0-1.0 anomaly score
                'confidence': float   # 0.0-1.0 prediction confidence
            }
        """
    
    def save(self, filepath: str):
        """Save model to JSON file."""
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from JSON file."""
```

## License

Part of the Nethical framework. See LICENSE for details.
