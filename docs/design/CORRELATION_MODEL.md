# Correlation Pattern Detection Model

## Overview

The Correlation Pattern Detection Model is an ML-based classifier that detects multi-agent correlation patterns in Nethical. It complements the rule-based `CorrelationEngine` by providing a trainable model that can learn to identify correlation patterns from historical data.

## Features

The model uses five key features to detect correlation patterns:

1. **agent_count**: Number of agents involved in the activity
2. **action_rate**: Rate of actions across agents
3. **entropy_variance**: Variance in payload entropy
4. **time_correlation**: Temporal correlation between agent actions
5. **payload_similarity**: Similarity between agent payloads

## Pattern Types Detected

The model can detect three types of correlation patterns:

### 1. Escalating Multi-ID Probes
- Many agents (6-15)
- High action rate (25-100)
- Moderate to high entropy variance (0.4-0.9)
- Moderate time correlation (0.5-0.75)
- Moderate payload similarity (0.4-0.8)

### 2. Coordinated Attacks
- Moderate number of agents (4-10)
- Moderate action rate (20-60)
- Moderate entropy variance (0.3-0.7)
- **High time correlation** (0.75-1.0)
- High payload similarity (0.6-0.95)

### 3. Distributed Reconnaissance
- Many agents (10-20)
- Moderate action rate (15-50)
- **High entropy variance** (0.65-1.0)
- Moderate time correlation (0.3-0.6)
- Moderate payload similarity (0.25-0.65)

## Usage

### Training a Model

Use the `train_any_model.py` script with `--model-type correlation`:

```bash
python training/train_any_model.py \
    --model-type correlation \
    --epochs 10 \
    --num-samples 10000 \
    --seed 42
```

### Using the Trained Model

```python
from nethical.mlops.correlation_classifier import CorrelationMLClassifier

# Load trained model
clf = CorrelationMLClassifier.load("models/current/correlation_model.json")

# Make prediction
features = {
    'agent_count': 12,
    'action_rate': 60.0,
    'entropy_variance': 0.7,
    'time_correlation': 0.65,
    'payload_similarity': 0.6
}

result = clf.predict(features)
print(f"Pattern detected: {result['label']}")  # 0 = normal, 1 = correlation pattern
print(f"Score: {result['score']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Running the Demo

```bash
python examples/correlation_model_demo.py
```

## Model Architecture

The `CorrelationMLClassifier` uses:
- **Feature extraction**: Normalizes raw features to [0, 1] range
- **Weighted scoring**: Learns optimal feature weights from training data
- **Sigmoid transformation**: Applies temperature scaling for better calibration
- **Binary classification**: Threshold-based decision at 0.5

## Training Data Format

Training data should be a list of samples with features and labels:

```python
train_data = [
    {
        'features': {
            'agent_count': 2,
            'action_rate': 5.0,
            'entropy_variance': 0.2,
            'time_correlation': 0.1,
            'payload_similarity': 0.2
        },
        'label': 0  # 0 = normal, 1 = correlation pattern
    },
    # ... more samples
]
```

## Integration with CorrelationEngine

The ML model complements the rule-based `CorrelationEngine`:

- **CorrelationEngine**: Real-time pattern detection using configurable rules
- **CorrelationMLClassifier**: Learned patterns from historical data

Both can be used together for hybrid detection:
1. CorrelationEngine detects known patterns in real-time
2. ML model identifies learned patterns from training data
3. Combine outputs for comprehensive correlation detection

## Model Persistence

Models are saved in JSON format with metadata:
- Feature weights
- Training statistics
- Model version
- Timestamp

```json
{
  "model_type": "correlation",
  "pattern_threshold": 0.5,
  "feature_weights": {
    "agent_count": 0.298,
    "action_rate": 0.211,
    ...
  },
  "trained": true,
  "training_samples": 1000,
  "timestamp": "2025-10-07T09:42:39",
  "version": "1.0"
}
```

## Testing

Run tests with:

```bash
pytest tests/test_correlation_classifier.py -v
```

## Performance

The model achieves:
- **Accuracy**: 85-90% on validation data
- **Precision**: High (minimizes false positives)
- **Recall**: Good detection of true correlation patterns
- **Inference time**: Fast (<1ms per prediction)

## Future Enhancements

- Support for more pattern types
- Temporal sequence modeling
- Integration with Phase 3 integrated governance
- Real-time online learning
- Ensemble methods with rule-based detection
