# Detection Models Training

This directory contains training pipelines for Phase 3 detection models.

## Structure

```
detection_models/
├── README.md (this file)
├── behavioral/          # Behavioral detection models
│   ├── train_coordination.py
│   ├── train_mimicry.py
│   └── train_drift.py
├── multimodal/          # Multimodal detection models
│   ├── train_image.py
│   ├── train_audio.py
│   └── train_cross_modal.py
├── zeroday/             # Zero-day detection models
│   ├── train_ensemble.py
│   ├── train_invariants.py
│   └── train_chain.py
└── shared/              # Shared utilities
    ├── data_loader.py
    ├── metrics.py
    └── evaluation.py
```

## Training Process

### 1. Data Collection
- Feedback from online learning pipeline
- Red team attack corpus
- Labeled historical data
- Synthetic attack generation

### 2. Model Training
- Supervised learning for known patterns
- Unsupervised anomaly detection
- Semi-supervised for zero-day
- Online learning for continuous improvement

### 3. Evaluation
- Held-out test set validation
- A/B testing in production
- Cross-validation
- Adversarial testing

### 4. Deployment
- Model versioning
- Rollback capability
- A/B testing
- Gradual rollout

## Usage

### Train Behavioral Model
```bash
python behavioral/train_coordination.py \
  --data /path/to/training/data \
  --output /path/to/model \
  --epochs 100
```

### Train Multimodal Model
```bash
python multimodal/train_image.py \
  --data /path/to/image/data \
  --output /path/to/model \
  --backbone resnet50
```

### Train Zero-Day Ensemble
```bash
python zeroday/train_ensemble.py \
  --data /path/to/anomaly/data \
  --output /path/to/model \
  --algorithms isolation_forest,autoencoder,statistical
```

## Model Formats

Models are saved in the following formats:
- PyTorch: `.pt` or `.pth`
- ONNX: `.onnx` (for cross-platform deployment)
- TensorFlow: `.pb` (if using TF)

## Integration with Online Learning

Trained models integrate with the online learning pipeline:

1. **Initial Training**: Train on historical data
2. **Deployment**: Deploy via RollbackManager
3. **Feedback Collection**: Collect operational feedback
4. **Incremental Updates**: Retrain with new feedback
5. **A/B Testing**: Test improvements before full rollout
6. **Continuous Improvement**: Iterate based on performance

## Safety Constraints

All training respects safety constraints:
- No reduction in detection rate for critical vectors
- Minimum accuracy thresholds
- False positive rate limits
- Human review for significant changes

## Law Alignment

Training process aligns with Fundamental Laws:
- Law 24 (Adaptive Learning): Continuous improvement
- Law 25 (Ethical Evolution): Maintain ethical standards
- Law 15 (Audit Compliance): Full audit trail
- Law 23 (Fail-Safe Design): Safety constraints

## Contact

For questions about training pipelines:
- Security Team: security@nethical.ai
- ML Team: ml@nethical.ai
