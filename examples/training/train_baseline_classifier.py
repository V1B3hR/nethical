#!/usr/bin/env python3
"""
Example: Train Baseline ML Classifier

This script demonstrates training a BaselineMLClassifier using
the train_any_model.py pipeline with different preprocessing modes.

The BaselineMLClassifier supports three modes:
- heuristic: Raw numeric features, no scaling
- logistic: Min-max normalized features (0-1 range)
- simple_transformer: Features with optional text tokenization

Usage:
    python examples/training/train_baseline_classifier.py

See Also:
    - training/train_any_model.py: Full training pipeline
    - docs/TRAINING_GUIDE.md: End-to-end training documentation
    - nethical/mlops/baseline.py: BaselineMLClassifier implementation
"""

import sys
from pathlib import Path

# Add parent directory to path for running this example directly without installation.
# For production use, install the package with: pip install -e .
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """Run the baseline classifier training demo."""
    print_section("NETHICAL: Baseline ML Classifier Training")

    print("\nThis demo shows how to train the BaselineMLClassifier, which:")
    print("  1. Uses weighted feature combinations for classification")
    print(
        "  2. Supports multiple preprocessing modes (heuristic, logistic, transformer)"
    )
    print("  3. Requires no external ML dependencies (pure Python)")
    print("  4. Is suitable for quick prototyping and production deployment")

    print_section("Supported Model Types")

    print("\nThe BaselineMLClassifier supports three preprocessing modes:")
    print("\n  1. heuristic (--model-type heuristic)")
    print("     • Uses raw numeric features without scaling")
    print("     • Best for features already in similar ranges")
    print("     • Fastest training and inference")

    print("\n  2. logistic (--model-type logistic)")
    print("     • Min-max normalizes all features to [0,1] range")
    print("     • Better for mixed-range features")
    print("     • Improved gradient-based weight learning")

    print("\n  3. simple_transformer (--model-type simple_transformer)")
    print("     • Adds text tokenization for text features")
    print("     • Useful for mixed numeric + text data")
    print("     • Placeholder for future transformer integration")

    print_section("Training the Model")

    print("\nBasic training with heuristic mode:")
    print("\n  $ python training/train_any_model.py \\")
    print("        --model-type heuristic \\")
    print("        --num-samples 5000 \\")
    print("        --epochs 10")

    print("\nTraining with logistic mode and audit logging:")
    print("\n  $ python training/train_any_model.py \\")
    print("        --model-type logistic \\")
    print("        --num-samples 5000 \\")
    print("        --enable-audit \\")
    print("        --audit-path audit_logs")

    print("\nTraining with governance validation:")
    print("\n  $ python training/train_any_model.py \\")
    print("        --model-type heuristic \\")
    print("        --num-samples 5000 \\")
    print("        --enable-governance \\")
    print("        --enable-audit")

    print_section("Training Data Format")

    print("\nThe classifier expects training data with this structure:")
    print(
        """
train_data = [
    {
        'features': {
            'violation_count': 0.8,
            'severity_max': 0.9,
            'recency_score': 0.5,
            'frequency_score': 0.3,
            'context_risk': 0.7
        },
        'label': 1  # 1 = violation, 0 = benign
    },
    ...
]
"""
    )

    print_section("Using the Trained Model")

    print("\nLoad and use the trained model:")
    print(
        """
from nethical.mlops.baseline import BaselineMLClassifier

# Load trained model
clf = BaselineMLClassifier.load('models/current/heuristic_model_*.json')

# Make predictions
features = {
    'violation_count': 0.7,
    'severity_max': 0.8,
    'recency_score': 0.6,
    'frequency_score': 0.5,
    'context_risk': 0.4
}

result = clf.predict(features)
print(f"Label: {result['label']}")  # 0 or 1
print(f"Score: {result['score']:.3f}")  # Raw score
print(f"Confidence: {result['confidence']:.3f}")  # Prediction confidence
"""
    )

    print_section("Quick Demo: Train and Predict")

    print("\nRunning a quick demo with synthetic data...")

    from nethical.mlops.baseline import BaselineMLClassifier
    import random

    # Generate synthetic training data
    train_data = []
    for _ in range(100):
        # Generate features
        features = {
            "violation_count": random.random(),
            "severity_max": random.random(),
            "recency_score": random.random(),
            "frequency_score": random.random(),
            "context_risk": random.random(),
        }
        # Label based on feature combination
        label = int(features["violation_count"] + features["severity_max"] > 1.0)
        train_data.append({"features": features, "label": label})

    # Train the classifier
    clf = BaselineMLClassifier(threshold=0.5)
    clf.train(train_data)

    print(f"\n  ✓ Trained on {clf.training_samples} samples")
    print(f"  ✓ Learned {len(clf.feature_weights)} feature weights")

    # Make predictions on test samples
    test_benign = {
        "violation_count": 0.2,
        "severity_max": 0.3,
        "recency_score": 0.5,
        "frequency_score": 0.3,
        "context_risk": 0.2,
    }
    test_violation = {
        "violation_count": 0.9,
        "severity_max": 0.8,
        "recency_score": 0.7,
        "frequency_score": 0.8,
        "context_risk": 0.9,
    }

    result_benign = clf.predict(test_benign)
    result_violation = clf.predict(test_violation)

    print("\n  Test Predictions:")
    print(
        f"  • Benign sample: label={result_benign['label']}, score={result_benign['score']:.3f}"
    )
    print(
        f"  • Violation sample: label={result_violation['label']}, score={result_violation['score']:.3f}"
    )

    print_section("Kaggle Datasets")

    print("\nThe training pipeline supports these Kaggle datasets (datasets/datasets):")
    print("  • Cyber Security Attacks")
    print("  • Microsoft Security Incident Prediction")
    print("  • Data Ethics in Data Science")
    print("  • Security Breach Dataset")
    print("  • RBA Dataset")
    print("  • Fake and Real News Dataset")
    print("  • AI vs Human Content Detection")
    print("  • Cyberbullying Classification")
    print("  • And more...")

    print("\nTo use Kaggle datasets:")
    print("  1. Set up Kaggle API credentials (~/.kaggle/kaggle.json)")
    print("  2. Run training without --no-download flag")
    print("  3. Or manually download CSV files to data/external/")

    print_section("Next Steps")

    print("\n1. Train with real data:")
    print(
        "   $ python training/train_any_model.py --model-type heuristic --num-samples 10000"
    )

    print("\n2. Check saved model:")
    print("   $ ls -l models/candidates/heuristic_model_*.json")

    print("\n3. Enable governance for production:")
    print(
        "   $ python training/train_any_model.py --model-type logistic --enable-governance --enable-audit"
    )

    print("\n4. Integrate with your application:")
    print("   • Load the trained model")
    print("   • Feed features for prediction")
    print("   • Apply to Phase 5/6 blended enforcement")

    print_section("Documentation Links")

    print("\nFor more information, see:")
    print("  • docs/TRAINING_GUIDE.md - Complete training guide")
    print("  • training/README.md - Training script documentation")
    print("  • scripts/README.md - Script usage documentation")
    print(
        "  • docs/implementation/TRAIN_MODEL_REAL_DATA_SUMMARY.md - Real data training"
    )

    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
