#!/usr/bin/env python3
"""
Example demonstrating correlation model training and usage.

This script shows how to:
1. Train a correlation pattern detection model
2. Make predictions on new multi-agent activity data
3. Save and load trained models
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.mlops.correlation_classifier import CorrelationMLClassifier


def generate_sample_data():
    """Generate sample training data for correlation patterns."""
    import random

    train_data = []

    # Normal patterns (no correlation) - 60 samples
    print("Generating normal multi-agent activity samples...")
    for _ in range(60):
        features = {
            "agent_count": random.randint(1, 3),
            "action_rate": random.uniform(1, 10),
            "entropy_variance": random.uniform(0.05, 0.3),
            "time_correlation": random.uniform(0, 0.25),
            "payload_similarity": random.uniform(0, 0.3),
        }
        train_data.append({"features": features, "label": 0})  # No correlation pattern

    # Correlation patterns - 40 samples
    print("Generating correlation pattern samples...")
    for _ in range(40):
        pattern_type = random.choice(["escalating", "coordinated", "distributed"])

        if pattern_type == "escalating":
            # Escalating multi-ID probes
            features = {
                "agent_count": random.randint(6, 15),
                "action_rate": random.uniform(25, 100),
                "entropy_variance": random.uniform(0.4, 0.9),
                "time_correlation": random.uniform(0.5, 0.75),
                "payload_similarity": random.uniform(0.4, 0.8),
            }
        elif pattern_type == "coordinated":
            # Coordinated attack
            features = {
                "agent_count": random.randint(4, 10),
                "action_rate": random.uniform(20, 60),
                "entropy_variance": random.uniform(0.3, 0.7),
                "time_correlation": random.uniform(0.75, 1.0),
                "payload_similarity": random.uniform(0.6, 0.95),
            }
        else:  # distributed
            # Distributed reconnaissance
            features = {
                "agent_count": random.randint(10, 20),
                "action_rate": random.uniform(15, 50),
                "entropy_variance": random.uniform(0.65, 1.0),
                "time_correlation": random.uniform(0.3, 0.6),
                "payload_similarity": random.uniform(0.25, 0.65),
            }

        train_data.append(
            {"features": features, "label": 1}  # Correlation pattern detected
        )

    # Shuffle the data
    random.shuffle(train_data)

    return train_data


def main():
    print("=" * 70)
    print("CORRELATION PATTERN DETECTION MODEL - DEMO")
    print("=" * 70)

    # 1. Generate sample data
    print("\n1. Generating sample training data...")
    train_data = generate_sample_data()
    print(f"   Generated {len(train_data)} training samples")

    # 2. Initialize and train the model
    print("\n2. Training correlation pattern detection model...")
    clf = CorrelationMLClassifier(pattern_threshold=0.5)
    clf.train(train_data)
    print(f"   Model trained on {clf.training_samples} samples")
    print(f"   Feature weights:")
    for name, weight in clf.feature_weights.items():
        print(f"     {name}: {weight:.3f}")

    # 3. Make predictions on test cases
    print("\n3. Making predictions on test cases...")

    # Test case 1: Normal activity
    print("\n   Test Case 1: Normal multi-agent activity")
    normal_features = {
        "agent_count": 2,
        "action_rate": 5.0,
        "entropy_variance": 0.15,
        "time_correlation": 0.1,
        "payload_similarity": 0.2,
    }
    result1 = clf.predict(normal_features)
    print(f"     Features: {normal_features}")
    print(
        f"     Prediction: {'Correlation Pattern' if result1['label'] == 1 else 'Normal'}"
    )
    print(f"     Score: {result1['score']:.3f}")
    print(f"     Confidence: {result1['confidence']:.3f}")

    # Test case 2: Escalating probes
    print("\n   Test Case 2: Escalating multi-ID probes")
    escalating_features = {
        "agent_count": 12,
        "action_rate": 60.0,
        "entropy_variance": 0.7,
        "time_correlation": 0.65,
        "payload_similarity": 0.6,
    }
    result2 = clf.predict(escalating_features)
    print(f"     Features: {escalating_features}")
    print(
        f"     Prediction: {'Correlation Pattern' if result2['label'] == 1 else 'Normal'}"
    )
    print(f"     Score: {result2['score']:.3f}")
    print(f"     Confidence: {result2['confidence']:.3f}")

    # Test case 3: Coordinated attack
    print("\n   Test Case 3: Coordinated attack pattern")
    coordinated_features = {
        "agent_count": 7,
        "action_rate": 40.0,
        "entropy_variance": 0.5,
        "time_correlation": 0.9,  # High time correlation
        "payload_similarity": 0.8,
    }
    result3 = clf.predict(coordinated_features)
    print(f"     Features: {coordinated_features}")
    print(
        f"     Prediction: {'Correlation Pattern' if result3['label'] == 1 else 'Normal'}"
    )
    print(f"     Score: {result3['score']:.3f}")
    print(f"     Confidence: {result3['confidence']:.3f}")

    # 4. Save the model
    print("\n4. Saving trained model...")
    model_path = "/tmp/correlation_model_demo.json"
    clf.save(model_path)
    print(f"   Model saved to: {model_path}")

    # 5. Load and verify the model
    print("\n5. Loading model and verifying...")
    loaded_clf = CorrelationMLClassifier.load(model_path)
    print(f"   Model loaded successfully")
    print(f"   Trained: {loaded_clf.trained}")
    print(f"   Training samples: {loaded_clf.training_samples}")

    # Verify predictions match
    result_loaded = loaded_clf.predict(escalating_features)
    print(
        f"   Verification: Loaded model prediction matches original: {result_loaded['label'] == result2['label']}"
    )

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  ✓ Correlation models detect multi-agent pattern anomalies")
    print("  ✓ Features include agent count, action rates, entropy, and timing")
    print("  ✓ Models can be trained, saved, and loaded for production use")
    print("  ✓ Use with train_any_model.py: --model-type correlation")
    print()


if __name__ == "__main__":
    main()
