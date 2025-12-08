"""Phase 5 Demo: ML Shadow Mode

Demonstrates:
- Passive ML inference alongside rule-based decisions
- Prediction logging and comparison
- Baseline metrics collection (precision, recall, F1, calibration)
- No impact on enforcement decisions
"""

import random
from nethical.core import MLShadowClassifier, MLModelType


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")


def simulate_action_features(agent_id, risk_level="low"):
    """Simulate feature extraction for an action."""
    base_features = {
        "violation_count": 0.0,
        "severity_max": 0.0,
        "recency_score": 0.0,
        "frequency_score": 0.0,
        "context_risk": 0.0,
    }

    if risk_level == "low":
        base_features["violation_count"] = random.uniform(0.0, 0.3)
        base_features["severity_max"] = random.uniform(0.0, 0.2)
    elif risk_level == "medium":
        base_features["violation_count"] = random.uniform(0.3, 0.6)
        base_features["severity_max"] = random.uniform(0.3, 0.6)
        base_features["recency_score"] = random.uniform(0.2, 0.5)
    else:  # high
        base_features["violation_count"] = random.uniform(0.6, 1.0)
        base_features["severity_max"] = random.uniform(0.7, 1.0)
        base_features["recency_score"] = random.uniform(0.5, 1.0)
        base_features["frequency_score"] = random.uniform(0.5, 0.9)

    return base_features


def simulate_rule_based_decision(features):
    """Simulate rule-based risk scoring and classification."""
    # Simple rule-based scoring
    risk_score = (
        features["violation_count"] * 0.4
        + features["severity_max"] * 0.3
        + features["recency_score"] * 0.2
        + features["frequency_score"] * 0.1
    )

    # Classification
    if risk_score >= 0.7:
        classification = "deny"
    elif risk_score >= 0.4:
        classification = "warn"
    else:
        classification = "allow"

    return risk_score, classification


def demo_shadow_mode():
    """Demonstrate ML shadow mode."""
    print_section("Phase 5: ML Shadow Mode Demo")

    # Initialize shadow classifier
    print("1. Initializing ML Shadow Classifier...")
    shadow_classifier = MLShadowClassifier(
        model_type=MLModelType.HEURISTIC,
        score_agreement_threshold=0.1,
        storage_path="./demo_data/phase5_shadow",
    )
    print("✓ Shadow classifier initialized")
    print(f"  Model type: {shadow_classifier.model_type.value}")
    print(f"  Agreement threshold: {shadow_classifier.score_agreement_threshold}")

    # Simulate various scenarios
    print("\n2. Running Shadow Predictions (Passive Mode)...")
    print("   ML predictions are logged but DO NOT affect decisions\n")

    scenarios = [
        ("agent_001", "low", 50),
        ("agent_002", "medium", 30),
        ("agent_003", "high", 20),
    ]

    for agent_id, risk_level, count in scenarios:
        print(f"   Scenario: {agent_id} ({risk_level} risk) - {count} actions")

        for i in range(count):
            action_id = f"{agent_id}_action_{i}"

            # Extract features
            features = simulate_action_features(agent_id, risk_level)

            # Get rule-based decision (this is what actually enforces)
            rule_score, rule_class = simulate_rule_based_decision(features)

            # Get shadow prediction (passive - no enforcement)
            prediction = shadow_classifier.predict(
                agent_id=agent_id,
                action_id=action_id,
                features=features,
                rule_risk_score=rule_score,
                rule_classification=rule_class,
            )

            # Show sample predictions
            if i < 3:
                print(
                    f"     Action {i}: Rule={rule_class}({rule_score:.3f}), "
                    f"ML={prediction.ml_classification}({prediction.ml_risk_score:.3f}), "
                    f"Agree={prediction.classifications_agree}"
                )

    print("\n✓ Shadow predictions completed")
    print(f"  Total predictions logged: {shadow_classifier.metrics.total_predictions}")

    # Show metrics
    print("\n3. Shadow Mode Metrics (Baseline Collection)...")
    metrics = shadow_classifier.get_metrics_report()

    print(f"\n   Classification Performance:")
    print(f"   - Precision: {metrics['precision']:.3f}")
    print(f"   - Recall: {metrics['recall']:.3f}")
    print(f"   - F1 Score: {metrics['f1_score']:.3f}")
    print(f"   - Accuracy: {metrics['accuracy']:.3f}")

    print(f"\n   Agreement with Rule-Based System:")
    print(f"   - Score Agreement: {metrics['score_agreement_rate']*100:.1f}%")
    print(
        f"   - Classification Agreement: {metrics['classification_agreement_rate']*100:.1f}%"
    )

    print(f"\n   Calibration:")
    print(
        f"   - Expected Calibration Error (ECE): {metrics['expected_calibration_error']:.3f}"
    )

    print(f"\n   Confusion Matrix (vs Rule-Based):")
    cm = metrics["confusion_matrix"]
    print(f"   - True Positives: {cm['true_positives']}")
    print(f"   - True Negatives: {cm['true_negatives']}")
    print(f"   - False Positives: {cm['false_positives']}")
    print(f"   - False Negatives: {cm['false_negatives']}")

    # Export predictions
    print("\n4. Exporting Predictions for Analysis...")
    predictions = shadow_classifier.export_predictions(limit=10)
    print(f"✓ Exported {len(predictions)} recent predictions")

    # Show sample
    if predictions:
        print(f"\n   Sample Prediction:")
        sample = predictions[0]
        print(f"   - Agent: {sample['agent_id']}")
        print(f"   - Timestamp: {sample['timestamp']}")
        print(
            f"   - Rule Classification: {sample['rule_classification']} ({sample['rule_risk_score']:.3f})"
        )
        print(
            f"   - ML Classification: {sample['ml_classification']} ({sample['ml_risk_score']:.3f})"
        )
        print(f"   - Classifications Agree: {sample['classifications_agree']}")

    print("\n5. Key Characteristics of Shadow Mode:")
    print("   ✓ ML predictions are PASSIVE ONLY")
    print("   ✓ No impact on actual enforcement decisions")
    print("   ✓ Predictions logged alongside rule-based outcomes")
    print("   ✓ Baseline metrics collected for future evaluation")
    print("   ✓ Safe environment to validate ML model behavior")

    return shadow_classifier


def demo_model_comparison():
    """Demonstrate comparing different models in shadow mode."""
    print_section("Model Comparison in Shadow Mode")

    print("Running multiple models in parallel (all passive)...\n")

    # Create two shadow classifiers with different configs
    models = {
        "heuristic": MLShadowClassifier(
            model_type=MLModelType.HEURISTIC, score_agreement_threshold=0.1
        ),
        "logistic": MLShadowClassifier(
            model_type=MLModelType.LOGISTIC, score_agreement_threshold=0.1
        ),
    }

    # Run same scenarios through both
    print("Testing with same inputs...")
    for i in range(50):
        agent_id = f"test_agent_{i % 5}"
        action_id = f"action_{i}"
        features = simulate_action_features(agent_id, "medium")
        rule_score, rule_class = simulate_rule_based_decision(features)

        for model_name, classifier in models.items():
            classifier.predict(
                agent_id=agent_id,
                action_id=action_id,
                features=features,
                rule_risk_score=rule_score,
                rule_classification=rule_class,
            )

    # Compare metrics
    print("✓ Predictions completed\n")
    print("Model Comparison:")
    for model_name, classifier in models.items():
        metrics = classifier.get_metrics_report()
        print(f"\n  {model_name.upper()}:")
        print(f"    - F1 Score: {metrics['f1_score']:.3f}")
        print(
            f"    - Agreement Rate: {metrics['classification_agreement_rate']*100:.1f}%"
        )
        print(f"    - ECE: {metrics['expected_calibration_error']:.3f}")

    print("\n✓ Both models evaluated safely in shadow mode")
    print("  No enforcement decisions were affected")


if __name__ == "__main__":
    # Run shadow mode demo
    classifier = demo_shadow_mode()

    # Run model comparison
    demo_model_comparison()

    print("\n" + "=" * 60)
    print("Phase 5 Demo Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("- Review logged predictions")
    print("- Analyze baseline metrics")
    print("- Compare ML vs rule-based performance")
    print("- Prepare for Phase 6 (ML Assisted Enforcement)")
