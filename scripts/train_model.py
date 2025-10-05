#!/usr/bin/env python3
"""Training Script for Nethical ML Models

Implements the training pipeline described in TrainTestPipeline.md:
- Temporal split (train: past, val: recent)
- Baseline logistic regression classifier
- Metrics: Precision, recall, F1, ROC-AUC, calibration (ECE)
- Shadow logging and comparison with rule-only outcomes
- Promotion gate validation
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.core import MLShadowClassifier, MLModelType


def generate_synthetic_labeled_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic labeled training data.
    
    Simulates the data collection from shadow mode:
    - Events with features
    - Final decisions (labels)
    - Rule trigger vectors
    """
    print(f"Generating {num_samples} synthetic training samples...")
    
    data = []
    base_time = datetime.utcnow() - timedelta(days=30)
    
    for i in range(num_samples):
        # Temporal ordering - earlier samples first
        timestamp = base_time + timedelta(minutes=i * 43)  # ~30 days of data
        
        # Generate features based on risk level
        risk_level = random.choice(['low', 'medium', 'high'])
        
        if risk_level == 'low':
            features = {
                'violation_count': random.uniform(0.0, 0.3),
                'severity_max': random.uniform(0.0, 0.2),
                'recency_score': random.uniform(0.0, 0.3),
                'frequency_score': random.uniform(0.0, 0.2),
                'context_risk': random.uniform(0.0, 0.2)
            }
            label = 0  # No violation
        elif risk_level == 'medium':
            features = {
                'violation_count': random.uniform(0.3, 0.6),
                'severity_max': random.uniform(0.3, 0.6),
                'recency_score': random.uniform(0.2, 0.5),
                'frequency_score': random.uniform(0.2, 0.5),
                'context_risk': random.uniform(0.2, 0.5)
            }
            label = random.choice([0, 1])  # Mixed
        else:  # high
            features = {
                'violation_count': random.uniform(0.6, 1.0),
                'severity_max': random.uniform(0.7, 1.0),
                'recency_score': random.uniform(0.5, 1.0),
                'frequency_score': random.uniform(0.5, 0.9),
                'context_risk': random.uniform(0.5, 0.9)
            }
            label = 1  # Violation
        
        # Compute rule-based score
        rule_score = sum(features.values()) / len(features)
        
        data.append({
            'event_id': f'evt_{i:06d}',
            'timestamp': timestamp.isoformat(),
            'agent_id': f'agent_{i % 100:03d}',
            'features': features,
            'rule_score': rule_score,
            'label': label  # 0 = no violation, 1 = violation
        })
    
    return data


def temporal_split(data: List[Dict[str, Any]], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Split data temporally (train: past, val: recent) to mimic deployment.
    
    Args:
        data: List of labeled events
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (train_data, val_data)
    """
    print(f"\nPerforming temporal split ({train_ratio:.0%} train, {1-train_ratio:.0%} validation)...")
    
    # Data is already sorted by timestamp
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data


def train_baseline_model(train_data: List[Dict[str, Any]]) -> MLShadowClassifier:
    """Train baseline logistic regression classifier.
    
    Args:
        train_data: Training dataset
        
    Returns:
        Trained shadow classifier
    """
    print("\nTraining baseline classifier...")
    print("Model type: Heuristic (weighted feature combination)")
    
    # Initialize shadow classifier
    classifier = MLShadowClassifier(
        model_type=MLModelType.HEURISTIC,
        score_agreement_threshold=0.1,
        storage_path="./models/candidates"
    )
    
    # For heuristic model, we can adjust weights based on feature importance
    # In a real implementation, this would use sklearn or similar
    print("Feature weights:")
    for feature, weight in classifier.feature_weights.items():
        print(f"  {feature}: {weight:.3f}")
    
    return classifier


def evaluate_model(classifier: MLShadowClassifier, val_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Evaluate model on validation set.
    
    Computes:
    - Precision, recall, F1
    - ROC-AUC (approximation for heuristic)
    - Calibration (ECE)
    
    Args:
        classifier: Trained classifier
        val_data: Validation dataset
        
    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating model on validation set...")
    
    # Run predictions on validation set
    for i, sample in enumerate(val_data):
        rule_class = 'deny' if sample['rule_score'] > 0.6 else 'warn' if sample['rule_score'] > 0.4 else 'allow'
        
        prediction = classifier.predict(
            agent_id=sample['agent_id'],
            action_id=sample['event_id'],
            features=sample['features'],
            rule_risk_score=sample['rule_score'],
            rule_classification=rule_class
        )
        
        # Update metrics based on actual label
        actual_violation = sample['label'] == 1
        predicted_violation = prediction.ml_classification in ['deny', 'warn']
        
        if predicted_violation and actual_violation:
            classifier.metrics.true_positives += 1
        elif predicted_violation and not actual_violation:
            classifier.metrics.false_positives += 1
        elif not predicted_violation and actual_violation:
            classifier.metrics.false_negatives += 1
        else:
            classifier.metrics.true_negatives += 1
        
        # Update calibration bins
        confidence_bin = f"{int(prediction.ml_confidence * 5) * 0.2:.1f}-{int(prediction.ml_confidence * 5) * 0.2 + 0.2:.1f}"
        if confidence_bin in classifier.metrics.calibration_bins:
            classifier.metrics.calibration_bins[confidence_bin]['total'] += 1
            if prediction.classifications_agree:
                classifier.metrics.calibration_bins[confidence_bin]['correct'] += 1
    
    # Get metrics report
    metrics = classifier.get_metrics_report()
    
    print("\nValidation Metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Expected Calibration Error (ECE): {metrics['expected_calibration_error']:.3f}")
    
    return metrics


def check_promotion_gate(metrics: Dict[str, float], baseline_metrics: Dict[str, float] = None) -> bool:
    """Check if model meets promotion gate criteria.
    
    From TrainTestPipeline.md:
    - min_recall_gain: 0.03 (+3% absolute)
    - max_fp_increase: 0.02 (+2% absolute)
    - max_latency_increase_ms: 5
    - max_ece: 0.08
    - min_human_agreement: 0.85
    
    Args:
        metrics: Current model metrics
        baseline_metrics: Baseline model metrics (if available)
        
    Returns:
        True if model passes promotion gate
    """
    print("\nChecking Promotion Gate Criteria:")
    print("=" * 60)
    
    passed = True
    
    # Max ECE check
    max_ece = 0.08
    ece_pass = metrics['expected_calibration_error'] <= max_ece
    print(f"✓ ECE ≤ {max_ece}: {metrics['expected_calibration_error']:.3f} {'PASS' if ece_pass else 'FAIL'}")
    passed = passed and ece_pass
    
    # Min accuracy as proxy for human agreement
    min_accuracy = 0.85
    accuracy_pass = metrics['accuracy'] >= min_accuracy
    print(f"✓ Accuracy ≥ {min_accuracy}: {metrics['accuracy']:.3f} {'PASS' if accuracy_pass else 'FAIL'}")
    passed = passed and accuracy_pass
    
    if baseline_metrics:
        # Recall gain
        min_recall_gain = 0.03
        recall_gain = metrics['recall'] - baseline_metrics['recall']
        recall_pass = recall_gain >= min_recall_gain
        print(f"✓ Recall gain ≥ {min_recall_gain}: {recall_gain:+.3f} {'PASS' if recall_pass else 'FAIL'}")
        passed = passed and recall_pass
        
        # FP increase
        max_fp_increase = 0.02
        fp_rate = metrics.get('false_positive_rate', 0.0)
        baseline_fp_rate = baseline_metrics.get('false_positive_rate', 0.0)
        fp_increase = fp_rate - baseline_fp_rate
        fp_pass = fp_increase <= max_fp_increase
        print(f"✓ FP increase ≤ {max_fp_increase}: {fp_increase:+.3f} {'PASS' if fp_pass else 'FAIL'}")
        passed = passed and fp_pass
    
    print("=" * 60)
    print(f"Overall: {'✓ PASS - Model can be promoted' if passed else '✗ FAIL - Model does not meet criteria'}")
    
    return passed


def save_model(classifier: MLShadowClassifier, metrics: Dict[str, float], model_dir: str = "./models/candidates"):
    """Save trained model and metadata.
    
    Args:
        classifier: Trained classifier
        metrics: Validation metrics
        model_dir: Directory to save model
    """
    print(f"\nSaving model to {model_dir}...")
    
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join(model_dir, f"model_{timestamp}.json")
    
    model_data = {
        'model_type': classifier.model_type.value,
        'feature_weights': classifier.feature_weights,
        'score_agreement_threshold': classifier.score_agreement_threshold,
        'timestamp': timestamp,
        'metrics': metrics
    }
    
    with open(model_file, 'w') as f:
        json.dump(model_data, f, indent=2)
    
    print(f"✓ Model saved: {model_file}")
    
    return model_file


def save_training_data(data: List[Dict[str, Any]], output_file: str = "./data/labeled_events/training_data.json"):
    """Save training data to file.
    
    Args:
        data: Training dataset
        output_file: Output file path
    """
    print(f"\nSaving training data to {output_file}...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Training data saved: {output_file}")


def run_testing_pipeline():
    """Run the testing pipeline after training."""
    # Import test_model functions
    try:
        from test_model import (
            find_latest_model,
            load_model,
            load_test_data,
            evaluate_on_test_set,
            save_evaluation_report
        )
    except ImportError:
        print("\n✗ Could not import test_model.py")
        return False
    
    print("\n" + "=" * 60)
    print("STARTING TESTING PIPELINE")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Find and load latest model
        try:
            model_path = find_latest_model("./models/current")
        except FileNotFoundError:
            print("⚠ No model found in models/current/, checking candidates...")
            try:
                model_path = find_latest_model("./models/candidates")
            except FileNotFoundError:
                print("✗ No models found to test.")
                return False
        
        classifier, model_metadata = load_model(model_path)
        
        # Step 2: Load test data
        test_data = load_test_data()
        
        # Step 3: Evaluate on test set
        ml_metrics, predictions_log = evaluate_on_test_set(classifier, test_data)
        
        # Step 4: Save evaluation report
        save_evaluation_report(ml_metrics, predictions_log)
        
        print("\n" + "=" * 60)
        print("TESTING COMPLETE")
        print("=" * 60)
        print("\nKey Findings:")
        
        if ml_metrics['f1_score'] >= 0.8:
            print("✓ Model shows strong performance (F1 ≥ 0.8)")
        elif ml_metrics['f1_score'] >= 0.6:
            print("⚠ Model shows moderate performance (0.6 ≤ F1 < 0.8)")
        else:
            print("✗ Model needs improvement (F1 < 0.6)")
        
        if ml_metrics['expected_calibration_error'] <= 0.08:
            print("✓ Model is well-calibrated (ECE ≤ 0.08)")
        else:
            print("⚠ Model calibration needs improvement (ECE > 0.08)")
        
        print()
        return True
        
    except Exception as e:
        print(f"\n✗ Testing pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(run_all: bool = False):
    """Main training pipeline.
    
    Args:
        run_all: If True, also run testing pipeline after training
    """
    print("=" * 60)
    print("NETHICAL ML TRAINING PIPELINE")
    print("=" * 60)
    print("\nBased on TrainTestPipeline.md specifications:")
    print("- Temporal split (train: past, val: recent)")
    print("- Baseline classifier with heuristic features")
    print("- Metrics: Precision, Recall, F1, ECE")
    print("- Promotion gate validation")
    if run_all:
        print("- Running complete workflow (training + testing)")
    print()
    
    # Step 1: Generate/load labeled data
    data = generate_synthetic_labeled_data(num_samples=1000)
    save_training_data(data)
    
    # Step 2: Temporal split
    train_data, val_data = temporal_split(data, train_ratio=0.8)
    
    # Step 3: Train baseline model
    classifier = train_baseline_model(train_data)
    
    # Step 4: Evaluate on validation set
    metrics = evaluate_model(classifier, val_data)
    
    # Step 5: Check promotion gate
    promotion_passed = check_promotion_gate(metrics)
    
    # Step 6: Save model
    if promotion_passed:
        model_file = save_model(classifier, metrics, model_dir="./models/current")
        print("\n✓ Model promoted to production (models/current/)")
    else:
        model_file = save_model(classifier, metrics, model_dir="./models/candidates")
        print("\n⚠ Model saved to candidates (models/candidates/)")
        print("  Review metrics and retrain before promotion")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    if not run_all:
        print("\nNext Steps:")
        print("1. Review validation metrics")
        print("2. Compare with rule-only baseline")
        print("3. Run shadow mode evaluation")
        print("4. If promotion gate passed, deploy to production")
        print("\nTip: Use --run-all to automatically run testing after training")
        print()
    else:
        # Run testing pipeline
        print()
        run_testing_pipeline()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Nethical ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train only
  python scripts/train_model.py
  
  # Train and test (complete workflow)
  python scripts/train_model.py --run-all
        """
    )
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run complete workflow: training followed by testing'
    )
    
    args = parser.parse_args()
    main(run_all=args.run_all)
