#!/usr/bin/env python3
"""Testing Script for Nethical ML Models

Implements the testing pipeline described in TrainTestPipeline.md:
- Load trained model
- Run on test dataset
- Compute comprehensive metrics (Precision, Recall, F1, ROC-AUC, ECE)
- Compare with rule-based baseline
- Generate evaluation report
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import glob

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.core import MLShadowClassifier, MLModelType


def load_model(model_path: str) -> tuple:
    """Load trained model from file.
    
    Args:
        model_path: Path to model JSON file
        
    Returns:
        Tuple of (classifier, model_metadata)
    """
    print(f"Loading model from {model_path}...")
    
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    
    # Reconstruct classifier
    classifier = MLShadowClassifier(
        model_type=MLModelType(model_data['model_type']),
        score_agreement_threshold=model_data['score_agreement_threshold'],
        storage_path="./models/current"
    )
    
    # Restore feature weights
    classifier.feature_weights = model_data['feature_weights']
    
    print(f"✓ Model loaded: {model_data['model_type']}")
    print(f"  Timestamp: {model_data['timestamp']}")
    
    return classifier, model_data


def load_test_data(data_path: str = "./data/labeled_events/training_data.json") -> List[Dict[str, Any]]:
    """Load test dataset.
    
    Args:
        data_path: Path to test data
        
    Returns:
        List of test samples
    """
    print(f"\nLoading test data from {data_path}...")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Use last 20% as test set
    test_size = int(len(data) * 0.2)
    test_data = data[-test_size:]
    
    print(f"✓ Loaded {len(test_data)} test samples")
    
    return test_data


def evaluate_on_test_set(classifier: MLShadowClassifier, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Run comprehensive evaluation on test set.
    
    Args:
        classifier: Trained classifier
        test_data: Test dataset
        
    Returns:
        Dictionary of metrics
    """
    print("\nRunning evaluation on test set...")
    print("=" * 60)
    
    # Reset metrics
    classifier.metrics.true_positives = 0
    classifier.metrics.true_negatives = 0
    classifier.metrics.false_positives = 0
    classifier.metrics.false_negatives = 0
    classifier.metrics.total_predictions = 0
    classifier.metrics.score_agreement_count = 0
    classifier.metrics.classification_agreement_count = 0
    
    predictions_log = []
    
    for sample in test_data:
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
        
        classifier.metrics.total_predictions += 1
        
        # Update calibration bins
        confidence_bin = f"{int(prediction.ml_confidence * 5) * 0.2:.1f}-{int(prediction.ml_confidence * 5) * 0.2 + 0.2:.1f}"
        if confidence_bin in classifier.metrics.calibration_bins:
            classifier.metrics.calibration_bins[confidence_bin]['total'] += 1
            if prediction.classifications_agree:
                classifier.metrics.calibration_bins[confidence_bin]['correct'] += 1
        
        # Log prediction
        predictions_log.append({
            'event_id': sample['event_id'],
            'actual_label': sample['label'],
            'ml_prediction': prediction.ml_classification,
            'ml_score': prediction.ml_risk_score,
            'rule_score': sample['rule_score'],
            'correct': (predicted_violation == actual_violation)
        })
    
    # Get metrics
    metrics = classifier.get_metrics_report()
    
    # Print detailed results
    print("\nTest Set Performance:")
    print("-" * 60)
    print(f"Total Predictions: {metrics['total_predictions']}")
    print()
    print("Classification Metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print()
    print("Calibration:")
    print(f"  Expected Calibration Error (ECE): {metrics['expected_calibration_error']:.3f}")
    print()
    print("Confusion Matrix:")
    print(f"  True Positives:  {metrics['confusion_matrix']['true_positives']}")
    print(f"  True Negatives:  {metrics['confusion_matrix']['true_negatives']}")
    print(f"  False Positives: {metrics['confusion_matrix']['false_positives']}")
    print(f"  False Negatives: {metrics['confusion_matrix']['false_negatives']}")
    print()
    print("Agreement with Rule-Based System:")
    print(f"  Score Agreement: {metrics['score_agreement_rate']:.1%}")
    print(f"  Classification Agreement: {metrics['classification_agreement_rate']:.1%}")
    print("=" * 60)
    
    return metrics, predictions_log


def compare_with_baseline(ml_metrics: Dict[str, float], rule_baseline: Dict[str, float]):
    """Compare ML model with rule-based baseline.
    
    Args:
        ml_metrics: ML model metrics
        rule_baseline: Rule-based baseline metrics
    """
    print("\nComparison with Rule-Based Baseline:")
    print("=" * 60)
    print(f"{'Metric':<30} {'Rule-Based':<15} {'ML Model':<15} {'Δ':<10}")
    print("-" * 60)
    
    metrics_to_compare = ['precision', 'recall', 'f1_score', 'accuracy']
    
    for metric in metrics_to_compare:
        baseline_val = rule_baseline.get(metric, 0.0)
        ml_val = ml_metrics.get(metric, 0.0)
        delta = ml_val - baseline_val
        delta_str = f"{delta:+.3f}"
        
        print(f"{metric.replace('_', ' ').title():<30} {baseline_val:<15.3f} {ml_val:<15.3f} {delta_str:<10}")
    
    print("=" * 60)


def save_evaluation_report(metrics: Dict[str, float], predictions_log: List[Dict], 
                          output_dir: str = "./data/labeled_events"):
    """Save evaluation report to file.
    
    Args:
        metrics: Evaluation metrics
        predictions_log: Detailed prediction log
        output_dir: Output directory
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    report = {
        'timestamp': timestamp,
        'metrics': metrics,
        'predictions': predictions_log[:100]  # Save first 100 predictions
    }
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Evaluation report saved: {report_file}")


def find_latest_model(model_dir: str = "./models/current") -> str:
    """Find the latest model in the directory.
    
    Args:
        model_dir: Directory to search
        
    Returns:
        Path to latest model file
    """
    model_files = glob.glob(os.path.join(model_dir, "model_*.json"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    # Sort by filename (which includes timestamp)
    latest_model = sorted(model_files)[-1]
    
    return latest_model


def main():
    """Main testing pipeline."""
    print("=" * 60)
    print("NETHICAL ML TESTING PIPELINE")
    print("=" * 60)
    print("\nComprehensive model evaluation:")
    print("- Load trained model")
    print("- Run on test dataset")
    print("- Compute metrics (Precision, Recall, F1, ECE)")
    print("- Compare with baseline")
    print("- Generate evaluation report")
    print()
    
    # Step 1: Find and load latest model
    try:
        model_path = find_latest_model("./models/current")
    except FileNotFoundError:
        print("⚠ No model found in models/current/, checking candidates...")
        try:
            model_path = find_latest_model("./models/candidates")
        except FileNotFoundError:
            print("✗ No models found. Please run train_model.py first.")
            sys.exit(1)
    
    classifier, model_metadata = load_model(model_path)
    
    # Step 2: Load test data
    try:
        test_data = load_test_data()
    except FileNotFoundError:
        print("✗ Test data not found. Please run train_model.py first to generate data.")
        sys.exit(1)
    
    # Step 3: Evaluate on test set
    ml_metrics, predictions_log = evaluate_on_test_set(classifier, test_data)
    
    # Step 4: Compare with rule-based baseline (if available)
    if 'metrics' in model_metadata:
        print("\nNote: Comparing with training metrics from model metadata")
    
    # Step 5: Save evaluation report
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


if __name__ == "__main__":
    main()
