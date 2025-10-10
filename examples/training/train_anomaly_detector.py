#!/usr/bin/env python3
"""Example: Train Anomaly Detection Model

This script demonstrates training an ML-based anomaly detector using
the train_any_model.py pipeline.

Usage:
    python examples/train_anomaly_detector.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """Run the anomaly detector training demo."""
    print_section("NETHICAL: Anomaly Detection Model Training")
    
    print("\nThis demo shows how to train an ML-based anomaly detector that:")
    print("  1. Learns patterns from normal and anomalous action sequences")
    print("  2. Detects unusual behavior in agent actions")
    print("  3. Can be deployed to production for real-time detection")
    
    print_section("Model Overview")
    
    print("\nThe AnomalyMLClassifier uses:")
    print("  • N-gram analysis to detect unusual action sequences")
    print("  • Action frequency analysis to spot rare actions")
    print("  • Entropy analysis to detect repetitive patterns")
    print("  • Pattern diversity analysis for behavioral profiling")
    
    print_section("Training the Model")
    
    print("\nTo train the anomaly detection model, run:")
    print("\n  $ python training/train_any_model.py --model-type anomaly --num-samples 5000")
    
    print("\nOptional parameters:")
    print("  --epochs N         : Number of training epochs (default: 10)")
    print("  --batch-size N     : Batch size for training (default: 32)")
    print("  --num-samples N    : Number of training samples (default: 10000)")
    print("  --seed N           : Random seed for reproducibility (default: 42)")
    
    print_section("Training Data")
    
    print("\nThe model trains on sequences of actions:")
    print("\nNormal patterns (70% of data):")
    print("  • ['read', 'process', 'write']")
    print("  • ['fetch', 'transform', 'load']")
    print("  • ['query', 'filter', 'aggregate']")
    print("  • ['request', 'authenticate', 'respond']")
    
    print("\nAnomalous patterns (30% of data):")
    print("  • ['delete', 'exfiltrate', 'cover_tracks']")
    print("  • ['escalate', 'access', 'exfiltrate']")
    print("  • ['scan', 'exploit', 'inject']")
    print("  • ['brute_force', 'access', 'modify']")
    
    print_section("Model Performance")
    
    print("\nExpected metrics on validation set:")
    print("  • Precision: ~100%")
    print("  • Recall: ~100%")
    print("  • Accuracy: ~100%")
    print("  • F1 Score: ~100%")
    
    print("\nThe model should pass the promotion gate:")
    print("  ✓ ECE (calibration error) <= 0.08")
    print("  ✓ Accuracy >= 0.85")
    
    print_section("Using the Trained Model")
    
    print("\nLoad and use the trained model:")
    print("""
from nethical.mlops.anomaly_classifier import AnomalyMLClassifier

# Load trained model
clf = AnomalyMLClassifier.load('models/current/anomaly_model_*.json')

# Predict on a new sequence
sequence = ['read', 'process', 'write']
result = clf.predict({'sequence': sequence})

print(f"Sequence: {sequence}")
print(f"Anomalous: {result['label'] == 1}")
print(f"Score: {result['score']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
""")
    
    print_section("Integration with Anomaly Detector")
    
    print("\nThe trained ML model can be integrated with the existing")
    print("anomaly_detector.py for enhanced detection:")
    print("\n  1. Use the rule-based detector for real-time sequence tracking")
    print("  2. Use the ML model for deeper behavioral analysis")
    print("  3. Combine both for robust anomaly detection")
    
    print_section("Next Steps")
    
    print("\n1. Train the model:")
    print("   $ python training/train_any_model.py --model-type anomaly --num-samples 10000")
    
    print("\n2. Check the saved model:")
    print("   $ ls -l models/current/anomaly_model_*.json")
    
    print("\n3. Integrate with your application:")
    print("   • Load the trained model")
    print("   • Feed action sequences for prediction")
    print("   • Trigger alerts on high anomaly scores")
    
    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    main()
