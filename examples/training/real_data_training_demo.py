#!/usr/bin/env python3
"""Demo: End-to-End Real Data Training Pipeline

This script demonstrates the complete training pipeline using real-world security datasets.

Usage:
    python examples/real_data_training_demo.py
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
    """Run the real data training demo."""
    print_section("NETHICAL: Real Data Training Pipeline Demo")
    
    print("\nThis demo showcases the end-to-end training orchestrator that:")
    print("  1. Downloads real-world security datasets from Kaggle")
    print("  2. Processes each dataset with specialized processors")
    print("  3. Merges all data into a unified training set")
    print("  4. Trains the BaselineMLClassifier")
    print("  5. Evaluates and saves the trained model")
    
    print_section("Step 1: Setup")
    
    print("\nDatasets used (from datasets/datasets):")
    datasets_file = Path("datasets/datasets")
    if datasets_file.exists():
        with open(datasets_file, 'r') as f:
            for i, line in enumerate(f, 1):
                url = line.strip()
                if url and url.startswith('http'):
                    print(f"  {i}. {url}")
    else:
        print("  No datasets file found")
    
    print("\nTo download datasets automatically:")
    print("  1. Install Kaggle API: pip install kaggle")
    print("  2. Get credentials from https://www.kaggle.com/account")
    print("  3. Save to ~/.kaggle/kaggle.json")
    
    print_section("Step 2: Dataset Processing")
    
    print("\nDataset processors available:")
    print("  - CyberSecurityAttacksProcessor: For network attack datasets")
    print("  - MicrosoftSecurityProcessor: For Microsoft incident datasets")
    print("  - GenericSecurityProcessor: For any security dataset")
    
    print("\nEach processor maps dataset fields to standard features:")
    print("  - violation_count: Number/frequency of violations")
    print("  - severity_max: Maximum severity level")
    print("  - recency_score: How recent the event is")
    print("  - frequency_score: Frequency of similar events")
    print("  - context_risk: Contextual risk factors")
    
    print_section("Step 3: Training Workflow")
    
    print("\nRun the complete pipeline:")
    print("  $ python scripts/baseline_orchestrator.py")
    print("\nOr step by step:")
    print("  $ python scripts/baseline_orchestrator.py --download")
    print("  $ python scripts/baseline_orchestrator.py --process-only")
    print("  $ python scripts/baseline_orchestrator.py --train-only")
    
    print_section("Step 4: Output Files")
    
    print("\nThe pipeline generates:")
    print("  - data/processed/*.json: Individual processed datasets")
    print("  - processed_train_data.json: Merged training data")
    print("  - models/candidates/baseline_model.json: Trained model")
    print("  - models/candidates/baseline_metrics.json: Validation metrics")
    
    print_section("Step 5: Using the Trained Model")
    
    print("\nLoad and use the trained model:")
    print("""
from nethical.mlops.baseline import BaselineMLClassifier

# Load trained model
clf = BaselineMLClassifier.load('models/candidates/baseline_model.json')

# Make predictions
features = {
    'violation_count': 0.7,
    'severity_max': 0.8,
    'recency_score': 0.6,
    'frequency_score': 0.5,
    'context_risk': 0.4
}

prediction = clf.predict(features)
print(f"Label: {prediction['label']}")
print(f"Score: {prediction['score']:.3f}")
print(f"Confidence: {prediction['confidence']:.3f}")
    """)
    
    print_section("Example Workflow")
    
    print("\nComplete example with sample data:")
    print("""
# 1. Create sample dataset (or download from Kaggle)
echo "attack_type,severity,label" > data/external/sample.csv
echo "DDoS,High,malicious" >> data/external/sample.csv
echo "Normal,Low,benign" >> data/external/sample.csv

# 2. Process datasets
python scripts/baseline_orchestrator.py --process-only

# 3. Train model
python scripts/baseline_orchestrator.py --train-only

# 4. Check results
cat models/candidates/baseline_metrics.json
    """)
    
    print_section("Next Steps")
    
    print("\nAfter training:")
    print("  1. Review validation metrics in baseline_metrics.json")
    print("  2. Test model with scripts/test_model.py")
    print("  3. Deploy to Phase 5 shadow mode (passive inference)")
    print("  4. Promote to Phase 6 blended enforcement (active decisions)")
    print("  5. Monitor performance with Phase 7 anomaly detection")
    
    print("\nSee also:")
    print("  - README.md: Training pipeline documentation")
    print("  - scripts/README.md: Detailed script documentation")
    print("  - TrainTestPipeline.md: Training/testing specifications")
    print("  - PHASE5-7_GUIDE.md: ML integration guide")
    
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
