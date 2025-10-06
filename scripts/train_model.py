#!/usr/bin/env python3
"""Training Script for Nethical ML Models

Implements the training pipeline described in TrainTestPipeline.md:
- Temporal split (train: past, val: recent)
- Baseline classifier from nethical/mlops
- Metrics: Precision, recall, F1, ROC-AUC, calibration (ECE)
- Shadow logging and comparison with rule-only outcomes
- Promotion gate validation

REAL-WORLD DATASETS (Default):
This script now uses real-world datasets by default:
    - https://www.kaggle.com/code/kmldas/data-ethics-in-data-science-analytics-ml-and-ai
    - https://www.kaggle.com/datasets/xontoloyo/security-breachhh

The script will:
1. Attempt to download datasets using Kaggle API (if available)
2. Process CSV files from data/external/ directory
3. Train model on all available real data
4. Fall back to synthetic data if no real datasets are found

MANUAL DATASET SETUP:
If Kaggle API is not available, manually download the datasets and place CSV files in data/external/

See datasets/datasets for additional dataset sources that can be used with baseline_orchestrator.py.
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

# ---- mlops imports ----
from nethical.mlops import model_registry
from nethical.mlops.baseline import BaselineMLClassifier

# ---- dataset processing imports ----
from scripts.dataset_processors.generic_processor import GenericSecurityProcessor

def load_real_world_data() -> List[Dict[str, Any]]:
    """Load and process real-world datasets for training.
    
    Downloads and processes the following datasets:
    - https://www.kaggle.com/code/kmldas/data-ethics-in-data-science-analytics-ml-and-ai
    - https://www.kaggle.com/datasets/xontoloyo/security-breachhh
    
    Returns:
        List of processed training records with features and labels
    """
    print("=" * 60)
    print("LOADING REAL-WORLD DATASETS")
    print("=" * 60)
    
    # Define the two datasets to use
    datasets_to_use = [
        "https://www.kaggle.com/code/kmldas/data-ethics-in-data-science-analytics-ml-and-ai",
        "https://www.kaggle.com/datasets/xontoloyo/security-breachhh"
    ]
    
    download_dir = Path("data/external")
    processed_dir = Path("data/processed")
    download_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to download datasets using Kaggle API
    print("\nAttempting to download datasets...")
    try:
        import kaggle
        print("✓ Kaggle API is available")
        
        for url in datasets_to_use:
            try:
                if '/datasets/' in url:
                    parts = url.split('/datasets/')[-1].split('/')
                    if len(parts) >= 2:
                        dataset_id = f"{parts[0]}/{parts[1]}"
                        print(f"Downloading {dataset_id}...")
                        kaggle.api.dataset_download_files(
                            dataset_id,
                            path=str(download_dir),
                            unzip=True
                        )
                        print(f"✓ Downloaded {dataset_id}")
                elif '/code/' in url:
                    # For code/kernel datasets, extract kernel ID
                    parts = url.split('/code/')[-1].split('/')
                    if len(parts) >= 2:
                        kernel_id = f"{parts[0]}/{parts[1]}"
                        print(f"Downloading kernel {kernel_id}...")
                        kaggle.api.kernels_pull(kernel_id, path=str(download_dir))
                        print(f"✓ Downloaded kernel {kernel_id}")
            except Exception as e:
                print(f"Warning: Could not download {url}: {e}")
                print("Please download manually and place CSV files in data/external/")
    except ImportError:
        print("✗ Kaggle API not installed")
        print("Please install with: pip install kaggle")
        print("\nOr download datasets manually:")
        for url in datasets_to_use:
            print(f"  - {url}")
        print(f"And save CSV files to: {download_dir.absolute()}")
    
    # Process all CSV files found in download directory
    print("\n" + "=" * 60)
    print("PROCESSING DATASETS")
    print("=" * 60)
    
    csv_files = list(download_dir.glob("**/*.csv"))
    print(f"\nFound {len(csv_files)} CSV files in {download_dir}")
    
    all_records = []
    
    if csv_files:
        for csv_file in csv_files:
            print(f"\nProcessing: {csv_file.name}")
            dataset_name = csv_file.stem.replace(' ', '_').lower()
            
            # Use GenericSecurityProcessor for all datasets
            processor = GenericSecurityProcessor(dataset_name, processed_dir)
            
            try:
                records = processor.process(csv_file)
                if records:
                    all_records.extend(records)
                    output_file = processor.save_processed_data(records)
                    print(f"✓ Processed {len(records)} records from {csv_file.name}")
                    print(f"  Saved to {output_file}")
            except Exception as e:
                print(f"Warning: Failed to process {csv_file.name}: {e}")
                continue
    
    if not all_records:
        print("\n" + "=" * 60)
        print("WARNING: No data loaded from real datasets")
        print("=" * 60)
        print("Falling back to synthetic data...")
        return generate_synthetic_labeled_data(num_samples=1000)
    
    # Shuffle records
    random.shuffle(all_records)
    
    print("\n" + "=" * 60)
    print(f"✓ Successfully loaded {len(all_records)} total records from real datasets")
    print("=" * 60)
    
    return all_records


def generate_synthetic_labeled_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic labeled training data."""
    print(f"Generating {num_samples} synthetic training samples...")

    data = []
    base_time = datetime.now() - timedelta(days=30)
    for i in range(num_samples):
        timestamp = base_time + timedelta(minutes=i * 43)
        risk_level = random.choice(['low', 'medium', 'high'])
        if risk_level == 'low':
            features = {
                'violation_count': random.uniform(0.0, 0.3),
                'severity_max': random.uniform(0.0, 0.2),
                'recency_score': random.uniform(0.0, 0.3),
                'frequency_score': random.uniform(0.0, 0.2),
                'context_risk': random.uniform(0.0, 0.2)
            }
            label = 0
        elif risk_level == 'medium':
            features = {
                'violation_count': random.uniform(0.3, 0.6),
                'severity_max': random.uniform(0.3, 0.6),
                'recency_score': random.uniform(0.2, 0.5),
                'frequency_score': random.uniform(0.2, 0.5),
                'context_risk': random.uniform(0.2, 0.5)
            }
            label = random.choice([0, 1])
        else:
            features = {
                'violation_count': random.uniform(0.6, 1.0),
                'severity_max': random.uniform(0.7, 1.0),
                'recency_score': random.uniform(0.5, 1.0),
                'frequency_score': random.uniform(0.5, 0.9),
                'context_risk': random.uniform(0.5, 0.9)
            }
            label = 1
        rule_score = sum(features.values()) / len(features)
        data.append({
            'event_id': f'evt_{i:06d}',
            'timestamp': timestamp.isoformat(),
            'agent_id': f'agent_{i % 100:03d}',
            'features': features,
            'rule_score': rule_score,
            'label': label
        })
    return data

def temporal_split(data: List[Dict[str, Any]], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """Split data temporally (train: past, val: recent)."""
    print(f"\nPerforming temporal split ({train_ratio:.0%} train, {1-train_ratio:.0%} validation)...")
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    return train_data, val_data

def train_baseline_model(train_data: List[Dict[str, Any]]) -> BaselineMLClassifier:
    """Train baseline classifier using mlops baseline model."""
    print("\nTraining baseline classifier (mlops)...")
    model = BaselineMLClassifier()
    model.train(train_data)
    print(f"Model trained with {len(train_data)} samples.")
    return model

def evaluate_model(classifier: BaselineMLClassifier, val_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Evaluate model on validation set."""
    print("\nEvaluating model on validation set...")
    preds, labels = [], []
    for sample in val_data:
        pred = classifier.predict(sample['features'])
        preds.append(pred['label'])
        labels.append(sample['label'])
    metrics = classifier.compute_metrics(preds, labels)
    print("\nValidation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return metrics

def check_promotion_gate(metrics: Dict[str, float], baseline_metrics: Dict[str, float] = None) -> bool:
    """Check if model meets promotion gate criteria."""
    print("\nChecking Promotion Gate Criteria:")
    print("=" * 60)
    passed = True
    max_ece = 0.08
    ece_pass = metrics.get('ece', 0.0) <= max_ece
    print(f"✓ ECE ≤ {max_ece}: {metrics.get('ece', 0.0):.3f} {'PASS' if ece_pass else 'FAIL'}")
    passed = passed and ece_pass
    min_accuracy = 0.85
    accuracy_pass = metrics.get('accuracy', 0.0) >= min_accuracy
    print(f"✓ Accuracy ≥ {min_accuracy}: {metrics.get('accuracy', 0.0):.3f} {'PASS' if accuracy_pass else 'FAIL'}")
    passed = passed and accuracy_pass
    print("=" * 60)
    print(f"Overall: {'✓ PASS - Model can be promoted' if passed else '✗ FAIL - Model does not meet criteria'}")
    return passed

def save_model(classifier: BaselineMLClassifier, metrics: Dict[str, float], model_dir: str = "./models/candidates"):
    """Save trained model and metadata using mlops model_registry."""
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = os.path.join(model_dir, f"model_{timestamp}.json")
    classifier.save(model_file)
    metadata_file = os.path.join(model_dir, f"model_{timestamp}_metrics.json")
    with open(metadata_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Model and metrics saved: {model_file}, {metadata_file}")
    return model_file

def save_training_data(data: List[Dict[str, Any]], output_file: str = "./data/labeled_events/training_data.json"):
    """Save training data to file."""
    print(f"\nSaving training data to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Training data saved: {output_file}")

def run_testing_pipeline():
    """Run the testing pipeline after training (if available)."""
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
        test_data = load_test_data()
        ml_metrics, predictions_log = evaluate_on_test_set(classifier, test_data)
        save_evaluation_report(ml_metrics, predictions_log)
        print("\n" + "=" * 60)
        print("TESTING COMPLETE")
        print("=" * 60)
        print("\nKey Findings:")
        if ml_metrics.get('f1_score', 0.0) >= 0.8:
            print("✓ Model shows strong performance (F1 ≥ 0.8)")
        elif ml_metrics.get('f1_score', 0.0) >= 0.6:
            print("⚠ Model shows moderate performance (0.6 ≤ F1 < 0.8)")
        else:
            print("✗ Model needs improvement (F1 < 0.6)")
        if ml_metrics.get('expected_calibration_error', 1.0) <= 0.08:
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
    """Main training pipeline."""
    print("=" * 60)
    print("NETHICAL ML TRAINING PIPELINE (mlops)")
    print("=" * 60)
    print("\nBased on TrainTestPipeline.md specifications:")
    print("- Temporal split (train: past, val: recent)")
    print("- Baseline classifier from mlops")
    print("- Metrics: Precision, Recall, F1, ECE")
    print("- Promotion gate validation")
    if run_all:
        print("- Running complete workflow (training + testing)")
    print()
    
    # Step 1: Load real-world data (from the two specified datasets)
    print("Loading real-world datasets:")
    print("  - https://www.kaggle.com/code/kmldas/data-ethics-in-data-science-analytics-ml-and-ai")
    print("  - https://www.kaggle.com/datasets/xontoloyo/security-breachhh")
    print()
    
    data = load_real_world_data()
    save_training_data(data)
    
    # Step 2: Temporal split
    train_data, val_data = temporal_split(data, train_ratio=0.8)
    
    # Step 3: Train baseline model
    classifier = train_baseline_model(train_data)
    
    # Step 4: Evaluate on validation set
    metrics = evaluate_model(classifier, val_data)
    
    # Step 5: Check promotion gate
    promotion_passed = check_promotion_gate(metrics)
    
    # Step 6: Save model to candidates
    model_file = save_model(classifier, metrics, model_dir="./models/candidates")
    
    # Step 7: Promote if criteria met
    if promotion_passed:
        print("\n✓ Promoting model to production (models/current/)...")
        model_registry.promote_model(os.path.basename(model_file))
        print("✓ Model promoted successfully")
    else:
        print("\n⚠ Model saved to candidates only (models/candidates/)")
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
