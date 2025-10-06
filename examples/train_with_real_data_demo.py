#!/usr/bin/env python3
"""
Demo: Training with Real Datasets

This script demonstrates how to use train_any_model.py with real datasets
from the datasets/datasets file.

Workflow:
1. Process raw datasets into standardized format
2. Train models using the processed data
3. Evaluate model performance

Requirements:
- Processed datasets in data/processed/ directory
- See scripts/dataset_processors/ for processing raw data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    """Run the training demonstration."""
    print_section("NETHICAL: Training with Real Datasets Demo")
    
    print("\n📋 Overview:")
    print("  This demo shows how to train ML models using real-world security datasets")
    print("  listed in datasets/datasets.")
    
    print_section("Step 1: Dataset Sources")
    
    print("\n📁 Datasets file location: datasets/datasets")
    print("   Contains URLs to real-world security datasets from Kaggle:")
    
    datasets_file = Path("datasets/datasets")
    if datasets_file.exists():
        with open(datasets_file, 'r') as f:
            for i, line in enumerate(f, 1):
                url = line.strip()
                if url and url.startswith('http'):
                    print(f"   {i}. {url.split('/')[-1]}")
    else:
        print("   ⚠️  datasets/datasets file not found")
    
    print_section("Step 2: Dataset Processing")
    
    print("\n🔧 Processing raw datasets:")
    print("   Dataset processors are available in scripts/dataset_processors/:")
    print("   - CyberSecurityAttacksProcessor: Network attack datasets")
    print("   - MicrosoftSecurityProcessor: Microsoft incident datasets")
    print("   - GenericSecurityProcessor: General security datasets")
    
    print("\n   Standard feature mapping:")
    print("   - violation_count: Number/frequency of violations")
    print("   - severity_max: Maximum severity level")
    print("   - recency_score: How recent the event is")
    print("   - frequency_score: Frequency of similar events")
    print("   - context_risk: Contextual risk factors")
    
    print("\n   Processed files are saved to: data/processed/*.json")
    
    print_section("Step 3: Training Models")
    
    print("\n🎯 Training with synthetic data (default):")
    print("   $ python training/train_any_model.py --model-type logistic --num-samples 5000")
    
    print("\n🎯 Training with real data:")
    print("   $ python training/train_any_model.py --model-type logistic --use-real-data")
    
    print("\n   The --use-real-data flag:")
    print("   ✓ Automatically loads all *_processed.json files from data/processed/")
    print("   ✓ Merges datasets from multiple sources")
    print("   ✓ Falls back to synthetic data if no processed files found")
    
    print_section("Step 4: Example Usage")
    
    print("\n💡 Complete workflow example:")
    print("""
# 1. Download datasets (manual or via Kaggle API)
#    See datasets/datasets for URLs

# 2. Process raw datasets
python -c "
from scripts.dataset_processors.cyber_security_processor import CyberSecurityAttacksProcessor
processor = CyberSecurityAttacksProcessor()
records = processor.process(Path('data/external/cyber_attacks.csv'))
processor.save_processed_data(records)
"

# 3. Train model with real data
python training/train_any_model.py --model-type logistic --use-real-data --epochs 10

# 4. Evaluate results
#    Model and metrics saved to models/candidates/ or models/current/
""")
    
    print_section("Step 5: Model Types")
    
    print("\n🤖 Supported model types:")
    print("   - logistic: Baseline logistic classifier")
    print("   - heuristic: Rule-based heuristic classifier")
    print("   - anomaly: Sequence-based anomaly detection")
    print("   - simple_transformer: Transformer-based classifier (if implemented)")
    
    print_section("Summary")
    
    print("\n✅ Key capabilities:")
    print("   • Load real security datasets from datasets/datasets")
    print("   • Process with specialized dataset processors")
    print("   • Train multiple model types")
    print("   • Automatic fallback to synthetic data")
    print("   • Promotion gates for model quality")
    
    print("\n📚 For more information:")
    print("   - README.md: General documentation")
    print("   - ANOMALY_DETECTION_TRAINING.md: Anomaly detection details")
    print("   - IMPLEMENTATION_SUMMARY.md: Real data training pipeline")
    
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
