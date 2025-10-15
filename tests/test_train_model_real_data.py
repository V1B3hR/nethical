#!/usr/bin/env python3
"""Test the updated train_model.py with real data loading."""
import json
import shutil
import tempfile
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.skip(reason="scripts.train_model module was deprecated - functionality moved to training/train_any_model.py")
def test_train_model_with_real_data():
    """Test train_model.py with sample real datasets."""
    print("\n" + "=" * 70)
    print("  Test: train_model.py with Real Data Loading")
    print("=" * 70)
    
    # Save current working directory
    import os
    original_dir = os.getcwd()
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create sample datasets in data/external/
            data_external = tmpdir_path / "data" / "external"
            data_external.mkdir(parents=True)
            
            # Create data ethics dataset
            ethics_csv = data_external / "data_ethics.csv"
            with open(ethics_csv, 'w') as f:
                f.write("ethical_issue,severity,risk_level,impact,label\n")
                f.write("privacy_breach,high,critical,major,malicious\n")
                f.write("consent_issue,medium,moderate,medium,suspicious\n")
                f.write("fair_algorithm,low,low,minor,benign\n")
                f.write("bias_detection,high,critical,major,malicious\n")
                f.write("transparency,medium,moderate,medium,suspicious\n")
                # Add more records to get better training
                for i in range(20):
                    if i % 2 == 0:
                        f.write(f"privacy_issue_{i},high,critical,major,malicious\n")
                    else:
                        f.write(f"minor_issue_{i},low,low,minor,benign\n")
            
            # Create security breach dataset
            breach_csv = data_external / "security_breach.csv"
            with open(breach_csv, 'w') as f:
                f.write("breach_type,severity,records_exposed,impact_level,label\n")
                f.write("ransomware,critical,1000000,high,malicious\n")
                f.write("phishing,high,50000,medium,malicious\n")
                f.write("minor_incident,low,100,low,benign\n")
                f.write("malware,high,100000,high,malicious\n")
                f.write("test_event,low,0,low,benign\n")
                # Add more records
                for i in range(20):
                    if i % 2 == 0:
                        f.write(f"attack_{i},high,100000,high,malicious\n")
                    else:
                        f.write(f"test_{i},low,0,low,benign\n")
            
            print(f"\n[1/4] Created sample datasets:")
            print(f"  ✓ {ethics_csv.name}: 25 records")
            print(f"  ✓ {breach_csv.name}: 25 records")
            
            # Change to temporary directory
            os.chdir(tmpdir_path)
            
            # Import and test load_real_world_data function
            print("\n[2/4] Testing load_real_world_data()...")
            from scripts.train_model import load_real_world_data
            
            data = load_real_world_data()
            
            print(f"  ✓ Loaded {len(data)} total records")
            
            # Verify data structure
            assert len(data) > 0, "No data loaded"
            assert all('features' in record for record in data), "Missing features"
            assert all('label' in record for record in data), "Missing labels"
            
            print("\n[3/4] Validating data structure...")
            sample = data[0]
            print(f"  Sample record keys: {list(sample.keys())}")
            print(f"  Sample features: {list(sample['features'].keys())}")
            print(f"  Sample label: {sample['label']}")
            
            required_features = ['violation_count', 'severity_max', 'recency_score', 
                               'frequency_score', 'context_risk']
            for feature in required_features:
                assert feature in sample['features'], f"Missing feature: {feature}"
            
            print("  ✓ All required features present")
            
            # Test with minimal training pipeline
            print("\n[4/4] Testing minimal training...")
            from scripts.train_model import temporal_split, train_baseline_model
            
            train_data, val_data = temporal_split(data, train_ratio=0.8)
            print(f"  ✓ Split data: {len(train_data)} train, {len(val_data)} val")
            
            classifier = train_baseline_model(train_data)
            print("  ✓ Model trained successfully")
            
            # Test prediction
            if val_data:
                sample_pred = classifier.predict(val_data[0]['features'])
                print(f"  ✓ Sample prediction: {sample_pred}")
            
            print("\n" + "=" * 70)
            print("  ✅ TEST PASSED")
            print("=" * 70)
            
    finally:
        # Restore original directory
        os.chdir(original_dir)


@pytest.mark.skip(reason="scripts.train_model module was deprecated - functionality moved to training/train_any_model.py")
def test_fallback_to_synthetic():
    """Test that train_model.py falls back to synthetic data when no real data available."""
    print("\n" + "=" * 70)
    print("  Test: Fallback to Synthetic Data")
    print("=" * 70)
    
    import os
    original_dir = os.getcwd()
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create empty data/external directory (no CSV files)
            data_external = tmpdir_path / "data" / "external"
            data_external.mkdir(parents=True)
            
            os.chdir(tmpdir_path)
            
            print("\n[1/2] Testing with no CSV files (should use synthetic)...")
            from scripts.train_model import load_real_world_data
            
            data = load_real_world_data()
            
            print(f"  ✓ Generated {len(data)} synthetic records")
            assert len(data) == 1000, f"Expected 1000 synthetic records, got {len(data)}"
            
            print("\n[2/2] Validating synthetic data structure...")
            sample = data[0]
            assert 'event_id' in sample, "Missing event_id in synthetic data"
            assert 'features' in sample, "Missing features in synthetic data"
            assert 'label' in sample, "Missing label in synthetic data"
            
            print("  ✓ Synthetic data structure valid")
            
            print("\n" + "=" * 70)
            print("  ✅ FALLBACK TEST PASSED")
            print("=" * 70)
            
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    test_train_model_with_real_data()
    test_fallback_to_synthetic()
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED ✅")
    print("=" * 70)
