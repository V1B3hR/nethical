#!/usr/bin/env python3
"""Test train_any_model.py with real data loading functionality."""
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_any_model import load_data, load_real_data_from_processed


def test_load_real_data_from_processed():
    """Test loading real data from processed JSON files."""
    print("\n" + "=" * 70)
    print("  Test: load_real_data_from_processed()")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        processed_dir = Path(tmpdir) / "processed"
        processed_dir.mkdir(parents=True)
        
        # Create sample processed data
        sample_data = [
            {
                'features': {
                    'violation_count': 0.8,
                    'severity_max': 0.9,
                    'recency_score': 0.7,
                    'frequency_score': 0.6,
                    'context_risk': 0.75
                },
                'label': 1
            },
            {
                'features': {
                    'violation_count': 0.1,
                    'severity_max': 0.2,
                    'recency_score': 0.3,
                    'frequency_score': 0.2,
                    'context_risk': 0.1
                },
                'label': 0
            }
        ]
        
        # Save to JSON file
        test_file = processed_dir / "test_dataset_processed.json"
        with open(test_file, 'w') as f:
            json.dump(sample_data, f)
        
        print(f"\n[1/3] Created test file: {test_file.name}")
        
        # Test loading
        print("\n[2/3] Loading data...")
        loaded_data = load_real_data_from_processed(data_dir=str(processed_dir))
        
        assert loaded_data is not None, "Failed to load data"
        assert len(loaded_data) == 2, f"Expected 2 records, got {len(loaded_data)}"
        print(f"  ✓ Loaded {len(loaded_data)} records")
        
        # Validate structure
        print("\n[3/3] Validating data structure...")
        for record in loaded_data:
            assert 'features' in record, "Missing 'features' key"
            assert 'label' in record, "Missing 'label' key"
            assert all(k in record['features'] for k in ['violation_count', 'severity_max', 
                                                          'recency_score', 'frequency_score', 
                                                          'context_risk']), "Missing required features"
        print("  ✓ All records have correct structure")
        
        print("\n" + "=" * 70)
        print("  ✅ TEST PASSED")
        print("=" * 70)


def test_load_data_with_use_real_data_flag():
    """Test load_data with use_real_data flag."""
    print("\n" + "=" * 70)
    print("  Test: load_data(use_real_data=True)")
    print("=" * 70)
    
    # Test synthetic fallback (no real data available)
    print("\n[1/2] Testing fallback to synthetic data...")
    data = load_data(num_samples=50, use_real_data=True)
    
    assert data is not None, "Failed to load data"
    assert len(data) == 50, f"Expected 50 samples, got {len(data)}"
    print(f"  ✓ Loaded {len(data)} synthetic samples (fallback)")
    
    # Test synthetic data without flag
    print("\n[2/2] Testing explicit synthetic data...")
    data = load_data(num_samples=30, use_real_data=False)
    
    assert data is not None, "Failed to load data"
    assert len(data) == 30, f"Expected 30 samples, got {len(data)}"
    print(f"  ✓ Loaded {len(data)} synthetic samples")
    
    print("\n" + "=" * 70)
    print("  ✅ TEST PASSED")
    print("=" * 70)


def test_multiple_processed_files():
    """Test loading from multiple processed JSON files."""
    print("\n" + "=" * 70)
    print("  Test: Multiple processed files")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        processed_dir = Path(tmpdir) / "processed"
        processed_dir.mkdir(parents=True)
        
        # Create multiple processed files
        for i in range(3):
            sample_data = [
                {
                    'features': {
                        'violation_count': 0.5 + i * 0.1,
                        'severity_max': 0.5,
                        'recency_score': 0.5,
                        'frequency_score': 0.5,
                        'context_risk': 0.5
                    },
                    'label': i % 2
                }
                for _ in range(5)
            ]
            
            test_file = processed_dir / f"dataset_{i}_processed.json"
            with open(test_file, 'w') as f:
                json.dump(sample_data, f)
        
        print(f"\n[1/2] Created 3 processed files with 5 records each")
        
        # Test loading all files
        print("\n[2/2] Loading data from all files...")
        loaded_data = load_real_data_from_processed(data_dir=str(processed_dir))
        
        assert loaded_data is not None, "Failed to load data"
        assert len(loaded_data) == 15, f"Expected 15 records, got {len(loaded_data)}"
        print(f"  ✓ Loaded {len(loaded_data)} records from 3 files")
        
        print("\n" + "=" * 70)
        print("  ✅ TEST PASSED")
        print("=" * 70)


if __name__ == "__main__":
    test_load_real_data_from_processed()
    test_load_data_with_use_real_data_flag()
    test_multiple_processed_files()
    
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED ✅")
    print("=" * 70)
