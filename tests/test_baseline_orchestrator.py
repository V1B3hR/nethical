"""Test baseline_orchestrator.py script."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_baseline_orchestrator_help():
    """Test that baseline_orchestrator shows help."""
    result = subprocess.run(
        [sys.executable, "scripts/baseline_orchestrator.py", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Script failed with: {result.stderr}"
    assert "Baseline ML Training Orchestrator" in result.stdout
    assert "--download" in result.stdout
    assert "--process-only" in result.stdout
    assert "--train-only" in result.stdout
    print("✓ Help message test passed")


def test_baseline_orchestrator_process_only():
    """Test baseline_orchestrator in process-only mode with synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test_data.json"
        
        # Run process-only mode
        result = subprocess.run(
            [
                sys.executable, 
                "scripts/baseline_orchestrator.py",
                "--process-only",
                "--data-file", str(data_file)
            ],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        assert data_file.exists(), "Data file was not created"
        
        # Load and validate data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        assert len(data) > 0, "No data generated"
        assert 'features' in data[0], "Missing features in data"
        assert 'label' in data[0], "Missing label in data"
        
        print(f"✓ Process-only test passed ({len(data)} records generated)")


def test_baseline_orchestrator_train_only():
    """Test baseline_orchestrator in train-only mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "train_data.json"
        
        # Create synthetic training data
        synthetic_data = []
        for i in range(100):
            features = {
                'violation_count': (i % 10) / 10.0,
                'severity_max': (i % 8) / 8.0,
                'recency_score': (i % 6) / 6.0,
                'frequency_score': (i % 5) / 5.0,
                'context_risk': (i % 4) / 4.0,
            }
            label = 1 if (features['violation_count'] + features['severity_max']) > 1.0 else 0
            synthetic_data.append({'features': features, 'label': label})
        
        with open(data_file, 'w') as f:
            json.dump(synthetic_data, f)
        
        # Change to temp directory for model output
        import os
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        # Create necessary directories
        Path("models/candidates").mkdir(parents=True, exist_ok=True)
        
        try:
            # Run train-only mode
            result = subprocess.run(
                [
                    sys.executable, 
                    str(Path(original_cwd) / "scripts/baseline_orchestrator.py"),
                    "--train-only",
                    "--data-file", str(data_file)
                ],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0, f"Script failed with: {result.stderr}"
            
            # Check that model and metrics were created
            model_file = Path("models/candidates/baseline_model.json")
            metrics_file = Path("models/candidates/baseline_metrics.json")
            
            assert model_file.exists(), "Model file was not created"
            assert metrics_file.exists(), "Metrics file was not created"
            
            # Validate model file
            with open(model_file, 'r') as f:
                model_data = json.load(f)
            
            assert model_data['model_type'] == 'baseline'
            assert model_data['trained'] == True
            assert 'feature_weights' in model_data
            
            # Validate metrics file
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            
            print("✓ Train-only test passed")
            
        finally:
            os.chdir(original_cwd)


def test_baseline_orchestrator_with_csv():
    """Test baseline_orchestrator with a real CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create external data directory
        external_dir = Path(tmpdir) / "data" / "external"
        external_dir.mkdir(parents=True)
        
        # Create a test CSV file
        csv_file = external_dir / "test_attacks.csv"
        with open(csv_file, 'w') as f:
            f.write("attack_type,severity,anomaly_score,packet_count,label\n")
            f.write("DDoS,High,85,1500,malicious\n")
            f.write("Normal,Low,10,50,normal\n")
            f.write("Port Scan,Medium,60,200,attack\n")
            f.write("SQL Injection,Critical,95,300,malicious\n")
            f.write("Normal Traffic,Low,5,40,benign\n")
        
        data_file = Path(tmpdir) / "csv_data.json"
        
        # Change to temp directory
        import os
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        
        # Create necessary directories
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        try:
            # Run process-only mode
            result = subprocess.run(
                [
                    sys.executable, 
                    str(Path(original_cwd) / "scripts/baseline_orchestrator.py"),
                    "--process-only",
                    "--data-file", str(data_file)
                ],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0, f"Script failed with: {result.stderr}"
            assert data_file.exists(), "Data file was not created"
            
            # Load and validate data
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 5, f"Expected 5 records, got {len(data)}"
            
            # Check that processing preserved labels
            labels = [d['label'] for d in data]
            assert 1 in labels, "No positive labels found"
            assert 0 in labels, "No negative labels found"
            
            print("✓ CSV processing test passed")
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    print("\n=== Testing Baseline Orchestrator ===\n")
    
    test_baseline_orchestrator_help()
    test_baseline_orchestrator_process_only()
    test_baseline_orchestrator_train_only()
    test_baseline_orchestrator_with_csv()
    
    print("\n=== All Tests Passed ===\n")
