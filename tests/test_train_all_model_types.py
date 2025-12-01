#!/usr/bin/env python3
"""Test training all model types in train_any_model.py."""
import json
import tempfile
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_all_model_types():
    """Test train_any_model.py with --model-type all."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with --model-type all")
    print("=" * 70)
    
    # Get the repo root directory
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create models directory
        models_dir = tmpdir_path / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "current").mkdir()
        (models_dir / "candidates").mkdir()
        
        # Create data directory
        data_dir = tmpdir_path / "data" / "external"
        data_dir.mkdir(parents=True)
        
        print("\n[1/5] Running training with --model-type all...")
        
        # Run training with --model-type all
        result = subprocess.run([
            sys.executable,
            str(repo_root / "training" / "train_any_model.py"),
            "--model-type", "all",
            "--epochs", "2",
            "--batch-size", "32",
            "--num-samples", "100",
            "--seed", "42"
        ], 
        cwd=tmpdir_path,
        capture_output=True,
        text=True,
        timeout=300  # 5 minutes for all models
        )
        
        combined_output = result.stdout + result.stderr
        
        print(f"  Exit code: {result.returncode}")
        
        # Check for successful training
        assert result.returncode == 0, f"Training failed with return code {result.returncode}\n{result.stderr}"
        print("  ✓ Training completed successfully")
        
        print("\n[2/5] Verifying all model types were trained...")
        
        # Expected model types
        expected_models = ["heuristic", "logistic", "simple_transformer", "anomaly", "correlation"]
        
        for model_type in expected_models:
            assert f"Training model" in combined_output and model_type in combined_output, \
                f"Model type {model_type} was not trained"
            print(f"  ✓ {model_type} was trained")
        
        print("\n[3/5] Checking training summary output...")
        
        # Check for training summary
        assert "TRAINING SUMMARY" in combined_output, "Training summary not printed"
        assert "Total models trained: 5" in combined_output, "Wrong number of models in summary"
        print("  ✓ Training summary printed correctly")
        
        print("\n[4/5] Checking model output files...")
        
        # Check that model files were created in candidates directory
        candidates_dir = models_dir / "candidates"
        model_files = list(candidates_dir.glob("*_model_*.json"))
        metrics_files = list(candidates_dir.glob("*_metrics_*.json"))
        
        assert len(model_files) >= 5, f"Expected at least 5 model files, found {len(model_files)}"
        assert len(metrics_files) >= 5, f"Expected at least 5 metrics files, found {len(metrics_files)}"
        
        print(f"  ✓ Found {len(model_files)} model files")
        print(f"  ✓ Found {len(metrics_files)} metrics files")
        
        print("\n[5/5] Checking individual model metrics...")
        
        for metrics_file in metrics_files:
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            assert "accuracy" in metrics, f"Accuracy missing from {metrics_file.name}"
            print(f"  ✓ {metrics_file.name}: accuracy={metrics['accuracy']:.4f}")
        
        print("\n" + "=" * 70)
        print("  ✅ ALL MODEL TYPE TESTS PASSED")
        print("=" * 70)


def test_train_all_with_audit_and_governance():
    """Test train_any_model.py --model-type all with audit and governance."""
    print("\n" + "=" * 70)
    print("  Test: --model-type all with Audit and Governance")
    print("=" * 70)
    
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create models directory
        models_dir = tmpdir_path / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "current").mkdir()
        (models_dir / "candidates").mkdir()
        
        # Create audit directory
        audit_dir = tmpdir_path / "test_audit"
        audit_dir.mkdir()
        
        # Create drift reports directory  
        drift_dir = tmpdir_path / "test_drift"
        drift_dir.mkdir()
        
        print("\n[1/4] Running training with all features enabled...")
        
        result = subprocess.run([
            sys.executable,
            str(repo_root / "training" / "train_any_model.py"),
            "--model-type", "all",
            "--epochs", "2",
            "--num-samples", "50",
            "--enable-audit",
            "--audit-path", str(audit_dir),
            "--enable-governance",
            "--enable-drift-tracking",
            "--drift-report-dir", str(drift_dir)
        ],
        cwd=tmpdir_path,
        capture_output=True,
        text=True,
        timeout=300
        )
        
        combined_output = result.stdout + result.stderr
        
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        print("  ✓ Training completed with all features enabled")
        
        print("\n[2/4] Checking audit summary...")
        
        audit_summary = audit_dir / "training_summary.json"
        assert audit_summary.exists(), "Audit summary not created"
        
        with open(audit_summary) as f:
            summary = json.load(f)
        
        # Check audit summary structure for batch training
        assert "merkle_root" in summary, "Merkle root missing"
        assert "model_types" in summary, "Model types missing from summary"
        assert "results" in summary, "Results missing from summary"
        assert "total_models" in summary, "Total models count missing"
        assert summary["total_models"] == 5, "Wrong total models count"
        
        print(f"  ✓ Audit summary created with {len(summary['results'])} model results")
        print(f"  ✓ Merkle root: {summary['merkle_root'][:16]}...")
        
        print("\n[3/4] Checking governance summary...")
        
        assert "governance" in summary, "Governance section missing"
        assert summary["governance"]["enabled"] is True, "Governance not marked as enabled"
        
        print(f"  ✓ Governance enabled: {summary['governance']['enabled']}")
        print(f"  ✓ Total data violations: {summary['governance'].get('data_violations', 0)}")
        print(f"  ✓ Total prediction violations: {summary['governance'].get('prediction_violations', 0)}")
        
        print("\n[4/4] Checking drift reports...")
        
        drift_files = list(drift_dir.glob("drift_*.json"))
        assert len(drift_files) > 0, "No drift report files created"
        
        with open(drift_files[0]) as f:
            drift_report = json.load(f)
        
        assert "drift_metrics" in drift_report, "Drift metrics missing"
        print(f"  ✓ Drift report created: {drift_files[0].name}")
        
        print("\n" + "=" * 70)
        print("  ✅ ALL FEATURES TEST PASSED")
        print("=" * 70)


def test_train_all_model_types_command_from_problem_statement():
    """Test the exact command from the problem statement."""
    print("\n" + "=" * 70)
    print("  Test: Exact Command from Problem Statement")
    print("=" * 70)
    
    repo_root = Path(__file__).parent.parent
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create necessary directories
        models_dir = tmpdir_path / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "current").mkdir()
        (models_dir / "candidates").mkdir()
        
        audit_dir = tmpdir_path / "training_audit_logs"
        audit_dir.mkdir()
        
        drift_dir = tmpdir_path / "training_drift_reports"
        drift_dir.mkdir()
        
        print("\n[1/3] Running exact command from problem statement (with reduced samples)...")
        
        # Run the exact command from problem statement (with reduced samples for testing)
        result = subprocess.run([
            sys.executable,
            str(repo_root / "training" / "train_any_model.py"),
            "--model-type", "all",
            "--epochs", "2",  # Reduced for testing
            "--batch-size", "64",
            "--num-samples", "100",  # Reduced for testing
            "--enable-audit",
            "--audit-path", str(audit_dir),
            "--promotion-min-accuracy", "0.85",
            "--promotion-max-ece", "0.08",
            "--enable-governance",
            "--enable-drift-tracking",
            "--drift-report-dir", str(drift_dir)
        ],
        cwd=tmpdir_path,
        capture_output=True,
        text=True,
        timeout=300
        )
        
        combined_output = result.stdout + result.stderr
        
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        print("  ✓ Command executed successfully")
        
        print("\n[2/3] Verifying promotion gate thresholds were applied...")
        
        # Check for promotion gate messages
        assert "Promotion Gate: ECE <= 0.080, Accuracy >= 0.850" in combined_output, \
            "Promotion gate thresholds not applied correctly"
        print("  ✓ Promotion gate thresholds correctly applied")
        
        print("\n[3/3] Verifying all 5 model types were trained...")
        
        assert "Total models trained: 5" in combined_output, "Not all models were trained"
        print("  ✓ All 5 model types trained")
        
        # Count successful models
        successful_count = combined_output.count("SUCCESS")
        assert successful_count >= 5, f"Expected at least 5 successful trainings, got {successful_count}"
        print(f"  ✓ {successful_count} successful model trainings")
        
        print("\n" + "=" * 70)
        print("  ✅ PROBLEM STATEMENT COMMAND TEST PASSED")
        print("=" * 70)


if __name__ == "__main__":
    try:
        test_train_all_model_types()
        test_train_all_with_audit_and_governance()
        test_train_all_model_types_command_from_problem_statement()
        print("\n" + "=" * 70)
        print("  ALL TESTS PASSED ✅")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
