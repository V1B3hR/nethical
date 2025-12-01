#!/usr/bin/env python3
"""Test training all model types with train_any_model.py."""
import json
import tempfile
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_all_model_types():
    """Test train_any_model.py with --model-type all option."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with --model-type all")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        audit_path = tmpdir_path / "test_audit_logs"
        models_dir = tmpdir_path / "models"
        
        # Run training for all model types
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "all",
            "--epochs", "2",
            "--num-samples", "100",
            "--seed", "42",
            "--enable-audit",
            "--audit-path", str(audit_path),
            "--models-dir", str(models_dir)
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        combined_output = result.stdout + result.stderr
        
        # Print summary lines
        print("\nOutput Summary:")
        for line in combined_output.split('\n'):
            if any(x in line for x in ['Training all model types', 'Training model', 
                                        'TRAINING SUMMARY', 'Total models', 'Successful',
                                        'Promoted', 'Failed', 'heuristic:', 'logistic:',
                                        'simple_transformer:', 'anomaly:', 'correlation:']):
                print(line)
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        print("\n✓ Training completed successfully")
        
        # Check that all model types were mentioned
        expected_types = ['heuristic', 'logistic', 'simple_transformer', 'anomaly', 'correlation']
        for model_type in expected_types:
            assert model_type in combined_output, f"Model type {model_type} not found in output"
        print(f"✓ All {len(expected_types)} model types were trained")
        
        # Check for "Training all model types" message
        assert "Training all model types" in combined_output, "Missing 'Training all model types' message"
        print("✓ Training all model types message found")
        
        # Check training summary
        assert "TRAINING SUMMARY" in combined_output, "Missing training summary"
        assert "Total models trained: 5" in combined_output, "Wrong total models count in summary"
        print("✓ Training summary shows 5 models trained")
        
        # Check audit summary
        summary_file = audit_path / "training_summary.json"
        assert summary_file.exists(), "Training summary not created"
        
        with open(summary_file) as f:
            summary = json.load(f)
        
        print("\nAudit Summary:")
        print(f"  Total models: {summary.get('total_models')}")
        print(f"  Successful: {summary.get('successful')}")
        print(f"  Promoted: {summary.get('promoted')}")
        
        assert summary['total_models'] == 5, f"Expected 5 models, got {summary['total_models']}"
        assert summary['model_types'] == expected_types, f"Model types mismatch: {summary['model_types']}"
        assert summary['successful'] >= 5, f"Expected all 5 models to succeed, got {summary['successful']}"
        print("✓ Audit summary verified")
        
        # Check results for each model
        assert 'results' in summary, "Results not in summary"
        assert len(summary['results']) == 5, f"Expected 5 results, got {len(summary['results'])}"
        
        for result_item in summary['results']:
            model_type = result_item['model_type']
            assert model_type in expected_types, f"Unexpected model type: {model_type}"
            assert result_item['success'] is True, f"Model {model_type} failed"
            assert 'metrics' in result_item, f"No metrics for {model_type}"
            print(f"✓ {model_type}: success={result_item['success']}, accuracy={result_item['metrics'].get('accuracy', 0):.4f}")
        
        print("\n" + "=" * 70)
        print("  ✅ ALL TESTS PASSED")
        print("=" * 70)


def test_train_all_model_types_with_full_options():
    """Test train_any_model.py with --model-type all and all options from the problem statement."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with --model-type all (full options)")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        audit_path = tmpdir_path / "audit_logs"
        drift_path = tmpdir_path / "drift_reports"
        models_dir = tmpdir_path / "models"
        
        # Run training with all options from problem statement
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "all",
            "--epochs", "5",  # Reduced for faster testing
            "--batch-size", "64",
            "--num-samples", "100",  # Reduced for faster testing
            "--enable-audit",
            "--audit-path", str(audit_path),
            "--promotion-min-accuracy", "0.85",
            "--promotion-max-ece", "0.08",
            "--enable-governance",
            "--enable-drift-tracking",
            "--drift-report-dir", str(drift_path),
            "--models-dir", str(models_dir)
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        combined_output = result.stdout + result.stderr
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        print("✓ Training completed successfully")
        
        # Check that all integrations were enabled
        assert "Merkle audit logging enabled" in combined_output, "Audit logging not enabled"
        print("✓ Audit logging enabled")
        
        assert "Governance validation enabled" in combined_output, "Governance not enabled"
        print("✓ Governance enabled")
        
        assert "Ethical drift tracking enabled" in combined_output, "Drift tracking not enabled"
        print("✓ Drift tracking enabled")
        
        # Check all model types were trained
        assert "Total models trained: 5" in combined_output, "Not all models were trained"
        print("✓ All 5 model types trained")
        
        # Check drift report
        assert "Drift Report ID:" in combined_output, "Drift report not generated"
        drift_files = list(drift_path.glob("drift_*.json"))
        assert len(drift_files) > 0, "No drift report files created"
        print(f"✓ Drift report generated: {drift_files[0].name}")
        
        # Check audit summary
        summary_file = audit_path / "training_summary.json"
        assert summary_file.exists(), "Audit summary not created"
        
        with open(summary_file) as f:
            summary = json.load(f)
        
        assert 'governance' in summary, "Governance section missing from audit"
        assert summary['governance']['enabled'] is True, "Governance not marked as enabled"
        print("✓ Governance metrics saved to audit summary")
        
        print("\n" + "=" * 70)
        print("  ✅ ALL TESTS PASSED")
        print("=" * 70)


if __name__ == "__main__":
    try:
        test_train_all_model_types()
        test_train_all_model_types_with_full_options()
        print("\n" + "=" * 70)
        print("  All Tests Passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
