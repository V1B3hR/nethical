#!/usr/bin/env python3
"""Test IntegratedGovernance integration in train_any_model.py."""
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_with_integrated_governance():
    """Test train_any_model.py with IntegratedGovernance enabled."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with IntegratedGovernance")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        governance_path = tmpdir_path / "test_governance_data"
        
        # Run training with IntegratedGovernance
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "50",
            "--seed", "42",
            "--enable-integrated-governance",
            "--governance-storage-dir", str(governance_path),
            "--cohort-id", "test_cohort"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print("\nSTDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check that IntegratedGovernance was initialized
        assert "IntegratedGovernance enabled" in result.stdout, "IntegratedGovernance not initialized"
        assert "All governance phases (3, 4, 5-7, 8-9) are active" in result.stdout, "Governance phases not active"
        
        # Check that governance data directory was created
        assert governance_path.exists(), "Governance data directory not created"
        
        # Check for expected subdirectories
        expected_dirs = [
            "merkle_data",
            "shadow_logs",
            "blended_logs",
            "drift_reports",
            "fairness_samples"
        ]
        
        for dir_name in expected_dirs:
            dir_path = governance_path / dir_name
            assert dir_path.exists(), f"Expected directory not created: {dir_name}"
        
        # Check for shadow predictions
        shadow_predictions_file = governance_path / "shadow_logs" / "shadow_predictions.jsonl"
        assert shadow_predictions_file.exists(), "Shadow predictions not created"
        
        # Check shadow predictions content
        with open(shadow_predictions_file) as f:
            lines = f.readlines()
            assert len(lines) > 0, "No shadow predictions logged"
            
            # Parse first prediction
            first_prediction = json.loads(lines[0])
            print("\nFirst Shadow Prediction:")
            print(json.dumps(first_prediction, indent=2))
            
            # Validate prediction structure
            required_fields = [
                'prediction_id',
                'timestamp',
                'agent_id',
                'ml_risk_score',
                'ml_classification',
                'rule_risk_score',
                'features'
            ]
            for field in required_fields:
                assert field in first_prediction, f"Missing field in shadow prediction: {field}"
        
        # Check for blended decisions
        blended_decisions_file = governance_path / "blended_logs" / "blended_decisions.jsonl"
        assert blended_decisions_file.exists(), "Blended decisions not created"
        
        # Check blended decisions content
        with open(blended_decisions_file) as f:
            lines = f.readlines()
            assert len(lines) > 0, "No blended decisions logged"
            
            # Parse first decision
            first_decision = json.loads(lines[0])
            print("\nFirst Blended Decision:")
            print(json.dumps(first_decision, indent=2))
            
            # Validate decision structure
            required_fields = [
                'decision_id',
                'timestamp',
                'agent_id',
                'blended_risk_score',
                'risk_zone',
                'ml_influenced'
            ]
            for field in required_fields:
                assert field in first_decision, f"Missing field in blended decision: {field}"
        
        # Check for governance system status in output
        assert "Governance System Status:" in result.stdout, "System status not reported"
        assert "Phase 3 (Risk & Correlation):" in result.stdout, "Phase 3 status not reported"
        assert "Phase 4 (Audit & Taxonomy):" in result.stdout, "Phase 4 status not reported"
        assert "Phase 5-7 (ML & Anomaly Detection):" in result.stdout, "Phase 5-7 status not reported"
        assert "Phase 8-9 (Human Oversight & Optimization):" in result.stdout, "Phase 8-9 status not reported"
        
        # Verify component statuses
        assert "Shadow classifier enabled: True" in result.stdout, "Shadow classifier not enabled"
        assert "ML blending enabled: True" in result.stdout, "ML blending not enabled"
        assert "Anomaly monitoring enabled: True" in result.stdout, "Anomaly monitoring not enabled"
        
        print("\n✓ All IntegratedGovernance tests passed!")
        return True


def test_train_with_different_model_types():
    """Test IntegratedGovernance with different model types."""
    print("\n" + "=" * 70)
    print("  Test: IntegratedGovernance with Different Model Types")
    print("=" * 70)
    
    model_types = ["heuristic", "logistic", "anomaly", "correlation"]
    
    for model_type in model_types:
        print(f"\n  Testing model type: {model_type}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            governance_path = tmpdir_path / f"test_governance_{model_type}"
            
            script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
            
            cmd = [
                sys.executable,
                str(script_path),
                "--model-type", model_type,
                "--epochs", "1",
                "--num-samples", "30",
                "--seed", "42",
                "--enable-integrated-governance",
                "--governance-storage-dir", str(governance_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Check that training succeeded
            assert result.returncode == 0, f"Training failed for {model_type} with return code {result.returncode}"
            
            # Check that IntegratedGovernance was used
            assert "IntegratedGovernance enabled" in result.stdout, f"IntegratedGovernance not enabled for {model_type}"
            
            # Check that governance data was created
            assert governance_path.exists(), f"Governance data not created for {model_type}"
            
            print(f"  ✓ {model_type} model training with IntegratedGovernance successful")
    
    print("\n✓ All model types tested successfully!")
    return True


def test_integrated_governance_supersedes_individual_flags():
    """Test that IntegratedGovernance supersedes individual audit and drift tracking flags."""
    print("\n" + "=" * 70)
    print("  Test: IntegratedGovernance Supersedes Individual Flags")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        governance_path = tmpdir_path / "test_governance"
        audit_path = tmpdir_path / "test_audit"  # Should be ignored
        drift_path = tmpdir_path / "test_drift"  # Should be ignored
        
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "1",
            "--num-samples", "30",
            "--seed", "42",
            "--enable-integrated-governance",
            "--governance-storage-dir", str(governance_path),
            "--enable-audit",  # Should be superseded
            "--audit-path", str(audit_path),
            "--enable-drift-tracking",  # Should be superseded
            "--drift-report-dir", str(drift_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check that IntegratedGovernance was used
        assert "IntegratedGovernance enabled" in result.stdout, "IntegratedGovernance not enabled"
        
        # Check that governance path was created
        assert governance_path.exists(), "Governance path not created"
        
        # Check that individual paths were NOT created
        # (IntegratedGovernance should handle everything)
        # Note: The individual paths might exist as empty dirs, but should not have the specific files
        if audit_path.exists():
            # Check that Merkle audit files are not in the separate audit path
            chunk_files = list(audit_path.glob("chunk_*.json"))
            assert len(chunk_files) == 0, "Separate audit files should not be created when using IntegratedGovernance"
        
        if drift_path.exists():
            # Check that drift report files are not in the separate drift path
            drift_files = list(drift_path.glob("drift_*.json"))
            assert len(drift_files) == 0, "Separate drift files should not be created when using IntegratedGovernance"
        
        # Verify that governance data contains the expected merged functionality
        assert (governance_path / "merkle_data").exists(), "Merkle data not in governance dir"
        assert (governance_path / "drift_reports").exists(), "Drift reports not in governance dir"
        
        print("\n✓ IntegratedGovernance correctly supersedes individual flags!")
        return True


if __name__ == "__main__":
    try:
        test_train_with_integrated_governance()
        test_train_with_different_model_types()
        test_integrated_governance_supersedes_individual_flags()
        print("\n" + "=" * 70)
        print("  All IntegratedGovernance Integration Tests Passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
