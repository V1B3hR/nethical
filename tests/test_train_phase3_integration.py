#!/usr/bin/env python3
"""Test Phase3 integration in train_any_model.py."""
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_with_phase3_integration():
    """Test train_any_model.py with Phase3 integrated governance enabled."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with Phase3 Integration")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run training WITH Phase3 integration
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        phase3_storage = Path(tmpdir) / "phase3_storage"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "50",
            "--seed", "42",
            "--enable-phase3",
            "--phase3-storage-dir", str(phase3_storage),
            "--cohort-id", "test_cohort_phase3"
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
        
        # Check that Phase3 integration was enabled
        assert "Phase3 integrated governance enabled" in result.stdout, "Phase3 integration not enabled"
        assert "Phase3 includes: Risk Engine, Correlation Engine" in result.stdout, "Phase3 components not listed"
        assert "Training cohort ID: test_cohort_phase3" in result.stdout, "Cohort ID not set"
        
        # Check that Phase3 drift report was generated
        assert "Generating ethical drift report using Phase3 integrated governance" in result.stdout, "Phase3 drift report not generated"
        assert "Drift Report ID:" in result.stdout, "Drift report ID not printed"
        
        # Check that Phase3 system status was printed
        assert "Phase3 System Status:" in result.stdout, "Phase3 system status not printed"
        assert "risk_engine" in result.stdout, "Risk engine status not printed"
        assert "correlation_engine" in result.stdout, "Correlation engine status not printed"
        assert "fairness_sampler" in result.stdout, "Fairness sampler status not printed"
        assert "ethical_drift_reporter" in result.stdout, "Ethical drift reporter status not printed"
        assert "performance_optimizer" in result.stdout, "Performance optimizer status not printed"
        
        # Check that Phase3 storage directory was created
        assert phase3_storage.exists(), "Phase3 storage directory not created"
        
        # Check for drift report files
        drift_reports_dir = phase3_storage / "drift_reports"
        assert drift_reports_dir.exists(), "Drift reports directory not created"
        
        drift_files = list(drift_reports_dir.glob("drift_*.json"))
        assert len(drift_files) > 0, "No drift report files created"
        
        # Check drift report contents
        with open(drift_files[0]) as f:
            drift_report = json.load(f)
        
        print("\nDrift Report:")
        print(json.dumps(drift_report, indent=2))
        
        assert 'report_id' in drift_report, "Report ID not in drift report"
        assert 'cohorts' in drift_report, "Cohorts not in drift report"
        assert 'drift_metrics' in drift_report, "Drift metrics not in drift report"
        assert 'recommendations' in drift_report, "Recommendations not in drift report"
        
        # Check that cohort was tracked
        assert 'test_cohort_phase3' in drift_report['cohorts'], "Test cohort not tracked"
        cohort_data = drift_report['cohorts']['test_cohort_phase3']
        
        print(f"\n✓ Cohort tracked: test_cohort_phase3")
        print(f"  - Actions: {cohort_data['action_count']}")
        print(f"  - Violations: {cohort_data['violation_stats']['total_count']}")
        print(f"  - Avg risk score: {cohort_data['avg_risk_score']:.4f}")
        
        print("\n✓ All Phase3 integration tests passed!")


def test_train_phase3_with_violations():
    """Test Phase3 integration with promotion gate failure (violations tracked)."""
    print("\n" + "=" * 70)
    print("  Test: Phase3 Integration with Violations")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run training WITH Phase3 integration (low samples to force failure)
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        phase3_storage = Path(tmpdir) / "phase3_storage"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "1",
            "--num-samples", "20",
            "--seed", "123",
            "--enable-phase3",
            "--phase3-storage-dir", str(phase3_storage),
            "--cohort-id", "test_cohort_violations"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Check that training completed
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check for drift report files
        drift_reports_dir = phase3_storage / "drift_reports"
        drift_files = list(drift_reports_dir.glob("drift_*.json"))
        assert len(drift_files) > 0, "No drift report files created"
        
        # Check drift report contents
        with open(drift_files[0]) as f:
            drift_report = json.load(f)
        
        # Check that violations were tracked
        cohort_data = drift_report['cohorts']['test_cohort_violations']
        violations = cohort_data['violation_stats']['total_count']
        
        print(f"\n✓ Cohort tracked: test_cohort_violations")
        print(f"  - Actions: {cohort_data['action_count']}")
        print(f"  - Violations: {violations}")
        print(f"  - Avg risk score: {cohort_data['avg_risk_score']:.4f}")
        
        # Since promotion gate likely failed with low samples, expect violations
        if "FAIL" in result.stdout:
            assert violations > 0, "Expected violations for failed promotion gate"
            print("\n✓ Violations tracked for promotion gate failure!")
        
        print("\n✓ Phase3 violation tracking test passed!")


def test_backward_compatibility():
    """Test that drift tracking still works when Phase3 is not enabled."""
    print("\n" + "=" * 70)
    print("  Test: Backward Compatibility (Drift Tracking without Phase3)")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run training WITHOUT Phase3 integration (use old drift tracking)
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        drift_dir = Path(tmpdir) / "drift_reports"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "1",
            "--num-samples", "30",
            "--seed", "42",
            "--enable-drift-tracking",  # Use old flag
            "--drift-report-dir", str(drift_dir),
            "--cohort-id", "test_cohort_compat"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check that old drift tracking was enabled (not Phase3)
        assert "Ethical drift tracking enabled" in result.stdout, "Drift tracking not enabled"
        assert "Phase3 integrated governance enabled" not in result.stdout, "Phase3 should not be enabled"
        
        # Check for drift report files
        assert drift_dir.exists(), "Drift reports directory not created"
        drift_files = list(drift_dir.glob("drift_*.json"))
        assert len(drift_files) > 0, "No drift report files created"
        
        print("\n✓ Backward compatibility maintained!")
        print("✓ Old drift tracking still works without Phase3!")


if __name__ == "__main__":
    try:
        test_train_with_phase3_integration()
        test_train_phase3_with_violations()
        test_backward_compatibility()
        print("\n" + "=" * 70)
        print("  All Phase3 Integration Tests Passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
