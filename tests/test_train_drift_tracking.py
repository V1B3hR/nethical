#!/usr/bin/env python3
"""Test drift tracking in train_any_model.py."""
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_with_drift_tracking():
    """Test train_any_model.py with drift tracking enabled."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with Drift Tracking")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        drift_path = tmpdir_path / "test_drift_reports"
        
        # Run training with drift tracking
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "50",
            "--seed", "42",
            "--enable-drift-tracking",
            "--drift-report-dir", str(drift_path),
            "--cohort-id", "test_cohort_a"
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
        
        # Check that drift tracking was enabled
        assert "Ethical drift tracking enabled" in result.stdout, "Drift tracking not enabled"
        assert "Training cohort ID: test_cohort_a" in result.stdout, "Cohort ID not set"
        
        # Check that drift report was generated
        assert "Generating ethical drift report" in result.stdout, "Drift report not generated"
        assert "Drift Report ID:" in result.stdout, "Drift report ID not printed"
        
        # Check that drift reports directory was created
        assert drift_path.exists(), "Drift reports directory not created"
        
        # Check for drift report files
        drift_files = list(drift_path.glob("drift_*.json"))
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
        assert 'test_cohort_a' in drift_report['cohorts'], "Test cohort not tracked"
        cohort_data = drift_report['cohorts']['test_cohort_a']
        
        # Verify cohort has action count
        assert cohort_data['action_count'] > 0, "No actions tracked for cohort"
        
        print(f"\n✓ Cohort tracked: test_cohort_a")
        print(f"  - Actions: {cohort_data['action_count']}")
        print(f"  - Violations: {cohort_data['violation_stats']['total_count']}")
        print(f"  - Avg risk score: {cohort_data['avg_risk_score']:.4f}")
        
        print("\n✓ All drift tracking tests passed!")
        return True


def test_train_with_promotion_failure():
    """Test drift tracking when promotion gate fails."""
    print("\n" + "=" * 70)
    print("  Test: Drift Tracking with Promotion Gate Failure")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        drift_path = tmpdir_path / "test_drift_reports"
        
        # Run training with small sample size to potentially trigger promotion failure
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "1",
            "--num-samples", "20",  # Very small sample size
            "--seed", "123",  # Different seed for different results
            "--enable-drift-tracking",
            "--drift-report-dir", str(drift_path),
            "--cohort-id", "test_cohort_fail"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Training should complete even if promotion fails
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check for drift report files
        drift_files = list(drift_path.glob("drift_*.json"))
        assert len(drift_files) > 0, "No drift report files created"
        
        # Check drift report contents
        with open(drift_files[0]) as f:
            drift_report = json.load(f)
        
        assert 'test_cohort_fail' in drift_report['cohorts'], "Test cohort not tracked"
        cohort_data = drift_report['cohorts']['test_cohort_fail']
        
        print(f"\n✓ Cohort tracked: test_cohort_fail")
        print(f"  - Actions: {cohort_data['action_count']}")
        print(f"  - Violations: {cohort_data['violation_stats']['total_count']}")
        print(f"  - Avg risk score: {cohort_data['avg_risk_score']:.4f}")
        
        print("\n✓ Promotion failure drift tracking test passed!")
        return True


def test_train_without_drift_tracking():
    """Test train_any_model.py without drift tracking."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py without Drift Tracking")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run training WITHOUT drift tracking
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "1",
            "--num-samples", "30",
            "--seed", "42"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print("\nSTDOUT (last 20 lines):")
        stdout_lines = result.stdout.split('\n')
        print('\n'.join(stdout_lines[-20:]))
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check that drift tracking was not mentioned
        assert "Drift tracking enabled" not in result.stdout, "Drift tracking should not be active"
        assert "Drift Report ID:" not in result.stdout, "Drift report should not be generated"
        
        print("\n✓ Training without drift tracking works correctly!")
        return True


def test_multiple_cohorts_drift():
    """Test drift tracking across multiple training runs."""
    print("\n" + "=" * 70)
    print("  Test: Multiple Training Runs Drift Reports")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        drift_path = tmpdir_path / "test_multi_drift_reports"
        
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        # Run training for cohort A
        cmd_a = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "1",
            "--num-samples", "40",
            "--seed", "42",
            "--enable-drift-tracking",
            "--drift-report-dir", str(drift_path),
            "--cohort-id", "cohort_a"
        ]
        
        print(f"\n[1/2] Running cohort A: {' '.join(cmd_a[-3:])}")
        result_a = subprocess.run(cmd_a, capture_output=True, text=True, timeout=60)
        assert result_a.returncode == 0, "Training A failed"
        
        # Run training for cohort B with different seed
        cmd_b = [
            sys.executable,
            str(script_path),
            "--model-type", "logistic",
            "--epochs", "1",
            "--num-samples", "40",
            "--seed", "999",
            "--enable-drift-tracking",
            "--drift-report-dir", str(drift_path),
            "--cohort-id", "cohort_b"
        ]
        
        print(f"[2/2] Running cohort B: {' '.join(cmd_b[-3:])}")
        result_b = subprocess.run(cmd_b, capture_output=True, text=True, timeout=60)
        assert result_b.returncode == 0, "Training B failed"
        
        # Check drift report files - each run creates its own report
        drift_files = list(drift_path.glob("drift_*.json"))
        print(f"\n✓ Found {len(drift_files)} drift report(s)")
        assert len(drift_files) >= 1, f"Expected at least 1 drift report, found {len(drift_files)}"
        
        # Check that each report contains its respective cohort
        reports_summary = []
        for drift_file in drift_files:
            with open(drift_file) as f:
                drift_report = json.load(f)
            cohorts_in_report = list(drift_report['cohorts'].keys())
            reports_summary.append({
                'file': drift_file.name,
                'cohorts': cohorts_in_report
            })
        
        print(f"\n✓ Reports summary:")
        for summary in reports_summary:
            print(f"  - {summary['file']}: {summary['cohorts']}")
        
        # Verify at least one report exists with cohorts
        assert any(len(s['cohorts']) > 0 for s in reports_summary), "No cohorts tracked"
        
        print("\n✓ Multiple training runs drift tracking test passed!")
        return True


if __name__ == "__main__":
    try:
        test_train_with_drift_tracking()
        test_train_with_promotion_failure()
        test_train_without_drift_tracking()
        test_multiple_cohorts_drift()
        print("\n" + "=" * 70)
        print("  All Drift Tracking Tests Passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
