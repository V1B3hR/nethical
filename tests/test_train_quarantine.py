#!/usr/bin/env python3
"""Test quarantine integration in train_any_model.py."""
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_with_quarantine_enabled():
    """Test train_any_model.py with quarantine enabled but no auto-quarantine."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with Quarantine Enabled (No Auto-Quarantine)")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Run training with quarantine enabled
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "50",
            "--seed", "42",
            "--enable-quarantine"
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
        
        # Check that quarantine system was enabled
        assert "Quarantine system enabled" in result.stdout
        assert "Quarantine System Summary" in result.stdout
        
        print("\n✅ Test passed: Quarantine system initialized successfully")


def test_train_with_quarantine_on_failure():
    """Test train_any_model.py with auto-quarantine on failure."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with Auto-Quarantine on Failure")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Run training with quarantine-on-failure
        # Use small dataset to increase chance of failure
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "logistic",
            "--epochs", "2",
            "--num-samples", "100",
            "--seed", "42",
            "--enable-quarantine",
            "--quarantine-on-failure"
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
        
        # Check that quarantine system was enabled
        assert "Quarantine system enabled" in result.stdout
        assert "Quarantine System Summary" in result.stdout
        
        # Check for either quarantine or no quarantine based on promotion result
        if "Promotion result: FAIL" in result.stdout:
            assert "Model failed promotion gate. Quarantining cohort" in result.stdout
            assert "Is Quarantined: YES" in result.stdout
            print("\n✅ Test passed: Model was quarantined after failing promotion gate")
        else:
            assert "Is Quarantined: NO" in result.stdout
            print("\n✅ Test passed: Model passed promotion gate and was not quarantined")


def test_train_with_quarantine_and_audit():
    """Test train_any_model.py with both quarantine and audit logging."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with Quarantine and Audit Logging")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        audit_path = tmpdir_path / "test_audit_logs"
        
        # Run training with both quarantine and audit
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "anomaly",
            "--epochs", "2",
            "--num-samples", "200",
            "--seed", "42",
            "--enable-quarantine",
            "--quarantine-on-failure",
            "--enable-audit",
            "--audit-path", str(audit_path)
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
        
        # Check that both systems were enabled
        assert "Merkle audit logging enabled" in result.stdout
        assert "Quarantine system enabled" in result.stdout
        assert "Quarantine System Summary" in result.stdout
        
        # Check that audit logs were created
        assert audit_path.exists(), "Audit logs directory was not created"
        
        # Check for audit summary file
        summary_file = audit_path / "training_summary.json"
        assert summary_file.exists(), "Audit summary file was not created"
        
        # Load and validate summary
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        assert 'merkle_root' in summary
        assert 'metrics' in summary
        assert 'promoted' in summary
        
        print("\n✅ Test passed: Quarantine and audit logging integrated successfully")


def test_quarantine_activation_time():
    """Test that quarantine activation time is under target (<15s)."""
    print("\n" + "=" * 70)
    print("  Test: Quarantine Activation Time")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Run training with quarantine-on-failure
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "logistic",
            "--epochs", "2",
            "--num-samples", "100",
            "--seed", "42",
            "--enable-quarantine",
            "--quarantine-on-failure"
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
        
        # If model was quarantined, check activation time
        if "Is Quarantined: YES" in result.stdout:
            # Extract activation time from output
            for line in result.stdout.split('\n'):
                if "Activation time:" in line and "ms" in line:
                    # Parse activation time
                    time_str = line.split("Activation time:")[1].strip().replace("ms", "")
                    activation_time_ms = float(time_str)
                    
                    # Should be under 15000ms (15 seconds)
                    assert activation_time_ms < 15000, f"Activation time {activation_time_ms}ms exceeds 15s target"
                    print(f"\n✅ Test passed: Quarantine activation time {activation_time_ms}ms is under 15s target")
                    break
        else:
            print("\n✅ Test passed: Model was not quarantined (passed promotion gate)")


def test_quarantine_statistics():
    """Test that quarantine statistics are reported correctly."""
    print("\n" + "=" * 70)
    print("  Test: Quarantine Statistics Reporting")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Run training with quarantine
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "50",
            "--seed", "42",
            "--enable-quarantine",
            "--quarantine-on-failure"
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
        
        # Check that statistics are present
        assert "Quarantine Statistics:" in result.stdout
        assert "Active quarantines:" in result.stdout
        assert "Total quarantines:" in result.stdout
        assert "Avg activation time:" in result.stdout
        assert "Target activation time:" in result.stdout
        
        print("\n✅ Test passed: Quarantine statistics reported successfully")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  Running Quarantine Integration Tests")
    print("=" * 70)
    
    tests = [
        test_train_with_quarantine_enabled,
        test_train_with_quarantine_on_failure,
        test_train_with_quarantine_and_audit,
        test_quarantine_activation_time,
        test_quarantine_statistics
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ Test error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"  Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
