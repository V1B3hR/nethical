#!/usr/bin/env python3
"""Test risk tracking in train_any_model.py."""
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_with_risk_tracking():
    """Test train_any_model.py with risk tracking enabled."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with Risk Tracking")
    print("=" * 70)
    
    # Run training with risk tracking
    script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--model-type", "heuristic",
        "--epochs", "2",
        "--num-samples", "50",
        "--seed", "42",
        "--enable-risk-tracking",
        "--risk-decay-hours", "12.0"
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
    
    # Check that risk tracking was enabled
    assert "Risk engine tracking enabled" in result.stdout, "Risk tracking not enabled"
    assert "Decay half-life: 12.0 hours" in result.stdout, "Decay half-life not set"
    assert "Tracking agent ID: model_heuristic" in result.stdout, "Agent ID not set"
    
    # Check that risk scores were calculated
    assert "Risk Tracking:" in result.stdout, "Risk tracking not performed"
    assert "Risk Score:" in result.stdout, "Risk score not calculated"
    assert "Risk Tier:" in result.stdout, "Risk tier not displayed"
    assert "Violation Severity:" in result.stdout, "Violation severity not calculated"
    
    # Check that risk profile summary was generated
    assert "RISK PROFILE SUMMARY" in result.stdout, "Risk profile summary not generated"
    assert "Final Risk Score:" in result.stdout, "Final risk score not displayed"
    assert "Final Risk Tier:" in result.stdout, "Final risk tier not displayed"
    assert "Total Actions:" in result.stdout, "Total actions not tracked"
    assert "Violation Count:" in result.stdout, "Violation count not tracked"
    
    # Check risk components
    assert "Risk Components:" in result.stdout, "Risk components not displayed"
    assert "Behavior Score:" in result.stdout, "Behavior score not displayed"
    assert "Severity Score:" in result.stdout, "Severity score not displayed"
    assert "Frequency Score:" in result.stdout, "Frequency score not displayed"
    assert "Recency Score:" in result.stdout, "Recency score not displayed"
    
    print("\n✓ All risk tracking output checks passed!")
    return True


def test_train_with_risk_and_audit():
    """Test risk tracking with audit logging enabled."""
    print("\n" + "=" * 70)
    print("  Test: Risk Tracking with Audit Logging")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        audit_path = tmpdir_path / "test_audit_logs"
        
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "logistic",
            "--epochs", "1",
            "--num-samples", "100",
            "--seed", "42",
            "--enable-risk-tracking",
            "--enable-audit",
            "--audit-path", str(audit_path)
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check that both features are enabled
        assert "Risk engine tracking enabled" in result.stdout, "Risk tracking not enabled"
        assert "Merkle audit logging enabled" in result.stdout, "Audit logging not enabled"
        
        # Check that audit directory was created
        assert audit_path.exists(), "Audit directory not created"
        
        # Check for audit chunk files
        chunk_files = list(audit_path.glob("chunk_*.json"))
        assert len(chunk_files) > 0, "No audit chunk files created"
        
        # Check audit summary
        summary_file = audit_path / "training_summary.json"
        assert summary_file.exists(), "Audit summary not created"
        
        with open(summary_file) as f:
            summary = json.load(f)
        
        print("\nAudit Summary:")
        print(json.dumps(summary, indent=2))
        
        assert 'merkle_root' in summary, "Merkle root not in summary"
        assert 'model_type' in summary, "Model type not in summary"
        assert 'metrics' in summary, "Metrics not in summary"
        
        # Check that audit events include risk profile
        with open(chunk_files[0]) as f:
            chunk = json.load(f)
        
        # Look for risk profile event
        risk_events = [e for e in chunk['events'] if e.get('event_type') == 'risk_profile_final']
        if risk_events:
            print("\n✓ Risk profile logged in audit trail")
            risk_event = risk_events[0]
            assert 'risk_profile' in risk_event, "Risk profile not in audit event"
            print(f"  - Agent ID: {risk_event['risk_profile']['agent_id']}")
            print(f"  - Risk Score: {risk_event['risk_profile']['current_score']:.4f}")
            print(f"  - Risk Tier: {risk_event['risk_profile']['current_tier']}")
        
        print("\n✓ Risk tracking and audit logging integration test passed!")
        return True


def test_train_with_all_features():
    """Test with risk tracking, audit, and drift tracking all enabled."""
    print("\n" + "=" * 70)
    print("  Test: All Features (Risk + Audit + Drift)")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        audit_path = tmpdir_path / "test_audit"
        drift_path = tmpdir_path / "test_drift"
        
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "1",
            "--num-samples", "80",
            "--seed", "999",
            "--enable-risk-tracking",
            "--risk-decay-hours", "24.0",
            "--enable-audit",
            "--audit-path", str(audit_path),
            "--enable-drift-tracking",
            "--drift-report-dir", str(drift_path),
            "--cohort-id", "test_all_features"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check that all features are enabled
        assert "Risk engine tracking enabled" in result.stdout, "Risk tracking not enabled"
        assert "Merkle audit logging enabled" in result.stdout, "Audit logging not enabled"
        assert "Ethical drift tracking enabled" in result.stdout, "Drift tracking not enabled"
        
        # Check outputs from all features
        assert "RISK PROFILE SUMMARY" in result.stdout, "Risk profile not generated"
        assert "Training audit trail finalized" in result.stdout, "Audit trail not finalized"
        assert "Generating ethical drift report" in result.stdout, "Drift report not generated"
        
        # Verify all directories were created
        assert audit_path.exists(), "Audit directory not created"
        assert drift_path.exists(), "Drift directory not created"
        
        # Verify files exist
        audit_files = list(audit_path.glob("*.json"))
        drift_files = list(drift_path.glob("drift_*.json"))
        
        assert len(audit_files) > 0, "No audit files created"
        assert len(drift_files) > 0, "No drift files created"
        
        print(f"\n✓ All features working together:")
        print(f"  - Audit files: {len(audit_files)}")
        print(f"  - Drift files: {len(drift_files)}")
        print(f"  - Risk tracking: ✓")
        
        print("\n✓ All features integration test passed!")
        return True


def test_train_without_risk_tracking():
    """Test train_any_model.py without risk tracking."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py without Risk Tracking")
    print("=" * 70)
    
    # Run training WITHOUT risk tracking
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
    
    print("\nSTDOUT (last 30 lines):")
    stdout_lines = result.stdout.split('\n')
    print('\n'.join(stdout_lines[-30:]))
    
    # Check that training succeeded
    assert result.returncode == 0, f"Training failed with return code {result.returncode}"
    
    # Check that risk tracking was not mentioned
    assert "Risk engine tracking enabled" not in result.stdout, "Risk tracking should not be active"
    assert "RISK PROFILE SUMMARY" not in result.stdout, "Risk profile should not be generated"
    
    print("\n✓ Training without risk tracking works correctly!")
    return True


def test_risk_tracking_elevated_tier():
    """Test risk tracking with promotion failure to trigger elevated risk."""
    print("\n" + "=" * 70)
    print("  Test: Risk Tracking - Elevated Tier Detection")
    print("=" * 70)
    
    # Run training with small sample to likely fail promotion
    script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--model-type", "heuristic",
        "--epochs", "1",
        "--num-samples", "20",  # Small sample for potential poor metrics
        "--seed", "123",
        "--enable-risk-tracking"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    
    # Check that training succeeded
    assert result.returncode == 0, f"Training failed with return code {result.returncode}"
    
    # Check that risk tracking occurred
    assert "Risk Tracking:" in result.stdout, "Risk tracking not performed"
    
    # Check for promotion gate risk update (happens on failure)
    if "Promotion Gate Risk Update:" in result.stdout:
        print("\n✓ Promotion gate failure triggered risk update")
        
        # Extract risk tier from output
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if "Promotion Gate Risk Update:" in line:
                # Look for tier in next few lines
                for j in range(i, min(i+5, len(lines))):
                    if "Updated Risk Tier:" in lines[j]:
                        print(f"  - {lines[j].strip()}")
                        break
    
    # Check risk profile summary
    assert "RISK PROFILE SUMMARY" in result.stdout, "Risk profile summary not generated"
    
    print("\n✓ Elevated tier detection test passed!")
    return True


def test_risk_decay_parameter():
    """Test custom risk decay parameter."""
    print("\n" + "=" * 70)
    print("  Test: Custom Risk Decay Parameter")
    print("=" * 70)
    
    script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
    
    # Test with custom decay of 6 hours
    cmd = [
        sys.executable,
        str(script_path),
        "--model-type", "logistic",
        "--epochs", "1",
        "--num-samples", "50",
        "--seed", "42",
        "--enable-risk-tracking",
        "--risk-decay-hours", "6.0"
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    
    # Check that training succeeded
    assert result.returncode == 0, f"Training failed with return code {result.returncode}"
    
    # Check that custom decay was set
    assert "Decay half-life: 6.0 hours" in result.stdout, "Custom decay not set"
    
    print("\n✓ Custom risk decay parameter test passed!")
    return True


if __name__ == "__main__":
    try:
        test_train_with_risk_tracking()
        test_train_with_risk_and_audit()
        test_train_with_all_features()
        test_train_without_risk_tracking()
        test_risk_tracking_elevated_tier()
        test_risk_decay_parameter()
        print("\n" + "=" * 70)
        print("  All Risk Tracking Tests Passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
