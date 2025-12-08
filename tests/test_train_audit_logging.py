#!/usr/bin/env python3
"""Test audit logging in train_any_model.py."""
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_with_audit_logging():
    """Test train_any_model.py with audit logging enabled."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with Audit Logging")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        audit_path = tmpdir_path / "test_audit_logs"

        # Run training with audit logging
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--model-type",
            "heuristic",
            "--epochs",
            "2",
            "--num-samples",
            "50",
            "--seed",
            "42",
            "--enable-audit",
            "--audit-path",
            str(audit_path),
        ]

        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        print("\nSTDOUT:")
        print(result.stdout)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        # Check that training succeeded
        assert (
            result.returncode == 0
        ), f"Training failed with return code {result.returncode}"

        # Check that audit logs were created
        assert audit_path.exists(), "Audit log directory not created"

        # Check for summary file
        summary_file = audit_path / "training_summary.json"
        assert summary_file.exists(), "Training summary not created"

        # Check summary contents
        with open(summary_file) as f:
            summary = json.load(f)

        print("\nAudit Summary:")
        print(json.dumps(summary, indent=2))

        assert "merkle_root" in summary, "Merkle root not in summary"
        assert "model_type" in summary, "Model type not in summary"
        assert "metrics" in summary, "Metrics not in summary"
        assert summary["model_type"] == "heuristic", "Wrong model type in summary"

        # Check that Merkle root is a valid hash (64 chars for SHA-256)
        merkle_root = summary["merkle_root"]
        assert len(merkle_root) == 64, f"Invalid Merkle root length: {len(merkle_root)}"
        assert all(
            c in "0123456789abcdef" for c in merkle_root
        ), "Invalid Merkle root format"

        # Check for chunk file
        chunk_files = list(audit_path.glob("chunk_*.json"))
        assert len(chunk_files) > 0, "No chunk files created"

        # Check chunk contents
        with open(chunk_files[0]) as f:
            chunk_data = json.load(f)

        print("\nAudit Chunk Events:")
        for event in chunk_data["events"]:
            print(f"  - {event['event_type']}")

        # Verify all expected events are present
        event_types = [e["event_type"] for e in chunk_data["events"]]
        expected_events = [
            "training_start",
            "data_loaded",
            "data_split",
            "training_completed",
            "validation_metrics",
            "model_saved",
        ]

        for expected in expected_events:
            assert expected in event_types, f"Missing event: {expected}"

        # Verify chunk has Merkle root
        assert chunk_data["merkle_root"] == merkle_root, "Merkle root mismatch"
        assert chunk_data["event_count"] == len(
            chunk_data["events"]
        ), "Event count mismatch"

        print("\n✓ All audit logging tests passed!")
        return True


def test_train_without_audit_logging():
    """Test train_any_model.py without audit logging."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py without Audit Logging")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Run training WITHOUT audit logging
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--model-type",
            "heuristic",
            "--epochs",
            "1",
            "--num-samples",
            "30",
            "--seed",
            "42",
        ]

        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        print("\nSTDOUT (last 20 lines):")
        stdout_lines = result.stdout.split("\n")
        print("\n".join(stdout_lines[-20:]))

        # Check that training succeeded
        assert (
            result.returncode == 0
        ), f"Training failed with return code {result.returncode}"

        # Check that no audit logs were mentioned
        assert "Merkle root:" not in result.stdout, "Audit logging should not be active"

        print("\n✓ Training without audit logging works correctly!")
        return True


if __name__ == "__main__":
    try:
        test_train_with_audit_logging()
        test_train_without_audit_logging()
        print("\n" + "=" * 70)
        print("  All Tests Passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
