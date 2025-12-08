#!/usr/bin/env python3
"""Test the governance integration in train_any_model.py."""
import json
import shutil
import tempfile
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_with_governance():
    """Test train_any_model.py with governance enabled."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with Governance Validation")
    print("=" * 70)

    # Get the repo root directory
    repo_root = Path(__file__).parent.parent

    try:
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

            print("\n[1/4] Running training with governance enabled...")

            # Run training with governance from repo root
            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "training" / "train_any_model.py"),
                    "--model-type",
                    "heuristic",
                    "--epochs",
                    "1",
                    "--num-samples",
                    "50",
                    "--enable-governance",
                ],
                cwd=tmpdir_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

            print(f"  ✓ Training completed with exit code: {result.returncode}")

            # Check for governance messages in output
            assert (
                "Governance validation enabled" in result.stdout
                or "Governance validation enabled" in result.stderr
            ), "Governance not initialized"
            print("  ✓ Governance system initialized")

            # Check for governance validation messages
            assert (
                "Running governance validation on training data samples"
                in result.stdout
                or "Running governance validation on training data samples"
                in result.stderr
            ), "Data validation not performed"
            print("  ✓ Data validation performed")

            assert (
                "Running governance validation on model predictions" in result.stdout
                or "Running governance validation on model predictions" in result.stderr
            ), "Prediction validation not performed"
            print("  ✓ Prediction validation performed")

            # Check for governance summary
            assert (
                "Governance Validation Summary" in result.stdout
                or "Governance Validation Summary" in result.stderr
            ), "Governance summary not printed"
            print("  ✓ Governance summary printed")

            print("\n[2/4] Training without governance (backward compatibility)...")

            # Run training without governance from repo root
            result2 = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "training" / "train_any_model.py"),
                    "--model-type",
                    "heuristic",
                    "--epochs",
                    "1",
                    "--num-samples",
                    "50",
                ],
                cwd=tmpdir_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

            print(f"  ✓ Training completed with exit code: {result2.returncode}")

            # Should not have governance messages
            assert (
                "Governance validation enabled" not in result2.stdout
                and "Governance validation enabled" not in result2.stderr
            ), "Governance should not be enabled"
            print("  ✓ Governance not initialized (as expected)")

            print("\n[3/4] Testing governance with audit logging...")

            audit_dir = tmpdir_path / "test_audit"
            audit_dir.mkdir()

            result3 = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "training" / "train_any_model.py"),
                    "--model-type",
                    "heuristic",
                    "--epochs",
                    "1",
                    "--num-samples",
                    "50",
                    "--enable-governance",
                    "--enable-audit",
                    "--audit-path",
                    str(audit_dir),
                ],
                cwd=tmpdir_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

            print(f"  ✓ Training completed with exit code: {result3.returncode}")

            # Check audit summary exists
            audit_summary = audit_dir / "training_summary.json"
            assert audit_summary.exists(), "Audit summary not created"
            print("  ✓ Audit summary created")

            # Load and verify audit summary
            with open(audit_summary) as f:
                summary = json.load(f)

            assert "governance" in summary, "Governance section missing from audit"
            assert (
                summary["governance"]["enabled"] is True
            ), "Governance not marked as enabled"
            assert (
                "data_violations" in summary["governance"]
            ), "Data violations not recorded"
            assert (
                "prediction_violations" in summary["governance"]
            ), "Prediction violations not recorded"
            print("  ✓ Governance metrics saved to audit summary")

            print("\n[4/4] Verifying governance metrics...")
            print(f"  Data violations: {summary['governance']['data_violations']}")
            print(
                f"  Prediction violations: {summary['governance']['prediction_violations']}"
            )

            if "total_violations_detected" in summary["governance"]:
                print(
                    f"  Total violations detected: {summary['governance']['total_violations_detected']}"
                )

            print("\n" + "=" * 70)
            print("  ✅ ALL GOVERNANCE TESTS PASSED")
            print("=" * 70)

    except subprocess.TimeoutExpired:
        print("  ❌ Test timed out")
        raise
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        raise


def test_governance_without_persistence():
    """Verify that governance works without persistence enabled."""
    print("\n" + "=" * 70)
    print("  Test: Governance Without Persistence")
    print("=" * 70)

    print("\n[INFO] Testing that governance initializes without database issues...")

    # Import and test
    from training.train_any_model import GOVERNANCE_AVAILABLE

    if not GOVERNANCE_AVAILABLE:
        print("  ⚠️  Governance not available, skipping test")
        return

    from nethical.core.governance import EnhancedSafetyGovernance, MonitoringConfig

    # Create governance with persistence disabled
    config = MonitoringConfig()
    config.enable_persistence = False

    gov = EnhancedSafetyGovernance(config=config)
    print("  ✓ Governance initialized without persistence")

    # Verify it has no database path
    assert gov.persistence is None, "Persistence should be disabled"
    print("  ✓ Persistence confirmed disabled")

    print("\n" + "=" * 70)
    print("  ✅ PERSISTENCE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_train_with_governance()
    test_governance_without_persistence()
    print("\n" + "=" * 70)
    print("  ALL TESTS PASSED ✅")
    print("=" * 70)
