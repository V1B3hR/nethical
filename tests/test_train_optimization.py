#!/usr/bin/env python3
"""Test optimization integration in train_any_model.py."""
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_with_optimization_baseline():
    """Test train_any_model.py with optimization - create baseline configuration."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with Optimization (Baseline)")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        opt_db = tmpdir_path / "optimization.db"
        
        # Run training with optimization enabled to create baseline
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "2",
            "--num-samples", "100",
            "--seed", "42",
            "--enable-optimization",
            "--optimization-db", str(opt_db)
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
        
        # Check that optimization database was created
        assert opt_db.exists(), "Optimization database not created"
        
        # Check that configuration was recorded
        assert "Recorded metrics for configuration:" in result.stdout, "Metrics not recorded"
        
        # Extract configuration ID from output
        config_id = None
        for line in result.stdout.split('\n'):
            if "Recorded metrics for configuration:" in line:
                config_id = line.split(":")[-1].strip()
                break
        
        assert config_id is not None, "Configuration ID not found in output"
        print(f"\n✓ Baseline configuration created: {config_id}")
        
        return config_id, str(opt_db)


def test_train_with_optimization_candidate():
    """Test train_any_model.py with optimization - test promotion gate."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py with Optimization (Candidate)")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        opt_db = tmpdir_path / "optimization.db"
        
        # First, create a baseline configuration
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        # Create baseline
        cmd_baseline = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "1",
            "--num-samples", "100",
            "--seed", "42",
            "--enable-optimization",
            "--optimization-db", str(opt_db)
        ]
        
        print(f"\nCreating baseline: {' '.join(cmd_baseline)}")
        result_baseline = subprocess.run(cmd_baseline, capture_output=True, text=True, timeout=60)
        
        assert result_baseline.returncode == 0, "Baseline training failed"
        
        # Extract baseline config ID
        baseline_id = None
        for line in result_baseline.stdout.split('\n'):
            if "Recorded metrics for configuration:" in line:
                baseline_id = line.split(":")[-1].strip()
                break
        
        assert baseline_id is not None, "Baseline configuration ID not found"
        print(f"✓ Baseline created: {baseline_id}")
        
        # Now create a candidate and test against baseline
        cmd_candidate = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "1",
            "--num-samples", "100",
            "--seed", "43",  # Different seed for variation
            "--enable-optimization",
            "--optimization-db", str(opt_db),
            "--baseline-config-id", baseline_id
        ]
        
        print(f"\nCreating candidate: {' '.join(cmd_candidate)}")
        result_candidate = subprocess.run(cmd_candidate, capture_output=True, text=True, timeout=60)
        
        print("\nCandidate STDOUT:")
        print(result_candidate.stdout)
        
        # Check that training succeeded
        assert result_candidate.returncode == 0, f"Candidate training failed with return code {result_candidate.returncode}"
        
        # Check that promotion gate was evaluated
        assert "Promotion Gate Results:" in result_candidate.stdout, "Promotion gate not evaluated"
        
        # Check if advanced promotion gate was used
        if "Using advanced promotion gate" in result_candidate.stdout:
            print("\n✓ Advanced promotion gate with multi-objective criteria was used")
        else:
            print("\n! Simple promotion gate was used (baseline might not have sufficient metrics)")
        
        print("\n✓ Candidate evaluation completed successfully")
        return True


def test_train_without_optimization():
    """Test train_any_model.py without optimization (ensure backward compatibility)."""
    print("\n" + "=" * 70)
    print("  Test: train_any_model.py without Optimization")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run training WITHOUT optimization
        script_path = Path(__file__).parent.parent / "training" / "train_any_model.py"
        
        cmd = [
            sys.executable,
            str(script_path),
            "--model-type", "heuristic",
            "--epochs", "1",
            "--num-samples", "50",
            "--seed", "42"
        ]
        
        print(f"\nRunning: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print("\nSTDOUT (last 20 lines):")
        stdout_lines = result.stdout.split('\n')
        print('\n'.join(stdout_lines[-20:]))
        
        # Check that training succeeded
        assert result.returncode == 0, f"Training failed with return code {result.returncode}"
        
        # Check that optimization was not mentioned
        assert "optimization enabled" not in result.stdout.lower(), "Optimization should not be active"
        
        # Check that simple promotion gate was used
        assert "Promotion Gate: ECE" in result.stdout, "Simple promotion gate not used"
        
        print("\n✓ Training without optimization works correctly!")
        return True


if __name__ == "__main__":
    try:
        test_train_with_optimization_baseline()
        test_train_with_optimization_candidate()
        test_train_without_optimization()
        print("\n" + "=" * 70)
        print("  All Optimization Tests Passed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
