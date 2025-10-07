#!/usr/bin/env python3
"""
Phase 8-9 Training Integration Demo

This example demonstrates how to use Phase89IntegratedGovernance
with the model training pipeline to:
1. Track configurations and metrics
2. Optimize hyperparameters
3. Validate with promotion gates

Usage:
    python examples/train_with_phase89.py
"""

import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_basic_training():
    """Run basic training with Phase89 enabled."""
    print("=" * 70)
    print("1. Basic Training with Phase89")
    print("=" * 70)
    print("\nRunning: Basic model training with Phase89 governance...\n")
    
    cmd = [
        "python", "training/train_any_model.py",
        "--model-type", "heuristic",
        "--epochs", "3",
        "--num-samples", "200",
        "--enable-phase89",
        "--seed", "123"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("Error:", result.stderr)
        return False
    
    return True


def run_training_with_optimization():
    """Run training with configuration optimization."""
    print("\n" + "=" * 70)
    print("2. Training with Configuration Optimization")
    print("=" * 70)
    print("\nRunning: Training with random search optimization...\n")
    
    cmd = [
        "python", "training/train_any_model.py",
        "--model-type", "logistic",
        "--epochs", "3",
        "--num-samples", "200",
        "--enable-phase89",
        "--optimize-config",
        "--optimization-technique", "random_search",
        "--optimization-iterations", "10",
        "--seed", "456"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("Error:", result.stderr)
        return False
    
    return True


def run_full_pipeline():
    """Run complete pipeline with all features."""
    print("\n" + "=" * 70)
    print("3. Complete Pipeline with All Features")
    print("=" * 70)
    print("\nRunning: Full pipeline with audit, drift, and Phase89...\n")
    
    cmd = [
        "python", "training/train_any_model.py",
        "--model-type", "heuristic",
        "--epochs", "3",
        "--num-samples", "200",
        "--enable-audit",
        "--enable-drift-tracking",
        "--enable-phase89",
        "--optimize-config",
        "--optimization-iterations", "5",
        "--seed", "789"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("Error:", result.stderr)
        return False
    
    return True


def main():
    """Run all demonstrations."""
    print("=" * 70)
    print("Phase 8-9 Training Integration Demo")
    print("=" * 70)
    print("\nThis demo shows how Phase89IntegratedGovernance integrates")
    print("with the model training pipeline.\n")
    
    # Run demonstrations
    success1 = run_basic_training()
    success2 = run_training_with_optimization()
    success3 = run_full_pipeline()
    
    # Summary
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    
    results = [
        ("Basic Training with Phase89", success1),
        ("Training with Optimization", success2),
        ("Full Pipeline", success3)
    ]
    
    print("\nResults:")
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print("\nKey Takeaways:")
    print("  ✓ Phase89 tracks configurations and metrics during training")
    print("  ✓ Optimization searches for better hyperparameter settings")
    print("  ✓ Promotion gates validate candidates before deployment")
    print("  ✓ Integrates seamlessly with audit and drift tracking")
    print("\nData stored in:")
    print("  - training_governance_data/    (Phase89 governance data)")
    print("  - training_audit_logs/         (Merkle audit trail)")
    print("  - training_drift_reports/      (Drift analysis)")
    print("  - models/                      (Trained models)")
    print()


if __name__ == "__main__":
    main()
