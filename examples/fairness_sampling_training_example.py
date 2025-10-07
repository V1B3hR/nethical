#!/usr/bin/env python3
"""
Example: Training with Fairness Sampling

This script demonstrates how to use the fairness sampling feature
in the training pipeline.
"""

import subprocess
import sys
from pathlib import Path

def run_training_with_fairness_sampling():
    """Run training with fairness sampling enabled."""
    
    print("=" * 70)
    print("Training with Fairness Sampling - Example")
    print("=" * 70)
    
    # Example 1: Basic fairness sampling
    print("\n[Example 1] Basic fairness sampling")
    print("-" * 70)
    
    cmd1 = [
        sys.executable,
        "training/train_any_model.py",
        "--model-type", "logistic",
        "--epochs", "1",
        "--num-samples", "100",
        "--enable-fairness-sampling",
        "--fairness-sample-size", "20"
    ]
    
    print(f"Command: {' '.join(cmd1)}")
    result = subprocess.run(cmd1, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Training completed successfully")
        # Extract and display fairness sampling output
        for line in result.stdout.split('\n'):
            if 'fairness' in line.lower() or 'coverage' in line.lower():
                print(f"  {line.strip()}")
    else:
        print(f"✗ Training failed: {result.stderr}")
        return
    
    # Example 2: Custom cohorts and storage
    print("\n[Example 2] Custom cohorts and storage location")
    print("-" * 70)
    
    cmd2 = [
        sys.executable,
        "training/train_any_model.py",
        "--model-type", "anomaly",
        "--epochs", "1",
        "--num-samples", "150",
        "--enable-fairness-sampling",
        "--fairness-cohorts", "train,validation",
        "--fairness-sample-size", "30",
        "--fairness-storage-dir", "./example_fairness_samples"
    ]
    
    print(f"Command: {' '.join(cmd2)}")
    result = subprocess.run(cmd2, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Training completed successfully")
        for line in result.stdout.split('\n'):
            if 'fairness' in line.lower() or 'coverage' in line.lower():
                print(f"  {line.strip()}")
    else:
        print(f"✗ Training failed: {result.stderr}")
        return
    
    # Example 3: Full integration (audit + drift + fairness)
    print("\n[Example 3] Complete integration (Audit + Drift + Fairness)")
    print("-" * 70)
    
    cmd3 = [
        sys.executable,
        "training/train_any_model.py",
        "--model-type", "logistic",
        "--epochs", "1",
        "--num-samples", "100",
        "--enable-audit",
        "--enable-drift-tracking",
        "--enable-fairness-sampling",
        "--fairness-sample-size", "25"
    ]
    
    print(f"Command: {' '.join(cmd3)}")
    result = subprocess.run(cmd3, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Training completed successfully")
        print("\nKey outputs:")
        for line in result.stdout.split('\n'):
            if any(keyword in line.lower() for keyword in ['fairness', 'audit', 'drift', 'merkle']):
                print(f"  {line.strip()}")
    else:
        print(f"✗ Training failed: {result.stderr}")
        return
    
    # Display generated files
    print("\n" + "=" * 70)
    print("Generated Files")
    print("=" * 70)
    
    dirs = [
        "fairness_samples",
        "training_audit_logs",
        "training_drift_reports",
        "example_fairness_samples"
    ]
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            files = list(dir_path.glob("*.json"))
            if files:
                print(f"\n{dir_name}/")
                for file in files[:3]:  # Show first 3 files
                    size = file.stat().st_size / 1024  # KB
                    print(f"  - {file.name} ({size:.1f} KB)")
    
    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Review fairness samples in: fairness_samples/")
    print("  2. Analyze coverage statistics for bias detection")
    print("  3. Integrate with fairness evaluation metrics")
    print("  4. Use samples for model auditing and compliance")
    print()

if __name__ == "__main__":
    run_training_with_fairness_sampling()
