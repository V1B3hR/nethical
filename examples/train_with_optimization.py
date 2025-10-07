#!/usr/bin/env python3
"""
Example: Training Models with Continuous Optimization

This example demonstrates how to use the continuous optimization feature
in train_any_model.py to systematically improve model performance.

Workflow:
1. Create a baseline configuration
2. Train candidate models with different parameters
3. Use the promotion gate to compare candidates against the baseline
4. Promote the best performing model to production

Run this script from the repository root:
    python examples/train_with_optimization.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

def run_training(model_type, epochs, num_samples, seed, opt_db, baseline_id=None):
    """Run training with optimization enabled."""
    cmd = [
        sys.executable,
        "training/train_any_model.py",
        "--model-type", model_type,
        "--epochs", str(epochs),
        "--num-samples", str(num_samples),
        "--seed", str(seed),
        "--enable-optimization",
        "--optimization-db", str(opt_db)
    ]
    
    if baseline_id:
        cmd.extend(["--baseline-config-id", baseline_id])
    
    print("\n" + "=" * 70)
    print(f"Running: {' '.join(cmd)}")
    print("=" * 70)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract configuration ID from output
    config_id = None
    for line in result.stdout.split('\n'):
        if "Recorded metrics for configuration:" in line:
            config_id = line.split(":")[-1].strip()
            break
    
    # Check if promoted
    promoted = "Configuration" in result.stdout and "promoted to production" in result.stdout
    
    return {
        'config_id': config_id,
        'promoted': promoted,
        'output': result.stdout,
        'returncode': result.returncode
    }


def main():
    """Run the optimization workflow example."""
    print("\n" + "=" * 70)
    print("  Continuous Optimization Workflow Example")
    print("=" * 70)
    print("\nThis example demonstrates:")
    print("  1. Creating a baseline configuration")
    print("  2. Training candidate models with different parameters")
    print("  3. Using the promotion gate to select the best model")
    print("=" * 70)
    
    # Create a temporary directory for the optimization database
    with tempfile.TemporaryDirectory() as tmpdir:
        opt_db = Path(tmpdir) / "optimization.db"
        
        # Step 1: Create baseline configuration
        print("\n[Step 1] Creating baseline configuration...")
        baseline_result = run_training(
            model_type="logistic",
            epochs=10,
            num_samples=1000,
            seed=42,
            opt_db=opt_db
        )
        
        if baseline_result['returncode'] != 0:
            print("\n✗ Baseline training failed!")
            return 1
        
        baseline_id = baseline_result['config_id']
        print(f"\n✓ Baseline created: {baseline_id}")
        
        # Step 2: Train candidate models with different parameters
        candidates = []
        
        print("\n[Step 2] Training candidate models...")
        
        # Candidate 1: More epochs
        print("\n  Candidate 1: More training epochs (20)")
        candidate1 = run_training(
            model_type="logistic",
            epochs=20,
            num_samples=1000,
            seed=42,
            opt_db=opt_db,
            baseline_id=baseline_id
        )
        candidates.append(("More epochs (20)", candidate1))
        
        # Candidate 2: More data
        print("\n  Candidate 2: More training data (2000 samples)")
        candidate2 = run_training(
            model_type="logistic",
            epochs=10,
            num_samples=2000,
            seed=42,
            opt_db=opt_db,
            baseline_id=baseline_id
        )
        candidates.append(("More data (2000)", candidate2))
        
        # Candidate 3: Different seed (simulates different initialization)
        print("\n  Candidate 3: Different initialization (seed=43)")
        candidate3 = run_training(
            model_type="logistic",
            epochs=10,
            num_samples=1000,
            seed=43,
            opt_db=opt_db,
            baseline_id=baseline_id
        )
        candidates.append(("Different seed (43)", candidate3))
        
        # Step 3: Summarize results
        print("\n" + "=" * 70)
        print("  Results Summary")
        print("=" * 70)
        
        print(f"\nBaseline: {baseline_id}")
        
        print("\nCandidates:")
        promoted_count = 0
        for name, result in candidates:
            status = "PROMOTED ✓" if result['promoted'] else "REJECTED ✗"
            print(f"  - {name} ({result['config_id']}): {status}")
            if result['promoted']:
                promoted_count += 1
        
        print(f"\nTotal candidates evaluated: {len(candidates)}")
        print(f"Candidates promoted: {promoted_count}")
        print(f"Candidates rejected: {len(candidates) - promoted_count}")
        
        # Step 4: Show promotion gate criteria
        print("\n" + "=" * 70)
        print("  Promotion Gate Criteria")
        print("=" * 70)
        print("\nFor a candidate to be promoted, it must meet ALL of:")
        print("  • Recall gain ≥ +3% (over baseline)")
        print("  • FP rate increase ≤ +2% (over baseline)")
        print("  • Latency increase ≤ +5ms (over baseline)")
        print("  • Human agreement ≥ 85%")
        print("  • Sample size ≥ 100 cases")
        
        print("\n" + "=" * 70)
        print("  Example Complete!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  • The optimization framework systematically compares models")
        print("  • Only models that meet strict criteria are promoted")
        print("  • All configurations and metrics are tracked in the database")
        print("  • This enables data-driven model selection and A/B testing")
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
