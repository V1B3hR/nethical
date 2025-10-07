#!/usr/bin/env python3
"""Example: Quarantine Integration in Model Training

This script demonstrates the quarantine functionality integrated into the
train_any_model.py training pipeline.

The quarantine system provides automated risk management for model training:
- Automatically quarantines models that fail promotion gate criteria
- Tracks quarantine status and activation times
- Integrates with audit logging for complete traceability

Usage:
    python examples/demo_train_with_quarantine.py
"""

import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def run_training_command(description, cmd):
    """Run a training command and display results."""
    print(f"\n{description}")
    print(f"\nCommand: {' '.join(cmd)}")
    print("-" * 70)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print relevant output lines
    for line in result.stdout.split('\n'):
        if any(keyword in line for keyword in [
            'Quarantine', 'quarantine', 'Promotion', 'Is Quarantined',
            'Cohort quarantined', 'Activation time', 'Statistics'
        ]):
            print(line)
    
    return result.returncode == 0


def main():
    """Run the quarantine demo."""
    print("\n" + "=" * 70)
    print("  QUARANTINE INTEGRATION DEMO")
    print("  Automated Model Risk Management")
    print("=" * 70)
    
    training_script = Path(__file__).parent.parent / "training" / "train_any_model.py"
    
    print_section("Overview")
    
    print("""
The quarantine system provides rapid incident response for model training:

1. Models are registered as cohorts in the quarantine system
2. Failed models can be automatically quarantined
3. Quarantine status is tracked and reported
4. Integration with audit logging ensures traceability

Promotion Gate Criteria:
- ECE (Expected Calibration Error) must be ≤ 0.08
- Accuracy must be ≥ 0.85

If a model fails either criterion, it can be quarantined.
""")
    
    print_section("Example 1: Basic Quarantine System")
    
    print("""
Enable the quarantine system without auto-quarantine.
The system will initialize and track the training cohort,
but won't automatically quarantine models that fail.
""")
    
    cmd1 = [
        sys.executable,
        str(training_script),
        "--model-type", "heuristic",
        "--epochs", "2",
        "--num-samples", "50",
        "--seed", "42",
        "--enable-quarantine"
    ]
    
    run_training_command("Running training with quarantine enabled...", cmd1)
    
    print_section("Example 2: Auto-Quarantine on Failure")
    
    print("""
Enable auto-quarantine for models that fail the promotion gate.
If the model fails, it will be automatically quarantined for 48 hours.
""")
    
    cmd2 = [
        sys.executable,
        str(training_script),
        "--model-type", "logistic",
        "--epochs", "2",
        "--num-samples", "100",
        "--seed", "42",
        "--enable-quarantine",
        "--quarantine-on-failure"
    ]
    
    run_training_command("Running training with auto-quarantine...", cmd2)
    
    print_section("Example 3: Quarantine with Audit Logging")
    
    print("""
Combine quarantine with Merkle audit logging for complete governance.
All quarantine events are recorded in the immutable audit trail.
""")
    
    cmd3 = [
        sys.executable,
        str(training_script),
        "--model-type", "anomaly",
        "--epochs", "2",
        "--num-samples", "200",
        "--seed", "42",
        "--enable-quarantine",
        "--quarantine-on-failure",
        "--enable-audit",
        "--audit-path", "/tmp/quarantine_demo_audit"
    ]
    
    run_training_command("Running training with quarantine and audit...", cmd3)
    
    print_section("Key Features")
    
    print("""
✓ Fast Activation: Quarantine activates in milliseconds (target: <15s)
✓ Risk-Based Categorization: Different reasons based on failure type
✓ Configurable Duration: Default 48 hours for failed models
✓ Audit Integration: Quarantine events logged to audit trail
✓ Statistics Tracking: Monitor quarantine system performance

Quarantine Reasons:
- high_risk_score: Both ECE and accuracy thresholds violated
- policy_violation: Single metric threshold violated
""")
    
    print_section("Next Steps")
    
    print("""
1. Review the documentation:
   $ cat QUARANTINE_TRAINING_INTEGRATION.md

2. Run tests:
   $ python tests/test_train_quarantine.py

3. Try different model types and parameters to see quarantine in action

4. Integrate with your deployment pipeline for automated quality gates

5. Monitor quarantine statistics to identify systematic issues
""")
    
    print("\n" + "=" * 70)
    print("  ✅ QUARANTINE DEMO COMPLETE")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
