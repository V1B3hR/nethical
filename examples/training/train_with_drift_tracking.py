#!/usr/bin/env python3
"""
Example: Using Ethical Drift Tracking in Model Training

This example demonstrates how to use the ethical drift tracking feature
in train_any_model.py to monitor model performance across different training
cohorts and detect drift in model behavior.

Usage:
    python examples/train_with_drift_tracking.py
"""

import subprocess
import sys
import json
from pathlib import Path


def main():
    print("=" * 80)
    print("  Ethical Drift Tracking Example")
    print("=" * 80)

    # Define training script path
    train_script = Path(__file__).parent.parent / "training" / "train_any_model.py"
    drift_report_dir = "example_drift_reports"

    print("\n1. Training Cohort A (heuristic model, seed=42)...")
    print("-" * 80)

    cmd_a = [
        sys.executable,
        str(train_script),
        "--model-type",
        "heuristic",
        "--epochs",
        "5",
        "--num-samples",
        "500",
        "--seed",
        "42",
        "--enable-drift-tracking",
        "--drift-report-dir",
        drift_report_dir,
        "--cohort-id",
        "cohort_a_heuristic",
    ]

    result_a = subprocess.run(cmd_a, capture_output=True, text=True)
    if result_a.returncode != 0:
        print(f"Error training cohort A: {result_a.stderr}")
        return

    print(result_a.stdout)

    print("\n2. Training Cohort B (logistic model, seed=999)...")
    print("-" * 80)

    cmd_b = [
        sys.executable,
        str(train_script),
        "--model-type",
        "logistic",
        "--epochs",
        "5",
        "--num-samples",
        "500",
        "--seed",
        "999",
        "--enable-drift-tracking",
        "--drift-report-dir",
        drift_report_dir,
        "--cohort-id",
        "cohort_b_logistic",
    ]

    result_b = subprocess.run(cmd_b, capture_output=True, text=True)
    if result_b.returncode != 0:
        print(f"Error training cohort B: {result_b.stderr}")
        return

    print(result_b.stdout)

    print("\n3. Analyzing Drift Reports...")
    print("-" * 80)

    # Read and display drift reports
    drift_path = Path(drift_report_dir)
    if drift_path.exists():
        drift_files = list(drift_path.glob("drift_*.json"))
        print(f"\nFound {len(drift_files)} drift report(s):\n")

        for drift_file in sorted(drift_files):
            with open(drift_file) as f:
                report = json.load(f)

            print(f"Report: {drift_file.name}")
            print(f"  Generated: {report['generated_at']}")
            print(
                f"  Drift Detected: {report['drift_metrics'].get('has_drift', False)}"
            )
            print(f"  Cohorts:")

            for cohort_id, cohort_data in report["cohorts"].items():
                print(f"    - {cohort_id}:")
                print(f"        Actions: {cohort_data['action_count']}")
                print(
                    f"        Violations: {cohort_data['violation_stats']['total_count']}"
                )
                print(f"        Avg Risk: {cohort_data['avg_risk_score']:.4f}")

            if report["recommendations"]:
                print(f"  Recommendations:")
                for rec in report["recommendations"][:3]:
                    print(f"    - {rec}")
            print()
    else:
        print(f"No drift reports found in {drift_report_dir}")

    print("=" * 80)
    print("  Example Complete!")
    print("=" * 80)
    print(f"\nDrift reports saved to: {drift_report_dir}/")
    print("You can analyze these reports to understand model behavior across cohorts.")


if __name__ == "__main__":
    main()
