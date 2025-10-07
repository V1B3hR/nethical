#!/usr/bin/env python3
"""
Example: Using Phase3 Integrated Governance in Model Training

This example demonstrates how to use the new Phase3 integration feature
in train_any_model.py to leverage all Phase 3 governance components during
model training:

- Risk Engine: Multi-factor risk scoring and tier management
- Correlation Engine: Multi-agent pattern detection
- Fairness Sampler: Stratified sampling across cohorts
- Ethical Drift Reporter: Cohort-based drift analysis
- Performance Optimizer: Risk-based detector gating

Usage:
    python examples/train_with_phase3.py
"""

import subprocess
import sys
import json
from pathlib import Path


def main():
    print("=" * 80)
    print("  Phase3 Integrated Governance in Model Training")
    print("=" * 80)
    
    # Define training script path
    train_script = Path(__file__).parent.parent / "training" / "train_any_model.py"
    phase3_storage_dir = "example_phase3_training"
    
    print("\n1. Training with Phase3 Integrated Governance...")
    print("-" * 80)
    print("\nThis training run will use all Phase 3 components:")
    print("  ✓ Risk Engine - Track model risk scores")
    print("  ✓ Correlation Engine - Detect training patterns")
    print("  ✓ Fairness Sampler - Ensure representative sampling")
    print("  ✓ Ethical Drift Reporter - Monitor drift across cohorts")
    print("  ✓ Performance Optimizer - Optimize detector invocation")
    print()
    
    cmd = [
        sys.executable,
        str(train_script),
        "--model-type", "logistic",
        "--epochs", "10",
        "--num-samples", "1000",
        "--seed", "42",
        "--enable-phase3",  # Enable Phase3 integration
        "--phase3-storage-dir", phase3_storage_dir,
        "--cohort-id", "production_v1"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: Training failed with return code {result.returncode}")
        print(result.stderr)
        return
    
    print(result.stdout)
    
    print("\n2. Analyzing Phase3 Results...")
    print("-" * 80)
    
    # Read and display drift reports
    phase3_path = Path(phase3_storage_dir)
    drift_reports_dir = phase3_path / "drift_reports"
    
    if drift_reports_dir.exists():
        drift_files = list(drift_reports_dir.glob("drift_*.json"))
        if drift_files:
            print(f"\n✓ Found {len(drift_files)} drift report(s)\n")
            
            for drift_file in drift_files[:3]:  # Show up to 3 reports
                with open(drift_file) as f:
                    report = json.load(f)
                
                print(f"Report: {drift_file.name}")
                print(f"  Report ID: {report['report_id']}")
                print(f"  Drift detected: {report['drift_metrics'].get('has_drift', False)}")
                print(f"  Cohorts tracked: {len(report['cohorts'])}")
                
                for cohort_id, cohort_data in report['cohorts'].items():
                    print(f"\n  Cohort: {cohort_id}")
                    print(f"    Actions: {cohort_data['action_count']}")
                    print(f"    Violations: {cohort_data['violation_stats']['total_count']}")
                    print(f"    Avg Risk Score: {cohort_data['avg_risk_score']:.4f}")
                
                if report.get('recommendations'):
                    print(f"\n  Recommendations:")
                    for i, rec in enumerate(report['recommendations'][:3], 1):
                        print(f"    {i}. {rec}")
                print()
        else:
            print(f"No drift reports found in {drift_reports_dir}")
    else:
        print(f"Drift reports directory not found: {drift_reports_dir}")
    
    print("\n3. Comparing with Traditional Training...")
    print("-" * 80)
    print("\nTraditional training (without Phase3):")
    
    cmd_traditional = [
        sys.executable,
        str(train_script),
        "--model-type", "logistic",
        "--epochs", "10",
        "--num-samples", "1000",
        "--seed", "42",
        "--enable-drift-tracking",  # Old flag
        "--drift-report-dir", "example_traditional_drift",
        "--cohort-id", "production_v1_traditional"
    ]
    
    print(f"Running: {' '.join(cmd_traditional)}\n")
    result_trad = subprocess.run(cmd_traditional, capture_output=True, text=True)
    
    if result_trad.returncode == 0:
        # Extract key lines
        for line in result_trad.stdout.split('\n'):
            if any(keyword in line for keyword in ['Ethical drift', 'Drift Report', 'Drift detected']):
                print(line)
    
    print("\n" + "=" * 80)
    print("  Comparison Summary")
    print("=" * 80)
    print("\nPhase3 Integration Benefits:")
    print("  ✓ Comprehensive governance with 5 integrated components")
    print("  ✓ Risk-based adaptive detection and optimization")
    print("  ✓ Multi-agent correlation pattern detection")
    print("  ✓ Performance optimization with CPU reduction targets")
    print("  ✓ Enhanced drift analysis with system-wide context")
    print("\nTraditional Drift Tracking:")
    print("  ✓ Lightweight drift monitoring")
    print("  ✓ Simple cohort-based analysis")
    print("  ✓ Backward compatible with existing workflows")
    
    print("\n" + "=" * 80)
    print("  Example Complete!")
    print("=" * 80)
    print(f"\nPhase3 data saved to: {phase3_storage_dir}/")
    print("You can explore the generated reports to understand model behavior.")
    print("\nTo use Phase3 in your training:")
    print("  python training/train_any_model.py --model-type <type> --enable-phase3")


if __name__ == "__main__":
    main()
