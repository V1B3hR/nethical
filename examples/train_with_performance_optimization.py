#!/usr/bin/env python3
"""
Example: Training with Performance Optimization

This example demonstrates how to use the performance optimization tracking
feature in train_any_model.py to monitor CPU usage during training phases.
"""

import subprocess
import sys
import json
from pathlib import Path

def run_training_example():
    """Run a training example with performance optimization."""
    print("=" * 70)
    print("Example: Training with Performance Optimization")
    print("=" * 70)
    
    # Get the path to train_any_model.py
    script_dir = Path(__file__).parent.parent / "training"
    script_path = script_dir / "train_any_model.py"
    
    # Run training with performance optimization enabled
    cmd = [
        sys.executable,
        str(script_path),
        "--model-type", "logistic",
        "--epochs", "5",
        "--num-samples", "500",
        "--seed", "42",
        "--enable-performance-optimization",
        "--performance-target-reduction", "30.0"
    ]
    
    print("\nCommand:")
    print(" ".join(cmd))
    print("\n" + "-" * 70)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"\nError: Training failed with return code {result.returncode}")
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        return False
    
    # Find and display the performance report
    report_dir = Path("training_performance_reports")
    if report_dir.exists():
        report_files = sorted(report_dir.glob("perf_report_logistic_*.json"))
        if report_files:
            latest_report = report_files[-1]
            print("\n" + "=" * 70)
            print("Performance Report Summary")
            print("=" * 70)
            
            with open(latest_report, 'r') as f:
                report = json.load(f)
            
            print(f"\nReport File: {latest_report.name}")
            print(f"Timestamp: {report['timestamp']}")
            
            print("\nTraining Phase Timing:")
            for phase_name, phase_stats in report['detector_stats']['detectors'].items():
                print(f"  {phase_name}:")
                print(f"    Tier: {phase_stats['tier']}")
                print(f"    Total Time: {phase_stats['total_cpu_time_ms']:.2f}ms")
                print(f"    Invocations: {phase_stats['total_invocations']}")
            
            print("\nOptimization Status:")
            opt = report['optimization']
            print(f"  Target CPU Reduction: {opt['target_cpu_reduction_pct']}%")
            print(f"  Current CPU Reduction: {opt['current_cpu_reduction_pct']:.2f}%")
            print(f"  Meeting Target: {opt['meeting_target']}")
            
            return True
    
    return False


def compare_with_without_optimization():
    """Compare training with and without optimization tracking."""
    print("\n" + "=" * 70)
    print("Comparison: With vs Without Optimization Tracking")
    print("=" * 70)
    
    script_dir = Path(__file__).parent.parent / "training"
    script_path = script_dir / "train_any_model.py"
    
    print("\n1. Running WITHOUT performance optimization...")
    cmd_without = [
        sys.executable,
        str(script_path),
        "--model-type", "heuristic",
        "--epochs", "2",
        "--num-samples", "100",
        "--seed", "42"
    ]
    
    result_without = subprocess.run(cmd_without, capture_output=True, text=True)
    
    # Check for performance metrics in output
    has_perf_metrics = "Performance Metrics:" in result_without.stdout
    print(f"   Performance metrics present: {has_perf_metrics}")
    
    print("\n2. Running WITH performance optimization...")
    cmd_with = [
        sys.executable,
        str(script_path),
        "--model-type", "heuristic",
        "--epochs", "2",
        "--num-samples", "100",
        "--seed", "42",
        "--enable-performance-optimization"
    ]
    
    result_with = subprocess.run(cmd_with, capture_output=True, text=True)
    
    # Check for performance metrics in output
    has_perf_metrics = "Performance Metrics:" in result_with.stdout
    print(f"   Performance metrics present: {has_perf_metrics}")
    
    print("\n3. Summary:")
    print("   - Default mode: No performance tracking overhead")
    print("   - Optimized mode: Detailed performance insights")
    print("   - Use --enable-performance-optimization when you need to:")
    print("     * Identify training bottlenecks")
    print("     * Measure optimization impact")
    print("     * Track performance over time")


if __name__ == "__main__":
    # Run the main example
    success = run_training_example()
    
    # Run the comparison
    compare_with_without_optimization()
    
    print("\n" + "=" * 70)
    print("Example completed successfully!" if success else "Example completed with warnings")
    print("=" * 70)
