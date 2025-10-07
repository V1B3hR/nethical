#!/usr/bin/env python3
"""
Example: Using Phase567 Integrated Governance in Model Training

This example demonstrates how to use the Phase567 integrated governance feature
in train_any_model.py to enable ML shadow mode, blended risk analysis, and
anomaly detection during model training.

The Phase567 integration provides:
- Phase 5: ML Shadow Mode for passive ML model validation
- Phase 6: ML Blended Risk for gray-zone decision assistance
- Phase 7: Anomaly & Drift Detection for behavioral monitoring

Usage:
    python examples/train_with_phase567.py
"""

import subprocess
import sys
from pathlib import Path


def run_training(model_type, num_samples, enable_components, description):
    """Run training with specified Phase567 configuration."""
    print("\n" + "=" * 80)
    print(f"  {description}")
    print("=" * 80)
    
    train_script = Path(__file__).parent.parent / "training" / "train_any_model.py"
    phase567_dir = "example_phase567_data"
    
    cmd = [
        sys.executable,
        str(train_script),
        "--model-type", model_type,
        "--epochs", "1",
        "--num-samples", str(num_samples),
        "--seed", "42",
        "--enable-phase567",
        "--phase567-storage-dir", phase567_dir
    ]
    
    # Add component-specific flags
    for component in enable_components:
        cmd.append(f"--enable-{component}")
    
    print(f"\nRunning: {' '.join(cmd[1:])}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error during training: {result.stderr}")
        return False
    
    print(result.stdout)
    return True


def display_phase567_reports(phase567_dir):
    """Display generated Phase567 reports."""
    print("\n" + "=" * 80)
    print("  Phase567 Reports")
    print("=" * 80)
    
    phase567_path = Path(phase567_dir)
    if not phase567_path.exists():
        print(f"\nNo Phase567 data found in {phase567_dir}")
        return
    
    # Find and display reports
    report_files = list(phase567_path.glob("phase567_report_*.md"))
    if not report_files:
        print(f"\nNo Phase567 reports found in {phase567_dir}")
        return
    
    print(f"\nFound {len(report_files)} Phase567 report(s):\n")
    
    for report_file in sorted(report_files):
        print(f"\nReport: {report_file.name}")
        print("-" * 80)
        with open(report_file) as f:
            content = f.read()
            # Display key sections
            lines = content.split('\n')
            in_relevant_section = False
            for line in lines:
                if line.startswith('## '):
                    in_relevant_section = True
                if in_relevant_section and line.strip():
                    print(line)
                if line.startswith('##') and not line.startswith('## System'):
                    in_relevant_section = False
        print()


def main():
    print("\n" + "=" * 80)
    print("  Phase567 Integrated Governance Training Example")
    print("=" * 80)
    print("\nThis example demonstrates training models with Phase567 integrated governance,")
    print("which includes ML shadow mode, blended risk, and anomaly detection.")
    
    # Example 1: Train with all Phase567 components enabled (default)
    success = run_training(
        model_type="heuristic",
        num_samples=200,
        enable_components=[],  # Empty means all components enabled by default
        description="Training with Full Phase567 Integration (Shadow + Blended + Anomaly)"
    )
    
    if not success:
        return
    
    # Example 2: Train with selective components (shadow mode and anomaly detection)
    success = run_training(
        model_type="logistic",
        num_samples=200,
        enable_components=["shadow-mode", "anomaly-detection"],
        description="Training with Selective Phase567 Components (Shadow + Anomaly)"
    )
    
    if not success:
        return
    
    # Display all generated reports
    display_phase567_reports("example_phase567_data")
    
    print("\n" + "=" * 80)
    print("  Example Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  • Phase567 integration processes validation samples through ML governance")
    print("  • Shadow mode validates ML predictions against rule-based decisions")
    print("  • Blended risk provides ML assistance in gray-zone decisions")
    print("  • Anomaly detection monitors for unusual patterns during training")
    print("  • Comprehensive reports are generated for analysis")
    print("\nPhase567 data saved to: example_phase567_data/")
    print("Review the markdown reports for detailed metrics and analysis.")
    print()


if __name__ == "__main__":
    main()
