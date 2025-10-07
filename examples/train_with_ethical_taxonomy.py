#!/usr/bin/env python3
"""
Example: Using Ethical Taxonomy in Model Training

This example demonstrates how to use the ethical taxonomy feature
in train_any_model.py to tag violations with ethical dimensions during
model training and evaluation.

Usage:
    python examples/train_with_ethical_taxonomy.py
"""

import subprocess
import sys
import json
from pathlib import Path

def main():
    print("=" * 80)
    print("  Ethical Taxonomy Training Example")
    print("=" * 80)
    
    # Define training script path
    train_script = Path(__file__).parent.parent / "training" / "train_any_model.py"
    
    print("\n1. Training model with ethical taxonomy enabled...")
    print("-" * 80)
    
    cmd = [
        sys.executable,
        str(train_script),
        "--model-type", "heuristic",
        "--epochs", "5",
        "--num-samples", "200",
        "--seed", "42",
        "--enable-ethical-taxonomy",
        "--taxonomy-path", "ethics_taxonomy.json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error during training: {result.stderr}")
        return
    
    print(result.stdout)
    
    print("\n2. Examining ethical taxonomy report...")
    print("-" * 80)
    
    # Find the most recent taxonomy report
    candidates_dir = Path("models/candidates")
    if candidates_dir.exists():
        taxonomy_files = list(candidates_dir.glob("heuristic_taxonomy_*.json"))
        
        if taxonomy_files:
            # Get the most recent file
            latest_file = max(taxonomy_files, key=lambda p: p.stat().st_mtime)
            
            print(f"\nReading taxonomy report: {latest_file}")
            
            with open(latest_file, 'r') as f:
                report = json.load(f)
            
            print(f"\nModel Type: {report['model_type']}")
            print(f"Timestamp: {report['timestamp']}")
            print(f"Promoted: {report['promoted']}")
            
            if report.get('violation_tags'):
                print(f"\nViolations Detected: {len(report['violation_tags'])}")
                
                for i, tag in enumerate(report['violation_tags'], 1):
                    print(f"\n  Violation {i}: {tag['violation_type']}")
                    print(f"    Severity: {tag['severity']}")
                    print(f"    Primary Dimension: {tag['primary_dimension']}")
                    print(f"    Dimension Scores:")
                    for dim, score in tag['dimensions'].items():
                        print(f"      - {dim}: {score:.2f}")
            else:
                print("\nNo violations detected - model passed promotion gate!")
            
            print(f"\nCoverage Statistics:")
            coverage = report['coverage_stats']
            print(f"  Total Violation Types Seen: {coverage['total_violation_types']}")
            print(f"  Tagged Types: {coverage['tagged_types']}")
            print(f"  Coverage: {coverage['coverage_percentage']:.1f}%")
            print(f"  Target: {coverage['target_percentage']:.1f}%")
            print(f"  Meets Target: {coverage['meets_target']}")
            
            print(f"\nEthical Dimensions:")
            for dim_name, dim_info in report['dimensions'].items():
                print(f"  {dim_name}:")
                print(f"    Description: {dim_info['description']}")
                print(f"    Weight: {dim_info['weight']}")
                print(f"    Severity Multiplier: {dim_info['severity_multiplier']}")
        else:
            print("No taxonomy reports found.")
    else:
        print("Candidates directory not found.")
    
    print("\n3. Training a model that passes promotion gate...")
    print("-" * 80)
    
    cmd_pass = [
        sys.executable,
        str(train_script),
        "--model-type", "heuristic",
        "--epochs", "5",
        "--num-samples", "50",
        "--seed", "123",
        "--enable-ethical-taxonomy"
    ]
    
    result_pass = subprocess.run(cmd_pass, capture_output=True, text=True)
    
    if result_pass.returncode != 0:
        print(f"Error during training: {result_pass.stderr}")
        return
    
    # Extract key lines from output
    for line in result_pass.stdout.split('\n'):
        if 'Promotion result' in line or 'Model saved' in line or 'Ethical Taxonomy' in line:
            print(line)
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("- Use --enable-ethical-taxonomy to tag violations with ethical dimensions")
    print("- Violations are tagged with privacy, manipulation, fairness, and safety scores")
    print("- Reports include primary dimension and coverage statistics")
    print("- Taxonomy reports saved alongside model metrics in JSON format")
    print("- Coverage target is 90% by default (configurable in ethics_taxonomy.json)")

if __name__ == "__main__":
    main()
