#!/usr/bin/env python3
"""
Model Performance Comparison Script

Compares newly trained models with baseline models to detect performance regression.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics from a JSON file"""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load metrics from {metrics_path}: {e}")
        return {}


def find_model_metrics(directory: Path) -> Dict[str, Dict[str, Any]]:
    """Find all model metrics files in a directory"""
    metrics_files = {}
    
    if not directory.exists():
        logging.warning(f"Directory does not exist: {directory}")
        return metrics_files
    
    # Look for metrics files
    for metrics_file in directory.glob("**/*_metrics.json"):
        # Extract model type from filename
        filename = metrics_file.stem
        model_type = filename.replace("_metrics", "")
        
        metrics = load_metrics(metrics_file)
        if metrics:
            metrics_files[model_type] = metrics
            logging.info(f"Loaded metrics for {model_type} from {metrics_file.name}")
    
    return metrics_files


def compare_metrics(
    current_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    threshold: float = 0.02
) -> Dict[str, Any]:
    """Compare two sets of metrics and detect regression"""
    comparison = {
        'has_regression': False,
        'improvements': [],
        'regressions': [],
        'unchanged': []
    }
    
    # Key metrics to compare
    important_metrics = ['accuracy', 'f1', 'precision', 'recall', 'ece']
    
    for metric_name in important_metrics:
        current_value = current_metrics.get(metric_name)
        baseline_value = baseline_metrics.get(metric_name)
        
        if current_value is None or baseline_value is None:
            continue
        
        # For ECE, lower is better
        if metric_name == 'ece':
            diff = baseline_value - current_value  # Inverted
            is_better = current_value < baseline_value
        else:
            diff = current_value - baseline_value
            is_better = current_value > baseline_value
        
        diff_pct = (diff / baseline_value * 100) if baseline_value != 0 else 0
        
        metric_comparison = {
            'metric': metric_name,
            'current': current_value,
            'baseline': baseline_value,
            'diff': diff,
            'diff_pct': diff_pct
        }
        
        # Check for significant change
        if abs(diff) > threshold:
            if is_better:
                comparison['improvements'].append(metric_comparison)
            else:
                comparison['regressions'].append(metric_comparison)
                comparison['has_regression'] = True
        else:
            comparison['unchanged'].append(metric_comparison)
    
    return comparison


def update_performance_history(
    history_path: Path,
    model_type: str,
    metrics: Dict[str, Any],
    timestamp: str
) -> None:
    """Update performance history with new metrics"""
    history_path.mkdir(parents=True, exist_ok=True)
    
    history_file = history_path / f"{model_type}_history.json"
    
    # Load existing history
    history = []
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load history: {e}")
    
    # Add new entry
    history.append({
        'timestamp': timestamp,
        'metrics': metrics
    })
    
    # Keep last 100 entries
    history = history[-100:]
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    logging.info(f"Updated performance history for {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Compare model performance")
    parser.add_argument(
        "--current-path",
        type=str,
        required=True,
        help="Path to directory with current model metrics"
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        required=True,
        help="Path to directory with baseline model metrics"
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default="performance_history",
        help="Path to store performance history"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="performance_comparison.json",
        help="Output file for comparison results"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Threshold for significant change (default: 0.02)"
    )
    
    args = parser.parse_args()
    
    current_path = Path(args.current_path)
    baseline_path = Path(args.baseline_path)
    history_path = Path(args.history_path)
    
    # Load metrics
    logging.info("Loading current model metrics...")
    current_metrics_all = find_model_metrics(current_path)
    
    logging.info("Loading baseline model metrics...")
    baseline_metrics_all = find_model_metrics(baseline_path)
    
    if not current_metrics_all:
        logging.warning("No current model metrics found")
        # Create empty comparison result
        result = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'has_regression': False,
            'message': 'No current model metrics found',
            'comparisons': []
        }
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        return 0
    
    # Compare each model type
    comparisons = []
    overall_regression = False
    timestamp = datetime.now(timezone.utc).isoformat()
    
    for model_type, current_metrics in current_metrics_all.items():
        logging.info(f"\nComparing {model_type}...")
        
        # Update history
        update_performance_history(
            history_path, model_type, current_metrics, timestamp
        )
        
        # Compare with baseline if available
        if model_type in baseline_metrics_all:
            baseline_metrics = baseline_metrics_all[model_type]
            comparison = compare_metrics(
                current_metrics, baseline_metrics, args.threshold
            )
            
            comparison['model_type'] = model_type
            comparisons.append(comparison)
            
            if comparison['has_regression']:
                overall_regression = True
                logging.warning(f"⚠️  Regression detected in {model_type}")
                for reg in comparison['regressions']:
                    logging.warning(
                        f"  {reg['metric']}: {reg['current']:.4f} vs "
                        f"{reg['baseline']:.4f} (diff: {reg['diff']:.4f})"
                    )
            else:
                logging.info(f"✓ No regression in {model_type}")
                if comparison['improvements']:
                    logging.info(f"  {len(comparison['improvements'])} improvements found")
        else:
            logging.info(f"No baseline found for {model_type} (new model)")
            comparisons.append({
                'model_type': model_type,
                'has_regression': False,
                'message': 'No baseline for comparison'
            })
    
    # Create final result
    result = {
        'timestamp': timestamp,
        'has_regression': overall_regression,
        'models_compared': len(comparisons),
        'comparisons': comparisons,
        'regressions': [
            {
                'model_type': c['model_type'],
                'regressions': c.get('regressions', [])
            }
            for c in comparisons if c.get('has_regression', False)
        ]
    }
    
    # Save result
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    logging.info(f"\nComparison results saved to {args.output}")
    
    if overall_regression:
        logging.warning("⚠️  Overall regression detected!")
        return 1
    else:
        logging.info("✓ No regressions detected")
        return 0


if __name__ == "__main__":
    sys.exit(main())
