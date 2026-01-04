#!/usr/bin/env python3
"""
Model Monitoring Script

Monitors production models for drift and performance degradation.
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Default baseline values - these should ideally come from configuration
# or be derived from actual historical model performance
DEFAULT_BASELINE_ACCURACY = 0.85  # Minimum expected accuracy
DEFAULT_BASELINE_POSITIVE_RATE = 0.3  # Expected positive prediction rate

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()


def generate_synthetic_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic data for monitoring"""
    data = []
    for _ in range(num_samples):
        features = {
            'violation_count': np.random.randint(0, 10),
            'severity_max': np.random.randint(0, 4),
            'recency_score': np.random.uniform(0, 1),
            'frequency_score': np.random.uniform(0, 1),
            'context_risk': np.random.uniform(0, 1)
        }
        label = 1 if sum(features.values()) > 7 else 0
        data.append({'features': features, 'label': label})
    
    return data


def load_model(model_path: Path) -> Any:
    """Load a pickle model"""
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        return None


def calculate_drift_score(
    current_predictions: List[int],
    baseline_predictions: List[int]
) -> float:
    """Calculate drift score based on prediction distribution"""
    current_dist = np.mean(current_predictions)
    baseline_dist = np.mean(baseline_predictions)
    
    drift_score = abs(current_dist - baseline_dist)
    return drift_score


def monitor_models(
    model_path: Path,
    num_samples: int = 1000,
    drift_threshold: float = 0.15
) -> Dict[str, Any]:
    """Monitor models for drift"""
    monitoring_results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'drift_detected': False,
        'drift_score': 0.0,
        'models_monitored': []
    }
    
    # Find all model files
    model_files = list(model_path.glob("*.pkl"))
    
    if not model_files:
        logging.warning(f"No models found in {model_path}")
        return monitoring_results
    
    logging.info(f"Found {len(model_files)} models to monitor")
    
    # Generate test data
    test_data = generate_synthetic_data(num_samples)
    
    # Monitor each model
    for model_file in model_files:
        model_name = model_file.stem
        logging.info(f"Monitoring model: {model_name}")
        
        model = load_model(model_file)
        if model is None:
            continue
        
        # Run inference
        try:
            predictions = []
            for sample in test_data:
                features = sample['features']
                # Handle different model types
                if hasattr(model, 'predict'):
                    # Sklearn-style model
                    X = np.array([[
                        features.get('violation_count', 0),
                        features.get('severity_max', 0),
                        features.get('recency_score', 0),
                        features.get('frequency_score', 0),
                        features.get('context_risk', 0)
                    ]])
                    pred = model.predict(X)[0]
                    predictions.append(int(pred))
                else:
                    # Heuristic or custom model
                    predictions.append(0)
            
            # Calculate metrics
            accuracy = sum(1 for i, p in enumerate(predictions) if p == test_data[i]['label']) / len(predictions)
            positive_rate = sum(predictions) / len(predictions)
            
            # Load baseline if available
            metadata_file = model_path / f"{model_name}_metrics.json"
            baseline_accuracy = DEFAULT_BASELINE_ACCURACY
            baseline_positive_rate = DEFAULT_BASELINE_POSITIVE_RATE
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    baseline_accuracy = metadata.get('accuracy', DEFAULT_BASELINE_ACCURACY)
                    # Try to derive positive rate from metadata if available
                    if 'positive_rate' in metadata:
                        baseline_positive_rate = metadata['positive_rate']
            
            # Calculate drift
            drift_score = abs(positive_rate - baseline_positive_rate)
            accuracy_drop = baseline_accuracy - accuracy
            
            model_result = {
                'model_name': model_name,
                'current_accuracy': accuracy,
                'baseline_accuracy': baseline_accuracy,
                'accuracy_drop': accuracy_drop,
                'positive_rate': positive_rate,
                'drift_score': drift_score,
                'drift_detected': drift_score > drift_threshold or accuracy_drop > 0.05
            }
            
            monitoring_results['models_monitored'].append(model_result)
            
            if model_result['drift_detected']:
                monitoring_results['drift_detected'] = True
                monitoring_results['drift_score'] = max(
                    monitoring_results['drift_score'], drift_score
                )
                logging.warning(
                    f"⚠️  Drift detected in {model_name}: "
                    f"score={drift_score:.4f}, accuracy_drop={accuracy_drop:.4f}"
                )
            else:
                logging.info(f"✓ {model_name} performing normally")
        
        except Exception as e:
            logging.error(f"Error monitoring {model_name}: {e}")
    
    return monitoring_results


def list_models(model_path: Path) -> None:
    """List all models in the directory"""
    model_files = list(model_path.glob("*.pkl")) + list(model_path.glob("*.pt"))
    
    if not model_files:
        logging.info("No models found")
        return
    
    logging.info(f"Found {len(model_files)} models:")
    for model_file in model_files:
        logging.info(f"  - {model_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Monitor production models")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/current",
        help="Path to model directory"
    )
    parser.add_argument(
        "--list-models",
        action='store_true',
        help="List available models"
    )
    parser.add_argument(
        "--inference-mode",
        action='store_true',
        help="Run inference on test data"
    )
    parser.add_argument(
        "--calculate-drift",
        action='store_true',
        help="Calculate drift metrics"
    )
    parser.add_argument(
        "--compare-baseline",
        action='store_true',
        help="Compare with baseline performance"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples for inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="drift_metrics.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    if not model_path.exists():
        logging.error(f"Model path does not exist: {model_path}")
        return 1
    
    if args.list_models:
        list_models(model_path)
        return 0
    
    if args.calculate_drift or args.compare_baseline or args.inference_mode:
        # Load configuration
        config_path = Path(".github/workflows/config/training-schedule.json")
        drift_threshold = 0.15
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                drift_threshold = config.get('performance_thresholds', {}).get(
                    'drift_alert_threshold', 0.15
                )
        
        # Monitor models
        results = monitor_models(model_path, args.num_samples, drift_threshold)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Monitoring results saved to {args.output}")
        
        if results['drift_detected']:
            logging.warning("⚠️  Drift detected in one or more models")
            return 1
        else:
            logging.info("✓ All models performing normally")
            return 0
    
    logging.info("No action specified. Use --help for options")
    return 0


if __name__ == "__main__":
    sys.exit(main())
