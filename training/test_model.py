#!/usr/bin/env python3
"""
Plug-and-Play Testing Script for Nethical ML Models

Usage:
    python test_model.py --model-type logistic --test-data-path data/test_data.json --metrics-output-path models/candidates/logistic_test_metrics.json

Steps:
    1. Load trained model from file
    2. Load test data (JSON, CSV, etc.)
    3. Run predictions
    4. Compute and save test metrics

Supported Model Types (see MLModelType):
    - heuristic
    - logistic
    - simple_transformer
    - deep_nn
    - [add more as needed]

"""

import argparse
import os
import json
import random
import numpy as np
from pathlib import Path

from nethical.core.ml_shadow import MLModelType
from nethical.mlops.baseline import BaselineMLClassifier
# from nethical.mlops.deep_nn import DeepNNClassifier   # Example for extensibility
# from nethical.mlops.transformer import TransformerClassifier

def get_model_class(model_type: str):
    registry = {
        "heuristic": BaselineMLClassifier,
        "logistic": BaselineMLClassifier,
        "simple_transformer": BaselineMLClassifier,
        # "deep_nn": DeepNNClassifier,
        # "transformer": TransformerClassifier,
    }
    if model_type not in registry:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: {list(registry.keys())}")
    return registry[model_type]

def load_model(model_type, model_path):
    ModelClass = get_model_class(model_type)
    return ModelClass.load(model_path)

def load_test_data(test_data_path):
    with open(test_data_path, "r") as f:
        if test_data_path.endswith(".json"):
            data = json.load(f)
        else:
            # Add CSV logic if needed
            raise NotImplementedError("Only JSON data supported by default")
    return data

def compute_metrics(preds, labels):
    tp = sum((p == 1 and l == 1) for p, l in zip(preds, labels))
    tn = sum((p == 0 and l == 0) for p, l in zip(preds, labels))
    fp = sum((p == 1 and l == 0) for p, l in zip(preds, labels))
    fn = sum((p == 0 and l == 1) for p, l in zip(preds, labels))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    ece = abs(accuracy - precision)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "ece": ece,
    }

def main():
    parser = argparse.ArgumentParser(description="Plug-and-Play Testing Pipeline for Nethical ML Models")
    parser.add_argument("--model-type", type=str, required=True, help="Model type (see MLModelType)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model file (JSON)")
    parser.add_argument("--test-data-path", type=str, required=True, help="Path to test data (JSON)")
    parser.add_argument("--metrics-output-path", type=str, default=None, help="Path to save test metrics (JSON)")
    args = parser.parse_args()

    # 1. Load Model
    model = load_model(args.model_type, args.model_path)
    print(f"[INFO] Loaded model from {args.model_path}")

    # 2. Load Test Data
    test_data = load_test_data(args.test_data_path)
    print(f"[INFO] Loaded test data: {len(test_data)} samples")

    # 3. Run Predictions
    preds = [model.predict(sample['features'])['label'] for sample in test_data]
    labels = [sample['label'] for sample in test_data]
    metrics = compute_metrics(preds, labels)
    print("[INFO] Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 4. Save Metrics
    if args.metrics_output_path:
        with open(args.metrics_output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Metrics saved to {args.metrics_output_path}")

if __name__ == "__main__":
    main()
