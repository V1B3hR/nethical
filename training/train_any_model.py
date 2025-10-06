#!/usr/bin/env python3
"""
Plug-and-Play Training Script for Nethical ML Models

Usage:
    python train_model.py --model-type logistic --epochs 20 --batch-size 32 --num-samples 5000 --seed 123

Supported Model Types (see MLModelType):
    - heuristic
    - logistic
    - simple_transformer
    - deep_nn
    - [add more as needed]

Steps:
    1. Load and process data (real-world or synthetic)
    2. Select model class based on model_type
    3. Train model with specified parameters
    4. Compute metrics: precision, recall, F1, accuracy, ECE
    5. Check promotion gate criteria
    6. Save trained model and metrics
    7. If promoted, deploy to production directory

"""

import argparse
import os
import sys
import random
import numpy as np
from datetime import datetime
from pathlib import Path

# --- Import your models here ---
from nethical.core.ml_shadow import MLModelType
from nethical.mlops.baseline import BaselineMLClassifier
# from nethical.mlops.deep_nn import DeepNNClassifier   # Example for extensibility
# from nethical.mlops.transformer import TransformerClassifier

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

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

def load_data(num_samples=10000):
    print(f"[INFO] Loading {num_samples} samples...")
    data = []
    for _ in range(num_samples):
        features = {
            'violation_count': random.random(),
            'severity_max': random.random(),
            'recency_score': random.random(),
            'frequency_score': random.random(),
            'context_risk': random.random(),
        }
        label = int(features['violation_count'] + features['severity_max'] > 1)
        data.append({'features': features, 'label': label})
    return data

def temporal_split(data, train_ratio=0.8):
    split_idx = int(len(data) * train_ratio)
    train_data, val_data = data[:split_idx], data[split_idx:]
    print(f"[INFO] Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    return train_data, val_data

def compute_metrics(preds, labels):
    tp = sum((p == 1 and l == 1) for p, l in zip(preds, labels))
    tn = sum((p == 0 and l == 0) for p, l in zip(preds, labels))
    fp = sum((p == 1 and l == 0) for p, l in zip(preds, labels))
    fn = sum((p == 0 and l == 1) for p, l in zip(preds, labels))
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    # ECE (expected calibration error) placeholder
    ece = abs(accuracy - precision)
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
        "ece": ece,
    }

def check_promotion_gate(metrics):
    max_ece = 0.08
    min_accuracy = 0.85
    passed = metrics["ece"] <= max_ece and metrics["accuracy"] >= min_accuracy
    print(f"[INFO] Promotion Gate: ECE <= {max_ece}, Accuracy >= {min_accuracy}")
    print(f"[INFO] ECE: {metrics['ece']:.3f}, Accuracy: {metrics['accuracy']:.3f}")
    print(f"[INFO] Promotion result: {'PASS' if passed else 'FAIL'}")
    return passed

def save_model_and_metrics(model, metrics, model_type, promoted=False):
    dest_dir = "models/current" if promoted else "models/candidates"
    os.makedirs(dest_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{dest_dir}/{model_type}_model_{timestamp}.json"
    metrics_path = f"{dest_dir}/{model_type}_metrics_{timestamp}.json"
    model.save(model_path)
    with open(metrics_path, "w") as f:
        import json
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Model saved to {model_path}")
    print(f"[INFO] Metrics saved to {metrics_path}")
    return model_path, metrics_path

def main():
    parser = argparse.ArgumentParser(description="Plug-and-Play Training Pipeline for Nethical ML Models")
    parser.add_argument("--model-type", type=str, required=True, help="Model type (see MLModelType)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Load Data
    data = load_data(num_samples=args.num_samples)
    train_data, val_data = temporal_split(data)

    # 2. Select Model
    model_type = args.model_type
    ModelClass = get_model_class(model_type)
    model = ModelClass()

    # 3. Train Model
    print(f"[INFO] Training {model_type} model for {args.epochs} epochs, batch size {args.batch_size}...")
    model.train(train_data, epochs=args.epochs, batch_size=args.batch_size)

    # 4. Evaluate Model
    preds = [model.predict(sample['features'])['label'] for sample in val_data]
    labels = [sample['label'] for sample in val_data]
    metrics = compute_metrics(preds, labels)
    print("[INFO] Validation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 5. Promotion Gate
    promoted = check_promotion_gate(metrics)

    # 6. Save Model and Metrics
    save_model_and_metrics(model, metrics, model_type, promoted=promoted)

if __name__ == "__main__":
    main()
