#!/usr/bin/env python3
"""
Plug-and-Play Training Script for Nethical ML Models

Usage:
    # Train with synthetic data:
    python train_model.py --model-type logistic --epochs 20 --batch-size 32 --num-samples 5000 --seed 123
    
    # Train with real data from datasets/datasets:
    python train_model.py --model-type logistic --epochs 20 --use-real-data

Supported Model Types (see MLModelType):
    - heuristic
    - logistic
    - simple_transformer
    - deep_nn
    - anomaly
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
import json
from datetime import datetime
from pathlib import Path

# --- Import your models here ---
from nethical.core.ml_shadow import MLModelType
from nethical.mlops.baseline import BaselineMLClassifier
from nethical.mlops.anomaly_classifier import AnomalyMLClassifier
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
        "anomaly": AnomalyMLClassifier,
        # "deep_nn": DeepNNClassifier,
        # "transformer": TransformerClassifier,
    }
    if model_type not in registry:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: {list(registry.keys())}")
    return registry[model_type]

def load_real_data_from_processed(data_dir="data/processed", num_samples=None):
    """Load real data from processed dataset files.
    
    Args:
        data_dir: Directory containing processed JSON files
        num_samples: Optional limit on number of samples to load
        
    Returns:
        List of data samples or None if no data available
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[WARNING] Data directory {data_dir} does not exist")
        return None
    
    # Find all processed JSON files
    json_files = list(data_path.glob("*_processed.json"))
    
    if not json_files:
        print(f"[WARNING] No processed JSON files found in {data_dir}")
        return None
    
    print(f"[INFO] Loading real data from {len(json_files)} processed file(s)...")
    all_data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                dataset = json.load(f)
                if isinstance(dataset, list):
                    all_data.extend(dataset)
                    print(f"[INFO] Loaded {len(dataset)} samples from {json_file.name}")
        except Exception as e:
            print(f"[WARNING] Failed to load {json_file.name}: {e}")
            continue
    
    if not all_data:
        print("[WARNING] No data loaded from processed files")
        return None
    
    # Shuffle the combined data
    random.shuffle(all_data)
    
    # Limit samples if requested
    if num_samples and num_samples < len(all_data):
        all_data = all_data[:num_samples]
        print(f"[INFO] Limited to {num_samples} samples")
    
    print(f"[INFO] Total real data samples loaded: {len(all_data)}")
    return all_data


def load_data(num_samples=10000, use_real_data=False):
    """Load training data (real or synthetic).
    
    Args:
        num_samples: Number of samples to load
        use_real_data: If True, attempt to load real data from processed files
        
    Returns:
        List of data samples
    """
    if use_real_data:
        real_data = load_real_data_from_processed(num_samples=num_samples)
        if real_data:
            return real_data
        print("[INFO] Falling back to synthetic data generation...")
    
    print(f"[INFO] Generating {num_samples} synthetic samples...")
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

def load_anomaly_data(num_samples=10000):
    """Load synthetic anomaly detection data with sequences.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        List of samples with sequence features
    """
    print(f"[INFO] Loading {num_samples} anomaly detection samples...")
    data = []
    
    # Define normal and anomalous patterns
    normal_patterns = [
        ['read', 'process', 'write'],
        ['read', 'validate', 'process'],
        ['fetch', 'transform', 'load'],
        ['query', 'filter', 'aggregate'],
        ['request', 'authenticate', 'respond'],
        ['connect', 'read', 'disconnect']
    ]
    
    anomalous_patterns = [
        ['delete', 'exfiltrate', 'cover_tracks'],
        ['escalate', 'access', 'exfiltrate'],
        ['scan', 'exploit', 'inject'],
        ['brute_force', 'access', 'modify'],
        ['bypass', 'escalate', 'execute']
    ]
    
    for i in range(num_samples):
        # 70% normal, 30% anomalous for balanced training
        if random.random() < 0.7:
            # Normal sample
            base_pattern = random.choice(normal_patterns)
            # Add some variation
            sequence = base_pattern.copy()
            if random.random() < 0.3:
                sequence.append(random.choice(['read', 'write', 'process', 'validate']))
            label = 0
        else:
            # Anomalous sample
            if random.random() < 0.8:
                # Use known anomalous pattern
                sequence = random.choice(anomalous_patterns).copy()
            else:
                # Create unusual sequence
                sequence = [random.choice(['unknown', 'suspicious', 'rare']) for _ in range(3)]
            label = 1
        
        features = {'sequence': sequence}
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
    parser.add_argument("--use-real-data", action="store_true", 
                        help="Use real data from data/processed directory (falls back to synthetic if unavailable)")
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Load Data
    if args.model_type == "anomaly":
        data = load_anomaly_data(num_samples=args.num_samples)
    else:
        data = load_data(num_samples=args.num_samples, use_real_data=args.use_real_data)
    train_data, val_data = temporal_split(data)

    # 2. Select Model
    model_type = args.model_type
    ModelClass = get_model_class(model_type)
    model = ModelClass()

    # 3. Train Model
    print(f"[INFO] Training {model_type} model...")
    # Some models (like anomaly) accept epochs and batch_size, others don't
    try:
        model.train(train_data, epochs=args.epochs, batch_size=args.batch_size)
    except TypeError:
        # Fallback for models that don't accept these parameters
        model.train(train_data)

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
