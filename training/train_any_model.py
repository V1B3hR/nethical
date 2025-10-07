#!/usr/bin/env python3
"""
Plug-and-Play Model Training Script for Nethical

- Downloads datasets from Kaggle using your API key (writes it to ~/.kaggle/kaggle.json if needed).
- Loads, preprocesses, and splits data according to the chosen model.
- Trains the model, computes metrics, checks promotion gate, and saves results.

Usage:
    python train_any_model.py --model-type logistic --epochs 30 --batch-size 64 --num-samples 4000 --seed 32

Dependencies: kaggle, pandas, numpy
"""

import argparse
import os
import sys
import random
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Merkle Anchor for audit logging
try:
    from nethical.core.audit_merkle import MerkleAnchor
    MERKLE_AVAILABLE = True
except ImportError:
    MERKLE_AVAILABLE = False
    print("[WARN] MerkleAnchor not available. Audit logging will be disabled.")

# ======= USER CONFIGURE HERE =======
KAGGLE_USERNAME = "andrzejmatewski"
KAGGLE_KEY = "bb8941672c5cc299926e65234a901284"

KAGGLE_DATASETS = [
    "teamincribo/cyber-security-attacks",
    "Microsoft/microsoft-security-incident-prediction",
    "kmldas/data-ethics-in-data-science-analytics-ml-and-ai",
    "xontoloyo/security-breachhh",
    "daylight-lab/cybersecurity-imagery-dataset",
    "dasgroup/rba-dataset",
    "mpwolke/cusersmarildownloadsphilosophycsv",
    "clmentbisaillon/fake-and-real-news-dataset",
    "pratyushpuri/ai-vs-human-content-detection-1000-record-in-2025",
    "hardkazakh/ai-generated-vs-human-written-text-dataset",
    "nelgiriyewithana/emotions",
    "andrewmvd/cyberbullying-classification",
    "bhavikjikadara/fake-news-detection",
]
DATA_EXTERNAL_DIR = Path("data/external")
# ====================================

def ensure_kaggle_json():
    kaggle_path = Path.home() / ".kaggle"
    kaggle_json_path = kaggle_path / "kaggle.json"
    kaggle_creds = {
        "username": KAGGLE_USERNAME,
        "key": KAGGLE_KEY
    }
    kaggle_path.mkdir(parents=True, exist_ok=True)
    if not kaggle_json_path.exists():
        print(f"[INFO] Writing Kaggle credentials to {kaggle_json_path}")
        with open(kaggle_json_path, "w") as f:
            json.dump(kaggle_creds, f)
        os.chmod(kaggle_json_path, 0o600)
    else:
        print(f"[INFO] Kaggle credentials already present at {kaggle_json_path}")

def download_kaggle_datasets():
    ensure_kaggle_json()
    try:
        import kaggle
    except ImportError:
        print("[ERROR] Please install the kaggle package: pip install kaggle")
        sys.exit(1)
    DATA_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in KAGGLE_DATASETS:
        print(f"[INFO] Downloading {dataset} ...")
        try:
            kaggle.api.dataset_download_files(dataset, path=str(DATA_EXTERNAL_DIR), unzip=True)
            print(f"[INFO] Downloaded and extracted: {dataset}")
        except Exception as e:
            print(f"[WARN] Could not download {dataset}: {e}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

# ----------- Model Registry & Preprocessing -----------
def get_model_class(model_type: str):
    from nethical.core.ml_shadow import MLModelType
    from nethical.mlops.baseline import BaselineMLClassifier
    from nethical.mlops.anomaly_classifier import AnomalyMLClassifier
    from nethical.mlops.correlation_classifier import CorrelationMLClassifier
    # from nethical.mlops.deep_nn import DeepNNClassifier
    # from nethical.mlops.transformer import TransformerClassifier
    registry = {
        "heuristic": (BaselineMLClassifier, preprocess_for_heuristic),
        "logistic": (BaselineMLClassifier, preprocess_for_logistic),
        "simple_transformer": (BaselineMLClassifier, preprocess_for_transformer),
        "anomaly": (AnomalyMLClassifier, preprocess_for_anomaly),
        "correlation": (CorrelationMLClassifier, preprocess_for_correlation),
        # "deep_nn": (DeepNNClassifier, preprocess_for_deep_nn),
        # "transformer": (TransformerClassifier, preprocess_for_transformer),
    }
    if model_type not in registry:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: {list(registry.keys())}")
    return registry[model_type]

def preprocess_for_heuristic(data):
    # Example: Use raw numeric features, no scaling
    for sample in data:
        feats = sample["features"]
        sample["features"] = {k: float(feats.get(k, 0.0)) for k in ["violation_count","severity_max","recency_score","frequency_score","context_risk"]}
    return data

def preprocess_for_logistic(data):
    # Example: Normalize all features to [0,1]
    features = [list(sample["features"].values()) for sample in data]
    arr = np.array(features)
    mins, maxs = arr.min(axis=0), arr.max(axis=0)
    for i, sample in enumerate(data):
        sample["features"] = {k: (v-mins[j])/(maxs[j]-mins[j]+1e-8)
                              for j,(k,v) in enumerate(sample["features"].items())}
    return data

def preprocess_for_transformer(data):
    # Example: Pretend we tokenize a text field (not real code, just placeholder)
    for sample in data:
        feats = sample["features"]
        # If the dataset has a 'text' field, replace with dummy token ids
        if "text" in feats:
            tokens = [ord(c) % 100 for c in feats["text"][:32]]
            sample["features"]["token_ids"] = tokens
            del sample["features"]["text"]
    return data

def preprocess_for_anomaly(data):
    # For anomaly detection, data should already be in the right format with sequences
    # Each sample should have features['sequence'] as a list of actions
    # No additional preprocessing needed
    return data

def preprocess_for_correlation(data):
    # For correlation detection, normalize all numeric features to [0,1]
    # Correlation features include: agent_count, action_rate, entropy_variance,
    # time_correlation, payload_similarity
    features = []
    for sample in data:
        feats = sample["features"]
        if isinstance(feats, dict):
            # Extract numeric features
            features.append([
                float(feats.get('agent_count', 0)),
                float(feats.get('action_rate', 0)),
                float(feats.get('entropy_variance', 0)),
                float(feats.get('time_correlation', 0)),
                float(feats.get('payload_similarity', 0))
            ])
    
    if features:
        arr = np.array(features)
        mins, maxs = arr.min(axis=0), arr.max(axis=0)
        
        feature_names = ['agent_count', 'action_rate', 'entropy_variance', 'time_correlation', 'payload_similarity']
        for i, sample in enumerate(data):
            normalized = {}
            for j, name in enumerate(feature_names):
                val = features[i][j]
                normalized[name] = (val - mins[j]) / (maxs[j] - mins[j] + 1e-8)
            sample["features"] = normalized
    
    return data

# def preprocess_for_deep_nn(data):
#     # Add real preprocessing for deep NN if needed (normalization, encoding, etc.)
#     return data

def load_data(num_samples=10000, model_type='logistic'):
    csv_files = list(DATA_EXTERNAL_DIR.glob("*.csv"))
    if csv_files:
        print(f"[INFO] Found {len(csv_files)} Kaggle CSVs. Loading and sampling data for training.")
        import pandas as pd
        data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Find numeric features, fallback to all float columns
                feature_cols = [c for c in df.columns if c.lower() in (
                    "violation_count", "severity_max", "recency_score", "frequency_score", "context_risk")]
                if not feature_cols:
                    feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    if not feature_cols:
                        continue
                label_col = next((c for c in df.columns if "label" in c.lower() or "target" in c.lower()), None)
                if not label_col:
                    continue
                for _, row in df.iterrows():
                    features = {col: row[col] for col in feature_cols if not pd.isna(row[col])}
                    if len(features) != len(feature_cols):
                        continue
                    label = int(row[label_col])
                    data.append({"features": features, "label": label})
                    if len(data) >= num_samples:
                        break
                if len(data) >= num_samples:
                    break
            except Exception as e:
                print(f"[WARN] Error reading {csv_file}: {e}")
        if not data:
            print("[WARN] No valid samples found in Kaggle data. Falling back to synthetic data.")
        else:
            return data
    print(f"[INFO] Loading {num_samples} synthetic samples...")
    data = []
    
    # Generate data based on model type
    if model_type == 'anomaly':
        # Generate synthetic anomaly detection data with action sequences
        # Normal patterns (70% of data)
        normal_patterns = [
            ['read', 'process', 'write'],
            ['fetch', 'transform', 'load'],
            ['query', 'filter', 'aggregate'],
            ['request', 'authenticate', 'respond'],
            ['open', 'read', 'close'],
            ['connect', 'send', 'receive'],
            ['validate', 'process', 'store'],
            ['load', 'analyze', 'report']
        ]
        
        # Anomalous patterns (30% of data)
        anomalous_patterns = [
            ['delete', 'exfiltrate', 'cover_tracks'],
            ['escalate', 'access', 'exfiltrate'],
            ['scan', 'exploit', 'inject'],
            ['brute_force', 'access', 'modify'],
            ['bypass', 'steal', 'hide'],
            ['probe', 'breach', 'extract'],
            ['intercept', 'decrypt', 'leak'],
            ['overflow', 'execute', 'control']
        ]
        
        normal_count = int(num_samples * 0.7)
        anomalous_count = num_samples - normal_count
        
        # Generate normal samples
        for _ in range(normal_count):
            pattern = random.choice(normal_patterns)
            data.append({
                'features': {'sequence': pattern.copy()},
                'label': 0  # Normal
            })
        
        # Generate anomalous samples
        for _ in range(anomalous_count):
            pattern = random.choice(anomalous_patterns)
            data.append({
                'features': {'sequence': pattern.copy()},
                'label': 1  # Anomalous
            })
        
        # Shuffle the data
        random.shuffle(data)
    elif model_type == 'correlation':
        # Generate synthetic correlation pattern detection data
        # Normal multi-agent activity (65% of data)
        normal_count = int(num_samples * 0.65)
        correlation_count = num_samples - normal_count
        
        # Generate normal samples (no correlation patterns)
        for _ in range(normal_count):
            features = {
                'agent_count': random.randint(1, 3),  # Few agents
                'action_rate': random.uniform(1, 10),  # Low action rate
                'entropy_variance': random.uniform(0.05, 0.3),  # Low entropy variance
                'time_correlation': random.uniform(0, 0.25),  # Low time correlation
                'payload_similarity': random.uniform(0, 0.3)  # Low similarity
            }
            data.append({
                'features': features,
                'label': 0  # No correlation pattern
            })
        
        # Generate correlation pattern samples
        for _ in range(correlation_count):
            # Simulate correlation patterns with specific characteristics
            pattern_type = random.choice(['escalating', 'coordinated', 'distributed'])
            
            if pattern_type == 'escalating':
                # Escalating multi-ID probes: many agents, increasing action rate
                features = {
                    'agent_count': random.randint(6, 15),
                    'action_rate': random.uniform(25, 100),
                    'entropy_variance': random.uniform(0.4, 0.9),
                    'time_correlation': random.uniform(0.5, 0.75),
                    'payload_similarity': random.uniform(0.4, 0.8)
                }
            elif pattern_type == 'coordinated':
                # Coordinated attack: moderate agents, high time correlation
                features = {
                    'agent_count': random.randint(4, 10),
                    'action_rate': random.uniform(20, 60),
                    'entropy_variance': random.uniform(0.3, 0.7),
                    'time_correlation': random.uniform(0.75, 1.0),  # High time correlation
                    'payload_similarity': random.uniform(0.6, 0.95)
                }
            else:  # distributed
                # Distributed reconnaissance: many agents, diverse payloads
                features = {
                    'agent_count': random.randint(10, 20),
                    'action_rate': random.uniform(15, 50),
                    'entropy_variance': random.uniform(0.65, 1.0),  # High entropy variance
                    'time_correlation': random.uniform(0.3, 0.6),
                    'payload_similarity': random.uniform(0.25, 0.65)
                }
            
            data.append({
                'features': features,
                'label': 1  # Correlation pattern detected
            })
        
        # Shuffle the data
        random.shuffle(data)
    else:
        # Original synthetic data for other model types
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
    parser.add_argument("--enable-audit", action="store_true", help="Enable Merkle audit logging")
    parser.add_argument("--audit-path", type=str, default="training_audit_logs", help="Path for audit logs")
    args = parser.parse_args()

    # Initialize Merkle Anchor for audit logging
    merkle_anchor = None
    if args.enable_audit and MERKLE_AVAILABLE:
        try:
            merkle_anchor = MerkleAnchor(
                storage_path=args.audit_path,
                chunk_size=100  # Smaller chunk size for training events
            )
            print(f"[INFO] Merkle audit logging enabled. Logs stored in: {args.audit_path}")
            
            # Log training start event
            merkle_anchor.add_event({
                'event_type': 'training_start',
                'model_type': args.model_type,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'num_samples': args.num_samples,
                'seed': args.seed,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            print(f"[WARN] Failed to initialize Merkle audit logging: {e}")
            merkle_anchor = None
    elif args.enable_audit and not MERKLE_AVAILABLE:
        print("[WARN] Audit logging requested but MerkleAnchor not available")

    set_seed(args.seed)
    download_kaggle_datasets()
    raw_data = load_data(num_samples=args.num_samples, model_type=args.model_type)
    
    # Log data loading event
    if merkle_anchor:
        merkle_anchor.add_event({
            'event_type': 'data_loaded',
            'num_samples': len(raw_data),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    ModelClass, preprocess_fn = get_model_class(args.model_type)
    print(f"[INFO] Preprocessing data for model type: {args.model_type}")
    processed_data = preprocess_fn(raw_data)

    train_data, val_data = temporal_split(processed_data)
    
    # Log data split event
    if merkle_anchor:
        merkle_anchor.add_event({
            'event_type': 'data_split',
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    model = ModelClass()
    print(f"[INFO] Training {args.model_type} model for {args.epochs} epochs, batch size {args.batch_size}...")
    training_start_time = datetime.now(timezone.utc)
    # Note: BaselineMLClassifier doesn't use epochs/batch_size, but we keep them for extensibility
    model.train(train_data)
    training_end_time = datetime.now(timezone.utc)
    
    # Log training completion event
    if merkle_anchor:
        merkle_anchor.add_event({
            'event_type': 'training_completed',
            'model_type': args.model_type,
            'epochs': args.epochs,
            'training_duration_seconds': (training_end_time - training_start_time).total_seconds(),
            'timestamp': training_end_time.isoformat()
        })

    preds = [model.predict(sample['features'])['label'] for sample in val_data]
    labels = [sample['label'] for sample in val_data]
    metrics = compute_metrics(preds, labels)
    print("[INFO] Validation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Log validation metrics event
    if merkle_anchor:
        merkle_anchor.add_event({
            'event_type': 'validation_metrics',
            'metrics': metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    promoted = check_promotion_gate(metrics)
    model_path, metrics_path = save_model_and_metrics(model, metrics, args.model_type, promoted=promoted)
    
    # Log model save event
    if merkle_anchor:
        merkle_anchor.add_event({
            'event_type': 'model_saved',
            'model_path': model_path,
            'metrics_path': metrics_path,
            'promoted': promoted,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Finalize chunk and get Merkle root
        try:
            if merkle_anchor.current_chunk.event_count > 0:
                merkle_root = merkle_anchor.finalize_chunk()
                print(f"[INFO] Training audit trail finalized. Merkle root: {merkle_root}")
                print(f"[INFO] Audit logs saved to: {args.audit_path}")
                
                # Save audit summary
                audit_summary_path = Path(args.audit_path) / "training_summary.json"
                audit_summary = {
                    'merkle_root': merkle_root,
                    'model_type': args.model_type,
                    'promoted': promoted,
                    'metrics': metrics,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                with open(audit_summary_path, 'w') as f:
                    json.dump(audit_summary, f, indent=2)
                print(f"[INFO] Audit summary saved to: {audit_summary_path}")
        except Exception as e:
            print(f"[WARN] Failed to finalize audit chunk: {e}")

if __name__ == "__main__":
    main()
