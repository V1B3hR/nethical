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

# Import Ethical Drift Reporter for drift tracking
try:
    from nethical.core.ethical_drift_reporter import EthicalDriftReporter
    DRIFT_REPORTER_AVAILABLE = True
except ImportError:
    DRIFT_REPORTER_AVAILABLE = False
    print("[WARN] EthicalDriftReporter not available. Drift tracking will be disabled.")

# Import Phase 8-9 Integration for human-in-the-loop and optimization
try:
    from nethical.core.phase89_integration import Phase89IntegratedGovernance
    PHASE89_AVAILABLE = True
except ImportError:
    PHASE89_AVAILABLE = False
    print("[WARN] Phase89IntegratedGovernance not available. Governance features will be disabled.")

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
        print("[WARN] Kaggle package not installed. Skipping dataset downloads. Using synthetic data.")
        return
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
    parser.add_argument("--enable-drift-tracking", action="store_true", help="Enable ethical drift tracking")
    parser.add_argument("--drift-report-dir", type=str, default="training_drift_reports", help="Directory for drift reports")
    parser.add_argument("--cohort-id", type=str, default=None, help="Cohort identifier for drift tracking (defaults to model-type_timestamp)")
    parser.add_argument("--enable-phase89", action="store_true", help="Enable Phase 8-9 governance integration")
    parser.add_argument("--phase89-storage", type=str, default="training_governance_data", help="Storage directory for Phase89 data")
    parser.add_argument("--optimize-config", action="store_true", help="Run configuration optimization using Phase89")
    parser.add_argument("--optimization-technique", type=str, default="random_search", choices=["grid_search", "random_search", "evolutionary"], help="Optimization technique")
    parser.add_argument("--optimization-iterations", type=int, default=20, help="Number of optimization iterations")
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

    # Initialize Ethical Drift Reporter for drift tracking
    drift_reporter = None
    cohort_id = args.cohort_id or f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.enable_drift_tracking and DRIFT_REPORTER_AVAILABLE:
        try:
            drift_reporter = EthicalDriftReporter(report_dir=args.drift_report_dir)
            print(f"[INFO] Ethical drift tracking enabled. Reports stored in: {args.drift_report_dir}")
            print(f"[INFO] Training cohort ID: {cohort_id}")
        except Exception as e:
            print(f"[WARN] Failed to initialize drift reporter: {e}")
            drift_reporter = None
    elif args.enable_drift_tracking and not DRIFT_REPORTER_AVAILABLE:
        print("[WARN] Drift tracking requested but EthicalDriftReporter not available")

    # Initialize Phase 8-9 Integrated Governance
    governance = None
    if args.enable_phase89 and PHASE89_AVAILABLE:
        try:
            governance = Phase89IntegratedGovernance(
                storage_dir=args.phase89_storage,
                triage_sla_seconds=3600,  # 1 hour
                resolution_sla_seconds=86400,  # 24 hours
                auto_escalate_on_block=True,
                auto_escalate_on_low_confidence=True,
                low_confidence_threshold=0.7
            )
            print(f"[INFO] Phase 8-9 governance enabled. Data stored in: {args.phase89_storage}")
        except Exception as e:
            print(f"[WARN] Failed to initialize Phase89 governance: {e}")
            governance = None
    elif args.enable_phase89 and not PHASE89_AVAILABLE:
        print("[WARN] Phase89 governance requested but Phase89IntegratedGovernance not available")

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
    
    # Track training performance for drift analysis
    if drift_reporter:
        # Track validation as an action with risk score based on accuracy
        # Higher accuracy = lower risk
        risk_score = 1.0 - metrics['accuracy']
        drift_reporter.track_action(
            agent_id=f"model_{args.model_type}",
            cohort=cohort_id,
            risk_score=risk_score
        )

    # Create and track configuration with Phase89
    config_version = f"{args.model_type}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    baseline_config_id = None
    if governance:
        try:
            print("\n[INFO] Recording configuration with Phase89 governance...")
            # Create configuration for this training run
            config = governance.create_configuration(
                config_version=config_version,
                classifier_threshold=0.5,
                confidence_threshold=0.7,
                gray_zone_lower=0.4,
                gray_zone_upper=0.6,
                metadata={
                    'model_type': args.model_type,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'num_samples': args.num_samples,
                    'seed': args.seed,
                    'training_timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            baseline_config_id = config.config_id
            print(f"[INFO] Configuration ID: {config.config_id}")
            
            # Record metrics with Phase89
            # Map training metrics to Phase89 metrics format
            phase89_metrics = governance.record_metrics(
                config_id=config.config_id,
                detection_recall=metrics['recall'],
                detection_precision=metrics['precision'],
                false_positive_rate=1.0 - metrics['precision'],  # Approximation
                decision_latency_ms=10.0,  # Placeholder - would be actual inference time
                human_agreement=0.85,  # Placeholder - would come from human review
                total_cases=len(val_data)
            )
            print(f"[INFO] Metrics recorded. Fitness score: {phase89_metrics.fitness_score:.4f}")
        except Exception as e:
            print(f"[WARN] Failed to record Phase89 metrics: {e}")

    promoted = check_promotion_gate(metrics)
    
    # Track promotion gate result for drift analysis
    if drift_reporter:
        if not promoted:
            # Track promotion gate failure as a violation
            if metrics['ece'] > 0.08:
                drift_reporter.track_violation(
                    agent_id=f"model_{args.model_type}",
                    cohort=cohort_id,
                    violation_type="calibration_error",
                    severity="high" if metrics['ece'] > 0.15 else "medium"
                )
            if metrics['accuracy'] < 0.85:
                drift_reporter.track_violation(
                    agent_id=f"model_{args.model_type}",
                    cohort=cohort_id,
                    violation_type="low_accuracy",
                    severity="high" if metrics['accuracy'] < 0.70 else "medium"
                )
    
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
    
    # Generate drift report if tracking is enabled
    if drift_reporter:
        try:
            print("\n[INFO] Generating ethical drift report...")
            report_start_time = training_start_time
            report_end_time = datetime.now(timezone.utc)
            
            drift_report = drift_reporter.generate_report(
                start_time=report_start_time,
                end_time=report_end_time
            )
            
            print(f"[INFO] Drift Report ID: {drift_report.report_id}")
            print(f"[INFO] Drift detected: {drift_report.drift_metrics.get('has_drift', False)}")
            
            if drift_report.recommendations:
                print("\n[INFO] Drift Analysis Recommendations:")
                for i, rec in enumerate(drift_report.recommendations[:5], 1):
                    print(f"  {i}. {rec}")
            
            # Save drift report path reference in model metadata
            drift_report_path = Path(args.drift_report_dir) / f"{drift_report.report_id}.json"
            print(f"[INFO] Drift report saved to: {drift_report_path}")
            
        except Exception as e:
            print(f"[WARN] Failed to generate drift report: {e}")
    
    # Run configuration optimization if requested
    if args.optimize_config and governance and baseline_config_id:
        try:
            print(f"\n[INFO] Running configuration optimization using {args.optimization_technique}...")
            
            # Define custom evaluation function for model configuration
            def evaluate_model_config(config):
                """Evaluate a model configuration."""
                # In a real scenario, this would retrain/evaluate the model with new hyperparameters
                # For now, we'll use the recorded metrics and add some variation
                import random
                base_recall = metrics['recall']
                base_precision = metrics['precision']
                
                # Simulate variation based on configuration parameters
                recall_var = random.uniform(-0.05, 0.05)
                precision_var = random.uniform(-0.05, 0.05)
                
                return governance.record_metrics(
                    config_id=config.config_id,
                    detection_recall=max(0.0, min(1.0, base_recall + recall_var)),
                    detection_precision=max(0.0, min(1.0, base_precision + precision_var)),
                    false_positive_rate=random.uniform(0.01, 0.15),
                    decision_latency_ms=random.uniform(8.0, 15.0),
                    human_agreement=random.uniform(0.80, 0.95),
                    total_cases=len(val_data)
                )
            
            # Run optimization
            if args.optimization_technique == "random_search":
                optimization_results = governance.optimize_configuration(
                    technique="random_search",
                    param_ranges={
                        'classifier_threshold': (0.4, 0.7),
                        'confidence_threshold': (0.6, 0.9),
                        'gray_zone_lower': (0.3, 0.5),
                        'gray_zone_upper': (0.5, 0.7)
                    },
                    n_iterations=args.optimization_iterations
                )
            elif args.optimization_technique == "grid_search":
                optimization_results = governance.optimize_configuration(
                    technique="grid_search",
                    param_grid={
                        'classifier_threshold': [0.4, 0.5, 0.6],
                        'gray_zone_lower': [0.3, 0.4, 0.5],
                        'gray_zone_upper': [0.5, 0.6, 0.7]
                    }
                )
            else:  # evolutionary
                base_config = governance.optimizer.configurations.get(baseline_config_id)
                optimization_results = governance.optimize_configuration(
                    technique="evolutionary",
                    base_config=base_config,
                    population_size=10,
                    n_generations=5,
                    mutation_rate=0.2
                )
            
            print(f"[INFO] Optimization completed. Evaluated {len(optimization_results)} configurations")
            
            # Display top 3 configurations
            print("\n[INFO] Top 3 Configurations:")
            for i, (config, opt_metrics) in enumerate(optimization_results[:3], 1):
                print(f"  {i}. {config.config_version}")
                print(f"     Fitness: {opt_metrics.fitness_score:.4f}")
                print(f"     Recall: {opt_metrics.detection_recall:.3f}, Precision: {opt_metrics.detection_precision:.3f}")
                print(f"     FP Rate: {opt_metrics.false_positive_rate:.3f}")
            
            # Check promotion gate for best configuration
            if len(optimization_results) > 0:
                best_config, best_metrics = optimization_results[0]
                print(f"\n[INFO] Checking promotion gate for best configuration...")
                passed, reasons = governance.check_promotion_gate(
                    candidate_id=best_config.config_id,
                    baseline_id=baseline_config_id
                )
                
                print(f"[INFO] Promotion gate: {'PASSED' if passed else 'FAILED'}")
                for reason in reasons:
                    print(f"  - {reason}")
                
                if passed:
                    print(f"[INFO] Promoting configuration {best_config.config_version} to production...")
                    promoted_success = governance.promote_configuration(best_config.config_id)
                    if promoted_success:
                        print(f"[INFO] Configuration promoted successfully")
                    else:
                        print(f"[WARN] Failed to promote configuration")
                        
        except Exception as e:
            print(f"[WARN] Failed to run configuration optimization: {e}")
            import traceback
            traceback.print_exc()
    
    # Finalize Merkle audit chunk if enabled
    if merkle_anchor:
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
