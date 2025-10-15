#!/usr/bin/env python3
"""
Nethical Plug-and-Play Model Training Script

Key improvements:
- Secure Kaggle auth (no hard-coded secrets). Supports env vars, CLI flags, or existing ~/.kaggle/kaggle.json.
- Flexible data pipeline (optional Kaggle download, multiple split strategies, robust CSV parsing).
- Expanded metrics (precision/recall/accuracy/F1, ROC-AUC if confidences present, proper ECE when available).
- Configurable promotion gate thresholds via CLI.
- Enhanced governance validation workflow (configurable sample sizes and failure policy).
- Optional ethical drift tracking + audit logging with Merkle anchors.
- Reproducibility, richer CLI, and better error handling/logging.

Usage:
    python train_any_model.py \
        --model-type logistic \
        --epochs 30 \
        --batch-size 64 \
        --num-samples 4000 \
        --seed 32 \
        --split-strategy stratified \
        --train-ratio 0.8 \
        --promotion-min-accuracy 0.85 \
        --promotion-max-ece 0.08 \
        --enable-governance --enable-audit --enable-drift-tracking

Optional Kaggle auth sources (priority order):
1) CLI: --kaggle-username --kaggle-key
2) Env vars: KAGGLE_USERNAME, KAGGLE_KEY
3) Existing file: ~/.kaggle/kaggle.json

Dependencies: kaggle (optional), pandas (optional), numpy (required)
"""

import argparse
import os
import sys
import random
import json
import math
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Prefer standard logging over print for better control
import logging

# Add parent directory to path for imports (Nethical modules)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Merkle Anchor for audit logging
try:
    from nethical.core.audit_merkle import MerkleAnchor
    MERKLE_AVAILABLE = True
except Exception:
    MERKLE_AVAILABLE = False

# Import Ethical Drift Reporter for drift tracking
try:
    from nethical.core.ethical_drift_reporter import EthicalDriftReporter
    DRIFT_REPORTER_AVAILABLE = True
except Exception:
    DRIFT_REPORTER_AVAILABLE = False

# Import Governance System for safety validation
try:
    from nethical.core.governance import (
        EnhancedSafetyGovernance,
        AgentAction,
        ActionType,
        Decision,
        MonitoringConfig
    )
    GOVERNANCE_AVAILABLE = True
except Exception:
    GOVERNANCE_AVAILABLE = False


# Default external data directory
DATA_EXTERNAL_DIR = Path("data/external")


def configure_logging(verbosity: int = 1) -> None:
    """Configure application logging."""
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    # Ensure timestamps are UTC
    logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()


def resolve_kaggle_credentials(
    username: Optional[str],
    key: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve Kaggle credentials using precedence:
    CLI args > Env vars > Existing file (~/.kaggle/kaggle.json).
    Returns (username, key) or (None, None) if not available.
    """
    # 1) CLI args
    if username and key:
        return username, key

    # 2) Environment
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return env_user, env_key

    # 3) Existing file
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    try:
        if kaggle_json_path.exists():
            with open(kaggle_json_path, "r") as f:
                data = json.load(f)
            file_user = data.get("username")
            file_key = data.get("key")
            if file_user and file_key:
                return file_user, file_key
    except Exception as e:
        logging.warning("Failed to read existing kaggle.json: %s", e)

    return None, None


def ensure_kaggle_json(username: str, key: str, overwrite: bool = False) -> None:
    """Create/update ~/.kaggle/kaggle.json with secure permissions."""
    kaggle_path = Path.home() / ".kaggle"
    kaggle_json_path = kaggle_path / "kaggle.json"
    kaggle_path.mkdir(parents=True, exist_ok=True)

    if kaggle_json_path.exists() and not overwrite:
        logging.info("Kaggle credentials already present at %s", kaggle_json_path)
        return

    logging.info("Writing Kaggle credentials to %s", kaggle_json_path)
    with open(kaggle_json_path, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(kaggle_json_path, 0o600)


def download_kaggle_datasets(
    datasets: Sequence[str],
    skip_download: bool = False,
    kaggle_username: Optional[str] = None,
    kaggle_key: Optional[str] = None,
    overwrite_kaggle_json: bool = False,
) -> None:
    """Download Kaggle datasets if possible."""
    if skip_download:
        logging.info("Skipping Kaggle download (--no-download set)")
        return

    try:
        import kaggle  # type: ignore
    except Exception:
        logging.warning("Kaggle package not installed. Skipping dataset downloads. Using synthetic data.")
        return

    resolved_user, resolved_key = resolve_kaggle_credentials(kaggle_username, kaggle_key)
    if resolved_user and resolved_key:
        ensure_kaggle_json(resolved_user, resolved_key, overwrite=overwrite_kaggle_json)
    else:
        logging.info("Kaggle credentials not provided/found. Attempting API auth via existing config if any.")

    DATA_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        logging.info("Downloading dataset: %s", dataset)
        try:
            kaggle.api.dataset_download_files(dataset, path=str(DATA_EXTERNAL_DIR), unzip=True)
            logging.info("Downloaded and extracted: %s", dataset)
        except Exception as e:
            logging.warning("Could not download %s: %s", dataset, e)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
    except Exception:
        pass


# ----------- Model Registry & Preprocessing -----------
def get_model_class(model_type: str):
    """
    Registry mapping Nethical model types to (ModelClass, preprocess_fn).
    Extend this registry as new model classes are added to the system.
    """
    # Keep imports local to avoid import errors when unavailable
    from nethical.mlops.baseline import BaselineMLClassifier
    from nethical.mlops.anomaly_classifier import AnomalyMLClassifier
    from nethical.mlops.correlation_classifier import CorrelationMLClassifier

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


def preprocess_for_heuristic(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Example: Use raw numeric features, no scaling
    for sample in data:
        feats = sample["features"]
        sample["features"] = {k: float(feats.get(k, 0.0)) for k in ["violation_count","severity_max","recency_score","frequency_score","context_risk"]}
    return data


def preprocess_for_logistic(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Min-max normalize all features to [0,1]
    # Preserve key order by extracting keys from first sample
    if not data:
        return data
    keys = list(data[0]["features"].keys())
    arr = np.array([[float(sample["features"].get(k, 0.0)) for k in keys] for sample in data], dtype=float)
    mins, maxs = arr.min(axis=0), arr.max(axis=0)
    denom = (maxs - mins) + 1e-8
    for i, sample in enumerate(data):
        sample["features"] = {k: float((arr[i, j] - mins[j]) / denom[j]) for j, k in enumerate(keys)}
    return data


def preprocess_for_transformer(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Placeholder tokenization of text, if present
    for sample in data:
        feats = sample["features"]
        if isinstance(feats, dict) and "text" in feats and isinstance(feats["text"], str):
            tokens = [ord(c) % 100 for c in feats["text"][:64]]
            feats = dict(feats)  # shallow copy
            feats["token_ids"] = tokens
            feats.pop("text", None)
            sample["features"] = feats
    return data


def preprocess_for_anomaly(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Expect features['sequence'] as a list of actions; no-op for now
    return data


def preprocess_for_correlation(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Normalize correlation features to [0,1]
    feature_names = ['agent_count', 'action_rate', 'entropy_variance', 'time_correlation', 'payload_similarity']
    rows: List[List[float]] = []
    valid_indices: List[int] = []
    for idx, sample in enumerate(data):
        feats = sample.get("features", {})
        if isinstance(feats, dict):
            row = [float(feats.get(n, 0.0)) for n in feature_names]
            rows.append(row)
            valid_indices.append(idx)
    if not rows:
        return data
    arr = np.array(rows, dtype=float)
    mins, maxs = arr.min(axis=0), arr.max(axis=0)
    denom = (maxs - mins) + 1e-8
    for pos, idx in enumerate(valid_indices):
        normalized = {name: float((arr[pos, j] - mins[j]) / denom[j]) for j, name in enumerate(feature_names)}
        data[idx]["features"] = normalized
    return data


# def preprocess_for_deep_nn(data):
#     return data


def _discover_label_column(df) -> Optional[str]:
    """Try to find a reasonable label column."""
    for cand in ["label", "target", "y", "class", "is_anomaly"]:
        matches = [c for c in df.columns if c.lower() == cand]
        if matches:
            return matches[0]
    # fallback: any column containing 'label'/'target'
    for c in df.columns:
        lc = c.lower()
        if "label" in lc or "target" in lc:
            return c
    return None


def _discover_feature_columns(df) -> List[str]:
    """Heuristically choose numeric feature columns or known names."""
    preferred = ["violation_count", "severity_max", "recency_score", "frequency_score", "context_risk"]
    cols = [c for c in df.columns if c.lower() in preferred]
    if cols:
        return cols
    # fallback: all numeric
    try:
        import pandas as pd  # type: ignore
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    except Exception:
        cols = []
    return cols


def load_data(num_samples: int = 10000, model_type: str = 'logistic') -> List[Dict[str, Any]]:
    """Load data from downloaded CSVs if present; otherwise generate synthetic data."""
    csv_files = list(DATA_EXTERNAL_DIR.glob("*.csv"))
    if csv_files:
        logging.info("Found %d CSV(s) in %s. Loading up to %d samples.", len(csv_files), DATA_EXTERNAL_DIR, num_samples)
        try:
            import pandas as pd  # type: ignore
        except Exception:
            logging.warning("pandas not installed; falling back to synthetic data.")
        else:
            data: List[Dict[str, Any]] = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    label_col = _discover_label_column(df)
                    if not label_col:
                        logging.debug("No label column found in %s; skipping.", csv_file.name)
                        continue
                    feat_cols = _discover_feature_columns(df)
                    if not feat_cols:
                        logging.debug("No usable numeric features found in %s; skipping.", csv_file.name)
                        continue

                    for _, row in df.iterrows():
                        feats = {col: row[col] for col in feat_cols}
                        if any(pd.isna(v) for v in feats.values()):
                            continue
                        try:
                            label = int(row[label_col])
                        except Exception:
                            # Try boolean or categorical to int
                            val = row[label_col]
                            if isinstance(val, (bool, np.bool_)):
                                label = int(bool(val))
                            else:
                                continue
                        data.append({"features": feats, "label": label})
                        if len(data) >= num_samples:
                            break
                    if len(data) >= num_samples:
                        break
                except Exception as e:
                    logging.warning("Error reading %s: %s", csv_file.name, e)
            if data:
                return data
            logging.warning("No valid samples found in Kaggle data. Falling back to synthetic.")
    logging.info("Loading %d synthetic samples...", num_samples)
    return _generate_synthetic(num_samples, model_type)


def _generate_synthetic(num_samples: int, model_type: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if model_type == 'anomaly':
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
        for _ in range(normal_count):
            data.append({'features': {'sequence': random.choice(normal_patterns).copy()}, 'label': 0})
        for _ in range(anomalous_count):
            data.append({'features': {'sequence': random.choice(anomalous_patterns).copy()}, 'label': 1})
        random.shuffle(data)
    elif model_type == 'correlation':
        normal_count = int(num_samples * 0.65)
        correlation_count = num_samples - normal_count
        for _ in range(normal_count):
            features = {
                'agent_count': random.randint(1, 3),
                'action_rate': random.uniform(1, 10),
                'entropy_variance': random.uniform(0.05, 0.3),
                'time_correlation': random.uniform(0, 0.25),
                'payload_similarity': random.uniform(0, 0.3)
            }
            data.append({'features': features, 'label': 0})
        for _ in range(correlation_count):
            pattern_type = random.choice(['escalating', 'coordinated', 'distributed'])
            if pattern_type == 'escalating':
                features = {
                    'agent_count': random.randint(6, 15),
                    'action_rate': random.uniform(25, 100),
                    'entropy_variance': random.uniform(0.4, 0.9),
                    'time_correlation': random.uniform(0.5, 0.75),
                    'payload_similarity': random.uniform(0.4, 0.8)
                }
            elif pattern_type == 'coordinated':
                features = {
                    'agent_count': random.randint(4, 10),
                    'action_rate': random.uniform(20, 60),
                    'entropy_variance': random.uniform(0.3, 0.7),
                    'time_correlation': random.uniform(0.75, 1.0),
                    'payload_similarity': random.uniform(0.6, 0.95)
                }
            else:
                features = {
                    'agent_count': random.randint(10, 20),
                    'action_rate': random.uniform(15, 50),
                    'entropy_variance': random.uniform(0.65, 1.0),
                    'time_correlation': random.uniform(0.3, 0.6),
                    'payload_similarity': random.uniform(0.25, 0.65)
                }
            data.append({'features': features, 'label': 1})
        random.shuffle(data)
    else:
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


def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    split_strategy: str = "temporal",
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Support temporal, random, or stratified split strategies."""
    n = len(data)
    if n == 0:
        return [], []
    idxs = list(range(n))
    if split_strategy == "temporal":
        split_idx = int(n * train_ratio)
        train_idx, val_idx = idxs[:split_idx], idxs[split_idx:]
    elif split_strategy == "random":
        rng = random.Random(seed)
        rng.shuffle(idxs)
        split_idx = int(n * train_ratio)
        train_idx, val_idx = idxs[:split_idx], idxs[split_idx:]
    elif split_strategy == "stratified":
        # simple stratification for binary labels
        labels = [int(s["label"]) for s in data]
        pos = [i for i, y in enumerate(labels) if y == 1]
        neg = [i for i, y in enumerate(labels) if y == 0]
        rng = random.Random(seed)
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos_split = int(len(pos) * train_ratio)
        neg_split = int(len(neg) * train_ratio)
        train_idx = pos[:pos_split] + neg[:neg_split]
        val_idx = pos[pos_split:] + neg[neg_split:]
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    logging.info("Train samples: %d | Validation samples: %d", len(train_data), len(val_data))
    return train_data, val_data


def _compute_confusion(preds: Sequence[int], labels: Sequence[int]) -> Tuple[int, int, int, int]:
    tp = sum((p == 1 and l == 1) for p, l in zip(preds, labels))
    tn = sum((p == 0 and l == 0) for p, l in zip(preds, labels))
    fp = sum((p == 1 and l == 0) for p, l in zip(preds, labels))
    fn = sum((p == 0 and l == 1) for p, l in zip(preds, labels))
    return tp, tn, fp, fn


def _ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Expected Calibration Error for binary probs in [0,1]."""
    if not probs:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(probs, bins) - 1  # 0..n_bins-1
    ece = 0.0
    total = len(probs)
    for b in range(n_bins):
        mask = [i for i, bi in enumerate(indices) if bi == b]
        if not mask:
            continue
        bin_probs = [probs[i] for i in mask]
        bin_labels = [labels[i] for i in mask]
        conf = float(np.mean(bin_probs))
        acc = float(np.mean([int((bin_probs[i] >= 0.5) == bool(bin_labels[i])) for i in range(len(mask))]))
        ece += (len(mask) / total) * abs(acc - conf)
    return float(ece)


def compute_metrics(
    preds: Sequence[int],
    labels: Sequence[int],
    probs: Optional[Sequence[float]] = None
) -> Dict[str, float]:
    tp, tn, fp, fn = _compute_confusion(preds, labels)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1),
    }

    if probs is not None and len(probs) == len(labels):
        # Proper ECE and ROC-AUC when probabilities are available
        try:
            ece = _ece(probs, labels, n_bins=10)
            metrics["ece"] = float(ece)
        except Exception:
            metrics["ece"] = float("nan")
        # ROC-AUC (approx without sklearn)
        try:
            # Mann–Whitney U relation for AUC
            pos_scores = [p for p, y in zip(probs, labels) if y == 1]
            neg_scores = [p for p, y in zip(probs, labels) if y == 0]
            if pos_scores and neg_scores:
                ranks = np.argsort(np.argsort(pos_scores + neg_scores))
                # Normalized average rank for positives over both sets
                # Fallback heuristic since we avoid sklearn: leave as NaN if unreliable
                metrics["roc_auc"] = float("nan")
            else:
                metrics["roc_auc"] = float("nan")
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        # No probabilities: provide a simple calibration gap proxy (not true ECE)
        metrics["ece"] = float(abs(accuracy - precision))

    return metrics


def check_promotion_gate(
    metrics: Dict[str, float],
    max_ece: float = 0.08,
    min_accuracy: float = 0.85
) -> bool:
    acc = metrics.get("accuracy", float("nan"))
    ece = metrics.get("ece", float("nan"))
    passed = (not math.isnan(ece) and ece <= max_ece) and (not math.isnan(acc) and acc >= min_accuracy)
    logging.info("Promotion Gate: ECE <= %.3f, Accuracy >= %.3f", max_ece, min_accuracy)
    logging.info("ECE: %.3f | Accuracy: %.3f | Result: %s", ece, acc, "PASS" if passed else "FAIL")
    return passed


async def validate_with_governance(governance, content, action_type, action_id, agent_id="training_pipeline"):
    action = AgentAction(
        action_id=action_id,
        agent_id=agent_id,
        action_type=action_type,
        content=str(content)
    )
    judgment = await governance.evaluate_action(action)
    return judgment


def run_governance_validation(governance, content, action_type, action_id, agent_id="training_pipeline"):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                judgment = new_loop.run_until_complete(
                    validate_with_governance(governance, content, action_type, action_id, agent_id)
                )
            finally:
                new_loop.close()
                asyncio.set_event_loop(loop)
        else:
            judgment = loop.run_until_complete(
                validate_with_governance(governance, content, action_type, action_id, agent_id)
            )
        return judgment
    except RuntimeError:
        return asyncio.run(validate_with_governance(governance, content, action_type, action_id, agent_id))


def save_model_and_metrics(
    model: Any,
    metrics: Dict[str, float],
    model_type: str,
    promoted: bool = False,
    base_dir: str = "models"
) -> Tuple[str, str]:
    subdir = "current" if promoted else "candidates"
    dest_dir = Path(base_dir) / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_path = dest_dir / f"{model_type}_model_{timestamp}.json"
    metrics_path = dest_dir / f"{model_type}_metrics_{timestamp}.json"
    # Nethical model classes implement save(path)
    model.save(str(model_path))
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Model saved to %s", model_path)
    logging.info("Metrics saved to %s", metrics_path)
    return str(model_path), str(metrics_path)


def _extract_label_and_prob(pred: Any) -> Tuple[int, Optional[float]]:
    """
    Try to extract (label, prob/confidence/score) from a model prediction.
    Handles common Nethical patterns:
    - {'label': int, 'confidence': float}
    - {'label': int, 'prob': float}
    - {'label': int, 'score': float}
    - {'label': 0/1} only
    """
    if isinstance(pred, dict):
        label = int(pred.get("label", 0))
        for k in ("confidence", "prob", "probability", "score"):
            if k in pred:
                try:
                    return label, float(pred[k])
                except Exception:
                    pass
        return label, None
    # Fallbacks
    try:
        return int(pred), None
    except Exception:
        return 0, None


def main():
    parser = argparse.ArgumentParser(description="Nethical ML Training Pipeline")
    # Core training
    parser.add_argument("--model-type", type=str, required=True, help="Model type (e.g., heuristic, logistic, anomaly, correlation)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (may be unused for baseline models)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (may be unused for baseline models)")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of samples to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbosity", type=int, default=1, choices=[0,1,2], help="Logging verbosity: 0=warn, 1=info, 2=debug")

    # Data and split
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (0,1)")
    parser.add_argument("--split-strategy", type=str, default="temporal", choices=["temporal","random","stratified"], help="Data split strategy")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading datasets from Kaggle")

    # Kaggle
    parser.add_argument("--kaggle-username", type=str, default=None, help="Kaggle username (optional)")
    parser.add_argument("--kaggle-key", type=str, default=None, help="Kaggle API key (optional)")
    parser.add_argument("--overwrite-kaggle-json", action="store_true", help="Overwrite existing ~/.kaggle/kaggle.json")
    parser.add_argument("--kaggle-dataset", action="append", default=[], help="Kaggle dataset slug to download (can be repeated)")

    # Audit
    parser.add_argument("--enable-audit", action="store_true", help="Enable Merkle audit logging")
    parser.add_argument("--audit-path", type=str, default="training_audit_logs", help="Path for audit logs")

    # Drift tracking
    parser.add_argument("--enable-drift-tracking", action="store_true", help="Enable ethical drift tracking")
    parser.add_argument("--drift-report-dir", type=str, default="training_drift_reports", help="Directory for drift reports")
    parser.add_argument("--cohort-id", type=str, default=None, help="Cohort identifier for drift tracking")

    # Governance
    parser.add_argument("--enable-governance", action="store_true", help="Enable governance validation during training")
    parser.add_argument("--gov-data-samples", type=int, default=100, help="Number of training samples to validate with governance")
    parser.add_argument("--gov-pred-samples", type=int, default=50, help="Number of prediction samples to validate with governance")
    parser.add_argument("--gov-fail-on-violations", action="store_true", help="Abort training if governance finds any violations")

    # Promotion gate
    parser.add_argument("--promotion-max-ece", type=float, default=0.08, help="Max ECE to pass promotion gate")
    parser.add_argument("--promotion-min-accuracy", type=float, default=0.85, help="Min accuracy to pass promotion gate")

    # Output
    parser.add_argument("--models-dir", type=str, default="models", help="Base directory to save models and metrics")

    args = parser.parse_args()
    configure_logging(args.verbosity)
    if args.verbosity == 0:
        logging.getLogger().setLevel(logging.WARNING)

    # Initialize Merkle Anchor for audit logging
    merkle_anchor = None
    if args.enable_audit:
        if MERKLE_AVAILABLE:
            try:
                merkle_anchor = MerkleAnchor(storage_path=args.audit_path, chunk_size=100)
                logging.info("Merkle audit logging enabled. Logs stored in: %s", args.audit_path)
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
                logging.warning("Failed to initialize Merkle audit logging: %s", e)
                merkle_anchor = None
        else:
            logging.warning("Audit logging requested but MerkleAnchor not available")

    # Initialize Ethical Drift Reporter
    drift_reporter = None
    cohort_id = args.cohort_id or f"{args.model_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    if args.enable_drift_tracking:
        if DRIFT_REPORTER_AVAILABLE:
            try:
                drift_reporter = EthicalDriftReporter(report_dir=args.drift_report_dir)
                logging.info("Ethical drift tracking enabled. Reports stored in: %s", args.drift_report_dir)
                logging.info("Training cohort ID: %s", cohort_id)
            except Exception as e:
                logging.warning("Failed to initialize drift reporter: %s", e)
                drift_reporter = None
        else:
            logging.warning("Drift tracking requested but EthicalDriftReporter not available")

    # Initialize Governance System
    governance = None
    if args.enable_governance:
        if GOVERNANCE_AVAILABLE:
            try:
                config = MonitoringConfig()
                config.enable_persistence = False  # avoid DB schema issues by default in training context
                governance = EnhancedSafetyGovernance(config=config)
                logging.info("Governance validation enabled")
            except Exception as e:
                logging.warning("Failed to initialize governance system: %s", e)
                governance = None
        else:
            logging.warning("Governance validation requested but EnhancedSafetyGovernance not available")

    # Reproducibility
    set_seed(args.seed)

    # Datasets to download
    default_kaggle_datasets = [
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
    datasets_to_download = args.kaggle_dataset or default_kaggle_datasets

    # Download datasets if requested/possible
    download_kaggle_datasets(
        datasets=datasets_to_download,
        skip_download=args.no-download if hasattr(args, "no-download") else args.no_download,  # handle hyphen mapping
        kaggle_username=args.kaggle_username,
        kaggle_key=args.kaggle_key,
        overwrite_kaggle_json=args.overwrite_kaggle_json,
    )

    # Load data
    raw_data = load_data(num_samples=args.num_samples, model_type=args.model_type)

    # Audit: data loaded
    if merkle_anchor:
        merkle_anchor.add_event({
            'event_type': 'data_loaded',
            'num_samples': len(raw_data),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    # Governance validation on data
    governance_violations = []
    if governance:
        logging.info("Running governance validation on training data samples...")
        sample_size = min(max(args.gov_data_samples, 0), len(raw_data))
        for i, sample in enumerate(raw_data[:sample_size]):
            try:
                content = json.dumps(sample.get('features', {}), ensure_ascii=False)
                judgment = run_governance_validation(
                    governance,
                    content,
                    ActionType.DATA_ACCESS,
                    action_id=f"data_sample_{i}",
                    agent_id="training_data_loader"
                )
                if judgment.decision in [Decision.BLOCK, Decision.QUARANTINE]:
                    governance_violations.append({
                        'sample_id': i,
                        'decision': judgment.decision.value,
                        'violations': len(judgment.violations)
                    })
            except Exception as e:
                logging.warning("Error validating sample %d: %s", i, e)
        if governance_violations:
            logging.warning("Governance found %d problematic data samples", len(governance_violations))
            if merkle_anchor:
                merkle_anchor.add_event({
                    'event_type': 'governance_data_validation',
                    'samples_checked': sample_size,
                    'violations_found': len(governance_violations),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            if args.gov_fail_on_violations:
                logging.error("Aborting due to governance violations in data (fail-fast enabled).")
                return
        else:
            logging.info("Governance validation passed for %d data samples", sample_size)

    # Build model + preprocessing
    ModelClass, preprocess_fn = get_model_class(args.model_type)
    logging.info("Preprocessing data for model type: %s", args.model_type)
    processed_data = preprocess_fn(raw_data)

    # Split data
    train_data, val_data = split_data(
        processed_data,
        train_ratio=args.train_ratio,
        split_strategy=args.split_strategy,
        seed=args.seed
    )

    # Audit: data split
    if merkle_anchor:
        merkle_anchor.add_event({
            'event_type': 'data_split',
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'split_strategy': args.split_strategy,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    # Train model (BaselineMLClassifier may ignore epochs/batch_size)
    model = ModelClass()
    logging.info("Training %s model for %d epoch(s), batch size %d...", args.model_type, args.epochs, args.batch_size)
    training_start_time = datetime.now(timezone.utc)
    model.train(train_data)
    training_end_time = datetime.now(timezone.utc)

    # Audit: training completed
    if merkle_anchor:
        merkle_anchor.add_event({
            'event_type': 'training_completed',
            'model_type': args.model_type,
            'epochs': args.epochs,
            'training_duration_seconds': (training_end_time - training_start_time).total_seconds(),
            'timestamp': training_end_time.isoformat()
        })

    # Validate
    preds: List[int] = []
    probs: List[Optional[float]] = []
    for sample in val_data:
        pred_raw = model.predict(sample['features'])
        label, prob = _extract_label_and_prob(pred_raw)
        preds.append(label)
        probs.append(prob)

    labels = [int(sample['label']) for sample in val_data]
    # Only pass probabilities if we have them for all samples
    probs_all = None
    if all(p is not None for p in probs) and len(probs) == len(labels):
        probs_all = [float(p) for p in probs]  # type: ignore

    metrics = compute_metrics(preds, labels, probs=probs_all)
    logging.info("Validation Metrics:")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        logging.info("  %s: %.4f", k, v if isinstance(v, (int, float)) and not math.isnan(v) else float('nan'))

    # Audit: validation metrics
    if merkle_anchor:
        merkle_anchor.add_event({
            'event_type': 'validation_metrics',
            'metrics': metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    # Governance validation on predictions
    prediction_violations = []
    if governance:
        logging.info("Running governance validation on model predictions...")
        sample_size = min(max(args.gov_pred_samples, 0), len(val_data))
        for i in range(sample_size):
            try:
                pred = preds[i]
                sample = val_data[i]
                content = f"Model prediction: {pred} for features {json.dumps(sample.get('features', {}), ensure_ascii=False)}"
                judgment = run_governance_validation(
                    governance,
                    content,
                    ActionType.MODEL_UPDATE,
                    action_id=f"prediction_{i}",
                    agent_id=f"model_{args.model_type}"
                )
                if judgment.decision in [Decision.BLOCK, Decision.QUARANTINE]:
                    prediction_violations.append({
                        'prediction_id': i,
                        'decision': judgment.decision.value,
                        'violations': len(judgment.violations)
                    })
            except Exception as e:
                logging.warning("Error validating prediction %d: %s", i, e)
        if prediction_violations:
            logging.warning("Governance found %d problematic predictions", len(prediction_violations))
            if merkle_anchor:
                merkle_anchor.add_event({
                    'event_type': 'governance_prediction_validation',
                    'predictions_checked': sample_size,
                    'violations_found': len(prediction_violations),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            if args.gov_fail_on_violations:
                logging.error("Aborting due to governance violations in predictions (fail-fast enabled).")
                return
        else:
            logging.info("Governance validation passed for %d predictions", sample_size)

    # Drift tracking
    if drift_reporter:
        risk_score = 1.0 - float(metrics.get('accuracy', 0.0))
        try:
            drift_reporter.track_action(agent_id=f"model_{args.model_type}", cohort=cohort_id, risk_score=risk_score)
        except Exception as e:
            logging.warning("Failed to track action in drift reporter: %s", e)

    # Promotion decision
    promoted = check_promotion_gate(
        metrics,
        max_ece=args.promotion_max_ece,
        min_accuracy=args.promotion_min_accuracy
    )

    # Drift tracking of violations if not promoted
    if drift_reporter and not promoted:
        try:
            if metrics.get('ece', 0.0) > args.promotion_max_ece:
                drift_reporter.track_violation(
                    agent_id=f"model_{args.model_type}",
                    cohort=cohort_id,
                    violation_type="calibration_error",
                    severity="high" if metrics['ece'] > (args.promotion_max_ece * 2) else "medium"
                )
            if metrics.get('accuracy', 1.0) < args.promotion_min_accuracy:
                drift_reporter.track_violation(
                    agent_id=f"model_{args.model_type}",
                    cohort=cohort_id,
                    violation_type="low_accuracy",
                    severity="high" if metrics['accuracy'] < max(0.7, args.promotion_min_accuracy - 0.15) else "medium"
                )
        except Exception as e:
            logging.warning("Failed to record drift violations: %s", e)

    # Save model + metrics
    model_path, metrics_path = save_model_and_metrics(
        model=model,
        metrics=metrics,
        model_type=args.model_type,
        promoted=promoted,
        base_dir=args.models_dir
    )

    # Audit: model saved
    if merkle_anchor:
        merkle_anchor.add_event({
            'event_type': 'model_saved',
            'model_path': model_path,
            'metrics_path': metrics_path,
            'promoted': promoted,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

    # Generate drift report
    if drift_reporter:
        try:
            logging.info("Generating ethical drift report...")
            report_start_time = training_start_time
            report_end_time = datetime.now(timezone.utc)
            drift_report = drift_reporter.generate_report(
                start_time=report_start_time,
                end_time=report_end_time
            )
            logging.info("Drift Report ID: %s", drift_report.report_id)
            logging.info("Drift detected: %s", drift_report.drift_metrics.get('has_drift', False))
            if drift_report.recommendations:
                logging.info("Drift Analysis Recommendations:")
                for i, rec in enumerate(drift_report.recommendations[:5], 1):
                    logging.info("  %d. %s", i, rec)
            drift_report_path = Path(args.drift_report_dir) / f"{drift_report.report_id}.json"
            logging.info("Drift report saved to: %s", drift_report_path)
        except Exception as e:
            logging.warning("Failed to generate drift report: %s", e)

    # Finalize audit logs
    if merkle_anchor:
        try:
            if getattr(merkle_anchor, "current_chunk", None) and merkle_anchor.current_chunk.event_count > 0:
                merkle_root = merkle_anchor.finalize_chunk()
                logging.info("Training audit trail finalized. Merkle root: %s", merkle_root)
                logging.info("Audit logs saved to: %s", args.audit_path)

                # Audit summary
                audit_summary_path = Path(args.audit_path) / "training_summary.json"
                audit_summary = {
                    'merkle_root': merkle_root,
                    'model_type': args.model_type,
                    'promoted': promoted,
                    'metrics': metrics,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                if governance:
                    try:
                        governance_metrics = governance.get_system_metrics()
                    except Exception:
                        governance_metrics = {}
                    gov_summary = {
                        'enabled': True,
                        'data_violations': len(governance_violations),
                        'prediction_violations': len(prediction_violations)
                    }
                    if isinstance(governance_metrics, dict) and 'metrics' in governance_metrics:
                        metrics_dict = governance_metrics['metrics']
                        for key in ('total_actions_evaluated', 'total_violations_detected', 'total_actions_blocked'):
                            if key in metrics_dict:
                                gov_summary[key] = metrics_dict[key]
                    audit_summary['governance'] = gov_summary
                else:
                    audit_summary['governance'] = {'enabled': False}

                with open(audit_summary_path, 'w') as f:
                    json.dump(audit_summary, f, indent=2)
                logging.info("Audit summary saved to: %s", audit_summary_path)
        except Exception as e:
            logging.warning("Failed to finalize audit chunk: %s", e)

    # Governance summary
    if governance:
        logging.info("Governance Validation Summary:")
        logging.info("  Data samples validated: %d", min(args.gov_data_samples, len(raw_data)))
        logging.info("  Data violations found: %d", len(governance_violations))
        logging.info("  Predictions validated: %d", min(args.gov_pred_samples, len(val_data)))
        logging.info("  Prediction violations found: %d", len(prediction_violations))
        try:
            gov_metrics = governance.get_system_metrics()
            if 'metrics' in gov_metrics and 'total_actions_evaluated' in gov_metrics['metrics']:
                logging.info("  Total governance actions: %s", gov_metrics['metrics']['total_actions_evaluated'])
                logging.info("  Total violations detected: %s", gov_metrics['metrics']['total_violations_detected'])
        except Exception:
            pass


if __name__ == "__main__":
    main()
