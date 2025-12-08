#!/usr/bin/env python3
"""Testing Script for Nethical ML Models

Implements the testing pipeline described in TrainTestPipeline.md:
- Load trained model
- Run on test dataset
- Compute comprehensive metrics (Precision, Recall, F1, ROC-AUC, ECE)
- Compare with rule-based baseline
- Generate evaluation report

Enhancements:
- CLI + reproducibility (seed)
- True rule-based baseline from rule_score thresholds (deny/warn/allow)
- ROC-AUC with sklearn or pure-Python fallback
- Robust ECE computation with configurable bins and score normalization
- Verbose logging, progress indicators (optional), and safer calibration binning
- CSV/JSON/JSONL output options for predictions and richer report
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import glob
import random

# Optional tqdm for progress display
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Optional sklearn for ROC-AUC; we also provide a pure-Python fallback
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

# Add parent directory to path so "nethical" imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.core import MLShadowClassifier, MLModelType


# ----------------------------
# Utilities and Metric Helpers
# ----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def normalize_score(score: float) -> float:
    """Clamp score to [0, 1] for calibration/ROC computations."""
    if score is None:
        return 0.0
    if score != score:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(score)))


def compute_confusion_components(
    y_true: List[int], y_pred: List[int]
) -> Tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
        else:
            tn += 1
    return tp, tn, fp, fn


def safe_div(n: float, d: float) -> float:
    return (n / d) if d else 0.0


def compute_basic_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp, tn, fp, fn = compute_confusion_components(y_true, y_pred)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = (
        safe_div(2 * precision * recall, precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "confusion_matrix": {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
        },
        "total_predictions": tp + tn + fp + fn,
    }


def compute_roc_auc(y_true: List[int], y_score: List[float]) -> Optional[float]:
    """Compute ROC-AUC with sklearn if available, else pure-Python fallback.
    Returns None if only one class present or if computation fails."""
    # Must have both classes present
    if len(set(y_true)) < 2:
        return None
    try:
        if roc_auc_score is not None:
            return float(roc_auc_score(y_true, y_score))
    except Exception:
        pass

    # Pure-Python fallback (Mann-Whitney U / rank-based AUC)
    try:
        # Sort by score ascending and compute rank sum for positives
        pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return None

        # Compute average ranks (1-based)
        # Handle ties by averaging ranks for tied groups
        ranks = [0.0] * len(pairs)
        i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
                j += 1
            avg_rank = (i + j + 2) / 2.0  # 1-based average
            for k in range(i, j + 1):
                ranks[k] = avg_rank
            i = j + 1

        # Sum ranks for positives
        rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
        # Mann-Whitney U for positives
        U = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
        auc = U / (n_pos * n_neg)
        return float(auc)
    except Exception:
        return None


def compute_ece(y_true: List[int], y_prob: List[float], n_bins: int = 10) -> float:
    """Expected Calibration Error using equal-width bins on [0,1].
    y_prob should be probabilities (or normalized scores)."""
    if not y_prob:
        return 0.0
    # normalize
    probs = [normalize_score(p) for p in y_prob]
    total = len(probs)
    if total == 0:
        return 0.0

    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # include right edge only for last bin
        if i < n_bins - 1:
            idxs = [k for k, p in enumerate(probs) if lo <= p < hi]
        else:
            idxs = [k for k, p in enumerate(probs) if lo <= p <= hi]
        if not idxs:
            continue
        bin_conf = sum(probs[k] for k in idxs) / len(idxs)
        bin_acc = sum(1 if y_true[k] == 1 else 0 for k in idxs) / len(idxs)
        ece += (len(idxs) / total) * abs(bin_acc - bin_conf)
    return float(ece)


# ----------------------------
# Loading
# ----------------------------


def load_model(model_path: str):
    """Load trained model from file.

    Args:
        model_path: Path to model JSON file

    Returns:
        Tuple of (classifier_or_baseline, model_metadata)
    """
    logging.info(f"Loading model from {model_path}...")
    with open(model_path, "r") as f:
        model_data = json.load(f)

    model_type = model_data.get("model_type", "shadow")
    if model_type == "baseline":
        # Load BaselineMLClassifier (if used in this project)
        from nethical.mlops.baseline import BaselineMLClassifier

        classifier = BaselineMLClassifier.load(model_path)
        logging.info(f"✓ Model loaded: baseline")
        logging.info(f"  Timestamp: {model_data.get('timestamp', 'unknown')}")
        return classifier, model_data

    # MLShadowClassifier
    classifier = MLShadowClassifier(
        model_type=MLModelType(model_data["model_type"]),
        score_agreement_threshold=model_data.get("score_agreement_threshold", 0.1),
        storage_path=str(Path(model_path).parent),
    )
    # Restore feature weights if present
    if "feature_weights" in model_data:
        classifier.feature_weights = model_data["feature_weights"]

    logging.info(f"✓ Model loaded: {model_data['model_type']}")
    logging.info(f"  Timestamp: {model_data.get('timestamp', 'unknown')}")
    return classifier, model_data


def find_latest_model(model_dir: str) -> str:
    """Find the latest model in the directory.

    Args:
        model_dir: Directory to search

    Returns:
        Path to latest model file
    """
    model_files = glob.glob(os.path.join(model_dir, "model_*.json"))
    # Filter out metrics files
    model_files = [f for f in model_files if not f.endswith("_metrics.json")]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    # Sort by filename (which includes timestamp)
    latest_model = sorted(model_files)[-1]
    return latest_model


# ----------------------------
# Data
# ----------------------------


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    logging.info(f"Loading dataset from {data_path}...")
    with open(data_path, "r") as f:
        data = json.load(f)
    logging.info(f"✓ Loaded {len(data)} samples")
    return data


def split_train_test(
    data: List[Dict[str, Any]], test_ratio: float, seed: int, stratify: bool = True
) -> List[Dict[str, Any]]:
    """Return test split (for run-only testing we just slice)."""
    if not data:
        return []
    n = len(data)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)

    if stratify:
        # Attempt stratified split if labels present
        try:
            pos = [i for i in idxs if int(data[i]["label"]) == 1]
            neg = [i for i in idxs if int(data[i]["label"]) == 0]
            n_test = max(1, int(n * test_ratio))
            n_pos_test = int(len(pos) * test_ratio)
            n_neg_test = n_test - n_pos_test
            pos_test = pos[:n_pos_test]
            neg_test = neg[:n_neg_test]
            test_idxs = pos_test + neg_test
            # If not enough due to rounding, top up
            if len(test_idxs) < n_test:
                remaining = [i for i in idxs if i not in test_idxs]
                test_idxs += remaining[: (n_test - len(test_idxs))]
            test_idxs = test_idxs[:n_test]
        except Exception:
            # Fallback to simple slice
            n_test = max(1, int(n * test_ratio))
            test_idxs = idxs[-n_test:]
    else:
        n_test = max(1, int(n * test_ratio))
        test_idxs = idxs[-n_test:]

    test_data = [data[i] for i in test_idxs]
    logging.info(
        f"✓ Prepared test set of {len(test_data)} samples (~{int(test_ratio*100)}%)"
    )
    return test_data


# ----------------------------
# Evaluation
# ----------------------------


def eval_rule_baseline(
    test_data: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate a pure rule-based baseline using rule_score thresholds:
    deny if > 0.6, warn if > 0.4, else allow.
    Positive class = violation if (deny or warn).
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []
    log_rows: List[Dict[str, Any]] = []

    iterator = tqdm(test_data, desc="Baseline eval", leave=False) if tqdm else test_data
    for sample in iterator:
        label = int(sample["label"])
        rule_score = float(sample.get("rule_score", 0.0))
        rule_class = (
            "deny" if rule_score > 0.6 else "warn" if rule_score > 0.4 else "allow"
        )
        predicted_violation = 1 if rule_class in ("deny", "warn") else 0

        y_true.append(label)
        y_pred.append(predicted_violation)
        y_score.append(normalize_score(rule_score))  # treat as risk prob in [0,1]

        log_rows.append(
            {
                "event_id": sample.get("event_id"),
                "actual_label": label,
                "ml_prediction": rule_class,  # keep field name for compatibility
                "ml_score": rule_score,
                "rule_score": rule_score,
                "correct": (predicted_violation == label),
            }
        )

    base_metrics = compute_basic_metrics(y_true, y_pred)
    base_metrics["expected_calibration_error"] = compute_ece(y_true, y_score, n_bins=10)
    base_metrics["roc_auc"] = compute_roc_auc(y_true, y_score)
    base_metrics.setdefault("score_agreement_rate", 0.0)
    base_metrics.setdefault("classification_agreement_rate", 0.0)
    return base_metrics, log_rows


def evaluate_on_test_set(
    classifier, test_data: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run comprehensive evaluation on test set.

    Supports:
      - BaselineMLClassifier (from nethical.mlops.baseline)
      - MLShadowClassifier (from nethical.core)

    Returns:
      (metrics dict, predictions_log list)
    """
    logging.info("Running evaluation on test set...")
    from nethical.mlops.baseline import BaselineMLClassifier

    is_baseline = isinstance(classifier, BaselineMLClassifier)

    predictions_log: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []

    iterator = tqdm(test_data, desc="Model eval", leave=False) if tqdm else test_data

    if is_baseline:
        # Evaluate using the BaselineMLClassifier API
        for sample in iterator:
            label = int(sample["label"])
            pred = classifier.predict(sample["features"])
            pred_label = pred.get("label")
            pred_score = float(pred.get("score", 0.0))

            # Map predicted label to binary violation target
            # Assume 'deny'/'warn' are violations; if baseline returns 0/1 use that
            if isinstance(pred_label, str):
                predicted_violation = 1 if pred_label in ("deny", "warn") else 0
            else:
                predicted_violation = int(pred_label)

            y_true.append(label)
            y_pred.append(predicted_violation)
            y_score.append(normalize_score(pred_score))

            predictions_log.append(
                {
                    "event_id": sample.get("event_id"),
                    "actual_label": label,
                    "ml_prediction": pred_label,
                    "ml_score": pred_score,
                    "rule_score": float(sample.get("rule_score", 0.0)),
                    "correct": (predicted_violation == label),
                }
            )

        # Prefer classifier's internal metrics if available; augment with ROC-AUC
        try:
            metrics = classifier.compute_metrics(y_pred, y_true)
            # Map to unified format
            formatted_metrics = {
                "total_predictions": metrics.get("total_predictions", len(y_true)),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1_score": metrics.get("f1_score", 0.0),
                "accuracy": metrics.get("accuracy", 0.0),
                "expected_calibration_error": metrics.get(
                    "ece", compute_ece(y_true, y_score)
                ),
                "confusion_matrix": {
                    "true_positives": metrics.get("true_positives", 0),
                    "true_negatives": metrics.get("true_negatives", 0),
                    "false_positives": metrics.get("false_positives", 0),
                    "false_negatives": metrics.get("false_negatives", 0),
                },
                "score_agreement_rate": 0.0,
                "classification_agreement_rate": 0.0,
            }
        except Exception:
            # Compute metrics ourselves
            formatted_metrics = compute_basic_metrics(y_true, y_pred)
            formatted_metrics["expected_calibration_error"] = compute_ece(
                y_true, y_score
            )
            formatted_metrics["score_agreement_rate"] = 0.0
            formatted_metrics["classification_agreement_rate"] = 0.0

        formatted_metrics["roc_auc"] = compute_roc_auc(y_true, y_score)

    else:
        # MLShadowClassifier flow
        # Reset metrics if the classifier exposes a metrics object
        try:
            m = classifier.metrics
            m.true_positives = 0
            m.true_negatives = 0
            m.false_positives = 0
            m.false_negatives = 0
            m.total_predictions = 0
            m.score_agreement_count = 0
            m.classification_agreement_count = 0
            # Ensure calibration bins are present (optional)
            if getattr(m, "calibration_bins", None) is not None:
                for k in m.calibration_bins.keys():
                    m.calibration_bins[k]["total"] = 0
                    m.calibration_bins[k]["correct"] = 0
        except Exception:
            pass

        for sample in iterator:
            label = int(sample["label"])
            rule_score = float(sample.get("rule_score", 0.0))
            rule_class = (
                "deny" if rule_score > 0.6 else "warn" if rule_score > 0.4 else "allow"
            )

            prediction = classifier.predict(
                agent_id=sample.get("agent_id"),
                action_id=sample.get("event_id"),
                features=sample["features"],
                rule_risk_score=rule_score,
                rule_classification=rule_class,
            )

            actual_violation = label == 1
            predicted_violation = prediction.ml_classification in ["deny", "warn"]

            y_true.append(label)
            y_pred.append(1 if predicted_violation else 0)

            # Prefer confidence if it represents probability; else use risk score
            prob_or_score = (
                prediction.ml_confidence
                if hasattr(prediction, "ml_confidence")
                else prediction.ml_risk_score
            )
            y_score.append(normalize_score(prob_or_score))

            # Log prediction
            predictions_log.append(
                {
                    "event_id": sample.get("event_id"),
                    "actual_label": label,
                    "ml_prediction": prediction.ml_classification,
                    "ml_score": float(prediction.ml_risk_score),
                    "rule_score": rule_score,
                    "correct": (predicted_violation == actual_violation),
                }
            )

            # If classifier exposes metrics, keep them consistent
            try:
                if predicted_violation and actual_violation:
                    classifier.metrics.true_positives += 1
                elif predicted_violation and not actual_violation:
                    classifier.metrics.false_positives += 1
                elif not predicted_violation and actual_violation:
                    classifier.metrics.false_negatives += 1
                else:
                    classifier.metrics.true_negatives += 1
                classifier.metrics.total_predictions += 1

                # Update calibration bins if available
                confidence = normalize_score(
                    getattr(prediction, "ml_confidence", prob_or_score)
                )
                if getattr(classifier.metrics, "calibration_bins", None) is not None:
                    # 10 bins [0.0,0.1) ... [0.9,1.0]
                    bin_lo = int(confidence * 10) / 10.0
                    bin_hi = 1.0 if bin_lo == 0.9 else round(bin_lo + 0.1, 1)
                    confidence_bin = f"{bin_lo:.1f}-{bin_hi:.1f}"
                    if confidence_bin in classifier.metrics.calibration_bins:
                        classifier.metrics.calibration_bins[confidence_bin][
                            "total"
                        ] += 1
                        # If predicted violation matches actual violation, count correct
                        classifier.metrics.calibration_bins[confidence_bin][
                            "correct"
                        ] += int(predicted_violation == actual_violation)
            except Exception:
                pass

        # Prefer classifier's report; augment with ROC-AUC if missing
        try:
            formatted_metrics = classifier.get_metrics_report()
            # Add ROC-AUC if not present
            if "roc_auc" not in formatted_metrics:
                formatted_metrics["roc_auc"] = compute_roc_auc(y_true, y_score)
            # If ECE missing, compute from scores
            if "expected_calibration_error" not in formatted_metrics:
                formatted_metrics["expected_calibration_error"] = compute_ece(
                    y_true, y_score
                )
        except Exception:
            # Compute ourselves
            formatted_metrics = compute_basic_metrics(y_true, y_pred)
            formatted_metrics["expected_calibration_error"] = compute_ece(
                y_true, y_score
            )
            formatted_metrics["score_agreement_rate"] = 0.0
            formatted_metrics["classification_agreement_rate"] = 0.0
            formatted_metrics["roc_auc"] = compute_roc_auc(y_true, y_score)

    # Pretty-print summary
    print("\nTest Set Performance:")
    print("-" * 60)
    print(f"Total Predictions: {formatted_metrics['total_predictions']}")
    print()
    print("Classification Metrics:")
    print(f"  Precision: {formatted_metrics['precision']:.3f}")
    print(f"  Recall: {formatted_metrics['recall']:.3f}")
    print(f"  F1 Score: {formatted_metrics['f1_score']:.3f}")
    print(f"  Accuracy: {formatted_metrics['accuracy']:.3f}")
    if formatted_metrics.get("roc_auc") is not None:
        print(f"  ROC-AUC:  {formatted_metrics['roc_auc']:.3f}")
    print()
    print("Calibration:")
    print(
        f"  Expected Calibration Error (ECE): {formatted_metrics['expected_calibration_error']:.3f}"
    )
    print()
    print("Confusion Matrix:")
    print(
        f"  True Positives:  {formatted_metrics['confusion_matrix']['true_positives']}"
    )
    print(
        f"  True Negatives:  {formatted_metrics['confusion_matrix']['true_negatives']}"
    )
    print(
        f"  False Positives: {formatted_metrics['confusion_matrix']['false_positives']}"
    )
    print(
        f"  False Negatives: {formatted_metrics['confusion_matrix']['false_negatives']}"
    )
    print()
    if not is_baseline:
        print("Agreement with Rule-Based System:")
        print(
            f"  Score Agreement: {formatted_metrics.get('score_agreement_rate', 0.0):.1%}"
        )
        print(
            f"  Classification Agreement: {formatted_metrics.get('classification_agreement_rate', 0.0):.1%}"
        )
    print("=" * 60)

    return formatted_metrics, predictions_log


def compare_with_baseline(
    ml_metrics: Dict[str, float], rule_baseline: Dict[str, float]
) -> None:
    """Compare ML model with rule-based baseline."""
    print("\nComparison with Rule-Based Baseline:")
    print("=" * 60)
    print(f"{'Metric':<30} {'Rule-Based':<15} {'ML Model':<15} {'Δ':<10}")
    print("-" * 60)

    metrics_to_compare = ["precision", "recall", "f1_score", "accuracy", "roc_auc"]
    for metric in metrics_to_compare:
        baseline_val = (
            rule_baseline.get(metric, 0.0)
            if rule_baseline.get(metric) is not None
            else 0.0
        )
        ml_val = (
            ml_metrics.get(metric, 0.0) if ml_metrics.get(metric) is not None else 0.0
        )
        delta = ml_val - baseline_val
        delta_str = f"{delta:+.3f}"
        print(
            f"{metric.replace('_', ' ').upper():<30} {baseline_val:<15.3f} {ml_val:<15.3f} {delta_str:<10}"
        )
    print("=" * 60)


# ----------------------------
# Reporting
# ----------------------------


def save_evaluation_report(
    ml_metrics: Dict[str, Any],
    predictions_log: List[Dict[str, Any]],
    output_dir: str = "./data/labeled_events",
    baseline_metrics: Optional[Dict[str, Any]] = None,
    save_preds: str = "json",
    max_preds: int = 100,
) -> str:
    """Save evaluation report to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
    report: Dict[str, Any] = {
        "timestamp": timestamp,
        "metrics": ml_metrics,
        "baseline_metrics": baseline_metrics or {},
        "predictions": predictions_log[
            :max_preds
        ],  # Save first N predictions for brevity
    }
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"✓ Evaluation report saved: {report_file}")

    # Optional detailed prediction dump
    if save_preds.lower() == "csv":
        csv_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        if predictions_log:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=predictions_log[0].keys())
                writer.writeheader()
                writer.writerows(predictions_log)
        logging.info(f"✓ Predictions CSV saved: {csv_path}")
    elif save_preds.lower() == "jsonl":
        jsonl_path = os.path.join(output_dir, f"predictions_{timestamp}.jsonl")
        with open(jsonl_path, "w") as f:
            for row in predictions_log:
                f.write(json.dumps(row) + "\n")
        logging.info(f"✓ Predictions JSONL saved: {jsonl_path}")
    # else 'json' already embedded in report

    return report_file


# ----------------------------
# CLI / Main
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nethical model testing pipeline")
    parser.add_argument(
        "--model", type=str, default="", help="Path to a model JSON file"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/current",
        help="Directory to search for latest model",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/labeled_events/training_data.json",
        help="Path to dataset JSON",
    )
    parser.add_argument(
        "--test-split", type=float, default=0.2, help="Test split ratio (0,1]"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for repeatability"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./data/labeled_events",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--no-baseline", action="store_true", help="Skip rule-based baseline comparison"
    )
    parser.add_argument(
        "--save-preds",
        type=str,
        default="json",
        choices=["json", "jsonl", "csv"],
        help="Format for saving predictions",
    )
    parser.add_argument(
        "--max-preds",
        type=int,
        default=100,
        help="Max predictions to embed in the main JSON report",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv)",
    )
    return parser.parse_args()


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    args = parse_args()
    configure_logging(args.verbose)
    set_seed(args.seed)

    print("=" * 60)
    print("NETHICAL ML TESTING PIPELINE")
    print("=" * 60)
    print("\nComprehensive model evaluation:")
    print("- Load trained model")
    print("- Run on test dataset")
    print("- Compute metrics (Precision, Recall, F1, ROC-AUC, ECE)")
    print("- Compare with rule-based baseline")
    print("- Generate evaluation report")
    print()

    # Step 1: Find and load model
    if args.model:
        model_path = args.model
    else:
        try:
            model_path = find_latest_model(args.model_dir)
        except FileNotFoundError:
            # Try candidates
            logging.warning(
                "No model found in %s, checking ./models/candidates ...", args.model_dir
            )
            try:
                model_path = find_latest_model("./models/candidates")
            except FileNotFoundError:
                print("✗ No models found. Please run train_model.py first.")
                sys.exit(1)

    classifier, model_metadata = load_model(model_path)

    # Step 2: Load and split test data
    try:
        full_data = load_dataset(args.data)
    except FileNotFoundError:
        print("✗ Dataset not found. Please run train_model.py first to generate data.")
        sys.exit(1)

    test_data = split_train_test(
        full_data, test_ratio=args.test_split, seed=args.seed, stratify=True
    )

    # Step 3: Evaluate on test set
    ml_metrics, predictions_log = evaluate_on_test_set(classifier, test_data)

    # Step 4: Rule-based baseline
    baseline_metrics = None
    if not args.no_baseline:
        baseline_metrics, _ = eval_rule_baseline(test_data)
        compare_with_baseline(ml_metrics, baseline_metrics)
    else:
        print("\nNote: Rule-based baseline comparison was skipped (--no-baseline).")

    # Step 5: Save evaluation report
    save_evaluation_report(
        ml_metrics=ml_metrics,
        predictions_log=predictions_log,
        output_dir=args.outdir,
        baseline_metrics=baseline_metrics,
        save_preds=args.save_preds,
        max_preds=args.max_preds,
    )

    # Final summary
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print("\nKey Findings:")
    if ml_metrics.get("f1_score", 0.0) >= 0.8:
        print("✓ Model shows strong performance (F1 ≥ 0.8)")
    elif ml_metrics.get("f1_score", 0.0) >= 0.6:
        print("⚠ Model shows moderate performance (0.6 ≤ F1 < 0.8)")
    else:
        print("✗ Model needs improvement (F1 < 0.6)")

    ece = ml_metrics.get("expected_calibration_error", 1.0)
    if ece <= 0.08:
        print("✓ Model is well-calibrated (ECE ≤ 0.08)")
    else:
        print("⚠ Model calibration needs improvement (ECE > 0.08)")
    if ml_metrics.get("roc_auc") is not None:
        print(f"ℹ ROC-AUC: {ml_metrics['roc_auc']:.3f}")
    print()


if __name__ == "__main__":
    main()
