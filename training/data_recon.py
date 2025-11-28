#!/usr/bin/env python3
"""
Data Reconnaissance Tool.

Scans datasets in datasets/datasets or data/external to provide intelligence on:
- Total samples and class balance (Safe vs Threat)
- Missing features critical for model training
- Warnings for unbalanced datasets or lack of adversarial examples

Does NOT remove toxic data. We need it for the models to learn.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [RECON] %(message)s")

# Expected features for different model types
EXPECTED_FEATURES = {
    "heuristic": {"violation_count", "severity_max", "recency_score", "frequency_score", "context_risk"},
    "logistic": {"violation_count", "severity_max", "recency_score", "frequency_score", "context_risk"},
    "anomaly": {"sequence"},
    "correlation": {"agent_count", "action_rate", "entropy_variance", "time_correlation", "payload_similarity"},
    "transformer": {"text"},
}

# Default directories to scan
DEFAULT_SCAN_DIRS = [
    "datasets/datasets",
    "data/external",
    "data/labeled_events",
]


def discover_label_column(df) -> Optional[str]:
    """Try to find a reasonable label column."""
    for cand in ["label", "target", "class", "is_threat", "is_toxic", "is_anomaly", "y"]:
        matches = [c for c in df.columns if c.lower() == cand]
        if matches:
            return matches[0]
    # Fallback: any column containing 'label'/'target'
    for c in df.columns:
        lc = c.lower()
        if "label" in lc or "target" in lc:
            return c
    return None


def discover_feature_columns(df) -> Set[str]:
    """Extract all potential feature columns from a DataFrame."""
    # Exclude obvious non-feature columns
    exclude_cols = {"id", "index", "unnamed", "timestamp", "date", "time"}
    feature_cols = set()

    for col in df.columns:
        col_lower = col.lower()
        if any(exc in col_lower for exc in exclude_cols):
            continue
        feature_cols.add(col)

    return feature_cols


def check_missing_features(available_features: Set[str], model_type: str = "heuristic") -> Set[str]:
    """Check which expected features are missing."""
    expected = EXPECTED_FEATURES.get(model_type, EXPECTED_FEATURES["heuristic"])
    # Normalize to lowercase for comparison
    available_lower = {f.lower() for f in available_features}
    expected_lower = {f.lower() for f in expected}
    return expected - {f for f in expected if f.lower() in available_lower}


def calculate_balance_ratio(threats: int, safe: int) -> Tuple[float, str]:
    """
    Calculate class balance ratio and determine status.

    Returns:
        Tuple of (ratio, status) where status is 'balanced', 'imbalanced', or 'critical'.
    """
    total = threats + safe
    if total == 0:
        return 0.0, "empty"

    minority = min(threats, safe)
    majority = max(threats, safe)

    if majority == 0:
        return 0.0, "single_class"

    ratio = minority / majority

    if ratio >= 0.4:
        return ratio, "balanced"
    elif ratio >= 0.1:
        return ratio, "imbalanced"
    else:
        return ratio, "critical"


def inspect_single_dataset(csv_path: Path, model_type: str = "heuristic") -> Dict:
    """
    Inspect a single dataset file.

    Returns:
        Dictionary with dataset statistics.
    """
    try:
        import pandas as pd
    except ImportError:
        return {"error": "pandas not installed"}

    result = {
        "file": csv_path.name,
        "path": str(csv_path),
        "total_samples": 0,
        "threats": 0,
        "safe": 0,
        "threat_ratio": 0.0,
        "balance_status": "unknown",
        "label_column": None,
        "available_features": [],
        "missing_features": [],
        "has_text_column": False,
        "warnings": [],
    }

    try:
        df = pd.read_csv(csv_path)
        result["total_samples"] = len(df)

        # Discover label column
        label_col = discover_label_column(df)
        result["label_column"] = label_col

        # Discover features
        available_features = discover_feature_columns(df)
        result["available_features"] = list(available_features)
        result["has_text_column"] = any("text" in f.lower() for f in available_features)

        # Check missing features
        missing = check_missing_features(available_features, model_type)
        result["missing_features"] = list(missing)

        if label_col:
            # Calculate class balance
            try:
                threats = len(df[df[label_col] == 1])
                safe = len(df[df[label_col] == 0])
                result["threats"] = threats
                result["safe"] = safe
                result["threat_ratio"] = (threats / len(df) * 100) if len(df) > 0 else 0

                ratio, status = calculate_balance_ratio(threats, safe)
                result["balance_status"] = status

                if status == "critical":
                    result["warnings"].append(
                        f"CRITICAL: Severely imbalanced dataset (ratio: {ratio:.2f})"
                    )
                elif status == "imbalanced":
                    result["warnings"].append(
                        f"WARNING: Imbalanced dataset (ratio: {ratio:.2f})"
                    )

                if threats == 0:
                    result["warnings"].append(
                        "ALERT: No threats found! Model cannot learn to defend."
                    )

            except Exception as e:
                result["warnings"].append(f"Failed to analyze labels: {e}")
        else:
            result["warnings"].append("Could not identify label column.")

        if missing:
            result["warnings"].append(
                f"Missing features for {model_type}: {', '.join(sorted(missing))}"
            )

    except Exception as e:
        result["error"] = str(e)

    return result


def inspect_datasets(
    data_dirs: List[str],
    model_type: str = "heuristic",
    check_adversarial: bool = True,
) -> Dict:
    """
    Scan multiple directories for datasets and generate reconnaissance report.

    Args:
        data_dirs: List of directories to scan.
        model_type: Model type to check expected features against.
        check_adversarial: Whether to warn about missing adversarial examples.

    Returns:
        Comprehensive reconnaissance report dictionary.
    """
    try:
        import pandas as pd
    except ImportError:
        logging.error("Pandas is required for reconnaissance. Please install it.")
        return {"error": "pandas not installed"}

    report = {
        "scanned_directories": data_dirs,
        "datasets_found": 0,
        "total_samples": 0,
        "total_threats": 0,
        "total_safe": 0,
        "overall_balance_status": "unknown",
        "datasets": [],
        "all_warnings": [],
        "recommendations": [],
    }

    for data_dir in data_dirs:
        path = Path(data_dir)
        if not path.exists():
            logging.warning(f"Directory {data_dir} does not exist. Skipping.")
            continue

        csv_files = list(path.glob("*.csv"))
        if not csv_files:
            logging.info(f"No CSV files found in {data_dir}")
            continue

        logging.info(f"Found {len(csv_files)} dataset(s) in {data_dir}")

        for csv_file in csv_files:
            result = inspect_single_dataset(csv_file, model_type)
            report["datasets"].append(result)

            if "error" not in result:
                report["datasets_found"] += 1
                report["total_samples"] += result["total_samples"]
                report["total_threats"] += result["threats"]
                report["total_safe"] += result["safe"]

                # Log dataset info
                logging.info(f"Dataset: {result['file']}")
                logging.info(f"  - Total: {result['total_samples']}")
                logging.info(f"  - Safe (0): {result['safe']}")
                logging.info(f"  - Threats (1): {result['threats']} ({result['threat_ratio']:.1f}%)")

                if result["missing_features"]:
                    logging.warning(f"  - Missing features: {', '.join(result['missing_features'])}")

                for warning in result["warnings"]:
                    logging.warning(f"  ! {warning}")
                    report["all_warnings"].append(f"{result['file']}: {warning}")

    # Overall statistics
    ratio, status = calculate_balance_ratio(report["total_threats"], report["total_safe"])
    report["overall_balance_status"] = status

    # Generate recommendations
    if report["total_threats"] < 100:
        report["recommendations"].append(
            "CRITICAL: Very few threat samples. Enable Adversarial Generator with --include-adversarial"
        )

    if status in ["imbalanced", "critical"]:
        report["recommendations"].append(
            f"Dataset is {status}. Consider oversampling threats or using class weights."
        )

    if check_adversarial and report["total_threats"] < report["total_samples"] * 0.2:
        report["recommendations"].append(
            "Less than 20% threats. Adversarial training recommended for robust models."
        )

    # Print summary
    logging.info("=" * 50)
    logging.info("RECONNAISSANCE SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Directories scanned: {len(data_dirs)}")
    logging.info(f"Datasets analyzed: {report['datasets_found']}")
    logging.info(f"Total samples: {report['total_samples']}")
    logging.info(f"Total threats: {report['total_threats']}")
    logging.info(f"Total safe: {report['total_safe']}")
    logging.info(f"Overall balance: {status}")

    if report["recommendations"]:
        logging.info("-" * 50)
        logging.info("RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            logging.warning(f"  {i}. {rec}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Reconnaissance Tool for Nethical Training Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        action="append",
        dest="data_dirs",
        help="Directory with CSV datasets (can be specified multiple times)",
    )
    parser.add_argument(
        "--scan-defaults",
        action="store_true",
        help="Scan default directories: datasets/datasets, data/external, data/labeled_events",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="heuristic",
        choices=["heuristic", "logistic", "anomaly", "correlation", "transformer"],
        help="Model type to check expected features against",
    )
    parser.add_argument(
        "--no-adversarial-check",
        action="store_true",
        help="Disable adversarial example warnings",
    )
    args = parser.parse_args()

    # Determine directories to scan
    dirs_to_scan = []
    if args.scan_defaults:
        dirs_to_scan.extend(DEFAULT_SCAN_DIRS)
    if args.data_dirs:
        dirs_to_scan.extend(args.data_dirs)
    if not dirs_to_scan:
        # Default to data/external if nothing specified
        dirs_to_scan = ["data/external"]

    inspect_datasets(
        data_dirs=dirs_to_scan,
        model_type=args.model_type,
        check_adversarial=not args.no_adversarial_check,
    )
