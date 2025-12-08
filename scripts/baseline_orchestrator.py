#!/usr/bin/env python3
"""Baseline Model Training Orchestrator

This script orchestrates the complete training pipeline for the baseline ML classifier:
1. Downloads real-world security datasets from Kaggle (optional)
2. Processes each dataset with specialized processors
3. Merges all data into a unified training set
4. Trains the BaselineMLClassifier
5. Evaluates and saves the trained model

Usage:
    # Full pipeline: download, process, and train
    python scripts/baseline_orchestrator.py

    # Download datasets only
    python scripts/baseline_orchestrator.py --download

    # Process existing CSV files only
    python scripts/baseline_orchestrator.py --process-only

    # Train on existing processed_train_data.json
    python scripts/baseline_orchestrator.py --train-only
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.mlops.baseline import BaselineMLClassifier
from scripts.dataset_processors.cyber_security_processor import (
    CyberSecurityAttacksProcessor,
)
from scripts.dataset_processors.microsoft_security_processor import (
    MicrosoftSecurityProcessor,
)
from scripts.dataset_processors.generic_processor import GenericSecurityProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directories
DATA_EXTERNAL_DIR = Path("data/external")
DATA_PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models/candidates")
DATASETS_FILE = Path("datasets/datasets")


def ensure_directories():
    """Create necessary directories if they don't exist."""
    DATA_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Directories verified/created")


def parse_kaggle_datasets() -> List[str]:
    """Parse dataset URLs from datasets file."""
    datasets = []
    if not DATASETS_FILE.exists():
        logger.warning(f"Datasets file not found: {DATASETS_FILE}")
        return datasets

    with open(DATASETS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("https://www.kaggle.com/datasets/"):
                # Extract dataset slug from URL
                # Format: https://www.kaggle.com/datasets/owner/dataset-name
                parts = line.split("/")
                if len(parts) >= 6:
                    owner = parts[4]
                    dataset = parts[5].split("?")[0]  # Remove query params
                    datasets.append(f"{owner}/{dataset}")

    logger.info(f"Found {len(datasets)} dataset URLs in {DATASETS_FILE}")
    return datasets


def download_datasets(dataset_slugs: List[str]) -> None:
    """Download datasets from Kaggle.

    Args:
        dataset_slugs: List of dataset slugs (e.g., 'owner/dataset-name')
    """
    try:
        import kaggle
    except ImportError:
        logger.warning("Kaggle package not installed. Install with: pip install kaggle")
        logger.info(
            "Skipping dataset downloads. Will use existing data or synthetic data."
        )
        return

    # Check for Kaggle API credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.warning("Kaggle API credentials not found at ~/.kaggle/kaggle.json")
        logger.info(
            "To download datasets, create credentials at: https://www.kaggle.com/account"
        )
        logger.info(
            "Skipping dataset downloads. Will use existing data or synthetic data."
        )
        return

    logger.info(f"Downloading {len(dataset_slugs)} datasets from Kaggle...")

    for slug in dataset_slugs:
        try:
            logger.info(f"Downloading: {slug}")
            kaggle.api.dataset_download_files(
                slug, path=str(DATA_EXTERNAL_DIR), unzip=True, quiet=False
            )
            logger.info(f"✓ Downloaded: {slug}")
        except Exception as e:
            logger.warning(f"Failed to download {slug}: {e}")
            continue


def process_datasets() -> List[Dict[str, Any]]:
    """Process all CSV files in the external data directory.

    Returns:
        List of processed records in standard format
    """
    logger.info("Processing datasets...")

    all_records = []
    csv_files = list(DATA_EXTERNAL_DIR.glob("**/*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {DATA_EXTERNAL_DIR}")
        logger.info("Generating synthetic data for training...")
        return generate_synthetic_data()

    logger.info(f"Found {len(csv_files)} CSV files")

    # Process each CSV file with appropriate processor
    for csv_file in csv_files:
        try:
            logger.info(f"Processing: {csv_file.name}")

            # Choose processor based on file name/content
            records = process_single_file(csv_file)

            if records:
                all_records.extend(records)
                logger.info(f"✓ Processed {len(records)} records from {csv_file.name}")
            else:
                logger.warning(f"No records extracted from {csv_file.name}")

        except Exception as e:
            logger.warning(f"Error processing {csv_file.name}: {e}")
            continue

    if not all_records:
        logger.warning("No records extracted from CSV files")
        logger.info("Generating synthetic data for training...")
        return generate_synthetic_data()

    logger.info(f"Total records processed: {len(all_records)}")
    return all_records


def process_single_file(csv_file: Path) -> List[Dict[str, Any]]:
    """Process a single CSV file with the appropriate processor.

    Args:
        csv_file: Path to CSV file

    Returns:
        List of processed records
    """
    filename_lower = csv_file.name.lower()

    # Choose processor based on filename
    if "cyber" in filename_lower or "attack" in filename_lower:
        processor = CyberSecurityAttacksProcessor(output_dir=DATA_PROCESSED_DIR)
    elif "microsoft" in filename_lower or "incident" in filename_lower:
        processor = MicrosoftSecurityProcessor(output_dir=DATA_PROCESSED_DIR)
    else:
        # Use generic processor for unknown datasets
        dataset_name = csv_file.stem
        processor = GenericSecurityProcessor(
            dataset_name, output_dir=DATA_PROCESSED_DIR
        )

    try:
        records = processor.process(csv_file)
        return records
    except Exception as e:
        logger.warning(f"Processor failed for {csv_file.name}: {e}")
        # Try generic processor as fallback
        try:
            processor = GenericSecurityProcessor(
                csv_file.stem, output_dir=DATA_PROCESSED_DIR
            )
            records = processor.process(csv_file)
            return records
        except Exception as e2:
            logger.error(f"Fallback processor also failed: {e2}")
            return []


def generate_synthetic_data(num_samples: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic training data.

    Args:
        num_samples: Number of samples to generate

    Returns:
        List of synthetic records
    """
    import random

    logger.info(f"Generating {num_samples} synthetic samples...")

    data = []
    for i in range(num_samples):
        # Generate random features
        features = {
            "violation_count": random.random(),
            "severity_max": random.random(),
            "recency_score": random.random(),
            "frequency_score": random.random(),
            "context_risk": random.random(),
        }

        # Simple labeling: positive if violation_count + severity_max > 1
        label = (
            1 if (features["violation_count"] + features["severity_max"]) > 1.0 else 0
        )

        data.append(
            {
                "features": features,
                "label": label,
                "meta": {"source": "synthetic", "index": i},
            }
        )

    logger.info(f"Generated {len(data)} synthetic samples")
    return data


def merge_and_save_data(
    records: List[Dict[str, Any]], output_file: str = "processed_train_data.json"
) -> None:
    """Merge and save processed data to JSON file.

    Args:
        records: List of processed records
        output_file: Output filename
    """
    output_path = Path(output_file)

    logger.info(f"Saving {len(records)} records to {output_path}")

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)

    logger.info(f"✓ Data saved to {output_path}")


def load_training_data(
    data_file: str = "processed_train_data.json",
) -> List[Dict[str, Any]]:
    """Load training data from JSON file.

    Args:
        data_file: Path to data file

    Returns:
        List of training records
    """
    data_path = Path(data_file)

    if not data_path.exists():
        logger.error(f"Training data file not found: {data_path}")
        logger.info("Run with --process-only first to generate training data")
        return []

    logger.info(f"Loading training data from {data_path}")

    with open(data_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} training samples")
    return data


def split_data(data: List[Dict[str, Any]], train_ratio: float = 0.8) -> tuple:
    """Split data into train and validation sets.

    Args:
        data: List of data records
        train_ratio: Fraction of data to use for training

    Returns:
        Tuple of (train_data, val_data)
    """
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    logger.info(f"Split: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data


def train_model(train_data: List[Dict[str, Any]]) -> BaselineMLClassifier:
    """Train the baseline classifier.

    Args:
        train_data: List of training samples

    Returns:
        Trained classifier
    """
    logger.info("Training BaselineMLClassifier...")

    classifier = BaselineMLClassifier()
    classifier.train(train_data)

    logger.info("✓ Training completed")
    logger.info(f"  Training samples: {classifier.training_samples}")
    logger.info(f"  Feature weights: {classifier.feature_weights}")

    return classifier


def evaluate_model(
    classifier: BaselineMLClassifier, val_data: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Evaluate the trained model.

    Args:
        classifier: Trained classifier
        val_data: Validation data

    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating model...")

    predictions = []
    labels = []

    for sample in val_data:
        pred = classifier.predict(sample["features"])
        predictions.append(pred["label"])
        labels.append(sample["label"])

    metrics = classifier.compute_metrics(predictions, labels)

    logger.info("Validation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return metrics


def save_model_and_metrics(
    classifier: BaselineMLClassifier, metrics: Dict[str, float]
) -> None:
    """Save model and metrics to disk.

    Args:
        classifier: Trained classifier
        metrics: Evaluation metrics
    """
    model_path = MODELS_DIR / "baseline_model.json"
    metrics_path = MODELS_DIR / "baseline_metrics.json"

    # Save model
    classifier.save(str(model_path))
    logger.info(f"✓ Model saved to {model_path}")

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✓ Metrics saved to {metrics_path}")


def main():
    """Main orchestrator function."""
    parser = argparse.ArgumentParser(
        description="Baseline ML Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python scripts/baseline_orchestrator.py
  
  # Download datasets only
  python scripts/baseline_orchestrator.py --download
  
  # Process existing CSVs only
  python scripts/baseline_orchestrator.py --process-only
  
  # Train on existing processed data
  python scripts/baseline_orchestrator.py --train-only
        """,
    )

    parser.add_argument(
        "--download", action="store_true", help="Download datasets from Kaggle and exit"
    )

    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Process datasets and exit (no training)",
    )

    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train on existing processed_train_data.json (no processing)",
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default="processed_train_data.json",
        help="Path to training data file (default: processed_train_data.json)",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8)",
    )

    args = parser.parse_args()

    # Ensure directories exist
    ensure_directories()

    # Download-only mode
    if args.download:
        logger.info("=== Download Mode ===")
        dataset_slugs = parse_kaggle_datasets()
        download_datasets(dataset_slugs)
        logger.info("Download completed")
        return

    # Process-only mode
    if args.process_only:
        logger.info("=== Process-Only Mode ===")
        records = process_datasets()
        merge_and_save_data(records, args.data_file)
        logger.info("Processing completed")
        return

    # Train-only mode
    if args.train_only:
        logger.info("=== Train-Only Mode ===")
        train_data_all = load_training_data(args.data_file)
        if not train_data_all:
            logger.error("No training data available")
            return

        train_data, val_data = split_data(train_data_all, args.train_ratio)
        classifier = train_model(train_data)
        metrics = evaluate_model(classifier, val_data)
        save_model_and_metrics(classifier, metrics)
        logger.info("Training completed")
        return

    # Full pipeline (default)
    logger.info("=== Full Training Pipeline ===")

    # Step 1: Download datasets (if Kaggle is available)
    logger.info("\n[Step 1/5] Downloading datasets...")
    dataset_slugs = parse_kaggle_datasets()
    download_datasets(dataset_slugs)

    # Step 2: Process datasets
    logger.info("\n[Step 2/5] Processing datasets...")
    records = process_datasets()

    # Step 3: Merge and save
    logger.info("\n[Step 3/5] Merging and saving data...")
    merge_and_save_data(records, args.data_file)

    # Step 4: Train model
    logger.info("\n[Step 4/5] Training model...")
    train_data, val_data = split_data(records, args.train_ratio)
    classifier = train_model(train_data)

    # Step 5: Evaluate and save
    logger.info("\n[Step 5/5] Evaluating and saving model...")
    metrics = evaluate_model(classifier, val_data)
    save_model_and_metrics(classifier, metrics)

    logger.info("\n=== Pipeline Complete ===")
    logger.info(f"Model: models/candidates/baseline_model.json")
    logger.info(f"Metrics: models/candidates/baseline_metrics.json")
    logger.info(f"Training data: {args.data_file}")


if __name__ == "__main__":
    main()
