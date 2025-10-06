#!/usr/bin/env python3
"""End-to-End Training Orchestrator for Real-World Datasets.

This script orchestrates the complete pipeline:
1. Download datasets (via Kaggle API or manual instructions)
2. Process each dataset using dedicated processors
3. Merge all processed data into processed_train_data.json
4. Train the BaselineMLClassifier
5. Evaluate and save the model

Usage:
    python scripts/baseline_orchestrator.py --download
    python scripts/baseline_orchestrator.py --process-only
    python scripts/baseline_orchestrator.py --train-only
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.mlops.baseline import BaselineMLClassifier
from scripts.dataset_processors.cyber_security_processor import CyberSecurityAttacksProcessor
from scripts.dataset_processors.microsoft_security_processor import MicrosoftSecurityProcessor
from scripts.dataset_processors.generic_processor import GenericSecurityProcessor

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Dataset configuration
DATASETS_FILE = Path("datasets/datasets")
DOWNLOAD_DIR = Path("data/external")
PROCESSED_DIR = Path("data/processed")
MERGED_FILE = Path("processed_train_data.json")


def read_dataset_urls() -> List[str]:
    """Read dataset URLs from datasets/datasets file.
    
    Returns:
        List of dataset URLs
    """
    urls = []
    if not DATASETS_FILE.exists():
        logger.warning(f"Dataset list file not found: {DATASETS_FILE}")
        return urls
    
    with open(DATASETS_FILE, 'r') as f:
        for line in f:
            url = line.strip()
            if url and url.startswith('http'):
                urls.append(url)
    
    logger.info(f"Found {len(urls)} dataset URLs")
    return urls


def download_datasets(urls: List[str]) -> None:
    """Download datasets from Kaggle.
    
    Note: This requires Kaggle API credentials to be configured.
    See: https://github.com/Kaggle/kaggle-api#api-credentials
    
    Args:
        urls: List of Kaggle dataset URLs
    """
    logger.info("=" * 60)
    logger.info("DATASET DOWNLOAD")
    logger.info("=" * 60)
    
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        import kaggle
        logger.info("Kaggle API is available")
    except ImportError:
        logger.error("Kaggle API not installed. Install with: pip install kaggle")
        logger.info("\nManual download instructions:")
        logger.info("Please download the following datasets manually:")
        for url in urls:
            logger.info(f"  - {url}")
        logger.info(f"\nSave CSV files to: {DOWNLOAD_DIR.absolute()}")
        return
    
    # Parse and download each dataset
    for url in urls:
        try:
            # Extract dataset identifier from URL
            # Example: https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks
            # -> teamincribo/cyber-security-attacks
            if '/datasets/' in url:
                parts = url.split('/datasets/')[-1].split('/')
                if len(parts) >= 2:
                    dataset_id = f"{parts[0]}/{parts[1]}"
                    logger.info(f"Downloading {dataset_id}...")
                    try:
                        kaggle.api.dataset_download_files(
                            dataset_id,
                            path=str(DOWNLOAD_DIR),
                            unzip=True
                        )
                        logger.info(f"✓ Downloaded {dataset_id}")
                    except Exception as e:
                        logger.error(f"Failed to download {dataset_id}: {e}")
            else:
                logger.warning(f"Skipping non-dataset URL: {url}")
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")


def process_datasets() -> List[Path]:
    """Process all downloaded datasets.
    
    Returns:
        List of paths to processed JSON files
    """
    logger.info("=" * 60)
    logger.info("DATASET PROCESSING")
    logger.info("=" * 60)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files in download directory
    csv_files = list(DOWNLOAD_DIR.glob("**/*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {DOWNLOAD_DIR}")
    
    if not csv_files:
        logger.warning("No CSV files found. Please download datasets first.")
        return []
    
    processed_files = []
    
    # Process each CSV file with appropriate processor
    for csv_file in csv_files:
        logger.info(f"\nProcessing: {csv_file.name}")
        
        # Choose processor based on filename
        filename_lower = csv_file.name.lower()
        
        if 'cyber' in filename_lower or 'attack' in filename_lower:
            processor = CyberSecurityAttacksProcessor(PROCESSED_DIR)
        elif 'microsoft' in filename_lower or 'incident' in filename_lower:
            processor = MicrosoftSecurityProcessor(PROCESSED_DIR)
        else:
            # Use generic processor
            dataset_name = csv_file.stem.replace(' ', '_').lower()
            processor = GenericSecurityProcessor(dataset_name, PROCESSED_DIR)
        
        try:
            records = processor.process(csv_file)
            if records:
                output_file = processor.save_processed_data(records)
                processed_files.append(output_file)
        except Exception as e:
            logger.error(f"Failed to process {csv_file.name}: {e}")
            continue
    
    logger.info(f"\n✓ Processed {len(processed_files)} datasets")
    return processed_files


def merge_processed_data(processed_files: List[Path]) -> Path:
    """Merge all processed datasets into a single file.
    
    Args:
        processed_files: List of processed JSON files
        
    Returns:
        Path to merged file
    """
    logger.info("=" * 60)
    logger.info("MERGING PROCESSED DATA")
    logger.info("=" * 60)
    
    all_records = []
    
    for pfile in processed_files:
        logger.info(f"Loading {pfile.name}...")
        try:
            with open(pfile, 'r') as f:
                records = json.load(f)
                all_records.extend(records)
                logger.info(f"  Added {len(records)} records")
        except Exception as e:
            logger.error(f"Failed to load {pfile}: {e}")
    
    logger.info(f"\nTotal records: {len(all_records)}")
    
    # Shuffle records
    random.shuffle(all_records)
    
    # Save merged data
    with open(MERGED_FILE, 'w') as f:
        json.dump(all_records, f, indent=2)
    
    logger.info(f"✓ Saved merged data to {MERGED_FILE}")
    return MERGED_FILE


def train_model(data_file: Path) -> None:
    """Train the BaselineMLClassifier on merged data.
    
    Args:
        data_file: Path to merged training data
    """
    logger.info("=" * 60)
    logger.info("MODEL TRAINING")
    logger.info("=" * 60)
    
    # Load data
    logger.info(f"Loading training data from {data_file}...")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} training samples")
    
    # Split into train and validation
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    # Train classifier
    logger.info("\nTraining BaselineMLClassifier...")
    clf = BaselineMLClassifier()
    clf.train(train_data)
    logger.info("✓ Training complete")
    
    # Evaluate on validation set
    logger.info("\nEvaluating on validation set...")
    val_features = [d["features"] for d in val_data]
    val_labels = [d["label"] for d in val_data]
    val_preds = [clf.predict(f)["label"] for f in val_features]
    
    metrics = clf.compute_metrics(val_preds, val_labels)
    
    logger.info("\nValidation Metrics:")
    logger.info("=" * 40)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"{k:20s}: {v:.4f}")
        else:
            logger.info(f"{k:20s}: {v}")
    
    # Save model
    model_dir = Path("models/candidates")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "baseline_model.json"
    
    clf.save(str(model_path))
    logger.info(f"\n✓ Model saved to {model_path}")
    
    # Save metrics
    metrics_path = model_dir / "baseline_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✓ Metrics saved to {metrics_path}")


def main():
    """Main orchestrator function."""
    parser = argparse.ArgumentParser(
        description="End-to-End Training Orchestrator for Real-World Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: download, process, and train
  python scripts/baseline_orchestrator.py

  # Download datasets only
  python scripts/baseline_orchestrator.py --download

  # Process existing CSV files only
  python scripts/baseline_orchestrator.py --process-only

  # Train on existing processed_train_data.json
  python scripts/baseline_orchestrator.py --train-only
        """
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Only download datasets (requires Kaggle API)'
    )
    parser.add_argument(
        '--process-only',
        action='store_true',
        help='Only process CSV files (skip download and training)'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train model (requires existing processed_train_data.json)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("NETHICAL END-TO-END TRAINING ORCHESTRATOR")
    logger.info("=" * 60)
    logger.info("")
    
    # Read dataset URLs
    urls = read_dataset_urls()
    
    if args.download:
        # Only download
        download_datasets(urls)
    elif args.process_only:
        # Only process
        processed_files = process_datasets()
        if processed_files:
            merge_processed_data(processed_files)
    elif args.train_only:
        # Only train
        if not MERGED_FILE.exists():
            logger.error(f"Merged data file not found: {MERGED_FILE}")
            logger.error("Please run processing first or use full pipeline")
            sys.exit(1)
        train_model(MERGED_FILE)
    else:
        # Full pipeline
        logger.info("Running full pipeline:")
        logger.info("1. Download datasets")
        logger.info("2. Process datasets")
        logger.info("3. Merge processed data")
        logger.info("4. Train model")
        logger.info("")
        
        # Step 1: Download
        download_datasets(urls)
        
        # Step 2: Process
        processed_files = process_datasets()
        
        if not processed_files:
            logger.error("No datasets were processed. Exiting.")
            sys.exit(1)
        
        # Step 3: Merge
        merged_file = merge_processed_data(processed_files)
        
        # Step 4: Train
        train_model(merged_file)
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
