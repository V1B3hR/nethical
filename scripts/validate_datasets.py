#!/usr/bin/env python3
"""
Dataset Validation Script

Validates dataset freshness and quality for ML training.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

# Constants for class imbalance detection
# Alert if any class is below MIN_CLASS_PCT or above MAX_CLASS_PCT
# These thresholds are chosen to detect severe imbalances that would
# significantly impact model training and fairness
MIN_CLASS_PCT_THRESHOLD = 5.0  # Minimum acceptable class percentage
MAX_CLASS_PCT_THRESHOLD = 95.0  # Maximum acceptable class percentage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()


def load_datasets_file() -> List[str]:
    """Load dataset URLs from datasets file"""
    datasets_file = Path("datasets/datasets")
    
    if not datasets_file.exists():
        logging.warning(f"Datasets file not found: {datasets_file}")
        return []
    
    datasets = []
    try:
        with open(datasets_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and line.startswith("https://"):
                    datasets.append(line)
    except Exception as e:
        logging.error(f"Failed to read datasets file: {e}")
        return []
    
    logging.info(f"Loaded {len(datasets)} datasets from file")
    return datasets


def check_kaggle_freshness(max_age_days: int = 30) -> Dict[str, Any]:
    """Check if Kaggle datasets are up to date"""
    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'datasets_stale': False,
        'stale_datasets': [],
        'max_age_days': max_age_days
    }
    
    datasets = load_datasets_file()
    
    if not datasets:
        result['message'] = "No datasets configured"
        return result
    
    # Check local data directory
    data_dir = Path("data/external")
    
    if not data_dir.exists():
        logging.warning("External data directory does not exist")
        result['datasets_stale'] = True
        result['message'] = "No local dataset files found"
        return result
    
    # Check file ages
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        logging.warning("No CSV files found in data directory")
        result['datasets_stale'] = True
        result['message'] = "No dataset files found"
        return result
    
    now = datetime.now(timezone.utc)
    cutoff_time = now - timedelta(days=max_age_days)
    
    for csv_file in csv_files:
        # Get file modification time
        mtime = datetime.fromtimestamp(csv_file.stat().st_mtime, tz=timezone.utc)
        age_days = (now - mtime).days
        
        if mtime < cutoff_time:
            result['stale_datasets'].append({
                'file': csv_file.name,
                'age_days': age_days,
                'last_modified': mtime.isoformat()
            })
            logging.warning(f"Stale dataset: {csv_file.name} (age: {age_days} days)")
    
    if result['stale_datasets']:
        result['datasets_stale'] = True
        logging.warning(f"Found {len(result['stale_datasets'])} stale datasets")
    else:
        logging.info("✓ All datasets are fresh")
    
    return result


def check_data_quality(min_samples: int = 10000) -> Dict[str, Any]:
    """Check data quality metrics"""
    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'quality_issues': False,
        'issues': [],
        'min_samples': min_samples
    }
    
    data_dir = Path("data/external")
    
    if not data_dir.exists():
        result['quality_issues'] = True
        result['message'] = "Data directory does not exist"
        return result
    
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        result['quality_issues'] = True
        result['message'] = "No CSV files found"
        return result
    
    try:
        import pandas as pd
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Check sample count
                if len(df) < min_samples:
                    result['issues'].append({
                        'file': csv_file.name,
                        'issue': 'insufficient_samples',
                        'samples': len(df),
                        'required': min_samples
                    })
                    result['quality_issues'] = True
                    logging.warning(
                        f"Insufficient samples in {csv_file.name}: "
                        f"{len(df)} < {min_samples}"
                    )
                
                # Check for missing values
                missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                if missing_pct > 20:
                    result['issues'].append({
                        'file': csv_file.name,
                        'issue': 'high_missing_values',
                        'missing_percentage': missing_pct
                    })
                    result['quality_issues'] = True
                    logging.warning(
                        f"High missing values in {csv_file.name}: {missing_pct:.1f}%"
                    )
                
                logging.info(
                    f"✓ {csv_file.name}: {len(df)} samples, "
                    f"{missing_pct:.1f}% missing"
                )
            
            except Exception as e:
                result['issues'].append({
                    'file': csv_file.name,
                    'issue': 'read_error',
                    'error': str(e)
                })
                result['quality_issues'] = True
                logging.error(f"Failed to read {csv_file.name}: {e}")
    
    except ImportError:
        logging.warning("pandas not installed, skipping quality checks")
        result['message'] = "pandas not available for quality checks"
    
    if not result['quality_issues']:
        logging.info("✓ No data quality issues found")
    
    return result


def check_distribution_shift() -> Dict[str, Any]:
    """Check for data distribution shifts"""
    result = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'distribution_shift': False,
        'shifts_detected': []
    }
    
    # This is a placeholder implementation
    # In a real scenario, you would compare current data distribution
    # with historical distributions
    
    data_dir = Path("data/external")
    
    if not data_dir.exists():
        result['message'] = "Data directory does not exist"
        return result
    
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        result['message'] = "No CSV files found"
        return result
    
    try:
        import pandas as pd
        import numpy as np
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Check label distribution
                label_cols = [c for c in df.columns if 'label' in c.lower() or 'target' in c.lower()]
                
                for label_col in label_cols:
                    if label_col in df.columns:
                        # Calculate label distribution
                        label_dist = df[label_col].value_counts(normalize=True)
                        
                        # Check for extreme imbalance (simplified check)
                        if len(label_dist) > 1:
                            min_class_pct = label_dist.min() * 100
                            max_class_pct = label_dist.max() * 100
                            
                            # Check against configured thresholds
                            if min_class_pct < MIN_CLASS_PCT_THRESHOLD or max_class_pct > MAX_CLASS_PCT_THRESHOLD:
                                result['shifts_detected'].append({
                                    'file': csv_file.name,
                                    'column': label_col,
                                    'issue': 'extreme_class_imbalance',
                                    'min_class_pct': min_class_pct,
                                    'max_class_pct': max_class_pct,
                                    'thresholds': {
                                        'min_threshold': MIN_CLASS_PCT_THRESHOLD,
                                        'max_threshold': MAX_CLASS_PCT_THRESHOLD
                                    }
                                })
                                result['distribution_shift'] = True
                                logging.warning(
                                    f"Extreme class imbalance in {csv_file.name}.{label_col}: "
                                    f"{min_class_pct:.1f}% / {max_class_pct:.1f}% "
                                    f"(thresholds: {MIN_CLASS_PCT_THRESHOLD}% - {MAX_CLASS_PCT_THRESHOLD}%)"
                                )
                
                logging.info(f"✓ Checked distribution for {csv_file.name}")
            
            except Exception as e:
                logging.warning(f"Failed to check distribution in {csv_file.name}: {e}")
    
    except ImportError:
        logging.warning("pandas/numpy not installed, skipping distribution checks")
        result['message'] = "pandas/numpy not available for distribution checks"
    
    if not result['distribution_shift']:
        logging.info("✓ No distribution shifts detected")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate datasets for ML training")
    parser.add_argument(
        "--check-kaggle-freshness",
        action='store_true',
        help="Check Kaggle dataset update dates"
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Maximum age for datasets in days"
    )
    parser.add_argument(
        "--check-quality",
        action='store_true',
        help="Validate data quality metrics"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10000,
        help="Minimum required samples"
    )
    parser.add_argument(
        "--check-distribution",
        action='store_true',
        help="Check for data distribution shifts"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    if args.check_kaggle_freshness:
        results = check_kaggle_freshness(args.max_age_days)
        output_file = args.output or "kaggle_check.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to {output_file}")
        
        if results['datasets_stale']:
            return 1
    
    elif args.check_quality:
        results = check_data_quality(args.min_samples)
        output_file = args.output or "quality_check.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to {output_file}")
        
        if results['quality_issues']:
            return 1
    
    elif args.check_distribution:
        results = check_distribution_shift()
        output_file = args.output or "distribution_check.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to {output_file}")
        
        if results['distribution_shift']:
            return 1
    
    else:
        logging.info("No action specified. Use --help for options")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
