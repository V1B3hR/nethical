#!/usr/bin/env python3
"""
Data Reconnaissance Tool.

Inspects datasets to ensure we have valid targets (Threats) for training.
Does NOT remove toxic data. We need it for the models to learn.
"""

import argparse
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [RECON] %(message)s")


def inspect_datasets(data_dir: str):
    path = Path(data_dir)
    if not path.exists():
        logging.error(f"Data directory {data_dir} does not exist.")
        return

    csv_files = list(path.glob("*.csv"))
    logging.info(f"Found {len(csv_files)} datasets in {data_dir}")

    try:
        import pandas as pd
    except ImportError:
        logging.error("Pandas is required for reconnaissance. Please install it.")
        return

    total_samples = 0
    total_threats = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Heuristic label detection
            label_col = None
            for cand in ["label", "target", "class", "is_threat", "is_toxic"]:
                if cand in df.columns:
                    label_col = cand
                    break

            if not label_col:
                # Try finding any numeric column with 0/1
                for col in df.columns:
                    if df[col].nunique() == 2 and sorted(df[col].unique().tolist()) == [0, 1]:
                        label_col = col
                        break

            if label_col:
                count = len(df)
                threats = len(df[df[label_col] == 1])
                safe = len(df[df[label_col] == 0])
                ratio = (threats / count) * 100 if count > 0 else 0

                logging.info(f"Dataset: {csv_file.name}")
                logging.info(f"  - Total: {count}")
                logging.info(f"  - Safe (0): {safe}")
                logging.info(f"  - Threats (1): {threats} ({ratio:.1f}%)")

                if threats == 0:
                    logging.warning(f"  ! ALERT: No threats found in {csv_file.name}. Model will not learn to defend!")
                else:
                    logging.info(f"  + OK: Targets confirmed.")

                total_samples += count
                total_threats += threats
            else:
                logging.warning(f"Dataset: {csv_file.name} - Could not identify label column.")

        except Exception as e:
            logging.error(f"Failed to inspect {csv_file.name}: {e}")

    logging.info("--- RECONNAISSANCE SUMMARY ---")
    logging.info(f"Total Intelligence gathered: {total_samples} samples")
    logging.info(f"Total Confirmed Threats: {total_threats}")

    if total_threats < 100:
        logging.warning("!!! CRITICAL: Very few threats identified. Recommendation: Enable Adversarial Generator.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Reconnaissance Tool")
    parser.add_argument("--data-dir", type=str, default="data/external", help="Directory with CSV datasets")
    args = parser.parse_args()

    inspect_datasets(args.data_dir)
