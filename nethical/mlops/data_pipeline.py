import os
import requests
from pathlib import Path

DATASET_LIST = Path("datasets/datasets")
DOWNLOAD_DIR = Path("data/external")

def fetch_kaggle_dataset(url):
    print(f"[INFO] Please manually download dataset from: {url}")
    # In production, integrate with Kaggle API or other automation

def ingest_all():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_LIST) as f:
        for line in f:
            url = line.strip()
            if url and url.startswith("http"):
                fetch_kaggle_dataset(url)
    print("[INFO] Dataset ingestion completed. Please check data/external/ directory.")

if __name__ == "__main__":
    ingest_all()
