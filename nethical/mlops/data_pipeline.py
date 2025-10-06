import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def fetch_kaggle_dataset(url):
    logging.info(f"Manual download required. Please download the dataset from: {url}")
    # TODO: Integrate with Kaggle API for automated downloads

def ingest_all(dataset_list_path=Path("datasets/datasets"), download_dir=Path("data/external")):
    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(dataset_list_path) as f:
            for line in f:
                url = line.strip()
                if url and url.startswith("http"):
                    fetch_kaggle_dataset(url)
        logging.info(f"Dataset ingestion completed. Please check {download_dir}/ directory.")
    except FileNotFoundError:
        logging.error(f"Dataset list file not found: {dataset_list_path}")
    except Exception as e:
        logging.error(f"An error occurred during ingestion: {e}")

if __name__ == "__main__":
    ingest_all()
