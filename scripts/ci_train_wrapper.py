#!/usr/bin/env python3
"""
CI Training Wrapper: robust dataset resolution + download filter + trainer invocation

Security note:
- You provided Kaggle credentials. Hardcoding secrets in repo or logs is risky.
- Prefer passing via environment variables or GitHub Secrets.
- This script will write ~/.kaggle/kaggle.json and chmod 600.

What this does:
- Resolves dataset slugs from both datasets/datasets and datasets/datasets.md
- Filters out disallowed/broken slugs (e.g., kmldas/data-ethics-in-data-science-analytics-ml-and-ai)
- Prints the final dataset list for CI debugging
- Downloads datasets to data/external/ using Kaggle API if available
- Invokes the main trainer with --no-download so training uses already-downloaded local data

Usage in CI (GitHub Actions step):
  python scripts/ci_train_wrapper.py \
    --model-type all \
    --epochs 70 \
    --batch-size 64 \
    --num-samples 20000 \
    --promotion-min-accuracy 0.85 \
    --promotion-max-ece 0.15 \
    --enable-audit \
    --enable-governance \
    --enable-drift-tracking \
    --verbosity 1
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Constants/paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_TXT = REPO_ROOT / "datasets" / "datasets"        # no extension file used by trainer
DATASETS_MD  = REPO_ROOT / "datasets" / "datasets.md"     # markdown list (human-friendly)
DATA_EXTERNAL_DIR = REPO_ROOT / "data" / "external"

# Slugs to exclude (blocklist)
EXCLUDE_SLUGS = {
    "kmldas/data-ethics-in-data-science-analytics-ml-and-ai",
}

# Default Kaggle credentials (provided by user; consider using secrets instead)
DEFAULT_KAGGLE_USERNAME = "andrzejmatewski"
DEFAULT_KAGGLE_KEY = "406272cc0df9e65d6d7fa69ff136bf5c"

# Regex helpers
KAGGLE_DATASETS_URL_RE = re.compile(r"^https://www\.kaggle\.com/datasets/([^/\s]+)/([^/\s]+)")

def parse_slugs_from_datasets_txt(path: Path) -> List[str]:
    """
    Parse slugs from datasets/datasets (expects Kaggle dataset URLs or plain 'owner/dataset' lines).
    """
    if not path.exists():
        return []
    slugs: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                # Accept raw slug "owner/dataset"
                if "/" in s and " " not in s and "https://" not in s:
                    slugs.append(s)
                    continue
                # Accept Kaggle dataset URL
                m = KAGGLE_DATASETS_URL_RE.match(s)
                if m:
                    owner, dataset = m.group(1).strip(), m.group(2).strip()
                    if owner and dataset:
                        slugs.append(f"{owner}/{dataset}")
    except Exception as e:
        print(f"[WARN] Failed to parse {path}: {e}", file=sys.stderr)
    return slugs

def parse_slugs_from_markdown(path: Path) -> List[str]:
    """
    Parse slugs from datasets/datasets.md (bullet list with Kaggle dataset URLs).
    """
    if not path.exists():
        return []
    slugs: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                m = KAGGLE_DATASETS_URL_RE.match(s)
                if m:
                    owner, dataset = m.group(1).strip(), m.group(2).strip()
                    if owner and dataset:
                        slugs.append(f"{owner}/{dataset}")
    except Exception as e:
        print(f"[WARN] Failed to parse {path}: {e}", file=sys.stderr)
    return slugs

def merge_and_filter_slugs(a: List[str], b: List[str], exclude: set) -> List[str]:
    merged = []
    seen = set()
    for src in (a, b):
        for slug in src:
            if slug in exclude:
                continue
            if slug not in seen:
                seen.add(slug)
                merged.append(slug)
    return merged

def resolve_kaggle_credentials(explicit_user: Optional[str], explicit_key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolution order:
    1) CLI flags
    2) Environment variables KAGGLE_USERNAME/KAGGLE_KEY
    3) Existing ~/.kaggle/kaggle.json
    4) Defaults embedded in script (not recommended; provided per user request)
    """
    if explicit_user and explicit_key:
        return explicit_user, explicit_key
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        return env_user, env_key
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    try:
        if kaggle_json_path.exists():
            with open(kaggle_json_path, "r") as f:
                data = json.load(f)
            user = data.get("username")
            key = data.get("key")
            if user and key:
                return user, key
    except Exception as e:
        print(f"[WARN] Failed to read existing kaggle.json: {e}", file=sys.stderr)
    # Fallback to defaults (provided by user)
    return DEFAULT_KAGGLE_USERNAME, DEFAULT_KAGGLE_KEY

def ensure_kaggle_json(username: str, key: str, overwrite: bool = False) -> None:
    kaggle_path = Path.home() / ".kaggle"
    kaggle_json_path = kaggle_path / "kaggle.json"
    kaggle_path.mkdir(parents=True, exist_ok=True)
    if kaggle_json_path.exists() and not overwrite:
        print(f("[INFO] Kaggle credentials already present at {kaggle_json_path}"))
        return
    print(f"[INFO] Writing Kaggle credentials to {kaggle_json_path}")
    with open(kaggle_json_path, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(kaggle_json_path, 0o600)

def kaggle_download(slugs: List[str]) -> None:
    try:
        import kaggle  # type: ignore
    except Exception:
        print("[WARN] Kaggle package not installed. Skipping dataset downloads.", file=sys.stderr)
        return
    DATA_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        print(f"[INFO] Downloading dataset: {slug}")
        try:
            kaggle.api.dataset_download_files(slug, path=str(DATA_EXTERNAL_DIR), unzip=True)
            print(f"[INFO] Downloaded and extracted: {slug}")
        except Exception as e:
            print(f"[WARN] Could not download {slug}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="CI wrapper for Nethical training")
    # Mirror trainer args that we pass-through
    parser.add_argument("--model-type", required=True, type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-samples", default=10000, type=int)
    parser.add_argument("--promotion-min-accuracy", default=0.85, type=float)
    parser.add_argument("--promotion-max-ece", default=0.08, type=float)
    parser.add_argument("--enable-audit", action="store_true")
    parser.add_argument("--enable-governance", action="store_true")
    parser.add_argument("--enable-drift-tracking", action="store_true")
    parser.add_argument("--verbosity", default=1, type=int)

    # Kaggle creds
    parser.add_argument("--kaggle-username", default=None, type=str)
    parser.add_argument("--kaggle-key", default=None, type=str)
    parser.add_argument("--overwrite-kaggle-json", action="store_true")

    args = parser.parse_args()

    # 1) Resolve dataset slugs from both files
    slugs_txt = parse_slugs_from_datasets_txt(DATASETS_TXT)
    slugs_md  = parse_slugs_from_markdown(DATASETS_MD)
    final_slugs = merge_and_filter_slugs(slugs_txt, slugs_md, EXCLUDE_SLUGS)

    # 2) Debug print
    print("[INFO] datasets/datasets exists:", DATASETS_TXT.exists())
    print("[INFO] datasets/datasets.md exists:", DATASETS_MD.exists())
    print("[INFO] Final dataset slugs to download (after filtering):")
    for i, s in enumerate(final_slugs, 1):
        print(f"  {i}. {s}")
    if not final_slugs:
        print("[WARN] No dataset slugs resolved. Training will proceed with synthetic or local data.")

    # 3) Kaggle credentials and download
    user, key = resolve_kaggle_credentials(args.kaggle_username, args.kaggle_key)
    if user and key:
        ensure_kaggle_json(user, key, overwrite=args.overwrite_kaggle_json)
    else:
        print("[INFO] Kaggle credentials not provided/found. Attempting API auth via existing config if any.")
    if final_slugs:
        kaggle_download(final_slugs)

    # 4) Invoke trainer with --no-download to avoid any internal dataset list fallback
    trainer = REPO_ROOT / "training" / "train_any_model.py"
    if not trainer.exists():
        print(f"[ERROR] Trainer not found at {trainer}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, str(trainer),
        "--model-type", args.model_type,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-samples", str(args.num_samples),
        "--promotion-min-accuracy", str(args.promotion_min_accuracy),
        "--promotion-max-ece", str(args.promotion_max_ece),
        "--verbosity", str(args.verbosity),
        "--no-download",  # prevent internal downloads; use what we just fetched
    ]
    if args.enable_audit:
        cmd.append("--enable-audit")
    if args.enable_governance:
        cmd.append("--enable-governance")
    if args.enable_drift_tracking:
        cmd.append("--enable-drift-tracking")

    print("[INFO] Launching trainer:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
