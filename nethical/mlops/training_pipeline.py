import argparse
import subprocess
from pathlib import Path

def run_training(params=""):
    print("[INFO] Running training pipeline...")
    cmd = f"python scripts/train_model.py {params}"
    print(f"[CMD] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-all', action='store_true', help='Run full workflow')
    args = parser.parse_args()
    params = "--run-all" if args.run_all else ""
    run_training(params)
