import argparse
import subprocess
import sys
from pathlib import Path

def run_training(params=None):
    print("[INFO] Running training pipeline...")
    script_path = Path("scripts") / "train_model.py"
    cmd = [sys.executable, str(script_path)]
    if params:
        cmd.extend(params)
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-all', action='store_true', help='Run full workflow')
    parser.add_argument('extra_args', nargs=argparse.REMAINDER, help='Additional arguments for training script')
    args = parser.parse_args()
    params = []
    if args.run_all:
        params.append('--run-all')
    params.extend(args.extra_args)
    run_training(params)
