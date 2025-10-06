import os
from pathlib import Path
import shutil

MODEL_DIR = Path("models")
CANDIDATES = MODEL_DIR / "candidates"
CURRENT = MODEL_DIR / "current"

def promote_model(model_filename):
    src = CANDIDATES / model_filename
    dst = CURRENT / model_filename
    CURRENT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[INFO] Promoted model {model_filename} to production.")

def list_models():
    print("[INFO] Candidate models:")
    for f in CANDIDATES.glob("*.json"):
        print(" -", f.name)
    print("[INFO] Current production models:")
    for f in CURRENT.glob("*.json"):
        print(" -", f.name)

if __name__ == "__main__":
    list_models()
