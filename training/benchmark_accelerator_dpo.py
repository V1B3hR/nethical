#!/usr/bin/env python3
"""Benchmark Suite: Vanilla PyTorch DPO vs AcceleratorAI Turbo DPO on NVIDIA RTX 4070.

Compares:
1. Vanilla PyTorch DPO: Standard AdamW, hard gradient truncation (clip_grad_norm_).
2. AcceleratorAI Turbo DPO:
   - VRAMPressureGuard proactive memory headroom management
   - Pneumatic tanh soft-clipping wastegate
   - KalmanLossGovernor loss trajectory tracking & plateau busting
   - InputGuard data validation and outlier scrubbing

Outputs detailed latency, throughput, VRAM efficiency, and convergence comparison table.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from training.train_dpo_ambassador import DPODatasetLoader, DPOTrainerEngine

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_accelerator_dpo")


def run_benchmark(
    dataset_path: Path,
    num_samples: int = 400,
    epochs: int = 2,
    batch_size: int = 16,
) -> Dict[str, Any]:
    loader = DPODatasetLoader(dataset_path)
    all_data = loader.load()
    bench_data = all_data[:num_samples]

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info(f"Rozpoczynanie benchmarku DPO na {device_name} (próbek: {len(bench_data)}, epok: {epochs}, batch: {batch_size})")

    results = {}

    # 1. Baseline: Vanilla PyTorch DPO (AcceleratorAI DISABLED)
    logger.info("\n>>> [1/2] Uruchamianie Baseline: Vanilla PyTorch DPO...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainer_vanilla = DPOTrainerEngine(
        dataset=bench_data,
        beta=0.1,
        learning_rate=5e-5,
        output_dir=REPO_ROOT / "models" / "benchmark_vanilla",
        use_accelerator=False,
        neural=True,
    )

    t0 = time.perf_counter()
    res_vanilla = trainer_vanilla.run_training(epochs=epochs, batch_size=batch_size)
    t_vanilla = time.perf_counter() - t0

    vram_vanilla = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    # 2. AcceleratorAI Turbo DPO (AcceleratorAI ENABLED)
    logger.info("\n>>> [2/2] Uruchamianie: AcceleratorAI Turbo DPO Engine...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainer_turbo = DPOTrainerEngine(
        dataset=bench_data,
        beta=0.1,
        learning_rate=5e-5,
        output_dir=REPO_ROOT / "models" / "benchmark_turbo",
        use_accelerator=True,
        neural=True,
    )

    t0 = time.perf_counter()
    res_turbo = trainer_turbo.run_training(epochs=epochs, batch_size=batch_size)
    t_turbo = time.perf_counter() - t0

    vram_turbo = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    # Metric calculations
    num_steps = epochs * math.ceil(len(bench_data) / batch_size)
    ms_vanilla = (t_vanilla / num_steps) * 1000.0
    ms_turbo = (t_turbo / num_steps) * 1000.0
    speedup = ms_vanilla / max(1e-6, ms_turbo)

    vanilla_init_loss = res_vanilla["history"][0]["loss"]
    vanilla_final_loss = res_vanilla["history"][-1]["loss"]
    turbo_init_loss = res_turbo["history"][0]["loss"]
    turbo_final_loss = res_turbo["history"][-1]["loss"]

    vanilla_margin = res_vanilla["history"][-1]["reward_margin"]
    turbo_margin = res_turbo["history"][-1]["reward_margin"]

    soft_clips = sum(h.get("soft_clips_count", 0) for h in res_turbo["history"])

    report = {
        "hardware": device_name,
        "samples_tested": len(bench_data),
        "epochs": epochs,
        "batch_size": batch_size,
        "total_steps": num_steps,
        "vanilla": {
            "total_time_s": round(t_vanilla, 2),
            "step_latency_ms": round(ms_vanilla, 2),
            "peak_vram_mb": round(vram_vanilla, 2),
            "init_loss": vanilla_init_loss,
            "final_loss": vanilla_final_loss,
            "reward_margin": vanilla_margin,
            "epistemic_honesty": res_vanilla["metrics"]["epistemic_honesty_rate"],
        },
        "accelerator_ai": {
            "total_time_s": round(t_turbo, 2),
            "step_latency_ms": round(ms_turbo, 2),
            "peak_vram_mb": round(vram_turbo, 2),
            "init_loss": turbo_init_loss,
            "final_loss": turbo_final_loss,
            "reward_margin": turbo_margin,
            "speedup_ratio": round(speedup, 2),
            "soft_clips_count": soft_clips,
            "epistemic_honesty": res_turbo["metrics"]["epistemic_honesty_rate"],
        },
    }

    # Print Formatted Benchmark Table
    print("\n" + "=" * 78)
    print(f"       BENCHMARK PORÓWNAWCZY DPO: VANILLA PyTorch vs ACCELERATORAI")
    print(f"       Platforma: {device_name} | Próbek: {len(bench_data)} | Epok: {epochs}")
    print("=" * 78)
    print(f"{'Metryka':<32} | {'Vanilla PyTorch':<18} | {'AcceleratorAI Turbo':<20}")
    print("-" * 78)
    print(f"{'Czas całkowity (s)':<32} | {report['vanilla']['total_time_s']:<18} | {report['accelerator_ai']['total_time_s']:<20}")
    print(f"{'Latencja kroku (ms/krok)':<32} | {report['vanilla']['step_latency_ms']:<18} | {report['accelerator_ai']['step_latency_ms']:<20}")
    print(f"{'Szczytowy VRAM GPU (MB)':<32} | {report['vanilla']['peak_vram_mb']:<18} | {report['accelerator_ai']['peak_vram_mb']:<20}")
    print(f"{'Strata początkowa (Loss Init)':<32} | {report['vanilla']['init_loss']:<18} | {report['accelerator_ai']['init_loss']:<20}")
    print(f"{'Strata końcowa (Loss Final)':<32} | {report['vanilla']['final_loss']:<18} | {report['accelerator_ai']['final_loss']:<20}")
    print(f"{'Margines Nagrody (Reward Margin)':<32} | {report['vanilla']['reward_margin']:<18} | {report['accelerator_ai']['reward_margin']:<20}")
    print(f"{'Wskaźnik Prawdomówności (Honesty)':<32} | {report['vanilla']['epistemic_honesty']*100:.1f}%{'':<12} | {report['accelerator_ai']['epistemic_honesty']*100:.1f}%")
    print(f"{'Tłumienia skoków tanh':<32} | {'0 (Hard clip)':<18} | {report['accelerator_ai']['soft_clips_count']}")
    print(f"{'Przyspieszenie (Speedup)':<32} | {'1.00x':<18} | {report['accelerator_ai']['speedup_ratio']:.2f}x")
    print("=" * 78)

    out_file = REPO_ROOT / "profiling_results" / "dpo_accelerator_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Zapisano raport benchmarkowy w: {out_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark DPO: Vanilla vs AcceleratorAI")
    parser.add_argument("--dataset", type=str, default="data/ambassador_dpo_dataset.jsonl")
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()
    dataset_path = REPO_ROOT / args.dataset
    run_benchmark(dataset_path, num_samples=args.samples, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
