"""
Benchmark Runner for Nethical

Orchestrates performance benchmarks for the Nethical governance platform.
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    name: str
    iterations: int = 1000
    warmup_iterations: int = 100
    concurrent_workers: int = 10
    timeout_seconds: float = 60.0
    output_dir: str = "benchmark_results"


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    timestamp: str
    total_iterations: int
    successful_iterations: int
    failed_iterations: int
    total_duration_seconds: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_rps: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Orchestrates benchmark execution."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []

    async def run_scenario(
        self,
        scenario_func: Any,
        scenario_name: str,
    ) -> BenchmarkResult:
        """Run a benchmark scenario."""
        logger.info(f"Running benchmark: {scenario_name}")

        # Warmup
        logger.info(f"Warming up with {self.config.warmup_iterations} iterations")
        for _ in range(self.config.warmup_iterations):
            try:
                await asyncio.wait_for(
                    scenario_func(), timeout=self.config.timeout_seconds
                )
            except Exception:
                pass

        # Main benchmark
        latencies: List[float] = []
        errors: List[str] = []
        successful = 0
        failed = 0

        start_time = time.perf_counter()

        semaphore = asyncio.Semaphore(self.config.concurrent_workers)

        async def run_iteration() -> Optional[float]:
            nonlocal successful, failed
            async with semaphore:
                try:
                    iter_start = time.perf_counter()
                    await asyncio.wait_for(
                        scenario_func(), timeout=self.config.timeout_seconds
                    )
                    iter_duration = (time.perf_counter() - iter_start) * 1000
                    successful += 1
                    return iter_duration
                except Exception as e:
                    failed += 1
                    errors.append(str(e)[:100])
                    return None

        tasks = [run_iteration() for _ in range(self.config.iterations)]
        results = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        total_duration = end_time - start_time

        latencies = [r for r in results if r is not None]

        if latencies:
            latencies.sort()
            avg_latency = sum(latencies) / len(latencies)
            p50 = latencies[int(len(latencies) * 0.50)]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[int(len(latencies) * 0.99)]
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            avg_latency = p50 = p95 = p99 = min_latency = max_latency = 0.0

        throughput = successful / total_duration if total_duration > 0 else 0

        result = BenchmarkResult(
            name=scenario_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_iterations=self.config.iterations,
            successful_iterations=successful,
            failed_iterations=failed,
            total_duration_seconds=total_duration,
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            throughput_rps=throughput,
            errors=errors[:10],  # Limit errors to first 10
            metadata={
                "config": {
                    "iterations": self.config.iterations,
                    "warmup_iterations": self.config.warmup_iterations,
                    "concurrent_workers": self.config.concurrent_workers,
                    "timeout_seconds": self.config.timeout_seconds,
                }
            },
        )

        self.results.append(result)
        logger.info(
            f"Benchmark {scenario_name} completed: "
            f"{throughput:.2f} rps, avg={avg_latency:.2f}ms, p99={p99:.2f}ms"
        )

        return result

    def save_results(self, output_path: Optional[str] = None) -> str:
        """Save benchmark results to JSON file."""
        if output_path is None:
            os.makedirs(self.config.output_dir, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config.output_dir, f"benchmark_{timestamp}.json"
            )

        results_data = {
            "benchmark_run": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "name": self.config.name,
                    "iterations": self.config.iterations,
                    "warmup_iterations": self.config.warmup_iterations,
                    "concurrent_workers": self.config.concurrent_workers,
                    "timeout_seconds": self.config.timeout_seconds,
                },
            },
            "results": [
                {
                    "name": r.name,
                    "timestamp": r.timestamp,
                    "total_iterations": r.total_iterations,
                    "successful_iterations": r.successful_iterations,
                    "failed_iterations": r.failed_iterations,
                    "total_duration_seconds": r.total_duration_seconds,
                    "avg_latency_ms": r.avg_latency_ms,
                    "p50_latency_ms": r.p50_latency_ms,
                    "p95_latency_ms": r.p95_latency_ms,
                    "p99_latency_ms": r.p99_latency_ms,
                    "min_latency_ms": r.min_latency_ms,
                    "max_latency_ms": r.max_latency_ms,
                    "throughput_rps": r.throughput_rps,
                    "errors": r.errors,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Results saved to {output_path}")
        return output_path


async def run_default_benchmarks() -> None:
    """Run all default benchmark scenarios."""
    from benchmarks.scenarios.throughput import ThroughputScenario
    from benchmarks.scenarios.latency import LatencyScenario

    config = BenchmarkConfig(
        name="default",
        iterations=int(os.getenv("BENCHMARK_ITERATIONS", "1000")),
        warmup_iterations=int(os.getenv("BENCHMARK_WARMUP", "100")),
        concurrent_workers=int(os.getenv("BENCHMARK_WORKERS", "10")),
    )

    runner = BenchmarkRunner(config)

    # Run throughput scenario
    throughput = ThroughputScenario()
    await runner.run_scenario(throughput.run, "throughput")

    # Run latency scenario
    latency = LatencyScenario()
    await runner.run_scenario(latency.run, "latency")

    runner.save_results()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_default_benchmarks())
