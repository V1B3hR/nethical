"""
Performance Profiling and Regression Testing

This module provides performance profiling capabilities and regression testing
for the Nethical project.

Features:
- Function and method profiling (sync and async)
- Performance regression detection with configurable tolerance
- Benchmark management and persistence
- Profiling reports and visualization
- Context manager sections for arbitrary code blocks
- Thread-safe result collection
- Optional detailed cProfile summaries for sync functions

Environment:
- Set NETHICAL_PROFILING=0 to disable profiling globally (default: enabled)
"""

from __future__ import annotations

import asyncio
import cProfile
import contextlib
import functools
import io
import json
import logging
import os
import pstats
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


logger = logging.getLogger("nethical.performanceprofiling")


@dataclass
class ProfileResult:
    """Performance profile result"""

    function_name: str
    execution_time_ms: float
    call_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stats: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "function_name": self.function_name,
            "execution_time_ms": self.execution_time_ms,
            "call_count": self.call_count,
            "timestamp": self.timestamp.isoformat(),
            "stats": self.stats,
            "metadata": self.metadata,
        }


@dataclass
class Benchmark:
    """Performance benchmark"""

    name: str
    baseline_ms: float
    tolerance_pct: float = 10.0  # 10% tolerance by default
    samples: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def check_regression(self, current_ms: float) -> tuple[bool, float]:
        """
        Check if current performance is a regression

        Returns:
            (is_regression, percent_change)
        """
        if self.baseline_ms <= 0:
            return False, 0.0
        percent_change = ((current_ms - self.baseline_ms) / self.baseline_ms) * 100
        is_regression = percent_change > self.tolerance_pct
        return is_regression, percent_change

    def update_baseline(self, new_baseline_ms: float):
        """Update baseline performance"""
        self.baseline_ms = new_baseline_ms
        self.samples = []
        self.last_updated = datetime.now(timezone.utc)

    def add_sample(self, sample_ms: float):
        """Add a performance sample"""
        self.samples.append(sample_ms)
        # Keep last 100 samples
        if len(self.samples) > 100:
            self.samples = self.samples[-100:]
        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        recent_avg = statistics.mean(self.samples) if self.samples else None
        recent_std = statistics.stdev(self.samples) if len(self.samples) > 1 else None
        return {
            "name": self.name,
            "baseline_ms": self.baseline_ms,
            "tolerance_pct": self.tolerance_pct,
            "samples_count": len(self.samples),
            "recent_avg_ms": recent_avg,
            "recent_std_ms": recent_std,
            "last_updated": self.last_updated.isoformat(),
        }


class PerformanceProfiler:
    """Profile function and method performance"""

    def __init__(
        self, results_dir: Union[str, Path] = "profiling_results", enabled: Optional[bool] = None
    ):
        # Enable/disable can be controlled by env var, default enabled
        if enabled is None:
            env = os.getenv("NETHICAL_PROFILING", "1").strip()
            self.enabled = env not in ("0", "false", "False", "")
        else:
            self.enabled = enabled

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, List[ProfileResult]] = {}
        self.benchmarks: Dict[str, Benchmark] = {}

        self._lock = threading.RLock()
        self._benchmarks_file = self.results_dir / "benchmarks.json"
        self._load_benchmarks()

    def is_enabled(self) -> bool:
        return self.enabled

    def set_enabled(self, value: bool) -> None:
        self.enabled = value

    def profile(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        detailed: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Decorator to profile function performance (sync and async).

        Usage:
            @profiler.profile()
            def my_function():
                pass

            @profiler.profile(name="custom_name", detailed=True)
            def my_function():
                pass
        """

        def decorator(f: Callable) -> Callable:
            func_name = name or f.__name__
            is_async = asyncio.iscoroutinefunction(f)

            if not self.enabled:
                # Fast-path: no wrapping overhead when disabled
                return f

            if is_async:

                @functools.wraps(f)
                async def async_wrapper(*args, **kwargs):
                    start_ns = time.perf_counter_ns()
                    result = await f(*args, **kwargs)
                    end_ns = time.perf_counter_ns()
                    execution_time = (end_ns - start_ns) / 1_000_000.0

                    profile_result = ProfileResult(
                        function_name=func_name,
                        execution_time_ms=execution_time,
                        call_count=1,
                        metadata=(metadata or {}) | {"async": True, "detailed": False},
                    )
                    self._record_result(func_name, profile_result, execution_time)
                    return result

                return async_wrapper

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                if detailed:
                    # Detailed profiling with cProfile (sync only)
                    profiler = cProfile.Profile()
                    profiler.enable()
                    start_ns = time.perf_counter_ns()
                    result = f(*args, **kwargs)
                    end_ns = time.perf_counter_ns()
                    profiler.disable()

                    execution_time = (end_ns - start_ns) / 1_000_000.0
                    stats_struct = self._extract_cprofile_stats(profiler, top_n=20)

                    profile_result = ProfileResult(
                        function_name=func_name,
                        execution_time_ms=execution_time,
                        call_count=1,
                        stats=stats_struct,
                        metadata=(metadata or {}) | {"async": False, "detailed": True},
                    )
                else:
                    # Simple timing
                    start_ns = time.perf_counter_ns()
                    result = f(*args, **kwargs)
                    end_ns = time.perf_counter_ns()

                    execution_time = (end_ns - start_ns) / 1_000_000.0
                    profile_result = ProfileResult(
                        function_name=func_name,
                        execution_time_ms=execution_time,
                        call_count=1,
                        metadata=(metadata or {}) | {"async": False, "detailed": False},
                    )

                self._record_result(func_name, profile_result, execution_time)
                return result

            return wrapper

        if func is None:
            return decorator
        else:
            return decorator(func)

    @contextlib.contextmanager
    def profile_section(self, name: str, *, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager to profile an arbitrary code block as a 'section'.

        Usage:
            with profiler.profile_section("load-data"):
                load_data()
        """
        section_name = f"section:{name}"
        if not self.enabled:
            yield
            return
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            execution_time = (end_ns - start_ns) / 1_000_000.0
            profile_result = ProfileResult(
                function_name=section_name,
                execution_time_ms=execution_time,
                call_count=1,
                metadata=(metadata or {}) | {"section": True},
            )
            self._record_result(section_name, profile_result, execution_time)

    def benchmark_function(
        self, func: Callable, iterations: int = 100, *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark a function by running it multiple times

        Args:
            func: Function to benchmark
            iterations: Number of iterations
            *args, **kwargs: Arguments to pass to function

        Returns:
            Benchmark statistics
        """
        func_name = func.__name__
        times_ms: List[float] = []

        logger.info("Benchmarking %s with %d iterations...", func_name, iterations)

        for _ in range(iterations):
            start_ns = time.perf_counter_ns()
            func(*args, **kwargs)
            end_ns = time.perf_counter_ns()
            times_ms.append((end_ns - start_ns) / 1_000_000.0)

        stats = {
            "function": func_name,
            "iterations": iterations,
            "mean_ms": statistics.mean(times_ms),
            "median_ms": statistics.median(times_ms),
            "std_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
            "p95_ms": self._percentile(times_ms, 95.0),
            "p99_ms": self._percentile(times_ms, 99.0),
        }

        logger.info(
            "Results for %s: mean=%.2fms, median=%.2fms, p95=%.2fms, p99=%.2fms",
            func_name,
            stats["mean_ms"],
            stats["median_ms"],
            stats["p95_ms"],
            stats["p99_ms"],
        )

        return stats

    def set_benchmark(self, name: str, baseline_ms: float, tolerance_pct: float = 10.0):
        """Set a performance benchmark"""
        with self._lock:
            self.benchmarks[name] = Benchmark(
                name=name,
                baseline_ms=baseline_ms,
                tolerance_pct=tolerance_pct,
            )
            self._save_benchmarks()
        logger.info(
            "Set benchmark '%s': %.2fms (tolerance: ±%.1f%%)", name, baseline_ms, tolerance_pct
        )

    def get_results(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get profiling results"""
        with self._lock:
            if function_name:
                results = self.results.get(function_name, [])
                if not results:
                    return {}

                times = [r.execution_time_ms for r in results]
                return {
                    "function": function_name,
                    "samples": len(times),
                    "mean_ms": statistics.mean(times),
                    "median_ms": statistics.median(times),
                    "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
                    "min_ms": min(times),
                    "max_ms": max(times),
                }
            else:
                # Return summary for all functions
                summary: Dict[str, Any] = {}
                for func_name, res_list in self.results.items():
                    times = [r.execution_time_ms for r in res_list]
                    summary[func_name] = {
                        "samples": len(times),
                        "mean_ms": statistics.mean(times),
                        "median_ms": statistics.median(times),
                    }
                return summary

    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate a performance report (text)"""
        report: List[str] = []
        report.append("=" * 80)
        report.append("PERFORMANCE PROFILING REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        report.append("")

        # Benchmarks
        with self._lock:
            if self.benchmarks:
                report.append("BENCHMARKS")
                report.append("-" * 80)
                for name, benchmark in sorted(self.benchmarks.items(), key=lambda x: x[0]):
                    report.append(f"  {name}:")
                    report.append(f"    Baseline: {benchmark.baseline_ms:.2f}ms")
                    report.append(f"    Tolerance: ±{benchmark.tolerance_pct}%")
                    if benchmark.samples:
                        recent_avg = statistics.mean(benchmark.samples)
                        is_regression, pct_change = benchmark.check_regression(recent_avg)
                        status = "REGRESSION" if is_regression else "OK"
                        report.append(f"    Recent Avg: {recent_avg:.2f}ms")
                        report.append(f"    Status: {status} ({pct_change:+.1f}%)")
                    report.append(f"    Last Updated: {benchmark.last_updated.isoformat()}")
                    report.append("")

            # Results
            if self.results:
                report.append("PROFILING RESULTS")
                report.append("-" * 80)
                for func_name in sorted(self.results.keys()):
                    stats = self.get_results(func_name)
                    if not stats:
                        continue
                    report.append(f"  {func_name}:")
                    report.append(f"    Samples: {stats['samples']}")
                    report.append(f"    Mean: {stats['mean_ms']:.2f}ms")
                    report.append(f"    Median: {stats['median_ms']:.2f}ms")
                    report.append(f"    Min: {stats['min_ms']:.2f}ms")
                    report.append(f"    Max: {stats['max_ms']:.2f}ms")
                    report.append("")

        report_text = "\n".join(report)

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info("Report saved to: %s", output_path)

        return report_text

    # Internal helpers

    def _extract_cprofile_stats(
        self, profiler: cProfile.Profile, top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Extract a structured cProfile summary.
        Returns a dict with 'top' list and 'text' truncated representation.
        """
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(top_n)
        text_output = s.getvalue()

        # ps.stats: {(filename, line, funcname): (cc, nc, tt, ct, callers)}
        entries = []
        for (filename, line, funcname), (cc, nc, tt, ct, callers) in ps.stats.items():
            entries.append(
                {
                    "file": filename,
                    "line": line,
                    "function": funcname,
                    "prim_calls": cc,
                    "total_calls": nc,
                    "tottime_s": tt,
                    "cumtime_s": ct,
                    "percall_tottime_ms": (tt / max(cc, 1)) * 1000.0,
                    "percall_cumtime_ms": (ct / max(nc, 1)) * 1000.0,
                }
            )

        # Top by cumulative time
        entries.sort(key=lambda e: e["cumtime_s"], reverse=True)
        top_entries = entries[:top_n]

        return {
            "top": top_entries,
            "text": text_output[:2000],  # Limit size to avoid bloating results
        }

    def _record_result(
        self, func_name: str, profile_result: ProfileResult, execution_time: float
    ) -> None:
        """Thread-safe storage and regression checking"""
        with self._lock:
            if func_name not in self.results:
                self.results[func_name] = []
            self.results[func_name].append(profile_result)

            # Check for regression if benchmark exists
            if func_name in self.benchmarks:
                benchmark = self.benchmarks[func_name]
                benchmark.add_sample(execution_time)
                is_regression, pct_change = benchmark.check_regression(execution_time)
                if is_regression:
                    logger.warning(
                        "Performance regression detected for %s: %.1f%% slower than baseline (%.2fms vs %.2fms)",
                        func_name,
                        pct_change,
                        execution_time,
                        benchmark.baseline_ms,
                    )
                # Persist benchmarks periodically
                self._save_benchmarks()

    def _save_benchmarks(self):
        """Save benchmarks to file"""
        with self._lock:
            data = {name: bench.to_dict() for name, bench in self.benchmarks.items()}
            self._benchmarks_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._benchmarks_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def _load_benchmarks(self):
        """Load benchmarks from file"""
        if self._benchmarks_file.exists():
            try:
                with open(self._benchmarks_file, encoding="utf-8") as f:
                    data = json.load(f)
                    for name, bench_data in data.items():
                        self.benchmarks[name] = Benchmark(
                            name=name,
                            baseline_ms=bench_data["baseline_ms"],
                            tolerance_pct=bench_data.get("tolerance_pct", 10.0),
                            samples=[],  # Do not restore historical samples to keep file small
                            last_updated=datetime.fromisoformat(
                                bench_data.get(
                                    "last_updated", datetime.now(timezone.utc).isoformat()
                                )
                            ),
                        )
            except Exception as e:
                logger.error("Failed to load benchmarks: %s", e)

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        # Use statistics.quantiles for robustness; fallback to max for small samples
        if len(values) < 2:
            return values[0]
        try:
            # statistics.quantiles expects n cut points; compute nearest-rank approximation
            # We'll derive index manually for clarity
            sorted_vals = sorted(values)
            rank = (p / 100.0) * (len(sorted_vals) - 1)
            lower = int(rank)
            upper = min(lower + 1, len(sorted_vals) - 1)
            weight = rank - lower
            return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight
        except Exception:
            return max(values)


# Global profiler instance
_profiler = PerformanceProfiler()


def profile(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    detailed: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Convenience decorator using global profiler"""
    return _profiler.profile(func, name=name, detailed=detailed, metadata=metadata)


@contextlib.contextmanager
def profile_section(name: str, *, metadata: Optional[Dict[str, Any]] = None):
    """Convenience context manager using global profiler"""
    with _profiler.profile_section(name, metadata=metadata):
        yield


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance"""
    return _profiler


if __name__ == "__main__":
    # Basic demo usage
    logging.basicConfig(level=logging.INFO)
    profiler = PerformanceProfiler()

    @profiler.profile()
    def example_function():
        time.sleep(0.01)
        return sum(range(1000))

    async def async_task():
        await asyncio.sleep(0.005)
        return sum(range(500))

    @profiler.profile(detailed=True)
    def cpu_heavy():
        s = 0
        for i in range(50000):
            s += i * i
        return s

    # Run the functions a few times
    for _ in range(5):
        example_function()
        cpu_heavy()

    # Profile a section
    with profiler.profile_section("section-demo"):
        time.sleep(0.003)

    # Async demo
    asyncio.run(profiler.profile()(async_task)())

    # Generate report
    print(profiler.generate_report())
