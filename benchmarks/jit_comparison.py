"""
JIT Comparison Benchmarks

Benchmarks comparing JIT-compiled functions vs pure Python implementations.
Target: 10-100x speedup for numerical operations.
"""

import time
from typing import Dict, Any, List, Callable

import numpy as np


def benchmark_function(
    func: Callable,
    args: tuple,
    iterations: int = 1000,
    warmup: int = 10,
) -> Dict[str, Any]:
    """
    Benchmark a function.

    Args:
        func: Function to benchmark
        args: Arguments to pass to function
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        Benchmark results
    """
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    n = len(latencies)

    return {
        "mean_ms": sum(latencies) / n,
        "p50_ms": latencies[int(n * 0.50)],
        "p95_ms": latencies[int(n * 0.95)],
        "p99_ms": latencies[int(n * 0.99)],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "iterations": iterations,
    }


def compare_implementations(
    jit_func: Callable,
    python_func: Callable,
    args: tuple,
    name: str,
    iterations: int = 1000,
) -> Dict[str, Any]:
    """
    Compare JIT and Python implementations.

    Args:
        jit_func: JIT-compiled function
        python_func: Pure Python function
        args: Arguments to pass
        name: Benchmark name
        iterations: Number of iterations

    Returns:
        Comparison results
    """
    jit_results = benchmark_function(jit_func, args, iterations)
    python_results = benchmark_function(python_func, args, iterations)

    speedup = (
        python_results["mean_ms"] / jit_results["mean_ms"]
        if jit_results["mean_ms"] > 0
        else 0
    )

    return {
        "name": name,
        "jit": jit_results,
        "python": python_results,
        "speedup": speedup,
        "target_met": speedup >= 10,  # Target: 10x speedup
    }


# Pure Python implementations for comparison


def python_risk_score(
    severities: np.ndarray,
    confidences: np.ndarray,
) -> float:
    """Pure Python risk score calculation."""
    if len(severities) == 0:
        return 0.0

    weighted_sum = 0.0
    for sev, conf in zip(severities, confidences):
        normalized = sev / 5.0
        weighted_sum += normalized * conf

    return min(1.0, weighted_sum / len(severities))


def python_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Pure Python cosine similarity."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def python_batch_similarity(
    vectors: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Pure Python batch similarity."""
    similarities = []
    ref_norm = sum(a * a for a in reference) ** 0.5

    for vec in vectors:
        vec_norm = sum(a * a for a in vec) ** 0.5
        if ref_norm == 0 or vec_norm == 0:
            similarities.append(0.0)
        else:
            dot = sum(a * b for a, b in zip(vec, reference))
            similarities.append(dot / (vec_norm * ref_norm))

    return np.array(similarities)


def python_outlier_detection(
    values: np.ndarray,
    threshold: float = 3.0,
) -> np.ndarray:
    """Pure Python z-score outlier detection."""
    if len(values) == 0:
        return np.array([], dtype=bool)

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance**0.5

    if std == 0:
        return np.zeros(len(values), dtype=bool)

    return np.array([abs((v - mean) / std) > threshold for v in values])


def run_all_benchmarks() -> List[Dict[str, Any]]:
    """
    Run all JIT comparison benchmarks.

    Returns:
        List of benchmark results
    """
    results = []

    try:
        from nethical.core.jit_optimizations import (
            calculate_risk_score_jit,
            cosine_similarity_jit,
            batch_cosine_similarity_jit,
            detect_outliers_zscore_jit,
            NUMBA_AVAILABLE,
        )

        if not NUMBA_AVAILABLE:
            print("Numba not available, skipping JIT benchmarks")
            return []

        # Benchmark 1: Risk Score Calculation
        severities = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20, dtype=np.float64)
        confidences = np.array([0.8, 0.9, 0.7, 0.85, 0.95] * 20, dtype=np.float64)

        result = compare_implementations(
            lambda s, c: calculate_risk_score_jit(s, c),
            python_risk_score,
            (severities, confidences),
            "Risk Score Calculation",
        )
        results.append(result)
        print(f"Risk Score: {result['speedup']:.1f}x speedup")

        # Benchmark 2: Cosine Similarity
        vec1 = np.random.randn(256).astype(np.float64)
        vec2 = np.random.randn(256).astype(np.float64)

        result = compare_implementations(
            cosine_similarity_jit,
            python_cosine_similarity,
            (vec1, vec2),
            "Cosine Similarity",
        )
        results.append(result)
        print(f"Cosine Similarity: {result['speedup']:.1f}x speedup")

        # Benchmark 3: Batch Similarity
        vectors = np.random.randn(100, 128).astype(np.float64)
        reference = np.random.randn(128).astype(np.float64)

        result = compare_implementations(
            batch_cosine_similarity_jit,
            python_batch_similarity,
            (vectors, reference),
            "Batch Cosine Similarity",
        )
        results.append(result)
        print(f"Batch Similarity: {result['speedup']:.1f}x speedup")

        # Benchmark 4: Outlier Detection
        values = np.random.randn(1000).astype(np.float64)

        result = compare_implementations(
            detect_outliers_zscore_jit,
            python_outlier_detection,
            (values,),
            "Z-Score Outlier Detection",
        )
        results.append(result)
        print(f"Outlier Detection: {result['speedup']:.1f}x speedup")

    except ImportError as e:
        print(f"Import error: {e}")

    return results


def print_benchmark_report(results: List[Dict[str, Any]]):
    """Print formatted benchmark report."""
    print("\n" + "=" * 60)
    print("JIT COMPARISON BENCHMARK REPORT")
    print("=" * 60)

    for result in results:
        print(f"\n{result['name']}")
        print("-" * 40)
        print(f"  JIT Mean:    {result['jit']['mean_ms']:.4f} ms")
        print(f"  Python Mean: {result['python']['mean_ms']:.4f} ms")
        print(f"  Speedup:     {result['speedup']:.1f}x")
        print(f"  Target Met:  {'✓' if result['target_met'] else '✗'} (10x)")

    print("\n" + "=" * 60)

    met = sum(1 for r in results if r["target_met"])
    print(f"Summary: {met}/{len(results)} benchmarks met 10x speedup target")


if __name__ == "__main__":
    print("Running JIT comparison benchmarks...")
    results = run_all_benchmarks()
    if results:
        print_benchmark_report(results)
