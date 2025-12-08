"""
Soak Load Test - Performance Requirement 5.3

Tests system stability over extended period (2 hours).
Detects memory leaks and performance degradation.

Memory leak threshold: <5% growth over test duration

Run with: pytest tests/performance/test_load_soak.py -v -s --run-soak
"""

import pytest
import asyncio
import time
import json
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from nethical.core.governance import (
    SafetyGovernance,
    MonitoringConfig,
    AgentAction,
    ActionType,
)


class SoakTestMetrics:
    """Collect and store soak test metrics"""

    def __init__(self):
        self.start_time = time.time()
        self.requests = []
        self.memory_samples = []
        self.cpu_samples = []
        self.gc_stats = []

    def record_request(self, duration_ms: float, success: bool):
        """Record a single request"""
        self.requests.append(
            {"timestamp": time.time(), "duration_ms": duration_ms, "success": success}
        )

    def record_system_metrics(self):
        """Record system metrics including memory"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()

        self.memory_samples.append(
            {
                "timestamp": time.time(),
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
            }
        )

    def get_memory_leak_analysis(self) -> Dict[str, Any]:
        """
        Analyze memory samples for leaks

        Returns analysis including:
        - Initial memory
        - Final memory
        - Growth percentage
        - Leak detected (>5% growth)
        """
        if len(self.memory_samples) < 2:
            return None

        # Use first 10% as baseline and last 10% as final
        baseline_count = max(1, len(self.memory_samples) // 10)
        initial_samples = self.memory_samples[:baseline_count]
        final_samples = self.memory_samples[-baseline_count:]

        initial_memory = sum(s["memory_mb"] for s in initial_samples) / len(
            initial_samples
        )
        final_memory = sum(s["memory_mb"] for s in final_samples) / len(final_samples)

        growth_mb = final_memory - initial_memory
        growth_percent = (growth_mb / initial_memory * 100) if initial_memory > 0 else 0

        leak_detected = growth_percent > 5.0

        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "growth_mb": growth_mb,
            "growth_percent": growth_percent,
            "leak_detected": leak_detected,
            "samples_count": len(self.memory_samples),
            "max_memory_mb": max(s["memory_mb"] for s in self.memory_samples),
            "min_memory_mb": min(s["memory_mb"] for s in self.memory_samples),
        }

    def get_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance over time"""
        if not self.requests:
            return None

        # Split requests into time windows
        duration = time.time() - self.start_time
        window_size = max(1, duration / 10)  # 10 windows

        windows = []
        for i in range(10):
            window_start = self.start_time + (i * window_size)
            window_end = window_start + window_size

            window_requests = [
                r for r in self.requests if window_start <= r["timestamp"] < window_end
            ]

            if window_requests:
                response_times = [r["duration_ms"] for r in window_requests]
                windows.append(
                    {
                        "window": i,
                        "count": len(window_requests),
                        "mean_response": sum(response_times) / len(response_times),
                        "p95_response": (
                            sorted(response_times)[int(len(response_times) * 0.95)]
                            if len(response_times) > 1
                            else response_times[0]
                        ),
                    }
                )

        # Check for performance degradation
        if len(windows) >= 2:
            first_window = windows[0]
            last_window = windows[-1]

            degradation_percent = (
                (
                    (last_window["mean_response"] - first_window["mean_response"])
                    / first_window["mean_response"]
                    * 100
                )
                if first_window["mean_response"] > 0
                else 0
            )
        else:
            degradation_percent = 0

        return {
            "windows": windows,
            "degradation_percent": degradation_percent,
            "degradation_detected": degradation_percent > 20.0,  # >20% degradation
        }

    def get_stats(self) -> Dict[str, Any]:
        """Calculate overall statistics"""
        duration = time.time() - self.start_time
        successful = sum(1 for r in self.requests if r["success"])

        response_times = [r["duration_ms"] for r in self.requests]

        return {
            "duration_seconds": duration,
            "duration_hours": duration / 3600,
            "total_requests": len(self.requests),
            "successful_requests": successful,
            "failed_requests": len(self.requests) - successful,
            "success_rate": successful / len(self.requests) if self.requests else 0,
            "throughput_rps": len(self.requests) / duration if duration > 0 else 0,
            "response_times": {
                "mean": (
                    sum(response_times) / len(response_times) if response_times else 0
                ),
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "p95": (
                    sorted(response_times)[int(len(response_times) * 0.95)]
                    if response_times
                    else 0
                ),
                "p99": (
                    sorted(response_times)[int(len(response_times) * 0.99)]
                    if response_times
                    else 0
                ),
            },
            "memory_analysis": self.get_memory_leak_analysis(),
            "performance_analysis": self.get_performance_analysis(),
        }

    def save_artifacts(self, output_dir: Path) -> Dict[str, str]:
        """Save test artifacts"""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        stats = self.get_stats()

        # Save raw data
        raw_file = output_dir / f"soak_test_raw_{timestamp}.json"
        with open(raw_file, "w") as f:
            json.dump(
                {
                    "requests": self.requests,
                    "memory_samples": self.memory_samples,
                },
                f,
                indent=2,
            )

        # Save summary report
        report_file = output_dir / f"soak_test_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(stats, f, indent=2)

        # Save human-readable report
        md_file = output_dir / f"soak_test_report_{timestamp}.md"
        with open(md_file, "w") as f:
            f.write("# Soak Load Test Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            f.write(f"## Summary\n\n")
            f.write(
                f"- Duration: {stats['duration_hours']:.2f} hours ({stats['duration_seconds']:.0f} seconds)\n"
            )
            f.write(f"- Total Requests: {stats['total_requests']}\n")
            f.write(f"- Success Rate: {stats['success_rate']*100:.2f}%\n")
            f.write(f"- Throughput: {stats['throughput_rps']:.2f} req/sec\n\n")

            f.write(f"## Response Times (ms)\n\n")
            f.write(f"- Mean: {stats['response_times']['mean']:.2f}\n")
            f.write(f"- P95: {stats['response_times']['p95']:.2f}\n")
            f.write(f"- P99: {stats['response_times']['p99']:.2f}\n\n")

            # Memory leak analysis
            if stats["memory_analysis"]:
                mem = stats["memory_analysis"]
                f.write(f"## Memory Leak Analysis\n\n")
                f.write(f"- Initial Memory: {mem['initial_memory_mb']:.2f} MB\n")
                f.write(f"- Final Memory: {mem['final_memory_mb']:.2f} MB\n")
                f.write(
                    f"- Growth: {mem['growth_mb']:.2f} MB ({mem['growth_percent']:+.2f}%)\n"
                )
                f.write(f"- Max Memory: {mem['max_memory_mb']:.2f} MB\n")
                f.write(
                    f"- Memory Leak Detected: {'❌ YES' if mem['leak_detected'] else '✅ NO'}\n\n"
                )

            # Performance degradation analysis
            if stats["performance_analysis"]:
                perf = stats["performance_analysis"]
                f.write(f"## Performance Degradation Analysis\n\n")
                f.write(
                    f"- Mean Response Time Degradation: {perf['degradation_percent']:+.2f}%\n"
                )
                f.write(
                    f"- Significant Degradation: {'❌ YES' if perf['degradation_detected'] else '✅ NO'}\n\n"
                )

            # Overall result
            passed = True
            if stats["memory_analysis"]:
                passed = passed and not stats["memory_analysis"]["leak_detected"]
            if stats["performance_analysis"]:
                passed = (
                    passed and not stats["performance_analysis"]["degradation_detected"]
                )
            passed = passed and stats["success_rate"] >= 0.95

            f.write(f"## Test Result: {'✅ PASSED' if passed else '❌ FAILED'}\n\n")
            f.write("### Pass Criteria\n\n")
            f.write(
                f"- Memory growth < 5%: {'✅' if not stats['memory_analysis']['leak_detected'] else '❌'}\n"
            )
            f.write(
                f"- Performance degradation < 20%: {'✅' if not stats['performance_analysis']['degradation_detected'] else '❌'}\n"
            )
            f.write(
                f"- Success rate ≥ 95%: {'✅' if stats['success_rate'] >= 0.95 else '❌'}\n"
            )

        return {
            "raw_file": str(raw_file),
            "report_file": str(report_file),
            "md_file": str(md_file),
        }


@pytest.fixture
def governance():
    """Create governance instance for testing"""
    config = MonitoringConfig(enable_persistence=False)
    return SafetyGovernance(config)


@pytest.fixture
def output_dir():
    """Output directory for test artifacts"""
    return Path("tests/performance/results/load_tests")


async def run_soak_test(
    governance: SafetyGovernance, duration_seconds: int, target_rps: int
) -> SoakTestMetrics:
    """
    Run soak test

    Args:
        governance: Governance instance
        duration_seconds: Test duration
        target_rps: Target requests per second
    """
    metrics = SoakTestMetrics()
    end_time = time.time() + duration_seconds
    request_interval = 1.0 / target_rps if target_rps > 0 else 0.1

    request_id = 0
    next_metric_time = time.time() + 10  # Record metrics every 10 seconds

    while time.time() < end_time:
        start = time.time()

        # Create and evaluate action
        action = AgentAction(
            action_id=f"soak_test_{request_id}",
            agent_id="soak_test_agent",
            action_type=ActionType.QUERY,
            content=f"Test query number {request_id} for soak testing",
        )

        try:
            await governance.evaluate_action(action)
            duration_ms = (time.time() - start) * 1000
            metrics.record_request(duration_ms, True)
        except Exception:
            duration_ms = (time.time() - start) * 1000
            metrics.record_request(duration_ms, False)

        # Record system metrics periodically
        if time.time() >= next_metric_time:
            metrics.record_system_metrics()
            next_metric_time = time.time() + 10

            # Print progress
            elapsed = time.time() - metrics.start_time
            print(
                f"  Progress: {elapsed/60:.1f} min / {duration_seconds/60:.1f} min ({elapsed/duration_seconds*100:.1f}%)"
            )

        # Sleep to maintain target RPS
        elapsed = time.time() - start
        if elapsed < request_interval:
            await asyncio.sleep(request_interval - elapsed)

        request_id += 1

    # Final metrics
    metrics.record_system_metrics()

    return metrics


@pytest.mark.asyncio
@pytest.mark.slow
async def test_soak_short(governance, output_dir):
    """
    Short soak test (5 minutes) - for CI/CD

    Validates:
    - Memory growth < 5%
    - Performance degradation < 20%
    - Success rate > 95%
    """
    print("\n=== Starting Short Soak Test (5 minutes) ===")

    # Run soak test
    metrics = await run_soak_test(
        governance=governance,
        duration_seconds=300,  # 5 minutes
        target_rps=5,  # 5 requests per second
    )

    # Get statistics
    stats = metrics.get_stats()
    mem = stats["memory_analysis"]
    perf = stats["performance_analysis"]

    print(f"\nTest completed:")
    print(f"  Duration: {stats['duration_hours']:.2f} hours")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Success rate: {stats['success_rate']*100:.2f}%")
    print(f"  Memory growth: {mem['growth_percent']:+.2f}%")
    print(f"  Performance degradation: {perf['degradation_percent']:+.2f}%")

    # Save artifacts
    files = metrics.save_artifacts(output_dir)
    print(f"\nArtifacts saved:")
    for key, path in files.items():
        print(f"  {key}: {path}")

    # Validate
    assert not mem[
        "leak_detected"
    ], f"Memory leak detected: {mem['growth_percent']:.2f}% growth"
    assert not perf[
        "degradation_detected"
    ], f"Performance degraded: {perf['degradation_percent']:.2f}%"
    assert (
        stats["success_rate"] >= 0.95
    ), f"Success rate too low: {stats['success_rate']*100:.2f}%"


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.skipif(
    "not config.getoption('--run-soak')", reason="Soak tests not enabled"
)
async def test_soak_2hour(governance, output_dir):
    """
    Full soak test (2 hours)

    Enable with: pytest --run-soak

    Validates:
    - Memory growth < 5% over 2 hours
    - No performance degradation > 20%
    - Success rate > 95%
    - System remains stable
    """
    print("\n=== Starting Full Soak Test (2 hours) ===")
    print("This test will run for 2 hours...")

    # Run soak test
    metrics = await run_soak_test(
        governance=governance,
        duration_seconds=7200,  # 2 hours
        target_rps=10,  # 10 requests per second
    )

    # Get statistics
    stats = metrics.get_stats()
    mem = stats["memory_analysis"]
    perf = stats["performance_analysis"]

    print(f"\n=== Test Completed ===")
    print(f"Duration: {stats['duration_hours']:.2f} hours")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Success rate: {stats['success_rate']*100:.2f}%")
    print(f"Memory growth: {mem['growth_percent']:+.2f}%")
    print(f"Performance degradation: {perf['degradation_percent']:+.2f}%")

    # Save artifacts
    files = metrics.save_artifacts(output_dir)
    print(f"\nArtifacts saved:")
    for key, path in files.items():
        print(f"  {key}: {path}")

    # Validate
    print("\n=== Validation ===")

    memory_pass = not mem["leak_detected"]
    print(
        f"Memory growth < 5%: {'✅ PASS' if memory_pass else '❌ FAIL'} ({mem['growth_percent']:+.2f}%)"
    )

    perf_pass = not perf["degradation_detected"]
    print(
        f"Performance degradation < 20%: {'✅ PASS' if perf_pass else '❌ FAIL'} ({perf['degradation_percent']:+.2f}%)"
    )

    success_pass = stats["success_rate"] >= 0.95
    print(
        f"Success rate ≥ 95%: {'✅ PASS' if success_pass else '❌ FAIL'} ({stats['success_rate']*100:.2f}%)"
    )

    overall_pass = memory_pass and perf_pass and success_pass
    print(f"\nOverall: {'✅ PASSED' if overall_pass else '❌ FAILED'}")

    # Assert
    assert memory_pass, f"Memory leak detected: {mem['growth_percent']:.2f}% growth"
    assert perf_pass, f"Performance degraded: {perf['degradation_percent']:.2f}%"
    assert success_pass, f"Success rate too low: {stats['success_rate']*100:.2f}%"


# Note: pytest_addoption is defined in tests/conftest.py
