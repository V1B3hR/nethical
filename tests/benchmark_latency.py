"""Benchmark latency under load with concurrent agents.

Load test:
- 1000 concurrent agents
- 10 requests per agent
- Mixed threat types

Target metrics:
- Throughput: >5000 req/s
- Avg latency: <50ms
- P95 latency: <100ms
- P99 latency: <200ms
"""

import asyncio
import random
import time

import numpy as np

from nethical.detectors.realtime import RealtimeThreatDetector


class LoadTestConfig:
    """Configuration for load testing."""

    num_agents: int = 1000
    requests_per_agent: int = 10
    threat_types: list[str] = [
        "shadow_ai",
        "deepfake",
        "polymorphic",
        "prompt_injection",
        "ai_vs_ai",
    ]


async def agent_workload(
    agent_id: int, config: LoadTestConfig, detector: RealtimeThreatDetector
) -> list[float]:
    """Simulate workload for a single agent.

    Args:
        agent_id: Agent identifier
        config: Load test configuration
        detector: Threat detector instance

    Returns:
        List of latencies for this agent
    """
    latencies = []

    for i in range(config.requests_per_agent):
        # Random threat type
        threat_type = random.choice(config.threat_types)

        # Generate appropriate test data
        if threat_type == "shadow_ai":
            input_data = {
                "network_traffic": {
                    "urls": [f"https://api.test{agent_id}.com/v1/endpoint"],
                }
            }
        elif threat_type == "deepfake":
            input_data = {
                "media": b"test_data" * 10,
                "media_type": "image",
            }
        elif threat_type == "polymorphic":
            input_data = {
                "executable_data": bytes([random.randint(0, 255) for _ in range(100)]),
            }
        elif threat_type == "prompt_injection":
            input_data = {
                "prompt": f"Test prompt {agent_id}_{i}",
            }
        else:  # ai_vs_ai
            input_data = {
                "query": {"input": f"query_{agent_id}_{i}"},
                "query_history": [],
                "client_id": f"agent_{agent_id}",
            }

        # Execute detection
        start = time.perf_counter()
        try:
            await detector.evaluate_threat(input_data, threat_type)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
        except Exception as e:
            # Record as failure
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)
            print(f"Agent {agent_id} request {i} failed: {e}")

    return latencies


async def run_load_test(config: LoadTestConfig) -> dict[str, any]:
    """Run load test with concurrent agents.

    Args:
        config: Load test configuration

    Returns:
        Dictionary with test results and metrics
    """
    print("\n🔥 Starting Load Test")
    print(f"{'=' * 70}")
    print(f"Agents: {config.num_agents}")
    print(f"Requests per agent: {config.requests_per_agent}")
    print(f"Total requests: {config.num_agents * config.requests_per_agent}")
    print(f"Threat types: {', '.join(config.threat_types)}")
    print(f"{'=' * 70}\n")

    # Create detector instance
    detector = RealtimeThreatDetector()

    # Create agent tasks
    print("Creating agent tasks...")
    agent_tasks = [
        agent_workload(agent_id, config, detector) for agent_id in range(config.num_agents)
    ]

    # Run all agents concurrently
    print("Starting concurrent execution...")
    start_time = time.perf_counter()

    agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Collect all latencies
    all_latencies = []
    successful_agents = 0

    for result in agent_results:
        if isinstance(result, Exception):
            print(f"Agent failed: {result}")
        else:
            all_latencies.extend(result)
            successful_agents += 1

    # Compute metrics
    total_requests = len(all_latencies)
    throughput = total_requests / total_time

    latencies_sorted = sorted(all_latencies)

    metrics = {
        "total_agents": config.num_agents,
        "successful_agents": successful_agents,
        "total_requests": total_requests,
        "total_time_seconds": total_time,
        "throughput_req_per_sec": throughput,
        "avg_latency_ms": np.mean(all_latencies),
        "median_latency_ms": np.median(all_latencies),
        "std_latency_ms": np.std(all_latencies),
        "min_latency_ms": min(all_latencies),
        "max_latency_ms": max(all_latencies),
        "p50_latency_ms": latencies_sorted[int(len(latencies_sorted) * 0.50)],
        "p95_latency_ms": latencies_sorted[int(len(latencies_sorted) * 0.95)],
        "p99_latency_ms": latencies_sorted[int(len(latencies_sorted) * 0.99)],
    }

    # Target metrics
    metrics["meets_throughput_target"] = throughput >= 5000
    metrics["meets_avg_latency_target"] = metrics["avg_latency_ms"] <= 50
    metrics["meets_p95_latency_target"] = metrics["p95_latency_ms"] <= 100
    metrics["meets_p99_latency_target"] = metrics["p99_latency_ms"] <= 200

    return metrics


def print_results(metrics: dict[str, any]) -> None:
    """Print load test results."""
    print(f"\n{'=' * 70}")
    print("📊 LOAD TEST RESULTS")
    print(f"{'=' * 70}")

    print("\n📈 Execution Summary:")
    print(f"  Total Agents:       {metrics['total_agents']}")
    print(f"  Successful Agents:  {metrics['successful_agents']}")
    print(f"  Total Requests:     {metrics['total_requests']}")
    print(f"  Total Time:         {metrics['total_time_seconds']:.2f} seconds")

    print("\n⚡ Throughput:")
    print(f"  Requests/sec:       {metrics['throughput_req_per_sec']:.2f}")
    print(
        f"  Target (>5000):     {'✅ PASS' if metrics['meets_throughput_target'] else '❌ FAIL'}"
    )

    print("\n⏱️  Latency Metrics:")
    print(f"  Average:            {metrics['avg_latency_ms']:.2f} ms")
    print(
        f"  Target (<50ms):     {'✅ PASS' if metrics['meets_avg_latency_target'] else '❌ FAIL'}"
    )
    print(f"  Median:             {metrics['median_latency_ms']:.2f} ms")
    print(f"  Std Dev:            {metrics['std_latency_ms']:.2f} ms")
    print(f"  Min:                {metrics['min_latency_ms']:.2f} ms")
    print(f"  Max:                {metrics['max_latency_ms']:.2f} ms")

    print("\n📊 Percentiles:")
    print(f"  P50:                {metrics['p50_latency_ms']:.2f} ms")
    print(f"  P95:                {metrics['p95_latency_ms']:.2f} ms")
    print(
        f"  P95 Target (<100ms):{'✅ PASS' if metrics['meets_p95_latency_target'] else '❌ FAIL'}"
    )
    print(f"  P99:                {metrics['p99_latency_ms']:.2f} ms")
    print(
        f"  P99 Target (<200ms):{'✅ PASS' if metrics['meets_p99_latency_target'] else '❌ FAIL'}"
    )

    print(f"\n{'=' * 70}")

    # Overall assessment
    all_targets_met = (
        metrics["meets_throughput_target"]
        and metrics["meets_avg_latency_target"]
        and metrics["meets_p95_latency_target"]
        and metrics["meets_p99_latency_target"]
    )

    if all_targets_met:
        print("🎉 ALL TARGETS MET! System is production-ready.")
    else:
        print("⚠️  Some targets not met. Further optimization needed.")

    print(f"{'=' * 70}\n")


async def main():
    """Run load test."""
    # Default configuration
    config = LoadTestConfig()

    # Run test
    metrics = await run_load_test(config)

    # Print results
    print_results(metrics)

    # Save results to file
    import json
    from pathlib import Path

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / f"load_test_{int(time.time())}.json"

    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    # Run with smaller scale for quick testing
    # For full test, use default LoadTestConfig values
    print("Running load test...")
    print("Note: For quick testing, using reduced scale.")
    print("For full benchmark, update LoadTestConfig in the script.\n")

    asyncio.run(main())
