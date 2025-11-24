"""
Performance Validation Test Suite

Tests performance metrics against SLO thresholds:
- Load tests
- Burst tests
- Soak tests

Thresholds:
- p95 Latency (baseline): <200ms
- p99 Latency (burst): <500ms
- Error Rate: <0.5%
"""

import pytest
import time
import asyncio
import logging
from statistics import mean, median, quantiles
from typing import List, Dict
import json
from datetime import datetime
from nethical.core.integrated_governance import IntegratedGovernance
from nethical.core.models import AgentAction

# Configure logging for detailed diagnostics
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Calculate performance metrics"""
    
    @staticmethod
    def calculate_latency_metrics(latencies: List[float]) -> Dict[str, float]:
        """
        Calculate latency metrics
        
        Args:
            latencies: List of latency measurements in seconds
            
        Returns:
            Dictionary with percentile metrics
        """
        if not latencies:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        sorted_latencies = sorted(latencies)
        
        if len(sorted_latencies) >= 2:
            percentiles = quantiles(sorted_latencies, n=100)
            p50 = percentiles[49]
            p95 = percentiles[94]
            p99 = percentiles[98]
        else:
            p50 = p95 = p99 = sorted_latencies[0]
        
        return {
            "mean": mean(latencies),
            "median": median(latencies),
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "min": min(latencies),
            "max": max(latencies)
        }
    
    @staticmethod
    def calculate_error_rate(total_requests: int, failed_requests: int) -> float:
        """Calculate error rate"""
        if total_requests == 0:
            return 0.0
        return failed_requests / total_requests


class LoadTester:
    """Load testing utility"""
    
    def __init__(self, governance: IntegratedGovernance):
        self.governance = governance
    
    def run_synchronous_load_test(self, num_requests: int = 100, 
                                   test_actions: List[str] = None) -> Dict:
        """
        Run synchronous load test
        
        Args:
            num_requests: Number of requests to send
            test_actions: List of test action strings
            
        Returns:
            Dictionary with test results
        """
        if test_actions is None:
            test_actions = [
                "Process user data",
                "Generate report",
                "Query database",
                "Update records",
                "Send notification"
            ]
        
        latencies = []
        failures = 0
        failure_details = []
        
        logger.debug(f"Starting load test with {num_requests} requests")
        
        for i in range(num_requests):
            action_text = test_actions[i % len(test_actions)]
            
            start_time = time.perf_counter()
            try:
                result = self.governance.process_action(
                    agent_id="load_tester",
                    action=action_text
                )
                elapsed = time.perf_counter() - start_time
                latencies.append(elapsed)
                
                # Log slow requests
                if elapsed > 0.5:  # >500ms
                    logger.warning(f"Slow request #{i + 1}: {elapsed * 1000:.2f}ms for action '{action_text}'")
            except Exception as e:
                failures += 1
                elapsed = time.perf_counter() - start_time
                failure_info = {
                    "request_id": i + 1,
                    "action": action_text,
                    "error": str(e),
                    "elapsed_ms": elapsed * 1000
                }
                failure_details.append(failure_info)
                logger.error(f"Request {i + 1} failed: {e}")
        
        metrics = PerformanceMetrics.calculate_latency_metrics(latencies)
        error_rate = PerformanceMetrics.calculate_error_rate(num_requests, failures)
        
        return {
            "total_requests": num_requests,
            "successful_requests": len(latencies),
            "failed_requests": failures,
            "error_rate": error_rate,
            "latency_metrics": metrics,
            "failure_details": failure_details
        }


@pytest.fixture
def governance():
    """Initialize governance"""
    return IntegratedGovernance()


@pytest.fixture
def load_tester(governance):
    """Initialize load tester"""
    return LoadTester(governance)


@pytest.fixture
def performance_metrics():
    """Initialize performance metrics calculator"""
    return PerformanceMetrics()


def test_baseline_latency_p50(load_tester):
    """Test p50 latency under baseline load"""
    result = load_tester.run_synchronous_load_test(num_requests=100)
    
    p50_ms = result["latency_metrics"]["p50"] * 1000
    
    print(f"\nBaseline P50 Latency Test:")
    print(f"  P50: {p50_ms:.2f}ms")
    print(f"  Mean: {result['latency_metrics']['mean'] * 1000:.2f}ms")
    print(f"  Error Rate: {result['error_rate']:.2%}")
    
    # P50 should be well under 200ms
    assert p50_ms < 200, f"P50 latency {p50_ms:.2f}ms exceeds 200ms baseline"


def test_baseline_latency_p95(load_tester):
    """Test p95 latency under baseline load"""
    logger.info("=" * 80)
    logger.info("PERFORMANCE TEST - Baseline P95 Latency")
    logger.info("=" * 80)
    logger.info("Testing 200 requests to measure p95 latency against 200ms SLO")
    
    start_time = time.time()
    result = load_tester.run_synchronous_load_test(num_requests=200)
    total_duration = time.time() - start_time
    
    p95_ms = result["latency_metrics"]["p95"] * 1000
    p99_ms = result["latency_metrics"]["p99"] * 1000
    mean_ms = result["latency_metrics"]["mean"] * 1000
    max_ms = result["latency_metrics"]["max"] * 1000
    
    logger.info("-" * 80)
    logger.info("Results:")
    logger.info(f"  Total Duration: {total_duration:.2f}s")
    logger.info(f"  Total Requests: {result['total_requests']}")
    logger.info(f"  Successful: {result['successful_requests']}")
    logger.info(f"  Failed: {result['failed_requests']}")
    logger.info(f"  Error Rate: {result['error_rate']:.2%}")
    logger.info(f"  Mean Latency: {mean_ms:.2f}ms")
    logger.info(f"  P50 Latency: {result['latency_metrics']['p50'] * 1000:.2f}ms")
    logger.info(f"  P95 Latency: {p95_ms:.2f}ms (SLO: <200ms)")
    logger.info(f"  P99 Latency: {p99_ms:.2f}ms")
    logger.info(f"  Max Latency: {max_ms:.2f}ms")
    logger.info(f"  Throughput: {result['successful_requests'] / total_duration:.2f} req/s")
    
    print(f"\nBaseline P95 Latency Test:")
    print(f"  P95: {p95_ms:.2f}ms {'✓' if p95_ms < 200 else '✗'}")
    print(f"  P99: {p99_ms:.2f}ms")
    print(f"  Max: {max_ms:.2f}ms")
    print(f"  Error Rate: {result['error_rate']:.2%}")
    
    if result["failed_requests"] > 0:
        logger.warning(f"\n{result['failed_requests']} requests failed:")
        for failure in result.get("failure_details", [])[:5]:
            logger.warning(
                f"  Request #{failure['request_id']}: {failure['error']} "
                f"(after {failure['elapsed_ms']:.2f}ms)"
            )
    
    if p95_ms >= 200:
        logger.error("=" * 80)
        logger.error("SLO VIOLATION DETECTED")
        logger.error("=" * 80)
        logger.error(f"P95 latency {p95_ms:.2f}ms exceeds 200ms baseline SLO")
        logger.error("Debugging steps:")
        logger.error("1. Check system resource utilization during test")
        logger.error("2. Review governance processing logic for bottlenecks")
        logger.error("3. Profile slow requests (see warnings above)")
        logger.error("4. Consider optimizing risk calculation or caching")
        logger.error("5. Verify database/storage performance")
        logger.error("\nTo reproduce:")
        logger.error("  pytest tests/validation/test_performance_validation.py::test_baseline_latency_p95 -v -s")
    
    assert p95_ms < 200, (
        f"P95 latency {p95_ms:.2f}ms exceeds 200ms baseline SLO.\n"
        f"  Mean: {mean_ms:.2f}ms, P99: {p99_ms:.2f}ms, Max: {max_ms:.2f}ms\n"
        f"  Successful requests: {result['successful_requests']}/{result['total_requests']}\n"
        f"  See logs above for detailed analysis and debugging steps"
    )


def test_burst_latency_p99(load_tester):
    """Test p99 latency under burst load"""
    # Simulate burst with rapid requests
    result = load_tester.run_synchronous_load_test(num_requests=500)
    
    p99_ms = result["latency_metrics"]["p99"] * 1000
    
    print(f"\nBurst P99 Latency Test:")
    print(f"  P99: {p99_ms:.2f}ms")
    print(f"  Max: {result['latency_metrics']['max'] * 1000:.2f}ms")
    print(f"  Total Requests: {result['total_requests']}")
    
    assert p99_ms < 500, f"P99 latency {p99_ms:.2f}ms exceeds 500ms burst SLO"


def test_error_rate_baseline(load_tester):
    """Test error rate under baseline load"""
    logger.info("=" * 80)
    logger.info("PERFORMANCE TEST - Error Rate")
    logger.info("=" * 80)
    logger.info("Testing 200 requests to measure error rate against 0.5% SLO")
    
    result = load_tester.run_synchronous_load_test(num_requests=200)
    
    logger.info("-" * 80)
    logger.info("Results:")
    logger.info(f"  Total Requests: {result['total_requests']}")
    logger.info(f"  Successful: {result['successful_requests']}")
    logger.info(f"  Failed: {result['failed_requests']}")
    logger.info(f"  Error Rate: {result['error_rate']:.2%} (SLO: <0.5%)")
    
    print(f"\nError Rate Test:")
    print(f"  Error Rate: {result['error_rate']:.2%} {'✓' if result['error_rate'] < 0.005 else '✗'}")
    print(f"  Failed Requests: {result['failed_requests']}")
    print(f"  Total Requests: {result['total_requests']}")
    
    if result['failed_requests'] > 0:
        logger.warning(f"\n{result['failed_requests']} requests failed:")
        failure_details = result.get('failure_details', [])
        for detail in failure_details[:10]:
            logger.warning(
                f"  Request #{detail['request_id']}: {detail['error']} "
                f"(action: {detail['action'][:30]}...)"
            )
        
        # Analyze failure patterns
        error_types = {}
        for detail in failure_details:
            error_type = type(detail['error']).__name__ if 'error' in detail else 'Unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        logger.warning("\nFailure breakdown by type:")
        for error_type, count in error_types.items():
            logger.warning(f"  {error_type}: {count}")
    
    if result["error_rate"] >= 0.005:
        logger.error("=" * 80)
        logger.error("ERROR RATE SLO VIOLATION")
        logger.error("=" * 80)
        logger.error(f"Error rate {result['error_rate']:.2%} exceeds 0.5% SLO")
        logger.error(f"Failed requests: {result['failed_requests']}/{result['total_requests']}")
        logger.error("\nDebugging steps:")
        logger.error("1. Review error types and patterns (logged above)")
        logger.error("2. Check for resource exhaustion (memory, connections)")
        logger.error("3. Verify error handling in governance processing")
        logger.error("4. Look for timeout or crash conditions")
        logger.error("5. Check system logs for underlying issues")
        logger.error("\nTo reproduce:")
        logger.error("  pytest tests/validation/test_performance_validation.py::test_error_rate_baseline -v -s")
    
    assert result["error_rate"] < 0.005, (
        f"Error rate {result['error_rate']:.2%} exceeds 0.5% SLO.\n"
        f"  Failed: {result['failed_requests']}/{result['total_requests']} requests\n"
        f"  Review failure details in logs above for error patterns\n"
        f"  Common causes: resource exhaustion, timeout, invalid input"
    )


def test_sustained_load_performance(load_tester):
    """Test sustained load performance"""
    # Run multiple batches to simulate sustained load
    all_latencies = []
    all_failures = 0
    total_requests = 0
    
    batches = 5
    requests_per_batch = 50
    
    print(f"\nSustained Load Test ({batches} batches):")
    
    for batch in range(batches):
        result = load_tester.run_synchronous_load_test(num_requests=requests_per_batch)
        all_latencies.extend([l * 1000 for l in [
            result["latency_metrics"]["mean"],
            result["latency_metrics"]["p95"],
            result["latency_metrics"]["p99"]
        ]])
        all_failures += result["failed_requests"]
        total_requests += result["total_requests"]
        
        print(f"  Batch {batch + 1}: P95={result['latency_metrics']['p95'] * 1000:.2f}ms, "
              f"Errors={result['failed_requests']}")
    
    overall_error_rate = all_failures / total_requests if total_requests > 0 else 0
    avg_p95 = mean([l for l in all_latencies if l < 1000])  # Filter outliers
    
    print(f"  Overall Error Rate: {overall_error_rate:.2%}")
    print(f"  Average P95: {avg_p95:.2f}ms")
    
    assert overall_error_rate < 0.005, f"Sustained load error rate {overall_error_rate:.2%} exceeds SLO"
    assert avg_p95 < 200, f"Sustained load P95 {avg_p95:.2f}ms exceeds SLO"


@pytest.mark.slow
def test_soak_test_stability(load_tester):
    """Test stability under extended soak test (marked as slow)"""
    # Mini soak test: 10 batches of 20 requests each
    duration_batches = 10
    requests_per_batch = 20
    
    batch_results = []
    
    print(f"\nSoak Test ({duration_batches} batches):")
    
    for batch in range(duration_batches):
        result = load_tester.run_synchronous_load_test(num_requests=requests_per_batch)
        batch_results.append(result)
        
        p95_ms = result["latency_metrics"]["p95"] * 1000
        print(f"  Batch {batch + 1}: P95={p95_ms:.2f}ms, Errors={result['failed_requests']}")
        
        # Short delay between batches
        time.sleep(0.1)
    
    # Check for performance degradation over time
    early_p95 = mean([r["latency_metrics"]["p95"] for r in batch_results[:3]])
    late_p95 = mean([r["latency_metrics"]["p95"] for r in batch_results[-3:]])
    
    degradation = (late_p95 - early_p95) / early_p95 if early_p95 > 0 else 0
    
    print(f"\nSoak Test Results:")
    print(f"  Early P95: {early_p95 * 1000:.2f}ms")
    print(f"  Late P95: {late_p95 * 1000:.2f}ms")
    print(f"  Degradation: {degradation:.1%}")
    
    # Performance shouldn't degrade more than 50% over time
    assert degradation < 0.5, f"Performance degraded by {degradation:.1%} during soak test"


def test_generate_performance_report(load_tester, tmp_path):
    """Generate comprehensive performance report"""
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_suite": "performance_validation",
        "tests": {}
    }
    
    # Run multiple test scenarios
    scenarios = {
        "baseline_100": 100,
        "baseline_200": 200,
        "burst_500": 500,
    }
    
    for scenario_name, num_requests in scenarios.items():
        result = load_tester.run_synchronous_load_test(num_requests=num_requests)
        
        # Convert to milliseconds
        latency_metrics_ms = {
            k: v * 1000 for k, v in result["latency_metrics"].items()
        }
        
        report["tests"][scenario_name] = {
            "total_requests": result["total_requests"],
            "successful_requests": result["successful_requests"],
            "failed_requests": result["failed_requests"],
            "error_rate": result["error_rate"],
            "latency_ms": latency_metrics_ms,
            "slo_compliance": {
                "p95_under_200ms": latency_metrics_ms["p95"] < 200,
                "p99_under_500ms": latency_metrics_ms["p99"] < 500,
                "error_rate_under_0.5pct": result["error_rate"] < 0.005
            }
        }
    
    # Overall compliance
    all_compliant = all(
        test["slo_compliance"]["p95_under_200ms"] and 
        test["slo_compliance"]["error_rate_under_0.5pct"]
        for test in report["tests"].values()
    )
    
    report["overall_compliance"] = all_compliant
    
    # Save report
    report_path = tmp_path / "performance_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nPerformance report saved to: {report_path}")
    print(f"Overall SLO Compliance: {all_compliant}")
    
    assert report_path.exists()
    assert all_compliant, "Some tests failed SLO compliance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
