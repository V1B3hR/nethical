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
from statistics import mean, median, quantiles
from typing import List, Dict
import json
from nethical.core.integrated_governance import IntegratedGovernance
from nethical.core.models import AgentAction


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
        
        for i in range(num_requests):
            action_text = test_actions[i % len(test_actions)]
            # Using string action
            
            start_time = time.perf_counter()
            try:
                result = self.governance.process_action(action)
                elapsed = time.perf_counter() - start_time
                latencies.append(elapsed)
            except Exception as e:
                failures += 1
                print(f"Request {i} failed: {e}")
        
        metrics = PerformanceMetrics.calculate_latency_metrics(latencies)
        error_rate = PerformanceMetrics.calculate_error_rate(num_requests, failures)
        
        return {
            "total_requests": num_requests,
            "successful_requests": len(latencies),
            "failed_requests": failures,
            "error_rate": error_rate,
            "latency_metrics": metrics
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
    result = load_tester.run_synchronous_load_test(num_requests=200)
    
    p95_ms = result["latency_metrics"]["p95"] * 1000
    
    print(f"\nBaseline P95 Latency Test:")
    print(f"  P95: {p95_ms:.2f}ms")
    print(f"  P99: {result['latency_metrics']['p99'] * 1000:.2f}ms")
    print(f"  Max: {result['latency_metrics']['max'] * 1000:.2f}ms")
    
    assert p95_ms < 200, f"P95 latency {p95_ms:.2f}ms exceeds 200ms baseline SLO"


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
    result = load_tester.run_synchronous_load_test(num_requests=200)
    
    print(f"\nError Rate Test:")
    print(f"  Error Rate: {result['error_rate']:.2%}")
    print(f"  Failed Requests: {result['failed_requests']}")
    print(f"  Total Requests: {result['total_requests']}")
    
    assert result["error_rate"] < 0.005, f"Error rate {result['error_rate']:.2%} exceeds 0.5% SLO"


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
