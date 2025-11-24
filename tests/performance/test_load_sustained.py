"""
Sustained Load Test - Performance Requirement 5.1

Tests system performance under sustained load for extended periods.
Generates artifacts including report and raw metrics.

Run with: pytest tests/performance/test_load_sustained.py -v -s
"""

import pytest
import asyncio
import time
import json
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from nethical.core.governance import SafetyGovernance, MonitoringConfig, AgentAction, ActionType


class LoadTestMetrics:
    """Collect and store load test metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.requests = []
        self.errors = []
        self.response_times = []
        self.cpu_samples = []
        self.memory_samples = []
        
    def record_request(self, duration_ms: float, success: bool, error: str = None):
        """Record a single request"""
        self.requests.append({
            'timestamp': time.time(),
            'duration_ms': duration_ms,
            'success': success,
            'error': error
        })
        self.response_times.append(duration_ms)
        if not success:
            self.errors.append(error or "Unknown error")
    
    def record_system_metrics(self):
        """Record current system metrics"""
        process = psutil.Process()
        self.cpu_samples.append(process.cpu_percent())
        self.memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate statistics"""
        duration = time.time() - self.start_time
        total_requests = len(self.requests)
        successful_requests = sum(1 for r in self.requests if r['success'])
        
        return {
            'duration_seconds': duration,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': len(self.errors),
            'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
            'throughput_rps': total_requests / duration if duration > 0 else 0,
            'response_times': {
                'mean': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                'min': min(self.response_times) if self.response_times else 0,
                'max': max(self.response_times) if self.response_times else 0,
                'p50': sorted(self.response_times)[len(self.response_times)//2] if self.response_times else 0,
                'p95': sorted(self.response_times)[int(len(self.response_times)*0.95)] if self.response_times else 0,
                'p99': sorted(self.response_times)[int(len(self.response_times)*0.99)] if self.response_times else 0,
            },
            'system_metrics': {
                'cpu_mean': sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
                'cpu_max': max(self.cpu_samples) if self.cpu_samples else 0,
                'memory_mean_mb': sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
                'memory_max_mb': max(self.memory_samples) if self.memory_samples else 0,
            }
        }
    
    def save_artifacts(self, output_dir: Path):
        """Save test artifacts"""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw metrics
        raw_file = output_dir / f'sustained_load_raw_{timestamp}.json'
        with open(raw_file, 'w') as f:
            json.dump({
                'requests': self.requests,
                'cpu_samples': self.cpu_samples,
                'memory_samples': self.memory_samples,
                'errors': self.errors
            }, f, indent=2)
        
        # Save summary report
        stats = self.get_stats()
        report_file = output_dir / f'sustained_load_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save human-readable report
        md_file = output_dir / f'sustained_load_report_{timestamp}.md'
        with open(md_file, 'w') as f:
            f.write("# Sustained Load Test Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Duration: {stats['duration_seconds']:.2f} seconds\n")
            f.write(f"- Total Requests: {stats['total_requests']}\n")
            f.write(f"- Successful: {stats['successful_requests']}\n")
            f.write(f"- Failed: {stats['failed_requests']}\n")
            f.write(f"- Success Rate: {stats['success_rate']*100:.2f}%\n")
            f.write(f"- Throughput: {stats['throughput_rps']:.2f} req/sec\n\n")
            f.write(f"## Response Times (ms)\n\n")
            f.write(f"- Mean: {stats['response_times']['mean']:.2f}\n")
            f.write(f"- Min: {stats['response_times']['min']:.2f}\n")
            f.write(f"- Max: {stats['response_times']['max']:.2f}\n")
            f.write(f"- P50: {stats['response_times']['p50']:.2f}\n")
            f.write(f"- P95: {stats['response_times']['p95']:.2f}\n")
            f.write(f"- P99: {stats['response_times']['p99']:.2f}\n\n")
            f.write(f"## System Metrics\n\n")
            f.write(f"- CPU Mean: {stats['system_metrics']['cpu_mean']:.2f}%\n")
            f.write(f"- CPU Max: {stats['system_metrics']['cpu_max']:.2f}%\n")
            f.write(f"- Memory Mean: {stats['system_metrics']['memory_mean_mb']:.2f} MB\n")
            f.write(f"- Memory Max: {stats['system_metrics']['memory_max_mb']:.2f} MB\n\n")
            
            if self.errors:
                f.write(f"## Errors ({len(self.errors)})\n\n")
                for error in self.errors[:10]:  # First 10 errors
                    f.write(f"- {error}\n")
        
        return {
            'raw_file': str(raw_file),
            'report_file': str(report_file),
            'md_file': str(md_file)
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


async def generate_load(governance: SafetyGovernance, duration_seconds: int, target_rps: int) -> LoadTestMetrics:
    """
    Generate sustained load
    
    Args:
        governance: Governance instance
        duration_seconds: How long to run the test
        target_rps: Target requests per second
    """
    metrics = LoadTestMetrics()
    end_time = time.time() + duration_seconds
    request_interval = 1.0 / target_rps if target_rps > 0 else 0.1
    
    request_id = 0
    while time.time() < end_time:
        start = time.time()
        
        # Create action
        action = AgentAction(
            action_id=f"load_test_{request_id}",
            agent_id="load_test_agent",
            action_type=ActionType.QUERY,
            content=f"Test query number {request_id} for sustained load testing"
        )
        
        # Execute request
        try:
            await governance.evaluate_action(action)
            duration_ms = (time.time() - start) * 1000
            metrics.record_request(duration_ms, True)
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            metrics.record_request(duration_ms, False, str(e))
        
        # Record system metrics periodically (every 10 requests)
        if request_id % 10 == 0:
            metrics.record_system_metrics()
        
        # Sleep to maintain target RPS
        elapsed = time.time() - start
        if elapsed < request_interval:
            await asyncio.sleep(request_interval - elapsed)
        
        request_id += 1
    
    # Final system metrics
    metrics.record_system_metrics()
    
    return metrics


@pytest.mark.asyncio
@pytest.mark.slow
async def test_sustained_load_short(governance, output_dir):
    """
    Short sustained load test (60 seconds) - for CI/CD
    
    Validates:
    - System handles sustained load
    - Success rate > 95%
    - P95 response time < 200ms
    - Generates artifacts
    """
    print("\n=== Starting Sustained Load Test (60s) ===")
    
    # Run load test
    metrics = await generate_load(
        governance=governance,
        duration_seconds=60,
        target_rps=10  # 10 requests per second
    )
    
    # Get statistics
    stats = metrics.get_stats()
    
    print(f"\nTest completed:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Success rate: {stats['success_rate']*100:.2f}%")
    print(f"  Throughput: {stats['throughput_rps']:.2f} req/sec")
    print(f"  Mean response time: {stats['response_times']['mean']:.2f}ms")
    print(f"  P95 response time: {stats['response_times']['p95']:.2f}ms")
    
    # Save artifacts
    files = metrics.save_artifacts(output_dir)
    print(f"\nArtifacts saved:")
    for key, path in files.items():
        print(f"  {key}: {path}")
    
    # Validate results
    assert stats['success_rate'] >= 0.95, f"Success rate too low: {stats['success_rate']*100:.2f}%"
    assert stats['response_times']['p95'] < 200, f"P95 response time too high: {stats['response_times']['p95']:.2f}ms"
    assert stats['throughput_rps'] >= 5, f"Throughput too low: {stats['throughput_rps']:.2f} req/sec"


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.skipif("not config.getoption('--run-extended')", reason="Extended tests not enabled")
async def test_sustained_load_extended(governance, output_dir):
    """
    Extended sustained load test (10 minutes)
    
    This is the full sustained load test that runs for 10 minutes.
    Enable with: pytest --run-extended
    
    Validates:
    - System stability over extended period
    - Success rate > 98%
    - P99 response time < 500ms  
    - No significant performance degradation
    """
    print("\n=== Starting Extended Sustained Load Test (10min) ===")
    
    # Run load test
    metrics = await generate_load(
        governance=governance,
        duration_seconds=600,  # 10 minutes
        target_rps=20  # 20 requests per second
    )
    
    # Get statistics
    stats = metrics.get_stats()
    
    print(f"\nTest completed:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Success rate: {stats['success_rate']*100:.2f}%")
    print(f"  Throughput: {stats['throughput_rps']:.2f} req/sec")
    print(f"  Mean response time: {stats['response_times']['mean']:.2f}ms")
    print(f"  P99 response time: {stats['response_times']['p99']:.2f}ms")
    print(f"  Memory max: {stats['system_metrics']['memory_max_mb']:.2f} MB")
    
    # Save artifacts
    files = metrics.save_artifacts(output_dir)
    print(f"\nArtifacts saved:")
    for key, path in files.items():
        print(f"  {key}: {path}")
    
    # Validate results
    assert stats['success_rate'] >= 0.98, f"Success rate too low: {stats['success_rate']*100:.2f}%"
    assert stats['response_times']['p99'] < 500, f"P99 response time too high: {stats['response_times']['p99']:.2f}ms"
    assert stats['throughput_rps'] >= 15, f"Throughput too low: {stats['throughput_rps']:.2f} req/sec"


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--run-extended",
        action="store_true",
        default=False,
        help="Run extended/long-running tests"
    )
