"""
Burst Load Test - Performance Requirement 5.2

Tests system performance under burst load (5× baseline).
Validates system can handle sudden traffic spikes.

Run with: pytest tests/performance/test_load_burst.py -v -s
"""

import pytest
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from nethical.core.governance import SafetyGovernance, MonitoringConfig, AgentAction, ActionType


class BurstTestMetrics:
    """Collect and store burst test metrics"""
    
    def __init__(self):
        self.baseline_stats = None
        self.burst_stats = None
        self.requests = []
        
    def record_request(self, duration_ms: float, success: bool, phase: str):
        """Record a single request"""
        self.requests.append({
            'timestamp': time.time(),
            'duration_ms': duration_ms,
            'success': success,
            'phase': phase
        })
    
    def calculate_phase_stats(self, phase: str) -> Dict[str, Any]:
        """Calculate statistics for a specific phase"""
        phase_requests = [r for r in self.requests if r['phase'] == phase]
        if not phase_requests:
            return None
        
        response_times = [r['duration_ms'] for r in phase_requests]
        successful = sum(1 for r in phase_requests if r['success'])
        
        return {
            'total_requests': len(phase_requests),
            'successful_requests': successful,
            'failed_requests': len(phase_requests) - successful,
            'success_rate': successful / len(phase_requests) if phase_requests else 0,
            'response_times': {
                'mean': sum(response_times) / len(response_times) if response_times else 0,
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'p95': sorted(response_times)[int(len(response_times)*0.95)] if response_times else 0,
                'p99': sorted(response_times)[int(len(response_times)*0.99)] if response_times else 0,
            }
        }
    
    def set_baseline_stats(self, stats: Dict[str, Any]):
        """Set baseline performance statistics"""
        self.baseline_stats = stats
    
    def set_burst_stats(self, stats: Dict[str, Any]):
        """Set burst performance statistics"""
        self.burst_stats = stats
    
    def get_comparison(self) -> Dict[str, Any]:
        """Compare baseline vs burst performance"""
        if not self.baseline_stats or not self.burst_stats:
            return None
        
        return {
            'baseline': self.baseline_stats,
            'burst': self.burst_stats,
            'degradation': {
                'mean_response_time': (
                    (self.burst_stats['response_times']['mean'] - 
                     self.baseline_stats['response_times']['mean']) /
                    self.baseline_stats['response_times']['mean'] * 100
                ) if self.baseline_stats['response_times']['mean'] > 0 else 0,
                'p95_response_time': (
                    (self.burst_stats['response_times']['p95'] - 
                     self.baseline_stats['response_times']['p95']) /
                    self.baseline_stats['response_times']['p95'] * 100
                ) if self.baseline_stats['response_times']['p95'] > 0 else 0,
                'success_rate': (
                    (self.burst_stats['success_rate'] - 
                     self.baseline_stats['success_rate']) * 100
                )
            }
        }
    
    def save_artifacts(self, output_dir: Path) -> Dict[str, str]:
        """Save test artifacts"""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw data
        raw_file = output_dir / f'burst_test_raw_{timestamp}.json'
        with open(raw_file, 'w') as f:
            json.dump({'requests': self.requests}, f, indent=2)
        
        # Save comparison report
        comparison = self.get_comparison()
        report_file = output_dir / f'burst_test_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Save human-readable report
        md_file = output_dir / f'burst_test_report_{timestamp}.md'
        with open(md_file, 'w') as f:
            f.write("# Burst Load Test Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            f.write("## Test Configuration\n\n")
            f.write("- Burst multiplier: 5×\n")
            f.write("- Test validates system handles 5× baseline load\n\n")
            
            if comparison:
                f.write("## Baseline Performance\n\n")
                bl = comparison['baseline']
                f.write(f"- Total Requests: {bl['total_requests']}\n")
                f.write(f"- Success Rate: {bl['success_rate']*100:.2f}%\n")
                f.write(f"- Mean Response Time: {bl['response_times']['mean']:.2f}ms\n")
                f.write(f"- P95 Response Time: {bl['response_times']['p95']:.2f}ms\n\n")
                
                f.write("## Burst Performance (5× baseline)\n\n")
                br = comparison['burst']
                f.write(f"- Total Requests: {br['total_requests']}\n")
                f.write(f"- Success Rate: {br['success_rate']*100:.2f}%\n")
                f.write(f"- Mean Response Time: {br['response_times']['mean']:.2f}ms\n")
                f.write(f"- P95 Response Time: {br['response_times']['p95']:.2f}ms\n\n")
                
                f.write("## Performance Degradation\n\n")
                deg = comparison['degradation']
                f.write(f"- Mean Response Time: {deg['mean_response_time']:+.2f}%\n")
                f.write(f"- P95 Response Time: {deg['p95_response_time']:+.2f}%\n")
                f.write(f"- Success Rate Change: {deg['success_rate']:+.2f}%\n\n")
                
                # Pass/Fail determination
                passed = (
                    br['success_rate'] >= 0.90 and  # 90% success rate minimum
                    deg['p95_response_time'] <= 300  # P95 degradation < 300%
                )
                f.write(f"## Test Result: {'✅ PASSED' if passed else '❌ FAILED'}\n\n")
                f.write("### Pass Criteria\n\n")
                f.write(f"- Success rate ≥ 90%: {'✅' if br['success_rate'] >= 0.90 else '❌'} ({br['success_rate']*100:.2f}%)\n")
                f.write(f"- P95 degradation ≤ 300%: {'✅' if deg['p95_response_time'] <= 300 else '❌'} ({deg['p95_response_time']:.2f}%)\n")
        
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


async def run_load_phase(governance: SafetyGovernance, num_requests: int, phase: str, 
                         metrics: BurstTestMetrics) -> None:
    """
    Run a load phase with specified number of requests
    
    Args:
        governance: Governance instance
        num_requests: Number of requests to send
        phase: Phase identifier ('baseline' or 'burst')
        metrics: Metrics collector
    """
    # Send requests concurrently in small batches
    batch_size = 10
    for batch_start in range(0, num_requests, batch_size):
        batch_end = min(batch_start + batch_size, num_requests)
        tasks = []
        
        for i in range(batch_start, batch_end):
            action = AgentAction(
                action_id=f"{phase}_test_{i}",
                agent_id="burst_test_agent",
                action_type=ActionType.QUERY,
                content=f"Test query number {i} for {phase} phase"
            )
            tasks.append(evaluate_with_timing(governance, action, phase, metrics))
        
        await asyncio.gather(*tasks)
        
        # Small delay between batches to avoid overwhelming the system
        await asyncio.sleep(0.1)


async def evaluate_with_timing(governance: SafetyGovernance, action: AgentAction, 
                               phase: str, metrics: BurstTestMetrics) -> None:
    """Evaluate action and record metrics"""
    start = time.time()
    try:
        await governance.evaluate_action(action)
        duration_ms = (time.time() - start) * 1000
        metrics.record_request(duration_ms, True, phase)
    except Exception:
        duration_ms = (time.time() - start) * 1000
        metrics.record_request(duration_ms, False, phase)


@pytest.mark.asyncio
async def test_burst_load_5x(governance, output_dir):
    """
    Burst load test - 5× baseline load
    
    Tests:
    1. Establish baseline performance (100 requests)
    2. Apply burst load (500 requests - 5× baseline)
    3. Validate system handles burst with acceptable degradation
    
    Pass Criteria:
    - Success rate ≥ 90% during burst
    - P95 response time degradation ≤ 300%
    """
    print("\n=== Starting Burst Load Test (5× baseline) ===")
    
    metrics = BurstTestMetrics()
    
    # Phase 1: Baseline load
    print("\nPhase 1: Running baseline load (100 requests)...")
    baseline_requests = 100
    await run_load_phase(governance, baseline_requests, 'baseline', metrics)
    baseline_stats = metrics.calculate_phase_stats('baseline')
    metrics.set_baseline_stats(baseline_stats)
    
    print(f"Baseline completed:")
    print(f"  Success rate: {baseline_stats['success_rate']*100:.2f}%")
    print(f"  Mean response time: {baseline_stats['response_times']['mean']:.2f}ms")
    print(f"  P95 response time: {baseline_stats['response_times']['p95']:.2f}ms")
    
    # Small cooldown
    await asyncio.sleep(2)
    
    # Phase 2: Burst load (5× baseline)
    print("\nPhase 2: Running burst load (500 requests - 5× baseline)...")
    burst_requests = baseline_requests * 5
    await run_load_phase(governance, burst_requests, 'burst', metrics)
    burst_stats = metrics.calculate_phase_stats('burst')
    metrics.set_burst_stats(burst_stats)
    
    print(f"Burst completed:")
    print(f"  Success rate: {burst_stats['success_rate']*100:.2f}%")
    print(f"  Mean response time: {burst_stats['response_times']['mean']:.2f}ms")
    print(f"  P95 response time: {burst_stats['response_times']['p95']:.2f}ms")
    
    # Get comparison
    comparison = metrics.get_comparison()
    degradation = comparison['degradation']
    
    print(f"\nPerformance degradation under burst:")
    print(f"  Mean response time: {degradation['mean_response_time']:+.2f}%")
    print(f"  P95 response time: {degradation['p95_response_time']:+.2f}%")
    print(f"  Success rate change: {degradation['success_rate']:+.2f}%")
    
    # Save artifacts
    files = metrics.save_artifacts(output_dir)
    print(f"\nArtifacts saved:")
    for key, path in files.items():
        print(f"  {key}: {path}")
    
    # Validate pass criteria
    print("\n=== Validating Pass Criteria ===")
    
    success_rate_pass = burst_stats['success_rate'] >= 0.90
    print(f"Success rate ≥ 90%: {'✅ PASS' if success_rate_pass else '❌ FAIL'} ({burst_stats['success_rate']*100:.2f}%)")
    
    p95_degradation_pass = degradation['p95_response_time'] <= 300
    print(f"P95 degradation ≤ 300%: {'✅ PASS' if p95_degradation_pass else '❌ FAIL'} ({degradation['p95_response_time']:.2f}%)")
    
    overall_pass = success_rate_pass and p95_degradation_pass
    print(f"\nOverall: {'✅ PASSED' if overall_pass else '❌ FAILED'}")
    
    # Assert test passes
    assert success_rate_pass, f"Success rate too low: {burst_stats['success_rate']*100:.2f}%"
    assert p95_degradation_pass, f"P95 degradation too high: {degradation['p95_response_time']:.2f}%"
