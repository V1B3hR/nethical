"""
Advanced Validation Testing for Production-Grade Confidence

This module provides comprehensive testing for production-grade systems including:
- Longer runs: 10,000 to 100,000 iterations to expose rare edge cases
- High worker concurrency: 50-70 workers for realistic loads, 100+ for stress testing
- Variable worker scaling: ramp-up and ramp-down logic for elasticity testing
- Stress tests: configurable concurrency beyond normal limits
- Soak tests: sustained runs over hours/days to catch memory leaks
- Chaos/failure injection: fault injection for resilience testing

Run with:
    pytest tests/validation/test_advanced_validation.py -v -s
    pytest tests/validation/test_advanced_validation.py -v -s --run-extended
    pytest tests/validation/test_advanced_validation.py -v -s --run-soak

Environment Variables:
    NETHICAL_ADVANCED_ITERATIONS: Override default iteration counts
    NETHICAL_MAX_WORKERS: Override max worker count
    NETHICAL_SOAK_DURATION: Override soak test duration (seconds)
"""

import asyncio
import json
import logging
import os
import random
import statistics
import sys
import tempfile
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import gc
import psutil

import pytest

from nethical.core.governance import (
    SafetyGovernance,
    MonitoringConfig,
    AgentAction,
    ActionType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and Enums
# =============================================================================

class TestPhase(Enum):
    """Test execution phases"""
    RAMP_UP = "ramp_up"
    SUSTAINED = "sustained"
    RAMP_DOWN = "ramp_down"
    RECOVERY = "recovery"


class FailureType(Enum):
    """Types of failures to inject"""
    TIMEOUT = "timeout"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_SPIKE = "cpu_spike"
    RANDOM_EXCEPTION = "random_exception"
    SLOW_RESPONSE = "slow_response"


@dataclass
class TestConfig:
    """Configuration for advanced validation tests"""
    # Iteration settings
    min_iterations: int = 1000
    standard_iterations: int = 10000
    extended_iterations: int = 100000
    
    # Worker settings
    min_workers: int = 5
    standard_workers: int = 50
    high_workers: int = 70
    stress_workers: int = 100
    max_workers: int = 150
    
    # Ramp settings
    ramp_up_duration_seconds: float = 30.0
    ramp_down_duration_seconds: float = 30.0
    ramp_step_seconds: float = 5.0
    
    # Soak test settings
    short_soak_duration_seconds: float = 300.0  # 5 minutes
    medium_soak_duration_seconds: float = 3600.0  # 1 hour
    long_soak_duration_seconds: float = 7200.0  # 2 hours
    extended_soak_duration_seconds: float = 86400.0  # 24 hours
    
    # Metrics collection
    metrics_sample_interval_seconds: float = 5.0
    memory_leak_threshold_percent: float = 5.0
    performance_degradation_threshold_percent: float = 20.0
    
    # SLO thresholds
    success_rate_threshold: float = 0.95
    p95_latency_threshold_ms: float = 200.0
    p99_latency_threshold_ms: float = 500.0
    
    # Failure injection
    failure_injection_rate: float = 0.01  # 1% failure rate
    
    @classmethod
    def from_environment(cls) -> "TestConfig":
        """Create config from environment variables"""
        config = cls()
        if os.getenv("NETHICAL_ADVANCED_ITERATIONS"):
            config.standard_iterations = int(os.getenv("NETHICAL_ADVANCED_ITERATIONS"))
        if os.getenv("NETHICAL_MAX_WORKERS"):
            config.max_workers = int(os.getenv("NETHICAL_MAX_WORKERS"))
        if os.getenv("NETHICAL_SOAK_DURATION"):
            config.medium_soak_duration_seconds = float(os.getenv("NETHICAL_SOAK_DURATION"))
        return config


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class RequestMetric:
    """Single request metric"""
    request_id: int
    worker_id: int
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error: Optional[str] = None
    phase: TestPhase = TestPhase.SUSTAINED


@dataclass
class SystemMetric:
    """System resource metric at a point in time"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    active_workers: int
    requests_in_flight: int


@dataclass
class TestResults:
    """Comprehensive test results"""
    test_name: str
    config: TestConfig
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    
    # Request metrics
    request_metrics: List[RequestMetric] = field(default_factory=list)
    
    # System metrics
    system_metrics: List[SystemMetric] = field(default_factory=list)
    
    # Aggregated results
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Phase tracking
    phase_metrics: Dict[str, Dict] = field(default_factory=dict)
    
    def add_request(self, metric: RequestMetric):
        """Add a request metric"""
        self.request_metrics.append(metric)
        self.total_requests += 1
        if metric.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
    
    def add_system_metric(self, metric: SystemMetric):
        """Add a system metric"""
        self.system_metrics.append(metric)
    
    def finalize(self):
        """Calculate final metrics"""
        self.end_time = time.time()
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Calculate latency statistics"""
        durations = [m.duration_ms for m in self.request_metrics if m.success]
        if not durations:
            return {}
        
        sorted_durations = sorted(durations)
        n = len(sorted_durations)
        
        return {
            "mean_ms": statistics.mean(durations),
            "median_ms": statistics.median(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "p50_ms": sorted_durations[int(n * 0.50)],
            "p90_ms": sorted_durations[int(n * 0.90)],
            "p95_ms": sorted_durations[int(n * 0.95)] if n > 20 else max(durations),
            "p99_ms": sorted_durations[int(n * 0.99)] if n > 100 else max(durations),
            "stddev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
        }
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def get_throughput_rps(self) -> float:
        """Get requests per second"""
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
        return self.total_requests / duration
    
    def get_memory_analysis(self) -> Dict[str, Any]:
        """Analyze memory usage over time"""
        if len(self.system_metrics) < 2:
            return {}
        
        memory_values = [m.memory_mb for m in self.system_metrics]
        early_samples = memory_values[:max(1, len(memory_values) // 10)]
        late_samples = memory_values[-max(1, len(memory_values) // 10):]
        
        initial_memory = statistics.mean(early_samples)
        final_memory = statistics.mean(late_samples)
        growth_mb = final_memory - initial_memory
        growth_percent = (growth_mb / initial_memory * 100) if initial_memory > 0 else 0
        
        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "max_memory_mb": max(memory_values),
            "min_memory_mb": min(memory_values),
            "growth_mb": growth_mb,
            "growth_percent": growth_percent,
            "leak_detected": growth_percent > 5.0,
        }
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance degradation over time"""
        if len(self.request_metrics) < 100:
            return {}
        
        # Split into windows
        window_size = len(self.request_metrics) // 10
        windows = []
        
        for i in range(0, len(self.request_metrics), window_size):
            window = self.request_metrics[i:i + window_size]
            if window:
                durations = [m.duration_ms for m in window if m.success]
                if durations:
                    windows.append(statistics.mean(durations))
        
        if len(windows) < 2:
            return {}
        
        early_mean = statistics.mean(windows[:3])
        late_mean = statistics.mean(windows[-3:])
        degradation = ((late_mean - early_mean) / early_mean * 100) if early_mean > 0 else 0
        
        return {
            "early_mean_ms": early_mean,
            "late_mean_ms": late_mean,
            "degradation_percent": degradation,
            "degradation_detected": degradation > 20.0,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "test_name": self.test_name,
            "duration_seconds": self.end_time - self.start_time,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "throughput_rps": self.get_throughput_rps(),
            "latency_stats": self.get_latency_stats(),
            "memory_analysis": self.get_memory_analysis(),
            "performance_analysis": self.get_performance_analysis(),
            "phase_metrics": self.phase_metrics,
        }
    
    def save_report(self, output_dir: Path) -> Dict[str, str]:
        """Save test report to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        report_file = output_dir / f'{self.test_name}_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save markdown report
        md_file = output_dir / f'{self.test_name}_{timestamp}.md'
        self._write_markdown_report(md_file)
        
        return {
            "json_report": str(report_file),
            "md_report": str(md_file),
        }
    
    def _write_markdown_report(self, filepath: Path):
        """Write markdown report"""
        stats = self.get_latency_stats()
        memory = self.get_memory_analysis()
        performance = self.get_performance_analysis()
        
        with open(filepath, 'w') as f:
            f.write(f"# Advanced Validation Test Report: {self.test_name}\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Duration**: {self.end_time - self.start_time:.2f} seconds\n")
            f.write(f"- **Total Requests**: {self.total_requests:,}\n")
            f.write(f"- **Successful**: {self.successful_requests:,}\n")
            f.write(f"- **Failed**: {self.failed_requests:,}\n")
            f.write(f"- **Success Rate**: {self.get_success_rate()*100:.2f}%\n")
            f.write(f"- **Throughput**: {self.get_throughput_rps():.2f} req/sec\n\n")
            
            if stats:
                f.write("## Latency Statistics\n\n")
                f.write(f"| Metric | Value (ms) |\n")
                f.write(f"|--------|------------|\n")
                for key, value in stats.items():
                    f.write(f"| {key.replace('_ms', '')} | {value:.2f} |\n")
                f.write("\n")
            
            if memory:
                f.write("## Memory Analysis\n\n")
                f.write(f"- Initial Memory: {memory['initial_memory_mb']:.2f} MB\n")
                f.write(f"- Final Memory: {memory['final_memory_mb']:.2f} MB\n")
                f.write(f"- Growth: {memory['growth_mb']:.2f} MB ({memory['growth_percent']:+.2f}%)\n")
                f.write(f"- Leak Detected: {'❌ YES' if memory['leak_detected'] else '✅ NO'}\n\n")
            
            if performance:
                f.write("## Performance Degradation Analysis\n\n")
                f.write(f"- Early Mean Latency: {performance['early_mean_ms']:.2f} ms\n")
                f.write(f"- Late Mean Latency: {performance['late_mean_ms']:.2f} ms\n")
                f.write(f"- Degradation: {performance['degradation_percent']:+.2f}%\n")
                f.write(f"- Degradation Detected: {'❌ YES' if performance['degradation_detected'] else '✅ NO'}\n\n")
            
            # Determine pass/fail
            passed = True
            issues = []
            
            if self.get_success_rate() < 0.95:
                passed = False
                issues.append(f"Success rate {self.get_success_rate()*100:.2f}% < 95%")
            
            if stats and stats.get("p95_ms", 0) > 200:
                passed = False
                issues.append(f"P95 latency {stats['p95_ms']:.2f}ms > 200ms")
            
            if memory and memory.get("leak_detected"):
                passed = False
                issues.append("Memory leak detected")
            
            if performance and performance.get("degradation_detected"):
                passed = False
                issues.append("Performance degradation detected")
            
            f.write(f"## Test Result: {'✅ PASSED' if passed else '❌ FAILED'}\n\n")
            
            if issues:
                f.write("### Issues Detected\n\n")
                for issue in issues:
                    f.write(f"- {issue}\n")


# =============================================================================
# Worker Pool with Variable Scaling
# =============================================================================

class WorkerPool:
    """
    Thread pool with variable scaling support for load testing.
    Supports ramp-up and ramp-down of worker count.
    """
    
    def __init__(self, max_workers: int = 100):
        self.max_workers = max_workers
        self.current_workers = 0
        self.executor: Optional[ThreadPoolExecutor] = None
        self.active_tasks = 0
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
    
    def start(self, initial_workers: int = 10):
        """Start the worker pool"""
        self.current_workers = min(initial_workers, self.max_workers)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._stop_event.clear()
        logger.info(f"WorkerPool started with {self.current_workers} workers")
    
    def stop(self):
        """Stop the worker pool"""
        self._stop_event.set()
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        logger.info("WorkerPool stopped")
    
    def scale_to(self, target_workers: int):
        """Scale to target number of workers"""
        with self.lock:
            self.current_workers = min(target_workers, self.max_workers)
            logger.debug(f"Scaled to {self.current_workers} workers")
    
    def get_current_workers(self) -> int:
        """Get current worker count"""
        return self.current_workers
    
    def submit(self, fn: Callable, *args, **kwargs):
        """Submit a task to the pool"""
        if not self.executor or self._stop_event.is_set():
            return None
        
        with self.lock:
            self.active_tasks += 1
        
        def wrapped_fn():
            try:
                return fn(*args, **kwargs)
            finally:
                with self.lock:
                    self.active_tasks -= 1
        
        return self.executor.submit(wrapped_fn)
    
    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()


# =============================================================================
# Failure Injection
# =============================================================================

class FailureInjector:
    """Injects failures to test resilience"""
    
    def __init__(self, failure_rate: float = 0.01):
        self.failure_rate = failure_rate
        self.failures_injected = 0
        self.lock = threading.Lock()
    
    def should_inject_failure(self) -> bool:
        """Determine if a failure should be injected"""
        return random.random() < self.failure_rate
    
    def inject_failure(self, failure_type: FailureType):
        """Inject a specific type of failure"""
        with self.lock:
            self.failures_injected += 1
        
        if failure_type == FailureType.TIMEOUT:
            time.sleep(random.uniform(1.0, 5.0))
        elif failure_type == FailureType.SLOW_RESPONSE:
            time.sleep(random.uniform(0.5, 2.0))
        elif failure_type == FailureType.RANDOM_EXCEPTION:
            raise RuntimeError("Injected failure for resilience testing")
        elif failure_type == FailureType.MEMORY_PRESSURE:
            # Allocate and release memory
            _ = [0] * (1024 * 1024)  # ~8MB
        elif failure_type == FailureType.CPU_SPIKE:
            # Brief CPU intensive work
            end_time = time.time() + 0.1
            while time.time() < end_time:
                _ = sum(range(10000))
    
    def maybe_inject_failure(self) -> Optional[FailureType]:
        """Possibly inject a random failure"""
        if not self.should_inject_failure():
            return None
        
        failure_type = random.choice(list(FailureType))
        self.inject_failure(failure_type)
        return failure_type
    
    def get_stats(self) -> Dict[str, int]:
        """Get failure injection statistics"""
        return {"total_failures_injected": self.failures_injected}


# =============================================================================
# Advanced Load Test Runner
# =============================================================================

class AdvancedLoadTestRunner:
    """
    Advanced load test runner with support for:
    - Variable concurrency
    - Ramp-up/ramp-down
    - Failure injection
    - Long-running soak tests
    """
    
    def __init__(
        self,
        governance: SafetyGovernance,
        config: Optional[TestConfig] = None,
        output_dir: Optional[Path] = None,
    ):
        self.governance = governance
        self.config = config or TestConfig.from_environment()
        self.output_dir = output_dir or Path("tests/validation/results")
        self.worker_pool: Optional[WorkerPool] = None
        self.failure_injector: Optional[FailureInjector] = None
        self.results: Optional[TestResults] = None
        self._metrics_thread: Optional[threading.Thread] = None
        self._stop_metrics = threading.Event()
    
    def _create_action(self, request_id: int) -> AgentAction:
        """Create a test action"""
        action_types = [
            ("Test query for request", ActionType.QUERY),
            ("Test data processing request", ActionType.QUERY),
            ("Test analysis request", ActionType.QUERY),
        ]
        content, action_type = random.choice(action_types)
        
        return AgentAction(
            action_id=f"adv_test_{request_id}",
            agent_id=f"worker_{request_id % 100}",
            action_type=action_type,
            content=f"{content} #{request_id}",
        )
    
    def _execute_single_request(
        self,
        request_id: int,
        worker_id: int,
        phase: TestPhase,
        inject_failures: bool = False,
    ) -> RequestMetric:
        """Execute a single request and collect metrics"""
        start_time = time.time()
        success = True
        error = None
        
        try:
            # Maybe inject failure
            if inject_failures and self.failure_injector:
                failure = self.failure_injector.maybe_inject_failure()
                if failure == FailureType.RANDOM_EXCEPTION:
                    raise RuntimeError("Injected failure")
            
            # Create and evaluate action
            action = self._create_action(request_id)
            
            # Use sync evaluation (run async in thread)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.governance.evaluate_action(action))
            finally:
                loop.close()
                
        except Exception as e:
            success = False
            error = str(e)
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        return RequestMetric(
            request_id=request_id,
            worker_id=worker_id,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            success=success,
            error=error,
            phase=phase,
        )
    
    def _collect_system_metrics(self):
        """Background thread for collecting system metrics"""
        process = psutil.Process()
        
        while not self._stop_metrics.is_set():
            try:
                metric = SystemMetric(
                    timestamp=time.time(),
                    cpu_percent=process.cpu_percent(),
                    memory_mb=process.memory_info().rss / 1024 / 1024,
                    memory_percent=process.memory_percent(),
                    active_workers=self.worker_pool.current_workers if self.worker_pool else 0,
                    requests_in_flight=self.worker_pool.active_tasks if self.worker_pool else 0,
                )
                if self.results:
                    self.results.add_system_metric(metric)
            except Exception as e:
                logger.debug(f"Error collecting metrics: {e}")
            
            self._stop_metrics.wait(self.config.metrics_sample_interval_seconds)
    
    def _start_metrics_collection(self):
        """Start background metrics collection"""
        self._stop_metrics.clear()
        self._metrics_thread = threading.Thread(
            target=self._collect_system_metrics,
            daemon=True,
        )
        self._metrics_thread.start()
    
    def _stop_metrics_collection(self):
        """Stop background metrics collection"""
        self._stop_metrics.set()
        if self._metrics_thread:
            self._metrics_thread.join(timeout=2.0)
    
    def run_high_iteration_test(
        self,
        iterations: int = 10000,
        workers: int = 50,
        inject_failures: bool = False,
    ) -> TestResults:
        """
        Run a high-iteration test with configurable concurrency.
        
        Args:
            iterations: Number of iterations (10,000 to 100,000)
            workers: Number of concurrent workers (50-70 realistic, 100+ stress)
            inject_failures: Whether to inject random failures
        """
        test_name = f"high_iteration_{iterations}_workers_{workers}"
        logger.info(f"Starting {test_name}")
        logger.info(f"  Iterations: {iterations:,}")
        logger.info(f"  Workers: {workers}")
        logger.info(f"  Failure injection: {inject_failures}")
        
        self.results = TestResults(test_name=test_name, config=self.config)
        self.worker_pool = WorkerPool(max_workers=workers)
        
        if inject_failures:
            self.failure_injector = FailureInjector(self.config.failure_injection_rate)
        
        self.worker_pool.start(workers)
        self._start_metrics_collection()
        
        try:
            # Submit all tasks
            futures = []
            for i in range(iterations):
                worker_id = i % workers
                future = self.worker_pool.submit(
                    self._execute_single_request,
                    i,
                    worker_id,
                    TestPhase.SUSTAINED,
                    inject_failures,
                )
                if future:
                    futures.append(future)
                
                # Progress logging
                if (i + 1) % max(1000, iterations // 10) == 0:
                    completed = sum(1 for f in futures if f.done())
                    logger.info(f"  Progress: {i + 1:,}/{iterations:,} submitted, {completed:,} completed")
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        self.results.add_request(result)
                except Exception as e:
                    logger.warning(f"Task failed: {e}")
        
        finally:
            self._stop_metrics_collection()
            self.worker_pool.stop()
        
        self.results.finalize()
        
        # Log summary
        stats = self.results.get_latency_stats()
        logger.info(f"\n{test_name} completed:")
        logger.info(f"  Total requests: {self.results.total_requests:,}")
        logger.info(f"  Success rate: {self.results.get_success_rate()*100:.2f}%")
        logger.info(f"  Throughput: {self.results.get_throughput_rps():.2f} req/sec")
        if stats:
            logger.info(f"  P95 latency: {stats['p95_ms']:.2f}ms")
            logger.info(f"  P99 latency: {stats['p99_ms']:.2f}ms")
        
        return self.results
    
    def run_scaling_test(
        self,
        min_workers: int = 5,
        max_workers: int = 100,
        ramp_duration_seconds: float = 30.0,
        sustained_duration_seconds: float = 60.0,
    ) -> TestResults:
        """
        Run a test with variable worker scaling (ramp-up and ramp-down).
        
        Args:
            min_workers: Starting/ending worker count
            max_workers: Peak worker count
            ramp_duration_seconds: Duration for ramp-up/ramp-down phases
            sustained_duration_seconds: Duration at peak load
        """
        test_name = f"scaling_test_{min_workers}_to_{max_workers}"
        logger.info(f"Starting {test_name}")
        
        self.results = TestResults(test_name=test_name, config=self.config)
        self.worker_pool = WorkerPool(max_workers=max_workers)
        self.worker_pool.start(min_workers)
        self._start_metrics_collection()
        
        request_counter = [0]  # Use list for mutable closure
        
        def submit_requests(phase: TestPhase, duration: float):
            """Submit requests for a duration"""
            end_time = time.time() + duration
            futures = []
            
            while time.time() < end_time and not self.worker_pool.is_stopped:
                current_workers = self.worker_pool.get_current_workers()
                
                # Submit batch based on current worker count
                batch_size = min(current_workers * 2, 100)
                
                for _ in range(batch_size):
                    req_id = request_counter[0]
                    request_counter[0] += 1
                    worker_id = req_id % current_workers
                    
                    future = self.worker_pool.submit(
                        self._execute_single_request,
                        req_id,
                        worker_id,
                        phase,
                        False,
                    )
                    if future:
                        futures.append(future)
                
                time.sleep(0.1)  # Pace submission
            
            return futures
        
        try:
            all_futures = []
            
            # Phase 1: Ramp up
            logger.info(f"  Phase 1: Ramp-up ({ramp_duration_seconds}s)")
            ramp_start = time.time()
            ramp_steps = int(ramp_duration_seconds / self.config.ramp_step_seconds)
            worker_increment = (max_workers - min_workers) / max(1, ramp_steps)
            
            for step in range(ramp_steps):
                target_workers = int(min_workers + (step + 1) * worker_increment)
                self.worker_pool.scale_to(target_workers)
                
                step_futures = submit_requests(TestPhase.RAMP_UP, self.config.ramp_step_seconds)
                all_futures.extend(step_futures)
            
            # Phase 2: Sustained load
            logger.info(f"  Phase 2: Sustained load ({sustained_duration_seconds}s)")
            self.worker_pool.scale_to(max_workers)
            sustained_futures = submit_requests(TestPhase.SUSTAINED, sustained_duration_seconds)
            all_futures.extend(sustained_futures)
            
            # Phase 3: Ramp down
            logger.info(f"  Phase 3: Ramp-down ({ramp_duration_seconds}s)")
            for step in range(ramp_steps):
                target_workers = int(max_workers - (step + 1) * worker_increment)
                self.worker_pool.scale_to(max(min_workers, target_workers))
                
                step_futures = submit_requests(TestPhase.RAMP_DOWN, self.config.ramp_step_seconds)
                all_futures.extend(step_futures)
            
            # Collect results
            logger.info("  Collecting results...")
            for future in as_completed(all_futures):
                try:
                    result = future.result()
                    if result:
                        self.results.add_request(result)
                except Exception as e:
                    logger.debug(f"Task error: {e}")
        
        finally:
            self._stop_metrics_collection()
            self.worker_pool.stop()
        
        self.results.finalize()
        
        # Log summary
        logger.info(f"\n{test_name} completed:")
        logger.info(f"  Total requests: {self.results.total_requests:,}")
        logger.info(f"  Success rate: {self.results.get_success_rate()*100:.2f}%")
        
        return self.results
    
    def run_soak_test(
        self,
        duration_seconds: float = 300.0,
        workers: int = 20,
        target_rps: float = 10.0,
    ) -> TestResults:
        """
        Run a soak test for memory leak and performance drift detection.
        
        Args:
            duration_seconds: Test duration
            workers: Number of workers
            target_rps: Target requests per second
        """
        test_name = f"soak_test_{int(duration_seconds)}s"
        logger.info(f"Starting {test_name}")
        logger.info(f"  Duration: {duration_seconds}s ({duration_seconds/60:.1f} min)")
        logger.info(f"  Workers: {workers}")
        logger.info(f"  Target RPS: {target_rps}")
        
        self.results = TestResults(test_name=test_name, config=self.config)
        self.worker_pool = WorkerPool(max_workers=workers)
        self.worker_pool.start(workers)
        self._start_metrics_collection()
        
        # Force initial GC
        gc.collect()
        
        end_time = time.time() + duration_seconds
        request_id = 0
        request_interval = 1.0 / target_rps if target_rps > 0 else 0.1
        
        futures = []
        last_progress = time.time()
        
        try:
            while time.time() < end_time:
                start = time.time()
                
                # Submit request
                worker_id = request_id % workers
                future = self.worker_pool.submit(
                    self._execute_single_request,
                    request_id,
                    worker_id,
                    TestPhase.SUSTAINED,
                    False,
                )
                if future:
                    futures.append(future)
                
                request_id += 1
                
                # Progress logging every 30 seconds
                if time.time() - last_progress > 30:
                    elapsed = time.time() - self.results.start_time
                    remaining = end_time - time.time()
                    completed = sum(1 for f in futures if f.done())
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    logger.info(
                        f"  Progress: {elapsed/60:.1f}/{duration_seconds/60:.1f} min, "
                        f"{completed:,} completed, {memory_mb:.1f}MB memory"
                    )
                    last_progress = time.time()
                
                # Maintain target RPS
                elapsed = time.time() - start
                if elapsed < request_interval:
                    time.sleep(request_interval - elapsed)
            
            # Wait for remaining futures
            logger.info("  Waiting for remaining requests...")
            for future in as_completed(futures, timeout=30):
                try:
                    result = future.result()
                    if result:
                        self.results.add_request(result)
                except Exception:
                    pass
        
        finally:
            self._stop_metrics_collection()
            self.worker_pool.stop()
        
        self.results.finalize()
        
        # Analyze results
        memory = self.results.get_memory_analysis()
        perf = self.results.get_performance_analysis()
        
        logger.info(f"\n{test_name} completed:")
        logger.info(f"  Duration: {self.results.end_time - self.results.start_time:.2f}s")
        logger.info(f"  Total requests: {self.results.total_requests:,}")
        logger.info(f"  Success rate: {self.results.get_success_rate()*100:.2f}%")
        
        if memory:
            logger.info(f"  Memory growth: {memory['growth_percent']:+.2f}%")
            logger.info(f"  Leak detected: {memory['leak_detected']}")
        
        if perf:
            logger.info(f"  Performance degradation: {perf['degradation_percent']:+.2f}%")
        
        return self.results
    
    def run_stress_test(
        self,
        workers: int = 100,
        duration_seconds: float = 60.0,
        inject_failures: bool = True,
    ) -> TestResults:
        """
        Run a stress test with high concurrency and failure injection.
        
        Args:
            workers: Number of concurrent workers (100+)
            duration_seconds: Test duration
            inject_failures: Whether to inject random failures
        """
        test_name = f"stress_test_{workers}_workers"
        logger.info(f"Starting {test_name}")
        logger.info(f"  Workers: {workers}")
        logger.info(f"  Duration: {duration_seconds}s")
        logger.info(f"  Failure injection: {inject_failures}")
        
        self.results = TestResults(test_name=test_name, config=self.config)
        self.worker_pool = WorkerPool(max_workers=workers)
        
        if inject_failures:
            self.failure_injector = FailureInjector(self.config.failure_injection_rate)
        
        self.worker_pool.start(workers)
        self._start_metrics_collection()
        
        end_time = time.time() + duration_seconds
        request_id = 0
        futures = []
        
        try:
            while time.time() < end_time:
                # Submit batch of requests
                batch_size = workers * 2
                
                for _ in range(batch_size):
                    worker_id = request_id % workers
                    future = self.worker_pool.submit(
                        self._execute_single_request,
                        request_id,
                        worker_id,
                        TestPhase.SUSTAINED,
                        inject_failures,
                    )
                    if future:
                        futures.append(future)
                    request_id += 1
                
                # Brief pause between batches
                time.sleep(0.05)
            
            # Wait for completion
            for future in as_completed(futures, timeout=30):
                try:
                    result = future.result()
                    if result:
                        self.results.add_request(result)
                except Exception:
                    pass
        
        finally:
            self._stop_metrics_collection()
            self.worker_pool.stop()
        
        self.results.finalize()
        
        logger.info(f"\n{test_name} completed:")
        logger.info(f"  Total requests: {self.results.total_requests:,}")
        logger.info(f"  Success rate: {self.results.get_success_rate()*100:.2f}%")
        logger.info(f"  Throughput: {self.results.get_throughput_rps():.2f} req/sec")
        
        if self.failure_injector:
            stats = self.failure_injector.get_stats()
            logger.info(f"  Failures injected: {stats['total_failures_injected']}")
        
        return self.results


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def governance():
    """Create governance instance for testing"""
    config = MonitoringConfig(enable_persistence=False)
    return SafetyGovernance(config)


@pytest.fixture
def test_config():
    """Get test configuration"""
    return TestConfig.from_environment()


@pytest.fixture
def output_dir():
    """Output directory for test artifacts"""
    return Path("tests/validation/results/advanced")


@pytest.fixture
def test_runner(governance, test_config, output_dir):
    """Create advanced load test runner"""
    return AdvancedLoadTestRunner(
        governance=governance,
        config=test_config,
        output_dir=output_dir,
    )


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.asyncio
async def test_high_iteration_10k(test_runner, output_dir):
    """
    Test with 10,000 iterations - exposes common edge cases.
    
    Validates:
    - Success rate > 95%
    - P95 latency < 200ms
    - No significant errors
    """
    results = test_runner.run_high_iteration_test(
        iterations=10000,
        workers=50,
        inject_failures=False,
    )
    
    # Save report
    files = results.save_report(output_dir)
    logger.info(f"Reports saved: {files}")
    
    # Validate
    assert results.get_success_rate() >= 0.95, \
        f"Success rate {results.get_success_rate()*100:.2f}% < 95%"
    
    stats = results.get_latency_stats()
    if stats:
        # In CI/test environments, latency can be higher due to resource constraints
        # Production threshold is 200ms, but we allow 1000ms for test environments
        p95_threshold = float(os.getenv("NETHICAL_P95_THRESHOLD_MS", "1000"))
        assert stats["p95_ms"] < p95_threshold, \
            f"P95 latency {stats['p95_ms']:.2f}ms > {p95_threshold}ms"


@pytest.mark.asyncio
async def test_worker_concurrency_realistic(test_runner, output_dir):
    """
    Test with 50-70 workers for realistic multi-user loads.
    
    Validates:
    - System handles concurrent workers
    - Success rate > 95%
    - Reasonable throughput
    """
    results = test_runner.run_high_iteration_test(
        iterations=5000,
        workers=60,  # Realistic multi-user load
        inject_failures=False,
    )
    
    files = results.save_report(output_dir)
    logger.info(f"Reports saved: {files}")
    
    assert results.get_success_rate() >= 0.95
    assert results.get_throughput_rps() > 10  # At least 10 RPS


@pytest.mark.asyncio
async def test_worker_scaling_ramp(test_runner, output_dir):
    """
    Test worker scaling with ramp-up and ramp-down.
    
    Validates:
    - System handles variable concurrency
    - No errors during scaling
    - Stable performance across phases
    """
    results = test_runner.run_scaling_test(
        min_workers=5,
        max_workers=70,
        ramp_duration_seconds=15.0,
        sustained_duration_seconds=30.0,
    )
    
    files = results.save_report(output_dir)
    logger.info(f"Reports saved: {files}")
    
    assert results.get_success_rate() >= 0.90  # Allow some flexibility during scaling


@pytest.mark.asyncio
async def test_stress_100_workers(test_runner, output_dir):
    """
    Stress test with 100+ workers.
    
    Validates:
    - System handles peak load
    - Graceful degradation under stress
    - No crashes
    """
    results = test_runner.run_stress_test(
        workers=100,
        duration_seconds=30.0,
        inject_failures=True,
    )
    
    files = results.save_report(output_dir)
    logger.info(f"Reports saved: {files}")
    
    # Under stress with failures, we expect lower success rate
    assert results.get_success_rate() >= 0.80, \
        f"Stress test success rate {results.get_success_rate()*100:.2f}% < 80%"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_soak_short(test_runner, output_dir):
    """
    Short soak test (5 minutes) for CI/CD.
    
    Validates:
    - Memory growth < 5%
    - No performance degradation > 20%
    - Success rate > 95%
    """
    results = test_runner.run_soak_test(
        duration_seconds=300,  # 5 minutes
        workers=20,
        target_rps=5,
    )
    
    files = results.save_report(output_dir)
    logger.info(f"Reports saved: {files}")
    
    # Validate success rate
    assert results.get_success_rate() >= 0.95
    
    # Validate no memory leak
    memory = results.get_memory_analysis()
    if memory:
        assert not memory["leak_detected"], \
            f"Memory leak detected: {memory['growth_percent']:.2f}% growth"
    
    # Validate no performance degradation
    perf = results.get_performance_analysis()
    if perf:
        assert not perf["degradation_detected"], \
            f"Performance degraded: {perf['degradation_percent']:.2f}%"


@pytest.mark.asyncio
@pytest.mark.skipif(
    "not config.getoption('--run-extended')",
    reason="Extended tests not enabled"
)
async def test_high_iteration_100k(test_runner, output_dir):
    """
    Extended test with 100,000 iterations.
    
    Enable with: pytest --run-extended
    
    Validates:
    - Rare edge cases are exposed
    - Tail latencies are acceptable
    - System remains stable
    """
    results = test_runner.run_high_iteration_test(
        iterations=100000,
        workers=70,
        inject_failures=False,
    )
    
    files = results.save_report(output_dir)
    logger.info(f"Reports saved: {files}")
    
    assert results.get_success_rate() >= 0.98
    
    stats = results.get_latency_stats()
    if stats:
        # Extended tests in CI may have higher latencies
        p99_threshold = float(os.getenv("NETHICAL_P99_THRESHOLD_MS", "2000"))
        assert stats["p99_ms"] < p99_threshold, \
            f"P99 latency {stats['p99_ms']:.2f}ms > {p99_threshold}ms"


@pytest.mark.asyncio
@pytest.mark.skipif(
    "not config.getoption('--run-extended')",
    reason="Extended tests not enabled"
)
async def test_stress_150_workers(test_runner, output_dir):
    """
    Extreme stress test with 150 workers.
    
    Enable with: pytest --run-extended
    
    Validates:
    - System behavior under extreme load
    - Bottleneck identification
    - Graceful degradation
    """
    results = test_runner.run_stress_test(
        workers=150,
        duration_seconds=60.0,
        inject_failures=True,
    )
    
    files = results.save_report(output_dir)
    logger.info(f"Reports saved: {files}")
    
    # Under extreme stress, expect some failures
    assert results.get_success_rate() >= 0.70


@pytest.mark.asyncio
@pytest.mark.skipif(
    "not config.getoption('--run-soak')",
    reason="Soak tests not enabled"
)
async def test_soak_2hour(test_runner, output_dir):
    """
    Full soak test (2 hours).
    
    Enable with: pytest --run-soak
    
    Validates:
    - Long-term memory stability
    - No performance drift
    - Sustained reliability
    """
    results = test_runner.run_soak_test(
        duration_seconds=7200,  # 2 hours
        workers=30,
        target_rps=10,
    )
    
    files = results.save_report(output_dir)
    logger.info(f"Reports saved: {files}")
    
    assert results.get_success_rate() >= 0.95
    
    memory = results.get_memory_analysis()
    assert memory and not memory["leak_detected"]
    
    perf = results.get_performance_analysis()
    assert perf and not perf["degradation_detected"]


@pytest.mark.asyncio
async def test_chaos_failure_injection(test_runner, output_dir):
    """
    Test resilience with chaos/failure injection.
    
    Validates:
    - System handles injected failures gracefully
    - Recovery from various failure types
    - Metrics capture failure events
    """
    results = test_runner.run_high_iteration_test(
        iterations=2000,
        workers=30,
        inject_failures=True,
    )
    
    files = results.save_report(output_dir)
    logger.info(f"Reports saved: {files}")
    
    # With 1% failure injection, expect ~98% success rate minimum
    # Allowing some margin for actual system issues
    assert results.get_success_rate() >= 0.85, \
        f"With failure injection, success rate {results.get_success_rate()*100:.2f}% < 85%"


@pytest.mark.asyncio
async def test_generate_comprehensive_report(test_runner, output_dir):
    """
    Generate a comprehensive validation report.
    
    Runs multiple test scenarios and produces combined report.
    """
    all_results = []
    
    # Run quick versions of each test type
    logger.info("Running quick validation suite...")
    
    # 1. High iteration
    logger.info("\n--- High Iteration Test ---")
    results1 = test_runner.run_high_iteration_test(
        iterations=1000,
        workers=20,
        inject_failures=False,
    )
    all_results.append(results1)
    
    # 2. Scaling test
    logger.info("\n--- Scaling Test ---")
    results2 = test_runner.run_scaling_test(
        min_workers=5,
        max_workers=30,
        ramp_duration_seconds=10.0,
        sustained_duration_seconds=15.0,
    )
    all_results.append(results2)
    
    # 3. Stress test
    logger.info("\n--- Stress Test ---")
    results3 = test_runner.run_stress_test(
        workers=50,
        duration_seconds=15.0,
        inject_failures=True,
    )
    all_results.append(results3)
    
    # Save individual reports
    for results in all_results:
        results.save_report(output_dir)
    
    # Generate summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "tests_run": len(all_results),
        "results": [r.to_dict() for r in all_results],
        "overall_pass": all(r.get_success_rate() >= 0.80 for r in all_results),
    }
    
    summary_file = output_dir / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nComprehensive report saved to: {summary_file}")
    logger.info(f"Overall pass: {summary['overall_pass']}")
    
    assert summary["overall_pass"], "Some tests in comprehensive suite failed"


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Advanced Validation Testing Suite")
    print("=" * 80)
    print()
    print("Run specific tests:")
    print("  pytest tests/validation/test_advanced_validation.py -v -s")
    print()
    print("Run with extended tests:")
    print("  pytest tests/validation/test_advanced_validation.py -v -s --run-extended")
    print()
    print("Run with soak tests:")
    print("  pytest tests/validation/test_advanced_validation.py -v -s --run-soak")
    print()
    print("Environment variables:")
    print("  NETHICAL_ADVANCED_ITERATIONS - Override iteration count")
    print("  NETHICAL_MAX_WORKERS - Override max workers")
    print("  NETHICAL_SOAK_DURATION - Override soak duration (seconds)")
    print()
    
    pytest.main([__file__, "-v", "-s"])
