"""
Performance Monitoring Probes

Runtime probes that monitor system performance metrics:
- Latency monitoring
- Throughput tracking
- Resource utilization

These probes ensure the system meets SLO/SLA requirements.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import psutil
import time

from .base_probe import BaseProbe, ProbeResult, ProbeStatus


class LatencyProbe(BaseProbe):
    """
    Latency Monitoring Probe
    
    Tracks request latency across different system components
    and validates against SLO targets.
    
    Monitors:
    - P50, P95, P99 latencies
    - Latency distribution
    - SLO compliance
    """
    
    def __init__(
        self,
        p95_target_ms: float = 100.0,
        p99_target_ms: float = 500.0,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize latency probe.
        
        Args:
            p95_target_ms: P95 latency SLO target
            p99_target_ms: P99 latency SLO target
            check_interval_seconds: Check interval
        """
        super().__init__(
            name="Latency-Monitor",
            check_interval_seconds=check_interval_seconds,
        )
        self.p95_target_ms = p95_target_ms
        self.p99_target_ms = p99_target_ms
        self._latency_samples: List[float] = []
        self._max_samples = 10000
    
    def record_latency(self, latency_ms: float, operation: str = ""):
        """Record a latency measurement"""
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > self._max_samples:
            self._latency_samples.pop(0)
    
    def check(self) -> ProbeResult:
        """Check latency metrics"""
        timestamp = datetime.utcnow()
        
        if not self._latency_samples:
            return ProbeResult(
                probe_name=self.name,
                status=ProbeStatus.HEALTHY,
                timestamp=timestamp,
                message="No latency samples recorded",
                metrics={},
            )
        
        # Calculate percentiles
        sorted_samples = sorted(self._latency_samples)
        n = len(sorted_samples)
        
        p50 = sorted_samples[int(n * 0.50)] if n > 0 else 0
        p95 = sorted_samples[int(n * 0.95)] if n > 0 else 0
        p99 = sorted_samples[int(n * 0.99)] if n > 0 else 0
        avg = sum(sorted_samples) / n if n > 0 else 0
        max_latency = max(sorted_samples) if sorted_samples else 0
        
        violations = []
        
        # Check against SLO targets
        if p95 > self.p95_target_ms:
            violations.append(
                f"P95 latency {p95:.2f}ms exceeds target {self.p95_target_ms}ms"
            )
        
        if p99 > self.p99_target_ms:
            violations.append(
                f"P99 latency {p99:.2f}ms exceeds target {self.p99_target_ms}ms"
            )
        
        # Determine status
        if p99 > self.p99_target_ms * 2:
            status = ProbeStatus.CRITICAL
            message = "Critical latency SLO violations"
        elif violations:
            status = ProbeStatus.WARNING
            message = "Latency SLO targets exceeded"
        else:
            status = ProbeStatus.HEALTHY
            message = "Latency within SLO targets"
        
        return ProbeResult(
            probe_name=self.name,
            status=status,
            timestamp=timestamp,
            message=message,
            metrics={
                "samples": n,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "avg_ms": avg,
                "max_ms": max_latency,
                "p95_target_ms": self.p95_target_ms,
                "p99_target_ms": self.p99_target_ms,
                "slo_compliance": p95 <= self.p95_target_ms and p99 <= self.p99_target_ms,
            },
            violations=violations,
        )


class ThroughputProbe(BaseProbe):
    """
    Throughput Monitoring Probe
    
    Tracks request throughput and validates against capacity targets.
    
    Monitors:
    - Requests per second
    - Throughput trends
    - Capacity utilization
    """
    
    def __init__(
        self,
        target_rps: float = 1000.0,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize throughput probe.
        
        Args:
            target_rps: Target requests per second
            check_interval_seconds: Check interval
        """
        super().__init__(
            name="Throughput-Monitor",
            check_interval_seconds=check_interval_seconds,
        )
        self.target_rps = target_rps
        self._request_timestamps: List[datetime] = []
        self._window_seconds = 60
    
    def record_request(self):
        """Record a request"""
        now = datetime.utcnow()
        self._request_timestamps.append(now)
        
        # Remove old timestamps outside window
        cutoff = now - timedelta(seconds=self._window_seconds)
        self._request_timestamps = [
            ts for ts in self._request_timestamps if ts > cutoff
        ]
    
    def check(self) -> ProbeResult:
        """Check throughput metrics"""
        timestamp = datetime.utcnow()
        
        # Count requests in window
        cutoff = timestamp - timedelta(seconds=self._window_seconds)
        recent_requests = [
            ts for ts in self._request_timestamps if ts > cutoff
        ]
        
        current_rps = len(recent_requests) / self._window_seconds
        capacity_utilization = (current_rps / self.target_rps) * 100 if self.target_rps > 0 else 0
        
        violations = []
        
        # Check for capacity issues
        if capacity_utilization > 90:
            violations.append(
                f"Capacity utilization at {capacity_utilization:.1f}% (target: {self.target_rps} RPS)"
            )
        
        # Determine status
        if capacity_utilization > 95:
            status = ProbeStatus.CRITICAL
            message = "Critical capacity utilization"
        elif capacity_utilization > 80:
            status = ProbeStatus.WARNING
            message = "High capacity utilization"
        else:
            status = ProbeStatus.HEALTHY
            message = f"Throughput: {current_rps:.1f} RPS"
        
        return ProbeResult(
            probe_name=self.name,
            status=status,
            timestamp=timestamp,
            message=message,
            metrics={
                "current_rps": current_rps,
                "target_rps": self.target_rps,
                "capacity_utilization_percent": capacity_utilization,
                "requests_in_window": len(recent_requests),
                "window_seconds": self._window_seconds,
            },
            violations=violations,
        )


class ResourceUtilizationProbe(BaseProbe):
    """
    Resource Utilization Monitoring Probe
    
    Tracks system resource usage (CPU, memory, disk, network).
    
    Monitors:
    - CPU utilization
    - Memory usage
    - Disk I/O
    - Network throughput
    """
    
    def __init__(
        self,
        cpu_threshold_percent: float = 80.0,
        memory_threshold_percent: float = 85.0,
        check_interval_seconds: int = 60,
    ):
        """
        Initialize resource utilization probe.
        
        Args:
            cpu_threshold_percent: CPU utilization alert threshold
            memory_threshold_percent: Memory utilization alert threshold
            check_interval_seconds: Check interval
        """
        super().__init__(
            name="ResourceUtilization-Monitor",
            check_interval_seconds=check_interval_seconds,
        )
        self.cpu_threshold = cpu_threshold_percent
        self.memory_threshold = memory_threshold_percent
    
    def check(self) -> ProbeResult:
        """Check resource utilization metrics"""
        timestamp = datetime.utcnow()
        violations = []
        
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
            
            # Memory utilization
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk I/O
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Check thresholds
            if cpu_percent > self.cpu_threshold:
                violations.append(
                    f"CPU utilization {cpu_percent:.1f}% exceeds threshold {self.cpu_threshold}%"
                )
            
            if memory_percent > self.memory_threshold:
                violations.append(
                    f"Memory utilization {memory_percent:.1f}% exceeds threshold {self.memory_threshold}%"
                )
            
            if disk_percent > 90:
                violations.append(
                    f"Disk utilization {disk_percent:.1f}% is critically high"
                )
            
            # Determine status
            if cpu_percent > 95 or memory_percent > 95 or disk_percent > 95:
                status = ProbeStatus.CRITICAL
                message = "Critical resource utilization"
            elif violations:
                status = ProbeStatus.WARNING
                message = "Resource utilization above thresholds"
            else:
                status = ProbeStatus.HEALTHY
                message = "Resource utilization normal"
            
            return ProbeResult(
                probe_name=self.name,
                status=status,
                timestamp=timestamp,
                message=message,
                metrics={
                    "cpu_percent": cpu_percent,
                    "cpu_per_core": cpu_per_core,
                    "cpu_threshold": self.cpu_threshold,
                    "memory_percent": memory_percent,
                    "memory_available_gb": memory_available_gb,
                    "memory_threshold": self.memory_threshold,
                    "disk_percent": disk_percent,
                    "disk_total_gb": disk.total / (1024**3),
                    "disk_used_gb": disk.used / (1024**3),
                },
                violations=violations,
            )
            
        except Exception as e:
            return ProbeResult(
                probe_name=self.name,
                status=ProbeStatus.UNKNOWN,
                timestamp=timestamp,
                message=f"Failed to collect resource metrics: {str(e)}",
                metrics={},
            )
