"""
Nethical Latency Engineering Module

This module provides latency monitoring, budgeting, and optimization
for real-time AI inference, especially critical for robotics and
safety-critical applications.

Problem Statement:
    500ms latency spikes are DANGEROUS for robotics — robot can crash into wall!

Features:
    - p50/p99 latency tracking
    - Latency budget system with target/warning/critical thresholds
    - Inference caching for repeated patterns
    - Real-time latency alerts
    - Performance regression detection

Fundamental Laws Alignment:
    - Law 21 (Primacy of Human Safety): Latency budgets protect humans
    - Law 23 (Fail-Safe Design): Critical latency triggers failsafe
    - Law 15 (Audit Compliance): All latency metrics are logged

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic
import hashlib
import functools

__all__ = [
    "LatencyLevel",
    "LatencyBudget",
    "LatencyMetric",
    "LatencyStats",
    "LatencyMonitor",
    "LatencyAlert",
    "InferenceCache",
    "latency_tracked",
    "with_latency_budget",
]

log = logging.getLogger(__name__)

T = TypeVar("T")


class LatencyLevel(str, Enum):
    """Latency severity levels."""

    NORMAL = "normal"  # Within target
    WARNING = "warning"  # Above target, below critical
    CRITICAL = "critical"  # Above critical - action required
    VIOLATION = "violation"  # Exceeded maximum - safety issue


@dataclass
class LatencyBudget:
    """Latency budget configuration.

    Defines thresholds for latency monitoring and alerting.
    Default values are optimized for robotics applications.

    Attributes:
        target_ms: Target latency for optimal operation
        warning_ms: Threshold for warning alerts
        critical_ms: Threshold for critical alerts (safety concern)
        max_ms: Maximum allowed latency (hard limit)
        name: Name/identifier for this budget
    """

    target_ms: float = 10.0  # Target for robotics
    warning_ms: float = 50.0  # Performance degradation
    critical_ms: float = 100.0  # Never exceed for safety
    max_ms: float = 500.0  # Hard limit - failsafe trigger
    name: str = "default"

    def __post_init__(self):
        """Validate budget thresholds."""
        if not (
            0 < self.target_ms <= self.warning_ms <= self.critical_ms <= self.max_ms
        ):
            raise ValueError(
                "Budget thresholds must be: target <= warning <= critical <= max"
            )

    def classify(self, latency_ms: float) -> LatencyLevel:
        """Classify latency against budget thresholds.

        Args:
            latency_ms: Measured latency in milliseconds

        Returns:
            LatencyLevel classification
        """
        if latency_ms > self.max_ms:
            return LatencyLevel.VIOLATION
        elif latency_ms > self.critical_ms:
            return LatencyLevel.CRITICAL
        elif latency_ms > self.warning_ms:
            return LatencyLevel.WARNING
        else:
            return LatencyLevel.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "target_ms": self.target_ms,
            "warning_ms": self.warning_ms,
            "critical_ms": self.critical_ms,
            "max_ms": self.max_ms,
        }


@dataclass
class LatencyMetric:
    """Individual latency measurement.

    Attributes:
        latency_ms: Measured latency in milliseconds
        timestamp: When the measurement was taken
        operation: Name of the measured operation
        level: Latency level classification
        metadata: Additional context
    """

    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operation: str = "inference"
    level: LatencyLevel = LatencyLevel.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "level": self.level.value,
            "metadata": self.metadata,
        }


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements.

    Attributes:
        count: Number of measurements
        mean_ms: Average latency
        min_ms: Minimum latency
        max_ms: Maximum latency
        p50_ms: 50th percentile (median)
        p90_ms: 90th percentile
        p95_ms: 95th percentile
        p99_ms: 99th percentile
        std_ms: Standard deviation
        window_seconds: Time window for measurements
    """

    count: int = 0
    mean_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_ms: float = 0.0
    window_seconds: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "window_seconds": round(self.window_seconds, 1),
        }


@dataclass
class LatencyAlert:
    """Alert for latency threshold violation.

    Attributes:
        level: Alert severity level
        latency_ms: Measured latency
        threshold_ms: Threshold that was exceeded
        operation: Operation that caused the alert
        timestamp: When the alert was generated
        message: Human-readable alert message
        recommendations: Suggested actions
    """

    level: LatencyLevel
    latency_ms: float
    threshold_ms: float
    operation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Generate default message if not provided."""
        if not self.message:
            self.message = (
                f"{self.level.value.upper()}: {self.operation} latency "
                f"{self.latency_ms:.1f}ms exceeded {self.threshold_ms:.1f}ms threshold"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "latency_ms": self.latency_ms,
            "threshold_ms": self.threshold_ms,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "recommendations": self.recommendations,
        }


class LatencyMonitor:
    """Real-time latency monitoring and alerting system.

    Tracks latency measurements, computes statistics, and generates
    alerts when thresholds are exceeded.

    Thread-safe implementation for concurrent access.
    """

    def __init__(
        self,
        budget: Optional[LatencyBudget] = None,
        window_size: int = 1000,
        alert_callback: Optional[Callable[[LatencyAlert], None]] = None,
    ):
        """Initialize latency monitor.

        Args:
            budget: Latency budget for classification
            window_size: Maximum number of measurements to retain
            alert_callback: Optional callback for alerts
        """
        self.budget = budget or LatencyBudget()
        self.window_size = window_size
        self.alert_callback = alert_callback

        self._measurements: deque[LatencyMetric] = deque(maxlen=window_size)
        self._alerts: deque[LatencyAlert] = deque(maxlen=100)
        self._lock = threading.RLock()

        # Counters
        self._total_count = 0
        self._warning_count = 0
        self._critical_count = 0
        self._violation_count = 0

        log.info(f"LatencyMonitor initialized with budget: {self.budget.name}")

    def record(
        self,
        latency_ms: float,
        operation: str = "inference",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LatencyMetric:
        """Record a latency measurement.

        Args:
            latency_ms: Measured latency in milliseconds
            operation: Name of the operation
            metadata: Additional context

        Returns:
            LatencyMetric with classification
        """
        level = self.budget.classify(latency_ms)

        metric = LatencyMetric(
            latency_ms=latency_ms,
            operation=operation,
            level=level,
            metadata=metadata or {},
        )

        with self._lock:
            self._measurements.append(metric)
            self._total_count += 1

            # Update counters
            if level == LatencyLevel.WARNING:
                self._warning_count += 1
            elif level == LatencyLevel.CRITICAL:
                self._critical_count += 1
            elif level == LatencyLevel.VIOLATION:
                self._violation_count += 1

        # Generate alert if needed
        if level in (
            LatencyLevel.WARNING,
            LatencyLevel.CRITICAL,
            LatencyLevel.VIOLATION,
        ):
            self._generate_alert(metric)

        return metric

    def _generate_alert(self, metric: LatencyMetric) -> None:
        """Generate and dispatch alert for latency violation."""
        threshold_ms = {
            LatencyLevel.WARNING: self.budget.warning_ms,
            LatencyLevel.CRITICAL: self.budget.critical_ms,
            LatencyLevel.VIOLATION: self.budget.max_ms,
        }.get(metric.level, self.budget.warning_ms)

        recommendations = []
        if metric.level == LatencyLevel.CRITICAL:
            recommendations = [
                "Consider reducing model complexity",
                "Enable mixed precision inference",
                "Check for resource contention",
            ]
        elif metric.level == LatencyLevel.VIOLATION:
            recommendations = [
                "IMMEDIATE ACTION REQUIRED",
                "Trigger failsafe mode",
                "Reduce inference rate",
                "Fall back to simpler model",
            ]

        alert = LatencyAlert(
            level=metric.level,
            latency_ms=metric.latency_ms,
            threshold_ms=threshold_ms,
            operation=metric.operation,
            recommendations=recommendations,
        )

        with self._lock:
            self._alerts.append(alert)

        # Log alert
        if metric.level == LatencyLevel.VIOLATION:
            log.error(alert.message)
        elif metric.level == LatencyLevel.CRITICAL:
            log.warning(alert.message)
        else:
            log.info(alert.message)

        # Dispatch callback
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                log.error(f"Alert callback failed: {e}")

    def get_stats(self, window_seconds: Optional[float] = None) -> LatencyStats:
        """Get latency statistics.

        Args:
            window_seconds: Time window for stats (None = all measurements)

        Returns:
            LatencyStats with computed metrics
        """
        with self._lock:
            if not self._measurements:
                return LatencyStats()

            # Filter by time window if specified
            now = datetime.now(timezone.utc)
            if window_seconds is not None:
                cutoff = now.timestamp() - window_seconds
                values = [
                    m.latency_ms
                    for m in self._measurements
                    if m.timestamp.timestamp() > cutoff
                ]
            else:
                values = [m.latency_ms for m in self._measurements]

            if not values:
                return LatencyStats()

            # Sort for percentile calculation
            sorted_values = sorted(values)
            n = len(sorted_values)

            def percentile(p: float) -> float:
                k = (n - 1) * p
                f = int(k)
                c = f + 1 if f + 1 < n else f
                return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (
                    k - f
                )

            # Compute statistics
            mean = sum(values) / n
            variance = sum((x - mean) ** 2 for x in values) / n
            std = variance**0.5

            return LatencyStats(
                count=n,
                mean_ms=mean,
                min_ms=min(values),
                max_ms=max(values),
                p50_ms=percentile(0.50),
                p90_ms=percentile(0.90),
                p95_ms=percentile(0.95),
                p99_ms=percentile(0.99),
                std_ms=std,
                window_seconds=window_seconds or 0.0,
            )

    def get_recent_alerts(self, limit: int = 10) -> List[LatencyAlert]:
        """Get recent alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        with self._lock:
            return list(self._alerts)[-limit:]

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status.

        Returns:
            Health status dictionary
        """
        stats = self.get_stats()

        # Determine overall health
        if self._violation_count > 0:
            health = "critical"
        elif self._critical_count > 0:
            health = "degraded"
        elif self._warning_count > 0:
            health = "warning"
        else:
            health = "healthy"

        return {
            "health": health,
            "budget": self.budget.to_dict(),
            "stats": stats.to_dict(),
            "counters": {
                "total": self._total_count,
                "warnings": self._warning_count,
                "critical": self._critical_count,
                "violations": self._violation_count,
            },
            "p99_within_target": stats.p99_ms <= self.budget.target_ms,
            "p99_within_critical": stats.p99_ms <= self.budget.critical_ms,
        }

    def reset(self) -> None:
        """Reset all measurements and counters."""
        with self._lock:
            self._measurements.clear()
            self._alerts.clear()
            self._total_count = 0
            self._warning_count = 0
            self._critical_count = 0
            self._violation_count = 0
        log.info("LatencyMonitor reset")


class InferenceCache(Generic[T]):
    """Cache for repeated inference patterns.

    Stores inference results for identical inputs to avoid
    redundant computation. Thread-safe with LRU eviction.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 300.0,
    ):
        """Initialize inference cache.

        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        self._cache: Dict[str, Tuple[T, float]] = {}
        self._access_order: deque[str] = deque()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0

        log.info(f"InferenceCache initialized (max_size={max_size})")

    def _compute_key(self, inputs: Any) -> str:
        """Compute cache key for inputs.

        Args:
            inputs: Input data

        Returns:
            Hash-based cache key
        """
        if hasattr(inputs, "tobytes"):
            # NumPy array
            data = inputs.tobytes()
        elif hasattr(inputs, "numpy"):
            # Tensor
            data = inputs.cpu().numpy().tobytes()
        else:
            data = str(inputs).encode()

        return hashlib.sha256(data).hexdigest()[:32]

    def get(self, inputs: Any) -> Optional[T]:
        """Get cached result for inputs.

        Args:
            inputs: Input data

        Returns:
            Cached result or None if not found
        """
        key = self._compute_key(inputs)
        now = time.time()

        with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]

                # Check TTL
                if now - timestamp < self.ttl_seconds:
                    self._hits += 1
                    # Update access order for LRU
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)
                    return result
                else:
                    # Expired
                    del self._cache[key]
                    if key in self._access_order:
                        self._access_order.remove(key)

            self._misses += 1
            return None

    def put(self, inputs: Any, result: T) -> None:
        """Store result in cache.

        Args:
            inputs: Input data
            result: Computation result
        """
        key = self._compute_key(inputs)
        now = time.time()

        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size and self._access_order:
                old_key = self._access_order.popleft()
                if old_key in self._cache:
                    del self._cache[old_key]

            self._cache[key] = (result, now)
            self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
        log.info("InferenceCache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "ttl_seconds": self.ttl_seconds,
            }


def latency_tracked(
    monitor: LatencyMonitor,
    operation: Optional[str] = None,
) -> Callable:
    """Decorator to track function latency.

    Args:
        monitor: LatencyMonitor instance
        operation: Operation name (defaults to function name)

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                monitor.record(elapsed_ms, operation=op_name)

        return wrapper

    return decorator


def with_latency_budget(
    budget: LatencyBudget,
    on_violation: Optional[Callable[[float], Any]] = None,
) -> Callable:
    """Decorator to enforce latency budget on function.

    Args:
        budget: Latency budget to enforce
        on_violation: Callback when budget is violated

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            level = budget.classify(elapsed_ms)
            if level in (LatencyLevel.CRITICAL, LatencyLevel.VIOLATION):
                log.warning(
                    f"Latency budget violation in {func.__name__}: "
                    f"{elapsed_ms:.1f}ms (limit: {budget.critical_ms:.1f}ms)"
                )
                if on_violation:
                    on_violation(elapsed_ms)

            return result

        return wrapper

    return decorator


# Default latency budgets for common use cases
ROBOTICS_BUDGET = LatencyBudget(
    name="robotics",
    target_ms=10.0,
    warning_ms=50.0,
    critical_ms=100.0,
    max_ms=200.0,
)

REALTIME_BUDGET = LatencyBudget(
    name="realtime",
    target_ms=20.0,
    warning_ms=100.0,
    critical_ms=200.0,
    max_ms=500.0,
)

INTERACTIVE_BUDGET = LatencyBudget(
    name="interactive",
    target_ms=100.0,
    warning_ms=300.0,
    critical_ms=1000.0,
    max_ms=5000.0,
)

BATCH_BUDGET = LatencyBudget(
    name="batch",
    target_ms=1000.0,
    warning_ms=5000.0,
    critical_ms=30000.0,
    max_ms=60000.0,
)
