"""
Adaptive Latency Optimization for Satellite Connections

Provides intelligent handling of variable satellite latencies including:
- Adaptive timeout configuration
- Request prioritization during high-latency periods
- Predictive latency modeling
- Bandwidth-aware request batching
"""

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class LatencyProfile(Enum):
    """Latency profile classifications."""

    EXCELLENT = "excellent"  # <30ms
    GOOD = "good"  # 30-50ms
    ACCEPTABLE = "acceptable"  # 50-100ms
    DEGRADED = "degraded"  # 100-250ms
    POOR = "poor"  # 250-500ms
    CRITICAL = "critical"  # >500ms


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""

    timestamp: datetime
    latency_ms: float
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    priority: RequestPriority = RequestPriority.NORMAL
    success: bool = True


@dataclass
class BatchRequest:
    """Batched request for satellite optimization."""

    request_id: str
    data: bytes
    priority: RequestPriority
    timeout_ms: float
    created_at: datetime
    callback: Optional[Callable] = None


@dataclass
class LatencyOptimizerConfig:
    """Configuration for latency optimization."""

    # Timeout settings
    base_timeout_ms: float = 1000.0
    min_timeout_ms: float = 500.0
    max_timeout_ms: float = 30000.0
    timeout_multiplier: float = 2.0

    # Latency thresholds
    excellent_threshold_ms: float = 30.0
    good_threshold_ms: float = 50.0
    acceptable_threshold_ms: float = 100.0
    degraded_threshold_ms: float = 250.0
    poor_threshold_ms: float = 500.0

    # Batching settings
    batching_enabled: bool = True
    batch_window_ms: float = 100.0
    max_batch_size: int = 10
    max_batch_bytes: int = 32768

    # Prediction settings
    prediction_enabled: bool = True
    measurement_window_seconds: int = 300
    max_measurements: int = 1000

    # Priority settings
    priority_timeout_multipliers: Dict[RequestPriority, float] = field(
        default_factory=lambda: {
            RequestPriority.LOW: 3.0,
            RequestPriority.NORMAL: 2.0,
            RequestPriority.HIGH: 1.5,
            RequestPriority.URGENT: 1.0,
        }
    )


class LatencyOptimizer:
    """
    Adaptive latency optimization for satellite connections.

    Provides intelligent handling of variable satellite latencies
    through adaptive timeouts, request prioritization, and batching.
    """

    def __init__(self, config: Optional[LatencyOptimizerConfig] = None):
        """
        Initialize latency optimizer.

        Args:
            config: Optimizer configuration
        """
        self.config = config or LatencyOptimizerConfig()

        # Latency measurements
        self._measurements: List[LatencyMeasurement] = []
        self._current_profile = LatencyProfile.GOOD

        # Request batching
        self._batch_queue: List[BatchRequest] = []
        self._batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None

        # Prediction model
        self._predicted_latency_ms: float = 50.0
        self._latency_trend: float = 0.0  # Positive = increasing, negative = decreasing

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "on_profile_change": [],
            "on_batch_ready": [],
        }

    @property
    def current_profile(self) -> LatencyProfile:
        """Get current latency profile."""
        return self._current_profile

    @property
    def predicted_latency_ms(self) -> float:
        """Get predicted latency in milliseconds."""
        return self._predicted_latency_ms

    @property
    def latency_trend(self) -> float:
        """Get latency trend (positive = worsening, negative = improving)."""
        return self._latency_trend

    def record_measurement(
        self,
        latency_ms: float,
        request_size_bytes: int = 0,
        response_size_bytes: int = 0,
        priority: RequestPriority = RequestPriority.NORMAL,
        success: bool = True,
    ):
        """
        Record a latency measurement.

        Args:
            latency_ms: Measured latency in milliseconds
            request_size_bytes: Request size in bytes
            response_size_bytes: Response size in bytes
            priority: Request priority
            success: Whether request succeeded
        """
        measurement = LatencyMeasurement(
            timestamp=datetime.utcnow(),
            latency_ms=latency_ms,
            request_size_bytes=request_size_bytes,
            response_size_bytes=response_size_bytes,
            priority=priority,
            success=success,
        )

        self._measurements.append(measurement)

        # Trim old measurements
        cutoff = datetime.utcnow() - timedelta(
            seconds=self.config.measurement_window_seconds
        )
        self._measurements = [m for m in self._measurements if m.timestamp > cutoff]

        if len(self._measurements) > self.config.max_measurements:
            self._measurements = self._measurements[-self.config.max_measurements :]

        # Update profile and predictions
        self._update_profile()
        if self.config.prediction_enabled:
            self._update_predictions()

    def _update_profile(self):
        """Update current latency profile based on recent measurements."""
        if not self._measurements:
            return

        # Calculate recent average
        recent = self._measurements[-20:] if len(self._measurements) > 20 else self._measurements
        avg_latency = statistics.mean(m.latency_ms for m in recent)

        # Determine profile
        old_profile = self._current_profile

        if avg_latency <= self.config.excellent_threshold_ms:
            self._current_profile = LatencyProfile.EXCELLENT
        elif avg_latency <= self.config.good_threshold_ms:
            self._current_profile = LatencyProfile.GOOD
        elif avg_latency <= self.config.acceptable_threshold_ms:
            self._current_profile = LatencyProfile.ACCEPTABLE
        elif avg_latency <= self.config.degraded_threshold_ms:
            self._current_profile = LatencyProfile.DEGRADED
        elif avg_latency <= self.config.poor_threshold_ms:
            self._current_profile = LatencyProfile.POOR
        else:
            self._current_profile = LatencyProfile.CRITICAL

        # Notify on profile change
        if old_profile != self._current_profile:
            logger.info(
                f"Latency profile changed: {old_profile.value} -> {self._current_profile.value}"
            )
            for callback in self._callbacks["on_profile_change"]:
                callback(old_profile, self._current_profile)

    def _update_predictions(self):
        """Update latency predictions using recent measurements."""
        if len(self._measurements) < 5:
            return

        # Calculate moving average
        recent = self._measurements[-50:]
        latencies = [m.latency_ms for m in recent]
        self._predicted_latency_ms = statistics.mean(latencies)

        # Calculate trend (simple linear regression slope)
        if len(latencies) >= 10:
            n = len(latencies)
            x_sum = sum(range(n))
            y_sum = sum(latencies)
            xy_sum = sum(i * latencies[i] for i in range(n))
            x2_sum = sum(i ** 2 for i in range(n))

            denominator = n * x2_sum - x_sum ** 2
            if denominator != 0:
                self._latency_trend = (n * xy_sum - x_sum * y_sum) / denominator

    def get_adaptive_timeout(
        self,
        priority: RequestPriority = RequestPriority.NORMAL,
        payload_size_bytes: int = 0,
    ) -> float:
        """
        Get adaptive timeout based on current conditions.

        Args:
            priority: Request priority
            payload_size_bytes: Request payload size

        Returns:
            Recommended timeout in milliseconds
        """
        # Start with predicted latency
        base = max(self._predicted_latency_ms, self.config.base_timeout_ms)

        # Apply profile multiplier
        profile_multipliers = {
            LatencyProfile.EXCELLENT: 1.5,
            LatencyProfile.GOOD: 2.0,
            LatencyProfile.ACCEPTABLE: 2.5,
            LatencyProfile.DEGRADED: 3.0,
            LatencyProfile.POOR: 4.0,
            LatencyProfile.CRITICAL: 5.0,
        }
        base *= profile_multipliers.get(self._current_profile, 2.0)

        # Apply priority multiplier
        priority_mult = self.config.priority_timeout_multipliers.get(priority, 2.0)
        base *= priority_mult

        # Adjust for payload size (estimate ~1ms per KB at 1Mbps)
        if payload_size_bytes > 0:
            size_adjustment = (payload_size_bytes / 1024) * 8  # ms at 1Mbps
            base += size_adjustment

        # Apply trend adjustment
        if self._latency_trend > 0:  # Worsening
            base *= 1 + (self._latency_trend * 0.1)

        # Clamp to min/max
        return max(self.config.min_timeout_ms, min(base, self.config.max_timeout_ms))

    async def queue_request(
        self,
        request_id: str,
        data: bytes,
        priority: RequestPriority = RequestPriority.NORMAL,
        callback: Optional[Callable] = None,
    ) -> bool:
        """
        Queue a request for batch processing.

        Args:
            request_id: Unique request identifier
            data: Request data
            priority: Request priority
            callback: Callback when request is batched

        Returns:
            True if queued successfully
        """
        if not self.config.batching_enabled:
            return False

        timeout = self.get_adaptive_timeout(priority, len(data))

        request = BatchRequest(
            request_id=request_id,
            data=data,
            priority=priority,
            timeout_ms=timeout,
            created_at=datetime.utcnow(),
            callback=callback,
        )

        async with self._batch_lock:
            self._batch_queue.append(request)

            # Check if batch is ready
            if self._should_flush_batch():
                await self._flush_batch()

        return True

    def _should_flush_batch(self) -> bool:
        """Check if batch should be flushed."""
        if not self._batch_queue:
            return False

        # Check batch size
        if len(self._batch_queue) >= self.config.max_batch_size:
            return True

        # Check total bytes
        total_bytes = sum(len(r.data) for r in self._batch_queue)
        if total_bytes >= self.config.max_batch_bytes:
            return True

        # Check for urgent requests
        if any(r.priority == RequestPriority.URGENT for r in self._batch_queue):
            return True

        # Check batch window
        oldest = min(r.created_at for r in self._batch_queue)
        age_ms = (datetime.utcnow() - oldest).total_seconds() * 1000
        if age_ms >= self.config.batch_window_ms:
            return True

        return False

    async def _flush_batch(self):
        """Flush the current batch."""
        if not self._batch_queue:
            return

        # Sort by priority (highest first)
        batch = sorted(self._batch_queue, key=lambda r: r.priority.value, reverse=True)
        self._batch_queue = []

        # Notify callbacks
        for callback in self._callbacks["on_batch_ready"]:
            callback(batch)

        # Call individual request callbacks
        for request in batch:
            if request.callback:
                try:
                    request.callback(request)
                except Exception as e:
                    logger.error(f"Error in batch callback: {e}")

    def should_defer_request(self, priority: RequestPriority) -> bool:
        """
        Check if request should be deferred based on current conditions.

        Args:
            priority: Request priority

        Returns:
            True if request should be deferred
        """
        if priority == RequestPriority.URGENT:
            return False

        if self._current_profile == LatencyProfile.CRITICAL:
            return priority == RequestPriority.LOW

        if self._current_profile == LatencyProfile.POOR:
            return priority == RequestPriority.LOW and self._latency_trend > 0

        return False

    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get optimization recommendations based on current conditions.

        Returns:
            Dictionary with recommendations
        """
        recommendations = []

        if self._current_profile in (LatencyProfile.POOR, LatencyProfile.CRITICAL):
            recommendations.append("Enable request batching to reduce overhead")
            recommendations.append("Defer non-urgent requests until conditions improve")
            recommendations.append("Consider enabling compression for payloads")

        if self._latency_trend > 0.5:
            recommendations.append("Latency is trending upward - preemptively increase timeouts")

        if self.config.batching_enabled and len(self._batch_queue) > 0:
            recommendations.append(
                f"Batch queue has {len(self._batch_queue)} pending requests"
            )

        return {
            "current_profile": self._current_profile.value,
            "predicted_latency_ms": self._predicted_latency_ms,
            "latency_trend": self._latency_trend,
            "recommended_timeout_ms": self.get_adaptive_timeout(),
            "recommendations": recommendations,
            "batching_enabled": self.config.batching_enabled,
            "batch_queue_size": len(self._batch_queue),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get latency statistics.

        Returns:
            Statistics dictionary
        """
        if not self._measurements:
            return {
                "count": 0,
                "profile": self._current_profile.value,
            }

        latencies = [m.latency_ms for m in self._measurements]
        successful = [m for m in self._measurements if m.success]

        return {
            "count": len(self._measurements),
            "profile": self._current_profile.value,
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
            "success_rate": len(successful) / len(self._measurements),
            "predicted_ms": self._predicted_latency_ms,
            "trend": self._latency_trend,
        }

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for optimizer events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
