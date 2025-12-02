"""
Circuit Breaker - Latency-Based Circuit Breaker

Protects system from latency spikes by failing fast.
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitConfig:
    """
    Circuit breaker configuration.

    Attributes:
        max_latency_ms: Maximum allowed latency
        failure_threshold: Number of failures before opening
        recovery_timeout_seconds: Time before trying again
        half_open_requests: Requests to allow in half-open state
        window_size: Number of samples for latency calculation
    """

    max_latency_ms: float = 10.0
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_requests: int = 3
    window_size: int = 100


class CircuitBreaker:
    """
    Latency-based circuit breaker.

    Protects system from cascading failures by:
    - Monitoring latency
    - Opening circuit when latency exceeds threshold
    - Failing fast when circuit is open
    - Gradually recovering in half-open state

    States:
    - CLOSED: Normal operation, monitoring latency
    - OPEN: Failing fast, not processing requests
    - HALF_OPEN: Testing if system has recovered
    """

    def __init__(
        self,
        max_latency_ms: float = 10.0,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        config: Optional[CircuitConfig] = None,
    ):
        """
        Initialize CircuitBreaker.

        Args:
            max_latency_ms: Maximum allowed latency
            failure_threshold: Failures before opening
            recovery_timeout_seconds: Recovery timeout
            config: Full configuration (overrides other args)
        """
        if config:
            self.config = config
        else:
            self.config = CircuitConfig(
                max_latency_ms=max_latency_ms,
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
            )

        # State
        self._state = CircuitState.CLOSED
        self._lock = threading.RLock()

        # Failure tracking
        self._consecutive_failures = 0
        self._last_failure_time: Optional[float] = None
        self._open_time: Optional[float] = None

        # Latency tracking
        self._latency_samples: List[float] = []

        # Half-open tracking
        self._half_open_successes = 0

        # Metrics
        self._total_requests = 0
        self._failed_requests = 0
        self._rejected_requests = 0

        logger.info(
            f"CircuitBreaker initialized: max_latency={self.config.max_latency_ms}ms"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._maybe_transition_state()
            return self._state

    def can_process(self) -> bool:
        """
        Check if circuit allows processing.

        Returns:
            True if request can be processed
        """
        with self._lock:
            self._maybe_transition_state()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                if self._half_open_successes < self.config.half_open_requests:
                    return True

            if self._state == CircuitState.OPEN:
                self._rejected_requests += 1

            return False

    def record_success(self, latency_ms: float):
        """
        Record a successful request.

        Args:
            latency_ms: Request latency in milliseconds
        """
        with self._lock:
            self._total_requests += 1

            # Record latency
            self._latency_samples.append(latency_ms)
            if len(self._latency_samples) > self.config.window_size:
                self._latency_samples.pop(0)

            # Check if latency is within threshold
            if latency_ms <= self.config.max_latency_ms:
                self._consecutive_failures = 0

                if self._state == CircuitState.HALF_OPEN:
                    self._half_open_successes += 1
                    if self._half_open_successes >= self.config.half_open_requests:
                        self._close()
            else:
                self._record_failure()

    def record_failure(self):
        """Record a failed request."""
        with self._lock:
            self._record_failure()

    def record_latency(self, latency_ms: float):
        """
        Record request latency.

        Args:
            latency_ms: Request latency in milliseconds
        """
        if latency_ms <= self.config.max_latency_ms:
            self.record_success(latency_ms)
        else:
            with self._lock:
                self._latency_samples.append(latency_ms)
                if len(self._latency_samples) > self.config.window_size:
                    self._latency_samples.pop(0)
                self._record_failure()

    def _record_failure(self):
        """Internal failure recording."""
        self._failed_requests += 1
        self._consecutive_failures += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._open()
        elif self._consecutive_failures >= self.config.failure_threshold:
            self._open()

    def _open(self):
        """Open the circuit."""
        if self._state != CircuitState.OPEN:
            logger.warning("Circuit breaker opened")
        self._state = CircuitState.OPEN
        self._open_time = time.time()

    def _close(self):
        """Close the circuit."""
        if self._state != CircuitState.CLOSED:
            logger.info("Circuit breaker closed")
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._half_open_successes = 0

    def _half_open(self):
        """Transition to half-open state."""
        if self._state != CircuitState.HALF_OPEN:
            logger.info("Circuit breaker half-open")
        self._state = CircuitState.HALF_OPEN
        self._half_open_successes = 0

    def _maybe_transition_state(self):
        """Check if state should transition."""
        if self._state == CircuitState.OPEN:
            if self._open_time is not None:
                elapsed = time.time() - self._open_time
                if elapsed >= self.config.recovery_timeout_seconds:
                    self._half_open()

    def force_open(self):
        """Force circuit to open state."""
        with self._lock:
            self._open()

    def force_close(self):
        """Force circuit to closed state."""
        with self._lock:
            self._close()

    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self._close()
            self._latency_samples.clear()
            self._total_requests = 0
            self._failed_requests = 0
            self._rejected_requests = 0

    def get_latency_percentile(self, percentile: float) -> Optional[float]:
        """
        Get latency percentile.

        Args:
            percentile: Percentile (0-100)

        Returns:
            Latency at percentile or None
        """
        with self._lock:
            if not self._latency_samples:
                return None
            sorted_samples = sorted(self._latency_samples)
            idx = int(len(sorted_samples) * percentile / 100)
            return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            return {
                "state": self._state.value,
                "consecutive_failures": self._consecutive_failures,
                "total_requests": self._total_requests,
                "failed_requests": self._failed_requests,
                "rejected_requests": self._rejected_requests,
                "p50_latency_ms": self.get_latency_percentile(50),
                "p95_latency_ms": self.get_latency_percentile(95),
                "p99_latency_ms": self.get_latency_percentile(99),
                "max_latency_ms": self.config.max_latency_ms,
                "failure_threshold": self.config.failure_threshold,
            }
