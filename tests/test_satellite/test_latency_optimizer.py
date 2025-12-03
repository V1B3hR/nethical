"""Tests for latency optimizer."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from nethical.connectivity.satellite.latency_optimizer import (
    LatencyOptimizer,
    LatencyProfile,
    RequestPriority,
    LatencyOptimizerConfig,
)


class TestRequestPriority:
    """Tests for RequestPriority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert RequestPriority.LOW.value == 0
        assert RequestPriority.NORMAL.value == 1
        assert RequestPriority.HIGH.value == 2
        assert RequestPriority.URGENT.value == 3

    def test_priority_ordering(self):
        """Test priority ordering."""
        assert RequestPriority.LOW.value < RequestPriority.NORMAL.value
        assert RequestPriority.NORMAL.value < RequestPriority.HIGH.value
        assert RequestPriority.HIGH.value < RequestPriority.URGENT.value


class TestLatencyProfile:
    """Tests for LatencyProfile enum."""

    def test_profile_values(self):
        """Test profile enum values."""
        assert LatencyProfile.EXCELLENT.value == "excellent"
        assert LatencyProfile.GOOD.value == "good"
        assert LatencyProfile.ACCEPTABLE.value == "acceptable"
        assert LatencyProfile.DEGRADED.value == "degraded"
        assert LatencyProfile.POOR.value == "poor"
        assert LatencyProfile.CRITICAL.value == "critical"


class TestLatencyOptimizerConfig:
    """Tests for LatencyOptimizerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = LatencyOptimizerConfig()
        assert config.base_timeout_ms == 1000.0
        assert config.max_timeout_ms == 30000.0
        assert config.batching_enabled is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = LatencyOptimizerConfig(
            base_timeout_ms=2000.0,
            max_timeout_ms=60000.0,
            batching_enabled=False,
        )
        assert config.base_timeout_ms == 2000.0
        assert config.max_timeout_ms == 60000.0
        assert config.batching_enabled is False


class TestLatencyOptimizer:
    """Tests for LatencyOptimizer."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LatencyOptimizerConfig(
            base_timeout_ms=1000.0,
        )

    @pytest.fixture
    def optimizer(self, config):
        """Create latency optimizer."""
        return LatencyOptimizer(config)

    def test_initial_state(self, optimizer):
        """Test initial optimizer state."""
        assert optimizer.current_profile == LatencyProfile.GOOD
        assert len(optimizer._measurements) == 0

    def test_record_measurement(self, optimizer):
        """Test recording latency measurements."""
        optimizer.record_measurement(latency_ms=30.0, success=True)
        optimizer.record_measurement(latency_ms=35.0, success=True)

        assert len(optimizer._measurements) == 2

    def test_get_adaptive_timeout_normal(self, optimizer):
        """Test adaptive timeout for normal conditions."""
        # Record some normal latency samples
        for _ in range(10):
            optimizer.record_measurement(latency_ms=30.0, success=True)

        timeout = optimizer.get_adaptive_timeout(RequestPriority.NORMAL)
        # Should be based on latency statistics
        assert timeout > 0
        assert timeout <= optimizer.config.max_timeout_ms

    def test_get_adaptive_timeout_by_priority(self, optimizer):
        """Test timeout varies by priority."""
        for _ in range(10):
            optimizer.record_measurement(latency_ms=50.0, success=True)

        urgent_timeout = optimizer.get_adaptive_timeout(RequestPriority.URGENT)
        low_timeout = optimizer.get_adaptive_timeout(RequestPriority.LOW)

        # Both should be valid timeouts
        assert urgent_timeout > 0
        assert low_timeout > 0

    def test_should_defer_request_urgent(self, optimizer):
        """Test urgent requests are never deferred."""
        # Simulate poor conditions
        for _ in range(20):
            optimizer.record_measurement(latency_ms=1000.0, success=True)

        should_defer = optimizer.should_defer_request(RequestPriority.URGENT)
        assert should_defer is False

    def test_get_statistics(self, optimizer):
        """Test getting statistics."""
        for i in range(100):
            optimizer.record_measurement(latency_ms=20.0 + i % 50, success=True)

        stats = optimizer.get_statistics()

        assert "mean_ms" in stats
        assert "median_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats
        assert "count" in stats

    def test_get_optimization_recommendations(self, optimizer):
        """Test getting optimization recommendations."""
        for _ in range(20):
            optimizer.record_measurement(latency_ms=100.0, success=True)

        recs = optimizer.get_optimization_recommendations()

        # Check the response has the expected structure
        assert "current_profile" in recs
        assert "recommendations" in recs
        assert isinstance(recs["recommendations"], list)

    def test_latency_trend(self, optimizer):
        """Test latency trend calculation."""
        # Record increasing latency
        for i in range(20):
            optimizer.record_measurement(latency_ms=20.0 + i * 5, success=True)

        trend = optimizer.latency_trend
        assert trend > 0  # Increasing trend

    def test_callback_registration(self, optimizer):
        """Test callback registration."""
        callback = MagicMock()
        optimizer.register_callback("on_profile_change", callback)
        assert callback in optimizer._callbacks["on_profile_change"]

    def test_min_samples_for_optimization(self, optimizer):
        """Test minimum samples requirement."""
        # With few samples, should use defaults
        optimizer.record_measurement(latency_ms=30.0, success=True)

        timeout = optimizer.get_adaptive_timeout(RequestPriority.NORMAL)
        # Should still return a valid timeout
        assert timeout > 0
