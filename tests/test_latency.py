"""
Tests for the latency monitoring and engineering module.

Tests cover:
- Latency budget configuration
- Latency monitoring and statistics
- Alert generation
- Inference caching
- Decorators
"""

import pytest
import time
import numpy as np
from datetime import datetime, timezone

from nethical.core.latency import (
    LatencyLevel,
    LatencyBudget,
    LatencyMetric,
    LatencyStats,
    LatencyAlert,
    LatencyMonitor,
    InferenceCache,
    latency_tracked,
    with_latency_budget,
    ROBOTICS_BUDGET,
    REALTIME_BUDGET,
    INTERACTIVE_BUDGET,
    BATCH_BUDGET,
)


class TestLatencyLevel:
    """Test LatencyLevel enum."""

    def test_level_values(self):
        """Test enum values."""
        assert LatencyLevel.NORMAL.value == "normal"
        assert LatencyLevel.WARNING.value == "warning"
        assert LatencyLevel.CRITICAL.value == "critical"
        assert LatencyLevel.VIOLATION.value == "violation"


class TestLatencyBudget:
    """Test LatencyBudget dataclass."""

    def test_default_values(self):
        """Test default budget values."""
        budget = LatencyBudget()
        assert budget.target_ms == 10.0
        assert budget.warning_ms == 50.0
        assert budget.critical_ms == 100.0
        assert budget.max_ms == 500.0
        assert budget.name == "default"

    def test_custom_values(self):
        """Test custom budget values."""
        budget = LatencyBudget(
            name="test",
            target_ms=5.0,
            warning_ms=20.0,
            critical_ms=50.0,
            max_ms=100.0,
        )
        assert budget.name == "test"
        assert budget.target_ms == 5.0

    def test_invalid_thresholds(self):
        """Test that invalid thresholds raise error."""
        with pytest.raises(ValueError):
            LatencyBudget(
                target_ms=100.0,
                warning_ms=50.0,  # Less than target - invalid
                critical_ms=200.0,
                max_ms=500.0,
            )

    def test_classify_normal(self):
        """Test classification of normal latency."""
        budget = LatencyBudget()
        assert budget.classify(5.0) == LatencyLevel.NORMAL
        assert budget.classify(10.0) == LatencyLevel.NORMAL

    def test_classify_warning(self):
        """Test classification of warning latency."""
        budget = LatencyBudget()
        assert budget.classify(51.0) == LatencyLevel.WARNING
        assert budget.classify(99.0) == LatencyLevel.WARNING

    def test_classify_critical(self):
        """Test classification of critical latency."""
        budget = LatencyBudget()
        assert budget.classify(101.0) == LatencyLevel.CRITICAL
        assert budget.classify(499.0) == LatencyLevel.CRITICAL

    def test_classify_violation(self):
        """Test classification of violation latency."""
        budget = LatencyBudget()
        assert budget.classify(501.0) == LatencyLevel.VIOLATION
        assert budget.classify(1000.0) == LatencyLevel.VIOLATION

    def test_to_dict(self):
        """Test serialization."""
        budget = LatencyBudget(name="test")
        result = budget.to_dict()
        assert result["name"] == "test"
        assert "target_ms" in result
        assert "warning_ms" in result


class TestLatencyMetric:
    """Test LatencyMetric dataclass."""

    def test_creation(self):
        """Test metric creation."""
        metric = LatencyMetric(
            latency_ms=15.5,
            operation="inference",
            level=LatencyLevel.NORMAL,
        )
        assert metric.latency_ms == 15.5
        assert metric.operation == "inference"
        assert metric.level == LatencyLevel.NORMAL
        assert isinstance(metric.timestamp, datetime)

    def test_to_dict(self):
        """Test serialization."""
        metric = LatencyMetric(
            latency_ms=15.5,
            operation="test",
            metadata={"key": "value"},
        )
        result = metric.to_dict()
        assert result["latency_ms"] == 15.5
        assert result["operation"] == "test"
        assert result["metadata"]["key"] == "value"


class TestLatencyStats:
    """Test LatencyStats dataclass."""

    def test_default_values(self):
        """Test default stats values."""
        stats = LatencyStats()
        assert stats.count == 0
        assert stats.mean_ms == 0.0
        assert stats.p99_ms == 0.0

    def test_to_dict(self):
        """Test serialization."""
        stats = LatencyStats(
            count=100,
            mean_ms=10.5,
            p99_ms=50.0,
        )
        result = stats.to_dict()
        assert result["count"] == 100
        assert result["mean_ms"] == 10.5
        assert result["p99_ms"] == 50.0


class TestLatencyAlert:
    """Test LatencyAlert dataclass."""

    def test_creation(self):
        """Test alert creation."""
        alert = LatencyAlert(
            level=LatencyLevel.CRITICAL,
            latency_ms=150.0,
            threshold_ms=100.0,
            operation="inference",
        )
        assert alert.level == LatencyLevel.CRITICAL
        assert alert.latency_ms == 150.0
        assert "CRITICAL" in alert.message

    def test_auto_message(self):
        """Test automatic message generation."""
        alert = LatencyAlert(
            level=LatencyLevel.WARNING,
            latency_ms=75.0,
            threshold_ms=50.0,
            operation="test",
        )
        assert "WARNING" in alert.message
        assert "75.0ms" in alert.message
        assert "50.0ms" in alert.message

    def test_to_dict(self):
        """Test serialization."""
        alert = LatencyAlert(
            level=LatencyLevel.VIOLATION,
            latency_ms=600.0,
            threshold_ms=500.0,
            operation="test",
            recommendations=["Action 1", "Action 2"],
        )
        result = alert.to_dict()
        assert result["level"] == "violation"
        assert len(result["recommendations"]) == 2


class TestLatencyMonitor:
    """Test LatencyMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = LatencyMonitor()
        assert monitor.budget is not None
        assert monitor.window_size == 1000

    def test_initialization_with_budget(self):
        """Test initialization with custom budget."""
        budget = LatencyBudget(name="custom")
        monitor = LatencyMonitor(budget=budget)
        assert monitor.budget.name == "custom"

    def test_record_normal(self):
        """Test recording normal latency."""
        monitor = LatencyMonitor()
        metric = monitor.record(5.0, operation="test")
        assert metric.latency_ms == 5.0
        assert metric.level == LatencyLevel.NORMAL

    def test_record_warning(self):
        """Test recording warning latency generates alert."""
        alerts_received = []

        def callback(alert):
            alerts_received.append(alert)

        monitor = LatencyMonitor(alert_callback=callback)
        monitor.record(75.0)  # Above warning threshold

        assert len(alerts_received) == 1
        assert alerts_received[0].level == LatencyLevel.WARNING

    def test_record_critical(self):
        """Test recording critical latency generates alert."""
        alerts_received = []

        def callback(alert):
            alerts_received.append(alert)

        monitor = LatencyMonitor(alert_callback=callback)
        monitor.record(150.0)  # Above critical threshold

        assert len(alerts_received) == 1
        assert alerts_received[0].level == LatencyLevel.CRITICAL
        assert len(alerts_received[0].recommendations) > 0

    def test_record_violation(self):
        """Test recording violation latency generates alert."""
        alerts_received = []

        def callback(alert):
            alerts_received.append(alert)

        monitor = LatencyMonitor(alert_callback=callback)
        monitor.record(600.0)  # Above max threshold

        assert len(alerts_received) == 1
        assert alerts_received[0].level == LatencyLevel.VIOLATION
        assert "IMMEDIATE" in alerts_received[0].recommendations[0]

    def test_get_stats(self):
        """Test statistics calculation."""
        monitor = LatencyMonitor()

        # Record some measurements
        for i in range(100):
            monitor.record(float(i))

        stats = monitor.get_stats()
        assert stats.count == 100
        assert 49.0 <= stats.mean_ms <= 50.0
        assert stats.min_ms == 0.0
        assert stats.max_ms == 99.0
        assert 48.0 <= stats.p50_ms <= 52.0

    def test_get_stats_with_window(self):
        """Test time-windowed statistics."""
        monitor = LatencyMonitor()

        # Record some measurements
        for i in range(10):
            monitor.record(float(i * 10))

        # Get stats for last 1 hour
        stats = monitor.get_stats(window_seconds=3600.0)
        assert stats.count == 10

    def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        monitor = LatencyMonitor()

        # Generate some alerts
        monitor.record(75.0)  # Warning
        monitor.record(150.0)  # Critical

        alerts = monitor.get_recent_alerts(limit=10)
        assert len(alerts) == 2

    def test_get_health_status(self):
        """Test health status reporting."""
        monitor = LatencyMonitor()

        # All normal
        for _ in range(10):
            monitor.record(5.0)

        health = monitor.get_health_status()
        assert health["health"] == "healthy"
        assert health["counters"]["violations"] == 0

    def test_get_health_status_degraded(self):
        """Test degraded health status."""
        monitor = LatencyMonitor()

        # One critical
        monitor.record(150.0)

        health = monitor.get_health_status()
        assert health["health"] == "degraded"

    def test_reset(self):
        """Test resetting the monitor."""
        monitor = LatencyMonitor()

        monitor.record(5.0)
        monitor.record(150.0)  # Critical

        monitor.reset()

        stats = monitor.get_stats()
        assert stats.count == 0

        health = monitor.get_health_status()
        assert health["counters"]["critical"] == 0


class TestInferenceCache:
    """Test InferenceCache class."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = InferenceCache(max_size=100, ttl_seconds=60.0)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 60.0

    def test_get_miss(self):
        """Test cache miss."""
        cache = InferenceCache()
        result = cache.get(np.array([1.0, 2.0, 3.0]))
        assert result is None

    def test_put_and_get(self):
        """Test putting and getting from cache."""
        cache = InferenceCache()

        inputs = np.array([1.0, 2.0, 3.0])
        output = np.array([2.0, 4.0, 6.0])

        cache.put(inputs, output)
        result = cache.get(inputs)

        np.testing.assert_array_equal(result, output)

    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = InferenceCache(max_size=3)

        # Fill cache
        for i in range(5):
            inputs = np.array([float(i)])
            cache.put(inputs, inputs * 2)

        # First two should be evicted
        assert cache.get(np.array([0.0])) is None
        assert cache.get(np.array([1.0])) is None

        # Last three should still be there
        assert cache.get(np.array([2.0])) is not None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = InferenceCache()

        inputs = np.array([1.0])
        cache.put(inputs, np.array([2.0]))

        # Hit
        cache.get(inputs)

        # Miss
        cache.get(np.array([999.0]))

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = InferenceCache()

        cache.put(np.array([1.0]), np.array([2.0]))
        cache.clear()

        assert cache.get(np.array([1.0])) is None
        assert cache.get_stats()["size"] == 0


class TestLatencyTrackedDecorator:
    """Test latency_tracked decorator."""

    def test_decorator_tracks_latency(self):
        """Test that decorator records latency."""
        monitor = LatencyMonitor()

        @latency_tracked(monitor, operation="test_op")
        def slow_function():
            time.sleep(0.01)  # 10ms
            return "result"

        result = slow_function()
        assert result == "result"

        stats = monitor.get_stats()
        assert stats.count == 1
        assert stats.mean_ms >= 10.0


class TestWithLatencyBudgetDecorator:
    """Test with_latency_budget decorator."""

    def test_decorator_enforces_budget(self):
        """Test that decorator enforces budget."""
        budget = LatencyBudget(
            target_ms=1.0,
            warning_ms=5.0,
            critical_ms=10.0,
            max_ms=50.0,
        )

        violations = []

        def on_violation(latency_ms):
            violations.append(latency_ms)

        @with_latency_budget(budget, on_violation=on_violation)
        def slow_function():
            time.sleep(0.02)  # 20ms - critical
            return "result"

        result = slow_function()
        assert result == "result"
        assert len(violations) == 1
        assert violations[0] >= 20.0


class TestPredefinedBudgets:
    """Test predefined latency budgets."""

    def test_robotics_budget(self):
        """Test robotics budget values."""
        assert ROBOTICS_BUDGET.name == "robotics"
        assert ROBOTICS_BUDGET.target_ms == 10.0
        assert ROBOTICS_BUDGET.max_ms == 200.0

    def test_realtime_budget(self):
        """Test realtime budget values."""
        assert REALTIME_BUDGET.name == "realtime"
        assert REALTIME_BUDGET.target_ms == 20.0

    def test_interactive_budget(self):
        """Test interactive budget values."""
        assert INTERACTIVE_BUDGET.name == "interactive"
        assert INTERACTIVE_BUDGET.target_ms == 100.0

    def test_batch_budget(self):
        """Test batch budget values."""
        assert BATCH_BUDGET.name == "batch"
        assert BATCH_BUDGET.target_ms == 1000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
