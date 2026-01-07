"""Integration tests for monitoring and alerting system."""

import pytest
import asyncio
from nethical.monitoring import PrometheusMetrics, MetricsServer, get_prometheus_metrics
from nethical.alerting import AlertManager, AlertSeverity, AlertChannel, RateLimiter
from nethical.profiling import FlamegraphProfiler


class TestPrometheusMetrics:
    """Test Prometheus metrics tracking."""
    
    def test_metrics_initialization(self):
        """Test metrics can be initialized."""
        metrics = PrometheusMetrics()
        assert metrics is not None
        # Metrics should be disabled if prometheus_client not installed
        # but should not raise errors
    
    def test_track_request(self):
        """Test tracking requests."""
        metrics = PrometheusMetrics()
        # Should not raise errors even if prometheus not installed
        metrics.track_request("test_detector", 0.1, "success")
    
    def test_track_threat(self):
        """Test tracking threats."""
        metrics = PrometheusMetrics()
        metrics.track_threat("test_detector", "HIGH", "test_category", 0.95)
    
    def test_track_cache(self):
        """Test tracking cache operations."""
        metrics = PrometheusMetrics()
        metrics.track_cache("test_cache", True)
        metrics.track_cache("test_cache", False)
    
    def test_track_error(self):
        """Test tracking errors."""
        metrics = PrometheusMetrics()
        metrics.track_error("test_detector", "timeout")
    
    def test_export_metrics(self):
        """Test metrics export."""
        metrics = PrometheusMetrics()
        output = metrics.export_metrics()
        assert isinstance(output, bytes)


@pytest.mark.asyncio
class TestMetricsServer:
    """Test metrics HTTP server."""
    
    async def test_server_initialization(self):
        """Test server can be initialized."""
        metrics = PrometheusMetrics()
        # This may fail if aiohttp not installed, which is expected
        try:
            server = MetricsServer(metrics, port=19091)  # Use different port for testing
            assert server is not None
        except RuntimeError as e:
            if "aiohttp" in str(e):
                pytest.skip("aiohttp not installed")
            raise


class TestAlertManager:
    """Test alert manager."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        config = {'enabled': True, 'max_alerts_per_minute': 10}
        manager = AlertManager(config)
        assert manager.enabled is True
    
    def test_alert_manager_disabled(self):
        """Test disabled alert manager."""
        config = {'enabled': False}
        manager = AlertManager(config)
        assert manager.enabled is False
    
    @pytest.mark.asyncio
    async def test_send_alert(self):
        """Test sending alerts."""
        config = {'enabled': True}
        manager = AlertManager(config)
        
        # Should not raise errors even without channels configured
        await manager.send_alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.INFO,
            channels=[AlertChannel.SLACK],
            metadata={'test': 'data'}
        )
        
        # Check alert history
        history = manager.get_alert_history()
        assert len(history) == 1
        assert history[0]['title'] == "Test Alert"
    
    def test_clear_history(self):
        """Test clearing alert history."""
        config = {'enabled': True}
        manager = AlertManager(config)
        manager.alert_history = [{'test': 'alert'}]
        
        count = manager.clear_history()
        assert count == 1
        assert len(manager.alert_history) == 0


class TestRateLimiter:
    """Test rate limiter."""
    
    def test_rate_limiter_allows_critical(self):
        """Test critical alerts always allowed."""
        limiter = RateLimiter(max_alerts_per_minute=1)
        
        # Critical alerts should always be allowed
        assert limiter.should_send("test", AlertSeverity.CRITICAL)
        assert limiter.should_send("test", AlertSeverity.CRITICAL)
        assert limiter.should_send("test", AlertSeverity.CRITICAL)
    
    def test_rate_limiter_limits_warnings(self):
        """Test warning alerts are rate limited."""
        limiter = RateLimiter(max_alerts_per_minute=1)
        
        # First warning should be allowed
        assert limiter.should_send("test", AlertSeverity.WARNING)
        
        # Second warning within 60s should be blocked
        assert not limiter.should_send("test", AlertSeverity.WARNING)
    
    def test_rate_limiter_different_keys(self):
        """Test different alert keys are tracked separately."""
        limiter = RateLimiter(max_alerts_per_minute=1)
        
        assert limiter.should_send("test1", AlertSeverity.WARNING)
        assert limiter.should_send("test2", AlertSeverity.WARNING)


class TestFlamegraphProfiler:
    """Test flamegraph profiler."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = FlamegraphProfiler(output_dir="/tmp/test_profiling")
        assert profiler.output_dir.exists()
    
    def test_profile_sync(self):
        """Test synchronous profiling."""
        profiler = FlamegraphProfiler(output_dir="/tmp/test_profiling")
        
        def test_function():
            total = 0
            for i in range(1000):
                total += i
            return total
        
        result, report_path = profiler.profile_sync(test_function)
        assert result == sum(range(1000))
        assert report_path.exists()


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for full monitoring system."""
    
    async def test_end_to_end_monitoring(self):
        """Test complete monitoring workflow."""
        # Initialize metrics
        metrics = get_prometheus_metrics()
        
        # Track some activity
        metrics.track_request("test_detector", 0.05, "success")
        metrics.track_threat("test_detector", "HIGH", "test", 0.9)
        metrics.track_cache("test_cache", True)
        
        # Export metrics
        output = metrics.export_metrics()
        assert isinstance(output, bytes)
    
    async def test_alerting_workflow(self):
        """Test complete alerting workflow."""
        # Setup alert manager
        config = {'enabled': True, 'max_alerts_per_minute': 10}
        manager = AlertManager(config)
        
        # Send test alert
        await manager.send_alert(
            title="Integration Test",
            message="Testing alerting",
            severity=AlertSeverity.INFO,
            channels=[AlertChannel.SLACK]
        )
        
        # Verify history
        history = manager.get_alert_history()
        assert len(history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
