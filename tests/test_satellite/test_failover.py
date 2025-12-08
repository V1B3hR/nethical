"""Tests for failover manager."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from nethical.connectivity.satellite.failover import (
    FailoverManager,
    FailoverConfig,
    FailoverEvent,
    FailoverReason,
    ConnectionType,
)
from nethical.connectivity.satellite.base import (
    ConnectionState,
    ConnectionConfig,
    ConnectionMetrics,
)


class MockProvider:
    """Mock satellite provider for testing."""

    def __init__(self, healthy=True):
        self._healthy = healthy
        self._state = (
            ConnectionState.CONNECTED if healthy else ConnectionState.DISCONNECTED
        )
        self._metrics = ConnectionMetrics()

    @property
    def state(self):
        return self._state

    @property
    def is_connected(self):
        return self._state == ConnectionState.CONNECTED

    @property
    def metrics(self):
        return self._metrics

    async def connect(self):
        self._state = ConnectionState.CONNECTED
        return True

    async def disconnect(self):
        self._state = ConnectionState.DISCONNECTED
        return True

    async def health_check(self):
        return self._healthy

    async def get_signal_info(self):
        return {"signal_strength_dbm": -65.0, "latency_ms": 30.0}


class TestFailoverConfig:
    """Tests for FailoverConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = FailoverConfig()
        assert config.latency_threshold_ms == 500.0
        assert config.packet_loss_threshold_percent == 5.0
        assert config.consecutive_failures_for_failover == 3
        assert config.auto_failback is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = FailoverConfig(
            latency_threshold_ms=200.0,
            packet_loss_threshold_percent=2.0,
            consecutive_failures_for_failover=5,
            auto_failback=False,
        )
        assert config.latency_threshold_ms == 200.0
        assert config.packet_loss_threshold_percent == 2.0
        assert config.consecutive_failures_for_failover == 5
        assert config.auto_failback is False


class TestConnectionType:
    """Tests for ConnectionType enum."""

    def test_connection_types(self):
        """Test connection type values."""
        assert ConnectionType.TERRESTRIAL.value == "terrestrial"
        assert ConnectionType.SATELLITE.value == "satellite"


class TestFailoverReason:
    """Tests for FailoverReason enum."""

    def test_failover_reasons(self):
        """Test failover reason values."""
        assert FailoverReason.LATENCY_THRESHOLD.value == "latency_threshold"
        assert FailoverReason.PACKET_LOSS.value == "packet_loss"
        assert FailoverReason.CONNECTION_LOST.value == "connection_lost"
        assert FailoverReason.MANUAL.value == "manual"


class TestFailoverEvent:
    """Tests for FailoverEvent."""

    def test_event_creation(self):
        """Test failover event creation."""
        event = FailoverEvent(
            timestamp=datetime.utcnow(),
            from_connection=ConnectionType.TERRESTRIAL,
            to_connection=ConnectionType.SATELLITE,
            reason=FailoverReason.LATENCY_THRESHOLD,
            duration_ms=150.0,
        )

        assert event.from_connection == ConnectionType.TERRESTRIAL
        assert event.to_connection == ConnectionType.SATELLITE
        assert event.reason == FailoverReason.LATENCY_THRESHOLD
        assert event.duration_ms == 150.0
        assert event.timestamp is not None


class TestFailoverManager:
    """Tests for FailoverManager."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return FailoverConfig(
            latency_threshold_ms=500.0,
            packet_loss_threshold_percent=5.0,
            health_check_interval_seconds=5.0,
            consecutive_failures_for_failover=3,
        )

    @pytest.fixture
    def satellite_provider(self):
        """Create mock satellite provider."""
        return MockProvider(healthy=True)

    @pytest.fixture
    def failover(self, config, satellite_provider):
        """Create failover manager."""
        return FailoverManager(config, satellite_provider)

    def test_initial_state(self, failover):
        """Test initial failover state."""
        assert failover.active_connection == ConnectionType.TERRESTRIAL
        assert failover._is_monitoring is False
        assert failover._terrestrial_healthy is True

    @pytest.mark.asyncio
    async def test_start_monitoring(self, failover):
        """Test starting monitoring."""
        await failover.start_monitoring()
        assert failover._is_monitoring is True
        await failover.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, failover):
        """Test stopping monitoring."""
        await failover.start_monitoring()
        await failover.stop_monitoring()
        assert failover._is_monitoring is False

    def test_get_status(self, failover):
        """Test getting failover status."""
        status = failover.get_status()

        assert "active_connection" in status
        assert "terrestrial_healthy" in status
        assert "satellite_healthy" in status
        assert "is_monitoring" in status

    def test_callback_registration(self, failover):
        """Test callback registration."""
        callback = MagicMock()
        failover.register_callback("on_failover", callback)
        assert callback in failover._callbacks["on_failover"]

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, failover):
        """Test graceful degradation behavior."""
        # Simulate both connections being unhealthy
        failover._terrestrial_healthy = False
        failover._satellite_healthy = False

        # System should still operate
        status = failover.get_status()
        assert status is not None
