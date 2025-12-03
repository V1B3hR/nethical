"""Tests for satellite base classes."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from nethical.connectivity.satellite.base import (
    SatelliteProvider,
    ConnectionState,
    ConnectionConfig,
    ConnectionMetrics,
    SatelliteConnectionError,
    SatelliteTimeoutError,
)


class MockSatelliteProvider(SatelliteProvider):
    """Mock implementation for testing."""

    @property
    def provider_name(self) -> str:
        return "MockProvider"

    @property
    def provider_type(self) -> str:
        return "TEST"

    async def connect(self) -> bool:
        self.state = ConnectionState.CONNECTED
        self._connection_start = datetime.utcnow()
        return True

    async def disconnect(self) -> bool:
        self.state = ConnectionState.DISCONNECTED
        return True

    async def send(self, data: bytes, priority: int = 0) -> bool:
        self._metrics.bytes_sent += len(data)
        return True

    async def receive(self, timeout=None) -> bytes:
        return b"test data"

    async def health_check(self) -> bool:
        return True

    async def get_signal_info(self):
        return {"signal": -70}


class TestConnectionConfig:
    """Tests for ConnectionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConnectionConfig()
        assert config.timeout_seconds == 30.0
        assert config.retry_attempts == 3
        assert config.compression_enabled is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ConnectionConfig(
            endpoint="192.168.1.1",
            timeout_seconds=60.0,
            retry_attempts=5,
        )
        assert config.endpoint == "192.168.1.1"
        assert config.timeout_seconds == 60.0
        assert config.retry_attempts == 5


class TestConnectionMetrics:
    """Tests for ConnectionMetrics."""

    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = ConnectionMetrics()
        assert metrics.latency_ms == 0.0
        assert metrics.packet_loss_percent == 0.0
        assert metrics.bytes_sent == 0

    def test_metrics_update(self):
        """Test metrics can be updated."""
        metrics = ConnectionMetrics()
        metrics.latency_ms = 25.0
        metrics.bytes_sent = 1024
        assert metrics.latency_ms == 25.0
        assert metrics.bytes_sent == 1024


class TestSatelliteConnectionError:
    """Tests for exception classes."""

    def test_error_attributes(self):
        """Test error has correct attributes."""
        error = SatelliteConnectionError(
            "Connection failed",
            "TestProvider",
            {"reason": "timeout"},
        )
        assert str(error) == "Connection failed"
        assert error.provider == "TestProvider"
        assert error.details == {"reason": "timeout"}
        assert error.timestamp is not None

    def test_timeout_error(self):
        """Test timeout error is subclass."""
        error = SatelliteTimeoutError("Timeout", "TestProvider")
        assert isinstance(error, SatelliteConnectionError)


class TestSatelliteProvider:
    """Tests for SatelliteProvider base class."""

    @pytest.fixture
    def provider(self):
        """Create mock provider."""
        return MockSatelliteProvider()

    def test_initial_state(self, provider):
        """Test initial provider state."""
        assert provider.state == ConnectionState.DISCONNECTED
        assert provider.is_connected is False
        assert provider.provider_name == "MockProvider"
        assert provider.provider_type == "TEST"

    @pytest.mark.asyncio
    async def test_connect(self, provider):
        """Test connection."""
        result = await provider.connect()
        assert result is True
        assert provider.state == ConnectionState.CONNECTED
        assert provider.is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect(self, provider):
        """Test disconnection."""
        await provider.connect()
        result = await provider.disconnect()
        assert result is True
        assert provider.state == ConnectionState.DISCONNECTED
        assert provider.is_connected is False

    @pytest.mark.asyncio
    async def test_send(self, provider):
        """Test sending data."""
        await provider.connect()
        result = await provider.send(b"test data")
        assert result is True
        assert provider.metrics.bytes_sent == 9

    @pytest.mark.asyncio
    async def test_receive(self, provider):
        """Test receiving data."""
        await provider.connect()
        data = await provider.receive()
        assert data == b"test data"

    @pytest.mark.asyncio
    async def test_health_check(self, provider):
        """Test health check."""
        result = await provider.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_get_signal_info(self, provider):
        """Test signal info."""
        info = await provider.get_signal_info()
        assert "signal" in info

    def test_callback_registration(self, provider):
        """Test callback registration."""
        callback = MagicMock()
        provider.register_callback("on_connect", callback)
        assert callback in provider._callbacks["on_connect"]

    def test_callback_unregistration(self, provider):
        """Test callback unregistration."""
        callback = MagicMock()
        provider.register_callback("on_connect", callback)
        provider.unregister_callback("on_connect", callback)
        assert callback not in provider._callbacks["on_connect"]

    def test_state_change_triggers_callback(self, provider):
        """Test state change triggers callbacks."""
        callback = MagicMock()
        provider.register_callback("on_state_change", callback)

        provider.state = ConnectionState.CONNECTED
        callback.assert_called_once()

    def test_update_metrics(self, provider):
        """Test metrics update."""
        provider._update_metrics(
            latency_ms=30.0,
            jitter_ms=5.0,
            signal_dbm=-65.0,
        )
        assert provider.metrics.latency_ms == 30.0
        assert provider.metrics.jitter_ms == 5.0
        assert provider.metrics.signal_strength_dbm == -65.0

    def test_metrics_summary(self, provider):
        """Test get_metrics_summary."""
        provider._update_metrics(latency_ms=25.0)
        summary = provider.get_metrics_summary()

        assert summary["provider"] == "MockProvider"
        assert summary["type"] == "TEST"
        assert summary["latency_ms"] == 25.0

    def test_repr(self, provider):
        """Test string representation."""
        result = repr(provider)
        assert "MockSatelliteProvider" in result
        assert "MockProvider" in result
