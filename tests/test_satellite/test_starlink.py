"""Tests for Starlink satellite provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from nethical.connectivity.satellite.starlink import (
    StarlinkProvider,
    StarlinkDishStatus,
)
from nethical.connectivity.satellite.base import (
    ConnectionState,
    ConnectionConfig,
)


class TestDishStatus:
    """Tests for StarlinkDishStatus dataclass."""

    def test_default_values(self):
        """Test default dish status."""
        status = StarlinkDishStatus()
        assert status.is_online is False
        assert status.pop_ping_latency_ms == 0.0
        assert status.obstruction_percent == 0.0
        assert status.device_id == ""

    def test_custom_values(self):
        """Test custom dish status."""
        status = StarlinkDishStatus(
            is_online=True,
            pop_ping_latency_ms=25.5,
            obstruction_percent=5.0,
            device_id="dish-123",
        )
        assert status.is_online is True
        assert status.pop_ping_latency_ms == 25.5
        assert status.obstruction_percent == 5.0
        assert status.device_id == "dish-123"


class TestStarlinkProvider:
    """Tests for StarlinkProvider."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ConnectionConfig(
            endpoint="192.168.100.1",
            timeout_seconds=30.0,
            provider_options={
                "dish_address": "192.168.100.1",
                "grpc_port": 9200,
                "enable_ipv6": True,
            },
        )

    @pytest.fixture
    def provider(self, config):
        """Create Starlink provider (using direct constructor for testing)."""
        return StarlinkProvider(config)

    @pytest.mark.asyncio
    async def test_factory_pattern(self, config):
        """Test async factory pattern."""
        # Test the recommended factory method
        provider = await StarlinkProvider.create(config)
        assert provider is not None
        assert isinstance(provider, StarlinkProvider)
        assert provider.state == ConnectionState.CONNECTED
        await provider.disconnect()

    def test_provider_properties(self, provider):
        """Test provider properties."""
        assert provider.provider_name == "Starlink"
        assert provider.provider_type == "LEO"
        assert provider._dish_address == "192.168.100.1"
        assert provider._grpc_port == 9200

    def test_initial_state(self, provider):
        """Test initial state."""
        assert provider.state == ConnectionState.DISCONNECTED
        assert provider._dish_status is None

    @pytest.mark.asyncio
    async def test_connect(self, provider):
        """Test connection."""
        result = await provider.connect()
        assert result is True
        assert provider.state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_disconnect(self, provider):
        """Test disconnection."""
        await provider.connect()
        result = await provider.disconnect()
        assert result is True
        assert provider.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_send(self, provider):
        """Test sending data."""
        await provider.connect()
        result = await provider.send(b"test payload")
        assert result is True
        assert provider.metrics.bytes_sent >= 12

    @pytest.mark.asyncio
    async def test_health_check(self, provider):
        """Test health check."""
        await provider.connect()
        result = await provider.health_check()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_get_signal_info(self, provider):
        """Test get signal info."""
        await provider.connect()
        info = await provider.get_signal_info()
        assert isinstance(info, dict)

    def test_default_dish_address(self):
        """Test default dish address."""
        provider = StarlinkProvider()
        assert provider._dish_address == "192.168.100.1"

    def test_default_grpc_port(self):
        """Test default gRPC port."""
        provider = StarlinkProvider()
        assert provider._grpc_port == 9200

    def test_ipv6_enabled(self, provider):
        """Test IPv6 enabled check."""
        assert provider._enable_ipv6 is True

    def test_metrics_summary(self, provider):
        """Test metrics summary."""
        provider._update_metrics(
            latency_ms=28.0,
            jitter_ms=3.0,
            signal_dbm=-62.0,
        )

        summary = provider.get_metrics_summary()
        assert summary["provider"] == "Starlink"
        assert summary["type"] == "LEO"
        assert summary["latency_ms"] == 28.0
