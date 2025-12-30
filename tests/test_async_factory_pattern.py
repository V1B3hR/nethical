"""Tests for async factory pattern implementation.

This test suite verifies that classes requiring async initialization
properly implement the async factory pattern as documented in
docs/ASYNC_FACTORY_PATTERN.md
"""

import asyncio
import pytest



class TestAsyncFactoryPattern:
    """Test suite for async factory pattern implementations."""

    @pytest.mark.asyncio
    async def test_nats_client_factory(self):
        """Test NATSClient async factory pattern."""
        from nethical.streaming.nats_client import NATSClient, NATSConfig

        # Test factory method
        config = NATSConfig(servers=["nats://localhost:4222"])
        client = await NATSClient.create(config)

        assert client is not None
        assert isinstance(client, NATSClient)
        assert client.config.servers == ["nats://localhost:4222"]

        # Verify metrics are initialized
        metrics = client.get_metrics()
        assert "connected" in metrics
        assert "messages_published" in metrics

        # Cleanup
        await client.close()

    @pytest.mark.asyncio
    async def test_nats_client_manual_init(self):
        """Test NATSClient manual initialization still works."""
        from nethical.streaming.nats_client import NATSClient, NATSConfig

        # Manual construction without factory
        config = NATSConfig(servers=["nats://localhost:4222"])
        client = NATSClient(config)

        assert client is not None
        assert not client.is_connected  # Not connected yet

        # Must call async_setup explicitly
        await client.async_setup()

        # Now should have attempted connection
        # (may not be connected if NATS server isn't running)
        metrics = client.get_metrics()
        assert "connected" in metrics

        await client.close()

    @pytest.mark.asyncio
    async def test_grpc_client_factory(self):
        """Test NethicalGRPCClient async factory pattern."""
        from nethical.grpc.client import NethicalGRPCClient

        # Test factory method
        client = await NethicalGRPCClient.create(
            address="localhost:50051",
            timeout_ms=5000
        )

        assert client is not None
        assert isinstance(client, NethicalGRPCClient)
        assert client.config.address == "localhost:50051"
        assert client.config.timeout_ms == 5000

        # Should be connected after create()
        assert client._connected is True

        # Test health check
        health = await client.health_check()
        assert health["status"] in ["healthy", "disconnected"]

        await client.close()

    @pytest.mark.asyncio
    async def test_grpc_client_context_manager(self):
        """Test NethicalGRPCClient with async context manager."""
        from nethical.grpc.client import NethicalGRPCClient

        async with await NethicalGRPCClient.create("localhost:50051") as client:
            assert client._connected is True
            health = await client.health_check()
            assert "status" in health

        # Should be closed after context manager exit
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_grpc_client_evaluate(self):
        """Test NethicalGRPCClient evaluation."""
        from nethical.grpc.client import NethicalGRPCClient

        client = await NethicalGRPCClient.create()

        # Test evaluation
        result = await client.evaluate(
            agent_id="test-agent",
            action="process data",
            action_type="data_processing"
        )

        assert result is not None
        assert result.decision in ["ALLOW", "BLOCK", "RESTRICT"]
        assert hasattr(result, "risk_score")
        assert hasattr(result, "decision_id")

        await client.close()

    @pytest.mark.asyncio
    async def test_starlink_provider_factory(self):
        """Test StarlinkProvider async factory pattern."""
        from nethical.connectivity.satellite.starlink import (
            StarlinkProvider,
            ConnectionConfig,
        )

        # Test factory method
        config = ConnectionConfig(
            provider_options={
                "dish_address": "192.168.100.1",
                "enable_ipv6": True,
            }
        )

        # Note: This will fail to connect without actual hardware
        # but we can verify the factory pattern works
        try:
            provider = await StarlinkProvider.create(config)
            assert provider is not None
            assert isinstance(provider, StarlinkProvider)
            assert provider._dish_address == "192.168.100.1"
            assert provider._enable_ipv6 is True
            await provider.disconnect()
        except Exception as e:
            # Expected if no Starlink hardware available
            # Verify the exception is a connection-related error, not a factory method error
            error_str = str(e).lower()
            assert any(keyword in error_str for keyword in ["connect", "reach", "satellite", "dish"]), \
                f"Unexpected error type: {type(e).__name__}: {e}"

    @pytest.mark.asyncio
    async def test_l2_redis_cache_factory(self):
        """Test L2RedisCache async factory pattern."""
        from nethical.cache.l2_redis import L2RedisCache, L2Config

        # Test factory method
        config = L2Config(host="localhost", port=6379)
        
        # Note: This will fail to connect without Redis server
        # but we can verify the factory pattern works
        try:
            cache = await L2RedisCache.create(config)
            assert cache is not None
            assert isinstance(cache, L2RedisCache)
            assert cache.config.host == "localhost"
            assert cache.config.port == 6379
        except ImportError as e:
            # Expected if Redis package not installed
            assert "redis" in str(e).lower()
        except Exception as e:
            # Expected if Redis server isn't running
            # Verify this is a connection error, not a factory error
            error_str = str(e).lower()
            assert any(keyword in error_str for keyword in ["redis", "connect", "connection"]), \
                f"Unexpected error type: {type(e).__name__}: {e}"

    @pytest.mark.asyncio
    async def test_backward_compatibility(self):
        """Test that synchronous classes still work."""
        from nethical.core.models import SafetyViolation

        # Synchronous classes don't need factory pattern
        violation = SafetyViolation(
            severity="HIGH",
            category="safety",
            description="Test violation",
        )

        assert violation is not None
        assert violation.severity == "HIGH"
        assert violation.category == "safety"


class TestAsyncFactoryPatternEdgeCases:
    """Test edge cases and error handling for async factory pattern."""

    @pytest.mark.asyncio
    async def test_multiple_async_setup_calls(self):
        """Test that multiple async_setup calls are handled gracefully."""
        from nethical.streaming.nats_client import NATSClient, NATSConfig

        config = NATSConfig(servers=["nats://localhost:4222"])
        client = NATSClient(config)

        # Call async_setup multiple times
        await client.async_setup()
        metrics1 = client.get_metrics()

        await client.async_setup()
        metrics2 = client.get_metrics()

        # Should be idempotent or at least not crash
        assert metrics1["messages_published"] == metrics2["messages_published"]

        await client.close()

    @pytest.mark.asyncio
    async def test_factory_with_invalid_config(self):
        """Test factory pattern with invalid configuration."""
        from nethical.grpc.client import NethicalGRPCClient

        # Empty address should still create client (may fail on actual use)
        client = await NethicalGRPCClient.create(
            address="",
            timeout_ms=1000
        )

        assert client is not None
        assert client.config.address == ""

        await client.close()

    @pytest.mark.asyncio
    async def test_concurrent_factory_calls(self):
        """Test concurrent factory method calls."""
        from nethical.grpc.client import NethicalGRPCClient

        # Create multiple clients concurrently
        clients = await asyncio.gather(
            NethicalGRPCClient.create("localhost:50051"),
            NethicalGRPCClient.create("localhost:50052"),
            NethicalGRPCClient.create("localhost:50053"),
        )

        assert len(clients) == 3
        for client in clients:
            assert client._connected is True
            await client.close()


class TestAsyncFactoryPatternPerformance:
    """Test performance characteristics of async factory pattern."""

    @pytest.mark.asyncio
    async def test_factory_method_latency(self):
        """Test that factory method doesn't add significant overhead."""
        from nethical.grpc.client import NethicalGRPCClient
        import time

        start = time.perf_counter()
        client = await NethicalGRPCClient.create()
        elapsed = time.perf_counter() - start

        # Factory method should be fast (< 100ms without network)
        assert elapsed < 0.1, f"Factory took {elapsed:.3f}s, expected < 0.1s"

        await client.close()

    @pytest.mark.asyncio
    async def test_batch_client_creation(self):
        """Test creating multiple clients in batch."""
        from nethical.grpc.client import NethicalGRPCClient

        num_clients = 10
        clients = []

        for _ in range(num_clients):
            client = await NethicalGRPCClient.create()
            clients.append(client)

        assert len(clients) == num_clients
        for client in clients:
            assert client._connected is True

        # Cleanup
        for client in clients:
            await client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
