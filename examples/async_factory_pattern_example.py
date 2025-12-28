#!/usr/bin/env python3
"""
Async Factory Pattern Example

This example demonstrates how to use the async factory pattern
for classes requiring asynchronous initialization in Nethical.

See docs/ASYNC_FACTORY_PATTERN.md for comprehensive documentation.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def example_nats_client():
    """Example: Using NATSClient with async factory pattern."""
    from nethical.streaming.nats_client import NATSClient, NATSConfig
    
    logger.info("=== NATSClient Example ===")
    
    # Configure NATS
    config = NATSConfig(
        servers=["nats://localhost:4222"],
        stream_prefix="example",
    )
    
    # ✅ Recommended: Use the factory method
    client = await NATSClient.create(config)
    logger.info(f"NATSClient created, connected: {client.is_connected}")
    
    # Use the client
    await client.publish("events", {"type": "test", "message": "Hello NATS!"})
    logger.info("Published message to NATS")
    
    # Get metrics
    metrics = client.get_metrics()
    logger.info(f"Client metrics: {metrics}")
    
    # Cleanup
    await client.close()
    logger.info("NATSClient closed\n")


async def example_grpc_client():
    """Example: Using gRPC client with async factory pattern."""
    from nethical.grpc.client import NethicalGRPCClient
    
    logger.info("=== gRPC Client Example ===")
    
    # ✅ Recommended: Use the factory method
    client = await NethicalGRPCClient.create(
        address="localhost:50051",
        timeout_ms=5000
    )
    logger.info("gRPC client created and connected")
    
    # Evaluate an action
    result = await client.evaluate(
        agent_id="example-agent",
        action="Process user data for analytics",
        action_type="data_processing",
        require_explanation=True
    )
    
    logger.info(f"Evaluation result: {result.decision}")
    logger.info(f"Risk score: {result.risk_score}")
    logger.info(f"Violations: {len(result.violations)}")
    if result.explanation:
        logger.info(f"Explanation: {result.explanation.summary}")
    
    # Check health
    health = await client.health_check()
    logger.info(f"Health status: {health['status']}")
    
    # Cleanup
    await client.close()
    logger.info("gRPC client closed\n")


async def example_grpc_with_context_manager():
    """Example: Using gRPC client with async context manager."""
    from nethical.grpc.client import NethicalGRPCClient
    
    logger.info("=== gRPC Client with Context Manager ===")
    
    # ✅ Best practice: Use with async context manager for automatic cleanup
    async with await NethicalGRPCClient.create("localhost:50051") as client:
        logger.info("Client created with context manager")
        
        # Use the client
        result = await client.evaluate(
            agent_id="context-agent",
            action="Read database records",
            action_type="data_access"
        )
        
        logger.info(f"Decision: {result.decision}")
    
    logger.info("Context manager automatically closed client\n")


async def example_satellite_connection():
    """Example: Using Starlink provider with async factory pattern."""
    from nethical.connectivity.satellite.starlink import (
        StarlinkProvider,
        ConnectionConfig
    )
    
    logger.info("=== Starlink Provider Example ===")
    
    # Configure connection
    config = ConnectionConfig(
        provider_options={
            "dish_address": "192.168.100.1",
            "enable_ipv6": True,
        }
    )
    
    try:
        # ✅ Recommended: Use the factory method
        provider = await StarlinkProvider.create(config)
        logger.info(f"Starlink provider created: {provider.provider_name}")
        logger.info(f"Connection state: {provider.state}")
        
        # Check dish status
        if provider.dish_status:
            logger.info(f"Dish status: online={provider.dish_status.is_online}")
            logger.info(f"SNR: {provider.dish_status.snr_db} dB")
        
        # Disconnect
        await provider.disconnect()
        logger.info("Starlink provider disconnected\n")
        
    except Exception as e:
        logger.warning(f"Starlink connection failed (expected without hardware): {e}\n")


async def example_concurrent_clients():
    """Example: Creating multiple clients concurrently."""
    from nethical.grpc.client import NethicalGRPCClient
    
    logger.info("=== Concurrent Client Creation ===")
    
    # Create multiple clients concurrently
    clients = await asyncio.gather(
        NethicalGRPCClient.create("localhost:50051"),
        NethicalGRPCClient.create("localhost:50051"),
        NethicalGRPCClient.create("localhost:50051"),
    )
    
    logger.info(f"Created {len(clients)} clients concurrently")
    
    # Use all clients concurrently
    results = await asyncio.gather(
        clients[0].evaluate(agent_id="agent1", action="action1"),
        clients[1].evaluate(agent_id="agent2", action="action2"),
        clients[2].evaluate(agent_id="agent3", action="action3"),
    )
    
    for i, result in enumerate(results):
        logger.info(f"Client {i+1} result: {result.decision}")
    
    # Cleanup
    for client in clients:
        await client.close()
    
    logger.info("All clients closed\n")


async def example_manual_initialization():
    """Example: Manual initialization (not recommended but supported)."""
    from nethical.streaming.nats_client import NATSClient, NATSConfig
    
    logger.info("=== Manual Initialization Example ===")
    
    # ⚠️ Not recommended: Manual construction
    config = NATSConfig(servers=["nats://localhost:4222"])
    client = NATSClient(config)
    logger.info("Client constructed (not yet initialized)")
    
    # Must call async_setup explicitly
    await client.async_setup()
    logger.info("Client initialized after explicit async_setup()")
    
    metrics = client.get_metrics()
    logger.info(f"Metrics: {metrics}")
    
    await client.close()
    logger.info("Manual initialization example complete\n")


async def example_error_handling():
    """Example: Error handling with async factory pattern."""
    from nethical.grpc.client import NethicalGRPCClient
    
    logger.info("=== Error Handling Example ===")
    
    try:
        # Create client
        client = await NethicalGRPCClient.create()
        
        # Evaluate with potential errors
        result = await client.evaluate(
            agent_id="error-test",
            action="delete all tables",  # Potentially dangerous
            action_type="database_operation"
        )
        
        logger.info(f"Decision: {result.decision}")
        logger.info(f"Risk score: {result.risk_score}")
        
        if result.decision == "BLOCK":
            logger.warning("Action was blocked due to high risk")
            for violation in result.violations:
                logger.warning(f"  - {violation.description} (severity: {violation.severity})")
        
        await client.close()
        
    except Exception as e:
        logger.error(f"Error occurred: {e}")
    
    logger.info("Error handling example complete\n")


async def main():
    """Run all examples."""
    logger.info("Async Factory Pattern Examples\n")
    logger.info("=" * 60 + "\n")
    
    try:
        # Run examples
        await example_nats_client()
        await example_grpc_client()
        await example_grpc_with_context_manager()
        await example_satellite_connection()
        await example_concurrent_clients()
        await example_manual_initialization()
        await example_error_handling()
        
    except KeyboardInterrupt:
        logger.info("\nExamples interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    
    logger.info("=" * 60)
    logger.info("All examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
