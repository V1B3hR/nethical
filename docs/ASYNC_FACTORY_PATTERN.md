# Async Factory Pattern Guide

## Overview

The Async Factory Pattern is used in Nethical for classes that require asynchronous initialization, particularly those involving:
- Network connections (gRPC, NATS, Redis, etc.)
- External API calls
- Database connections
- File I/O operations
- Resource-intensive setup

This pattern ensures that asynchronous setup is properly handled and makes the initialization contract explicit.

## The Pattern

### Basic Structure

```python
class AsyncResource:
    """Example class requiring async initialization."""
    
    def __init__(self, param: str):
        """
        Synchronous constructor - only sets basic attributes.
        
        Args:
            param: Configuration parameter
        """
        self.param = param
        self._connected = False
        self._resource = None
    
    async def async_setup(self) -> None:
        """
        Perform asynchronous initialization.
        
        This method handles all async operations like:
        - Network connections
        - API calls
        - Resource allocation
        
        Raises:
            ConnectionError: If setup fails
        """
        # Perform async initialization
        self._resource = await self._establish_connection()
        self._connected = True
    
    @classmethod
    async def create(cls, param: str) -> "AsyncResource":
        """
        Async factory method for creating instances.
        
        This is the recommended way to instantiate this class.
        
        Args:
            param: Configuration parameter
            
        Returns:
            Fully initialized instance
            
        Raises:
            ConnectionError: If initialization fails
            
        Example:
            >>> resource = await AsyncResource.create("config")
            >>> # resource is ready to use
        """
        obj = cls(param)
        await obj.async_setup()
        return obj
    
    async def _establish_connection(self):
        """Internal async helper."""
        # Simulate async operation
        import asyncio
        await asyncio.sleep(0.1)
        return "connected"
```

### Usage

```python
import asyncio

async def main():
    # Recommended: Use the factory method
    resource = await AsyncResource.create("my-config")
    
    # Use the resource
    print(resource._connected)  # True
    
asyncio.run(main())
```

## When to Use This Pattern

✅ **Use the async factory pattern when:**
- Establishing network connections (TCP, HTTP, gRPC, WebSocket)
- Connecting to databases or caches (Redis, PostgreSQL, etc.)
- Initializing message queues (NATS, Kafka, RabbitMQ)
- Loading large files or datasets asynchronously
- Performing health checks or validation during initialization
- Any I/O-bound initialization that would block

❌ **Don't use this pattern when:**
- Only setting attributes from constructor parameters
- All initialization is CPU-bound and fast (<1ms)
- The class doesn't interact with external resources

## Real-World Examples in Nethical

### Example 1: NATS Client

```python
from nethical.streaming.nats_client import NATSClient, NATSConfig

async def example_nats():
    # Create configuration
    config = NATSConfig(
        servers=["nats://localhost:4222"],
        stream_prefix="myapp"
    )
    
    # Use factory method to create and connect
    client = await NATSClient.create(config)
    
    # Client is ready to use
    await client.publish("events", {"type": "test"})
    
    # Cleanup
    await client.close()
```

### Example 2: gRPC Client

```python
from nethical.grpc.client import NethicalGRPCClient

async def example_grpc():
    # Use factory method
    client = await NethicalGRPCClient.create(
        address="localhost:50051",
        timeout_ms=5000
    )
    
    # Client is connected and ready
    result = await client.evaluate(
        agent_id="my-agent",
        action="process_data",
        action_type="data_processing"
    )
    
    # Cleanup
    await client.close()
```

### Example 3: Satellite Connectivity

```python
from nethical.connectivity.satellite.starlink import StarlinkProvider, ConnectionConfig

async def example_satellite():
    # Configure connection
    config = ConnectionConfig(
        provider_options={
            "dish_address": "192.168.100.1",
            "enable_ipv6": True
        }
    )
    
    # Use factory method
    provider = await StarlinkProvider.create(config)
    
    # Provider is connected
    await provider.send("Hello from space!")
    
    # Cleanup
    await provider.disconnect()
```

## Context Manager Pattern

For resources that need cleanup, combine the async factory pattern with async context managers:

```python
class ManagedAsyncResource:
    """Resource with automatic cleanup."""
    
    def __init__(self, param: str):
        self.param = param
        self._connected = False
    
    async def async_setup(self) -> None:
        """Initialize the resource."""
        self._connected = True
    
    @classmethod
    async def create(cls, param: str) -> "ManagedAsyncResource":
        """Factory method."""
        obj = cls(param)
        await obj.async_setup()
        return obj
    
    async def close(self) -> None:
        """Cleanup resources."""
        self._connected = False
    
    async def __aenter__(self) -> "ManagedAsyncResource":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

# Usage with context manager
async def use_managed_resource():
    async with await ManagedAsyncResource.create("config") as resource:
        # Use resource
        pass
    # Automatically cleaned up
```

## Backward Compatibility

Classes that don't require async initialization can be instantiated normally:

```python
from nethical.core.models import SafetyViolation

# Synchronous classes work as before
violation = SafetyViolation(
    severity="HIGH",
    category="safety",
    description="Detected unsafe action"
)
```

## Migration Guide

### Before (Anti-pattern)

```python
class OldClient:
    def __init__(self, address: str):
        self.address = address
        # ❌ BAD: Blocking I/O in constructor
        self._connection = self._connect_sync()
    
    def _connect_sync(self):
        # Blocks the event loop!
        import time
        time.sleep(1)
        return "connected"
```

### After (Correct pattern)

```python
class NewClient:
    def __init__(self, address: str):
        self.address = address
        self._connection = None
    
    async def async_setup(self) -> None:
        """Async initialization."""
        self._connection = await self._connect_async()
    
    @classmethod
    async def create(cls, address: str) -> "NewClient":
        """Factory method."""
        obj = cls(address)
        await obj.async_setup()
        return obj
    
    async def _connect_async(self):
        import asyncio
        await asyncio.sleep(1)
        return "connected"
```

## Testing Async Factory Pattern

```python
import pytest

class TestAsyncResource:
    @pytest.mark.asyncio
    async def test_create_factory(self):
        """Test the factory method."""
        resource = await AsyncResource.create("test-param")
        
        assert resource.param == "test-param"
        assert resource._connected is True
    
    @pytest.mark.asyncio
    async def test_manual_initialization_fails(self):
        """Test that manual init without async_setup fails."""
        resource = AsyncResource("test-param")
        
        # Not yet initialized
        assert resource._connected is False
        
        # Must call async_setup
        await resource.async_setup()
        assert resource._connected is True
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with await ManagedAsyncResource.create("test") as resource:
            assert resource._connected is True
        
        # Verify cleanup happened
        assert resource._connected is False
```

## Best Practices

1. **Clear Documentation**: Always document that a class uses the async factory pattern
2. **Fail Fast**: Raise clear exceptions if initialization fails
3. **Idempotency**: Make `async_setup()` idempotent when possible
4. **Type Hints**: Use proper type hints for better IDE support
5. **Logging**: Log initialization steps for debugging
6. **Timeouts**: Set reasonable timeouts for async operations
7. **Resource Cleanup**: Implement cleanup methods and context managers
8. **Testing**: Write tests for both factory method and manual initialization

## Common Pitfalls

### Pitfall 1: Forgetting to await

```python
# ❌ Wrong
client = NATSClient.create(config)  # Returns a coroutine, not a client!

# ✅ Correct
client = await NATSClient.create(config)
```

### Pitfall 2: Mixing sync and async

```python
# ❌ Wrong
class BadClient:
    def __init__(self):
        asyncio.run(self._async_init())  # Creates new event loop!
    
    async def _async_init(self):
        pass

# ✅ Correct
class GoodClient:
    @classmethod
    async def create(cls):
        obj = cls()
        await obj._async_init()
        return obj
```

### Pitfall 3: Not handling initialization errors

```python
# ❌ Wrong
@classmethod
async def create(cls, param):
    obj = cls(param)
    await obj.async_setup()  # No error handling
    return obj

# ✅ Correct
@classmethod
async def create(cls, param):
    obj = cls(param)
    try:
        await obj.async_setup()
    except Exception as e:
        # Log and re-raise with context
        logger.error(f"Failed to initialize {cls.__name__}: {e}")
        raise
    return obj
```

## Summary

The Async Factory Pattern provides a clean, explicit way to handle asynchronous initialization in Python classes. It:

- Makes async requirements clear in the API
- Prevents blocking operations in constructors
- Ensures proper initialization before use
- Works well with modern Python async/await syntax
- Maintains backward compatibility for sync classes

Use this pattern consistently across Nethical for all classes requiring async initialization.
