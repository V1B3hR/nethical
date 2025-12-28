"""
Abstract Base Class for Satellite Providers

Defines the interface that all satellite connectivity providers must implement.
Provides common functionality for connection management, metrics collection,
and state tracking.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    DEGRADED = "degraded"


class SatelliteConnectionError(Exception):
    """Base exception for satellite connection errors."""

    def __init__(self, message: str, provider: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.provider = provider
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class SatelliteTimeoutError(SatelliteConnectionError):
    """Exception raised when satellite connection times out."""

    pass


@dataclass
class ConnectionConfig:
    """Configuration for satellite connections."""

    # Connection settings
    endpoint: str = ""
    api_key: str = ""
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 5.0

    # Performance settings
    max_latency_ms: float = 500.0
    min_bandwidth_kbps: float = 100.0
    compression_enabled: bool = True

    # Failover settings
    auto_reconnect: bool = True
    failover_threshold_ms: float = 1000.0
    health_check_interval_seconds: float = 30.0

    # Provider-specific settings
    provider_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionMetrics:
    """Metrics for a satellite connection."""

    latency_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_percent: float = 0.0
    bandwidth_kbps: float = 0.0
    signal_strength_dbm: float = -100.0
    uptime_seconds: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors_count: int = 0
    last_update: Optional[datetime] = None


class SatelliteProvider(ABC):
    """
    Abstract base class for satellite connectivity providers with async factory pattern.

    All satellite providers (Starlink, Kuiper, OneWeb, Iridium) must
    implement this interface to ensure consistent behavior across
    different satellite systems.
    
    Note:
        Subclasses should use the async factory pattern by implementing their own
        `create()` class method or using the base implementation. See
        docs/ASYNC_FACTORY_PATTERN.md for details.
    """

    def __init__(self, config: Optional[ConnectionConfig] = None):
        """
        Initialize satellite provider (synchronous constructor).
        
        Note:
            This only sets up basic attributes. Subclasses should provide an
            async factory method `create()` for proper initialization with connection.

        Args:
            config: Connection configuration
        """
        self.config = config or ConnectionConfig()
        self._state = ConnectionState.DISCONNECTED
        self._metrics = ConnectionMetrics()
        self._callbacks: Dict[str, List[Callable]] = {
            "on_connect": [],
            "on_disconnect": [],
            "on_error": [],
            "on_state_change": [],
        }
        self._connection_start: Optional[datetime] = None

    async def async_setup(self) -> None:
        """
        Perform asynchronous initialization.
        
        This method establishes the connection. Subclasses can override
        this method to add additional setup logic.
        
        Raises:
            SatelliteConnectionError: If connection fails
        """
        await self.connect()

    @classmethod
    async def create(cls, config: Optional[ConnectionConfig] = None) -> "SatelliteProvider":
        """
        Async factory method for creating a connected satellite provider.
        
        This is the recommended way to instantiate satellite providers as it ensures
        the connection is established before the instance is returned.
        
        Subclasses can override this method to customize the creation process.
        
        Args:
            config: Connection configuration
            
        Returns:
            Fully initialized and connected provider instance
            
        Raises:
            SatelliteConnectionError: If connection fails
            
        Example:
            >>> from nethical.connectivity.satellite.starlink import StarlinkProvider
            >>> provider = await StarlinkProvider.create(config)
        """
        obj = cls(config)
        await obj.async_setup()
        return obj

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Return the provider type (LEO, MEO, GEO, etc.)."""
        pass

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @state.setter
    def state(self, new_state: ConnectionState):
        """Set connection state and trigger callbacks."""
        old_state = self._state
        self._state = new_state
        if old_state != new_state:
            self._trigger_callbacks("on_state_change", old_state, new_state)

    @property
    def metrics(self) -> ConnectionMetrics:
        """Get current connection metrics."""
        if self._connection_start:
            self._metrics.uptime_seconds = (
                datetime.utcnow() - self._connection_start
            ).total_seconds()
        return self._metrics

    @property
    def is_connected(self) -> bool:
        """Check if provider is connected."""
        return self._state == ConnectionState.CONNECTED

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to satellite network.

        Returns:
            True if connection successful, False otherwise

        Raises:
            SatelliteConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from satellite network.

        Returns:
            True if disconnection successful
        """
        pass

    @abstractmethod
    async def send(self, data: bytes, priority: int = 0) -> bool:
        """
        Send data over satellite connection.

        Args:
            data: Data to send
            priority: Message priority (0=normal, 1=high, 2=urgent)

        Returns:
            True if send successful

        Raises:
            SatelliteConnectionError: If send fails
        """
        pass

    @abstractmethod
    async def receive(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Receive data from satellite connection.

        Args:
            timeout: Receive timeout in seconds

        Returns:
            Received data or None if timeout

        Raises:
            SatelliteConnectionError: If receive fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check connection health.

        Returns:
            True if connection is healthy
        """
        pass

    @abstractmethod
    async def get_signal_info(self) -> Dict[str, Any]:
        """
        Get current signal information.

        Returns:
            Dictionary with signal details
        """
        pass

    def register_callback(self, event: str, callback: Callable):
        """
        Register a callback for an event.

        Args:
            event: Event name (on_connect, on_disconnect, on_error, on_state_change)
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown event type: {event}")

    def unregister_callback(self, event: str, callback: Callable):
        """
        Unregister a callback.

        Args:
            event: Event name
            callback: Callback function to remove
        """
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """
        Trigger all callbacks for an event.

        Args:
            event: Event name
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks
        """
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {event} callback: {e}")

    def _update_metrics(
        self,
        latency_ms: Optional[float] = None,
        jitter_ms: Optional[float] = None,
        packet_loss: Optional[float] = None,
        bandwidth_kbps: Optional[float] = None,
        signal_dbm: Optional[float] = None,
    ):
        """
        Update connection metrics.

        Args:
            latency_ms: Current latency in milliseconds
            jitter_ms: Current jitter in milliseconds
            packet_loss: Packet loss percentage
            bandwidth_kbps: Bandwidth in kbps
            signal_dbm: Signal strength in dBm
        """
        if latency_ms is not None:
            self._metrics.latency_ms = latency_ms
        if jitter_ms is not None:
            self._metrics.jitter_ms = jitter_ms
        if packet_loss is not None:
            self._metrics.packet_loss_percent = packet_loss
        if bandwidth_kbps is not None:
            self._metrics.bandwidth_kbps = bandwidth_kbps
        if signal_dbm is not None:
            self._metrics.signal_strength_dbm = signal_dbm
        self._metrics.last_update = datetime.utcnow()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of connection metrics.

        Returns:
            Dictionary with metrics summary
        """
        return {
            "provider": self.provider_name,
            "type": self.provider_type,
            "state": self._state.value,
            "is_connected": self.is_connected,
            "latency_ms": self._metrics.latency_ms,
            "jitter_ms": self._metrics.jitter_ms,
            "packet_loss_percent": self._metrics.packet_loss_percent,
            "bandwidth_kbps": self._metrics.bandwidth_kbps,
            "signal_strength_dbm": self._metrics.signal_strength_dbm,
            "uptime_seconds": self.metrics.uptime_seconds,
            "bytes_sent": self._metrics.bytes_sent,
            "bytes_received": self._metrics.bytes_received,
            "errors_count": self._metrics.errors_count,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name}, state={self._state.value})"
