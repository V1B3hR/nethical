"""
SpaceX Starlink LEO Satellite Integration

Provides integration with Starlink's LEO satellite constellation,
including support for:
- 20-40ms typical latency with spikes to 100ms+
- IPv6 native support
- Dishy McFlatface API integration for local dish status
- Obstruction detection awareness
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .base import (
    ConnectionConfig,
    ConnectionMetrics,
    ConnectionState,
    SatelliteConnectionError,
    SatelliteProvider,
    SatelliteTimeoutError,
)

logger = logging.getLogger(__name__)


@dataclass
class StarlinkDishStatus:
    """Starlink dish (Dishy McFlatface) status information."""

    device_id: str = ""
    hardware_version: str = ""
    software_version: str = ""
    uptime_seconds: float = 0.0
    is_online: bool = False

    # Signal metrics
    snr_db: float = 0.0
    downlink_throughput_bps: float = 0.0
    uplink_throughput_bps: float = 0.0
    pop_ping_latency_ms: float = 0.0

    # Obstruction info
    obstruction_percent: float = 0.0
    obstruction_duration_seconds: float = 0.0
    is_obstructed: bool = False

    # Orientation
    tilt_angle_deg: float = 0.0
    boresight_azimuth_deg: float = 0.0
    boresight_elevation_deg: float = 0.0

    # Alerts
    alerts: list = None

    def __post_init__(self):
        if self.alerts is None:
            self.alerts = []


class StarlinkProvider(SatelliteProvider):
    """
    SpaceX Starlink LEO satellite integration.

    Starlink provides low-latency broadband internet via a constellation
    of LEO satellites. Typical latency is 20-40ms with occasional spikes
    to 100ms+ during satellite handoffs or obstructions.

    Features:
    - Native IPv6 support
    - Local dish API integration
    - Obstruction detection and awareness
    - Adaptive latency handling for variable conditions
    """

    # Starlink typical performance ranges
    TYPICAL_LATENCY_MS_MIN = 20.0
    TYPICAL_LATENCY_MS_MAX = 40.0
    SPIKE_LATENCY_MS = 100.0
    TYPICAL_BANDWIDTH_MBPS = 100.0

    def __init__(self, config: Optional[ConnectionConfig] = None):
        """
        Initialize Starlink provider.

        Args:
            config: Connection configuration with Starlink-specific options
        """
        super().__init__(config)

        # Starlink-specific configuration
        self._dish_address = self.config.provider_options.get(
            "dish_address", "192.168.100.1"
        )
        self._grpc_port = self.config.provider_options.get("grpc_port", 9200)
        self._enable_ipv6 = self.config.provider_options.get("enable_ipv6", True)

        # Dish status
        self._dish_status: Optional[StarlinkDishStatus] = None
        self._last_dish_poll: Optional[datetime] = None

        # Connection tracking
        self._send_queue: asyncio.Queue = asyncio.Queue()
        self._receive_queue: asyncio.Queue = asyncio.Queue()

    @property
    def provider_name(self) -> str:
        return "Starlink"

    @property
    def provider_type(self) -> str:
        return "LEO"

    @property
    def dish_status(self) -> Optional[StarlinkDishStatus]:
        """Get current dish status."""
        return self._dish_status

    async def connect(self) -> bool:
        """
        Establish connection to Starlink network.

        Returns:
            True if connection successful

        Raises:
            SatelliteConnectionError: If connection fails
        """
        try:
            self.state = ConnectionState.CONNECTING
            logger.info(f"Connecting to Starlink via dish at {self._dish_address}")

            # Check dish connectivity
            dish_reachable = await self._check_dish_connectivity()
            if not dish_reachable:
                raise SatelliteConnectionError(
                    "Cannot reach Starlink dish",
                    self.provider_name,
                    {"dish_address": self._dish_address},
                )

            # Get initial dish status
            await self._poll_dish_status()

            # Check if dish is online
            if self._dish_status and not self._dish_status.is_online:
                raise SatelliteConnectionError(
                    "Starlink dish is offline",
                    self.provider_name,
                    {"status": "offline"},
                )

            # Test internet connectivity
            internet_ok = await self._test_internet_connectivity()
            if not internet_ok:
                self.state = ConnectionState.DEGRADED
                logger.warning("Starlink connected but internet connectivity limited")
            else:
                self.state = ConnectionState.CONNECTED

            self._connection_start = datetime.utcnow()
            self._trigger_callbacks("on_connect")

            logger.info(
                f"Starlink connected. Latency: {self._dish_status.pop_ping_latency_ms:.1f}ms"
            )
            return True

        except SatelliteConnectionError:
            self.state = ConnectionState.ERROR
            raise
        except Exception as e:
            self.state = ConnectionState.ERROR
            raise SatelliteConnectionError(
                f"Starlink connection failed: {str(e)}",
                self.provider_name,
                {"error": str(e)},
            )

    async def disconnect(self) -> bool:
        """
        Disconnect from Starlink network.

        Returns:
            True if disconnection successful
        """
        try:
            logger.info("Disconnecting from Starlink")
            self.state = ConnectionState.DISCONNECTED
            self._connection_start = None
            self._trigger_callbacks("on_disconnect")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Starlink: {e}")
            return False

    async def send(self, data: bytes, priority: int = 0) -> bool:
        """
        Send data over Starlink connection.

        Args:
            data: Data to send
            priority: Message priority

        Returns:
            True if send successful
        """
        if not self.is_connected:
            raise SatelliteConnectionError(
                "Not connected to Starlink",
                self.provider_name,
            )

        try:
            # Update metrics
            self._metrics.bytes_sent += len(data)

            # In a real implementation, this would send via the network
            await self._send_queue.put((data, priority))

            return True
        except Exception as e:
            self._metrics.errors_count += 1
            raise SatelliteConnectionError(
                f"Send failed: {str(e)}",
                self.provider_name,
            )

    async def receive(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Receive data from Starlink connection.

        Args:
            timeout: Receive timeout in seconds

        Returns:
            Received data or None if timeout
        """
        if not self.is_connected:
            raise SatelliteConnectionError(
                "Not connected to Starlink",
                self.provider_name,
            )

        try:
            timeout = timeout or self.config.timeout_seconds
            data = await asyncio.wait_for(
                self._receive_queue.get(),
                timeout=timeout,
            )
            self._metrics.bytes_received += len(data)
            return data
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self._metrics.errors_count += 1
            raise SatelliteConnectionError(
                f"Receive failed: {str(e)}",
                self.provider_name,
            )

    async def health_check(self) -> bool:
        """
        Check Starlink connection health.

        Returns:
            True if connection is healthy
        """
        try:
            # Poll dish status
            await self._poll_dish_status()

            if not self._dish_status:
                return False

            # Check online status
            if not self._dish_status.is_online:
                self.state = ConnectionState.ERROR
                return False

            # Check for obstructions
            if self._dish_status.is_obstructed:
                if self._dish_status.obstruction_percent > 50:
                    self.state = ConnectionState.DEGRADED
                else:
                    logger.warning(
                        f"Starlink experiencing {self._dish_status.obstruction_percent:.1f}% obstruction"
                    )

            # Update metrics from dish status
            self._update_metrics(
                latency_ms=self._dish_status.pop_ping_latency_ms,
                bandwidth_kbps=self._dish_status.downlink_throughput_bps / 1000,
                signal_dbm=self._dish_status.snr_db,
            )

            return self._dish_status.is_online

        except Exception as e:
            logger.error(f"Starlink health check failed: {e}")
            self._metrics.errors_count += 1
            return False

    async def get_signal_info(self) -> Dict[str, Any]:
        """
        Get current Starlink signal information.

        Returns:
            Dictionary with signal details
        """
        await self._poll_dish_status()

        if not self._dish_status:
            return {"error": "No dish status available"}

        return {
            "provider": self.provider_name,
            "type": self.provider_type,
            "dish_id": self._dish_status.device_id,
            "hardware_version": self._dish_status.hardware_version,
            "software_version": self._dish_status.software_version,
            "is_online": self._dish_status.is_online,
            "snr_db": self._dish_status.snr_db,
            "latency_ms": self._dish_status.pop_ping_latency_ms,
            "downlink_mbps": self._dish_status.downlink_throughput_bps / 1_000_000,
            "uplink_mbps": self._dish_status.uplink_throughput_bps / 1_000_000,
            "obstruction_percent": self._dish_status.obstruction_percent,
            "is_obstructed": self._dish_status.is_obstructed,
            "tilt_angle_deg": self._dish_status.tilt_angle_deg,
            "azimuth_deg": self._dish_status.boresight_azimuth_deg,
            "elevation_deg": self._dish_status.boresight_elevation_deg,
            "alerts": self._dish_status.alerts,
        }

    async def get_obstruction_map(self) -> Optional[Dict[str, Any]]:
        """
        Get dish obstruction map.

        Returns:
            Obstruction map data or None
        """
        # In a real implementation, this would fetch the obstruction
        # map from the Dishy gRPC API
        return {
            "obstruction_percent": (
                self._dish_status.obstruction_percent if self._dish_status else 0.0
            ),
            "wedge_obstructions": [],
            "last_update": datetime.utcnow().isoformat(),
        }

    async def _check_dish_connectivity(self) -> bool:
        """Check if the Starlink dish is reachable."""
        # In a real implementation, this would ping the dish
        # or attempt a gRPC connection
        return True

    async def _test_internet_connectivity(self) -> bool:
        """Test internet connectivity through Starlink."""
        # In a real implementation, this would test actual connectivity
        return True

    async def _poll_dish_status(self):
        """Poll the Starlink dish for current status."""
        try:
            # In a real implementation, this would use the Starlink gRPC API
            # to fetch actual dish telemetry
            self._dish_status = StarlinkDishStatus(
                device_id="ut-01234567",
                hardware_version="rev3_proto2",
                software_version="2024.01.01.mr12345",
                uptime_seconds=self.metrics.uptime_seconds,
                is_online=True,
                snr_db=9.5,
                downlink_throughput_bps=150_000_000,  # 150 Mbps
                uplink_throughput_bps=20_000_000,  # 20 Mbps
                pop_ping_latency_ms=25.0,
                obstruction_percent=0.0,
                obstruction_duration_seconds=0.0,
                is_obstructed=False,
                tilt_angle_deg=45.0,
                boresight_azimuth_deg=180.0,
                boresight_elevation_deg=45.0,
                alerts=[],
            )
            self._last_dish_poll = datetime.utcnow()

        except Exception as e:
            logger.error(f"Failed to poll Starlink dish: {e}")

    def is_latency_within_normal(self) -> bool:
        """Check if current latency is within normal Starlink range."""
        if not self._dish_status:
            return False

        return (
            self.TYPICAL_LATENCY_MS_MIN
            <= self._dish_status.pop_ping_latency_ms
            <= self.TYPICAL_LATENCY_MS_MAX
        )
