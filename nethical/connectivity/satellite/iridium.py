"""
Iridium Satellite Network Integration

Provides integration with the Iridium satellite network for
global voice and data connectivity, including polar regions.
Iridium is ideal for safety-critical and maritime applications.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .base import (
    ConnectionConfig,
    ConnectionState,
    SatelliteConnectionError,
    SatelliteProvider,
)

logger = logging.getLogger(__name__)


@dataclass
class IridiumModemStatus:
    """Iridium modem status information."""

    modem_id: str = ""
    imei: str = ""
    firmware_version: str = ""
    is_registered: bool = False

    # Signal metrics
    signal_bars: int = 0  # 0-5 bars
    signal_quality_percent: float = 0.0

    # Network info
    network_available: bool = False
    satellite_visible: bool = False

    # SBD (Short Burst Data) status
    mo_queue_length: int = 0  # Mobile Originated queue
    mt_queue_length: int = 0  # Mobile Terminated queue


class IridiumProvider(SatelliteProvider):
    """
    Iridium satellite network integration.

    Iridium provides truly global coverage including polar regions
    through a constellation of 66 active LEO satellites. While
    bandwidth is limited compared to newer LEO constellations,
    Iridium excels in reliability and polar coverage.

    Specifications:
    - Latency: ~100-300ms
    - Bandwidth: 2.4 kbps (SBD), up to 704 kbps (Certus)
    - Coverage: True global (including poles)
    - Reliability: Very high
    """

    # Iridium performance specifications
    TYPICAL_LATENCY_MS = 200.0
    SBD_BANDWIDTH_KBPS = 2.4
    CERTUS_BANDWIDTH_KBPS = 704.0

    def __init__(self, config: Optional[ConnectionConfig] = None):
        """
        Initialize Iridium provider.

        Args:
            config: Connection configuration
        """
        super().__init__(config)
        self._modem_status: Optional[IridiumModemStatus] = None
        self._serial_port = self.config.provider_options.get(
            "serial_port", "/dev/ttyUSB0"
        )
        self._service_type = self.config.provider_options.get("service_type", "sbd")

    @property
    def provider_name(self) -> str:
        return "Iridium"

    @property
    def provider_type(self) -> str:
        return "LEO"

    @property
    def service_type(self) -> str:
        """Get service type (sbd or certus)."""
        return self._service_type

    async def connect(self) -> bool:
        """
        Establish connection to Iridium network.

        Returns:
            True if connection successful

        Raises:
            SatelliteConnectionError: If connection fails
        """
        try:
            self.state = ConnectionState.CONNECTING
            logger.info(f"Connecting to Iridium via {self._serial_port}")

            # Initialize modem
            modem_ok = await self._initialize_modem()
            if not modem_ok:
                raise SatelliteConnectionError(
                    "Failed to initialize Iridium modem",
                    self.provider_name,
                )

            # Register with network
            registered = await self._register_network()
            if not registered:
                raise SatelliteConnectionError(
                    "Failed to register with Iridium network",
                    self.provider_name,
                )

            # Get initial status
            await self._poll_modem_status()

            self._connection_start = datetime.utcnow()
            self.state = ConnectionState.CONNECTED
            self._trigger_callbacks("on_connect")

            logger.info("Iridium connection established")
            return True

        except SatelliteConnectionError:
            self.state = ConnectionState.ERROR
            raise
        except Exception as e:
            self.state = ConnectionState.ERROR
            raise SatelliteConnectionError(
                f"Iridium connection failed: {str(e)}",
                self.provider_name,
            )

    async def disconnect(self) -> bool:
        """
        Disconnect from Iridium network.

        Returns:
            True if disconnection successful
        """
        try:
            self.state = ConnectionState.DISCONNECTED
            self._connection_start = None
            self._trigger_callbacks("on_disconnect")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Iridium: {e}")
            return False

    async def send(self, data: bytes, priority: int = 0) -> bool:
        """
        Send data over Iridium connection (SBD or Certus).

        Args:
            data: Data to send (max 340 bytes for SBD MO)
            priority: Message priority

        Returns:
            True if send successful
        """
        if not self.is_connected:
            raise SatelliteConnectionError(
                "Not connected to Iridium",
                self.provider_name,
            )

        # Check message size for SBD
        if self._service_type == "sbd" and len(data) > 340:
            raise SatelliteConnectionError(
                f"SBD message too large: {len(data)} bytes (max 340)",
                self.provider_name,
            )

        try:
            self._metrics.bytes_sent += len(data)
            logger.debug(f"Iridium SBD send: {len(data)} bytes")
            return True
        except Exception as e:
            self._metrics.errors_count += 1
            raise SatelliteConnectionError(
                f"Send failed: {str(e)}",
                self.provider_name,
            )

    async def receive(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Receive data from Iridium connection.

        Args:
            timeout: Receive timeout in seconds

        Returns:
            Received data or None if timeout
        """
        if not self.is_connected:
            raise SatelliteConnectionError(
                "Not connected to Iridium",
                self.provider_name,
            )

        try:
            timeout = timeout or self.config.timeout_seconds
            # Check for queued MT messages
            if self._modem_status and self._modem_status.mt_queue_length > 0:
                # Stub implementation:
                # In production, this would retrieve actual MT messages from the modem.
                # The simulated response is for testing and development purposes only.
                await asyncio.sleep(min(timeout, 0.1))
                return b"MT message data"  # Simulated MT message
            return None
        except Exception as e:
            self._metrics.errors_count += 1
            raise SatelliteConnectionError(
                f"Receive failed: {str(e)}",
                self.provider_name,
            )

    async def health_check(self) -> bool:
        """
        Check Iridium connection health.

        Returns:
            True if connection is healthy
        """
        try:
            await self._poll_modem_status()

            if not self._modem_status:
                return False

            if not self._modem_status.is_registered:
                self.state = ConnectionState.ERROR
                return False

            if not self._modem_status.network_available:
                self.state = ConnectionState.DEGRADED
                return False

            bandwidth = (
                self.CERTUS_BANDWIDTH_KBPS
                if self._service_type == "certus"
                else self.SBD_BANDWIDTH_KBPS
            )

            self._update_metrics(
                latency_ms=self.TYPICAL_LATENCY_MS,
                bandwidth_kbps=bandwidth,
                signal_dbm=-100 + (self._modem_status.signal_bars * 10),
            )

            return True

        except Exception as e:
            logger.error(f"Iridium health check failed: {e}")
            return False

    async def get_signal_info(self) -> Dict[str, Any]:
        """
        Get current Iridium signal information.

        Returns:
            Dictionary with signal details
        """
        await self._poll_modem_status()

        if not self._modem_status:
            return {"error": "No modem status available"}

        return {
            "provider": self.provider_name,
            "type": self.provider_type,
            "service_type": self._service_type,
            "modem_id": self._modem_status.modem_id,
            "imei": self._modem_status.imei,
            "is_registered": self._modem_status.is_registered,
            "signal_bars": self._modem_status.signal_bars,
            "signal_quality_percent": self._modem_status.signal_quality_percent,
            "network_available": self._modem_status.network_available,
            "satellite_visible": self._modem_status.satellite_visible,
            "mo_queue_length": self._modem_status.mo_queue_length,
            "mt_queue_length": self._modem_status.mt_queue_length,
        }

    async def send_sbd_message(self, data: bytes) -> Optional[str]:
        """
        Send an SBD (Short Burst Data) message.

        Args:
            data: Message data (max 340 bytes)

        Returns:
            Message ID if successful, None otherwise
        """
        if len(data) > 340:
            raise ValueError("SBD message cannot exceed 340 bytes")

        success = await self.send(data)
        if success:
            return f"SBD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        return None

    async def check_mailbox(self) -> int:
        """
        Check for waiting MT (Mobile Terminated) messages.

        Returns:
            Number of waiting messages
        """
        await self._poll_modem_status()
        if self._modem_status:
            return self._modem_status.mt_queue_length
        return 0

    async def _initialize_modem(self) -> bool:
        """Initialize the Iridium modem."""
        # In a real implementation, this would send AT commands
        return True

    async def _register_network(self) -> bool:
        """Register with the Iridium network."""
        # In a real implementation, this would perform network registration
        return True

    async def _poll_modem_status(self):
        """Poll the Iridium modem for current status."""
        try:
            self._modem_status = IridiumModemStatus(
                modem_id="9602-001",
                imei="300234063904190",
                firmware_version="33105",
                is_registered=True,
                signal_bars=4,
                signal_quality_percent=80.0,
                network_available=True,
                satellite_visible=True,
                mo_queue_length=0,
                mt_queue_length=0,
            )
        except Exception as e:
            logger.error(f"Failed to poll Iridium modem: {e}")
