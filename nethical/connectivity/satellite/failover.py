"""
Terrestrial ↔ Satellite Automatic Failover

Provides automatic switching between terrestrial and satellite
connectivity based on health checks, latency thresholds, and
connection quality scoring.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .base import (
    ConnectionConfig,
    ConnectionState,
    SatelliteProvider,
)

logger = logging.getLogger(__name__)


class ConnectionType(Enum):
    """Connection type enumeration."""

    TERRESTRIAL = "terrestrial"
    SATELLITE = "satellite"
    HYBRID = "hybrid"


class FailoverReason(Enum):
    """Reason for failover event."""

    LATENCY_THRESHOLD = "latency_threshold"
    PACKET_LOSS = "packet_loss"
    CONNECTION_LOST = "connection_lost"
    BANDWIDTH_LOW = "bandwidth_low"
    MANUAL = "manual"
    HEALTH_CHECK_FAILED = "health_check_failed"
    QUALITY_SCORE = "quality_score"


@dataclass
class FailoverEvent:
    """Represents a failover event."""

    timestamp: datetime
    from_connection: ConnectionType
    to_connection: ConnectionType
    reason: FailoverReason
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    success: bool = True


@dataclass
class FailoverConfig:
    """Configuration for failover behavior."""

    # Thresholds
    latency_threshold_ms: float = 500.0
    packet_loss_threshold_percent: float = 5.0
    bandwidth_threshold_kbps: float = 100.0
    quality_score_threshold: float = 0.5

    # Health check settings
    health_check_interval_seconds: float = 30.0
    health_check_timeout_seconds: float = 10.0
    consecutive_failures_for_failover: int = 3

    # Failover behavior
    auto_failback: bool = True
    failback_delay_seconds: float = 60.0
    min_time_on_backup_seconds: float = 30.0

    # Preferred connections
    preferred_connection: ConnectionType = ConnectionType.TERRESTRIAL
    always_prefer_terrestrial: bool = True


class FailoverManager:
    """
    Manages automatic failover between terrestrial and satellite connections.

    Features:
    - Automatic switching based on configurable thresholds
    - Health check probes for each connection type
    - Connection quality scoring
    - Event logging and callbacks
    - Configurable failover/failback policies
    """

    def __init__(
        self,
        config: Optional[FailoverConfig] = None,
        satellite_provider: Optional[SatelliteProvider] = None,
    ):
        """
        Initialize failover manager.

        Args:
            config: Failover configuration
            satellite_provider: Satellite provider instance
        """
        self.config = config or FailoverConfig()
        self._satellite_provider = satellite_provider

        self._active_connection = self.config.preferred_connection
        self._is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Connection states
        self._terrestrial_healthy = True
        self._satellite_healthy = False
        self._terrestrial_quality_score = 1.0
        self._satellite_quality_score = 0.0

        # Failure tracking
        self._terrestrial_consecutive_failures = 0
        self._satellite_consecutive_failures = 0

        # Event history
        self._failover_events: List[FailoverEvent] = []
        self._max_events = 100

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "on_failover": [],
            "on_failback": [],
            "on_quality_change": [],
        }

        # Timing
        self._last_failover: Optional[datetime] = None
        self._last_failback: Optional[datetime] = None

    @property
    def active_connection(self) -> ConnectionType:
        """Get currently active connection type."""
        return self._active_connection

    @property
    def is_on_satellite(self) -> bool:
        """Check if currently using satellite connection."""
        return self._active_connection == ConnectionType.SATELLITE

    @property
    def is_on_terrestrial(self) -> bool:
        """Check if currently using terrestrial connection."""
        return self._active_connection == ConnectionType.TERRESTRIAL

    @property
    def failover_events(self) -> List[FailoverEvent]:
        """Get failover event history."""
        return self._failover_events.copy()

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Failover monitoring started")

    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Failover monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                await self._perform_health_checks()
                await self._evaluate_failover()
                await asyncio.sleep(self.config.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in failover monitoring: {e}")
                await asyncio.sleep(5)

    async def _perform_health_checks(self):
        """Perform health checks on all connections."""
        # Check terrestrial connection
        terrestrial_ok = await self._check_terrestrial_health()
        if terrestrial_ok:
            self._terrestrial_consecutive_failures = 0
            self._terrestrial_healthy = True
        else:
            self._terrestrial_consecutive_failures += 1
            if (
                self._terrestrial_consecutive_failures
                >= self.config.consecutive_failures_for_failover
            ):
                self._terrestrial_healthy = False

        # Check satellite connection
        if self._satellite_provider:
            satellite_ok = await self._check_satellite_health()
            if satellite_ok:
                self._satellite_consecutive_failures = 0
                self._satellite_healthy = True
            else:
                self._satellite_consecutive_failures += 1
                if (
                    self._satellite_consecutive_failures
                    >= self.config.consecutive_failures_for_failover
                ):
                    self._satellite_healthy = False

        # Update quality scores
        self._update_quality_scores()

    async def _check_terrestrial_health(self) -> bool:
        """Check terrestrial connection health."""
        # In a real implementation, this would ping/test terrestrial connectivity
        # Simulated health check
        return True

    async def _check_satellite_health(self) -> bool:
        """Check satellite connection health."""
        if not self._satellite_provider:
            return False

        try:
            return await self._satellite_provider.health_check()
        except Exception as e:
            logger.warning(f"Satellite health check failed: {e}")
            return False

    def _update_quality_scores(self):
        """Update connection quality scores."""
        # Terrestrial quality based on health
        if self._terrestrial_healthy:
            self._terrestrial_quality_score = 1.0
        else:
            self._terrestrial_quality_score = max(
                0, 1.0 - (self._terrestrial_consecutive_failures * 0.2)
            )

        # Satellite quality based on metrics
        if self._satellite_provider and self._satellite_healthy:
            metrics = self._satellite_provider.metrics
            latency_score = max(
                0, 1.0 - (metrics.latency_ms / self.config.latency_threshold_ms)
            )
            loss_score = max(
                0,
                1.0
                - (metrics.packet_loss_percent / self.config.packet_loss_threshold_percent),
            )
            self._satellite_quality_score = (latency_score + loss_score) / 2
        else:
            self._satellite_quality_score = 0.0

        # Trigger quality change callbacks
        for callback in self._callbacks["on_quality_change"]:
            callback(
                self._terrestrial_quality_score,
                self._satellite_quality_score,
            )

    async def _evaluate_failover(self):
        """Evaluate if failover is needed."""
        # Check if we should fail over to satellite
        if self.is_on_terrestrial and not self._terrestrial_healthy:
            if self._satellite_healthy:
                await self._do_failover(
                    ConnectionType.TERRESTRIAL,
                    ConnectionType.SATELLITE,
                    FailoverReason.CONNECTION_LOST,
                )

        # Check if we should fail back to terrestrial
        elif self.is_on_satellite and self.config.auto_failback:
            if self._terrestrial_healthy:
                # Check minimum time on backup
                if self._last_failover:
                    time_on_satellite = (
                        datetime.utcnow() - self._last_failover
                    ).total_seconds()
                    if time_on_satellite < self.config.min_time_on_backup_seconds:
                        return

                # Check failback delay
                if self._last_failback:
                    time_since_failback = (
                        datetime.utcnow() - self._last_failback
                    ).total_seconds()
                    if time_since_failback < self.config.failback_delay_seconds:
                        return

                await self._do_failback(
                    ConnectionType.SATELLITE,
                    ConnectionType.TERRESTRIAL,
                )

    async def _do_failover(
        self,
        from_conn: ConnectionType,
        to_conn: ConnectionType,
        reason: FailoverReason,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Execute failover."""
        start_time = datetime.utcnow()
        success = True

        try:
            logger.info(f"Initiating failover: {from_conn.value} -> {to_conn.value}")

            # Activate satellite if failing over to it
            if to_conn == ConnectionType.SATELLITE and self._satellite_provider:
                if not self._satellite_provider.is_connected:
                    await self._satellite_provider.connect()

            self._active_connection = to_conn
            self._last_failover = datetime.utcnow()

        except Exception as e:
            logger.error(f"Failover failed: {e}")
            success = False

        # Record event
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        event = FailoverEvent(
            timestamp=start_time,
            from_connection=from_conn,
            to_connection=to_conn,
            reason=reason,
            details=details or {},
            duration_ms=duration,
            success=success,
        )
        self._record_event(event)

        # Trigger callbacks
        for callback in self._callbacks["on_failover"]:
            callback(event)

        if success:
            logger.info(
                f"Failover complete: {from_conn.value} -> {to_conn.value} "
                f"({duration:.1f}ms)"
            )

    async def _do_failback(
        self,
        from_conn: ConnectionType,
        to_conn: ConnectionType,
    ):
        """Execute failback to preferred connection."""
        start_time = datetime.utcnow()
        success = True

        try:
            logger.info(f"Initiating failback: {from_conn.value} -> {to_conn.value}")
            self._active_connection = to_conn
            self._last_failback = datetime.utcnow()

        except Exception as e:
            logger.error(f"Failback failed: {e}")
            success = False

        # Record event
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        event = FailoverEvent(
            timestamp=start_time,
            from_connection=from_conn,
            to_connection=to_conn,
            reason=FailoverReason.MANUAL,
            details={"type": "failback"},
            duration_ms=duration,
            success=success,
        )
        self._record_event(event)

        # Trigger callbacks
        for callback in self._callbacks["on_failback"]:
            callback(event)

    def _record_event(self, event: FailoverEvent):
        """Record a failover event."""
        self._failover_events.append(event)
        if len(self._failover_events) > self._max_events:
            self._failover_events.pop(0)

    async def force_failover_to_satellite(self) -> bool:
        """
        Force immediate failover to satellite.

        Returns:
            True if failover successful
        """
        if self.is_on_satellite:
            return True

        if not self._satellite_healthy:
            logger.warning("Cannot failover: satellite connection unhealthy")
            return False

        await self._do_failover(
            ConnectionType.TERRESTRIAL,
            ConnectionType.SATELLITE,
            FailoverReason.MANUAL,
        )
        return True

    async def force_failback_to_terrestrial(self) -> bool:
        """
        Force immediate failback to terrestrial.

        Returns:
            True if failback successful
        """
        if self.is_on_terrestrial:
            return True

        if not self._terrestrial_healthy:
            logger.warning("Cannot failback: terrestrial connection unhealthy")
            return False

        await self._do_failback(
            ConnectionType.SATELLITE,
            ConnectionType.TERRESTRIAL,
        )
        return True

    def register_callback(self, event: str, callback: Callable):
        """Register a callback for failover events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def get_status(self) -> Dict[str, Any]:
        """
        Get failover manager status.

        Returns:
            Status dictionary
        """
        return {
            "active_connection": self._active_connection.value,
            "is_monitoring": self._is_monitoring,
            "terrestrial_healthy": self._terrestrial_healthy,
            "satellite_healthy": self._satellite_healthy,
            "terrestrial_quality_score": self._terrestrial_quality_score,
            "satellite_quality_score": self._satellite_quality_score,
            "failover_count": len(
                [e for e in self._failover_events if e.success]
            ),
            "last_failover": (
                self._last_failover.isoformat() if self._last_failover else None
            ),
            "last_failback": (
                self._last_failback.isoformat() if self._last_failback else None
            ),
        }
