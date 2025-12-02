"""
Network Monitor - Connectivity Detection

Monitors network connectivity for offline mode detection.
"""

import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStatus:
    """
    Network connection status.

    Attributes:
        is_connected: Whether network is available
        latency_ms: Network latency in milliseconds
        last_check: Timestamp of last check
        consecutive_failures: Number of consecutive failures
        quality: Connection quality (good, degraded, poor, none)
    """

    is_connected: bool
    latency_ms: Optional[float] = None
    last_check: float = 0.0
    consecutive_failures: int = 0
    quality: str = "unknown"


class NetworkMonitor:
    """
    Network connectivity monitor.

    Features:
    - Heartbeat monitoring
    - Network quality assessment
    - Graceful degradation triggers
    """

    # Default endpoints to check
    DEFAULT_ENDPOINTS = [
        ("8.8.8.8", 53),  # Google DNS
        ("1.1.1.1", 53),  # Cloudflare DNS
    ]

    def __init__(
        self,
        endpoints: Optional[List[tuple]] = None,
        check_interval_seconds: int = 10,
        timeout_seconds: float = 2.0,
        failure_threshold: int = 3,
    ):
        """
        Initialize NetworkMonitor.

        Args:
            endpoints: List of (host, port) tuples to check
            check_interval_seconds: Interval between checks
            timeout_seconds: Socket timeout
            failure_threshold: Consecutive failures before offline
        """
        self.endpoints = endpoints or self.DEFAULT_ENDPOINTS
        self.check_interval_seconds = check_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.failure_threshold = failure_threshold

        # State
        self._status = ConnectionStatus(is_connected=True, last_check=time.time())
        self._lock = threading.RLock()

        # Background monitoring
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        logger.info("NetworkMonitor initialized")

    def start_monitoring(self):
        """Start background connectivity monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Network monitoring started")

    def stop_monitoring(self):
        """Stop background connectivity monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        logger.info("Network monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self.check_connectivity()
            except Exception as e:
                logger.error(f"Network check error: {e}")
            time.sleep(self.check_interval_seconds)

    def check_connectivity(self) -> ConnectionStatus:
        """
        Check network connectivity.

        Returns:
            ConnectionStatus with result
        """
        start_time = time.perf_counter()
        is_connected = False
        latency_ms = None

        for host, port in self.endpoints:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout_seconds)

                connect_start = time.perf_counter()
                sock.connect((host, port))
                connect_time = time.perf_counter() - connect_start

                sock.close()

                is_connected = True
                latency_ms = connect_time * 1000
                break  # Success on first endpoint

            except (socket.timeout, socket.error):
                continue

        with self._lock:
            if is_connected:
                self._status.is_connected = True
                self._status.latency_ms = latency_ms
                self._status.consecutive_failures = 0
                self._status.quality = self._assess_quality(latency_ms)
            else:
                self._status.consecutive_failures += 1
                if self._status.consecutive_failures >= self.failure_threshold:
                    self._status.is_connected = False
                    self._status.quality = "none"

            self._status.last_check = time.time()

            return ConnectionStatus(
                is_connected=self._status.is_connected,
                latency_ms=self._status.latency_ms,
                last_check=self._status.last_check,
                consecutive_failures=self._status.consecutive_failures,
                quality=self._status.quality,
            )

    def _assess_quality(self, latency_ms: Optional[float]) -> str:
        """Assess connection quality based on latency."""
        if latency_ms is None:
            return "unknown"
        if latency_ms < 50:
            return "good"
        if latency_ms < 200:
            return "degraded"
        return "poor"

    def get_status(self, force_check: bool = False) -> ConnectionStatus:
        """
        Get current connection status.

        Args:
            force_check: Force immediate check

        Returns:
            Current ConnectionStatus
        """
        if force_check:
            return self.check_connectivity()

        # Return cached status if recent
        with self._lock:
            age = time.time() - self._status.last_check
            if age < self.check_interval_seconds:
                return ConnectionStatus(
                    is_connected=self._status.is_connected,
                    latency_ms=self._status.latency_ms,
                    last_check=self._status.last_check,
                    consecutive_failures=self._status.consecutive_failures,
                    quality=self._status.quality,
                )

        return self.check_connectivity()

    def is_connected(self) -> bool:
        """Quick check if connected (uses cached status)."""
        with self._lock:
            return self._status.is_connected

    def get_latency(self) -> Optional[float]:
        """Get last measured latency in milliseconds."""
        with self._lock:
            return self._status.latency_ms

    def add_endpoint(self, host: str, port: int):
        """Add an endpoint to check."""
        self.endpoints.append((host, port))

    def get_metrics(self) -> dict:
        """Get monitor metrics."""
        with self._lock:
            return {
                "is_connected": self._status.is_connected,
                "latency_ms": self._status.latency_ms,
                "quality": self._status.quality,
                "consecutive_failures": self._status.consecutive_failures,
                "last_check": self._status.last_check,
                "endpoints_count": len(self.endpoints),
            }
