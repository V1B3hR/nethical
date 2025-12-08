"""
Satellite Connection Quality Metrics

Provides comprehensive metrics collection and analysis for
satellite connections including signal quality, jitter,
packet loss, and connection reliability.
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SignalQuality(Enum):
    """Signal quality classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    NO_SIGNAL = "no_signal"


@dataclass
class ConnectionMetrics:
    """Connection quality metrics snapshot."""

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Latency metrics
    latency_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_avg_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Jitter metrics
    jitter_ms: float = 0.0
    jitter_avg_ms: float = 0.0

    # Packet loss
    packet_loss_percent: float = 0.0
    packets_sent: int = 0
    packets_received: int = 0
    packets_lost: int = 0

    # Bandwidth
    bandwidth_download_kbps: float = 0.0
    bandwidth_upload_kbps: float = 0.0

    # Signal
    signal_strength_dbm: float = -100.0
    signal_quality: SignalQuality = SignalQuality.NO_SIGNAL
    snr_db: float = 0.0

    # Connection
    uptime_seconds: float = 0.0
    connection_drops: int = 0
    reconnection_count: int = 0

    # Data transfer
    bytes_sent: int = 0
    bytes_received: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "latency": {
                "current_ms": self.latency_ms,
                "min_ms": self.latency_min_ms,
                "max_ms": self.latency_max_ms,
                "avg_ms": self.latency_avg_ms,
                "p95_ms": self.latency_p95_ms,
                "p99_ms": self.latency_p99_ms,
            },
            "jitter": {
                "current_ms": self.jitter_ms,
                "avg_ms": self.jitter_avg_ms,
            },
            "packet_loss": {
                "percent": self.packet_loss_percent,
                "sent": self.packets_sent,
                "received": self.packets_received,
                "lost": self.packets_lost,
            },
            "bandwidth": {
                "download_kbps": self.bandwidth_download_kbps,
                "upload_kbps": self.bandwidth_upload_kbps,
            },
            "signal": {
                "strength_dbm": self.signal_strength_dbm,
                "quality": self.signal_quality.value,
                "snr_db": self.snr_db,
            },
            "connection": {
                "uptime_seconds": self.uptime_seconds,
                "drops": self.connection_drops,
                "reconnections": self.reconnection_count,
            },
            "data_transfer": {
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
            },
        }


@dataclass
class MetricsSample:
    """Individual metrics sample for time series."""

    timestamp: datetime
    latency_ms: float
    jitter_ms: float
    signal_dbm: float
    packet_success: bool


class SatelliteMetrics:
    """
    Comprehensive satellite connection metrics collection.

    Collects and analyzes:
    - Latency statistics (min, max, avg, percentiles)
    - Jitter measurements
    - Packet loss tracking
    - Signal quality monitoring
    - Connection reliability metrics
    """

    def __init__(
        self,
        sample_window_seconds: int = 300,
        max_samples: int = 10000,
    ):
        """
        Initialize satellite metrics collector.

        Args:
            sample_window_seconds: Time window for metrics calculation
            max_samples: Maximum samples to retain
        """
        self._sample_window = timedelta(seconds=sample_window_seconds)
        self._max_samples = max_samples

        # Sample storage
        self._samples: List[MetricsSample] = []

        # Cumulative counters
        self._total_packets_sent = 0
        self._total_packets_received = 0
        self._total_bytes_sent = 0
        self._total_bytes_received = 0
        self._connection_drops = 0
        self._reconnection_count = 0

        # Connection tracking
        self._connection_start: Optional[datetime] = None
        self._last_update: Optional[datetime] = None

        # Current values
        self._current_latency_ms = 0.0
        self._current_jitter_ms = 0.0
        self._current_signal_dbm = -100.0
        self._current_bandwidth_down_kbps = 0.0
        self._current_bandwidth_up_kbps = 0.0

    def record_sample(
        self,
        latency_ms: float,
        jitter_ms: float = 0.0,
        signal_dbm: float = -100.0,
        packet_success: bool = True,
    ):
        """
        Record a metrics sample.

        Args:
            latency_ms: Measured latency
            jitter_ms: Measured jitter
            signal_dbm: Signal strength
            packet_success: Whether packet was successful
        """
        sample = MetricsSample(
            timestamp=datetime.utcnow(),
            latency_ms=latency_ms,
            jitter_ms=jitter_ms,
            signal_dbm=signal_dbm,
            packet_success=packet_success,
        )

        self._samples.append(sample)

        # Update current values
        self._current_latency_ms = latency_ms
        self._current_jitter_ms = jitter_ms
        self._current_signal_dbm = signal_dbm
        self._last_update = datetime.utcnow()

        # Update counters
        self._total_packets_sent += 1
        if packet_success:
            self._total_packets_received += 1

        # Trim old samples
        self._trim_samples()

    def _trim_samples(self):
        """Remove samples outside the window."""
        cutoff = datetime.utcnow() - self._sample_window
        self._samples = [s for s in self._samples if s.timestamp > cutoff]

        if len(self._samples) > self._max_samples:
            self._samples = self._samples[-self._max_samples :]

    def record_bytes_sent(self, bytes_count: int):
        """Record bytes sent."""
        self._total_bytes_sent += bytes_count

    def record_bytes_received(self, bytes_count: int):
        """Record bytes received."""
        self._total_bytes_received += bytes_count

    def record_connection_drop(self):
        """Record a connection drop event."""
        self._connection_drops += 1

    def record_reconnection(self):
        """Record a reconnection event."""
        self._reconnection_count += 1

    def set_connection_start(self, start_time: Optional[datetime] = None):
        """Set connection start time."""
        self._connection_start = start_time or datetime.utcnow()

    def update_bandwidth(self, download_kbps: float, upload_kbps: float):
        """Update current bandwidth measurements."""
        self._current_bandwidth_down_kbps = download_kbps
        self._current_bandwidth_up_kbps = upload_kbps

    def get_current_metrics(self) -> ConnectionMetrics:
        """
        Get current metrics snapshot.

        Returns:
            ConnectionMetrics with current values
        """
        latency_stats = self._calculate_latency_stats()
        jitter_stats = self._calculate_jitter_stats()
        packet_loss = self._calculate_packet_loss()

        uptime = 0.0
        if self._connection_start:
            uptime = (datetime.utcnow() - self._connection_start).total_seconds()

        return ConnectionMetrics(
            timestamp=datetime.utcnow(),
            latency_ms=self._current_latency_ms,
            latency_min_ms=latency_stats["min"],
            latency_max_ms=latency_stats["max"],
            latency_avg_ms=latency_stats["avg"],
            latency_p95_ms=latency_stats["p95"],
            latency_p99_ms=latency_stats["p99"],
            jitter_ms=self._current_jitter_ms,
            jitter_avg_ms=jitter_stats["avg"],
            packet_loss_percent=packet_loss["percent"],
            packets_sent=self._total_packets_sent,
            packets_received=self._total_packets_received,
            packets_lost=packet_loss["lost"],
            bandwidth_download_kbps=self._current_bandwidth_down_kbps,
            bandwidth_upload_kbps=self._current_bandwidth_up_kbps,
            signal_strength_dbm=self._current_signal_dbm,
            signal_quality=self._classify_signal_quality(),
            snr_db=self._estimate_snr(),
            uptime_seconds=uptime,
            connection_drops=self._connection_drops,
            reconnection_count=self._reconnection_count,
            bytes_sent=self._total_bytes_sent,
            bytes_received=self._total_bytes_received,
        )

    def _calculate_latency_stats(self) -> Dict[str, float]:
        """Calculate latency statistics."""
        if not self._samples:
            return {"min": 0, "max": 0, "avg": 0, "p95": 0, "p99": 0}

        latencies = [s.latency_ms for s in self._samples]
        sorted_latencies = sorted(latencies)

        return {
            "min": min(latencies),
            "max": max(latencies),
            "avg": statistics.mean(latencies),
            "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
            "p99": sorted_latencies[int(len(sorted_latencies) * 0.99)],
        }

    def _calculate_jitter_stats(self) -> Dict[str, float]:
        """Calculate jitter statistics."""
        if not self._samples:
            return {"avg": 0}

        jitters = [s.jitter_ms for s in self._samples if s.jitter_ms > 0]
        if not jitters:
            return {"avg": 0}

        return {"avg": statistics.mean(jitters)}

    def _calculate_packet_loss(self) -> Dict[str, Any]:
        """Calculate packet loss statistics."""
        if self._total_packets_sent == 0:
            return {"percent": 0.0, "lost": 0}

        lost = self._total_packets_sent - self._total_packets_received
        percent = (lost / self._total_packets_sent) * 100

        return {"percent": percent, "lost": lost}

    def _classify_signal_quality(self) -> SignalQuality:
        """Classify signal quality based on strength."""
        dbm = self._current_signal_dbm

        if dbm >= -50:
            return SignalQuality.EXCELLENT
        elif dbm >= -60:
            return SignalQuality.GOOD
        elif dbm >= -70:
            return SignalQuality.FAIR
        elif dbm >= -80:
            return SignalQuality.POOR
        elif dbm >= -90:
            return SignalQuality.CRITICAL
        else:
            return SignalQuality.NO_SIGNAL

    def _estimate_snr(self) -> float:
        """Estimate SNR from signal strength."""
        # Simplified SNR estimation
        # In real implementation, this would use actual noise measurements
        noise_floor_dbm = -100.0
        return max(0, self._current_signal_dbm - noise_floor_dbm)

    def get_quality_score(self) -> float:
        """
        Calculate overall connection quality score (0-1).

        Returns:
            Quality score between 0 and 1
        """
        if not self._samples:
            return 0.0

        # Latency score (0-1, lower is better)
        latency_stats = self._calculate_latency_stats()
        latency_score = max(0, 1 - (latency_stats["avg"] / 500))  # 500ms = 0

        # Packet loss score (0-1, lower is better)
        packet_loss = self._calculate_packet_loss()
        loss_score = max(0, 1 - (packet_loss["percent"] / 10))  # 10% = 0

        # Signal score (0-1)
        signal = self._classify_signal_quality()
        signal_scores = {
            SignalQuality.EXCELLENT: 1.0,
            SignalQuality.GOOD: 0.8,
            SignalQuality.FAIR: 0.6,
            SignalQuality.POOR: 0.4,
            SignalQuality.CRITICAL: 0.2,
            SignalQuality.NO_SIGNAL: 0.0,
        }
        signal_score = signal_scores.get(signal, 0.0)

        # Weighted average
        return latency_score * 0.4 + loss_score * 0.4 + signal_score * 0.2

    def get_health_assessment(self) -> Dict[str, Any]:
        """
        Get connection health assessment.

        Returns:
            Health assessment dictionary
        """
        quality_score = self.get_quality_score()
        metrics = self.get_current_metrics()

        # Determine health status
        if quality_score >= 0.8:
            status = "healthy"
        elif quality_score >= 0.6:
            status = "degraded"
        elif quality_score >= 0.4:
            status = "poor"
        else:
            status = "critical"

        # Generate recommendations
        recommendations = []
        if metrics.latency_avg_ms > 200:
            recommendations.append("High latency detected - consider request batching")
        if metrics.packet_loss_percent > 2:
            recommendations.append("Packet loss above threshold - check signal quality")
        if metrics.signal_quality in (SignalQuality.POOR, SignalQuality.CRITICAL):
            recommendations.append("Poor signal - verify antenna alignment")

        return {
            "status": status,
            "quality_score": quality_score,
            "signal_quality": metrics.signal_quality.value,
            "latency_avg_ms": metrics.latency_avg_ms,
            "packet_loss_percent": metrics.packet_loss_percent,
            "uptime_seconds": metrics.uptime_seconds,
            "recommendations": recommendations,
            "last_update": self._last_update.isoformat() if self._last_update else None,
        }

    def reset(self):
        """Reset all metrics."""
        self._samples.clear()
        self._total_packets_sent = 0
        self._total_packets_received = 0
        self._total_bytes_sent = 0
        self._total_bytes_received = 0
        self._connection_drops = 0
        self._reconnection_count = 0
        self._connection_start = None
        self._last_update = None
        logger.info("Satellite metrics reset")
