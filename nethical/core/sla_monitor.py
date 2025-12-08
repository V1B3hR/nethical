"""SLA Monitoring for Phase 4.5: SLA & Performance.

This module implements:
- P95 latency tracking
- Load testing validation
- SLA documentation and validation
- Performance guarantee monitoring
"""

import statistics
from typing import Dict, List, Optional, Any, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import deque
from enum import Enum


class SLAStatus(str, Enum):
    """SLA compliance status."""

    COMPLIANT = "compliant"
    WARNING = "warning"
    BREACH = "breach"
    UNKNOWN = "unknown"


@dataclass
class LatencyWindow:
    """Sliding window of latency measurements."""

    measurements: Deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    timestamps: Deque[datetime] = field(default_factory=lambda: deque(maxlen=10000))
    window_size_seconds: float = 300.0  # 5 minutes


@dataclass
class SLATarget:
    """SLA target definition."""

    metric_name: str
    target_value: float
    percentile: Optional[float] = None  # e.g., 95 for P95
    unit: str = "ms"
    description: str = ""


@dataclass
class SLABreach:
    """SLA breach record."""

    metric_name: str
    target_value: float
    actual_value: float
    breach_percentage: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SLAMonitor:
    """SLA monitoring and validation system."""

    def __init__(
        self,
        target_p95_ms: float = 220.0,
        target_p99_ms: float = 500.0,
        target_avg_ms: float = 100.0,
        warning_threshold_pct: float = 0.9,  # Warn at 90% of target
        breach_threshold_count: int = 10,  # Consecutive breaches to trigger alert
        window_size_seconds: float = 300.0,
    ):
        """Initialize SLA monitor.

        Args:
            target_p95_ms: Target P95 latency in milliseconds
            target_p99_ms: Target P99 latency in milliseconds
            target_avg_ms: Target average latency in milliseconds
            warning_threshold_pct: Warning threshold as percentage of target
            breach_threshold_count: Breaches needed to trigger alert
            window_size_seconds: Rolling window size
        """
        self.target_p95_ms = target_p95_ms
        self.target_p99_ms = target_p99_ms
        self.target_avg_ms = target_avg_ms
        self.warning_threshold_pct = warning_threshold_pct
        self.breach_threshold_count = breach_threshold_count
        self.window_size_seconds = window_size_seconds

        # Latency measurements
        self.latency_window = LatencyWindow(window_size_seconds=window_size_seconds)

        # SLA targets
        self.sla_targets: Dict[str, SLATarget] = {
            "p95_latency": SLATarget(
                metric_name="p95_latency",
                target_value=target_p95_ms,
                percentile=95,
                unit="ms",
                description="95th percentile latency",
            ),
            "p99_latency": SLATarget(
                metric_name="p99_latency",
                target_value=target_p99_ms,
                percentile=99,
                unit="ms",
                description="99th percentile latency",
            ),
            "avg_latency": SLATarget(
                metric_name="avg_latency",
                target_value=target_avg_ms,
                unit="ms",
                description="Average latency",
            ),
        }

        # Breach tracking
        self.breaches: List[SLABreach] = []
        self.consecutive_breaches = 0
        self.last_breach_check: Optional[datetime] = None

        # Load tracking
        self.current_load_multiplier = 1.0
        self.load_history: List[Dict[str, Any]] = []

    def record_latency(self, latency_ms: float, timestamp: Optional[datetime] = None):
        """Record a latency measurement.

        Args:
            latency_ms: Latency in milliseconds
            timestamp: Optional timestamp (defaults to now)
        """
        ts = timestamp or datetime.now(timezone.utc)

        self.latency_window.measurements.append(latency_ms)
        self.latency_window.timestamps.append(ts)

        # Cleanup old measurements outside window
        self._cleanup_window()

    def _cleanup_window(self):
        """Remove measurements outside the rolling window."""
        if not self.latency_window.timestamps:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.window_size_seconds
        )

        while (
            self.latency_window.timestamps
            and self.latency_window.timestamps[0] < cutoff
        ):
            self.latency_window.measurements.popleft()
            self.latency_window.timestamps.popleft()

    def _calculate_percentile(
        self, measurements: List[float], percentile: float
    ) -> float:
        """Calculate percentile from measurements.

        Args:
            measurements: List of measurements
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not measurements:
            return 0.0

        sorted_measurements = sorted(measurements)
        index = int(len(sorted_measurements) * percentile / 100.0)
        index = min(index, len(sorted_measurements) - 1)

        return sorted_measurements[index]

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics.

        Returns:
            Dictionary of metrics
        """
        measurements = list(self.latency_window.measurements)

        if not measurements:
            return {
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "avg_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "sample_count": 0,
            }

        return {
            "p50_latency_ms": self._calculate_percentile(measurements, 50),
            "p95_latency_ms": self._calculate_percentile(measurements, 95),
            "p99_latency_ms": self._calculate_percentile(measurements, 99),
            "avg_latency_ms": statistics.mean(measurements),
            "max_latency_ms": max(measurements),
            "min_latency_ms": min(measurements),
            "sample_count": len(measurements),
        }

    def check_sla_compliance(self) -> Dict[str, Any]:
        """Check SLA compliance.

        Returns:
            Compliance status dictionary
        """
        metrics = self.get_current_metrics()

        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": SLAStatus.COMPLIANT,
            "metrics": metrics,
            "targets": {},
            "breaches": [],
        }

        # Check each target
        for target_name, target in self.sla_targets.items():
            if target.percentile:
                metric_key = f"p{int(target.percentile)}_latency_ms"
            else:
                metric_key = "avg_latency_ms"

            actual_value = metrics.get(metric_key, 0.0)
            target_value = target.target_value

            # Calculate compliance
            if actual_value <= target_value:
                status = SLAStatus.COMPLIANT
            elif actual_value <= target_value * (1 + (1 - self.warning_threshold_pct)):
                status = SLAStatus.WARNING
            else:
                status = SLAStatus.BREACH

            breach_pct = (
                ((actual_value - target_value) / target_value * 100)
                if actual_value > target_value
                else 0.0
            )

            results["targets"][target_name] = {
                "target": target_value,
                "actual": actual_value,
                "status": status.value,
                "breach_percentage": breach_pct,
                "description": target.description,
            }

            # Track breaches
            if status == SLAStatus.BREACH:
                breach = SLABreach(
                    metric_name=target_name,
                    target_value=target_value,
                    actual_value=actual_value,
                    breach_percentage=breach_pct,
                )
                self.breaches.append(breach)
                results["breaches"].append(
                    {
                        "metric": target_name,
                        "target": target_value,
                        "actual": actual_value,
                        "breach_pct": breach_pct,
                    }
                )
                results["overall_status"] = SLAStatus.BREACH
            elif (
                status == SLAStatus.WARNING
                and results["overall_status"] == SLAStatus.COMPLIANT
            ):
                results["overall_status"] = SLAStatus.WARNING

        self.last_breach_check = datetime.now(timezone.utc)

        return results

    def get_sla_report(self) -> Dict[str, Any]:
        """Get comprehensive SLA report.

        Returns:
            SLA report dictionary
        """
        compliance = self.check_sla_compliance()
        metrics = compliance["metrics"]

        # Calculate uptime metrics
        total_measurements = metrics["sample_count"]
        breach_count = len(
            [
                b
                for b in self.breaches
                if b.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
            ]
        )

        # P95 specific check (primary SLA)
        p95_actual = metrics["p95_latency_ms"]
        p95_target = self.target_p95_ms
        p95_met = p95_actual <= p95_target

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": compliance["overall_status"],
            "sla_met": p95_met,
            "p95_latency_ms": p95_actual,
            "p95_target_ms": p95_target,
            "p95_margin_ms": p95_target - p95_actual,
            "p95_margin_pct": (
                ((p95_target - p95_actual) / p95_target * 100) if p95_target > 0 else 0
            ),
            "metrics": metrics,
            "targets": compliance["targets"],
            "total_measurements": total_measurements,
            "breach_count_24h": breach_count,
            "current_load_multiplier": self.current_load_multiplier,
            "window_size_seconds": self.window_size_seconds,
        }

    def set_load_multiplier(self, multiplier: float):
        """Set current load multiplier for testing.

        Args:
            multiplier: Load multiplier (e.g., 2.0 for 2× load)
        """
        self.current_load_multiplier = multiplier

        self.load_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "multiplier": multiplier,
            }
        )

    def validate_under_load(
        self, load_multiplier: float = 2.0, duration_seconds: float = 60.0
    ) -> Dict[str, Any]:
        """Validate SLA under specific load.

        Args:
            load_multiplier: Load multiplier
            duration_seconds: Test duration

        Returns:
            Validation results
        """
        # This is a placeholder for actual load testing
        # In practice, this would integrate with load testing tools

        self.set_load_multiplier(load_multiplier)

        # Simulate validation (actual implementation would run real load test)
        return {
            "load_multiplier": load_multiplier,
            "duration_seconds": duration_seconds,
            "target_p95_ms": self.target_p95_ms,
            "validation_status": "simulated",
            "message": "Use actual load testing tools for production validation",
        }

    def get_breach_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get breach summary for time period.

        Args:
            hours: Hours to look back

        Returns:
            Breach summary
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_breaches = [b for b in self.breaches if b.timestamp > cutoff]

        # Group by metric
        breaches_by_metric = {}
        for breach in recent_breaches:
            if breach.metric_name not in breaches_by_metric:
                breaches_by_metric[breach.metric_name] = []
            breaches_by_metric[breach.metric_name].append(breach)

        return {
            "period_hours": hours,
            "total_breaches": len(recent_breaches),
            "breaches_by_metric": {
                metric: len(breaches) for metric, breaches in breaches_by_metric.items()
            },
            "most_recent": (
                {
                    "metric": recent_breaches[-1].metric_name,
                    "target": recent_breaches[-1].target_value,
                    "actual": recent_breaches[-1].actual_value,
                    "timestamp": recent_breaches[-1].timestamp.isoformat(),
                }
                if recent_breaches
                else None
            ),
        }

    def export_sla_documentation(self) -> str:
        """Export SLA documentation.

        Returns:
            SLA documentation as markdown
        """
        lines = []
        lines.append("# SLA Documentation")
        lines.append("")
        lines.append("## Performance Targets")
        lines.append("")

        for target_name, target in self.sla_targets.items():
            lines.append(f"### {target.description}")
            lines.append(f"- **Metric**: {target.metric_name}")
            lines.append(f"- **Target**: {target.target_value} {target.unit}")
            if target.percentile:
                lines.append(f"- **Percentile**: P{int(target.percentile)}")
            lines.append("")

        lines.append("## Current Performance")
        lines.append("")

        report = self.get_sla_report()
        lines.append(f"- **Status**: {report['overall_status']}")
        lines.append(
            f"- **P95 Latency**: {report['p95_latency_ms']:.1f}ms (target: {report['p95_target_ms']}ms)"
        )
        lines.append(f"- **SLA Met**: {'✅ Yes' if report['sla_met'] else '❌ No'}")
        lines.append("")

        lines.append("## Compliance Guarantees")
        lines.append("")
        lines.append("- P95 latency <220ms under 2× nominal load")
        lines.append("- P99 latency <500ms under normal load")
        lines.append("- Average latency <100ms under normal load")
        lines.append("")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics.

        Returns:
            Statistics dictionary
        """
        report = self.get_sla_report()

        return {
            "total_measurements": len(self.latency_window.measurements),
            "total_breaches": len(self.breaches),
            "current_status": report["overall_status"],
            "sla_compliance": report["sla_met"],
            "p95_latency_ms": report["p95_latency_ms"],
            "p95_target_ms": self.target_p95_ms,
            "window_size_seconds": self.window_size_seconds,
        }
