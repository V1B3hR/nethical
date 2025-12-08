"""Performance Optimizer for Phase 3.5: Performance & Cost Optimization.

This module implements:
- Risk-based gating for selective detector invocation
- CPU usage tracking and optimization
- Performance metrics and monitoring
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
from enum import Enum


class DetectorTier(str, Enum):
    """Detector performance tiers."""

    FAST = "fast"  # Low-cost, always-on detectors
    STANDARD = "standard"  # Medium-cost detectors
    ADVANCED = "advanced"  # High-cost detectors, only for elevated risk
    PREMIUM = "premium"  # Very expensive detectors, only for critical cases


@dataclass
class DetectorMetrics:
    """Performance metrics for a detector."""

    name: str
    tier: DetectorTier
    total_invocations: int = 0
    total_cpu_time_ms: float = 0.0
    avg_cpu_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    skip_count: int = 0  # Times skipped due to gating

    def update(self, cpu_time_ms: float, was_cached: bool = False):
        """Update metrics with new invocation."""
        self.total_invocations += 1
        self.total_cpu_time_ms += cpu_time_ms
        self.avg_cpu_time_ms = self.total_cpu_time_ms / self.total_invocations

        if was_cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_skip(self):
        """Record that detector was skipped."""
        self.skip_count += 1

    def get_skip_rate(self) -> float:
        """Get skip rate (0-1)."""
        total = self.total_invocations + self.skip_count
        if total == 0:
            return 0.0
        return self.skip_count / total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "tier": self.tier.value,
            "total_invocations": self.total_invocations,
            "total_cpu_time_ms": self.total_cpu_time_ms,
            "avg_cpu_time_ms": self.avg_cpu_time_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "skip_count": self.skip_count,
            "skip_rate": self.get_skip_rate(),
        }


@dataclass
class ActionMetrics:
    """Performance metrics for action processing."""

    total_actions: int = 0
    total_cpu_time_ms: float = 0.0
    avg_cpu_time_ms: float = 0.0
    detector_invocations: int = 0
    avg_detectors_per_action: float = 0.0

    # Recent window for tracking improvements
    recent_cpu_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    def update(self, cpu_time_ms: float, detectors_invoked: int):
        """Update action metrics."""
        self.total_actions += 1
        self.total_cpu_time_ms += cpu_time_ms
        self.avg_cpu_time_ms = self.total_cpu_time_ms / self.total_actions

        self.detector_invocations += detectors_invoked
        self.avg_detectors_per_action = self.detector_invocations / self.total_actions

        self.recent_cpu_times.append(cpu_time_ms)

    def get_recent_avg_cpu_ms(self) -> float:
        """Get recent average CPU time."""
        if not self.recent_cpu_times:
            return 0.0
        return sum(self.recent_cpu_times) / len(self.recent_cpu_times)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_actions": self.total_actions,
            "total_cpu_time_ms": self.total_cpu_time_ms,
            "avg_cpu_time_ms": self.avg_cpu_time_ms,
            "detector_invocations": self.detector_invocations,
            "avg_detectors_per_action": self.avg_detectors_per_action,
            "recent_avg_cpu_ms": self.get_recent_avg_cpu_ms(),
        }


class PerformanceOptimizer:
    """Performance optimizer with risk-based gating and CPU tracking."""

    def __init__(
        self,
        target_cpu_reduction_pct: float = 30.0,
        risk_gate_thresholds: Optional[Dict[DetectorTier, float]] = None,
    ):
        """Initialize performance optimizer.

        Args:
            target_cpu_reduction_pct: Target CPU reduction percentage (default 30%)
            risk_gate_thresholds: Risk score thresholds for each detector tier
        """
        self.target_cpu_reduction_pct = target_cpu_reduction_pct

        # Default thresholds: higher tier = higher risk required
        self.risk_gate_thresholds = risk_gate_thresholds or {
            DetectorTier.FAST: 0.0,  # Always run
            DetectorTier.STANDARD: 0.25,  # Run if risk >= LOW
            DetectorTier.ADVANCED: 0.5,  # Run if risk >= HIGH
            DetectorTier.PREMIUM: 0.75,  # Run if risk >= ELEVATED
        }

        # Metrics tracking
        self.detector_metrics: Dict[str, DetectorMetrics] = {}
        self.action_metrics = ActionMetrics()

        # Registered detectors
        self.detector_registry: Dict[str, DetectorTier] = {}

        # Baseline tracking
        self.baseline_cpu_ms: Optional[float] = None
        self.baseline_established = False

    def register_detector(self, name: str, tier: DetectorTier):
        """Register a detector with its performance tier.

        Args:
            name: Detector name
            tier: Performance tier
        """
        self.detector_registry[name] = tier
        if name not in self.detector_metrics:
            self.detector_metrics[name] = DetectorMetrics(name=name, tier=tier)

    def should_invoke_detector(
        self, detector_name: str, risk_score: float, force: bool = False
    ) -> bool:
        """Determine if detector should be invoked based on risk gating.

        Args:
            detector_name: Name of the detector
            risk_score: Current risk score (0-1)
            force: Force invocation regardless of risk

        Returns:
            True if detector should be invoked
        """
        if force:
            return True

        # Unknown detectors are always invoked (conservative)
        if detector_name not in self.detector_registry:
            return True

        tier = self.detector_registry[detector_name]
        threshold = self.risk_gate_thresholds.get(tier, 0.0)

        should_invoke = risk_score >= threshold

        # Record skip if gated
        if not should_invoke:
            if detector_name in self.detector_metrics:
                self.detector_metrics[detector_name].record_skip()

        return should_invoke

    def track_detector_invocation(
        self, detector_name: str, cpu_time_ms: float, was_cached: bool = False
    ):
        """Track detector invocation metrics.

        Args:
            detector_name: Name of the detector
            cpu_time_ms: CPU time in milliseconds
            was_cached: Whether result was from cache
        """
        if detector_name not in self.detector_metrics:
            tier = self.detector_registry.get(detector_name, DetectorTier.STANDARD)
            self.detector_metrics[detector_name] = DetectorMetrics(
                name=detector_name, tier=tier
            )

        self.detector_metrics[detector_name].update(cpu_time_ms, was_cached)

    def track_action_processing(self, cpu_time_ms: float, detectors_invoked: int):
        """Track action processing metrics.

        Args:
            cpu_time_ms: Total CPU time for action
            detectors_invoked: Number of detectors invoked
        """
        self.action_metrics.update(cpu_time_ms, detectors_invoked)

        # Establish baseline if not set
        if not self.baseline_established and self.action_metrics.total_actions >= 100:
            self.baseline_cpu_ms = self.action_metrics.avg_cpu_time_ms
            self.baseline_established = True

    def get_cpu_reduction_pct(self) -> float:
        """Get current CPU reduction percentage compared to baseline.

        Returns:
            CPU reduction percentage (positive = improvement)
        """
        if not self.baseline_established or self.baseline_cpu_ms is None:
            return 0.0

        current_avg = self.action_metrics.get_recent_avg_cpu_ms()
        if self.baseline_cpu_ms == 0:
            return 0.0

        reduction = ((self.baseline_cpu_ms - current_avg) / self.baseline_cpu_ms) * 100
        return reduction

    def is_meeting_target(self) -> bool:
        """Check if CPU reduction target is being met.

        Returns:
            True if target reduction is achieved
        """
        return self.get_cpu_reduction_pct() >= self.target_cpu_reduction_pct

    def get_detector_stats(self) -> Dict[str, Any]:
        """Get detector performance statistics.

        Returns:
            Dictionary of detector statistics
        """
        stats = {
            "detectors": {},
            "summary": {
                "total_detectors": len(self.detector_metrics),
                "total_invocations": 0,
                "total_skips": 0,
                "total_cpu_time_ms": 0.0,
            },
        }

        for name, metrics in self.detector_metrics.items():
            stats["detectors"][name] = metrics.to_dict()
            stats["summary"]["total_invocations"] += metrics.total_invocations
            stats["summary"]["total_skips"] += metrics.skip_count
            stats["summary"]["total_cpu_time_ms"] += metrics.total_cpu_time_ms

        # Calculate overall skip rate
        total = stats["summary"]["total_invocations"] + stats["summary"]["total_skips"]
        stats["summary"]["overall_skip_rate"] = (
            stats["summary"]["total_skips"] / total if total > 0 else 0.0
        )

        return stats

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report.

        Returns:
            Performance report dictionary
        """
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_metrics": self.action_metrics.to_dict(),
            "detector_stats": self.get_detector_stats(),
            "optimization": {
                "baseline_cpu_ms": self.baseline_cpu_ms,
                "baseline_established": self.baseline_established,
                "current_cpu_reduction_pct": self.get_cpu_reduction_pct(),
                "target_cpu_reduction_pct": self.target_cpu_reduction_pct,
                "meeting_target": self.is_meeting_target(),
            },
        }

        return report

    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on current metrics.

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Check if target is met
        if not self.is_meeting_target():
            reduction = self.get_cpu_reduction_pct()
            suggestions.append(
                f"CPU reduction ({reduction:.1f}%) below target "
                f"({self.target_cpu_reduction_pct}%). "
                f"Consider raising risk thresholds for expensive detectors."
            )

        # Check for expensive detectors
        for name, metrics in self.detector_metrics.items():
            if metrics.avg_cpu_time_ms > 100:  # 100ms threshold
                suggestions.append(
                    f"Detector '{name}' has high average CPU time "
                    f"({metrics.avg_cpu_time_ms:.1f}ms). "
                    f"Consider caching or optimizing."
                )

            # Check cache efficiency
            if metrics.total_invocations > 10:
                cache_hit_rate = metrics.cache_hits / metrics.total_invocations
                if cache_hit_rate < 0.3:  # Less than 30% cache hit rate
                    suggestions.append(
                        f"Detector '{name}' has low cache hit rate "
                        f"({cache_hit_rate:.1%}). "
                        f"Review caching strategy."
                    )

        # Check detector invocation rate
        if self.action_metrics.avg_detectors_per_action > 10:
            suggestions.append(
                f"High average detectors per action "
                f"({self.action_metrics.avg_detectors_per_action:.1f}). "
                f"Consider more aggressive risk gating."
            )

        if not suggestions:
            suggestions.append("Performance optimization targets are being met.")

        return suggestions

    def adjust_thresholds(self, tier: DetectorTier, new_threshold: float):
        """Adjust risk threshold for a detector tier.

        Args:
            tier: Detector tier to adjust
            new_threshold: New risk threshold (0-1)
        """
        if 0.0 <= new_threshold <= 1.0:
            self.risk_gate_thresholds[tier] = new_threshold
