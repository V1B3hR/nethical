"""Track conditions analyzer for the Adaptive Guardian.

Analyzes the "track conditions" (threat landscape) to recommend appropriate mode.
"""

import time
from collections import deque, defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional, Deque

from .guardian_modes import GuardianMode, get_mode_for_threat_score
from .tripwires import TripwireAlert


@dataclass
class ThreatAnalysis:
    """Result of threat analysis."""
    
    overall_threat_score: float  # 0.0-1.0
    recommended_mode: GuardianMode
    contributing_factors: Dict[str, float]
    anomaly_modules: List[str]
    alert_count: int
    error_rate: float
    response_time_trend: str  # "stable", "increasing", "decreasing"
    correlation_score: float
    timestamp: float


class TrackAnalyzer:
    """Analyzes track conditions to compute threat scores and recommend modes.
    
    Continuously monitors:
    - Recent alerts count and severity
    - Number of modules with anomalies
    - Error rate trends
    - Response time trends
    - Cross-module correlation score
    - External signals (maintenance windows, known attacks)
    """
    
    # Weights for threat score calculation
    WEIGHT_RECENT_ALERTS = 0.25
    WEIGHT_ANOMALY_MODULES = 0.20
    WEIGHT_ERROR_RATE = 0.20
    WEIGHT_RESPONSE_TIME = 0.15
    WEIGHT_CORRELATION = 0.15
    WEIGHT_EXTERNAL = 0.05
    
    # Thresholds
    HIGH_ALERT_COUNT_THRESHOLD = 10  # per minute
    HIGH_ERROR_RATE_THRESHOLD = 0.1  # 10%
    HIGH_ANOMALY_MODULE_THRESHOLD = 3
    
    # Window sizes
    ALERT_WINDOW_S = 60  # 1 minute
    METRIC_WINDOW_S = 300  # 5 minutes
    TREND_WINDOW_SIZE = 20
    
    def __init__(self):
        """Initialize track analyzer."""
        self._lock = Lock()
        
        # Alert tracking
        self._recent_alerts: Deque[TripwireAlert] = deque()
        
        # Module metrics
        self._module_metrics: Dict[str, Deque[tuple[float, float, bool]]] = defaultdict(
            lambda: deque(maxlen=100)
        )  # (timestamp, response_time, error)
        
        # Anomaly tracking
        self._anomaly_modules: set = set()
        
        # External signals
        self._maintenance_mode = False
        self._known_attack_active = False
        self._external_threat_level = 0.0  # 0.0-1.0
        
        # Correlation tracking
        self._correlation_events: Deque[tuple[float, str, str]] = deque(
            maxlen=1000
        )  # (timestamp, module1, module2)
        
        # Response time trends
        self._response_time_history: Deque[tuple[float, float]] = deque(
            maxlen=self.TREND_WINDOW_SIZE
        )
        
        # Statistics
        self._stats = {
            "analyses_performed": 0,
            "mode_recommendations": defaultdict(int),
            "avg_threat_score": 0.0,
        }
    
    def record_alert(self, alert: TripwireAlert) -> None:
        """Record a tripwire alert for analysis.
        
        Args:
            alert: Tripwire alert
        """
        with self._lock:
            self._recent_alerts.append(alert)
            
            # Mark module as anomalous if critical/high alert
            if alert.severity in ["CRITICAL", "HIGH"]:
                self._anomaly_modules.add(alert.module)
    
    def record_metric(
        self, module: str, response_time_ms: float, error: bool
    ) -> None:
        """Record module metrics for analysis.
        
        Args:
            module: Module name
            response_time_ms: Response time in milliseconds
            error: Whether an error occurred
        """
        with self._lock:
            now = time.time()
            self._module_metrics[module].append((now, response_time_ms, error))
            
            # Update response time history (global average)
            avg_rt = self._calculate_avg_response_time()
            self._response_time_history.append((now, avg_rt))
    
    def record_correlation(self, module1: str, module2: str) -> None:
        """Record a correlation event between modules.
        
        Args:
            module1: First module
            module2: Second module
        """
        with self._lock:
            now = time.time()
            self._correlation_events.append((now, module1, module2))
    
    def set_maintenance_mode(self, enabled: bool) -> None:
        """Set maintenance mode flag.
        
        During maintenance, threat scores are artificially lowered.
        
        Args:
            enabled: Whether maintenance mode is active
        """
        with self._lock:
            self._maintenance_mode = enabled
    
    def set_known_attack(self, active: bool) -> None:
        """Set known attack flag.
        
        When a known attack is detected externally, threat scores increase.
        
        Args:
            active: Whether a known attack is active
        """
        with self._lock:
            self._known_attack_active = active
    
    def set_external_threat_level(self, level: float) -> None:
        """Set external threat level from threat intelligence.
        
        Args:
            level: Threat level (0.0-1.0)
        """
        with self._lock:
            self._external_threat_level = max(0.0, min(1.0, level))
    
    def analyze(self) -> ThreatAnalysis:
        """Analyze current track conditions and compute threat score.
        
        Returns:
            ThreatAnalysis with threat score and recommended mode
        """
        with self._lock:
            now = time.time()
            
            # Clean old data
            self._clean_old_data(now)
            
            # Calculate individual factors
            alert_score = self._calculate_alert_score(now)
            anomaly_score = self._calculate_anomaly_score()
            error_score = self._calculate_error_score(now)
            response_time_score = self._calculate_response_time_score()
            correlation_score = self._calculate_correlation_score(now)
            external_score = self._calculate_external_score()
            
            # Compute overall threat score
            threat_score = (
                alert_score * self.WEIGHT_RECENT_ALERTS
                + anomaly_score * self.WEIGHT_ANOMALY_MODULES
                + error_score * self.WEIGHT_ERROR_RATE
                + response_time_score * self.WEIGHT_RESPONSE_TIME
                + correlation_score * self.WEIGHT_CORRELATION
                + external_score * self.WEIGHT_EXTERNAL
            )
            
            # Clamp to valid range
            threat_score = max(0.0, min(1.0, threat_score))
            
            # Get recommended mode
            recommended_mode = get_mode_for_threat_score(threat_score)
            
            # Build contributing factors
            contributing_factors = {
                "recent_alerts": alert_score,
                "anomaly_modules": anomaly_score,
                "error_rate": error_score,
                "response_time": response_time_score,
                "correlation": correlation_score,
                "external_signals": external_score,
            }
            
            # Get response time trend
            rt_trend = self._get_response_time_trend()
            
            # Count recent alerts
            alert_count = len(
                [a for a in self._recent_alerts if now - a.timestamp < self.ALERT_WINDOW_S]
            )
            
            # Calculate current error rate
            error_rate = self._calculate_current_error_rate(now)
            
            # Update statistics
            self._stats["analyses_performed"] += 1
            self._stats["mode_recommendations"][recommended_mode.value] += 1
            prev_avg = self._stats["avg_threat_score"]
            count = self._stats["analyses_performed"]
            self._stats["avg_threat_score"] = (
                prev_avg * (count - 1) + threat_score
            ) / count
            
            return ThreatAnalysis(
                overall_threat_score=threat_score,
                recommended_mode=recommended_mode,
                contributing_factors=contributing_factors,
                anomaly_modules=list(self._anomaly_modules),
                alert_count=alert_count,
                error_rate=error_rate,
                response_time_trend=rt_trend,
                correlation_score=correlation_score,
                timestamp=now,
            )
    
    def _clean_old_data(self, now: float) -> None:
        """Remove old data from tracking structures."""
        # Clean alerts
        cutoff_alerts = now - self.ALERT_WINDOW_S
        while self._recent_alerts and self._recent_alerts[0].timestamp < cutoff_alerts:
            self._recent_alerts.popleft()
        
        # Clean correlation events
        cutoff_correlation = now - self.METRIC_WINDOW_S
        while (
            self._correlation_events
            and self._correlation_events[0][0] < cutoff_correlation
        ):
            self._correlation_events.popleft()
        
        # Reset anomaly modules if no recent alerts
        if not self._recent_alerts:
            self._anomaly_modules.clear()
    
    def _calculate_alert_score(self, now: float) -> float:
        """Calculate score based on recent alerts."""
        recent = [
            a for a in self._recent_alerts if now - a.timestamp < self.ALERT_WINDOW_S
        ]
        
        if not recent:
            return 0.0
        
        # Count by severity
        critical_count = sum(1 for a in recent if a.severity == "CRITICAL")
        high_count = sum(1 for a in recent if a.severity == "HIGH")
        medium_count = sum(1 for a in recent if a.severity == "MEDIUM")
        
        # Weighted score
        weighted_count = critical_count * 3 + high_count * 2 + medium_count * 1
        
        # Normalize (10+ weighted alerts = max score)
        return min(weighted_count / 10.0, 1.0)
    
    def _calculate_anomaly_score(self) -> float:
        """Calculate score based on anomalous modules."""
        count = len(self._anomaly_modules)
        
        # Normalize (3+ anomalous modules = max score)
        return min(count / self.HIGH_ANOMALY_MODULE_THRESHOLD, 1.0)
    
    def _calculate_error_score(self, now: float) -> float:
        """Calculate score based on error rate."""
        error_rate = self._calculate_current_error_rate(now)
        
        # Normalize (10%+ error rate = max score)
        return min(error_rate / self.HIGH_ERROR_RATE_THRESHOLD, 1.0)
    
    def _calculate_current_error_rate(self, now: float) -> float:
        """Calculate current error rate across all modules."""
        cutoff = now - self.METRIC_WINDOW_S
        
        total_requests = 0
        error_requests = 0
        
        for module_data in self._module_metrics.values():
            for timestamp, _, error in module_data:
                if timestamp >= cutoff:
                    total_requests += 1
                    if error:
                        error_requests += 1
        
        return error_requests / total_requests if total_requests > 0 else 0.0
    
    def _calculate_response_time_score(self) -> float:
        """Calculate score based on response time trend."""
        trend = self._get_response_time_trend()
        
        if trend == "stable":
            return 0.0
        elif trend == "increasing":
            # Check magnitude of increase
            if len(self._response_time_history) < 2:
                return 0.0
            
            first_half = [rt for _, rt in list(self._response_time_history)[: len(self._response_time_history) // 2]]
            second_half = [rt for _, rt in list(self._response_time_history)[len(self._response_time_history) // 2 :]]
            
            if not first_half or not second_half:
                return 0.0
            
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            if avg_first == 0:
                return 0.0
            
            increase_ratio = (avg_second - avg_first) / avg_first
            
            # 50%+ increase = max score
            return min(increase_ratio / 0.5, 1.0)
        else:  # decreasing
            return 0.0  # Decreasing is good
    
    def _get_response_time_trend(self) -> str:
        """Determine response time trend."""
        if len(self._response_time_history) < 10:
            return "stable"
        
        # Simple linear trend detection
        recent = list(self._response_time_history)
        first_half_avg = sum(rt for _, rt in recent[: len(recent) // 2]) / (
            len(recent) // 2
        )
        second_half_avg = sum(rt for _, rt in recent[len(recent) // 2 :]) / (
            len(recent) - len(recent) // 2
        )
        
        if second_half_avg > first_half_avg * 1.2:  # 20% increase
            return "increasing"
        elif second_half_avg < first_half_avg * 0.8:  # 20% decrease
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time across all modules."""
        total_time = 0.0
        count = 0
        
        for module_data in self._module_metrics.values():
            for _, response_time, _ in module_data:
                total_time += response_time
                count += 1
        
        return total_time / count if count > 0 else 0.0
    
    def _calculate_correlation_score(self, now: float) -> float:
        """Calculate score based on cross-module correlations."""
        cutoff = now - self.METRIC_WINDOW_S
        recent_correlations = [
            c for c in self._correlation_events if c[0] >= cutoff
        ]
        
        if not recent_correlations:
            return 0.0
        
        # Count unique module pairs
        unique_pairs = set(
            (min(m1, m2), max(m1, m2)) for _, m1, m2 in recent_correlations
        )
        
        # Normalize (5+ unique correlation pairs = max score)
        return min(len(unique_pairs) / 5.0, 1.0)
    
    def _calculate_external_score(self) -> float:
        """Calculate score based on external signals."""
        if self._maintenance_mode:
            # During maintenance, reduce threat score
            return -0.5  # Can go negative to reduce overall score
        
        if self._known_attack_active:
            return 1.0  # Max score
        
        return self._external_threat_level
    
    def get_statistics(self) -> Dict:
        """Get analyzer statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                **self._stats,
                "modules_tracked": len(self._module_metrics),
                "anomaly_modules": list(self._anomaly_modules),
                "recent_alert_count": len(self._recent_alerts),
                "correlation_events": len(self._correlation_events),
                "maintenance_mode": self._maintenance_mode,
                "known_attack_active": self._known_attack_active,
                "external_threat_level": self._external_threat_level,
            }
