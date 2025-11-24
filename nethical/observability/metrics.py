"""Metrics Collection System

Prometheus-compatible metrics for governance actions and violations.
Includes:
- actions_total: Counter for all governance actions
- violations_total: Counter for detected violations
- action_latency: Histogram for action processing latency
- violation_latency: Histogram for violation detection latency
"""

from __future__ import annotations

import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# Optional Prometheus client import
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class MetricValue:
    """Simple metric value container."""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Centralized metrics collection for governance system."""
    
    def __init__(self, enable_prometheus: bool = True):
        """Initialize metrics collector.
        
        Args:
            enable_prometheus: Whether to use Prometheus client if available
        """
        self._lock = threading.RLock()
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        # In-memory metric storage (fallback)
        self._counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._histograms: Dict[str, list] = defaultdict(list)
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        else:
            self.registry = None
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metric objects."""
        if not self.enable_prometheus:
            return
        
        # Actions counter
        self.actions_total = Counter(
            'nethical_actions_total',
            'Total number of governance actions evaluated',
            ['action_type', 'decision', 'region'],
            registry=self.registry
        )
        
        # Violations counter
        self.violations_total = Counter(
            'nethical_violations_total',
            'Total number of violations detected',
            ['violation_type', 'severity', 'detector'],
            registry=self.registry
        )
        
        # Action latency histogram
        self.action_latency = Histogram(
            'nethical_action_latency_seconds',
            'Latency for processing governance actions',
            ['action_type'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        # Violation detection latency histogram
        self.violation_latency = Histogram(
            'nethical_violation_detection_latency_seconds',
            'Latency for violation detection',
            ['detector_type'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        # Active sessions gauge
        self.active_sessions = Gauge(
            'nethical_active_sessions',
            'Number of active governance sessions',
            registry=self.registry
        )
        
        # Error rate gauge
        self.error_rate = Gauge(
            'nethical_error_rate',
            'Current error rate',
            ['component'],
            registry=self.registry
        )
    
    def record_action(self, 
                     action_type: str,
                     decision: str,
                     region: str = "GLOBAL",
                     latency_seconds: Optional[float] = None) -> None:
        """Record a governance action.
        
        Args:
            action_type: Type of action (e.g., 'api_call', 'file_access')
            decision: Decision made (e.g., 'ALLOW', 'DENY')
            region: Region where action occurred
            latency_seconds: Optional processing latency
        """
        with self._lock:
            # In-memory tracking
            key = f"{action_type}:{decision}:{region}"
            self._counters['actions'][key] += 1
            
            # Prometheus
            if self.enable_prometheus:
                self.actions_total.labels(
                    action_type=action_type,
                    decision=decision,
                    region=region
                ).inc()
                
                if latency_seconds is not None:
                    self.action_latency.labels(action_type=action_type).observe(latency_seconds)
    
    def record_violation(self,
                        violation_type: str,
                        severity: str,
                        detector: str,
                        latency_seconds: Optional[float] = None) -> None:
        """Record a detected violation.
        
        Args:
            violation_type: Type of violation detected
            severity: Severity level
            detector: Name of detector that found it
            latency_seconds: Optional detection latency
        """
        with self._lock:
            # In-memory tracking
            key = f"{violation_type}:{severity}:{detector}"
            self._counters['violations'][key] += 1
            
            # Prometheus
            if self.enable_prometheus:
                self.violations_total.labels(
                    violation_type=violation_type,
                    severity=severity,
                    detector=detector
                ).inc()
                
                if latency_seconds is not None:
                    self.violation_latency.labels(detector_type=detector).observe(latency_seconds)
    
    def record_latency(self,
                      metric_name: str,
                      latency_seconds: float,
                      labels: Optional[Dict[str, str]] = None) -> None:
        """Record a latency measurement.
        
        Args:
            metric_name: Name of the metric
            latency_seconds: Latency value in seconds
            labels: Optional labels for the metric
        """
        with self._lock:
            self._histograms[metric_name].append(latency_seconds)
            
            # Keep only recent values (last 1000)
            if len(self._histograms[metric_name]) > 1000:
                self._histograms[metric_name] = self._histograms[metric_name][-1000:]
    
    def set_gauge(self,
                 metric_name: str,
                 value: float,
                 labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value.
        
        Args:
            metric_name: Name of the gauge
            value: Value to set
            labels: Optional labels
        """
        with self._lock:
            label_key = str(labels) if labels else "default"
            self._gauges[metric_name][label_key] = value
            
            # Update Prometheus if available
            if self.enable_prometheus:
                if metric_name == 'active_sessions':
                    self.active_sessions.set(value)
                elif metric_name == 'error_rate' and labels and 'component' in labels:
                    self.error_rate.labels(component=labels['component']).set(value)
    
    def get_counter(self, counter_name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value.
        
        Args:
            counter_name: Name of the counter
            labels: Optional labels to filter by
            
        Returns:
            Current counter value
        """
        with self._lock:
            if labels:
                # Build key from labels
                key_parts = [f"{k}:{v}" for k, v in sorted(labels.items())]
                key = ":".join(key_parts)
                return self._counters[counter_name].get(key, 0)
            else:
                # Sum all values for this counter
                return sum(self._counters[counter_name].values())
    
    def get_histogram_stats(self, histogram_name: str) -> Dict[str, float]:
        """Get statistics for a histogram metric.
        
        Args:
            histogram_name: Name of the histogram
            
        Returns:
            Dictionary with min, max, mean, median, p95, p99
        """
        with self._lock:
            values = self._histograms.get(histogram_name, [])
            
            if not values:
                return {'count': 0}
            
            sorted_values = sorted(values)
            count = len(sorted_values)
            
            return {
                'count': count,
                'min': sorted_values[0],
                'max': sorted_values[-1],
                'mean': sum(sorted_values) / count,
                'median': sorted_values[count // 2],
                'p95': sorted_values[int(count * 0.95)],
                'p99': sorted_values[int(count * 0.99)] if count > 1 else sorted_values[-1]
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics.
        
        Returns:
            Dictionary with all metrics data
        """
        with self._lock:
            metrics = {
                'counters': {},
                'histograms': {},
                'gauges': {}
            }
            
            # Counters
            for counter_name, values in self._counters.items():
                metrics['counters'][counter_name] = dict(values)
            
            # Histograms
            for hist_name in self._histograms.keys():
                metrics['histograms'][hist_name] = self.get_histogram_stats(hist_name)
            
            # Gauges
            for gauge_name, values in self._gauges.items():
                metrics['gauges'][gauge_name] = dict(values)
            
            return metrics
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()


# Global singleton instance
_metrics_collector: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics_collector(enable_prometheus: bool = True) -> MetricsCollector:
    """Get or create the global metrics collector instance.
    
    Args:
        enable_prometheus: Whether to enable Prometheus metrics
        
    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    
    if _metrics_collector is None:
        with _metrics_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector(enable_prometheus=enable_prometheus)
    
    return _metrics_collector


# Convenience functions
def record_action(action_type: str, decision: str, region: str = "GLOBAL",
                 latency_seconds: Optional[float] = None) -> None:
    """Record a governance action (convenience function)."""
    collector = get_metrics_collector()
    collector.record_action(action_type, decision, region, latency_seconds)


def record_violation(violation_type: str, severity: str, detector: str,
                    latency_seconds: Optional[float] = None) -> None:
    """Record a violation (convenience function)."""
    collector = get_metrics_collector()
    collector.record_violation(violation_type, severity, detector, latency_seconds)
