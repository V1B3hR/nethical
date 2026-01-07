"""Prometheus metrics exporter for Nethical threat detection system.

This module exports comprehensive metrics for all threat detectors including:
- Request counts by detector type
- Latency histograms (P50, P95, P99)
- Threat detection rates
- Error rates
- Cache hit/miss rates
- Model inference times
"""

from datetime import datetime, timezone
from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create stub classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def inc(self, value=1):
            pass
    
    class Histogram:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def observe(self, value):
            pass
    
    class Gauge:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def set(self, value):
            pass
        def inc(self, value=1):
            pass
        def dec(self, value=1):
            pass
    
    class Summary:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, **kwargs):
            return self
        def observe(self, value):
            pass
    
    class CollectorRegistry:
        def __init__(self):
            pass
    
    def generate_latest(registry):
        return b"# Prometheus client not installed\n"
    
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


class PrometheusMetrics:
    """Export metrics for all threat detectors.
    
    Metrics categories:
    - Request counts by detector type
    - Latency histograms (P50, P95, P99)
    - Threat detection rates
    - Error rates
    - Cache hit/miss rates
    - Model inference times
    - Queue metrics
    - Throughput summaries
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics.
        
        Args:
            registry: Optional Prometheus registry. If None, creates a new one.
        """
        self.registry = registry or CollectorRegistry()
        self.enabled = PROMETHEUS_AVAILABLE
        
        if not self.enabled:
            import warnings
            warnings.warn(
                "prometheus_client not installed. Metrics collection disabled. "
                "Install with: pip install prometheus-client>=0.19.0"
            )
            return
        
        # Request metrics
        self.requests_total = Counter(
            'nethical_requests_total',
            'Total threat detection requests',
            ['detector_type', 'status'],
            registry=self.registry
        )
        
        # Latency metrics
        self.request_latency = Histogram(
            'nethical_request_latency_seconds',
            'Request latency in seconds',
            ['detector_type'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        # Threat detection metrics
        self.threats_detected = Counter(
            'nethical_threats_detected_total',
            'Total threats detected',
            ['detector_type', 'threat_level', 'threat_category'],
            registry=self.registry
        )
        
        # Model inference metrics
        self.model_inference_time = Histogram(
            'nethical_model_inference_seconds',
            'Model inference time',
            ['model_name'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'nethical_cache_hits_total',
            'Cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'nethical_cache_misses_total',
            'Cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'nethical_errors_total',
            'Total errors',
            ['detector_type', 'error_type'],
            registry=self.registry
        )
        
        # Active detectors gauge
        self.active_detectors = Gauge(
            'nethical_active_detectors',
            'Number of active detector instances',
            ['detector_type'],
            registry=self.registry
        )
        
        # Queue metrics
        self.queue_size = Gauge(
            'nethical_queue_size',
            'Current queue size',
            ['queue_name'],
            registry=self.registry
        )
        
        # Throughput summary
        self.throughput = Summary(
            'nethical_throughput_requests_per_second',
            'Throughput in requests per second',
            ['detector_type'],
            registry=self.registry
        )
        
        # Additional metrics for comprehensive monitoring
        self.confidence_score = Histogram(
            'nethical_confidence_score',
            'Confidence scores for threat detections',
            ['detector_type'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry
        )
        
        self.false_positives = Counter(
            'nethical_false_positives_total',
            'Total false positives reported',
            ['detector_type'],
            registry=self.registry
        )
        
        self.detector_health = Gauge(
            'nethical_detector_health',
            'Detector health status (0=unhealthy, 1=healthy)',
            ['detector_type'],
            registry=self.registry
        )
    
    def track_request(self, detector_type: str, latency: float, status: str) -> None:
        """Track a single request.
        
        Args:
            detector_type: Type of detector used
            latency: Request latency in seconds
            status: Request status (success, failure, etc.)
        """
        if not self.enabled:
            return
        
        self.requests_total.labels(
            detector_type=detector_type,
            status=status
        ).inc()
        
        self.request_latency.labels(
            detector_type=detector_type
        ).observe(latency)
    
    def track_threat(
        self,
        detector_type: str,
        threat_level: str,
        category: str,
        confidence: Optional[float] = None
    ) -> None:
        """Track detected threat.
        
        Args:
            detector_type: Type of detector that found the threat
            threat_level: Severity level of threat
            category: Category of threat
            confidence: Optional confidence score (0.0-1.0)
        """
        if not self.enabled:
            return
        
        self.threats_detected.labels(
            detector_type=detector_type,
            threat_level=threat_level,
            threat_category=category
        ).inc()
        
        if confidence is not None:
            self.confidence_score.labels(
                detector_type=detector_type
            ).observe(confidence)
    
    def track_model_inference(self, model_name: str, inference_time: float) -> None:
        """Track model inference time.
        
        Args:
            model_name: Name of the model
            inference_time: Inference time in seconds
        """
        if not self.enabled:
            return
        
        self.model_inference_time.labels(
            model_name=model_name
        ).observe(inference_time)
    
    def track_cache(self, cache_type: str, hit: bool) -> None:
        """Track cache hit or miss.
        
        Args:
            cache_type: Type of cache
            hit: True for cache hit, False for miss
        """
        if not self.enabled:
            return
        
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()
    
    def track_error(self, detector_type: str, error_type: str) -> None:
        """Track an error.
        
        Args:
            detector_type: Type of detector where error occurred
            error_type: Type of error
        """
        if not self.enabled:
            return
        
        self.errors_total.labels(
            detector_type=detector_type,
            error_type=error_type
        ).inc()
    
    def set_active_detectors(self, detector_type: str, count: int) -> None:
        """Set number of active detectors.
        
        Args:
            detector_type: Type of detector
            count: Number of active instances
        """
        if not self.enabled:
            return
        
        self.active_detectors.labels(
            detector_type=detector_type
        ).set(count)
    
    def set_queue_size(self, queue_name: str, size: int) -> None:
        """Set current queue size.
        
        Args:
            queue_name: Name of the queue
            size: Current size
        """
        if not self.enabled:
            return
        
        self.queue_size.labels(
            queue_name=queue_name
        ).set(size)
    
    def track_throughput(self, detector_type: str, requests_per_second: float) -> None:
        """Track throughput.
        
        Args:
            detector_type: Type of detector
            requests_per_second: Throughput measurement
        """
        if not self.enabled:
            return
        
        self.throughput.labels(
            detector_type=detector_type
        ).observe(requests_per_second)
    
    def track_false_positive(self, detector_type: str) -> None:
        """Track a false positive.
        
        Args:
            detector_type: Type of detector
        """
        if not self.enabled:
            return
        
        self.false_positives.labels(
            detector_type=detector_type
        ).inc()
    
    def set_detector_health(self, detector_type: str, healthy: bool) -> None:
        """Set detector health status.
        
        Args:
            detector_type: Type of detector
            healthy: True if healthy, False otherwise
        """
        if not self.enabled:
            return
        
        self.detector_health.labels(
            detector_type=detector_type
        ).set(1 if healthy else 0)
    
    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        if not self.enabled:
            return b"# Prometheus client not installed\n"
        
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get the content type for Prometheus metrics.
        
        Returns:
            Content type string
        """
        return CONTENT_TYPE_LATEST


# Global singleton instance
_prometheus_metrics: Optional[PrometheusMetrics] = None


def get_prometheus_metrics(registry: Optional[CollectorRegistry] = None) -> PrometheusMetrics:
    """Get or create the global Prometheus metrics instance.
    
    Args:
        registry: Optional Prometheus registry
        
    Returns:
        PrometheusMetrics instance
    """
    global _prometheus_metrics
    
    if _prometheus_metrics is None:
        _prometheus_metrics = PrometheusMetrics(registry=registry)
    
    return _prometheus_metrics
