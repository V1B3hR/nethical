"""
MLOps Monitoring and Alerting Module (Enhanced)

This module provides advanced monitoring, observability, and alerting utilities
for machine learning models, pipelines, and supporting services.

Key Features:
- Configurable monitoring via MonitoringConfig
- Thread-safe alert & metric collectors with retention policies
- Pluggable alert handlers (email, slack, webhook, custom)
- Async / threaded alert dispatch with retry & backoff
- Structured JSON logging (optional)
- Model prediction logging with automatic SLA & latency tracking
- Extensible drift detection framework (mean, KS-test, PSI)
- Prometheus & OpenTelemetry integration (optional, auto-detected)
- Persistence to JSONL for metrics, predictions, alerts (batched)
- Metric summaries: min/max/mean/percentiles/rates
- Health snapshot & diagnostics
- Context managers & decorators for timing instrumentation
- Backward compatible legacy functions (setup_logger, log_event, ...)

NOTE:
This file is an enhanced version tailored for a scalable "nethical" system.
You may disable certain features via config flags if not required.

Author: nethical system enhancement
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import random
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
)

# Optional dependencies guarded
try:
    from prometheus_client import Counter, Gauge, Histogram  # type: ignore

    _PROM_ENABLED = True
except Exception:
    _PROM_ENABLED = False

try:
    from scipy import stats  # type: ignore

    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False

# OpenTelemetry placeholders
try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.metrics import get_meter  # type: ignore

    _OTEL_ENABLED = True
except Exception:
    _OTEL_ENABLED = False

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------


def _setup_root_logging(structured_json: bool):
    """Configure root logging only once."""
    logger = logging.getLogger()
    if getattr(logger, "_nethical_configured", False):
        return
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        if structured_json:
            formatter = logging.Formatter(
                '{"timestamp":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}'
            )
        else:
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger._nethical_configured = True


# ---------------------------------------------------------------------------
# Enums & Data Structures
# ---------------------------------------------------------------------------


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
        }


@dataclass
class MetricPoint:
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


@dataclass
class MetricDefinition:
    name: str
    metric_type: MetricType
    description: str = ""
    unit: Optional[str] = None
    histogram_buckets: Optional[List[float]] = None


@dataclass
class MonitoringConfig:
    model_name: str
    # Retention
    metric_retention_hours: int = 24
    prediction_retention: int = 10000
    alert_retention: int = 2000
    # SLA thresholds
    latency_sla_ms: float = 1000.0
    error_rate_sla: float = 0.05
    # Persistence
    enable_persistence: bool = True
    persistence_dir: Path = Path("logs/mlops")
    persistence_batch_size: int = 50
    # Logging
    structured_logging: bool = False
    # Prometheus / OTEL
    enable_prometheus: bool = True
    enable_opentelemetry: bool = True
    # Alert dispatch
    alert_dispatch_workers: int = 4
    alert_dispatch_timeout: float = 10.0
    max_alert_handlers: int = 32
    handler_retry_attempts: int = 3
    handler_retry_backoff: float = 1.5  # exponential backoff factor
    # Drift detection defaults
    drift_alert_severity: AlertSeverity = AlertSeverity.WARNING
    # Percentiles
    latency_percentiles: Tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)
    # Rate window
    rate_window_minutes: int = 15
    # Flush interval (seconds) for persistence
    flush_interval_sec: int = 30
    # Write asynchronously
    async_persistence: bool = True


# ---------------------------------------------------------------------------
# Protocols / Interfaces
# ---------------------------------------------------------------------------


class DriftDetector(Protocol):
    def detect(
        self,
        current: Dict[str, Any],
        baseline: Dict[str, Any],
        feature: str,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns a dict with drift info if drift is detected, else None.
        """
        ...


# ---------------------------------------------------------------------------
# Drift Detectors
# ---------------------------------------------------------------------------


class SimpleMeanDriftDetector:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def detect(
        self, current: Dict[str, Any], baseline: Dict[str, Any], feature: str, **_
    ) -> Optional[Dict[str, Any]]:
        if feature not in current or feature not in baseline:
            return None
        b = baseline[feature]
        c = current[feature]
        if b == 0:
            return None
        drift_ratio = abs(c - b) / abs(b)
        if drift_ratio > self.threshold:
            return {
                "feature": feature,
                "type": "mean_ratio",
                "baseline": b,
                "current": c,
                "ratio": drift_ratio,
                "threshold": self.threshold,
            }
        return None


class KSTestDriftDetector:
    """Requires distribution samples, not just summary stats. Uses SciPy KS-test if available."""

    def __init__(self, p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold

    def detect(
        self, current: Dict[str, Any], baseline: Dict[str, Any], feature: str, **_
    ) -> Optional[Dict[str, Any]]:
        if not _SCIPY_AVAILABLE:
            return None
        # Expect arrays under keys like f"{feature}_samples"
        cur_key = f"{feature}_samples"
        base_key = f"{feature}_baseline_samples"
        if cur_key not in current or base_key not in baseline:
            return None
        cur_samples = current[cur_key]
        base_samples = baseline[base_key]
        if not isinstance(cur_samples, (list, tuple)) or not isinstance(
            base_samples, (list, tuple)
        ):
            return None
        if len(cur_samples) < 5 or len(base_samples) < 5:
            return None
        stat, p = stats.ks_2samp(base_samples, cur_samples)
        if p < self.p_value_threshold:
            return {
                "feature": feature,
                "type": "ks_test",
                "ks_stat": stat,
                "p_value": p,
                "p_value_threshold": self.p_value_threshold,
            }
        return None


class PopulationStabilityIndexDetector:
    """PSI-based detector for bucketed distributions: expects 'hist' arrays or bin percentages."""

    def __init__(self, psi_threshold: float = 0.2):
        self.psi_threshold = psi_threshold

    def detect(
        self, current: Dict[str, Any], baseline: Dict[str, Any], feature: str, **_
    ) -> Optional[Dict[str, Any]]:
        cur_key = f"{feature}_hist"
        base_key = f"{feature}_baseline_hist"
        if cur_key not in current or base_key not in baseline:
            return None
        cur_hist = current[cur_key]
        base_hist = baseline[base_key]
        if len(cur_hist) != len(base_hist) or len(cur_hist) == 0:
            return None

        def _safe(p):
            return max(p, 1e-12)

        psi = 0.0
        for c, b in zip(cur_hist, base_hist):
            c = _safe(c)
            b = _safe(b)
            psi += (c - b) * math.log(c / b)

        if psi > self.psi_threshold:
            return {
                "feature": feature,
                "type": "psi",
                "psi": psi,
                "threshold": self.psi_threshold,
            }
        return None


# ---------------------------------------------------------------------------
# Alert Manager
# ---------------------------------------------------------------------------


class AlertManager:
    """
    Thread-safe alert management system with async dispatch & retry.

    Handlers are simple callables: handler(alert: Alert) -> None
    """

    def __init__(self, config: MonitoringConfig):
        self._config = config
        self._alerts: Deque[Alert] = deque(maxlen=config.alert_retention)
        self._handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        self._dispatch_queue: "queue.Queue[Alert]" = queue.Queue()
        self._workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._start_workers()

    def _start_workers(self):
        for i in range(self._config.alert_dispatch_workers):
            t = threading.Thread(target=self._worker_loop, name=f"alert-dispatch-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                alert = self._dispatch_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._dispatch(alert)
            self._dispatch_queue.task_done()

    def register_handler(self, handler: Callable[[Alert], None]):
        with self._lock:
            if len(self._handlers) >= self._config.max_alert_handlers:
                raise RuntimeError("Maximum number of alert handlers reached")
            self._handlers.append(handler)

    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: Optional[str] = None,
        **metadata,
    ) -> Alert:
        import uuid

        alert = Alert(
            alert_id=str(uuid.uuid4())[:10],
            severity=severity,
            title=title,
            message=message,
            source=source,
            metadata=metadata,
        )
        with self._lock:
            self._alerts.append(alert)
        # Enqueue for async dispatch
        self._dispatch_queue.put(alert)
        return alert

    def _dispatch(self, alert: Alert):
        for handler in list(self._handlers):
            attempt = 0
            while attempt < self._config.handler_retry_attempts:
                try:
                    handler(alert)
                    break
                except Exception as e:
                    backoff = (self._config.handler_retry_backoff**attempt) + random.uniform(0, 0.1)
                    logging.warning(
                        f"Alert handler error (attempt {attempt+1}/{self._config.handler_retry_attempts}): {e}. Backoff {backoff:.2f}s"
                    )
                    attempt += 1
                    if attempt < self._config.handler_retry_attempts:
                        time.sleep(backoff)

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        since: Optional[datetime] = None,
        acknowledged: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[Alert]:
        with self._lock:
            alerts = list(self._alerts)
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        alerts = sorted(alerts, key=lambda a: a.timestamp, reverse=True)
        if limit is not None:
            alerts = alerts[:limit]
        return alerts

    def acknowledge(self, alert_id: str) -> bool:
        with self._lock:
            for a in self._alerts:
                if a.alert_id == alert_id:
                    a.acknowledged = True
                    return True
        return False

    def stop(self):
        self._stop_event.set()
        for t in self._workers:
            t.join(timeout=1.0)

    def __del__(self):
        self.stop()


# ---------------------------------------------------------------------------
# Metrics Collector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """
    Thread-safe metric collector with retention, definitions & summaries.
    """

    def __init__(self, retention_hours: int):
        self._retention_hours = retention_hours
        self._metrics: Dict[str, Deque[MetricPoint]] = {}
        self._definitions: Dict[str, MetricDefinition] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

        # Prometheus registries (only created if imported)
        self._prom_counters: Dict[str, Any] = {}
        self._prom_gauges: Dict[str, Any] = {}
        self._prom_histograms: Dict[str, Any] = {}

        if _OTEL_ENABLED:
            self._tracer = trace.get_tracer(__name__)
            self._meter = get_meter(__name__)
        else:
            self._tracer = None
            self._meter = None

    def register_metric(self, definition: MetricDefinition):
        with self._lock:
            if definition.name not in self._definitions:
                self._definitions[definition.name] = definition
                if _PROM_ENABLED:
                    if definition.metric_type == MetricType.COUNTER:
                        self._prom_counters[definition.name] = Counter(
                            definition.name,
                            definition.description,
                        )
                    elif definition.metric_type == MetricType.GAUGE:
                        self._prom_gauges[definition.name] = Gauge(
                            definition.name,
                            definition.description,
                        )
                    elif definition.metric_type == MetricType.HISTOGRAM:
                        self._prom_histograms[definition.name] = Histogram(
                            definition.name,
                            definition.description,
                            buckets=definition.histogram_buckets
                            or (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
                        )

    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
    ):
        tags = tags or {}
        point = MetricPoint(name=name, value=value, metric_type=metric_type, tags=tags)
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=20000)
            self._metrics[name].append(point)

            # Prometheus update
            if _PROM_ENABLED:
                if metric_type == MetricType.COUNTER and name in self._prom_counters:
                    self._prom_counters[name].inc(value)
                elif metric_type == MetricType.GAUGE and name in self._prom_gauges:
                    self._prom_gauges[name].set(value)
                elif metric_type == MetricType.HISTOGRAM and name in self._prom_histograms:
                    self._prom_histograms[name].observe(value)

            # Cleanup hourly
            if time.time() - self._last_cleanup > 3600:
                self._cleanup()

    def _cleanup(self):
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._retention_hours)
        for name, dq in self._metrics.items():
            while dq and dq[0].timestamp < cutoff:
                dq.popleft()
        self._last_cleanup = time.time()

    def get_metrics(self, name: str, since: Optional[datetime] = None) -> List[MetricPoint]:
        with self._lock:
            if name not in self._metrics:
                return []
            metrics = list(self._metrics[name])
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        return metrics

    def summary(
        self,
        name: str,
        window_minutes: int = 60,
        percentiles: Iterable[float] = (0.5, 0.9, 0.95),
    ) -> Dict[str, Any]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        points = self.get_metrics(name, since=cutoff)
        if not points:
            return {"count": 0}

        values = [p.value for p in points]
        values_sorted = sorted(values)
        result = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1],
        }
        for q in percentiles:
            idx = int(q * (len(values_sorted) - 1))
            result[f"p{int(q*100)}"] = values_sorted[idx]
        return result

    def rate_per_minute(self, name: str, window_minutes: int) -> float:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        n = len(self.get_metrics(name, since=cutoff))
        return n / window_minutes if window_minutes > 0 else float("nan")

    def definitions(self) -> Dict[str, MetricDefinition]:
        with self._lock:
            return dict(self._definitions)


# ---------------------------------------------------------------------------
# Model Monitor
# ---------------------------------------------------------------------------


class ModelMonitor:
    """
    Core monitoring orchestrator.

    Responsibilities:
    - Logging predictions
    - Recording metrics
    - Detecting drift via pluggable detectors
    - Generating alerts on SLA violation or drift
    - Persistence (alerts, metrics, predictions)
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        _setup_root_logging(config.structured_logging)

        self._lock = threading.RLock()
        self.metrics = MetricsCollector(retention_hours=config.metric_retention_hours)
        self.alerts = AlertManager(config)
        self._predictions: Deque[Dict[str, Any]] = deque(maxlen=config.prediction_retention)

        # Register internal alert handler
        self.alerts.register_handler(self._default_alert_handler)

        # Persistence
        self._persist_buffers = {
            "predictions": [],
            "metrics": [],
            "alerts": [],
        }
        self._last_flush = time.time()
        self._persistence_lock = threading.RLock()
        self._stop_event = threading.Event()
        if config.enable_persistence:
            config.persistence_dir.mkdir(parents=True, exist_ok=True)
            if config.async_persistence:
                self._persistence_thread = threading.Thread(
                    target=self._persistence_loop, name="monitor-persistence", daemon=True
                )
                self._persistence_thread.start()

        # Drift detectors (can be extended)
        self._drift_detectors: List[DriftDetector] = [
            SimpleMeanDriftDetector(threshold=0.1),
            KSTestDriftDetector(p_value_threshold=0.05),
            PopulationStabilityIndexDetector(psi_threshold=0.2),
        ]

        # Metric definitions
        self.metrics.register_metric(
            MetricDefinition(
                name="prediction_latency_ms",
                metric_type=MetricType.HISTOGRAM,
                description="Prediction latency in milliseconds",
                unit="ms",
            )
        )
        self.metrics.register_metric(
            MetricDefinition(
                name="prediction_success_total",
                metric_type=MetricType.COUNTER,
                description="Count of successful predictions",
            )
        )
        self.metrics.register_metric(
            MetricDefinition(
                name="prediction_error_total",
                metric_type=MetricType.COUNTER,
                description="Count of errored predictions",
            )
        )

    # -----------------------------
    # Prediction Logging
    # -----------------------------

    def log_prediction(
        self,
        input_data: Any,
        prediction: Any,
        latency_ms: float,
        error: Optional[str] = None,
        **metadata,
    ):
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.config.model_name,
            "latency_ms": latency_ms,
            "error": error,
            "input": self._safe_str(input_data),
            "prediction": self._safe_str(prediction),
            "metadata": metadata,
        }
        with self._lock:
            self._predictions.append(record)

        # Metrics
        self.metrics.record(
            "prediction_latency_ms",
            latency_ms,
            metric_type=MetricType.HISTOGRAM,
            tags={"model": self.config.model_name},
        )
        if error:
            self.metrics.record(
                "prediction_error_total",
                1,
                metric_type=MetricType.COUNTER,
                tags={"model": self.config.model_name},
            )
        else:
            self.metrics.record(
                "prediction_success_total",
                1,
                metric_type=MetricType.COUNTER,
                tags={"model": self.config.model_name},
            )

        # SLA evaluation
        if latency_ms > self.config.latency_sla_ms:
            self.alerts.create_alert(
                severity=AlertSeverity.WARNING,
                title="Latency SLA Violation",
                message=f"Latency {latency_ms:.2f}ms exceeded SLA {self.config.latency_sla_ms}ms",
                source=self.config.model_name,
                latency_ms=latency_ms,
            )

        # Persistence buffer
        self._buffer_for_persistence("predictions", record)

    def log_model_metrics(self, **metrics: float):
        """
        Log arbitrary model performance metrics (accuracy=0.9, f1=0.8, etc.)
        """
        for name, val in metrics.items():
            metric_name = f"model_{name}"
            self.metrics.record(
                metric_name,
                float(val),
                metric_type=MetricType.GAUGE,
                tags={"model": self.config.model_name},
            )
            self._buffer_for_persistence(
                "metrics",
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": metric_name,
                    "value": float(val),
                    "model": self.config.model_name,
                },
            )

    # -----------------------------
    # Drift Detection
    # -----------------------------

    def run_drift_detection(
        self,
        current_distribution: Dict[str, Any],
        baseline_distribution: Dict[str, Any],
        features: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Iterates all detectors for given features. If any detector flags drift, an alert is emitted.
        Returns list of drift events.
        """
        if features is None:
            # Use intersection of keys
            features = set(baseline_distribution.keys()) & set(current_distribution.keys())

        drift_events = []
        for feature in features:
            for detector in self._drift_detectors:
                try:
                    result = detector.detect(current_distribution, baseline_distribution, feature)
                except Exception as e:
                    logging.debug(f"Drift detector error for {feature}: {e}")
                    continue
                if result:
                    drift_events.append(result)
                    self.alerts.create_alert(
                        severity=self.config.drift_alert_severity,
                        title="Data Drift Detected",
                        message=f"Feature '{feature}' drift: {result.get('type')}",
                        source=self.config.model_name,
                        **result,
                    )
                    # We can break on first detection per feature or continue to gather all
        return drift_events

    # -----------------------------
    # Dashboard / Snapshot
    # -----------------------------

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)
        pred_last_hour = [
            p
            for p in list(self._predictions)
            if datetime.fromisoformat(p["timestamp"]) >= one_hour_ago
        ]

        errors = [p for p in pred_last_hour if p["error"]]
        error_rate = len(errors) / len(pred_last_hour) if pred_last_hour else 0.0

        latency_summary = self.metrics.summary(
            "prediction_latency_ms",
            window_minutes=60,
            percentiles=self.config.latency_percentiles,
        )

        # Rolling SLA error rate evaluation over configured window
        window_summary = self._error_rate_window()

        active_alerts = self.alerts.get_alerts(
            since=now - timedelta(hours=24),
            acknowledged=False,
        )

        return {
            "model_name": self.config.model_name,
            "predictions_last_hour": len(pred_last_hour),
            "errors_last_hour": len(errors),
            "error_rate_last_hour": error_rate,
            "error_rate_window": window_summary,
            "latency_summary": latency_summary,
            "sla_violations": {
                "latency": (
                    latency_summary.get("max", 0) > self.config.latency_sla_ms
                    if latency_summary.get("count", 0)
                    else False
                ),
                "error_rate": window_summary.get("error_rate", 0) > self.config.error_rate_sla,
            },
            "active_alerts": len(active_alerts),
            "timestamp": now.isoformat(),
        }

    def _error_rate_window(self) -> Dict[str, Any]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.config.rate_window_minutes)
        preds = [
            p for p in list(self._predictions) if datetime.fromisoformat(p["timestamp"]) >= cutoff
        ]
        if not preds:
            return {"count": 0, "error_rate": 0.0}
        errors = sum(1 for p in preds if p["error"])
        return {
            "count": len(preds),
            "errors": errors,
            "error_rate": errors / len(preds),
            "window_minutes": self.config.rate_window_minutes,
        }

    def health(self) -> Dict[str, Any]:
        return {
            "model": self.config.model_name,
            "predictions_buffered": len(self._predictions),
            "alerts": len(self.alerts.get_alerts(limit=10000)),
            "persistence_backlog": {k: len(v) for k, v in self._persist_buffers.items()},
            "uptime_sec": self._uptime(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _uptime(self) -> float:
        if not hasattr(self, "_start_time"):
            self._start_time = time.time()
        return time.time() - self._start_time

    # -----------------------------
    # Persistence
    # -----------------------------

    def _buffer_for_persistence(self, category: str, record: Dict[str, Any]):
        if not self.config.enable_persistence:
            return
        with self._persistence_lock:
            self._persist_buffers[category].append(record)
        if not self.config.async_persistence:
            # Flush synchronously if batch size met
            self._maybe_flush(force=False)

    def _persistence_loop(self):
        while not self._stop_event.is_set():
            time.sleep(1.0)
            try:
                self._maybe_flush(force=False)
            except Exception as e:
                logging.error(f"Persistence loop error: {e}")

    def flush(self):
        """Force flush buffers."""
        self._maybe_flush(force=True)

    def _maybe_flush(self, force: bool):
        if not self.config.enable_persistence:
            return
        now = time.time()
        if not force and (now - self._last_flush < self.config.flush_interval_sec):
            # Only flush if any buffer exceeds batch size
            if not any(
                len(buf) >= self.config.persistence_batch_size
                for buf in self._persist_buffers.values()
            ):
                return

        with self._persistence_lock:
            for category, buffer in self._persist_buffers.items():
                if not buffer:
                    continue
                self._write_jsonl(category, buffer)
                buffer.clear()
            self._last_flush = time.time()

    def _write_jsonl(self, category: str, rows: List[Dict[str, Any]]):
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        file_path = self.config.persistence_dir / f"{category}_{date_str}.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # -----------------------------
    # Alert Handlers
    # -----------------------------

    def _default_alert_handler(self, alert: Alert):
        """
        Writes alerts to daily file + logs severity.
        """
        if not self.config.enable_persistence:
            pass
        else:
            self._buffer_for_persistence("alerts", alert.to_dict())

        level = (
            logging.ERROR
            if alert.severity in (AlertSeverity.ERROR, AlertSeverity.CRITICAL)
            else logging.WARNING
        )
        logging.log(
            level,
            f"[ALERT] {alert.severity.value.upper()} {alert.title} - {alert.message}",
        )

    # Example placeholders (user can implement actual send logic)
    def register_slack_handler(self, webhook_url: str):
        def slack_handler(alert: Alert):
            # Placeholder: Integrate requests.post(webhook_url, json=payload)
            logging.info(f"Slack handler placeholder for alert {alert.alert_id}")

        self.alerts.register_handler(slack_handler)

    def register_email_handler(self, to_addr: str, from_addr: str, smtp_server: str):
        def email_handler(alert: Alert):
            logging.info(f"Email handler placeholder: sending to {to_addr}")

        self.alerts.register_handler(email_handler)

    # -----------------------------
    # Timing Helpers
    # -----------------------------

    @contextmanager
    def timed_prediction(self, input_data: Any, **metadata):
        """
        Context manager to measure prediction latency automatically.

        Usage:
            with monitor.timed_prediction(input) as finish:
                pred = model.predict(input)
                finish(prediction=pred)
        """
        start = time.time()
        error_holder = {"err": None}

        def _finish(prediction=None, error: Optional[str] = None):
            latency_ms = (time.time() - start) * 1000.0
            self.log_prediction(
                input_data=input_data,
                prediction=prediction,
                latency_ms=latency_ms,
                error=error,
                **metadata,
            )

        try:
            yield _finish
        except Exception as e:
            error_holder["err"] = str(e)
            raise
        finally:
            if error_holder["err"]:
                _finish(prediction=None, error=error_holder["err"])

    def timed(self, name: str):
        """
        Decorator for timing arbitrary functions; records metric <name>_latency_ms.
        """
        metric_name = f"{name}_latency_ms"
        self.metrics.register_metric(
            MetricDefinition(
                name=metric_name,
                metric_type=MetricType.HISTOGRAM,
                description=f"Latency for {name}",
            )
        )

        def decorator(fn: Callable):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return fn(*args, **kwargs)
                finally:
                    latency_ms = (time.time() - start) * 1000.0
                    self.metrics.record(metric_name, latency_ms, MetricType.HISTOGRAM)

            return wrapper

        return decorator

    # -----------------------------
    # Utilities / Cleanup
    # -----------------------------

    def _safe_str(self, obj: Any, max_len: int = 1000) -> str:
        s = str(obj)
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s

    def close(self):
        self._stop_event.set()
        # Flush persistence
        try:
            self.flush()
        except Exception as e:
            logging.error(f"Error during flush on close: {e}")
        # Stop alerts manager workers
        self.alerts.stop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Legacy API (Backward Compatibility)
# ---------------------------------------------------------------------------

_LEGACY_LOGGER_NAME = "mlops"
_legacy_logger = None


def setup_logger(logfile="logs/mlops.log", to_console=True):
    """
    Legacy function (deprecated).
    Prefer using ModelMonitor + MonitoringConfig.

    Maintains old behavior for compatibility.
    """
    global _legacy_logger
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger = logging.getLogger(_LEGACY_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)
        if to_console:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(ch)
    _legacy_logger = logger
    logging.warning("setup_logger is deprecated. Use ModelMonitor instead.")
    return logger


def get_logger():
    """Legacy accessor."""
    logger = logging.getLogger(_LEGACY_LOGGER_NAME)
    if not logger.handlers:
        setup_logger()
    return logger


def log_event(event, **kwargs):
    """Legacy event logging."""
    logger = get_logger()
    logger.info(f"{event}: {json.dumps(kwargs)}" if kwargs else event)


# ---------------------------------------------------------------------------
# Demo / Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = MonitoringConfig(
        model_name="demo_model",
        structured_logging=False,
        enable_prometheus=False,
        enable_persistence=True,
    )
    monitor = ModelMonitor(cfg)
    print("Enhanced ModelMonitor initialized")

    # Simulate predictions
    for i in range(5):
        with monitor.timed_prediction({"x": i}) as finish:
            time.sleep(0.01 * i)
            finish(prediction={"y": i * 2})

    # Force a latency SLA violation
    monitor.log_prediction({"x": 999}, {"y": 999}, latency_ms=cfg.latency_sla_ms + 250)

    # Log some model metrics
    monitor.log_model_metrics(accuracy=0.92, f1=0.88)

    # Drift detection example
    base = {"feature_a": 100}
    cur = {"feature_a": 120}
    drift_events = monitor.run_drift_detection(cur, base, features=["feature_a"])
    print("Drift events:", drift_events)

    print("Dashboard snapshot:")
    print(json.dumps(monitor.get_dashboard_metrics(), indent=2))

    monitor.flush()
    monitor.close()
