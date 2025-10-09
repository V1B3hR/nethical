"""
MLOps Monitoring and Alerting Module

This module provides comprehensive monitoring, alerting, and observability
for machine learning models and pipelines.

Features:
- Model performance monitoring
- Data drift detection
- Prediction monitoring and logging
- Alert management (email, slack, webhooks)
- Dashboard metrics collection
- SLA tracking
"""

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Alert object"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'metadata': self.metadata,
            'acknowledged': self.acknowledged
        }


@dataclass
class Metric:
    """Metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }


class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self):
        self.alerts: deque = deque(maxlen=1000)
        self.handlers: List[Callable] = []
    
    def register_handler(self, handler: Callable[[Alert], None]):
        """Register an alert handler"""
        self.handlers.append(handler)
    
    def create_alert(self,
                     severity: AlertSeverity,
                     title: str,
                     message: str,
                     source: Optional[str] = None,
                     **metadata) -> Alert:
        """Create and dispatch an alert"""
        import uuid
        alert = Alert(
            alert_id=str(uuid.uuid4())[:8],
            severity=severity,
            title=title,
            message=message,
            source=source,
            metadata=metadata
        )
        
        self.alerts.append(alert)
        
        # Dispatch to handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
        
        return alert
    
    def get_alerts(self, 
                   severity: Optional[AlertSeverity] = None,
                   since: Optional[datetime] = None,
                   acknowledged: Optional[bool] = None) -> List[Alert]:
        """Get alerts with optional filtering"""
        alerts = list(self.alerts)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False


class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, deque] = {}
        self.retention_hours = retention_hours
        self._last_cleanup = time.time()
    
    def record(self, 
               name: str,
               value: float,
               metric_type: MetricType = MetricType.GAUGE,
               **tags):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags
        )
        
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=10000)
        
        self.metrics[name].append(metric)
        
        # Periodic cleanup
        if time.time() - self._last_cleanup > 3600:  # Every hour
            self._cleanup_old_metrics()
    
    def get_metrics(self,
                    name: str,
                    since: Optional[datetime] = None) -> List[Metric]:
        """Get metrics for a name"""
        if name not in self.metrics:
            return []
        
        metrics = list(self.metrics[name])
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        return metrics
    
    def get_summary(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get metric summary statistics"""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        metrics = self.get_metrics(name, since=cutoff)
        
        if not metrics:
            return {'count': 0}
        
        values = [m.value for m in metrics]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'latest': values[-1]
        }
    
    def _cleanup_old_metrics(self):
        """Remove old metrics beyond retention period"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        for name in self.metrics:
            metrics = self.metrics[name]
            # Remove old metrics
            while metrics and metrics[0].timestamp < cutoff:
                metrics.popleft()
        
        self._last_cleanup = time.time()


class ModelMonitor:
    """Monitor ML model performance and behavior"""
    
    def __init__(self, model_name: str, log_dir: Optional[Path] = None):
        self.model_name = model_name
        self.log_dir = Path(log_dir or "logs/mlops")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        
        # Register default alert handler
        self.alerts.register_handler(self._log_alert)
        
        # Prediction tracking
        self.predictions: deque = deque(maxlen=10000)
        
        # SLA thresholds
        self.latency_sla_ms = 1000  # 1 second
        self.error_rate_sla = 0.05  # 5%
    
    def log_prediction(self,
                       input_data: Any,
                       prediction: Any,
                       latency_ms: float,
                       error: Optional[str] = None,
                       **metadata):
        """Log a model prediction"""
        prediction_log = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'input': str(input_data),
            'prediction': str(prediction),
            'latency_ms': latency_ms,
            'error': error,
            'metadata': metadata
        }
        
        self.predictions.append(prediction_log)
        
        # Record metrics
        self.metrics.record('prediction_latency', latency_ms, 
                           MetricType.HISTOGRAM, model=self.model_name)
        
        if error:
            self.metrics.record('prediction_errors', 1, 
                               MetricType.COUNTER, model=self.model_name)
        else:
            self.metrics.record('prediction_success', 1, 
                               MetricType.COUNTER, model=self.model_name)
        
        # Check SLA violations
        if latency_ms > self.latency_sla_ms:
            self.alerts.create_alert(
                AlertSeverity.WARNING,
                "Latency SLA Violation",
                f"Prediction latency {latency_ms:.2f}ms exceeded SLA of {self.latency_sla_ms}ms",
                source=self.model_name,
                latency=latency_ms
            )
    
    def log_model_metrics(self,
                          accuracy: Optional[float] = None,
                          precision: Optional[float] = None,
                          recall: Optional[float] = None,
                          f1_score: Optional[float] = None,
                          **custom_metrics):
        """Log model performance metrics"""
        if accuracy is not None:
            self.metrics.record('model_accuracy', accuracy, 
                               MetricType.GAUGE, model=self.model_name)
        if precision is not None:
            self.metrics.record('model_precision', precision,
                               MetricType.GAUGE, model=self.model_name)
        if recall is not None:
            self.metrics.record('model_recall', recall,
                               MetricType.GAUGE, model=self.model_name)
        if f1_score is not None:
            self.metrics.record('model_f1', f1_score,
                               MetricType.GAUGE, model=self.model_name)
        
        for name, value in custom_metrics.items():
            self.metrics.record(f'model_{name}', value,
                               MetricType.GAUGE, model=self.model_name)
    
    def check_data_drift(self, 
                         current_distribution: Dict[str, Any],
                         baseline_distribution: Dict[str, Any],
                         threshold: float = 0.1) -> bool:
        """
        Simple data drift detection
        
        Args:
            current_distribution: Current data statistics
            baseline_distribution: Baseline data statistics
            threshold: Drift threshold
            
        Returns:
            True if drift detected
        """
        # Simple implementation - compare means
        drift_detected = False
        
        for key in baseline_distribution:
            if key in current_distribution:
                baseline_val = baseline_distribution[key]
                current_val = current_distribution[key]
                
                if baseline_val != 0:
                    drift = abs(current_val - baseline_val) / abs(baseline_val)
                    if drift > threshold:
                        drift_detected = True
                        self.alerts.create_alert(
                            AlertSeverity.WARNING,
                            "Data Drift Detected",
                            f"Feature '{key}' has drifted {drift:.2%} from baseline",
                            source=self.model_name,
                            feature=key,
                            drift_percentage=drift
                        )
        
        return drift_detected
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        # Get recent predictions
        recent_predictions = len([p for p in self.predictions 
                                 if datetime.fromisoformat(p['timestamp']) > 
                                    datetime.now() - timedelta(hours=1)])
        
        # Get recent errors
        recent_errors = len([p for p in self.predictions 
                            if p['error'] is not None and 
                               datetime.fromisoformat(p['timestamp']) > 
                               datetime.now() - timedelta(hours=1)])
        
        error_rate = recent_errors / recent_predictions if recent_predictions > 0 else 0
        
        # Get latency stats
        latency_summary = self.metrics.get_summary('prediction_latency', window_minutes=60)
        
        # Get recent alerts
        recent_alerts = self.alerts.get_alerts(
            since=datetime.now() - timedelta(hours=24),
            acknowledged=False
        )
        
        return {
            'model_name': self.model_name,
            'predictions_last_hour': recent_predictions,
            'errors_last_hour': recent_errors,
            'error_rate': error_rate,
            'latency_stats': latency_summary,
            'sla_violations': {
                'latency': latency_summary.get('max', 0) > self.latency_sla_ms if latency_summary.get('count', 0) > 0 else False,
                'error_rate': error_rate > self.error_rate_sla
            },
            'active_alerts': len(recent_alerts),
            'timestamp': datetime.now().isoformat()
        }
    
    def _log_alert(self, alert: Alert):
        """Default alert handler - log to file"""
        alert_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(alert_file, 'a') as f:
            json.dump(alert.to_dict(), f)
            f.write('\n')
        
        # Also log to console
        logging.log(
            logging.ERROR if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] else logging.WARNING,
            f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}"
        )


# Legacy functions for backward compatibility
def setup_logger(logfile="logs/mlops.log", to_console=True):
    """Legacy function - use ModelMonitor instead"""
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    logger = logging.getLogger("mlops")
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if logger is reused
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

        # Optional: Console handler
        if to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(console_handler)
    return logger


def get_logger():
    """Legacy function"""
    return logging.getLogger("mlops")


def log_event(event, **kwargs):
    """Legacy function"""
    logger = get_logger()
    # Serialize kwargs for clarity
    logger.info(f"{event}: {json.dumps(kwargs)}" if kwargs else event)


if __name__ == "__main__":
    # Demo usage
    monitor = ModelMonitor("demo_model")
    print("Model monitor initialized")
    print(f"Monitoring model: {monitor.model_name}")
