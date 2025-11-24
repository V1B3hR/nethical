"""Alert Rules Management

Defines and evaluates alert rules for:
- Latency thresholds
- Error rate thresholds
- Drift detection
- Quota saturation
"""

from __future__ import annotations

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert firing state."""
    ACTIVE = "active"
    PENDING = "pending"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    description: str
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]  # Function that evaluates metrics
    threshold: float
    duration: int  # Seconds the condition must be true before firing
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # State tracking
    first_triggered: Optional[float] = None
    last_evaluated: Optional[float] = None
    state: AlertState = AlertState.RESOLVED
    firing_count: int = 0


@dataclass
class Alert:
    """An active alert instance."""
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    state: AlertState = AlertState.ACTIVE
    resolved_at: Optional[datetime] = None


class AlertRuleManager:
    """Manages alert rules and evaluates them against metrics."""
    
    def __init__(self):
        """Initialize alert rule manager."""
        self._lock = threading.RLock()
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Initialize default rules
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Register default alert rules."""
        
        # High latency alert
        self.register_rule(AlertRule(
            name='high_latency',
            description='Action processing latency exceeds threshold',
            severity=AlertSeverity.WARNING,
            condition=lambda metrics: self._check_latency(metrics, 'action_latency', 1.0),
            threshold=1.0,  # 1 second
            duration=60,  # Fire if true for 60 seconds
            labels={'component': 'governance'},
            annotations={'summary': 'High latency detected in action processing'}
        ))
        
        # Critical latency alert
        self.register_rule(AlertRule(
            name='critical_latency',
            description='Action processing latency critically high',
            severity=AlertSeverity.CRITICAL,
            condition=lambda metrics: self._check_latency(metrics, 'action_latency', 5.0),
            threshold=5.0,  # 5 seconds
            duration=30,  # Fire if true for 30 seconds
            labels={'component': 'governance'},
            annotations={'summary': 'Critical latency detected - immediate action required'}
        ))
        
        # High error rate alert
        self.register_rule(AlertRule(
            name='high_error_rate',
            description='Error rate exceeds acceptable threshold',
            severity=AlertSeverity.ERROR,
            condition=lambda metrics: self._check_error_rate(metrics, 0.05),
            threshold=0.05,  # 5%
            duration=120,  # Fire if true for 2 minutes
            labels={'component': 'governance'},
            annotations={'summary': 'Error rate above 5%'}
        ))
        
        # Drift detection alert
        self.register_rule(AlertRule(
            name='drift_detected',
            description='Data or behavior drift detected',
            severity=AlertSeverity.WARNING,
            condition=lambda metrics: self._check_drift(metrics),
            threshold=0.2,  # 20% drift
            duration=300,  # Fire if true for 5 minutes
            labels={'component': 'ml_monitor'},
            annotations={'summary': 'Significant drift detected in model behavior'}
        ))
        
        # Quota saturation alert
        self.register_rule(AlertRule(
            name='quota_saturation',
            description='Resource quota approaching limit',
            severity=AlertSeverity.WARNING,
            condition=lambda metrics: self._check_quota_saturation(metrics, 0.9),
            threshold=0.9,  # 90% usage
            duration=60,
            labels={'component': 'quota_manager'},
            annotations={'summary': 'Quota usage above 90%'}
        ))
        
        # Critical quota alert
        self.register_rule(AlertRule(
            name='quota_critical',
            description='Resource quota critically high',
            severity=AlertSeverity.CRITICAL,
            condition=lambda metrics: self._check_quota_saturation(metrics, 0.95),
            threshold=0.95,  # 95% usage
            duration=30,
            labels={'component': 'quota_manager'},
            annotations={'summary': 'Quota usage above 95% - immediate action required'}
        ))
    
    def _check_latency(self, metrics: Dict[str, Any], metric_name: str, threshold: float) -> bool:
        """Check if latency exceeds threshold."""
        latency_data = metrics.get('histograms', {}).get(metric_name, {})
        p95 = latency_data.get('p95', 0)
        return p95 > threshold
    
    def _check_error_rate(self, metrics: Dict[str, Any], threshold: float) -> bool:
        """Check if error rate exceeds threshold."""
        gauges = metrics.get('gauges', {}).get('error_rate', {})
        for value in gauges.values():
            if value > threshold:
                return True
        return False
    
    def _check_drift(self, metrics: Dict[str, Any]) -> bool:
        """Check for drift indicators."""
        # Check for drift_score gauge
        drift_data = metrics.get('gauges', {}).get('drift_score', {})
        for value in drift_data.values():
            if value > 0.2:  # 20% drift threshold
                return True
        return False
    
    def _check_quota_saturation(self, metrics: Dict[str, Any], threshold: float) -> bool:
        """Check if quota usage exceeds threshold."""
        quota_data = metrics.get('gauges', {}).get('quota_usage', {})
        for value in quota_data.values():
            if value > threshold:
                return True
        return False
    
    def register_rule(self, rule: AlertRule) -> None:
        """Register a new alert rule.
        
        Args:
            rule: AlertRule to register
        """
        with self._lock:
            self.rules[rule.name] = rule
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule.
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if rule was removed
        """
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                return True
        return False
    
    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register an alert handler callback.
        
        Args:
            handler: Callback function to handle alerts
        """
        with self._lock:
            self.alert_handlers.append(handler)
    
    def evaluate_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate all rules against current metrics.
        
        Args:
            metrics: Current metrics data
            
        Returns:
            List of newly fired alerts
        """
        newly_fired = []
        now = time.time()
        
        with self._lock:
            for rule_name, rule in self.rules.items():
                try:
                    # Evaluate condition
                    condition_met = rule.condition(metrics)
                    rule.last_evaluated = now
                    
                    if condition_met:
                        # Track first trigger time
                        if rule.first_triggered is None:
                            rule.first_triggered = now
                            rule.state = AlertState.PENDING
                        
                        # Check if duration threshold met
                        time_active = now - rule.first_triggered
                        if time_active >= rule.duration and rule.state != AlertState.ACTIVE:
                            # Fire alert
                            rule.state = AlertState.ACTIVE
                            rule.firing_count += 1
                            
                            alert = Alert(
                                rule_name=rule.name,
                                severity=rule.severity,
                                message=f"{rule.description} (threshold: {rule.threshold})",
                                timestamp=datetime.utcnow(),
                                labels=rule.labels.copy(),
                                annotations=rule.annotations.copy(),
                                state=AlertState.ACTIVE
                            )
                            
                            self.active_alerts[rule_name] = alert
                            self.alert_history.append(alert)
                            newly_fired.append(alert)
                            
                            # Notify handlers
                            for handler in self.alert_handlers:
                                try:
                                    handler(alert)
                                except Exception as e:
                                    print(f"Alert handler error: {e}")
                    
                    else:
                        # Condition no longer met
                        if rule.state != AlertState.RESOLVED:
                            rule.state = AlertState.RESOLVED
                            rule.first_triggered = None
                            
                            # Resolve active alert if exists
                            if rule_name in self.active_alerts:
                                alert = self.active_alerts[rule_name]
                                alert.state = AlertState.RESOLVED
                                alert.resolved_at = datetime.utcnow()
                                del self.active_alerts[rule_name]
                
                except Exception as e:
                    print(f"Error evaluating rule {rule_name}: {e}")
        
        return newly_fired
    
    def get_active_alerts(self, 
                         severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get currently active alerts.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts
    
    def get_alert_history(self,
                         since: Optional[datetime] = None,
                         severity: Optional[AlertSeverity] = None,
                         limit: int = 100) -> List[Alert]:
        """Get alert history.
        
        Args:
            since: Optional datetime filter
            severity: Optional severity filter
            limit: Maximum number of alerts to return
            
        Returns:
            List of historical alerts
        """
        with self._lock:
            history = self.alert_history.copy()
        
        if since:
            history = [a for a in history if a.timestamp >= since]
        
        if severity:
            history = [a for a in history if a.severity == severity]
        
        # Sort by timestamp descending
        history.sort(key=lambda a: a.timestamp, reverse=True)
        
        return history[:limit]
    
    def get_rule_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all rules.
        
        Returns:
            Dictionary with rule status information
        """
        with self._lock:
            status = {}
            
            for name, rule in self.rules.items():
                status[name] = {
                    'name': name,
                    'severity': rule.severity.value,
                    'state': rule.state.value,
                    'firing_count': rule.firing_count,
                    'last_evaluated': rule.last_evaluated,
                    'threshold': rule.threshold,
                    'duration': rule.duration
                }
        
        return status
    
    def clear_history(self, before: Optional[datetime] = None) -> int:
        """Clear alert history.
        
        Args:
            before: Optional datetime - clear alerts before this time
            
        Returns:
            Number of alerts cleared
        """
        with self._lock:
            if before:
                original_len = len(self.alert_history)
                self.alert_history = [
                    a for a in self.alert_history 
                    if a.timestamp >= before
                ]
                return original_len - len(self.alert_history)
            else:
                count = len(self.alert_history)
                self.alert_history.clear()
                return count


# Convenience function for default console handler
def console_alert_handler(alert: Alert) -> None:
    """Simple console alert handler."""
    severity_prefix = {
        AlertSeverity.INFO: "ℹ️",
        AlertSeverity.WARNING: "⚠️",
        AlertSeverity.ERROR: "❌",
        AlertSeverity.CRITICAL: "🚨"
    }
    
    prefix = severity_prefix.get(alert.severity, "•")
    print(f"{prefix} [{alert.severity.value.upper()}] {alert.rule_name}: {alert.message}")
    if alert.annotations:
        print(f"   Summary: {alert.annotations.get('summary', 'N/A')}")


if __name__ == '__main__':
    # Demo usage
    manager = AlertRuleManager()
    
    # Register console handler
    manager.register_handler(console_alert_handler)
    
    # Simulate metrics with high latency
    test_metrics = {
        'histograms': {
            'action_latency': {
                'p95': 6.0  # 6 seconds - exceeds both thresholds
            }
        },
        'gauges': {
            'error_rate': {
                'default': 0.08  # 8% error rate
            },
            'quota_usage': {
                'agent_001': 0.92  # 92% quota usage
            }
        }
    }
    
    print("Evaluating alert rules...")
    print("=" * 60)
    
    # Evaluate rules
    fired = manager.evaluate_rules(test_metrics)
    
    print(f"\n{len(fired)} alerts fired")
    print(f"{len(manager.get_active_alerts())} active alerts\n")
    
    # Show rule status
    print("Rule Status:")
    print("-" * 60)
    for name, status in manager.get_rule_status().items():
        print(f"{name}: {status['state']} (fired {status['firing_count']} times)")
