"""Tests for Observability features."""

import pytest
import time
from datetime import datetime, timedelta

from nethical.observability.metrics import MetricsCollector, get_metrics_collector
from nethical.observability.sanitization import LogSanitizer, sanitize_log
from nethical.observability.tracing import TracingManager, get_tracer
from nethical.observability.alerts import AlertRuleManager, AlertSeverity, AlertRule


class TestMetrics:
    """Test metrics collection."""
    
    def test_metrics_collector_creation(self):
        """Test creating metrics collector."""
        collector = MetricsCollector(enable_prometheus=False)
        assert collector is not None
    
    def test_record_action(self):
        """Test recording actions."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.record_action("api_call", "ALLOW", "US")
        collector.record_action("api_call", "DENY", "EU")
        
        # Check counters
        actions = collector.get_counter("actions")
        assert actions > 0
    
    def test_record_violation(self):
        """Test recording violations."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.record_violation("manipulation", "high", "detector1")
        collector.record_violation("privacy", "critical", "detector2")
        
        violations = collector.get_counter("violations")
        assert violations > 0
    
    def test_latency_recording(self):
        """Test latency histogram."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.record_latency("test_operation", 0.5)
        collector.record_latency("test_operation", 1.0)
        collector.record_latency("test_operation", 0.8)
        
        stats = collector.get_histogram_stats("test_operation")
        assert stats['count'] == 3
        assert stats['min'] == 0.5
        assert stats['max'] == 1.0
    
    def test_gauge_metrics(self):
        """Test gauge metrics."""
        collector = MetricsCollector(enable_prometheus=False)
        
        collector.set_gauge("active_sessions", 10)
        collector.set_gauge("error_rate", 0.02, {"component": "api"})
        
        metrics = collector.get_all_metrics()
        assert "active_sessions" in metrics['gauges']


class TestSanitization:
    """Test log sanitization."""
    
    def test_ssn_redaction(self):
        """Test SSN redaction."""
        sanitizer = LogSanitizer()
        
        text = "My SSN is 123-45-6789"
        sanitized = sanitizer.sanitize(text)
        
        assert "123-45-6789" not in sanitized
        assert "[SSN-REDACTED]" in sanitized
    
    def test_email_redaction(self):
        """Test email redaction."""
        sanitizer = LogSanitizer()
        
        text = "Contact me at john.doe@example.com"
        sanitized = sanitizer.sanitize(text)
        
        assert "john.doe@example.com" not in sanitized
        assert "[EMAIL-REDACTED]" in sanitized
    
    def test_phone_redaction(self):
        """Test phone number redaction."""
        sanitizer = LogSanitizer()
        
        text = "Call me at +1-555-123-4567"
        sanitized = sanitizer.sanitize(text)
        
        assert "555-123-4567" not in sanitized
        assert "[PHONE-REDACTED]" in sanitized
    
    def test_api_key_redaction(self):
        """Test API key redaction."""
        sanitizer = LogSanitizer()
        
        text = 'api_key="sk_live_abc123xyz789"'
        sanitized = sanitizer.sanitize(text)
        
        assert "sk_live_abc123xyz789" not in sanitized
        assert "[KEY-REDACTED]" in sanitized
    
    def test_password_redaction(self):
        """Test password redaction."""
        sanitizer = LogSanitizer()
        
        text = 'password="secret123"'
        sanitized = sanitizer.sanitize(text)
        
        assert "secret123" not in sanitized
        assert "[PASSWORD-REDACTED]" in sanitized
    
    def test_dict_sanitization(self):
        """Test dictionary sanitization."""
        sanitizer = LogSanitizer()
        
        data = {
            'user': 'john.doe@example.com',
            'password': 'secret',
            'api_key': 'sk_test_123',
            'message': 'Normal message'
        }
        
        sanitized = sanitizer.sanitize_dict(data)
        
        assert '[EMAIL-REDACTED]' in sanitized['user']
        assert sanitized['password'] == '[REDACTED]'
        assert sanitized['api_key'] == '[REDACTED]'
        assert sanitized['message'] == 'Normal message'
    
    def test_nested_dict_sanitization(self):
        """Test recursive dictionary sanitization."""
        sanitizer = LogSanitizer()
        
        data = {
            'user': 'test@example.com',
            'metadata': {
                'password': 'secret',
                'note': 'SSN is 123-45-6789'
            }
        }
        
        sanitized = sanitizer.sanitize_dict(data, recursive=True)
        
        assert '[EMAIL-REDACTED]' in sanitized['user']
        assert sanitized['metadata']['password'] == '[REDACTED]'
        # Check that SSN in note field gets redacted by text sanitization
        assert '[SSN-REDACTED]' in sanitized['metadata']['note']
    
    def test_convenience_function(self):
        """Test convenience sanitization function."""
        text = "My email is test@example.com"
        sanitized = sanitize_log(text)
        
        assert "test@example.com" not in sanitized
        assert "[EMAIL-REDACTED]" in sanitized


class TestTracing:
    """Test distributed tracing."""
    
    def test_tracing_manager_creation(self):
        """Test creating tracing manager."""
        tracer = TracingManager(enable_otel=False)
        assert tracer is not None
    
    def test_span_creation(self):
        """Test creating spans."""
        tracer = TracingManager(enable_otel=False)
        
        with tracer.start_span("test_operation") as span:
            assert span is not None
            assert span.name == "test_operation"
    
    def test_span_attributes(self):
        """Test adding span attributes."""
        tracer = TracingManager(enable_otel=False)
        
        with tracer.start_span("test_op", attributes={"key": "value"}) as span:
            tracer.add_span_attribute("additional", "data")
            assert "key" in span.attributes
    
    def test_span_events(self):
        """Test adding span events."""
        tracer = TracingManager(enable_otel=False)
        
        with tracer.start_span("test_op") as span:
            tracer.add_span_event("checkpoint", {"step": 1})
            assert len(span.events) > 0
    
    def test_sampling_rate(self):
        """Test sampling logic."""
        tracer = TracingManager(baseline_sample_rate=0.0, enable_otel=False)
        
        # With 0% baseline, should not sample normal ops
        assert tracer.should_sample(is_error=False) is False
        
        # With 100% error rate, should always sample errors
        assert tracer.should_sample(is_error=True) is True
    
    def test_nested_spans(self):
        """Test nested span creation."""
        tracer = TracingManager(enable_otel=False)
        
        with tracer.start_span("parent") as parent_span:
            with tracer.start_span("child") as child_span:
                assert child_span.parent_span_id == parent_span.span_id


class TestAlertRules:
    """Test alert rule management."""
    
    def test_alert_manager_creation(self):
        """Test creating alert manager."""
        manager = AlertRuleManager()
        assert manager is not None
        assert len(manager.rules) > 0  # Should have default rules
    
    def test_register_rule(self):
        """Test registering custom rule."""
        manager = AlertRuleManager()
        
        rule = AlertRule(
            name="test_rule",
            description="Test alert",
            severity=AlertSeverity.WARNING,
            condition=lambda m: True,
            threshold=1.0,
            duration=60
        )
        
        manager.register_rule(rule)
        assert "test_rule" in manager.rules
    
    def test_remove_rule(self):
        """Test removing rules."""
        manager = AlertRuleManager()
        
        rule = AlertRule(
            name="temp_rule",
            description="Temporary",
            severity=AlertSeverity.INFO,
            condition=lambda m: False,
            threshold=1.0,
            duration=60
        )
        
        manager.register_rule(rule)
        assert "temp_rule" in manager.rules
        
        removed = manager.remove_rule("temp_rule")
        assert removed is True
        assert "temp_rule" not in manager.rules
    
    def test_evaluate_rules(self):
        """Test rule evaluation."""
        manager = AlertRuleManager()
        
        # Create metrics that trigger high latency
        metrics = {
            'histograms': {
                'action_latency': {
                    'p95': 6.0  # Exceeds threshold
                }
            },
            'gauges': {
                'error_rate': {},
                'drift_score': {},
                'quota_usage': {}
            }
        }
        
        # Evaluate (won't fire immediately due to duration requirement)
        fired = manager.evaluate_rules(metrics)
        
        # Check rule states updated
        rule_status = manager.get_rule_status()
        assert 'critical_latency' in rule_status
    
    def test_alert_handler(self):
        """Test alert handler registration."""
        manager = AlertRuleManager()
        
        fired_alerts = []
        
        def test_handler(alert):
            fired_alerts.append(alert)
        
        manager.register_handler(test_handler)
        
        # Handler should be registered
        assert len(manager.alert_handlers) > 0
    
    def test_get_active_alerts(self):
        """Test retrieving active alerts."""
        manager = AlertRuleManager()
        
        active = manager.get_active_alerts()
        assert isinstance(active, list)
    
    def test_get_rule_status(self):
        """Test getting rule status."""
        manager = AlertRuleManager()
        
        status = manager.get_rule_status()
        
        assert isinstance(status, dict)
        assert len(status) > 0
        
        # Check default rules exist
        assert 'high_latency' in status
        assert 'high_error_rate' in status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
