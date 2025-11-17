"""
Tests for Phase 7 Anomaly Detection and Alert System

Tests anomaly detection, alert creation, and escalation.
"""

import pytest
from datetime import datetime, timedelta
from probes import ProbeResult, ProbeStatus
from probes.anomaly_detector import (
    AnomalyDetector,
    AlertSystem,
    Alert,
    AlertSeverity,
)


class TestAnomalyDetector:
    """Test anomaly detector"""
    
    def test_creation(self):
        """Test creating anomaly detector"""
        detector = AnomalyDetector(sensitivity=2.0, lookback_window=100)
        assert detector.sensitivity == 2.0
        assert detector.lookback_window == 100
    
    def test_analyze_healthy_result(self):
        """Test analyzing healthy probe result"""
        detector = AnomalyDetector()
        
        result = ProbeResult(
            probe_name="test-probe",
            status=ProbeStatus.HEALTHY,
            timestamp=datetime.utcnow(),
            message="All good",
            metrics={"latency_ms": 100.0}
        )
        
        anomaly = detector.analyze(result)
        assert anomaly is None or 'status' not in anomaly
    
    def test_analyze_critical_result(self):
        """Test analyzing critical probe result"""
        detector = AnomalyDetector()
        
        result = ProbeResult(
            probe_name="test-probe",
            status=ProbeStatus.CRITICAL,
            timestamp=datetime.utcnow(),
            message="Critical failure",
            metrics={}
        )
        
        anomaly = detector.analyze(result)
        assert anomaly is not None
        assert 'status' in anomaly
    
    def test_statistical_anomaly_detection(self):
        """Test statistical anomaly detection"""
        detector = AnomalyDetector(sensitivity=2.0)
        
        # Build baseline
        for i in range(50):
            result = ProbeResult(
                probe_name="test-probe",
                status=ProbeStatus.HEALTHY,
                timestamp=datetime.utcnow(),
                message="Normal",
                metrics={"latency_ms": 100.0 + i * 0.5}
            )
            detector.analyze(result)
        
        # Add outlier
        outlier = ProbeResult(
            probe_name="test-probe",
            status=ProbeStatus.HEALTHY,
            timestamp=datetime.utcnow(),
            message="Outlier",
            metrics={"latency_ms": 500.0}  # Way above baseline
        )
        
        anomaly = detector.analyze(outlier)
        assert anomaly is not None
        assert 'latency_ms' in anomaly
    
    def test_trend_detection(self):
        """Test trend detection"""
        detector = AnomalyDetector()
        
        # Build increasing trend
        for i in range(10):
            result = ProbeResult(
                probe_name="test-probe",
                status=ProbeStatus.HEALTHY,
                timestamp=datetime.utcnow(),
                message="Normal",
                metrics={"latency_ms": 100.0 + i * 10.0}
            )
            detector.analyze(result)
        
        trend = detector.detect_trend("test-probe", "latency_ms", min_samples=5)
        assert trend is not None
        assert "Increasing" in trend


class TestAlertSystem:
    """Test alert system"""
    
    def test_creation(self):
        """Test creating alert system"""
        system = AlertSystem(escalation_threshold_seconds=3600)
        assert system.escalation_threshold_seconds == 3600
    
    def test_create_alert(self):
        """Test creating alert"""
        system = AlertSystem()
        
        alert = system.create_alert(
            severity=AlertSeverity.CRITICAL,
            probe_name="test-probe",
            message="Test alert",
            metrics={"value": 100},
            violations=["violation 1"]
        )
        
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.probe_name == "test-probe"
        assert len(system._alerts) == 1
    
    def test_alert_deduplication(self):
        """Test alert deduplication"""
        system = AlertSystem()
        
        # Create same alert twice
        alert1 = system.create_alert(
            severity=AlertSeverity.WARNING,
            probe_name="test-probe",
            message="Same alert"
        )
        
        alert2 = system.create_alert(
            severity=AlertSeverity.WARNING,
            probe_name="test-probe",
            message="Same alert"
        )
        
        # Should be same alert (deduplicated)
        assert alert1.alert_id == alert2.alert_id
        assert len(system._alerts) == 1
    
    def test_acknowledge_alert(self):
        """Test acknowledging alert"""
        system = AlertSystem()
        
        alert = system.create_alert(
            severity=AlertSeverity.WARNING,
            probe_name="test-probe",
            message="Test"
        )
        
        success = system.acknowledge_alert(alert.alert_id)
        assert success
        assert alert.acknowledged
    
    def test_resolve_alert(self):
        """Test resolving alert"""
        system = AlertSystem()
        
        alert = system.create_alert(
            severity=AlertSeverity.WARNING,
            probe_name="test-probe",
            message="Test"
        )
        
        success = system.resolve_alert(alert.alert_id)
        assert success
        assert alert.resolved
    
    def test_get_active_alerts(self):
        """Test getting active alerts"""
        system = AlertSystem()
        
        # Create alerts
        system.create_alert(AlertSeverity.CRITICAL, "probe1", "Alert 1")
        system.create_alert(AlertSeverity.WARNING, "probe2", "Alert 2")
        system.create_alert(AlertSeverity.INFO, "probe3", "Alert 3")
        
        # Resolve one
        alerts = system.get_active_alerts()
        system.resolve_alert(alerts[0].alert_id)
        
        # Get active alerts
        active = system.get_active_alerts()
        assert len(active) == 2
    
    def test_get_active_alerts_by_severity(self):
        """Test filtering active alerts by severity"""
        system = AlertSystem()
        
        system.create_alert(AlertSeverity.CRITICAL, "probe1", "Critical")
        system.create_alert(AlertSeverity.CRITICAL, "probe2", "Critical")
        system.create_alert(AlertSeverity.WARNING, "probe3", "Warning")
        
        critical = system.get_active_alerts(severity=AlertSeverity.CRITICAL)
        assert len(critical) == 2
        assert all(a.severity == AlertSeverity.CRITICAL for a in critical)
    
    def test_check_escalation(self):
        """Test alert escalation"""
        system = AlertSystem(escalation_threshold_seconds=1)  # 1 second for testing
        
        # Create alert
        alert = system.create_alert(
            AlertSeverity.CRITICAL,
            "test-probe",
            "Test escalation"
        )
        
        # Initially not escalated
        assert not alert.escalated
        
        # Immediately check - should not escalate yet
        to_escalate = system.check_escalation()
        assert len(to_escalate) == 0
        
        # Wait and check again (in production, would wait >threshold)
        # For testing, manually set timestamp
        alert.timestamp = datetime.utcnow() - timedelta(seconds=2)
        
        to_escalate = system.check_escalation()
        assert len(to_escalate) == 1
        assert to_escalate[0].alert_id == alert.alert_id
        assert alert.escalated
    
    def test_escalation_with_acknowledgment(self):
        """Test that acknowledged alerts don't escalate"""
        system = AlertSystem(escalation_threshold_seconds=1)
        
        # Create and acknowledge alert
        alert = system.create_alert(
            AlertSeverity.CRITICAL,
            "test-probe",
            "Test"
        )
        system.acknowledge_alert(alert.alert_id)
        
        # Make it old enough to escalate
        alert.timestamp = datetime.utcnow() - timedelta(seconds=2)
        
        # Should not escalate (acknowledged)
        to_escalate = system.check_escalation()
        assert len(to_escalate) == 0
    
    def test_register_handler(self):
        """Test registering alert handler"""
        system = AlertSystem()
        
        handled_alerts = []
        
        def handler(alert: Alert):
            handled_alerts.append(alert)
        
        system.register_handler(AlertSeverity.CRITICAL, handler)
        
        # Create critical alert
        system.create_alert(
            AlertSeverity.CRITICAL,
            "test-probe",
            "Critical alert"
        )
        
        # Handler should have been called
        assert len(handled_alerts) == 1
    
    def test_get_metrics(self):
        """Test getting alert system metrics"""
        system = AlertSystem()
        
        # Create various alerts
        system.create_alert(AlertSeverity.CRITICAL, "probe1", "Critical")
        system.create_alert(AlertSeverity.WARNING, "probe2", "Warning")
        system.create_alert(AlertSeverity.INFO, "probe3", "Info")
        
        metrics = system.get_metrics()
        assert metrics['total_alerts'] == 3
        assert metrics['active_alerts'] == 3
        assert metrics['critical_alerts'] == 1
        assert metrics['warning_alerts'] == 1


class TestAlertModel:
    """Test Alert data model"""
    
    def test_alert_creation(self):
        """Test creating alert"""
        alert = Alert(
            alert_id="ALT-001",
            severity=AlertSeverity.WARNING,
            probe_name="test-probe",
            message="Test alert",
            timestamp=datetime.utcnow(),
            metrics={"value": 100},
            violations=["violation"]
        )
        
        assert alert.alert_id == "ALT-001"
        assert alert.severity == AlertSeverity.WARNING
        assert not alert.acknowledged
        assert not alert.resolved
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary"""
        alert = Alert(
            alert_id="ALT-001",
            severity=AlertSeverity.CRITICAL,
            probe_name="test-probe",
            message="Test",
            timestamp=datetime.utcnow(),
            metrics={},
            violations=[]
        )
        
        alert_dict = alert.to_dict()
        assert alert_dict['alert_id'] == "ALT-001"
        assert alert_dict['severity'] == "critical"
        assert alert_dict['probe_name'] == "test-probe"
        assert 'timestamp' in alert_dict


class TestAnomalyDetectionIntegration:
    """Integration tests for anomaly detection and alerting"""
    
    def test_end_to_end_anomaly_detection_and_alerting(self):
        """Test complete workflow from probe to alert"""
        detector = AnomalyDetector()
        alert_system = AlertSystem()
        
        # Simulate probe results over time
        for i in range(20):
            result = ProbeResult(
                probe_name="latency-probe",
                status=ProbeStatus.HEALTHY,
                timestamp=datetime.utcnow(),
                message="Normal",
                metrics={"latency_ms": 100.0 + i}
            )
            detector.analyze(result)
        
        # Add critical result
        critical_result = ProbeResult(
            probe_name="latency-probe",
            status=ProbeStatus.CRITICAL,
            timestamp=datetime.utcnow(),
            message="High latency detected",
            metrics={"latency_ms": 500.0}
        )
        
        anomaly = detector.analyze(critical_result)
        assert anomaly is not None
        
        # Create alert based on anomaly
        alert = alert_system.create_alert(
            severity=AlertSeverity.CRITICAL,
            probe_name="latency-probe",
            message="Latency anomaly detected",
            metrics=critical_result.metrics,
            violations=["latency_ms > threshold"]
        )
        
        assert alert is not None
        assert len(alert_system.get_active_alerts()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
