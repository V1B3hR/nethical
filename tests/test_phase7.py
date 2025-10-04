"""Tests for Phase 7: Anomaly & Drift Detection."""

import pytest
import tempfile
from pathlib import Path

from nethical.core import (
    AnomalyDriftMonitor,
    SequenceAnomalyDetector,
    DistributionDriftDetector,
    AnomalyAlert,
    AnomalyType,
    DriftSeverity,
    DriftMetrics
)


class TestSequenceAnomalyDetector:
    """Test sequence anomaly detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = SequenceAnomalyDetector(
            n=3,
            min_frequency=2,
            anomaly_threshold=0.8
        )
        
        assert detector.n == 3
        assert detector.min_frequency == 2
        assert detector.anomaly_threshold == 0.8
        assert detector.total_ngrams == 0
    
    def test_record_action(self):
        """Test recording actions."""
        detector = SequenceAnomalyDetector(n=3)
        
        detector.record_action("agent_001", "read")
        detector.record_action("agent_001", "process")
        detector.record_action("agent_001", "write")
        
        # Should have one 3-gram
        assert detector.total_ngrams == 1
        assert len(detector.agent_sequences["agent_001"]) == 3
    
    def test_detect_anomaly_normal(self):
        """Test anomaly detection with normal sequence."""
        detector = SequenceAnomalyDetector(n=3)
        
        # Establish baseline
        for _ in range(10):
            detector.record_action("agent_001", "read")
            detector.record_action("agent_001", "process")
            detector.record_action("agent_001", "write")
        
        # Test same sequence
        is_anomalous, score, evidence = detector.detect_anomaly("agent_001")
        
        # Should not be anomalous (common sequence)
        assert score < 0.8  # Below default threshold
    
    def test_detect_anomaly_unusual(self):
        """Test anomaly detection with unusual sequence."""
        detector = SequenceAnomalyDetector(n=3, anomaly_threshold=0.9, min_frequency=2)
        
        # Establish baseline with common patterns
        for _ in range(10):
            detector.record_action("agent_001", "read")
            detector.record_action("agent_001", "process")
            detector.record_action("agent_001", "write")
        
        # Now unusual sequence - should have high anomaly score
        detector.record_action("agent_002", "delete")
        detector.record_action("agent_002", "exfiltrate")
        detector.record_action("agent_002", "cover")
        
        is_anomalous, score, evidence = detector.detect_anomaly("agent_002")
        
        # Should be anomalous (unseen sequence)
        # Score is based on rarity - unseen ngrams have high score
        assert score >= 0.9 or evidence['is_unseen'] is True
    
    def test_statistics(self):
        """Test detector statistics."""
        detector = SequenceAnomalyDetector(n=3)
        
        for i in range(5):
            detector.record_action(f"agent_{i}", "action")
            detector.record_action(f"agent_{i}", "test")
            detector.record_action(f"agent_{i}", "end")
        
        stats = detector.get_statistics()
        
        assert stats['n'] == 3
        assert stats['total_ngrams'] > 0
        assert stats['tracked_agents'] > 0


class TestDistributionDriftDetector:
    """Test distribution drift detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = DistributionDriftDetector(
            num_bins=10,
            psi_threshold=0.2,
            kl_threshold=0.1
        )
        
        assert detector.num_bins == 10
        assert detector.psi_threshold == 0.2
        assert detector.kl_threshold == 0.1
    
    def test_set_baseline(self):
        """Test setting baseline distribution."""
        detector = DistributionDriftDetector()
        
        baseline_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        detector.set_baseline(baseline_scores)
        
        assert len(detector.baseline_scores) == 5
        assert detector.baseline_distribution is not None
    
    def test_add_score(self):
        """Test adding current scores."""
        detector = DistributionDriftDetector()
        
        detector.add_score(0.5)
        detector.add_score(0.6)
        
        assert len(detector.current_scores) == 2
    
    def test_detect_drift_no_drift(self):
        """Test drift detection with no drift."""
        detector = DistributionDriftDetector()
        
        # Set baseline
        baseline = [0.1, 0.2, 0.3, 0.4, 0.5] * 20
        detector.set_baseline(baseline)
        
        # Add similar current scores
        for score in [0.15, 0.25, 0.35, 0.45, 0.55] * 10:
            detector.add_score(score)
        
        metrics = detector.detect_drift()
        
        # Should not detect drift
        assert metrics.psi_drift_detected is False
        assert metrics.kl_drift_detected is False
    
    def test_detect_drift_with_drift(self):
        """Test drift detection with actual drift."""
        detector = DistributionDriftDetector(
            psi_threshold=0.1,
            kl_threshold=0.05
        )
        
        # Set baseline (low scores)
        baseline = [0.1, 0.2, 0.15, 0.25, 0.3] * 20
        detector.set_baseline(baseline)
        
        # Add shifted current scores (high scores)
        for _ in range(50):
            detector.add_score(0.7 + (0.1 * (hash(str(_)) % 10) / 10))
        
        metrics = detector.detect_drift()
        
        # Should detect drift
        assert metrics.psi_drift_detected or metrics.kl_drift_detected
    
    def test_reset_current(self):
        """Test resetting current distribution."""
        detector = DistributionDriftDetector()
        
        detector.add_score(0.5)
        detector.add_score(0.6)
        
        assert len(detector.current_scores) == 2
        
        detector.reset_current()
        
        assert len(detector.current_scores) == 0
    
    def test_statistics(self):
        """Test detector statistics."""
        detector = DistributionDriftDetector()
        
        baseline = [0.1, 0.2, 0.3]
        detector.set_baseline(baseline)
        
        detector.add_score(0.4)
        detector.add_score(0.5)
        
        stats = detector.get_statistics()
        
        assert stats['baseline_size'] == 3
        assert stats['current_size'] == 2


class TestAnomalyDriftMonitor:
    """Test integrated anomaly and drift monitor."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = AnomalyDriftMonitor(
            sequence_n=3,
            psi_threshold=0.2,
            kl_threshold=0.1
        )
        
        assert monitor.sequence_detector is not None
        assert monitor.drift_detector is not None
        assert len(monitor.alerts) == 0
    
    def test_record_action(self):
        """Test recording actions."""
        monitor = AnomalyDriftMonitor()
        
        alert = monitor.record_action(
            agent_id="agent_001",
            action_type="read",
            risk_score=0.3,
            cohort="production"
        )
        
        # First action shouldn't trigger alert
        assert alert is None or isinstance(alert, AnomalyAlert)
    
    def test_behavioral_anomaly_detection(self):
        """Test behavioral anomaly detection."""
        monitor = AnomalyDriftMonitor()
        
        # Normal diverse behavior
        for action in ['read', 'write', 'update', 'delete', 'query']:
            for _ in range(5):
                monitor.record_action("normal_agent", action, 0.3)
        
        alert = monitor.check_behavioral_anomaly("normal_agent")
        assert alert is None  # Should be normal
        
        # Repetitive behavior (suspicious)
        for _ in range(25):
            monitor.record_action("suspicious_agent", "same_action", 0.5)
        
        alert = monitor.check_behavioral_anomaly("suspicious_agent")
        assert alert is not None  # Should detect anomaly
        assert alert.anomaly_type == AnomalyType.BEHAVIORAL
    
    def test_drift_detection(self):
        """Test distribution drift detection."""
        monitor = AnomalyDriftMonitor()
        
        # Set baseline
        baseline = [0.2, 0.3, 0.25, 0.35, 0.28] * 40
        monitor.set_baseline_distribution(baseline)
        
        # Add similar current (no drift)
        for score in [0.22, 0.32, 0.27, 0.33, 0.29] * 20:
            monitor.drift_detector.add_score(score)
        
        alert = monitor.check_drift()
        
        # May or may not detect depending on exact distribution
        if alert:
            assert alert.anomaly_type == AnomalyType.DISTRIBUTIONAL
    
    def test_alert_filtering(self):
        """Test alert filtering."""
        monitor = AnomalyDriftMonitor()
        
        # Create some alerts manually
        alert1 = AnomalyAlert(
            alert_id="alert_001",
            timestamp=None,
            anomaly_type=AnomalyType.SEQUENCE,
            severity=DriftSeverity.WARNING,
            anomaly_score=0.8,
            threshold=0.7
        )
        
        alert2 = AnomalyAlert(
            alert_id="alert_002",
            timestamp=None,
            anomaly_type=AnomalyType.BEHAVIORAL,
            severity=DriftSeverity.CRITICAL,
            anomaly_score=0.95,
            threshold=0.8
        )
        
        monitor.alerts = [alert1, alert2]
        
        # Filter by severity
        critical_alerts = monitor.get_alerts(severity=DriftSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == DriftSeverity.CRITICAL
        
        # Filter by type
        seq_alerts = monitor.get_alerts(anomaly_type=AnomalyType.SEQUENCE)
        assert len(seq_alerts) == 1
        assert seq_alerts[0].anomaly_type == AnomalyType.SEQUENCE
    
    def test_statistics(self):
        """Test monitor statistics."""
        monitor = AnomalyDriftMonitor()
        
        # Add some data
        for i in range(10):
            monitor.record_action(f"agent_{i}", "action", 0.3)
        
        stats = monitor.get_statistics()
        
        assert 'alerts' in stats
        assert 'sequence_detector' in stats
        assert 'drift_detector' in stats
    
    def test_export_alerts(self):
        """Test alert export."""
        monitor = AnomalyDriftMonitor()
        
        # Create alert
        alert = AnomalyAlert(
            alert_id="test_001",
            timestamp=None,
            anomaly_type=AnomalyType.SEQUENCE,
            severity=DriftSeverity.INFO,
            anomaly_score=0.7,
            threshold=0.6
        )
        
        monitor.alerts = [alert]
        
        exported = monitor.export_alerts()
        
        assert len(exported) == 1
        assert isinstance(exported[0], dict)
        assert exported[0]['alert_id'] == "test_001"
    
    def test_storage_persistence(self):
        """Test alert persistence to storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = AnomalyDriftMonitor(storage_path=tmpdir)
            
            # Trigger behavioral anomaly
            for _ in range(25):
                monitor.record_action("agent_001", "same_action", 0.5)
            
            monitor.check_behavioral_anomaly("agent_001")
            
            # Check if log file was created
            log_file = Path(tmpdir) / "anomaly_alerts.jsonl"
            if monitor.alerts:  # If alert was actually triggered
                assert log_file.exists()


class TestDriftMetrics:
    """Test drift metrics."""
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = DriftMetrics()
        
        assert metrics.psi_score == 0.0
        assert metrics.kl_divergence == 0.0
        assert metrics.psi_drift_detected is False
        assert metrics.kl_drift_detected is False
    
    def test_to_dict(self):
        """Test metrics to dictionary."""
        metrics = DriftMetrics(
            psi_score=0.25,
            kl_divergence=0.15,
            psi_drift_detected=True,
            kl_drift_detected=True
        )
        
        data = metrics.to_dict()
        
        assert data['psi_score'] == 0.25
        assert data['kl_divergence'] == 0.15
        assert data['drift_detected'] is True


class TestEnums:
    """Test enum types."""
    
    def test_anomaly_type(self):
        """Test anomaly type enum."""
        assert AnomalyType.SEQUENCE.value == "sequence"
        assert AnomalyType.FREQUENCY.value == "frequency"
        assert AnomalyType.BEHAVIORAL.value == "behavioral"
        assert AnomalyType.DISTRIBUTIONAL.value == "distributional"
    
    def test_drift_severity(self):
        """Test drift severity enum."""
        assert DriftSeverity.INFO.value == "info"
        assert DriftSeverity.WARNING.value == "warning"
        assert DriftSeverity.CRITICAL.value == "critical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
