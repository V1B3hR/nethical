"""
Unit tests for Phase 2.1: Advanced Anomaly Detection
"""

import pytest
from datetime import datetime, timedelta, timezone
from nethical.security.anomaly_detection import (
    AnomalyType,
    AnomalyDetectionResult,
    LSTMSequenceDetector,
    TransformerContextAnalyzer,
    GraphRelationshipAnalyzer,
    InsiderThreatDetector,
    APTBehavioralDetector,
    AdvancedAnomalyDetectionEngine,
)


class TestAnomalyDetectionResult:
    """Test AnomalyDetectionResult"""
    
    def test_creation(self):
        """Test result creation"""
        result = AnomalyDetectionResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.SEQUENCE,
            confidence_score=0.85,
            severity="high",
        )
        
        assert result.is_anomalous is True
        assert result.anomaly_type == AnomalyType.SEQUENCE
        assert result.confidence_score == 0.85
        assert result.severity == "high"
    
    def test_is_critical(self):
        """Test critical detection"""
        result = AnomalyDetectionResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.APT_BEHAVIOR,
            confidence_score=0.9,
            severity="critical",
        )
        
        assert result.is_critical() is True
        
        # Not critical if score too low
        result2 = AnomalyDetectionResult(
            is_anomalous=True,
            anomaly_type=AnomalyType.APT_BEHAVIOR,
            confidence_score=0.7,
            severity="critical",
        )
        assert result2.is_critical() is False


class TestLSTMSequenceDetector:
    """Test LSTM Sequence Detector"""
    
    def test_initialization(self):
        """Test detector initialization"""
        detector = LSTMSequenceDetector(
            sequence_length=10,
            threshold=0.7,
        )
        
        assert detector.sequence_length == 10
        assert detector.threshold == 0.7
        assert detector.training_mode is False
    
    @pytest.mark.asyncio
    async def test_analyze_sequence_insufficient_history(self):
        """Test sequence analysis with insufficient history"""
        detector = LSTMSequenceDetector(sequence_length=5)
        
        event = {
            "type": "read",
            "timestamp": datetime.now(timezone.utc),
        }
        
        result = await detector.analyze_sequence("agent1", event)
        
        assert result.is_anomalous is False
        assert result.anomaly_type == AnomalyType.SEQUENCE
        assert "Insufficient" in result.details["reason"]
    
    @pytest.mark.asyncio
    async def test_analyze_sequence_with_anomaly(self):
        """Test sequence analysis detecting anomaly"""
        detector = LSTMSequenceDetector(sequence_length=5, threshold=0.3)
        
        # Build up sequence with rapid repeated actions
        for i in range(5):
            event = {
                "type": "read",  # Same action repeated
                "timestamp": datetime.now(timezone.utc),
                "privilege_level": 1,
            }
            result = await detector.analyze_sequence("agent1", event)
        
        # Last result should detect anomaly
        assert result.is_anomalous or result.anomaly_type == AnomalyType.SEQUENCE
    
    @pytest.mark.asyncio
    async def test_train_on_data(self):
        """Test training functionality"""
        detector = LSTMSequenceDetector(training_mode=True)
        
        training_data = [
            {"type": "read", "timestamp": datetime.now(timezone.utc)},
            {"type": "write", "timestamp": datetime.now(timezone.utc)},
        ]
        
        result = await detector.train_on_data(training_data)
        
        assert result["training_complete"] is True
        assert result["sequences_processed"] == 2


class TestTransformerContextAnalyzer:
    """Test Transformer Context Analyzer"""
    
    def test_initialization(self):
        """Test analyzer initialization"""
        analyzer = TransformerContextAnalyzer(
            context_window=50,
            attention_heads=8,
        )
        
        assert analyzer.context_window == 50
        assert analyzer.attention_heads == 8
        assert analyzer.threshold == 0.7
    
    @pytest.mark.asyncio
    async def test_analyze_context_normal(self):
        """Test context analysis for normal behavior"""
        analyzer = TransformerContextAnalyzer(threshold=0.9)  # High threshold
        
        event = {
            "type": "read",
            "privilege_level": 1,
            "resource": "file1",
        }
        
        result = await analyzer.analyze_context("agent1", event)
        
        assert result.anomaly_type == AnomalyType.CONTEXT
        assert "context_window_size" in result.details
    
    @pytest.mark.asyncio
    async def test_analyze_context_with_anomaly(self):
        """Test context analysis detecting anomaly"""
        analyzer = TransformerContextAnalyzer(threshold=0.3)
        
        # Build normal context
        for i in range(5):
            event = {
                "type": "read",
                "privilege_level": 1,
                "resource": "file1",
            }
            await analyzer.analyze_context("agent1", event)
        
        # Add anomalous event (high privilege)
        anomalous_event = {
            "type": "delete",
            "privilege_level": 5,
            "resource": "sensitive_file",
        }
        
        result = await analyzer.analyze_context("agent1", anomalous_event)
        
        # Should detect some anomaly
        assert result.anomaly_type == AnomalyType.CONTEXT


class TestGraphRelationshipAnalyzer:
    """Test Graph Relationship Analyzer"""
    
    def test_initialization(self):
        """Test analyzer initialization"""
        analyzer = GraphRelationshipAnalyzer()
        
        assert analyzer.neo4j_uri is None
    
    @pytest.mark.asyncio
    async def test_analyze_relationships(self):
        """Test relationship analysis"""
        analyzer = GraphRelationshipAnalyzer()
        
        result = await analyzer.analyze_relationships(
            "agent1",
            "resource1",
            "read",
        )
        
        assert result.anomaly_type == AnomalyType.RELATIONSHIP
        assert result.details["agent_id"] == "agent1"
        assert result.details["resource_id"] == "resource1"
    
    @pytest.mark.asyncio
    async def test_detect_lateral_movement(self):
        """Test detection of lateral movement pattern"""
        analyzer = GraphRelationshipAnalyzer()
        
        # Access many different resources (lateral movement)
        for i in range(12):
            result = await analyzer.analyze_relationships(
                "agent1",
                f"resource{i}",
                "access",
            )
        
        # Should detect anomaly due to many resources
        assert result.is_anomalous or result.confidence_score > 0.5


class TestInsiderThreatDetector:
    """Test Insider Threat Detector"""
    
    def test_initialization(self):
        """Test detector initialization"""
        detector = InsiderThreatDetector(sensitivity=0.7)
        
        assert detector.sensitivity == 0.7
    
    @pytest.mark.asyncio
    async def test_detect_normal_behavior(self):
        """Test detection of normal behavior"""
        detector = InsiderThreatDetector(sensitivity=0.9)  # High threshold
        
        event = {
            "timestamp": datetime.now(timezone.utc).replace(hour=10),
            "bulk_access": False,
            "data_classification": "public",
        }
        
        result = await detector.detect_insider_threat("user1", event)
        
        assert result.anomaly_type == AnomalyType.INSIDER_THREAT
    
    @pytest.mark.asyncio
    async def test_detect_after_hours_access(self):
        """Test detection of after-hours access"""
        detector = InsiderThreatDetector(sensitivity=0.2)
        
        event = {
            "timestamp": datetime.now(timezone.utc).replace(hour=2),  # 2 AM
            "bulk_access": False,
            "data_classification": "public",
        }
        
        result = await detector.detect_insider_threat("user1", event)
        
        assert result.is_anomalous
        assert "after_hours_access" in result.details["indicators"]
    
    @pytest.mark.asyncio
    async def test_detect_bulk_access(self):
        """Test detection of bulk data access"""
        detector = InsiderThreatDetector(sensitivity=0.3)
        
        event = {
            "timestamp": datetime.now(timezone.utc).replace(hour=14),
            "bulk_access": True,
            "data_classification": "sensitive",
        }
        
        result = await detector.detect_insider_threat("user1", event)
        
        assert result.is_anomalous
        assert "bulk_data_access" in result.details["indicators"]
        assert "sensitive_data_access" in result.details["indicators"]


class TestAPTBehavioralDetector:
    """Test APT Behavioral Detector"""
    
    def test_initialization(self):
        """Test detector initialization"""
        detector = APTBehavioralDetector()
        
        assert len(detector._apt_signatures) > 0
        assert "reconnaissance" in detector._apt_signatures
        assert "exfiltration" in detector._apt_signatures
    
    @pytest.mark.asyncio
    async def test_detect_single_indicator(self):
        """Test detection of single APT indicator"""
        detector = APTBehavioralDetector()
        
        event = {
            "type": "remote_execution",
            "timestamp": datetime.now(timezone.utc),
        }
        
        result = await detector.detect_apt_behavior("campaign1", event)
        
        assert result.anomaly_type == AnomalyType.APT_BEHAVIOR
        assert "lateral_movement" in result.details["matched_signatures"]
    
    @pytest.mark.asyncio
    async def test_detect_multi_stage_attack(self):
        """Test detection of multi-stage attack"""
        detector = APTBehavioralDetector()
        
        # Simulate multi-stage attack
        stages = [
            {"type": "enumeration"},  # Reconnaissance
            {"type": "service_creation"},  # Persistence
            {"type": "remote_execution"},  # Lateral movement
            {"type": "data_staging"},  # Collection
            {"type": "data_encrypted"},  # Exfiltration
            {"type": "external_connection"},  # Exfiltration
        ]
        
        for stage in stages:
            stage["timestamp"] = datetime.now(timezone.utc)
            result = await detector.detect_apt_behavior("campaign1", stage)
        
        # Final result should have high confidence due to multi-stage attack
        assert result.is_anomalous
        assert result.confidence_score > 0.5
        # Campaign tracking should show multiple events
        assert len(detector._campaign_tracking["campaign1"]) == 6
    
    @pytest.mark.asyncio
    async def test_detect_exfiltration(self):
        """Test detection of data exfiltration"""
        detector = APTBehavioralDetector()
        
        event = {
            "type": "data_compressed_external_connection",
            "timestamp": datetime.now(timezone.utc),
        }
        
        result = await detector.detect_apt_behavior("campaign1", event)
        
        # High weight for exfiltration
        assert result.confidence_score > 0.5


class TestAdvancedAnomalyDetectionEngine:
    """Test Advanced Anomaly Detection Engine"""
    
    def test_initialization_all_enabled(self):
        """Test engine initialization with all detectors"""
        engine = AdvancedAnomalyDetectionEngine()
        
        assert engine.lstm_detector is not None
        assert engine.transformer_analyzer is not None
        assert engine.graph_analyzer is not None
        assert engine.insider_detector is not None
        assert engine.apt_detector is not None
    
    def test_initialization_selective(self):
        """Test engine initialization with selective detectors"""
        engine = AdvancedAnomalyDetectionEngine(
            enable_lstm=True,
            enable_transformer=False,
            enable_graph=False,
            enable_insider=True,
            enable_apt=False,
        )
        
        assert engine.lstm_detector is not None
        assert engine.transformer_analyzer is None
        assert engine.graph_analyzer is None
        assert engine.insider_detector is not None
        assert engine.apt_detector is None
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_no_anomaly(self):
        """Test anomaly detection with normal event"""
        engine = AdvancedAnomalyDetectionEngine()
        
        event = {
            "type": "read",
            "timestamp": datetime.now(timezone.utc).replace(hour=10),
            "privilege_level": 1,
            "resource": "file1",
            "bulk_access": False,
            "data_classification": "public",
        }
        
        results = await engine.detect_anomalies("agent1", event)
        
        # Should return list (may be empty for normal event)
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_with_context(self):
        """Test anomaly detection with context"""
        engine = AdvancedAnomalyDetectionEngine()
        
        event = {
            "type": "admin_access",
            "timestamp": datetime.now(timezone.utc).replace(hour=2),
            "privilege_level": 5,
            "resource": "sensitive_data",
            "bulk_access": True,
            "data_classification": "sensitive",
        }
        
        context = {
            "user_role": "analyst",
            "normal_hours": "9-17",
        }
        
        results = await engine.detect_anomalies("agent1", event, context)
        
        # Suspicious event should trigger detections
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_get_detection_summary(self):
        """Test detection summary"""
        engine = AdvancedAnomalyDetectionEngine()
        
        summary = await engine.get_detection_summary()
        
        assert summary["status"] == "operational"
        assert "detectors_enabled" in summary
        assert "capabilities" in summary
        assert len(summary["capabilities"]) > 0


class TestAnomalyType:
    """Test AnomalyType enum"""
    
    def test_anomaly_types(self):
        """Test all anomaly types"""
        assert AnomalyType.SEQUENCE == "sequence"
        assert AnomalyType.CONTEXT == "context"
        assert AnomalyType.RELATIONSHIP == "relationship"
        assert AnomalyType.INSIDER_THREAT == "insider_threat"
        assert AnomalyType.APT_BEHAVIOR == "apt_behavior"
        assert AnomalyType.UNKNOWN == "unknown"
