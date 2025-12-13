"""
Comprehensive Test Suite for Phase 5: Detection Omniscience.

Tests all Phase 5 components:
- Threat Intelligence Integration
- Predictive Modeling
- Proactive Hardening
- Detector Formal Verification
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta

# Phase 5 imports
from nethical.ml.threat_intelligence import (
    ThreatFeedIntegrator,
    ThreatSource,
    ThreatIntelligence,
    ThreatSeverity,
    PredictiveModeler,
    AttackPrediction,
    ProactiveHardener,
    HardeningPriority,
)

from nethical.core.verification import (
    DetectorVerifier,
    DetectorProperty,
    VerificationStatus,
)


# ============================================================================
# Threat Intelligence Integration Tests
# ============================================================================

class TestThreatFeedIntegrator:
    """Test threat feed integration."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test integrator initialization."""
        integrator = ThreatFeedIntegrator()
        
        assert integrator is not None
        assert len(integrator.sources) > 0
        assert integrator.active_threats == 0
        assert integrator.total_threats_ingested == 0
    
    @pytest.mark.asyncio
    async def test_ingest_threat(self):
        """Test threat ingestion."""
        integrator = ThreatFeedIntegrator()
        
        threat = ThreatIntelligence(
            threat_id="test_threat_001",
            source=ThreatSource.CVE_DATABASE,
            severity=ThreatSeverity.HIGH,
            title="Test Vulnerability",
            description="A test vulnerability for demonstration",
            indicators=["pattern1", "pattern2"],
            attack_vectors=["prompt_injection"],
            confidence=0.9,
        )
        
        result = await integrator.ingest_threat(threat)
        
        assert result["status"] == "success"
        assert result["threat_id"] == "test_threat_001"
        assert integrator.active_threats == 1
        assert integrator.total_threats_ingested == 1
    
    @pytest.mark.asyncio
    async def test_duplicate_threat_handling(self):
        """Test handling of duplicate threats."""
        integrator = ThreatFeedIntegrator()
        
        threat1 = ThreatIntelligence(
            threat_id="dup_threat",
            source=ThreatSource.CVE_DATABASE,
            severity=ThreatSeverity.MEDIUM,
            title="Duplicate Threat",
            description="First version",
            indicators=["ind1"],
            confidence=0.7,
        )
        
        threat2 = ThreatIntelligence(
            threat_id="dup_threat",  # Same ID
            source=ThreatSource.AI_RESEARCH_FEEDS,
            severity=ThreatSeverity.HIGH,
            title="Duplicate Threat",
            description="Updated version",
            indicators=["ind2"],
            confidence=0.9,
        )
        
        result1 = await integrator.ingest_threat(threat1)
        result2 = await integrator.ingest_threat(threat2)
        
        assert result1["action"] == "created"
        assert result2["action"] == "updated"
        assert integrator.active_threats == 1  # Only one threat stored
        
        # Check merged data
        stored_threat = integrator.threats["dup_threat"]
        assert len(stored_threat.indicators) == 2  # Both indicators
        assert stored_threat.confidence == 0.9  # Higher confidence
    
    @pytest.mark.asyncio
    async def test_get_threats_by_severity(self):
        """Test filtering threats by severity."""
        integrator = ThreatFeedIntegrator()
        
        # Add threats of different severities
        for i, severity in enumerate([ThreatSeverity.CRITICAL, ThreatSeverity.HIGH, ThreatSeverity.LOW]):
            threat = ThreatIntelligence(
                threat_id=f"threat_{i}",
                source=ThreatSource.CVE_DATABASE,
                severity=severity,
                title=f"Threat {i}",
                description=f"Test threat {i}",
                confidence=0.8,
            )
            await integrator.ingest_threat(threat)
        
        critical_threats = await integrator.get_threats_by_severity(ThreatSeverity.CRITICAL)
        high_threats = await integrator.get_threats_by_severity(ThreatSeverity.HIGH)
        
        assert len(critical_threats) == 1
        assert len(high_threats) == 1
    
    @pytest.mark.asyncio
    async def test_search_threats(self):
        """Test threat search functionality."""
        integrator = ThreatFeedIntegrator()
        
        threat1 = ThreatIntelligence(
            threat_id="search_test_1",
            source=ThreatSource.CVE_DATABASE,
            severity=ThreatSeverity.HIGH,
            title="Prompt Injection Vulnerability",
            description="SQL injection through prompts",
            attack_vectors=["prompt_injection"],
            confidence=0.9,
        )
        
        threat2 = ThreatIntelligence(
            threat_id="search_test_2",
            source=ThreatSource.AI_RESEARCH_FEEDS,
            severity=ThreatSeverity.MEDIUM,
            title="Model Extraction Attack",
            description="Extracting model weights",
            attack_vectors=["model_extraction"],
            confidence=0.6,
        )
        
        await integrator.ingest_threat(threat1)
        await integrator.ingest_threat(threat2)
        
        # Search by keyword
        results = await integrator.search_threats(keywords=["injection"])
        assert len(results) == 1
        assert results[0].threat_id == "search_test_1"
        
        # Search by attack vector
        results = await integrator.search_threats(attack_vectors=["model_extraction"])
        assert len(results) == 1
        assert results[0].threat_id == "search_test_2"
        
        # Search by confidence
        results = await integrator.search_threats(min_confidence=0.85)
        assert len(results) == 1
        assert results[0].threat_id == "search_test_1"
    
    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test statistics retrieval."""
        integrator = ThreatFeedIntegrator()
        
        # Add some threats
        for i in range(3):
            threat = ThreatIntelligence(
                threat_id=f"stat_threat_{i}",
                source=ThreatSource.CVE_DATABASE,
                severity=ThreatSeverity.HIGH,
                title=f"Threat {i}",
                description="Test",
                confidence=0.8,
            )
            await integrator.ingest_threat(threat)
        
        stats = integrator.get_statistics()
        
        assert stats["total_threats_ingested"] == 3
        assert stats["active_threats"] == 3
        assert "threats_by_severity" in stats
        assert "threats_by_source" in stats


# ============================================================================
# Predictive Modeling Tests
# ============================================================================

class TestPredictiveModeler:
    """Test predictive modeling."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test modeler initialization."""
        modeler = PredictiveModeler()
        
        assert modeler is not None
        assert modeler.prediction_threshold == 0.7
        assert len(modeler.time_horizons) > 0
        assert modeler.total_predictions == 0
    
    @pytest.mark.asyncio
    async def test_analyze_trends(self):
        """Test trend analysis."""
        modeler = PredictiveModeler()
        
        # Create historical attack data
        attack_history = [
            {
                "type": "prompt_injection",
                "timestamp": datetime.now(timezone.utc) - timedelta(days=i),
            }
            for i in range(14)  # 2 weeks of data
        ]
        
        trends = await modeler.analyze_trends(attack_history)
        
        assert "prompt_injection" in trends
        # Verify trend is a valid ThreatTrend value
        from nethical.ml.threat_intelligence.predictive_modeling import ThreatTrend
        assert trends["prompt_injection"] in ThreatTrend
    
    @pytest.mark.asyncio
    async def test_predict_attacks(self):
        """Test attack prediction."""
        modeler = PredictiveModeler(prediction_threshold=0.5)
        
        # Create mock threat intelligence
        threat = ThreatIntelligence(
            threat_id="pred_threat",
            source=ThreatSource.AI_RESEARCH_FEEDS,
            severity=ThreatSeverity.HIGH,
            title="Emerging Attack",
            description="New attack pattern",
            attack_vectors=["prompt_injection"],
            indicators=["pattern1", "pattern2"],
            confidence=0.8,
        )
        
        predictions = await modeler.predict_attacks([threat])
        
        assert len(predictions) > 0
        
        # Check prediction structure
        pred = predictions[0]
        assert pred.attack_type is not None
        assert 0.0 <= pred.probability <= 1.0
        assert 0.0 <= pred.confidence <= 1.0
        assert pred.time_horizon_days > 0
    
    @pytest.mark.asyncio
    async def test_model_threat_evolution(self):
        """Test threat evolution modeling."""
        modeler = PredictiveModeler()
        
        # Create historical data
        historical_data = [
            {
                "family": "prompt_injection",
                "variant": f"variant_{i}",
                "timestamp": datetime.now(timezone.utc) - timedelta(days=i*30),
                "sophistication": 0.5 + (i * 0.1),
            }
            for i in range(5)
        ]
        
        model = await modeler.model_threat_evolution(
            "prompt_injection", historical_data
        )
        
        assert model.threat_family == "prompt_injection"
        assert len(model.current_variants) > 0
        assert model.mutation_rate >= 0.0
        assert model.sophistication_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_validate_prediction(self):
        """Test prediction validation."""
        modeler = PredictiveModeler(prediction_threshold=0.7)
        
        # Create a prediction
        threat = ThreatIntelligence(
            threat_id="val_threat",
            source=ThreatSource.CVE_DATABASE,
            severity=ThreatSeverity.HIGH,
            title="Test",
            description="Test",
            attack_vectors=["test"],
            confidence=0.9,
        )
        
        predictions = await modeler.predict_attacks([threat])
        
        if predictions:
            pred = predictions[0]
            
            # Validate as true positive
            result = await modeler.validate_prediction(
                pred.prediction_id, actually_occurred=True
            )
            
            assert result["status"] == "success"
            assert result["result"] == "true_positive"
            assert modeler.accurate_predictions == 1
    
    @pytest.mark.asyncio
    async def test_accuracy_metrics(self):
        """Test accuracy metrics."""
        modeler = PredictiveModeler()
        
        metrics = modeler.get_accuracy_metrics()
        
        assert "total_predictions" in metrics
        assert "accurate_predictions" in metrics
        assert "false_positives" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics


# ============================================================================
# Proactive Hardening Tests
# ============================================================================

class TestProactiveHardener:
    """Test proactive hardening."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test hardener initialization."""
        hardener = ProactiveHardener()
        
        assert hardener is not None
        assert hardener.auto_deploy_threshold == 0.8
        assert hardener.max_concurrent_deployments == 5
        assert hardener.total_actions_created == 0
    
    @pytest.mark.asyncio
    async def test_create_hardening_action(self):
        """Test hardening action creation."""
        hardener = ProactiveHardener()
        
        # Create a mock prediction
        prediction = AttackPrediction(
            prediction_id="test_pred",
            attack_type="prompt_injection",
            predicted_vector="PI-013",
            probability=0.75,
            confidence=0.8,
            time_horizon_days=30,
            indicators=["ind1", "ind2"],
            recommended_defenses=["Deploy detector", "Update rules"],
        )
        
        action = await hardener.create_hardening_action(prediction)
        
        assert action is not None
        assert action.action_id.startswith("harden_")
        assert action.priority in HardeningPriority
        assert action.threat_probability == 0.75
        assert len(action.deployment_steps) > 0
        assert hardener.total_actions_created == 1
    
    @pytest.mark.asyncio
    async def test_approve_action(self):
        """Test action approval."""
        hardener = ProactiveHardener()
        
        # Create action with lower probability to require approval
        prediction = AttackPrediction(
            prediction_id="approve_test",
            attack_type="test_attack",
            predicted_vector="TEST-001",
            probability=0.6,
            confidence=0.7,
            time_horizon_days=90,
        )
        
        action = await hardener.create_hardening_action(prediction)
        
        # Approve the action
        result = await hardener.approve_action(action.action_id, "admin")
        
        assert result["status"] == "success"
        assert hardener.actions[action.action_id].approved_by == "admin"
    
    @pytest.mark.asyncio
    async def test_deploy_action(self):
        """Test action deployment."""
        hardener = ProactiveHardener(auto_deploy_threshold=1.0)  # Disable auto-deploy
        
        prediction = AttackPrediction(
            prediction_id="deploy_test",
            attack_type="test_attack",
            predicted_vector="TEST-002",
            probability=0.75,
            confidence=0.8,
            time_horizon_days=30,
        )
        
        action = await hardener.create_hardening_action(prediction)
        await hardener.approve_action(action.action_id, "admin")
        
        # Deploy the action
        result = await hardener.deploy_action(action.action_id)
        
        assert result["status"] == "success"
        assert hardener.successful_deployments == 1
        assert action.action_id in hardener.deployed_actions
    
    @pytest.mark.asyncio
    async def test_rollback_action(self):
        """Test action rollback."""
        hardener = ProactiveHardener(auto_deploy_threshold=1.0)
        
        prediction = AttackPrediction(
            prediction_id="rollback_test",
            attack_type="test_attack",
            predicted_vector="TEST-003",
            probability=0.8,
            confidence=0.85,
            time_horizon_days=7,
        )
        
        action = await hardener.create_hardening_action(prediction)
        await hardener.approve_action(action.action_id, "admin")
        await hardener.deploy_action(action.action_id)
        
        # Rollback the action
        result = await hardener.rollback_action(
            action.action_id, "Testing rollback"
        )
        
        assert result["status"] == "success"
        assert hardener.rollbacks == 1
        assert action.action_id not in hardener.deployed_actions
    
    @pytest.mark.asyncio
    async def test_get_pending_actions(self):
        """Test retrieving pending actions."""
        hardener = ProactiveHardener(auto_deploy_threshold=1.0)
        
        # Create multiple actions with different priorities
        for i, prob in enumerate([0.9, 0.7, 0.5]):
            prediction = AttackPrediction(
                prediction_id=f"pending_{i}",
                attack_type=f"attack_{i}",
                predicted_vector=f"TEST-{i}",
                probability=prob,
                confidence=0.8,
                time_horizon_days=30,
            )
            await hardener.create_hardening_action(prediction)
        
        pending = await hardener.get_pending_actions()
        
        assert len(pending) > 0
        # Check they're sorted by priority
        for i in range(len(pending) - 1):
            assert pending[i].priority.value <= pending[i + 1].priority.value
    
    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test statistics retrieval."""
        hardener = ProactiveHardener()
        
        stats = hardener.get_statistics()
        
        assert "total_actions_created" in stats
        assert "successful_deployments" in stats
        assert "failed_deployments" in stats
        assert "rollbacks" in stats
        assert "actions_by_status" in stats
        assert "actions_by_priority" in stats


# ============================================================================
# Detector Verification Tests
# ============================================================================

class TestDetectorVerifier:
    """Test detector formal verification."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test verifier initialization."""
        verifier = DetectorVerifier()
        
        assert verifier is not None
        assert verifier.enable_runtime_monitoring is True
        assert verifier.total_verifications == 0
    
    @pytest.mark.asyncio
    async def test_verify_detector(self):
        """Test detector verification."""
        verifier = DetectorVerifier()
        
        results = await verifier.verify_detector(
            "test_detector",
            properties=[DetectorProperty.DETERMINISTIC_BEHAVIOR],
        )
        
        assert len(results) == 1
        assert results[0].detector_id == "test_detector"
        assert results[0].property == DetectorProperty.DETERMINISTIC_BEHAVIOR
        assert results[0].status in VerificationStatus
        assert verifier.total_verifications == 1
    
    @pytest.mark.asyncio
    async def test_verify_all_properties(self):
        """Test verifying all properties."""
        verifier = DetectorVerifier()
        
        results = await verifier.verify_detector("test_detector_all")
        
        # Should verify all properties
        assert len(results) == len(DetectorProperty)
        assert verifier.total_verifications == len(DetectorProperty)
    
    @pytest.mark.asyncio
    async def test_runtime_monitoring(self):
        """Test runtime property monitoring."""
        verifier = DetectorVerifier(enable_runtime_monitoring=True)
        
        # First verify a property
        await verifier.verify_detector(
            "monitored_detector",
            properties=[DetectorProperty.DETERMINISTIC_BEHAVIOR],
        )
        
        # Then monitor at runtime
        detection_result = {
            "violations": [],
            "confidence": 0.9,
        }
        
        result = await verifier.monitor_runtime_property(
            "monitored_detector",
            DetectorProperty.DETERMINISTIC_BEHAVIOR,
            detection_result,
        )
        
        assert result["status"] in ["ok", "violation", "not_monitored"]
    
    @pytest.mark.asyncio
    async def test_get_verification_status(self):
        """Test getting verification status."""
        verifier = DetectorVerifier()
        
        # Verify some properties
        await verifier.verify_detector(
            "status_test_detector",
            properties=[
                DetectorProperty.DETERMINISTIC_BEHAVIOR,
                DetectorProperty.BOUNDED_FALSE_POSITIVES,
            ],
        )
        
        status = await verifier.get_verification_status("status_test_detector")
        
        assert status["detector_id"] == "status_test_detector"
        assert status["properties_verified"] >= 0
        assert status["total_properties"] == 2
    
    @pytest.mark.asyncio
    async def test_statistics(self):
        """Test statistics retrieval."""
        verifier = DetectorVerifier()
        
        # Verify a detector
        await verifier.verify_detector(
            "stats_detector",
            properties=[DetectorProperty.DETERMINISTIC_BEHAVIOR],
        )
        
        stats = verifier.get_statistics()
        
        assert "total_verifications" in stats
        assert "verified_count" in stats
        assert "failed_count" in stats
        assert "runtime_violations" in stats
        assert stats["total_verifications"] >= 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase5Integration:
    """Integration tests for Phase 5 components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_threat_prediction_hardening(self):
        """Test complete flow from threat intelligence to hardening."""
        # Step 1: Ingest threat intelligence
        integrator = ThreatFeedIntegrator()
        
        threat = ThreatIntelligence(
            threat_id="integration_threat",
            source=ThreatSource.AI_RESEARCH_FEEDS,
            severity=ThreatSeverity.HIGH,
            title="Emerging Attack Pattern",
            description="New sophisticated attack",
            attack_vectors=["prompt_injection"],
            indicators=["indicator1", "indicator2"],
            confidence=0.9,
        )
        
        await integrator.ingest_threat(threat)
        
        # Step 2: Generate predictions
        modeler = PredictiveModeler(prediction_threshold=0.5)
        high_threats = await integrator.get_threats_by_severity(ThreatSeverity.HIGH)
        predictions = await modeler.predict_attacks(high_threats)
        
        assert len(predictions) > 0
        
        # Step 3: Create hardening actions
        hardener = ProactiveHardener(auto_deploy_threshold=1.0)
        action = await hardener.create_hardening_action(predictions[0])
        
        assert action is not None
        
        # Step 4: Deploy hardening
        await hardener.approve_action(action.action_id, "integration_test")
        result = await hardener.deploy_action(action.action_id)
        
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_verification_integration(self):
        """Test verification integration with threat response."""
        verifier = DetectorVerifier()
        
        # Verify a detector that would handle predicted threats
        results = await verifier.verify_detector(
            "integration_detector",
            properties=[
                DetectorProperty.NO_FALSE_NEGATIVES_CRITICAL,
                DetectorProperty.BOUNDED_FALSE_POSITIVES,
            ],
        )
        
        verified = sum(
            1 for r in results if r.status == VerificationStatus.VERIFIED
        )
        
        # At least one property should be verified
        assert verified > 0
        
        # Get verification status
        status = await verifier.get_verification_status("integration_detector")
        assert status["status"] in ["verified", "partial"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
