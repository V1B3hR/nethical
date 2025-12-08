"""Tests for Unified Integrated Governance."""

import pytest
from nethical.core import IntegratedGovernance


class TestIntegratedGovernance:
    """Test suite for unified IntegratedGovernance class."""

    def test_initialization(self):
        """Test that unified governance initializes all components."""
        gov = IntegratedGovernance(
            storage_dir="./test_unified_data",
            enable_performance_optimization=True,
            enable_shadow_mode=True,
            enable_ml_blending=True,
            enable_anomaly_detection=True,
        )

        # Verify all phase components are initialized
        assert gov.risk_engine is not None
        assert gov.correlation_engine is not None
        assert gov.fairness_sampler is not None
        assert gov.ethical_drift_reporter is not None
        assert gov.performance_optimizer is not None

        assert gov.merkle_anchor is not None
        assert gov.policy_auditor is not None
        assert gov.quarantine_manager is not None
        assert gov.ethical_taxonomy is not None
        assert gov.sla_monitor is not None

        assert gov.shadow_classifier is not None
        assert gov.blended_engine is not None
        assert gov.anomaly_monitor is not None

        assert gov.escalation_queue is not None
        assert gov.optimizer is not None

    def test_components_enabled_flags(self):
        """Test that component flags are properly set."""
        gov = IntegratedGovernance(
            storage_dir="./test_unified_data2",
            enable_performance_optimization=False,
            enable_shadow_mode=False,
            enable_ml_blending=False,
            enable_anomaly_detection=False,
            enable_merkle_anchoring=False,
            enable_quarantine=False,
            enable_ethical_taxonomy=False,
            enable_sla_monitoring=False,
        )

        assert gov.components_enabled["performance_optimizer"] is False
        assert gov.components_enabled["shadow_mode"] is False
        assert gov.components_enabled["ml_blending"] is False
        assert gov.components_enabled["anomaly_detection"] is False
        assert gov.components_enabled["merkle_anchoring"] is False
        assert gov.components_enabled["quarantine"] is False
        assert gov.components_enabled["ethical_taxonomy"] is False
        assert gov.components_enabled["sla_monitoring"] is False

    def test_process_action_basic(self):
        """Test basic action processing through unified governance."""
        gov = IntegratedGovernance(
            storage_dir="./test_unified_data3", enable_performance_optimization=True
        )

        result = gov.process_action(
            agent_id="agent_123",
            action="test action",
            cohort="test_cohort",
            violation_detected=False,
        )

        # Verify we get results from all phases
        assert "phase3" in result
        assert "phase4" in result
        assert "phase567" in result
        assert "phase89" in result

        # Phase 3 results
        assert "risk_score" in result["phase3"]
        assert "risk_tier" in result["phase3"]
        assert "correlations" in result["phase3"]

        # Phase 4 results
        assert "merkle" in result["phase4"]
        assert "latency_ms" in result["phase4"]

    def test_process_action_with_ml_features(self):
        """Test action processing with ML features."""
        gov = IntegratedGovernance(
            storage_dir="./test_unified_data4",
            enable_shadow_mode=True,
            enable_ml_blending=True,
            enable_anomaly_detection=True,
        )

        result = gov.process_action(
            agent_id="agent_123",
            action="test action",
            cohort="test_cohort",
            violation_detected=True,
            violation_type="safety",
            violation_severity="high",
            action_id="action_456",
            action_type="response",
            features={"violation_count": 0.5, "severity_max": 0.6, "ml_score": 0.7},
            rule_risk_score=0.65,
            rule_classification="warn",
        )

        # Verify ML phase results
        assert "phase567" in result
        assert "shadow" in result["phase567"]
        assert "blended" in result["phase567"]
        assert "ml_risk_score" in result["phase567"]["shadow"]
        assert "blended_risk_score" in result["phase567"]["blended"]

    def test_get_system_status(self):
        """Test system status retrieval."""
        gov = IntegratedGovernance(storage_dir="./test_unified_data5")

        status = gov.get_system_status()

        assert "timestamp" in status
        assert "components_enabled" in status
        assert "phase3" in status
        assert "phase4" in status
        assert "phase567" in status
        assert "phase89" in status

        # Verify component status
        assert "risk_engine" in status["phase3"]
        assert "merkle_anchor" in status["phase4"]
        assert "shadow_classifier" in status["phase567"]
        assert "escalation_queue" in status["phase89"]

    def test_selective_component_initialization(self):
        """Test that we can selectively enable/disable components."""
        gov = IntegratedGovernance(
            storage_dir="./test_unified_data6",
            enable_performance_optimization=False,
            enable_merkle_anchoring=False,
            enable_shadow_mode=False,
        )

        assert gov.performance_optimizer is None
        assert gov.merkle_anchor is None
        assert gov.shadow_classifier is None

        # But other components should still be initialized
        assert gov.risk_engine is not None
        assert gov.escalation_queue is not None
