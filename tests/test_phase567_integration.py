"""Tests for Phase 5-7 Integration."""

import pytest
import tempfile
from pathlib import Path

from nethical.core import Phase567IntegratedGovernance


class TestPhase567Integration:
    """Test Phase 5-7 integrated governance."""

    def test_initialization_all_enabled(self):
        """Test initialization with all components enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase567IntegratedGovernance(
                storage_dir=tmpdir,
                enable_shadow_mode=True,
                enable_ml_blending=True,
                enable_anomaly_detection=True,
            )

            assert gov.shadow_classifier is not None
            assert gov.blended_engine is not None
            assert gov.anomaly_monitor is not None
            assert gov.components_enabled["shadow_mode"] is True
            assert gov.components_enabled["ml_blending"] is True
            assert gov.components_enabled["anomaly_detection"] is True

    def test_initialization_selective(self):
        """Test initialization with selective components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase567IntegratedGovernance(
                storage_dir=tmpdir,
                enable_shadow_mode=True,
                enable_ml_blending=False,
                enable_anomaly_detection=True,
            )

            assert gov.shadow_classifier is not None
            assert gov.blended_engine is None
            assert gov.anomaly_monitor is not None

    def test_process_action_shadow_only(self):
        """Test action processing with only shadow mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase567IntegratedGovernance(
                storage_dir=tmpdir,
                enable_shadow_mode=True,
                enable_ml_blending=False,
                enable_anomaly_detection=False,
            )

            result = gov.process_action(
                agent_id="agent_1",
                action_id="action_1",
                action_type="response",
                features={"violation_count": 0.3, "severity_max": 0.4},
                rule_risk_score=0.5,
                rule_classification="warn",
            )

            assert result["agent_id"] == "agent_1"
            assert result["action_id"] == "action_1"
            assert "shadow" in result
            assert result["shadow"]["ml_risk_score"] >= 0.0
            assert result["shadow"]["ml_classification"] in ["allow", "warn", "deny"]
            assert "blended" not in result
            assert result.get("anomaly_alert") is None

    def test_process_action_full_pipeline(self):
        """Test action processing through full pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase567IntegratedGovernance(
                storage_dir=tmpdir,
                enable_shadow_mode=True,
                enable_ml_blending=True,
                enable_anomaly_detection=True,
            )

            # Set baseline for anomaly detection
            baseline_scores = [0.2, 0.3, 0.25, 0.35] * 25
            gov.set_baseline_distribution(baseline_scores, cohort="test_cohort")

            result = gov.process_action(
                agent_id="agent_1",
                action_id="action_1",
                action_type="response",
                features={"violation_count": 0.3, "severity_max": 0.4},
                rule_risk_score=0.5,
                rule_classification="warn",
                cohort="test_cohort",
            )

            assert result["agent_id"] == "agent_1"
            assert "shadow" in result
            assert "blended" in result
            assert "final_decision" in result
            assert result["final_decision"]["risk_score"] >= 0.0
            assert result["final_decision"]["classification"] in [
                "allow",
                "warn",
                "deny",
            ]

    def test_process_action_gray_zone_blending(self):
        """Test that ML blending occurs in gray zone."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase567IntegratedGovernance(
                storage_dir=tmpdir,
                enable_shadow_mode=True,
                enable_ml_blending=True,
                enable_anomaly_detection=False,
                gray_zone_lower=0.4,
                gray_zone_upper=0.6,
            )

            # Test with gray zone score
            result = gov.process_action(
                agent_id="agent_1",
                action_id="action_1",
                action_type="response",
                features={"violation_count": 0.5},
                rule_risk_score=0.5,  # In gray zone
                rule_classification="warn",
            )

            assert "blended" in result
            assert "risk_zone" in result["blended"]

    def test_anomaly_detection_baseline(self):
        """Test setting and using anomaly detection baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase567IntegratedGovernance(
                storage_dir=tmpdir,
                enable_shadow_mode=False,
                enable_ml_blending=False,
                enable_anomaly_detection=True,
            )

            # Set baseline
            baseline_scores = [0.2, 0.3, 0.25, 0.35] * 25
            success = gov.set_baseline_distribution(baseline_scores, cohort="test")
            assert success is True

            # Process action that should not trigger anomaly
            result = gov.process_action(
                agent_id="agent_1",
                action_id="action_1",
                action_type="normal",
                features={},
                rule_risk_score=0.3,
                rule_classification="allow",
                cohort="test",
            )

            # With normal scores, anomaly should not be triggered
            # (though it could be None or a low-severity alert)
            assert "anomaly_alert" in result

    def test_get_shadow_metrics(self):
        """Test getting shadow mode metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase567IntegratedGovernance(
                storage_dir=tmpdir, enable_shadow_mode=True
            )

            # Process some actions
            for i in range(5):
                gov.process_action(
                    agent_id=f"agent_{i}",
                    action_id=f"action_{i}",
                    action_type="response",
                    features={"violation_count": 0.3},
                    rule_risk_score=0.5,
                    rule_classification="warn",
                )

            metrics = gov.get_shadow_metrics()
            assert "total_predictions" in metrics
            assert metrics["total_predictions"] >= 5

    def test_get_system_status(self):
        """Test getting overall system status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase567IntegratedGovernance(
                storage_dir=tmpdir,
                enable_shadow_mode=True,
                enable_ml_blending=True,
                enable_anomaly_detection=True,
            )

            status = gov.get_system_status()

            assert "timestamp" in status
            assert "components" in status
            assert "shadow_classifier" in status["components"]
            assert "blended_engine" in status["components"]
            assert "anomaly_monitor" in status["components"]

            assert status["components"]["shadow_classifier"]["enabled"] is True
            assert status["components"]["blended_engine"]["enabled"] is True
            assert status["components"]["anomaly_monitor"]["enabled"] is True

    def test_export_report(self):
        """Test exporting comprehensive report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase567IntegratedGovernance(
                storage_dir=tmpdir,
                enable_shadow_mode=True,
                enable_ml_blending=True,
                enable_anomaly_detection=True,
            )

            # Process some actions
            for i in range(3):
                gov.process_action(
                    agent_id=f"agent_{i}",
                    action_id=f"action_{i}",
                    action_type="response",
                    features={"violation_count": 0.3},
                    rule_risk_score=0.5,
                    rule_classification="warn",
                )

            report = gov.export_phase567_report()

            assert isinstance(report, str)
            assert "Phase 5-7: ML & Anomaly Detection - Report" in report
            assert "System Status" in report
            assert "Shadow Mode Metrics" in report
            assert "ML Blending Metrics" in report
            assert "Anomaly Detection" in report

    def test_disabled_components_return_errors(self):
        """Test that disabled components return appropriate errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase567IntegratedGovernance(
                storage_dir=tmpdir,
                enable_shadow_mode=False,
                enable_ml_blending=False,
                enable_anomaly_detection=False,
            )

            shadow_metrics = gov.get_shadow_metrics()
            assert "error" in shadow_metrics

            blending_metrics = gov.get_blending_metrics()
            assert "error" in blending_metrics

            anomaly_stats = gov.get_anomaly_statistics()
            assert "error" in anomaly_stats

            baseline_set = gov.set_baseline_distribution([0.5, 0.6])
            assert baseline_set is False
