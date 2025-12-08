"""Tests for Phase 6: ML Assisted Enforcement."""

import pytest
import tempfile
from pathlib import Path

from nethical.core import (
    MLBlendedRiskEngine,
    BlendedDecision,
    BlendingMetrics,
    RiskZone,
)


class TestMLBlendedRiskEngine:
    """Test ML blended risk engine."""

    def test_initialization(self):
        """Test blended engine initialization."""
        engine = MLBlendedRiskEngine(
            gray_zone_lower=0.4, gray_zone_upper=0.6, rule_weight=0.7, ml_weight=0.3
        )

        assert engine.gray_zone_lower == 0.4
        assert engine.gray_zone_upper == 0.6
        assert engine.rule_weight == 0.7
        assert engine.ml_weight == 0.3
        assert engine.enable_ml_blending is True

    def test_weight_validation(self):
        """Test weight sum validation."""
        with pytest.raises(ValueError):
            MLBlendedRiskEngine(rule_weight=0.6, ml_weight=0.3)  # Doesn't sum to 1.0

    def test_clear_allow_zone(self):
        """Test decision in clear allow zone (no ML influence)."""
        engine = MLBlendedRiskEngine()

        decision = engine.compute_blended_risk(
            agent_id="agent_001",
            action_id="action_001",
            rule_risk_score=0.2,
            rule_classification="allow",
            ml_risk_score=0.8,  # ML disagrees, but should be ignored
            ml_confidence=0.9,
        )

        assert decision.risk_zone == RiskZone.CLEAR_ALLOW
        assert decision.ml_influenced is False
        assert decision.blended_risk_score == 0.2  # Uses rule score only
        assert decision.rule_weight == 1.0
        assert decision.ml_weight == 0.0

    def test_gray_zone_with_ml(self):
        """Test decision in gray zone with ML influence."""
        engine = MLBlendedRiskEngine(
            gray_zone_lower=0.4, gray_zone_upper=0.6, rule_weight=0.7, ml_weight=0.3
        )

        decision = engine.compute_blended_risk(
            agent_id="agent_001",
            action_id="action_001",
            rule_risk_score=0.5,
            rule_classification="warn",
            ml_risk_score=0.7,
            ml_confidence=0.85,
        )

        assert decision.risk_zone == RiskZone.GRAY_ZONE
        assert decision.ml_influenced is True

        # Check blending formula
        expected = 0.7 * 0.5 + 0.3 * 0.7
        assert abs(decision.blended_risk_score - expected) < 0.001

        assert decision.rule_weight == 0.7
        assert decision.ml_weight == 0.3

    def test_clear_deny_zone(self):
        """Test decision in clear deny zone (no ML influence)."""
        engine = MLBlendedRiskEngine()

        decision = engine.compute_blended_risk(
            agent_id="agent_001",
            action_id="action_001",
            rule_risk_score=0.8,
            rule_classification="deny",
            ml_risk_score=0.2,  # ML disagrees, but should be ignored
            ml_confidence=0.9,
        )

        assert decision.risk_zone == RiskZone.CLEAR_DENY
        assert decision.ml_influenced is False
        assert decision.blended_risk_score == 0.8

    def test_gray_zone_without_ml(self):
        """Test gray zone when ML score not available."""
        engine = MLBlendedRiskEngine()

        decision = engine.compute_blended_risk(
            agent_id="agent_001",
            action_id="action_001",
            rule_risk_score=0.5,
            rule_classification="warn",
            ml_risk_score=None,  # ML not available
            ml_confidence=None,
        )

        assert decision.risk_zone == RiskZone.GRAY_ZONE
        assert decision.ml_influenced is False
        assert decision.blended_risk_score == 0.5  # Falls back to rule score

    def test_classification_change_tracking(self):
        """Test tracking of classification changes."""
        engine = MLBlendedRiskEngine(rule_weight=0.5, ml_weight=0.5)

        # Scenario where ML pushes classification up
        decision = engine.compute_blended_risk(
            agent_id="agent_001",
            action_id="action_001",
            rule_risk_score=0.45,  # "warn" at boundary
            rule_classification="warn",
            ml_risk_score=0.75,  # "deny"
            ml_confidence=0.9,
        )

        # Blended: 0.5 * 0.45 + 0.5 * 0.75 = 0.6 ("warn")
        # If it changed to deny, should track it
        if decision.blended_classification != decision.rule_classification:
            assert decision.classification_changed is True

    def test_metrics_tracking(self):
        """Test metrics are tracked correctly."""
        engine = MLBlendedRiskEngine()

        # Generate decisions in different zones
        engine.compute_blended_risk("a1", "act1", 0.2, "allow", 0.3, 0.8)  # Clear allow
        engine.compute_blended_risk("a2", "act2", 0.5, "warn", 0.6, 0.8)  # Gray zone
        engine.compute_blended_risk("a3", "act3", 0.8, "deny", 0.9, 0.8)  # Clear deny

        assert engine.metrics.total_decisions == 3
        assert engine.metrics.clear_allow_count == 1
        assert engine.metrics.gray_zone_count == 1
        assert engine.metrics.clear_deny_count == 1
        assert engine.metrics.ml_influenced_count == 1  # Only gray zone

    def test_export_decisions(self):
        """Test decision export functionality."""
        engine = MLBlendedRiskEngine()

        # Add some decisions
        for i in range(5):
            engine.compute_blended_risk(
                f"agent_{i}", f"action_{i}", 0.5, "warn", 0.6, 0.8
            )

        decisions = engine.export_decisions(limit=3)

        assert len(decisions) == 3
        assert all(isinstance(d, dict) for d in decisions)

    def test_export_gray_zone_only(self):
        """Test exporting only gray zone decisions."""
        engine = MLBlendedRiskEngine()

        # Mix of zones
        engine.compute_blended_risk("a1", "act1", 0.2, "allow", 0.3, 0.8)
        engine.compute_blended_risk("a2", "act2", 0.5, "warn", 0.6, 0.8)
        engine.compute_blended_risk("a3", "act3", 0.8, "deny", 0.9, 0.8)

        gray_only = engine.export_decisions(risk_zone=RiskZone.GRAY_ZONE)

        assert len(gray_only) == 1
        assert gray_only[0]["risk_zone"] == RiskZone.GRAY_ZONE.value

    def test_storage_persistence(self):
        """Test decision persistence to storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MLBlendedRiskEngine(storage_path=tmpdir)

            engine.compute_blended_risk(
                "agent_001", "action_001", 0.5, "warn", 0.6, 0.8
            )

            log_file = Path(tmpdir) / "blended_decisions.jsonl"
            assert log_file.exists()

    def test_ml_blending_toggle(self):
        """Test toggling ML blending on/off."""
        engine = MLBlendedRiskEngine(enable_ml_blending=False)

        decision = engine.compute_blended_risk(
            "agent_001", "action_001", 0.5, "warn", 0.7, 0.9
        )

        # Even in gray zone, ML should not influence when disabled
        assert decision.ml_influenced is False
        assert decision.blended_risk_score == 0.5


class TestBlendingMetrics:
    """Test blending metrics calculations."""

    def test_zone_percentages(self):
        """Test zone percentage calculations."""
        metrics = BlendingMetrics()

        metrics.total_decisions = 100
        metrics.clear_allow_count = 30
        metrics.gray_zone_count = 40
        metrics.clear_deny_count = 30

        assert metrics.gray_zone_percentage == 40.0

    def test_ml_influence_rate(self):
        """Test ML influence rate calculation."""
        metrics = BlendingMetrics()

        metrics.gray_zone_count = 50
        metrics.ml_influenced_count = 45

        assert metrics.ml_influence_rate == 90.0

    def test_fp_delta(self):
        """Test false positive delta calculation."""
        metrics = BlendingMetrics()

        metrics.baseline_false_positives = 10
        metrics.blended_false_positives = 12

        assert metrics.fp_delta == 2
        assert metrics.fp_delta_percentage == 20.0

    def test_detection_improvement(self):
        """Test detection improvement calculation."""
        metrics = BlendingMetrics()

        metrics.baseline_true_positives = 80
        metrics.blended_true_positives = 85

        assert metrics.detection_improvement == 5
        assert metrics.detection_improvement_rate == 6.25

    def test_gate_check_pass(self):
        """Test gate check that passes."""
        metrics = BlendingMetrics()

        # Setup for passing gate
        metrics.gray_zone_count = 150
        metrics.baseline_false_positives = 20
        metrics.blended_false_positives = 21  # 5% increase
        metrics.baseline_true_positives = 80
        metrics.blended_true_positives = 85  # Improvement

        passes, reason = metrics.gate_check(max_fp_delta_pct=5.0)

        assert passes is True

    def test_gate_check_fail_fp_delta(self):
        """Test gate check that fails due to FP delta."""
        metrics = BlendingMetrics()

        metrics.gray_zone_count = 150
        metrics.baseline_false_positives = 20
        metrics.blended_false_positives = 23  # 15% increase
        metrics.baseline_true_positives = 80
        metrics.blended_true_positives = 85

        passes, reason = metrics.gate_check(max_fp_delta_pct=5.0)

        assert passes is False
        assert "FP delta" in reason

    def test_gate_check_fail_no_improvement(self):
        """Test gate check that fails due to no detection improvement."""
        metrics = BlendingMetrics()

        metrics.gray_zone_count = 150
        metrics.baseline_false_positives = 20
        metrics.blended_false_positives = 20
        metrics.baseline_true_positives = 80
        metrics.blended_true_positives = 80  # No improvement

        passes, reason = metrics.gate_check()

        assert passes is False
        assert "detection rate" in reason

    def test_gate_check_fail_insufficient_samples(self):
        """Test gate check that fails due to insufficient samples."""
        metrics = BlendingMetrics()

        metrics.gray_zone_count = 50  # Too few
        metrics.baseline_true_positives = 10
        metrics.blended_true_positives = 11

        passes, reason = metrics.gate_check()

        assert passes is False
        assert "Insufficient" in reason


class TestRiskZone:
    """Test risk zone enum."""

    def test_risk_zones(self):
        """Test risk zone values."""
        assert RiskZone.CLEAR_ALLOW.value == "clear_allow"
        assert RiskZone.GRAY_ZONE.value == "gray_zone"
        assert RiskZone.CLEAR_DENY.value == "clear_deny"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
