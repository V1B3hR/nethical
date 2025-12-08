"""Tests for Phase 5: ML Shadow Mode."""

import pytest
import tempfile
import shutil
from pathlib import Path

from nethical.core import (
    MLShadowClassifier,
    ShadowPrediction,
    ShadowMetrics,
    MLModelType,
)


class TestMLShadowClassifier:
    """Test ML shadow mode classifier."""

    def test_initialization(self):
        """Test shadow classifier initialization."""
        classifier = MLShadowClassifier(
            model_type=MLModelType.HEURISTIC, score_agreement_threshold=0.1
        )

        assert classifier.model_type == MLModelType.HEURISTIC
        assert classifier.score_agreement_threshold == 0.1
        assert len(classifier.predictions) == 0
        assert classifier.metrics.total_predictions == 0

    def test_predict_basic(self):
        """Test basic prediction."""
        classifier = MLShadowClassifier()

        features = {
            "violation_count": 0.5,
            "severity_max": 0.6,
            "recency_score": 0.4,
            "frequency_score": 0.3,
            "context_risk": 0.2,
        }

        prediction = classifier.predict(
            agent_id="agent_001",
            action_id="action_001",
            features=features,
            rule_risk_score=0.5,
            rule_classification="warn",
        )

        assert isinstance(prediction, ShadowPrediction)
        assert prediction.agent_id == "agent_001"
        assert prediction.action_id == "action_001"
        assert 0 <= prediction.ml_risk_score <= 1
        assert 0 <= prediction.ml_confidence <= 1
        assert prediction.ml_classification in ["allow", "warn", "deny"]

    def test_metrics_update(self):
        """Test metrics are updated correctly."""
        classifier = MLShadowClassifier()

        # Make predictions
        for i in range(10):
            features = {"violation_count": 0.3, "severity_max": 0.2}
            classifier.predict(
                agent_id=f"agent_{i}",
                action_id=f"action_{i}",
                features=features,
                rule_risk_score=0.3,
                rule_classification="allow",
            )

        assert classifier.metrics.total_predictions == 10
        assert len(classifier.predictions) == 10

    def test_agreement_tracking(self):
        """Test agreement tracking between ML and rules."""
        classifier = MLShadowClassifier(score_agreement_threshold=0.05)

        # Perfect agreement
        features = {"violation_count": 0.5}
        prediction = classifier.predict(
            agent_id="agent_001",
            action_id="action_001",
            features=features,
            rule_risk_score=0.5,
            rule_classification="warn",
        )

        # Check if scores agree (within threshold)
        assert isinstance(prediction.scores_agree, bool)
        assert isinstance(prediction.classifications_agree, bool)

    def test_metrics_report(self):
        """Test metrics report generation."""
        classifier = MLShadowClassifier()

        # Generate some predictions
        for i in range(20):
            features = {"violation_count": 0.2 if i < 10 else 0.8}
            rule_score = 0.2 if i < 10 else 0.8
            rule_class = "allow" if i < 10 else "deny"

            classifier.predict(
                agent_id=f"agent_{i}",
                action_id=f"action_{i}",
                features=features,
                rule_risk_score=rule_score,
                rule_classification=rule_class,
            )

        report = classifier.get_metrics_report()

        assert "precision" in report
        assert "recall" in report
        assert "f1_score" in report
        assert "accuracy" in report
        assert "expected_calibration_error" in report
        assert report["total_predictions"] == 20

    def test_export_predictions(self):
        """Test prediction export."""
        classifier = MLShadowClassifier()

        # Add predictions
        for i in range(5):
            features = {"violation_count": 0.3}
            classifier.predict(
                agent_id=f"agent_{i}",
                action_id=f"action_{i}",
                features=features,
                rule_risk_score=0.3,
                rule_classification="allow",
            )

        predictions = classifier.export_predictions(limit=3)

        assert len(predictions) == 3
        assert all(isinstance(p, dict) for p in predictions)
        assert all("agent_id" in p for p in predictions)

    def test_storage_persistence(self):
        """Test prediction persistence to storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            classifier = MLShadowClassifier(storage_path=tmpdir)

            features = {"violation_count": 0.5}
            classifier.predict(
                agent_id="agent_001",
                action_id="action_001",
                features=features,
                rule_risk_score=0.5,
                rule_classification="warn",
            )

            # Check if log file was created
            log_file = Path(tmpdir) / "shadow_predictions.jsonl"
            assert log_file.exists()

    def test_reset_metrics(self):
        """Test metrics reset."""
        classifier = MLShadowClassifier()

        # Add some predictions
        features = {"violation_count": 0.5}
        classifier.predict(
            agent_id="agent_001",
            action_id="action_001",
            features=features,
            rule_risk_score=0.5,
            rule_classification="warn",
        )

        assert classifier.metrics.total_predictions > 0

        # Reset
        classifier.reset_metrics()

        assert classifier.metrics.total_predictions == 0
        assert len(classifier.predictions) == 0


class TestShadowMetrics:
    """Test shadow metrics calculations."""

    def test_precision_calculation(self):
        """Test precision calculation."""
        metrics = ShadowMetrics()

        metrics.true_positives = 8
        metrics.false_positives = 2

        assert metrics.precision == 0.8

    def test_recall_calculation(self):
        """Test recall calculation."""
        metrics = ShadowMetrics()

        metrics.true_positives = 8
        metrics.false_negatives = 2

        assert metrics.recall == 0.8

    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        metrics = ShadowMetrics()

        metrics.true_positives = 8
        metrics.false_positives = 2
        metrics.false_negatives = 2

        precision = 8 / 10  # 0.8
        recall = 8 / 10  # 0.8
        expected_f1 = 2 * (precision * recall) / (precision + recall)

        assert metrics.f1_score == expected_f1

    def test_calibration_error(self):
        """Test expected calibration error calculation."""
        metrics = ShadowMetrics()

        # Add some calibration data
        metrics.calibration_bins["0.8-1.0"]["total"] = 10
        metrics.calibration_bins["0.8-1.0"]["correct"] = 9

        ece = metrics.get_calibration_error()

        assert isinstance(ece, float)
        assert 0 <= ece <= 1


class TestMLModelType:
    """Test ML model type enum."""

    def test_model_types(self):
        """Test model type values."""
        assert MLModelType.HEURISTIC.value == "heuristic"
        assert MLModelType.LOGISTIC.value == "logistic"
        assert MLModelType.SIMPLE_TRANSFORMER.value == "simple_transformer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
