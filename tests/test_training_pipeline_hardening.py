#!/usr/bin/env python3
"""
Tests for training pipeline hardening.

These tests verify:
1. Zero-denominator handling in compute_metrics (train_any_model.py)
2. Dynamic feature selection in preprocess_for_heuristic
3. Empty data handling in preprocess_for_logistic
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_any_model import (
    compute_metrics,
    preprocess_for_heuristic,
    preprocess_for_logistic,
)


class TestComputeMetricsZeroDenominator:
    """Tests for zero-denominator handling in compute_metrics."""

    def test_empty_predictions_and_labels(self):
        """Test with empty predictions and labels."""
        metrics = compute_metrics([], [])

        # All metrics should be 0.0 when there's no data
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["accuracy"] == 0.0
        assert metrics["f1"] == 0.0

    def test_all_true_negatives(self):
        """Test when all predictions are true negatives (tp=0, fp=0)."""
        # All predictions and labels are 0 (negative class)
        preds = [0, 0, 0, 0, 0]
        labels = [0, 0, 0, 0, 0]

        metrics = compute_metrics(preds, labels)

        # precision = tp / (tp + fp) = 0 / 0 -> should be 0.0, not error
        assert metrics["precision"] == 0.0
        # recall = tp / (tp + fn) = 0 / 0 -> should be 0.0, not error
        assert metrics["recall"] == 0.0
        # accuracy = (tp + tn) / total = 5/5 = 1.0
        assert metrics["accuracy"] == 1.0
        # f1 = 2 * precision * recall / (precision + recall) = 0 / 0 -> 0.0
        assert metrics["f1"] == 0.0

    def test_all_false_negatives(self):
        """Test when all predictions are false negatives (tp=0, fn>0)."""
        # Predictions are all 0, but labels are all 1
        preds = [0, 0, 0, 0, 0]
        labels = [1, 1, 1, 1, 1]

        metrics = compute_metrics(preds, labels)

        # precision = tp / (tp + fp) = 0 / (0 + 0) = 0 (no positive predictions)
        assert metrics["precision"] == 0.0
        # recall = tp / (tp + fn) = 0 / (0 + 5) = 0
        assert metrics["recall"] == 0.0
        # accuracy = 0 / 5 = 0.0
        assert metrics["accuracy"] == 0.0
        # f1 = 0.0
        assert metrics["f1"] == 0.0

    def test_all_false_positives(self):
        """Test when all predictions are false positives (tp=0, fp>0)."""
        # Predictions are all 1, but labels are all 0
        preds = [1, 1, 1, 1, 1]
        labels = [0, 0, 0, 0, 0]

        metrics = compute_metrics(preds, labels)

        # precision = tp / (tp + fp) = 0 / (0 + 5) = 0.0
        assert metrics["precision"] == 0.0
        # recall = tp / (tp + fn) = 0 / (0 + 0) = 0.0 (fn=0 because no positive labels exist to miss)
        assert metrics["recall"] == 0.0
        # accuracy = 0 / 5 = 0.0
        assert metrics["accuracy"] == 0.0
        # f1 = 0.0
        assert metrics["f1"] == 0.0

    def test_all_true_positives(self):
        """Test when all predictions are true positives."""
        preds = [1, 1, 1, 1, 1]
        labels = [1, 1, 1, 1, 1]

        metrics = compute_metrics(preds, labels)

        # precision = 5 / 5 = 1.0
        assert metrics["precision"] == 1.0
        # recall = 5 / 5 = 1.0
        assert metrics["recall"] == 1.0
        # accuracy = 5 / 5 = 1.0
        assert metrics["accuracy"] == 1.0
        # f1 = 2 * 1 * 1 / 2 = 1.0
        assert metrics["f1"] == 1.0

    def test_normal_mixed_predictions(self):
        """Test with normal mixed predictions."""
        # tp=2, tn=2, fp=1, fn=1
        preds = [1, 1, 1, 0, 0, 0]
        labels = [1, 1, 0, 0, 0, 1]

        metrics = compute_metrics(preds, labels)

        # precision = 2 / 3 = 0.666...
        assert abs(metrics["precision"] - 2 / 3) < 0.001
        # recall = 2 / 3 = 0.666...
        assert abs(metrics["recall"] - 2 / 3) < 0.001
        # accuracy = 4 / 6 = 0.666...
        assert abs(metrics["accuracy"] - 4 / 6) < 0.001
        # f1 = 2 * (2/3) * (2/3) / (4/3) = 0.666...
        assert abs(metrics["f1"] - 2 / 3) < 0.001


class TestPreprocessForHeuristic:
    """Tests for dynamic feature selection in preprocess_for_heuristic."""

    def test_with_preferred_keys_present(self):
        """Test when preferred keys are present in the data."""
        data = [
            {
                "features": {
                    "violation_count": 0.5,
                    "severity_max": 0.8,
                    "recency_score": 0.3,
                    "frequency_score": 0.7,
                    "context_risk": 0.4,
                },
                "label": 1,
            }
        ]

        result = preprocess_for_heuristic(data)

        # Should use all preferred keys
        assert "violation_count" in result[0]["features"]
        assert "severity_max" in result[0]["features"]
        assert "recency_score" in result[0]["features"]
        assert "frequency_score" in result[0]["features"]
        assert "context_risk" in result[0]["features"]

        # Values should be preserved as floats
        assert result[0]["features"]["violation_count"] == 0.5
        assert result[0]["features"]["severity_max"] == 0.8

    def test_with_some_preferred_keys_present(self):
        """Test when only some preferred keys are present."""
        data = [
            {
                "features": {
                    "violation_count": 0.5,
                    "severity_max": 0.8,
                    # Other preferred keys are missing
                },
                "label": 1,
            }
        ]

        result = preprocess_for_heuristic(data)

        # Should still include all preferred keys (missing ones default to 0.0)
        assert "violation_count" in result[0]["features"]
        assert "severity_max" in result[0]["features"]
        assert "recency_score" in result[0]["features"]
        assert result[0]["features"]["recency_score"] == 0.0

    def test_with_no_preferred_keys_fallback(self):
        """Test fallback to dynamic feature selection when no preferred keys are present."""
        data = [
            {
                "features": {
                    "custom_feature_a": 1.5,
                    "custom_feature_b": 2.0,
                    "custom_feature_c": 3.5,
                    "text_field": "some text",  # Non-numeric, should be excluded
                },
                "label": 0,
            }
        ]

        result = preprocess_for_heuristic(data)

        # Should use dynamic feature selection - all numeric features
        assert "custom_feature_a" in result[0]["features"]
        assert result[0]["features"]["custom_feature_a"] == 1.5
        assert "custom_feature_b" in result[0]["features"]
        assert result[0]["features"]["custom_feature_b"] == 2.0
        assert "custom_feature_c" in result[0]["features"]
        assert result[0]["features"]["custom_feature_c"] == 3.5
        # Non-numeric fields should be excluded
        assert "text_field" not in result[0]["features"]

    def test_mixed_data_with_and_without_preferred_keys(self):
        """Test with mixed data where some samples have preferred keys."""
        data = [
            {
                "features": {
                    "violation_count": 0.5,
                    "severity_max": 0.8,
                },
                "label": 1,
            },
            {
                "features": {
                    "custom_feature_x": 1.0,
                    "custom_feature_y": 2.0,
                },
                "label": 0,
            },
        ]

        result = preprocess_for_heuristic(data)

        # First sample should use preferred keys
        assert "violation_count" in result[0]["features"]

        # Second sample should use dynamic feature selection
        assert "custom_feature_x" in result[1]["features"]
        assert "custom_feature_y" in result[1]["features"]

    def test_with_empty_features(self):
        """Test with empty features dictionary."""
        data = [{"features": {}, "label": 0}]

        result = preprocess_for_heuristic(data)

        # Should return empty features since no keys are present
        assert result[0]["features"] == {}

    def test_with_integer_values(self):
        """Test that integer values are converted to floats."""
        data = [
            {
                "features": {
                    "int_feature": 5,
                    "float_feature": 3.14,
                },
                "label": 0,
            }
        ]

        result = preprocess_for_heuristic(data)

        # Both should be floats now
        assert isinstance(result[0]["features"]["int_feature"], float)
        assert result[0]["features"]["int_feature"] == 5.0
        assert result[0]["features"]["float_feature"] == 3.14


class TestPreprocessForLogistic:
    """Tests for preprocess_for_logistic empty data handling."""

    def test_empty_data_returns_empty(self):
        """Test that empty data returns empty list."""
        result = preprocess_for_logistic([])
        assert result == []

    def test_non_empty_data_normalizes(self):
        """Test that non-empty data is normalized."""
        data = [
            {"features": {"f1": 0.0, "f2": 10.0}, "label": 0},
            {"features": {"f1": 5.0, "f2": 20.0}, "label": 1},
            {"features": {"f1": 10.0, "f2": 30.0}, "label": 1},
        ]

        result = preprocess_for_logistic(data)

        # Should have normalized values in [0, 1] range
        assert len(result) == 3

        # First sample should have min values (normalized to ~0)
        assert result[0]["features"]["f1"] < 0.1
        assert result[0]["features"]["f2"] < 0.1

        # Last sample should have max values (normalized to ~1)
        assert result[2]["features"]["f1"] > 0.9
        assert result[2]["features"]["f2"] > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
