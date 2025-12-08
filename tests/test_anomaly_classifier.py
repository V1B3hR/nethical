"""Tests for AnomalyMLClassifier."""

import pytest
from nethical.mlops.anomaly_classifier import AnomalyMLClassifier


class TestAnomalyMLClassifier:
    """Test anomaly ML classifier."""

    def test_initialization(self):
        """Test classifier initialization."""
        clf = AnomalyMLClassifier(n=3, anomaly_threshold=0.3)

        assert clf.n == 3
        assert clf.anomaly_threshold == 0.3
        assert clf.trained is False
        assert clf.training_samples == 0

    def test_train_with_sequences(self):
        """Test training with sequence data."""
        # Create training data
        train_data = []

        # Normal patterns
        for _ in range(50):
            train_data.append(
                {"features": {"sequence": ["read", "process", "write"]}, "label": 0}
            )

        # Anomalous patterns
        for _ in range(50):
            train_data.append(
                {
                    "features": {"sequence": ["delete", "exfiltrate", "cover_tracks"]},
                    "label": 1,
                }
            )

        clf = AnomalyMLClassifier(n=3, anomaly_threshold=0.3)
        clf.train(train_data)

        assert clf.trained is True
        assert clf.training_samples == 100
        assert len(clf.normal_ngrams) > 0
        assert len(clf.anomalous_ngrams) > 0

    def test_predict_normal_sequence(self):
        """Test prediction on normal sequence."""
        # Create training data
        train_data = []
        for _ in range(50):
            train_data.append(
                {"features": {"sequence": ["read", "process", "write"]}, "label": 0}
            )
        for _ in range(50):
            train_data.append(
                {
                    "features": {"sequence": ["delete", "exfiltrate", "cover_tracks"]},
                    "label": 1,
                }
            )

        clf = AnomalyMLClassifier(n=3, anomaly_threshold=0.3)
        clf.train(train_data)

        # Test normal sequence
        result = clf.predict({"sequence": ["read", "process", "write"]})

        assert result["label"] == 0
        assert 0.0 <= result["score"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_anomalous_sequence(self):
        """Test prediction on anomalous sequence."""
        # Create training data
        train_data = []
        for _ in range(50):
            train_data.append(
                {"features": {"sequence": ["read", "process", "write"]}, "label": 0}
            )
        for _ in range(50):
            train_data.append(
                {
                    "features": {"sequence": ["delete", "exfiltrate", "cover_tracks"]},
                    "label": 1,
                }
            )

        clf = AnomalyMLClassifier(n=3, anomaly_threshold=0.3)
        clf.train(train_data)

        # Test anomalous sequence
        result = clf.predict({"sequence": ["delete", "exfiltrate", "cover_tracks"]})

        assert result["label"] == 1
        assert 0.0 <= result["score"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

    def test_save_and_load(self):
        """Test model save and load."""
        import tempfile
        import os

        # Create and train model
        train_data = []
        for _ in range(20):
            train_data.append(
                {"features": {"sequence": ["read", "process", "write"]}, "label": 0}
            )
        for _ in range(20):
            train_data.append(
                {
                    "features": {"sequence": ["delete", "exfiltrate", "cover_tracks"]},
                    "label": 1,
                }
            )

        clf = AnomalyMLClassifier(n=3, anomaly_threshold=0.3)
        clf.train(train_data)

        # Save model
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            clf.save(filepath)

            # Load model
            loaded_clf = AnomalyMLClassifier.load(filepath)

            assert loaded_clf.n == clf.n
            assert loaded_clf.anomaly_threshold == clf.anomaly_threshold
            assert loaded_clf.trained == clf.trained
            assert loaded_clf.training_samples == clf.training_samples

            # Test that loaded model makes same predictions
            test_seq = {"sequence": ["read", "process", "write"]}
            result1 = clf.predict(test_seq)
            result2 = loaded_clf.predict(test_seq)

            assert result1["label"] == result2["label"]
            assert abs(result1["score"] - result2["score"]) < 0.01
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_extract_ngrams(self):
        """Test n-gram extraction."""
        clf = AnomalyMLClassifier(n=3)

        sequence = ["a", "b", "c", "d"]
        ngrams = clf._extract_ngrams(sequence)

        assert len(ngrams) == 2
        assert ngrams[0] == ("a", "b", "c")
        assert ngrams[1] == ("b", "c", "d")

    def test_calculate_entropy(self):
        """Test entropy calculation."""
        clf = AnomalyMLClassifier()

        # Uniform distribution should have high entropy
        uniform_seq = ["a", "b", "c", "d"]
        entropy1 = clf._calculate_entropy(uniform_seq)

        # Repetitive sequence should have low entropy
        repetitive_seq = ["a", "a", "a", "a"]
        entropy2 = clf._calculate_entropy(repetitive_seq)

        assert entropy1 > entropy2
        assert entropy2 == 0.0  # All same elements
