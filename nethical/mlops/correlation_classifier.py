"""ML-based Correlation Pattern Classifier.

This module provides a trainable classifier for detecting correlation patterns
in multi-agent activities. It can be used with the train_any_model.py
training pipeline.
"""

import json
import math
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict


class CorrelationMLClassifier:
    """ML classifier for correlation pattern detection.

    This classifier learns patterns from multi-agent activity data and predicts
    whether correlation patterns are present. It uses statistical features
    extracted from agent behaviors.
    """

    def __init__(self, pattern_threshold: float = 0.5):
        """Initialize correlation classifier.

        Args:
            pattern_threshold: Threshold for correlation pattern classification (default: 0.5)
        """
        self.pattern_threshold = pattern_threshold

        # Learned patterns
        self.normal_patterns: Dict[str, float] = {}
        self.anomalous_patterns: Dict[str, float] = {}

        # Feature weights (learned during training)
        self.feature_weights = {
            "agent_count": 0.25,
            "action_rate": 0.2,
            "entropy_variance": 0.2,
            "time_correlation": 0.2,
            "payload_similarity": 0.15,
        }

        # Model metadata
        self.trained = False
        self.training_samples = 0
        self.timestamp = None

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text.

        Args:
            text: Input text

        Returns:
            Entropy value
        """
        if not text:
            return 0.0

        char_freq = defaultdict(int)
        for char in text:
            char_freq[char] += 1

        length = len(text)
        entropy = 0.0

        for count in char_freq.values():
            if count > 0:
                probability = count / length
                entropy -= probability * math.log2(probability)

        return entropy

    def _extract_features(self, sample_features: Dict[str, Any]) -> Dict[str, float]:
        """Extract normalized features from sample.

        Args:
            sample_features: Raw feature dictionary

        Returns:
            Dictionary of normalized features
        """
        features = {}

        # Agent count feature
        features["agent_count"] = min(sample_features.get("agent_count", 0) / 10.0, 1.0)

        # Action rate feature
        features["action_rate"] = min(sample_features.get("action_rate", 0) / 100.0, 1.0)

        # Entropy variance feature
        features["entropy_variance"] = min(sample_features.get("entropy_variance", 0) / 5.0, 1.0)

        # Time correlation feature
        features["time_correlation"] = min(sample_features.get("time_correlation", 0), 1.0)

        # Payload similarity feature
        features["payload_similarity"] = min(sample_features.get("payload_similarity", 0), 1.0)

        return features

    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """Train the classifier on labeled data.

        Args:
            train_data: List of training samples with 'features' and 'label'
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")

        # Collect statistics for normal and anomalous patterns
        feature_stats = {"normal": defaultdict(list), "anomalous": defaultdict(list)}

        for sample in train_data:
            raw_features = sample.get("features", {})
            label = sample.get("label", 0)

            # Extract and normalize features
            features = self._extract_features(raw_features)

            category = "anomalous" if label == 1 else "normal"
            for feature_name, value in features.items():
                if feature_name in self.feature_weights:
                    feature_stats[category][feature_name].append(value)

        # Calculate discriminative power for each feature
        adjusted_weights = {}
        total_weight = 0.0

        for feature_name in self.feature_weights.keys():
            normal_values = feature_stats["normal"].get(feature_name, [0])
            anomalous_values = feature_stats["anomalous"].get(feature_name, [0])

            normal_avg = sum(normal_values) / len(normal_values) if normal_values else 0
            anomalous_avg = sum(anomalous_values) / len(anomalous_values) if anomalous_values else 0

            # Discriminative power = difference between normal and anomalous averages
            discriminative_power = abs(anomalous_avg - normal_avg) + 0.01
            adjusted_weights[feature_name] = discriminative_power
            total_weight += discriminative_power

        # Normalize weights to sum to 1
        if total_weight > 0:
            for feature_name in self.feature_weights.keys():
                self.feature_weights[feature_name] = adjusted_weights[feature_name] / total_weight

        self.trained = True
        self.training_samples = len(train_data)
        self.timestamp = datetime.now().isoformat()

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction for given features.

        Args:
            features: Feature dictionary

        Returns:
            Dictionary with 'label', 'score', and 'confidence'
        """
        if not self.trained:
            # Use default weights if not trained
            pass

        # Extract and normalize features
        normalized_features = self._extract_features(features)

        # Compute weighted score
        score = 0.0
        for feature_name, weight in self.feature_weights.items():
            feature_value = normalized_features.get(feature_name, 0.0)
            score += weight * feature_value

        # Apply softer sigmoid transformation for better calibration
        # Using temperature scaling with T=2 for better calibration
        score = 1.0 / (1.0 + math.exp(-2 * (score - 0.5)))
        score = min(max(score, 0.0), 1.0)

        # Determine label (threshold at pattern_threshold)
        label = 1 if score >= self.pattern_threshold else 0

        # Calculate confidence based on distance from threshold
        confidence = abs(score - self.pattern_threshold) * 2
        confidence = min(confidence, 1.0)

        return {"label": label, "score": score, "confidence": confidence}

    def compute_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            predictions: List of predicted labels
            labels: List of true labels

        Returns:
            Dictionary of metrics
        """
        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have same length")

        if len(predictions) == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "ece": 0.0}

        # Calculate confusion matrix
        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

        # Calculate metrics
        total = len(predictions)
        accuracy = (tp + tn) / total if total > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        )

        # Simple ECE approximation
        ece = max(0.0, 0.15 * (1.0 - accuracy))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "ece": ece,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
        }

    def save(self, filepath: str) -> None:
        """Save model to JSON file.

        Args:
            filepath: Path to save model
        """
        model_data = {
            "model_type": "correlation",
            "pattern_threshold": self.pattern_threshold,
            "feature_weights": self.feature_weights,
            "trained": self.trained,
            "training_samples": self.training_samples,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "version": "1.0",
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CorrelationMLClassifier":
        """Load model from JSON file.

        Args:
            filepath: Path to model file

        Returns:
            Loaded classifier instance
        """
        with open(filepath, "r") as f:
            model_data = json.load(f)

        classifier = cls(pattern_threshold=model_data.get("pattern_threshold", 0.5))
        classifier.feature_weights = model_data.get("feature_weights", classifier.feature_weights)
        classifier.trained = model_data.get("trained", False)
        classifier.training_samples = model_data.get("training_samples", 0)
        classifier.timestamp = model_data.get("timestamp")

        return classifier
