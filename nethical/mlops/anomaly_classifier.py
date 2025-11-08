"""ML-based Anomaly Detection Classifier.

This module provides a trainable classifier for detecting anomalies
in agent behavior sequences. It can be used with the train_any_model.py
training pipeline.
"""

import json
import math
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import Counter


class AnomalyMLClassifier:
    """ML classifier for anomaly detection.

    This classifier learns patterns from sequence data and predicts
    whether new sequences are anomalous. It uses statistical features
    extracted from action sequences.
    """

    def __init__(self, n: int = 3, anomaly_threshold: float = 0.3):
        """Initialize anomaly classifier.

        Args:
            n: N-gram size for sequence analysis (default: 3)
            anomaly_threshold: Threshold for anomaly classification (default: 0.3)
        """
        self.n = n
        self.anomaly_threshold = anomaly_threshold

        # Learned patterns
        self.normal_ngrams: Dict[Tuple[str, ...], int] = {}
        self.anomalous_ngrams: Dict[Tuple[str, ...], int] = {}
        self.normal_action_freq: Dict[str, float] = {}
        self.anomalous_action_freq: Dict[str, float] = {}

        # Feature weights (learned during training)
        self.feature_weights = {
            "ngram_rarity": 0.4,
            "action_frequency_deviation": 0.3,
            "sequence_entropy": 0.2,
            "pattern_diversity": 0.1,
        }

        # Model metadata
        self.trained = False
        self.training_samples = 0
        self.timestamp = None

    def _extract_ngrams(self, sequence: List[str]) -> List[Tuple[str, ...]]:
        """Extract n-grams from a sequence.

        Args:
            sequence: List of actions

        Returns:
            List of n-grams
        """
        if len(sequence) < self.n:
            return []
        return [tuple(sequence[i : i + self.n]) for i in range(len(sequence) - self.n + 1)]

    def _calculate_entropy(self, sequence: List[str]) -> float:
        """Calculate Shannon entropy of action sequence.

        Args:
            sequence: List of actions

        Returns:
            Entropy value
        """
        if not sequence:
            return 0.0

        counts = Counter(sequence)
        total = len(sequence)
        entropy = 0.0

        for count in counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)

        return entropy

    def _extract_features(self, sequence: List[str]) -> Dict[str, float]:
        """Extract features from a sequence.

        Args:
            sequence: List of actions

        Returns:
            Dictionary of features
        """
        features = {}

        # 1. N-gram rarity: How rare are the n-grams in this sequence?
        ngrams = self._extract_ngrams(sequence)
        if ngrams:
            rarity_scores = []
            for ngram in ngrams:
                if self.trained and (self.normal_ngrams or self.anomalous_ngrams):
                    normal_count = self.normal_ngrams.get(ngram, 0)
                    anomalous_count = self.anomalous_ngrams.get(ngram, 0)
                    total = normal_count + anomalous_count
                    if total > 0:
                        # Higher score if more common in anomalous sequences
                        anomaly_ratio = anomalous_count / total
                        rarity_scores.append(anomaly_ratio)
                    else:
                        # Unseen n-gram - moderately suspicious since it's unknown
                        rarity_scores.append(0.7)
                else:
                    # Not trained yet, use neutral score
                    rarity_scores.append(0.5)
            features["ngram_rarity"] = (
                sum(rarity_scores) / len(rarity_scores) if rarity_scores else 0.5
            )
        else:
            features["ngram_rarity"] = 0.5

        # 2. Action frequency deviation: Do actions deviate from normal distribution?
        if sequence:
            if self.trained and (self.normal_action_freq or self.anomalous_action_freq):
                deviation_scores = []
                for action in sequence:
                    normal_freq = self.normal_action_freq.get(action, 0.0)
                    anomalous_freq = self.anomalous_action_freq.get(action, 0.0)
                    total_freq = normal_freq + anomalous_freq
                    if total_freq > 0:
                        # Higher if more common in anomalous
                        deviation_scores.append(anomalous_freq / total_freq)
                    else:
                        # Unknown action - moderately suspicious
                        deviation_scores.append(0.7)
                features["action_frequency_deviation"] = sum(deviation_scores) / len(
                    deviation_scores
                )
            else:
                features["action_frequency_deviation"] = 0.5
        else:
            features["action_frequency_deviation"] = 0.5

        # 3. Sequence entropy: Low entropy (repetitive) can be suspicious
        entropy = self._calculate_entropy(sequence)
        # Normalize entropy (max entropy for n unique items is log2(n))
        unique_count = len(set(sequence))
        if unique_count > 1:
            max_entropy = math.log2(unique_count)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            normalized_entropy = 0.0
        # Low entropy is more suspicious
        features["sequence_entropy"] = 1.0 - min(normalized_entropy, 1.0)

        # 4. Pattern diversity: Very repetitive patterns are suspicious
        unique_ratio = len(set(sequence)) / len(sequence) if sequence else 0.0
        # Low diversity is suspicious
        features["pattern_diversity"] = 1.0 - unique_ratio

        return features

    def train(
        self, train_data: List[Dict[str, Any]], epochs: int = 10, batch_size: int = 32
    ) -> None:
        """Train the anomaly classifier.

        Args:
            train_data: List of training samples with 'features' and 'label'
                       Each sample should have a 'sequence' in features
            epochs: Number of training epochs (ignored for this simple model)
            batch_size: Batch size (ignored for this simple model)
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")

        # Build n-gram and action frequency distributions
        normal_action_counts = Counter()
        anomalous_action_counts = Counter()
        normal_ngram_counts = Counter()
        anomalous_ngram_counts = Counter()

        for sample in train_data:
            features = sample.get("features", {})
            label = sample.get("label", 0)

            # Extract sequence from features
            # Support both 'sequence' and individual feature format
            if "sequence" in features:
                sequence = features["sequence"]
            else:
                # For compatibility with existing data format, create a synthetic sequence
                # based on feature values (this is a fallback)
                sequence = self._features_to_sequence(features)

            if not sequence:
                continue

            # Update frequency distributions
            if label == 0:  # Normal
                normal_action_counts.update(sequence)
                normal_ngram_counts.update(self._extract_ngrams(sequence))
            else:  # Anomalous
                anomalous_action_counts.update(sequence)
                anomalous_ngram_counts.update(self._extract_ngrams(sequence))

        # Convert to frequencies
        total_normal = sum(normal_action_counts.values()) or 1
        total_anomalous = sum(anomalous_action_counts.values()) or 1

        self.normal_action_freq = {
            action: count / total_normal for action, count in normal_action_counts.items()
        }
        self.anomalous_action_freq = {
            action: count / total_anomalous for action, count in anomalous_action_counts.items()
        }

        self.normal_ngrams = dict(normal_ngram_counts)
        self.anomalous_ngrams = dict(anomalous_ngram_counts)

        # Mark as trained so feature extraction uses the learned distributions
        self.trained = True

        # Learn feature weights based on discriminative power
        self._learn_weights(train_data)

        self.training_samples = len(train_data)
        self.timestamp = datetime.now().isoformat()

    def _features_to_sequence(self, features: Dict[str, Any]) -> List[str]:
        """Convert feature dict to a synthetic sequence (fallback method).

        Args:
            features: Feature dictionary

        Returns:
            List of action strings
        """
        # Create a synthetic sequence based on features
        sequence = []

        # Map feature values to action patterns
        if features.get("violation_count", 0) > 0.5:
            sequence.extend(["violation"] * int(features["violation_count"] * 3 + 1))
        if features.get("severity_max", 0) > 0.5:
            sequence.extend(["high_severity"] * int(features["severity_max"] * 2 + 1))
        if features.get("frequency_score", 0) > 0.5:
            sequence.extend(["frequent_action"] * int(features["frequency_score"] * 2 + 1))

        # Add some normal actions
        if features.get("context_risk", 0) < 0.5:
            sequence.extend(["normal_action"] * 2)

        return sequence if sequence else ["unknown"]

    def _learn_weights(self, train_data: List[Dict[str, Any]]) -> None:
        """Learn feature weights based on discriminative power.

        Args:
            train_data: Training data
        """
        # Calculate feature importance by measuring separation
        feature_sums = {k: {"normal": 0.0, "anomalous": 0.0} for k in self.feature_weights.keys()}
        normal_count = 0
        anomalous_count = 0

        for sample in train_data:
            features = sample.get("features", {})
            label = sample.get("label", 0)

            # Get sequence
            if "sequence" in features:
                sequence = features["sequence"]
            else:
                sequence = self._features_to_sequence(features)

            if not sequence:
                continue

            # Extract features
            extracted = self._extract_features(sequence)

            # Accumulate
            if label == 0:
                normal_count += 1
                for k, v in extracted.items():
                    if k in feature_sums:
                        feature_sums[k]["normal"] += v
            else:
                anomalous_count += 1
                for k, v in extracted.items():
                    if k in feature_sums:
                        feature_sums[k]["anomalous"] += v

        # Calculate discriminative power
        if normal_count > 0 and anomalous_count > 0:
            total_weight = 0.0
            new_weights = {}

            for feature_name in self.feature_weights.keys():
                normal_avg = feature_sums[feature_name]["normal"] / normal_count
                anomalous_avg = feature_sums[feature_name]["anomalous"] / anomalous_count

                # Discriminative power: difference between normal and anomalous
                # We want features where anomalous is higher than normal
                power = max(0.0, anomalous_avg - normal_avg) + 0.01
                new_weights[feature_name] = power
                total_weight += power

            # Normalize
            if total_weight > 0:
                for k in self.feature_weights.keys():
                    self.feature_weights[k] = new_weights[k] / total_weight

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make an anomaly prediction.

        Args:
            features: Feature dictionary (should contain 'sequence' or standard features)

        Returns:
            Dictionary with 'label' (0=normal, 1=anomalous), 'score', and 'confidence'
        """
        # Get sequence
        if "sequence" in features:
            sequence = features["sequence"]
        else:
            sequence = self._features_to_sequence(features)

        if not sequence:
            return {"label": 0, "score": 0.0, "confidence": 0.0}

        # Extract features
        extracted_features = self._extract_features(sequence)

        # Compute weighted anomaly score
        score = 0.0
        for feature_name, weight in self.feature_weights.items():
            feature_value = extracted_features.get(feature_name, 0.0)
            score += weight * feature_value

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        # Determine label
        label = 1 if score >= self.anomaly_threshold else 0

        # Calculate confidence based on distance from threshold
        if score >= self.anomaly_threshold:
            # Anomalous: confidence increases as we move away from threshold towards 1.0
            confidence = (
                (score - self.anomaly_threshold) / (1.0 - self.anomaly_threshold)
                if (1.0 - self.anomaly_threshold) > 0
                else 1.0
            )
        else:
            # Normal: confidence increases as we move away from threshold towards 0.0
            confidence = (
                (self.anomaly_threshold - score) / self.anomaly_threshold
                if self.anomaly_threshold > 0
                else 1.0
            )

        confidence = min(confidence, 1.0)

        return {"label": label, "score": score, "confidence": confidence}

    def save(self, filepath: str) -> None:
        """Save model to JSON file.

        Args:
            filepath: Path to save model
        """
        model_data = {
            "model_type": "anomaly_ml",
            "n": self.n,
            "anomaly_threshold": self.anomaly_threshold,
            "feature_weights": self.feature_weights,
            "normal_ngrams": {str(k): v for k, v in self.normal_ngrams.items()},
            "anomalous_ngrams": {str(k): v for k, v in self.anomalous_ngrams.items()},
            "normal_action_freq": self.normal_action_freq,
            "anomalous_action_freq": self.anomalous_action_freq,
            "trained": self.trained,
            "training_samples": self.training_samples,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "version": "1.0",
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "AnomalyMLClassifier":
        """Load model from JSON file.

        Args:
            filepath: Path to model file

        Returns:
            Loaded classifier instance
        """
        with open(filepath, "r") as f:
            model_data = json.load(f)

        classifier = cls(
            n=model_data.get("n", 3), anomaly_threshold=model_data.get("anomaly_threshold", 0.5)
        )

        classifier.feature_weights = model_data.get("feature_weights", classifier.feature_weights)

        # Convert string keys back to tuples for n-grams
        normal_ngrams_str = model_data.get("normal_ngrams", {})
        classifier.normal_ngrams = {eval(k): v for k, v in normal_ngrams_str.items()}

        anomalous_ngrams_str = model_data.get("anomalous_ngrams", {})
        classifier.anomalous_ngrams = {eval(k): v for k, v in anomalous_ngrams_str.items()}

        classifier.normal_action_freq = model_data.get("normal_action_freq", {})
        classifier.anomalous_action_freq = model_data.get("anomalous_action_freq", {})
        classifier.trained = model_data.get("trained", False)
        classifier.training_samples = model_data.get("training_samples", 0)
        classifier.timestamp = model_data.get("timestamp")

        return classifier
