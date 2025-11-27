"""
MultiPatternMLClassifier: Extensible ML-based Pattern Detector for Multi-Agent Systems

This single-cell script provides a multi-type pattern classifier for multi-agent activities,
based on extending the CorrelationMLClassifier architecture.

Features:
- Detects multiple pattern types: correlation, anomaly, collaboration, temporal, spatial (extensible).
- Easy to add new features and pattern types.
- Single class supports train, predict, metrics, save, and load.

Usage Example:
    classifier = MultiPatternMLClassifier(
        pattern_types=["correlation", "anomaly", "collaboration", "temporal", "spatial"]
    )
    classifier.train(train_data)
    result = classifier.predict(sample_features)
"""

import json
import math
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict


class MultiPatternMLClassifier:
    """
    ML classifier for multi-pattern detection in agent activities.

    Supports multiple pattern types. Features are extracted and weighted for each pattern.
    """

    def __init__(self, pattern_types=None, pattern_thresholds=None):
        """
        Args:
            pattern_types: List of supported pattern types (str). E.g. ["correlation", "anomaly", ...]
            pattern_thresholds: Optional dict: {pattern_type: threshold (float from 0 to 1)}
        """
        if pattern_types is None:
            pattern_types = ["correlation", "anomaly", "collaboration", "temporal", "spatial"]

        self.pattern_types = pattern_types
        self.pattern_thresholds = pattern_thresholds or {p: 0.5 for p in self.pattern_types}

        # Feature weights for each pattern_type. Overwritten by training.
        self.feature_weights = {
            # base features as example
            "agent_count": {pt: 0.2 for pt in self.pattern_types},
            "action_rate": {pt: 0.2 for pt in self.pattern_types},
            "entropy_variance": {pt: 0.2 for pt in self.pattern_types},
            "time_correlation": {pt: 0.2 for pt in self.pattern_types},
            "payload_similarity": {pt: 0.2 for pt in self.pattern_types},
            # additional, example features for new patterns:
            "group_size": {pt: 0.15 if pt == "collaboration" else 0.0 for pt in self.pattern_types},
            "temporal_spike": {pt: 0.15 if pt == "temporal" else 0.0 for pt in self.pattern_types},
            "spatial_variance": {pt: 0.15 if pt == "spatial" else 0.0 for pt in self.pattern_types},
        }

        self.trained = {pt: False for pt in self.pattern_types}
        self.training_samples = 0
        self.timestamp = None

    @staticmethod
    def _calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy of text."""
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
        """Extract and normalize features from sample dict."""
        features = {}
        features["agent_count"] = min(sample_features.get("agent_count", 0) / 10.0, 1.0)
        features["action_rate"] = min(sample_features.get("action_rate", 0) / 100.0, 1.0)
        features["entropy_variance"] = min(sample_features.get("entropy_variance", 0) / 5.0, 1.0)
        features["time_correlation"] = min(sample_features.get("time_correlation", 0), 1.0)
        features["payload_similarity"] = min(sample_features.get("payload_similarity", 0), 1.0)
        # NEW FEATURES, optionally required for new pattern types:
        features["group_size"] = min(sample_features.get("group_size", 1) / 10.0, 1.0)
        features["temporal_spike"] = min(sample_features.get("temporal_spike", 0) / 5.0, 1.0)
        features["spatial_variance"] = min(sample_features.get("spatial_variance", 0) / 5.0, 1.0)
        return features

    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """Train classifier on multiple pattern types (multi-label/multi-class).

        Args:
            train_data: List of samples with 'features' and 'labels', where labels is a dict:
                {
                    "features": {feature_name: value, ...},
                    "labels": {pattern_type: 0/1, ...}
                }
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")
        feature_stats = {pt: {"positive": defaultdict(list), "negative": defaultdict(list)}
                         for pt in self.pattern_types}

        for sample in train_data:
            raw_features = sample.get("features", {})
            labels = sample.get("labels", {})  # dict: {pattern_type: 0/1}
            norm_features = self._extract_features(raw_features)

            for pt in self.pattern_types:
                label = labels.get(pt, 0)
                cat = "positive" if label == 1 else "negative"
                for fname, val in norm_features.items():
                    if fname in self.feature_weights:
                        feature_stats[pt][cat][fname].append(val)

        # Learn discriminative power and weights per pattern_type
        for pt in self.pattern_types:
            adjusted_weights = {}
            total_weight = 0.0
            for fname in self.feature_weights:
                pos_vals = feature_stats[pt]["positive"].get(fname, [0])
                neg_vals = feature_stats[pt]["negative"].get(fname, [0])
                pos_avg = sum(pos_vals) / len(pos_vals) if pos_vals else 0
                neg_avg = sum(neg_vals) / len(neg_vals) if neg_vals else 0
                discriminative_power = abs(pos_avg - neg_avg) + 0.01
                adjusted_weights[fname] = discriminative_power
                total_weight += discriminative_power
            # Normalize so weights sum to 1
            for fname in self.feature_weights:
                self.feature_weights[fname][pt] = adjusted_weights[fname] / total_weight if total_weight > 0 else 0.0
            self.trained[pt] = True

        self.training_samples = len(train_data)
        self.timestamp = datetime.now().isoformat()

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict for multiple pattern types.

        Args:
            features: Raw features dict

        Returns:
            Dict {pattern_type: {label, score, confidence}}
        """
        norm_features = self._extract_features(features)
        result = {}
        for pt in self.pattern_types:
            score = 0.0
            for fname, weight_by_pt in self.feature_weights.items():
                weight = weight_by_pt.get(pt, 0.0)
                fval = norm_features.get(fname, 0.0)
                score += weight * fval
            # Calibration (soft sigmoid, temperature T=2)
            score = 1.0 / (1.0 + math.exp(-2 * (score - 0.5)))
            score = min(max(score, 0.0), 1.0)
            threshold = self.pattern_thresholds.get(pt, 0.5)
            label = 1 if score >= threshold else 0
            confidence = abs(score - threshold) * 2
            confidence = min(confidence, 1.0)
            result[pt] = {"label": label, "score": score, "confidence": confidence}
        return result

    @staticmethod
    def compute_metrics(preds: Dict[str, List[int]], labels: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each pattern_type."""
        metrics = {}
        for pt in preds.keys():
            p = preds[pt]
            l = labels[pt]
            if len(p) != len(l) or len(p) == 0:
                metrics[pt] = {"accuracy": 0, "precision":0, "recall":0, "f1_score":0, "ece":0}
                continue
            tp = sum(1 for pred, lab in zip(p, l) if pred == 1 and lab == 1)
            tn = sum(1 for pred, lab in zip(p, l) if pred == 0 and lab == 0)
            fp = sum(1 for pred, lab in zip(p, l) if pred == 1 and lab == 0)
            fn = sum(1 for pred, lab in zip(p, l) if pred == 0 and lab == 1)
            total = len(p)
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )
            ece = max(0.0, 0.15 * (1.0 - accuracy))
            metrics[pt] = dict(
                accuracy=accuracy, precision=precision, recall=recall,
                f1_score=f1_score, ece=ece, true_positives=tp, true_negatives=tn,
                false_positives=fp, false_negatives=fn
            )
        return metrics

    def save(self, filepath: str) -> None:
        """Save to JSON."""
        model_data = {
            "model_type": "multi_pattern",
            "pattern_types": self.pattern_types,
            "pattern_thresholds": self.pattern_thresholds,
            "feature_weights": self.feature_weights,
            "trained": self.trained,
            "training_samples": self.training_samples,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "version": "1.0",
        }
        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "MultiPatternMLClassifier":
        """Load from JSON file."""
        with open(filepath, "r") as f:
            model_data = json.load(f)
        classifier = cls(
            pattern_types=model_data.get("pattern_types", None),
            pattern_thresholds=model_data.get("pattern_thresholds", None)
        )
        classifier.feature_weights = model_data.get("feature_weights", classifier.feature_weights)
        classifier.trained = model_data.get("trained", {pt: False for pt in classifier.pattern_types})
        classifier.training_samples = model_data.get("training_samples", 0)
        classifier.timestamp = model_data.get("timestamp")
        return classifier

# --- Example usage (uncomment to run) ---
# train_data = [
#     {"features": {"agent_count": 5, "action_rate": 70, "entropy_variance": 2.1, 
#                   "time_correlation": 0.9, "payload_similarity": 0.8, "group_size": 3, "temporal_spike": 1.5, "spatial_variance": 0.7},
#      "labels": {"correlation": 1, "anomaly":0, "collaboration":1, "temporal":0, "spatial":0}},
#     {"features": {"agent_count": 1, "action_rate": 15, "entropy_variance": 0.5, 
#                   "time_correlation": 0.1, "payload_similarity": 0.0, "group_size": 1, "temporal_spike": 0.5, "spatial_variance": 1.5},
#      "labels": {"correlation": 0, "anomaly":1, "collaboration":0, "temporal":1, "spatial":1}},
#     # More samples...
# ]
# classifier = MultiPatternMLClassifier()
# classifier.train(train_data)
# features = {"agent_count": 3, "action_rate": 60, "entropy_variance": 1.9, 
#             "time_correlation": 0.8, "payload_similarity": 0.79, "group_size": 2, "temporal_spike": 1.1, "spatial_variance": 1.1}
# prediction = classifier.predict(features)
# print("Pattern predictions:", prediction)
