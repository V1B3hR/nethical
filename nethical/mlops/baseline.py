"""
Baseline ML Classifier Module

This module provides baseline ML classifiers for the Nethical framework:
- BaselineMLClassifier: Lightweight heuristic-based classifier (no external dependencies)
- AdvancedMLClassifier: Deep learning classifier using PyTorch transformers

The BaselineMLClassifier uses weighted feature combinations for classification,
supporting heuristic, logistic, and simple transformer-like preprocessing modes.
"""

import json
import math
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class BaselineMLClassifier:
    """
    Lightweight baseline classifier for ethical violation detection.

    This classifier uses weighted feature combinations without requiring
    heavy ML dependencies like PyTorch or TensorFlow. It's suitable for:
    - Quick prototyping and testing
    - Resource-constrained environments
    - Production deployment with minimal overhead

    Supports three preprocessing modes via train_any_model.py:
    - heuristic: Raw numeric features
    - logistic: Min-max normalized features (0-1 range)
    - simple_transformer: Features with optional text tokenization

    Example usage:
        >>> clf = BaselineMLClassifier()
        >>> train_data = [
        ...     {'features': {'violation_count': 0.8, 'severity_max': 0.9}, 'label': 1},
        ...     {'features': {'violation_count': 0.1, 'severity_max': 0.2}, 'label': 0},
        ... ]
        >>> clf.train(train_data)
        >>> result = clf.predict({'violation_count': 0.7, 'severity_max': 0.6})
        >>> print(result['label'], result['confidence'])

    See Also:
        - training/train_any_model.py: Full training pipeline with this classifier
        - docs/TRAINING_GUIDE.md: End-to-end training documentation
        - examples/training/: Example training scripts
    """

    def __init__(self, threshold: float = 0.5, learning_rate: float = 0.1):
        """
        Initialize the baseline classifier.

        Args:
            threshold: Classification threshold (0-1). Scores >= threshold are class 1.
            learning_rate: Learning rate for weight updates during training.
        """
        self.threshold = threshold
        self.learning_rate = learning_rate

        # Default feature weights (will be learned during training)
        self.feature_weights: Dict[str, float] = {}
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}

        # Training state
        self.trained = False
        self.training_samples = 0
        self.timestamp: Optional[str] = None
        self.version = "1.0"

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text for feature extraction."""
        if not text:
            return 0.0
        char_freq: Dict[str, int] = defaultdict(int)
        for char in text:
            char_freq[char] += 1
        length = len(text)
        entropy = 0.0
        for count in char_freq.values():
            if count > 0:
                probability = count / length
                entropy -= probability * math.log2(probability)
        return entropy

    def _extract_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Extract and normalize features from input dict."""
        result: Dict[str, float] = {}
        for key, value in features.items():
            if isinstance(value, (int, float)):
                result[key] = float(value)
            elif isinstance(value, str):
                # For text features, use length and entropy
                result[f"{key}_length"] = min(len(value) / 100.0, 1.0)
                result[f"{key}_entropy"] = self._calculate_entropy(value) / 8.0
            elif isinstance(value, list):
                # For list features (like token_ids), use length and mean
                result[f"{key}_len"] = min(len(value) / 64.0, 1.0)
                if value and all(isinstance(v, (int, float)) for v in value):
                    result[f"{key}_mean"] = sum(value) / len(value) / 100.0
        return result

    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train the classifier on labeled data.

        This method learns feature weights based on discriminative power
        (difference in feature values between positive and negative classes).

        Args:
            train_data: List of training samples, each with 'features' (dict) and 'label' (0/1).
                       Example: [{'features': {'f1': 0.5, 'f2': 0.3}, 'label': 1}, ...]

        Raises:
            ValueError: If train_data is empty.
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")

        # Collect feature statistics
        positive_features: Dict[str, List[float]] = defaultdict(list)
        negative_features: Dict[str, List[float]] = defaultdict(list)

        all_features: Dict[str, List[float]] = defaultdict(list)

        for sample in train_data:
            raw_features = sample.get("features", {})
            label = int(sample.get("label", 0))

            extracted = self._extract_features(raw_features)

            for key, value in extracted.items():
                all_features[key].append(value)
                if label == 1:
                    positive_features[key].append(value)
                else:
                    negative_features[key].append(value)

        # Calculate feature statistics for normalization
        for key, values in all_features.items():
            self.feature_means[key] = sum(values) / len(values) if values else 0.0
            variance = (
                sum((v - self.feature_means[key]) ** 2 for v in values) / len(values)
                if values
                else 1.0
            )
            self.feature_stds[key] = math.sqrt(variance) if variance > 0 else 1.0

        # Learn feature weights based on discriminative power
        total_weight = 0.0
        for key in all_features.keys():
            pos_vals = positive_features.get(key, [0])
            neg_vals = negative_features.get(key, [0])

            pos_avg = sum(pos_vals) / len(pos_vals) if pos_vals else 0.0
            neg_avg = sum(neg_vals) / len(neg_vals) if neg_vals else 0.0

            # Discriminative power is the difference between class means
            discriminative_power = abs(pos_avg - neg_avg) + 0.01

            # Weight direction: positive if higher values correlate with positive class
            direction = 1.0 if pos_avg > neg_avg else -1.0

            self.feature_weights[key] = discriminative_power * direction
            total_weight += abs(discriminative_power)

        # Normalize weights to sum to 1
        if total_weight > 0:
            for key in self.feature_weights:
                self.feature_weights[key] /= total_weight

        self.trained = True
        self.training_samples = len(train_data)
        self.timestamp = datetime.now().isoformat()

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict class label for input features.

        Args:
            features: Feature dictionary (same format as training data features).

        Returns:
            Dictionary with:
            - 'label': Predicted class (0 or 1)
            - 'score': Raw prediction score (0-1)
            - 'confidence': Prediction confidence (0-1)

        Example:
            >>> result = clf.predict({'violation_count': 0.7, 'severity_max': 0.8})
            >>> print(result)
            {'label': 1, 'score': 0.75, 'confidence': 0.50}
        """
        extracted = self._extract_features(features)

        # Calculate weighted sum
        score = 0.5  # Base score
        for key, weight in self.feature_weights.items():
            if key in extracted:
                # Normalize feature value
                value = extracted[key]
                normalized = (
                    value - self.feature_means.get(key, 0)
                ) / self.feature_stds.get(key, 1)
                normalized = max(min(normalized, 3), -3)  # Clip to [-3, 3]

                # Apply weight
                score += weight * normalized * 0.5

        # Apply sigmoid to keep score in [0, 1]
        score = 1.0 / (1.0 + math.exp(-4 * (score - 0.5)))
        score = max(min(score, 1.0), 0.0)

        label = 1 if score >= self.threshold else 0
        confidence = abs(score - self.threshold) * 2
        confidence = min(confidence, 1.0)

        return {"label": label, "score": score, "confidence": confidence}

    def compute_metrics(
        self, predictions: List[int], labels: List[int]
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            predictions: List of predicted labels (0/1)
            labels: List of true labels (0/1)

        Returns:
            Dictionary with accuracy, precision, recall, f1_score, and ece.
        """
        if len(predictions) != len(labels) or len(predictions) == 0:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0, "ece": 0}

        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

        total = len(predictions)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Simplified ECE (Expected Calibration Error)
        ece = max(0.0, 0.15 * (1.0 - accuracy))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "ece": ece,
        }

    def save(self, filepath: str) -> None:
        """
        Save trained model to JSON file.

        Args:
            filepath: Path to save the model (will create .json file)
        """
        # Ensure filepath ends with .json
        if not filepath.endswith(".json"):
            filepath = (
                filepath + ".json" if "." not in filepath.split("/")[-1] else filepath
            )

        model_data = {
            "model_type": "baseline",
            "threshold": self.threshold,
            "learning_rate": self.learning_rate,
            "feature_weights": self.feature_weights,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "trained": self.trained,
            "training_samples": self.training_samples,
            "timestamp": self.timestamp,
            "version": self.version,
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "BaselineMLClassifier":
        """
        Load trained model from JSON file.

        Args:
            filepath: Path to the saved model file

        Returns:
            Loaded BaselineMLClassifier instance
        """
        with open(filepath, "r") as f:
            model_data = json.load(f)

        classifier = cls(
            threshold=model_data.get("threshold", 0.5),
            learning_rate=model_data.get("learning_rate", 0.1),
        )
        classifier.feature_weights = model_data.get("feature_weights", {})
        classifier.feature_means = model_data.get("feature_means", {})
        classifier.feature_stds = model_data.get("feature_stds", {})
        classifier.trained = model_data.get("trained", False)
        classifier.training_samples = model_data.get("training_samples", 0)
        classifier.timestamp = model_data.get("timestamp")
        classifier.version = model_data.get("version", "1.0")

        return classifier


# Optional: Import PyTorch-based AdvancedMLClassifier
# This is kept for backward compatibility and advanced use cases
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    _TORCH_AVAILABLE = True
except ImportError:
    pass  # PyTorch not available, AdvancedMLClassifier will not be defined


if _TORCH_AVAILABLE:

    class AdvancedMLClassifier(nn.Module):
        """Advanced ML classifier with feature scaling and multi-class transformer architecture."""

        def __init__(
            self,
            input_dim: int,
            num_classes: int = 2,
            num_layers: int = 2,
            d_model: int = 64,
            num_heads: int = 4,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.num_classes = num_classes
            self.d_model = d_model

            # Input encoder
            self.input_linear = nn.Linear(input_dim, d_model)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=128, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

            # Output layer (multi-class)
            self.output_layer = nn.Linear(d_model, num_classes)
            self.softmax = nn.Softmax(dim=1)

            # Scaler for features
            self.scaler: Optional[StandardScaler] = None

            # Metadata
            self.trained = False
            self.training_samples = 0
            self.timestamp = None

        def forward(self, x):
            x = self.input_linear(x).unsqueeze(1)  # (batch, seq_len=1, d_model)
            x = self.transformer(x)
            x = x.squeeze(1)
            logits = self.output_layer(x)
            probs = self.softmax(logits)
            return probs

        def fit_scaler(self, X: np.ndarray):
            """Fit feature scaler."""
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        def scale_features(self, X: np.ndarray) -> np.ndarray:
            """Scale features with fitted scaler."""
            if self.scaler is None:
                raise ValueError("Feature scaler not fitted.")
            return self.scaler.transform(X)

        def train_model(
            self, train_data: List[Dict[str, Any]], epochs: int = 50, lr: float = 1e-3
        ):
            """Train classifier with multi-class support and feature scaling."""
            # Prepare features in consistent order
            feature_keys = sorted(train_data[0]["features"].keys())
            X = []
            y = []
            for sample in train_data:
                feat_vector = np.array(
                    [sample["features"][k] for k in feature_keys], dtype=np.float32
                )
                X.append(feat_vector)
                y.append(int(sample["label"]))
            X = np.stack(X)
            self.fit_scaler(X)
            X_scaled = self.scale_features(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)

            optimizer = optim.Adam(self.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            self.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                preds = self.forward(X_tensor)
                loss = criterion(preds, y_tensor)
                loss.backward()
                optimizer.step()

            self.trained = True
            self.training_samples = len(train_data)
            self.timestamp = datetime.now().isoformat()
            self.feature_keys = feature_keys

        def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
            feat_vector = np.array(
                [features[k] for k in self.feature_keys], dtype=np.float32
            )
            X_scaled = self.scale_features(feat_vector.reshape(1, -1))
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            self.eval()
            with torch.no_grad():
                probs = self.forward(X_tensor).cpu().numpy()[0]  # shape: (num_classes,)
            label = int(np.argmax(probs))
            confidence = float(np.max(probs))
            return {
                "label": label,
                "class_probs": {str(i): float(prob) for i, prob in enumerate(probs)},
                "confidence": confidence,
            }

        def compute_metrics(
            self, predictions: List[int], labels: List[int]
        ) -> Dict[str, float]:
            # Multi-class metrics
            confusion = np.zeros((self.num_classes, self.num_classes), dtype=int)
            for pred, label in zip(predictions, labels):
                confusion[label, pred] += 1

            accuracy = sum([confusion[i, i] for i in range(self.num_classes)]) / max(
                len(labels), 1
            )
            per_class_accuracy = [
                confusion[c, c] / max(sum(confusion[c]), 1)
                for c in range(self.num_classes)
            ]

            # Macro-averaged precision/recall/f1
            precision_list = []
            recall_list = []
            f1_list = []
            for c in range(self.num_classes):
                tp = confusion[c, c]
                fp = sum(confusion[:, c]) - tp
                fn = sum(confusion[c, :]) - tp
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = (
                    2 * precision * recall / max(precision + recall, 1e-8)
                    if (precision + recall)
                    else 0
                )
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)

            macro_prec = sum(precision_list) / self.num_classes
            macro_rec = sum(recall_list) / self.num_classes
            macro_f1 = sum(f1_list) / self.num_classes

            return {
                "accuracy": accuracy,
                "macro_precision": macro_prec,
                "macro_recall": macro_rec,
                "macro_f1_score": macro_f1,
                "per_class_accuracy": per_class_accuracy,
                "confusion_matrix": confusion.tolist(),
            }

        def save(self, filepath: str) -> None:
            torch.save(self.state_dict(), filepath + ".pt")
            meta = {
                "input_dim": self.input_dim,
                "d_model": self.d_model,
                "num_classes": self.num_classes,
                "trained": self.trained,
                "training_samples": self.training_samples,
                "timestamp": self.timestamp,
                "feature_keys": getattr(self, "feature_keys", None),
                "scaler_mean": self.scaler.mean_.tolist() if self.scaler else None,
                "scaler_scale": self.scaler.scale_.tolist() if self.scaler else None,
                "version": "2.1",
            }
            with open(filepath + ".meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        @classmethod
        def load(cls, filepath: str) -> "AdvancedMLClassifier":
            # Load meta
            with open(filepath + ".meta.json", "r") as f:
                meta = json.load(f)
            model = cls(
                meta["input_dim"],
                num_classes=meta["num_classes"],
                d_model=meta["d_model"],
            )
            model.load_state_dict(torch.load(filepath + ".pt"))
            model.trained = meta["trained"]
            model.training_samples = meta["training_samples"]
            model.timestamp = meta["timestamp"]
            model.feature_keys = meta["feature_keys"]
            # Restore scaler
            if (
                meta.get("scaler_mean") is not None
                and meta.get("scaler_scale") is not None
            ):
                model.scaler = StandardScaler()
                model.scaler.mean_ = np.array(meta["scaler_mean"])
                model.scaler.scale_ = np.array(meta["scaler_scale"])
            return model
