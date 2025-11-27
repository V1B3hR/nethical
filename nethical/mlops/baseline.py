import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler


class AdvancedMLClassifier(nn.Module):
    """Advanced ML classifier with feature scaling and multi-class transformer architecture."""

    def __init__(self, input_dim: int, num_classes: int = 2, num_layers: int = 2, d_model: int = 64, num_heads: int = 4):
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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

    def train_model(self, train_data: List[Dict[str, Any]], epochs: int = 50, lr: float = 1e-3):
        """Train classifier with multi-class support and feature scaling."""
        # Prepare features in consistent order
        feature_keys = sorted(train_data[0]['features'].keys())
        X = []
        y = []
        for sample in train_data:
            feat_vector = np.array([sample["features"][k] for k in feature_keys], dtype=np.float32)
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
        feat_vector = np.array([features[k] for k in self.feature_keys], dtype=np.float32)
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
            "confidence": confidence
        }

    def compute_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        # Multi-class metrics
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for pred, label in zip(predictions, labels):
            confusion[label, pred] += 1

        accuracy = sum([confusion[i, i] for i in range(self.num_classes)]) / max(len(labels), 1)
        per_class_accuracy = [
            confusion[c, c] / max(sum(confusion[c]), 1) for c in range(self.num_classes)
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
            f1 = 2 * precision * recall / max(precision + recall, 1e-8) if (precision + recall) else 0
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
            "confusion_matrix": confusion.tolist()
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
            "version": "2.1"
        }
        with open(filepath + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "AdvancedMLClassifier":
        # Load meta
        with open(filepath + ".meta.json", "r") as f:
            meta = json.load(f)
        model = cls(meta["input_dim"], num_classes=meta["num_classes"], d_model=meta["d_model"])
        model.load_state_dict(torch.load(filepath + ".pt"))
        model.trained = meta["trained"]
        model.training_samples = meta["training_samples"]
        model.timestamp = meta["timestamp"]
        model.feature_keys = meta["feature_keys"]
        # Restore scaler
        if meta.get("scaler_mean") is not None and meta.get("scaler_scale") is not None:
            model.scaler = StandardScaler()
            model.scaler.mean_ = np.array(meta["scaler_mean"])
            model.scaler.scale_ = np.array(meta["scaler_scale"])
        return model
