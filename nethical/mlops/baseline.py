import json
import math
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

# --- BaselineMLClassifier definition (from baseline.py) ---

class BaselineMLClassifier:
    """Simple baseline classifier for ML training pipeline.
    
    This classifier uses a weighted feature combination approach similar to
    the MLShadowClassifier but optimized for supervised learning with labels.
    """
    
    def __init__(self):
        """Initialize baseline classifier."""
        # Default feature weights
        self.feature_weights = {
            'violation_count': 0.3,
            'severity_max': 0.25,
            'recency_score': 0.2,
            'frequency_score': 0.15,
            'context_risk': 0.1
        }
        
        # Model metadata
        self.trained = False
        self.training_samples = 0
        self.timestamp = None
    
    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """Train the classifier on labeled data.
        
        For this baseline implementation, we use a simple approach:
        - Learn optimal feature weights from training data
        - No complex optimization, just basic statistics
        
        Args:
            train_data: List of training samples with 'features' and 'label'
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")
        
        # Simple training: adjust weights based on feature importance
        # Calculate correlation of each feature with labels
        feature_sums = {k: 0.0 for k in self.feature_weights.keys()}
        label_sums = {k: {'positive': 0.0, 'negative': 0.0} for k in self.feature_weights.keys()}
        
        for sample in train_data:
            features = sample.get('features', {})
            label = sample.get('label', 0)
            
            for feature_name in self.feature_weights.keys():
                feature_value = features.get(feature_name, 0.0)
                if isinstance(feature_value, (int, float)):
                    feature_sums[feature_name] += feature_value
                    if label == 1:
                        label_sums[feature_name]['positive'] += feature_value
                    else:
                        label_sums[feature_name]['negative'] += feature_value
        
        # Adjust weights based on discriminative power
        # Features that differ more between positive/negative get higher weight
        total_weight = 0.0
        adjusted_weights = {}
        
        for feature_name in self.feature_weights.keys():
            pos_avg = label_sums[feature_name]['positive']
            neg_avg = label_sums[feature_name]['negative']
            
            # Discriminative power = difference between positive and negative averages
            discriminative_power = abs(pos_avg - neg_avg) + 0.01  # Add small epsilon
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
        
        # Compute weighted score
        score = 0.0
        for feature_name, weight in self.feature_weights.items():
            feature_value = features.get(feature_name, 0.0)
            if isinstance(feature_value, (int, float)):
                score += weight * min(feature_value, 1.0)
        
        # Apply sigmoid-like transformation for non-linearity
        score = 1.0 - math.exp(-2 * score)
        score = min(score, 1.0)
        
        # Determine label (threshold at 0.5)
        label = 1 if score >= 0.5 else 0
        
        # Calculate confidence based on distance from threshold
        confidence = abs(score - 0.5) * 2  # Scale 0-0.5 to 0-1
        confidence = min(confidence, 1.0)
        
        return {
            'label': label,
            'score': score,
            'confidence': confidence
        }
    
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
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'ece': 0.0
            }
        
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
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Simple ECE approximation (would need probability scores for proper calculation)
        # For now, use a placeholder based on accuracy
        ece = max(0.0, 0.15 * (1.0 - accuracy))  # Higher error when accuracy is lower
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'ece': ece,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def save(self, filepath: str) -> None:
        """Save model to JSON file.
        
        Args:
            filepath: Path to save model
        """
        model_data = {
            'model_type': 'baseline',
            'feature_weights': self.feature_weights,
            'trained': self.trained,
            'training_samples': self.training_samples,
            'timestamp': self.timestamp or datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaselineMLClassifier':
        """Load model from JSON file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded classifier instance
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        classifier = cls()
        classifier.feature_weights = model_data.get('feature_weights', classifier.feature_weights)
        classifier.trained = model_data.get('trained', False)
        classifier.training_samples = model_data.get('training_samples', 0)
        classifier.timestamp = model_data.get('timestamp')
        
        return classifier

# --- End of BaselineMLClassifier definition ---

# --- Main script for training and evaluation ---

# Path to your processed dataset (must be a list of {"features": {...}, "label": ...})
DATA_PATH = "processed_train_data.json"  # Update this to your actual file

# Load data
with open(DATA_PATH, "r") as f:
    data = json.load(f)

# Shuffle and split into train and validation sets
random.shuffle(data)
split = int(0.8 * len(data))
train_data = data[:split]
val_data = data[split:]

# Train model
clf = BaselineMLClassifier()
clf.train(train_data)

# Evaluate on validation set
val_features = [d["features"] for d in val_data]
val_labels = [d["label"] for d in val_data]
val_preds = [clf.predict(f)["label"] for f in val_features]

metrics = clf.compute_metrics(val_preds, val_labels)
print("Validation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Save trained model
clf.save("baseline_model.json")
print("Model saved to baseline_model.json")
