"""
Anomaly Detection ML Classifier Module

This module provides anomaly detection classifiers for the Nethical framework:
- AnomalyMLClassifier: Lightweight n-gram based anomaly detector for action sequences
- TransformerAnomalyClassifier: Deep learning transformer-based detector (PyTorch)

The AnomalyMLClassifier uses n-gram analysis and entropy calculations to detect
unusual patterns in action sequences, suitable for identifying malicious behavior.

Example usage:
    >>> clf = AnomalyMLClassifier()
    >>> train_data = [
    ...     {'features': {'sequence': ['read', 'process', 'write']}, 'label': 0},
    ...     {'features': {'sequence': ['delete', 'exfiltrate', 'cover']}, 'label': 1},
    ... ]
    >>> clf.train(train_data)
    >>> result = clf.predict({'sequence': ['read', 'process', 'write']})
    >>> print(result['label'], result['score'])

See Also:
    - training/train_any_model.py: Training pipeline with --model-type anomaly
    - docs/TRAINING_GUIDE.md: End-to-end training documentation
    - examples/training/train_anomaly_detector.py: Anomaly detector example
"""

import json
import math
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class AnomalyMLClassifier:
    """
    Lightweight anomaly classifier for action sequence detection.
    
    This classifier uses n-gram analysis and statistical features to detect
    anomalous patterns in action sequences without requiring heavy ML dependencies.
    
    Features analyzed:
    - N-gram patterns (trigrams by default)
    - Action frequency distributions
    - Sequence entropy
    - Pattern diversity
    
    Training data format:
        Each sample should have:
        - 'features': dict with 'sequence' key containing a list of action strings
        - 'label': 0 for normal, 1 for anomalous
    
    Example:
        >>> clf = AnomalyMLClassifier(n=3, anomaly_threshold=0.5)
        >>> clf.train([
        ...     {'features': {'sequence': ['read', 'process', 'write']}, 'label': 0},
        ...     {'features': {'sequence': ['exploit', 'escalate', 'exfil']}, 'label': 1}
        ... ])
        >>> result = clf.predict({'sequence': ['read', 'process', 'write']})
        >>> print(f"Label: {result['label']}, Score: {result['score']:.3f}")
    """
    
    def __init__(self, n: int = 3, anomaly_threshold: float = 0.5):
        """
        Initialize anomaly classifier.
        
        Args:
            n: N-gram size for pattern analysis (default: 3 for trigrams)
            anomaly_threshold: Score threshold for anomaly classification (0-1)
        """
        self.n = n
        self.anomaly_threshold = anomaly_threshold
        
        # Learned patterns
        self.normal_ngrams: Dict[Tuple, int] = {}
        self.anomalous_ngrams: Dict[Tuple, int] = {}
        self.action_frequencies: Dict[str, float] = {}
        self.normal_entropy_stats: Dict[str, float] = {"mean": 0.0, "std": 1.0}
        
        # Training state
        self.trained = False
        self.training_samples = 0
        self.timestamp: Optional[str] = None
        self.version = "1.0"
    
    def _extract_ngrams(self, sequence: List[str]) -> List[Tuple]:
        """Extract n-grams from a sequence."""
        if len(sequence) < self.n:
            return [tuple(sequence)]
        return [tuple(sequence[i:i+self.n]) for i in range(len(sequence) - self.n + 1)]
    
    def _calculate_entropy(self, sequence: List[str]) -> float:
        """Calculate Shannon entropy of sequence."""
        if not sequence:
            return 0.0
        counter = Counter(sequence)
        length = len(sequence)
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                probability = count / length
                entropy -= probability * math.log2(probability)
        return entropy
    
    def _extract_sequence(self, features: Dict[str, Any]) -> List[str]:
        """Extract sequence from features dict."""
        if isinstance(features, dict) and 'sequence' in features:
            seq = features['sequence']
            if isinstance(seq, list):
                return [str(a) for a in seq]
        return []
    
    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train classifier on labeled sequence data.
        
        Args:
            train_data: List of samples with 'features' (containing 'sequence') and 'label'
        
        Raises:
            ValueError: If training data is empty
        """
        if not train_data:
            raise ValueError("Training data cannot be empty")
        
        # Reset learned patterns
        self.normal_ngrams = defaultdict(int)
        self.anomalous_ngrams = defaultdict(int)
        action_counts: Dict[str, int] = defaultdict(int)
        normal_entropies: List[float] = []
        
        for sample in train_data:
            features = sample.get("features", {})
            label = int(sample.get("label", 0))
            
            sequence = self._extract_sequence(features)
            if not sequence:
                continue
            
            # Count actions
            for action in sequence:
                action_counts[action] += 1
            
            # Extract n-grams
            ngrams = self._extract_ngrams(sequence)
            target_dict = self.anomalous_ngrams if label == 1 else self.normal_ngrams
            for ng in ngrams:
                target_dict[ng] += 1
            
            # Collect entropy stats for normal sequences
            if label == 0:
                normal_entropies.append(self._calculate_entropy(sequence))
        
        # Convert to regular dicts
        self.normal_ngrams = dict(self.normal_ngrams)
        self.anomalous_ngrams = dict(self.anomalous_ngrams)
        
        # Compute action frequencies
        total_actions = sum(action_counts.values())
        self.action_frequencies = {
            action: count / total_actions 
            for action, count in action_counts.items()
        }
        
        # Compute normal entropy statistics
        if normal_entropies:
            mean_entropy = sum(normal_entropies) / len(normal_entropies)
            variance = sum((e - mean_entropy) ** 2 for e in normal_entropies) / len(normal_entropies)
            self.normal_entropy_stats = {
                "mean": mean_entropy,
                "std": math.sqrt(variance) if variance > 0 else 1.0
            }
        
        self.trained = True
        self.training_samples = len(train_data)
        self.timestamp = datetime.now().isoformat()
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict anomaly score for a sequence.
        
        Args:
            features: Dict with 'sequence' key containing list of action strings
        
        Returns:
            Dict with 'label' (0/1), 'score' (0-1), and 'confidence' (0-1)
        """
        sequence = self._extract_sequence(features)
        if not sequence:
            return {"label": 0, "score": 0.0, "confidence": 0.0}
        
        # Calculate anomaly indicators
        scores = []
        
        # 1. N-gram novelty score
        ngrams = self._extract_ngrams(sequence)
        if ngrams:
            normal_matches = sum(1 for ng in ngrams if ng in self.normal_ngrams)
            anomal_matches = sum(1 for ng in ngrams if ng in self.anomalous_ngrams)
            
            # Novelty: proportion not in normal patterns
            novelty = 1.0 - (normal_matches / len(ngrams)) if ngrams else 0.0
            # Suspicious: proportion matching anomalous patterns
            suspicious = anomal_matches / len(ngrams) if ngrams else 0.0
            
            ngram_score = (novelty * 0.4 + suspicious * 0.6)
            scores.append(ngram_score)
        
        # 2. Action rarity score
        rarity_scores = []
        for action in sequence:
            freq = self.action_frequencies.get(action, 0.0)
            # Rare actions (low frequency) get high scores
            rarity = 1.0 - min(freq * 10, 1.0)
            rarity_scores.append(rarity)
        if rarity_scores:
            scores.append(sum(rarity_scores) / len(rarity_scores))
        
        # 3. Entropy deviation score
        entropy = self._calculate_entropy(sequence)
        mean = self.normal_entropy_stats["mean"]
        std = self.normal_entropy_stats["std"]
        if std > 0:
            z_score = abs(entropy - mean) / std
            entropy_score = min(z_score / 3.0, 1.0)  # Cap at 3 std devs
        else:
            entropy_score = 0.5
        scores.append(entropy_score)
        
        # 4. Pattern diversity score
        unique_ratio = len(set(sequence)) / len(sequence) if sequence else 0
        # Both very low and very high diversity can be suspicious
        diversity_score = abs(unique_ratio - 0.5) * 2
        scores.append(diversity_score)
        
        # Combine scores
        final_score = sum(scores) / len(scores) if scores else 0.0
        final_score = max(min(final_score, 1.0), 0.0)
        
        label = 1 if final_score >= self.anomaly_threshold else 0
        confidence = abs(final_score - self.anomaly_threshold) * 2
        confidence = min(confidence, 1.0)
        
        return {
            "label": label,
            "score": final_score,
            "confidence": confidence
        }
    
    def compute_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Compute classification metrics."""
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
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        ece = max(0.0, 0.15 * (1.0 - accuracy))
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "ece": ece
        }
    
    def save(self, filepath: str) -> None:
        """Save trained model to JSON file."""
        if not filepath.endswith('.json'):
            filepath = filepath + '.json' if '.' not in filepath.split('/')[-1] else filepath
        
        # Convert tuple keys to strings for JSON serialization
        normal_ngrams_str = {str(k): v for k, v in self.normal_ngrams.items()}
        anomal_ngrams_str = {str(k): v for k, v in self.anomalous_ngrams.items()}
        
        model_data = {
            "model_type": "anomaly",
            "n": self.n,
            "anomaly_threshold": self.anomaly_threshold,
            "normal_ngrams": normal_ngrams_str,
            "anomalous_ngrams": anomal_ngrams_str,
            "action_frequencies": self.action_frequencies,
            "normal_entropy_stats": self.normal_entropy_stats,
            "trained": self.trained,
            "training_samples": self.training_samples,
            "timestamp": self.timestamp,
            "version": self.version
        }
        
        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "AnomalyMLClassifier":
        """Load trained model from JSON file."""
        with open(filepath, "r") as f:
            model_data = json.load(f)
        
        classifier = cls(
            n=model_data.get("n", 3),
            anomaly_threshold=model_data.get("anomaly_threshold", 0.5)
        )
        
        # Convert string keys back to tuples using ast.literal_eval for safety
        import ast
        normal_ngrams_str = model_data.get("normal_ngrams", {})
        anomal_ngrams_str = model_data.get("anomalous_ngrams", {})
        classifier.normal_ngrams = {ast.literal_eval(k): v for k, v in normal_ngrams_str.items()}
        classifier.anomalous_ngrams = {ast.literal_eval(k): v for k, v in anomal_ngrams_str.items()}
        
        classifier.action_frequencies = model_data.get("action_frequencies", {})
        classifier.normal_entropy_stats = model_data.get("normal_entropy_stats", {"mean": 0.0, "std": 1.0})
        classifier.trained = model_data.get("trained", False)
        classifier.training_samples = model_data.get("training_samples", 0)
        classifier.timestamp = model_data.get("timestamp")
        classifier.version = model_data.get("version", "1.0")
        
        return classifier


# ============================================================================
# Legacy Transformer-Based Anomaly Classifier (requires PyTorch)
# ============================================================================
# The following code provides a deep learning transformer-based classifier
# for more advanced anomaly detection scenarios.

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    from scipy.stats import entropy as scipy_entropy
    from sklearn.metrics import roc_auc_score
    
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    from itertools import groupby
    from collections import Counter

    # --- Data Utilities (PyTorch-based) ---

    def build_tokenizer(sequences):
        """Creates a dictionary mapping actions to integer IDs."""
        actions = set(a for seq in sequences for a in seq)
        tok = {a: i+1 for i, a in enumerate(sorted(actions))}  # 0 is reserved for padding
        return tok

    def encode_sequences(sequences, tok, max_len=None):
        """Encodes action sequences using tokenizer."""
        if max_len is None:
            max_len = max(len(seq) for seq in sequences)
        encoded = []
        for seq in sequences:
            seq_ids = [tok.get(a, 0) for a in seq]
            if len(seq_ids) < max_len:
                seq_ids += [0]*(max_len - len(seq_ids))
            else:
                seq_ids = seq_ids[:max_len]
            encoded.append(seq_ids)
        return np.array(encoded)

    def inverse_tokenizer(tok):
        return {v:k for k,v in tok.items()}

    # --- Feature Engineering (PyTorch-based) ---

    def extract_features(sequence):
        """Extract features from a sequence for transformer model."""
        features = {}

        # N-gram rarity
        n = 3
        ngrams = [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
        features['ngram_count'] = len(ngrams)
        features['unique_ngrams'] = len(set(ngrams))/max(len(ngrams),1)
        
        # Action frequency deviation
        act_counts = Counter(sequence)
        freq = [v/len(sequence) for v in act_counts.values()]
        features['freq_entropy'] = scipy_entropy(freq)
        
        # Sequence entropy (Shannon)
        features['shannon_entropy'] = scipy_entropy([act_counts[a]/len(sequence) for a in act_counts])
        
        # Pattern diversity
        features['pattern_diversity'] = len(set(sequence))/len(sequence)
        
        # Markov transitions
        transitions = [tuple(sequence[i:i+2]) for i in range(len(sequence)-1)]
        trans_counts = Counter(transitions)
        trans_freq = [v/len(transitions) for v in trans_counts.values()] if transitions else [1.0]
        features['markov_entropy'] = scipy_entropy(trans_freq)
        
        # Palindromic pattern detection
        features['is_palindrome'] = int(sequence == sequence[::-1])
        
        # Rare event statistics
        rare_events = [action for action, cnt in act_counts.items() if cnt < (0.05 * len(sequence))]
        features['rare_event_ratio'] = len(rare_events)/max(len(set(sequence)), 1)
        
        # Interleaved actions (count alternations)
        alternations = sum(1 for i in range(1,len(sequence)) if sequence[i] != sequence[i-1])
        features['alternation_ratio'] = alternations/max(len(sequence), 1)
        
        # Motif detection: looks for certain subsequences (here, all sequences of length 2)
        unique_pairs = set(tuple(sequence[i:i+2]) for i in range(len(sequence)-1))
        features['motif_count'] = len(unique_pairs)
        
        # Duration/frequency statistics
        features['avg_action_runlength'] = np.mean([len(list(group)) for _, group in groupby(sequence)])
        
        # Position-based (are anomalies at head/tail)
        features['head_action'] = sequence[0] if sequence else ''
        features['tail_action'] = sequence[-1] if sequence else ''
        
        return features

    # --- Transformer Model (PyTorch-based) ---

    class TransformerAnomalyClassifier(nn.Module):
        """
        Deep learning transformer-based anomaly classifier.
        
        Requires PyTorch. For a lightweight alternative without PyTorch,
        use AnomalyMLClassifier instead.
        """
        def __init__(self, vocab_size, feature_dim, d_model=64, nhead=4, num_layers=2, max_len=48):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size+1, d_model, padding_idx=0)
            self.pos_embedding = nn.Embedding(max_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Feature input
            self.feature_fc = nn.Linear(feature_dim, d_model)
            # Output
            self.final_fc = nn.Linear(d_model*2, 1)
            
        def forward(self, x, features):
            # x: [B, L]
            B, L = x.shape
            x_emb = self.embedding(x)                   # [B, L, d_model]
            pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            pos_emb = self.pos_embedding(pos_ids)       # [B, L, d_model]
            h_seq = self.transformer(x_emb + pos_emb)   # [B, L, d_model]
            h_seq_sum = h_seq.mean(dim=1)               # [B, d_model] (simple pooling)
            
            h_feat = self.feature_fc(features)          # [B, d_model]
            h = torch.cat([h_seq_sum, h_feat], dim=1)   # [B, 2*d_model]
            out = self.final_fc(h)
            return torch.sigmoid(out.squeeze(1))        # [B]
