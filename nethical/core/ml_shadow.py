"""Phase 5: ML Shadow Mode Implementation.

This module implements:
- Minimal classifier (logistic regression / simple model)
- Passive inference with no enforcement authority
- Prediction logging alongside rule-based outcomes
- Baseline metrics collection (precision, recall, F1, calibration)
- No impact on enforcement decisions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import json
import math
from collections import defaultdict


class MLModelType(str, Enum):
    """Types of ML models supported in shadow mode."""
    LOGISTIC = "logistic"
    SIMPLE_TRANSFORMER = "simple_transformer"
    HEURISTIC = "heuristic"


@dataclass
class ShadowPrediction:
    """Shadow mode prediction result."""
    prediction_id: str
    timestamp: datetime
    agent_id: str
    action_id: str
    
    # ML prediction
    ml_risk_score: float  # 0-1
    ml_classification: str  # "allow", "warn", "deny"
    ml_confidence: float  # 0-1
    
    # Rule-based comparison
    rule_risk_score: float
    rule_classification: str
    
    # Agreement tracking
    scores_agree: bool  # Within threshold
    classifications_agree: bool
    
    # Features used
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'prediction_id': self.prediction_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'action_id': self.action_id,
            'ml_risk_score': self.ml_risk_score,
            'ml_classification': self.ml_classification,
            'ml_confidence': self.ml_confidence,
            'rule_risk_score': self.rule_risk_score,
            'rule_classification': self.rule_classification,
            'scores_agree': self.scores_agree,
            'classifications_agree': self.classifications_agree,
            'features': self.features
        }


@dataclass
class ShadowMetrics:
    """Metrics for shadow mode evaluation."""
    total_predictions: int = 0
    
    # Classification metrics (comparing to rule-based ground truth)
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Agreement metrics
    score_agreement_count: int = 0
    classification_agreement_count: int = 0
    
    # Calibration bins (for reliability diagram)
    calibration_bins: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        '0.0-0.2': {'correct': 0, 'total': 0},
        '0.2-0.4': {'correct': 0, 'total': 0},
        '0.4-0.6': {'correct': 0, 'total': 0},
        '0.6-0.8': {'correct': 0, 'total': 0},
        '0.8-1.0': {'correct': 0, 'total': 0}
    })
    
    @property
    def precision(self) -> float:
        """Calculate precision."""
        denom = self.true_positives + self.false_positives
        if denom == 0:
            return 0.0
        return self.true_positives / denom
    
    @property
    def recall(self) -> float:
        """Calculate recall."""
        denom = self.true_positives + self.false_negatives
        if denom == 0:
            return 0.0
        return self.true_positives / denom
    
    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    @property
    def score_agreement_rate(self) -> float:
        """Calculate score agreement rate."""
        if self.total_predictions == 0:
            return 0.0
        return self.score_agreement_count / self.total_predictions
    
    @property
    def classification_agreement_rate(self) -> float:
        """Calculate classification agreement rate."""
        if self.total_predictions == 0:
            return 0.0
        return self.classification_agreement_count / self.total_predictions
    
    def get_calibration_error(self) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        total_samples = sum(bin_data['total'] for bin_data in self.calibration_bins.values())
        if total_samples == 0:
            return 0.0
        
        ece = 0.0
        bin_centers = {
            '0.0-0.2': 0.1,
            '0.2-0.4': 0.3,
            '0.4-0.6': 0.5,
            '0.6-0.8': 0.7,
            '0.8-1.0': 0.9
        }
        
        for bin_name, bin_data in self.calibration_bins.items():
            if bin_data['total'] > 0:
                bin_accuracy = bin_data['correct'] / bin_data['total']
                bin_confidence = bin_centers[bin_name]
                bin_weight = bin_data['total'] / total_samples
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'total_predictions': self.total_predictions,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy,
            'score_agreement_rate': self.score_agreement_rate,
            'classification_agreement_rate': self.classification_agreement_rate,
            'expected_calibration_error': self.get_calibration_error(),
            'confusion_matrix': {
                'true_positives': self.true_positives,
                'true_negatives': self.true_negatives,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives
            }
        }


class MLShadowClassifier:
    """ML Shadow Mode classifier - passive inference only."""
    
    def __init__(
        self,
        model_type: MLModelType = MLModelType.HEURISTIC,
        score_agreement_threshold: float = 0.1,
        storage_path: Optional[str] = None
    ):
        """Initialize shadow classifier.
        
        Args:
            model_type: Type of ML model to use
            score_agreement_threshold: Threshold for score agreement (default: 0.1)
            storage_path: Path for storing predictions and metrics
        """
        self.model_type = model_type
        self.score_agreement_threshold = score_agreement_threshold
        self.storage_path = storage_path
        
        # Prediction log
        self.predictions: List[ShadowPrediction] = []
        
        # Metrics
        self.metrics = ShadowMetrics()
        
        # Model weights (for heuristic model)
        self.feature_weights = {
            'violation_count': 0.3,
            'severity_max': 0.25,
            'recency_score': 0.2,
            'frequency_score': 0.15,
            'context_risk': 0.1
        }
    
    def predict(
        self,
        agent_id: str,
        action_id: str,
        features: Dict[str, Any],
        rule_risk_score: float,
        rule_classification: str
    ) -> ShadowPrediction:
        """Make shadow prediction (passive - no enforcement).
        
        Args:
            agent_id: Agent identifier
            action_id: Action identifier
            features: Extracted features for prediction
            rule_risk_score: Rule-based risk score for comparison
            rule_classification: Rule-based classification for comparison
            
        Returns:
            Shadow prediction result
        """
        # Generate ML prediction
        ml_risk_score, ml_confidence = self._compute_ml_score(features)
        ml_classification = self._score_to_classification(ml_risk_score)
        
        # Compare with rule-based outcome
        scores_agree = abs(ml_risk_score - rule_risk_score) <= self.score_agreement_threshold
        classifications_agree = ml_classification == rule_classification
        
        # Create prediction record
        prediction = ShadowPrediction(
            prediction_id=f"shadow_{int(datetime.utcnow().timestamp() * 1000000)}",
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            action_id=action_id,
            ml_risk_score=ml_risk_score,
            ml_classification=ml_classification,
            ml_confidence=ml_confidence,
            rule_risk_score=rule_risk_score,
            rule_classification=rule_classification,
            scores_agree=scores_agree,
            classifications_agree=classifications_agree,
            features=features
        )
        
        # Log prediction
        self.predictions.append(prediction)
        
        # Update metrics
        self._update_metrics(prediction)
        
        # Persist if configured
        if self.storage_path:
            self._persist_prediction(prediction)
        
        return prediction
    
    def _compute_ml_score(
        self,
        features: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Compute ML risk score and confidence.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tuple of (risk_score, confidence)
        """
        if self.model_type == MLModelType.HEURISTIC:
            return self._heuristic_model(features)
        elif self.model_type == MLModelType.LOGISTIC:
            # Placeholder for actual logistic regression
            return self._heuristic_model(features)
        else:
            # Default to heuristic
            return self._heuristic_model(features)
    
    def _heuristic_model(
        self,
        features: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Simple heuristic-based model for shadow mode.
        
        This serves as a baseline and placeholder for actual ML models.
        """
        score = 0.0
        confidence = 0.8  # Base confidence
        
        # Weighted feature combination
        for feature_name, weight in self.feature_weights.items():
            feature_value = features.get(feature_name, 0.0)
            if isinstance(feature_value, (int, float)):
                score += weight * min(feature_value, 1.0)
        
        # Add some non-linearity
        score = 1.0 - math.exp(-2 * score)  # Sigmoid-like transformation
        
        # Adjust confidence based on feature completeness
        available_features = sum(1 for k in self.feature_weights.keys() if k in features)
        confidence *= (available_features / len(self.feature_weights))
        
        return min(score, 1.0), min(confidence, 1.0)
    
    def _score_to_classification(self, score: float) -> str:
        """Convert risk score to classification."""
        if score >= 0.7:
            return "deny"
        elif score >= 0.4:
            return "warn"
        else:
            return "allow"
    
    def _update_metrics(self, prediction: ShadowPrediction) -> None:
        """Update metrics based on prediction.
        
        Uses rule-based classification as ground truth for evaluation.
        """
        self.metrics.total_predictions += 1
        
        # Update agreement metrics
        if prediction.scores_agree:
            self.metrics.score_agreement_count += 1
        
        if prediction.classifications_agree:
            self.metrics.classification_agreement_count += 1
        
        # Update confusion matrix (treating "deny" as positive class)
        ml_positive = prediction.ml_classification == "deny"
        rule_positive = prediction.rule_classification == "deny"
        
        if ml_positive and rule_positive:
            self.metrics.true_positives += 1
        elif not ml_positive and not rule_positive:
            self.metrics.true_negatives += 1
        elif ml_positive and not rule_positive:
            self.metrics.false_positives += 1
        else:  # not ml_positive and rule_positive
            self.metrics.false_negatives += 1
        
        # Update calibration bins
        confidence = prediction.ml_confidence
        correct = prediction.classifications_agree
        
        bin_name = self._get_calibration_bin(confidence)
        self.metrics.calibration_bins[bin_name]['total'] += 1
        if correct:
            self.metrics.calibration_bins[bin_name]['correct'] += 1
    
    def _get_calibration_bin(self, confidence: float) -> str:
        """Get calibration bin for confidence value."""
        if confidence < 0.2:
            return '0.0-0.2'
        elif confidence < 0.4:
            return '0.2-0.4'
        elif confidence < 0.6:
            return '0.4-0.6'
        elif confidence < 0.8:
            return '0.6-0.8'
        else:
            return '0.8-1.0'
    
    def _persist_prediction(self, prediction: ShadowPrediction) -> None:
        """Persist prediction to storage."""
        if not self.storage_path:
            return
        
        try:
            import os
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Append to predictions log
            log_file = os.path.join(self.storage_path, "shadow_predictions.jsonl")
            with open(log_file, 'a') as f:
                f.write(json.dumps(prediction.to_dict()) + '\n')
        except Exception:
            pass  # Silent fail for logging
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics report.
        
        Returns:
            Dictionary with all metrics and analysis
        """
        report = self.metrics.to_dict()
        
        # Add additional analysis
        report['model_type'] = self.model_type.value
        report['total_logged_predictions'] = len(self.predictions)
        
        # Recent predictions analysis
        if len(self.predictions) > 0:
            recent_predictions = self.predictions[-100:]
            recent_agreement = sum(
                1 for p in recent_predictions 
                if p.classifications_agree
            ) / len(recent_predictions)
            report['recent_classification_agreement'] = recent_agreement
        
        return report
    
    def export_predictions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Export predictions for analysis.
        
        Args:
            limit: Optional limit on number of predictions to export
            
        Returns:
            List of prediction dictionaries
        """
        predictions_to_export = self.predictions[-limit:] if limit else self.predictions
        return [p.to_dict() for p in predictions_to_export]
    
    def reset_metrics(self) -> None:
        """Reset metrics (useful for evaluation periods)."""
        self.metrics = ShadowMetrics()
        self.predictions.clear()
