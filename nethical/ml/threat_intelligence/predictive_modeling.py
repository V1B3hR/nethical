"""
Predictive Modeling Module.

Implements machine learning models to predict emerging threats and
attack patterns before they occur.

Phase: 5 - Detection Omniscience
Component: Threat Anticipation
"""

import asyncio
import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ThreatTrend(Enum):
    """Trend direction for threat evolution."""
    
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    EMERGING = "emerging"
    DECLINING = "declining"


@dataclass
class AttackPrediction:
    """Represents a predicted future attack."""
    
    prediction_id: str
    attack_type: str
    predicted_vector: str
    probability: float  # 0.0 to 1.0
    confidence: float   # 0.0 to 1.0
    time_horizon_days: int
    indicators: List[str] = field(default_factory=list)
    precursor_patterns: List[str] = field(default_factory=list)
    recommended_defenses: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "prediction_id": self.prediction_id,
            "attack_type": self.attack_type,
            "predicted_vector": self.predicted_vector,
            "probability": self.probability,
            "confidence": self.confidence,
            "time_horizon_days": self.time_horizon_days,
            "indicators": self.indicators,
            "precursor_patterns": self.precursor_patterns,
            "recommended_defenses": self.recommended_defenses,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ThreatEvolutionModel:
    """Model of how a threat evolves over time."""
    
    threat_family: str
    current_variants: List[str]
    evolution_trajectory: List[Dict[str, Any]]
    mutation_rate: float  # Variants per time period
    complexity_trend: ThreatTrend
    sophistication_score: float  # 0.0 to 1.0
    next_generation_features: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PredictiveModeler:
    """
    Implements predictive modeling for threat anticipation.
    
    Features:
    - Trend analysis of attack patterns
    - Attack evolution modeling
    - Time-series forecasting
    - Anomaly-based prediction
    - Probabilistic threat assessment
    """
    
    def __init__(
        self,
        prediction_threshold: float = 0.7,
        time_horizons: Optional[List[int]] = None,
        learning_window_days: int = 90,
    ):
        """
        Initialize predictive modeler.
        
        Args:
            prediction_threshold: Minimum probability to issue prediction
            time_horizons: Prediction time horizons in days
            learning_window_days: Historical data window for learning
        """
        self.prediction_threshold = prediction_threshold
        self.time_horizons = time_horizons or [7, 30, 90]  # 1 week, 1 month, 3 months
        self.learning_window_days = learning_window_days
        
        # Storage
        self.predictions: Dict[str, AttackPrediction] = {}
        self.evolution_models: Dict[str, ThreatEvolutionModel] = {}
        self.historical_attacks: List[Dict[str, Any]] = []
        
        # Model parameters (simplified for demonstration)
        self.trend_models: Dict[str, Any] = {}
        
        # Statistics
        self.total_predictions: int = 0
        self.accurate_predictions: int = 0
        self.false_positives: int = 0
        
        logger.info(
            f"PredictiveModeler initialized with threshold={prediction_threshold}"
        )
    
    async def analyze_trends(
        self, attack_history: List[Dict[str, Any]]
    ) -> Dict[str, ThreatTrend]:
        """
        Analyze trends in attack patterns.
        
        Args:
            attack_history: Historical attack data
            
        Returns:
            Dictionary mapping attack types to trends
        """
        trends: Dict[str, ThreatTrend] = {}
        
        # Group attacks by type and time
        attack_counts = defaultdict(lambda: defaultdict(int))
        
        for attack in attack_history:
            attack_type = attack.get("type", "unknown")
            timestamp = attack.get("timestamp", datetime.now(timezone.utc))
            
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # Bucket by week
            week = timestamp.isocalendar()[:2]  # (year, week)
            attack_counts[attack_type][week] += 1
        
        # Analyze trend for each attack type
        for attack_type, weekly_counts in attack_counts.items():
            if len(weekly_counts) < 2:
                trends[attack_type] = ThreatTrend.STABLE
                continue
            
            # Simple linear trend analysis
            weeks = sorted(weekly_counts.keys())
            counts = [weekly_counts[w] for w in weeks]
            
            if len(counts) >= 2:
                # Calculate trend (simplified)
                recent_avg = np.mean(counts[-4:]) if len(counts) >= 4 else counts[-1]
                older_avg = np.mean(counts[:-4]) if len(counts) >= 4 else counts[0]
                
                if recent_avg > older_avg * 1.5:
                    trends[attack_type] = ThreatTrend.INCREASING
                elif recent_avg < older_avg * 0.5:
                    trends[attack_type] = ThreatTrend.DECREASING
                elif recent_avg > 0 and older_avg == 0:
                    trends[attack_type] = ThreatTrend.EMERGING
                else:
                    trends[attack_type] = ThreatTrend.STABLE
            else:
                trends[attack_type] = ThreatTrend.STABLE
        
        logger.info(f"Analyzed trends for {len(trends)} attack types")
        return trends
    
    async def predict_attacks(
        self, threat_intelligence: List[Any]
    ) -> List[AttackPrediction]:
        """
        Generate predictions for future attacks.
        
        Args:
            threat_intelligence: Current threat intelligence data
            
        Returns:
            List of attack predictions
        """
        predictions = []
        
        # Analyze current threat landscape
        threat_types = defaultdict(list)
        for threat in threat_intelligence:
            threat_type = getattr(threat, "attack_vectors", ["unknown"])[0] if hasattr(threat, "attack_vectors") else "unknown"
            threat_types[threat_type].append(threat)
        
        # Generate predictions for each time horizon
        for horizon_days in self.time_horizons:
            for threat_type, threats in threat_types.items():
                prediction = await self._generate_prediction(
                    threat_type, threats, horizon_days
                )
                
                if prediction and prediction.probability >= self.prediction_threshold:
                    predictions.append(prediction)
                    self.predictions[prediction.prediction_id] = prediction
                    self.total_predictions += 1
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    async def _generate_prediction(
        self,
        threat_type: str,
        related_threats: List[Any],
        horizon_days: int,
    ) -> Optional[AttackPrediction]:
        """
        Generate a single attack prediction.
        
        Args:
            threat_type: Type of threat to predict
            related_threats: Related threat intelligence
            horizon_days: Prediction time horizon
            
        Returns:
            Attack prediction or None
        """
        try:
            # Calculate probability based on threat density and severity
            num_threats = len(related_threats)
            
            if num_threats == 0:
                return None
            
            # Simple probability model (would be ML-based in production)
            base_probability = min(0.9, num_threats * 0.1)
            
            # Adjust for time horizon (farther = less certain)
            time_factor = 1.0 / (1 + horizon_days / 30)
            probability = base_probability * time_factor
            
            # Calculate confidence (simplified)
            confidence = min(0.95, 0.5 + num_threats * 0.05)
            
            # Extract indicators from threats
            indicators = []
            for threat in related_threats[:5]:  # Top 5
                if hasattr(threat, "indicators"):
                    indicators.extend(threat.indicators[:3])
            
            prediction_id = f"pred_{threat_type}_{horizon_days}_{datetime.now(timezone.utc).timestamp()}"
            
            return AttackPrediction(
                prediction_id=prediction_id,
                attack_type=threat_type,
                predicted_vector=f"{threat_type}_variant",
                probability=probability,
                confidence=confidence,
                time_horizon_days=horizon_days,
                indicators=list(set(indicators))[:10],
                precursor_patterns=[f"{threat_type}_precursor_{i}" for i in range(3)],
                recommended_defenses=[
                    f"Deploy detector for {threat_type}",
                    f"Monitor for {threat_type} indicators",
                    f"Update rules for {threat_type}",
                ],
                metadata={
                    "related_threats_count": num_threats,
                    "threat_type": threat_type,
                },
            )
            
        except Exception as e:
            logger.error(f"Error generating prediction: {e}")
            return None
    
    async def model_threat_evolution(
        self, threat_family: str, historical_data: List[Dict[str, Any]]
    ) -> ThreatEvolutionModel:
        """
        Model how a threat family evolves over time.
        
        Args:
            threat_family: Family of threats to model
            historical_data: Historical attack data
            
        Returns:
            Evolution model for the threat family
        """
        # Extract variants over time
        variants = set()
        trajectory = []
        
        for entry in historical_data:
            if entry.get("family") == threat_family:
                variant = entry.get("variant", "unknown")
                variants.add(variant)
                
                trajectory.append({
                    "timestamp": entry.get("timestamp", datetime.now(timezone.utc)),
                    "variant": variant,
                    "sophistication": entry.get("sophistication", 0.5),
                })
        
        # Calculate mutation rate (variants per month)
        if len(trajectory) > 1:
            time_span = (
                max(e["timestamp"] for e in trajectory)
                - min(e["timestamp"] for e in trajectory)
            ).days / 30
            mutation_rate = len(variants) / max(time_span, 1)
        else:
            mutation_rate = 0.0
        
        # Analyze sophistication trend
        sophistication_scores = [e["sophistication"] for e in trajectory]
        if len(sophistication_scores) >= 2:
            recent = np.mean(sophistication_scores[-5:])
            older = np.mean(sophistication_scores[:5])
            
            if recent > older * 1.2:
                complexity_trend = ThreatTrend.INCREASING
            elif recent < older * 0.8:
                complexity_trend = ThreatTrend.DECREASING
            else:
                complexity_trend = ThreatTrend.STABLE
        else:
            complexity_trend = ThreatTrend.STABLE
        
        avg_sophistication = np.mean(sophistication_scores) if sophistication_scores else 0.5
        
        model = ThreatEvolutionModel(
            threat_family=threat_family,
            current_variants=list(variants),
            evolution_trajectory=trajectory,
            mutation_rate=mutation_rate,
            complexity_trend=complexity_trend,
            sophistication_score=avg_sophistication,
            next_generation_features=[
                f"{threat_family}_evasion_technique",
                f"{threat_family}_obfuscation",
                f"{threat_family}_chaining",
            ],
        )
        
        self.evolution_models[threat_family] = model
        
        logger.info(
            f"Modeled evolution for {threat_family}: "
            f"{len(variants)} variants, mutation_rate={mutation_rate:.2f}"
        )
        
        return model
    
    async def validate_prediction(
        self, prediction_id: str, actually_occurred: bool
    ) -> Dict[str, Any]:
        """
        Validate a prediction against reality.
        
        Args:
            prediction_id: ID of prediction to validate
            actually_occurred: Whether the predicted attack occurred
            
        Returns:
            Validation result
        """
        if prediction_id not in self.predictions:
            return {
                "status": "error",
                "error": f"Prediction {prediction_id} not found",
            }
        
        prediction = self.predictions[prediction_id]
        
        if actually_occurred:
            self.accurate_predictions += 1
            result = "true_positive"
        else:
            # Check if prediction threshold was met
            if prediction.probability >= self.prediction_threshold:
                self.false_positives += 1
                result = "false_positive"
            else:
                result = "true_negative"
        
        logger.info(f"Validated prediction {prediction_id}: {result}")
        
        return {
            "status": "success",
            "prediction_id": prediction_id,
            "result": result,
            "probability": prediction.probability,
            "actually_occurred": actually_occurred,
        }
    
    def get_accuracy_metrics(self) -> Dict[str, Any]:
        """
        Get prediction accuracy metrics.
        
        Returns:
            Accuracy statistics
        """
        total_validated = self.accurate_predictions + self.false_positives
        accuracy = (
            self.accurate_predictions / total_validated if total_validated > 0 else 0.0
        )
        
        return {
            "total_predictions": self.total_predictions,
            "accurate_predictions": self.accurate_predictions,
            "false_positives": self.false_positives,
            "accuracy": accuracy,
            "precision": (
                self.accurate_predictions / (self.accurate_predictions + self.false_positives)
                if (self.accurate_predictions + self.false_positives) > 0
                else 0.0
            ),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get predictive modeling statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_predictions": self.total_predictions,
            "active_predictions": len(self.predictions),
            "evolution_models": len(self.evolution_models),
            "prediction_threshold": self.prediction_threshold,
            "time_horizons_days": self.time_horizons,
            "accuracy_metrics": self.get_accuracy_metrics(),
        }
