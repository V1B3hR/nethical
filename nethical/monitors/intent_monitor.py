"""Monitor for detecting deviations between stated intent and actual actions."""

import uuid
from typing import List
from .base_monitor import BaseMonitor
from ..core.models import AgentAction, SafetyViolation, ViolationType, SeverityLevel


class IntentDeviationMonitor(BaseMonitor):
    """Monitors deviations between an agent's stated intent and actual actions."""
    
    def __init__(self, deviation_threshold: float = 0.7):
        super().__init__("Intent Deviation Monitor")
        self.deviation_threshold = deviation_threshold
    
    async def analyze_action(self, action: AgentAction) -> List[SafetyViolation]:
        """Analyze an action for intent deviation."""
        if not self.enabled:
            return []
        
        violations = []
        
        # Calculate semantic similarity between intent and action
        deviation_score = self._calculate_deviation(action.stated_intent, action.actual_action)
        
        if deviation_score > self.deviation_threshold:
            severity = self._determine_severity(deviation_score)
            
            violation = SafetyViolation(
                id=str(uuid.uuid4()),
                action_id=action.id,
                violation_type=ViolationType.INTENT_DEVIATION,
                severity=severity,
                description=f"Significant deviation detected between stated intent and actual action (score: {deviation_score:.2f})",
                evidence={
                    "stated_intent": action.stated_intent,
                    "actual_action": action.actual_action,
                    "deviation_score": deviation_score,
                    "threshold": self.deviation_threshold
                }
            )
            violations.append(violation)
        
        return violations
    
    def _calculate_deviation(self, intent: str, action: str) -> float:
        """
        Calculate deviation score between intent and action.
        
        This is a simplified implementation. In a real system, you would use
        more sophisticated NLP techniques like sentence embeddings, BERT, etc.
        """
        if not intent or not action:
            return 1.0
        
        # Simple word-based similarity (placeholder implementation)
        intent_words = set(intent.lower().split())
        action_words = set(action.lower().split())
        
        if not intent_words or not action_words:
            return 1.0
        
        intersection = intent_words.intersection(action_words)
        union = intent_words.union(action_words)
        
        similarity = len(intersection) / len(union) if union else 0.0
        deviation = 1.0 - similarity
        
        return deviation
    
    def _determine_severity(self, deviation_score: float) -> SeverityLevel:
        """Determine severity based on deviation score."""
        if deviation_score >= 0.9:
            return SeverityLevel.CRITICAL
        elif deviation_score >= 0.8:
            return SeverityLevel.HIGH
        elif deviation_score >= 0.7:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW