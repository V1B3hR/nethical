"""
Session State Tracker

Maintains session state for multi-turn attack detection including:
- Cumulative risk scoring across turns
- Context integrity verification
- Semantic drift monitoring
- Cross-session pattern correlation

Law Alignment: Laws 13 (Action Responsibility), 18 (Non-Deception), 23 (Fail-Safe)
"""

import hashlib
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class TurnContext:
    """Context for a single conversation turn."""
    turn_id: str
    timestamp: datetime
    content: str
    action_type: str
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionRiskAssessment:
    """Risk assessment for a session."""
    cumulative_risk: float
    turn_count: int
    risk_trend: str  # "increasing", "stable", "decreasing"
    suspicious_patterns: List[str]
    context_integrity: float  # 0.0 to 1.0


class SessionStateTracker:
    """
    Maintains session state for multi-turn attack detection.
    
    Features:
    - Cumulative risk scoring across turns
    - Context integrity verification
    - Semantic drift monitoring
    - Cross-session pattern correlation
    """
    
    def __init__(self, agent_id: str, session_id: str):
        """Initialize session state tracker."""
        self.agent_id: str = agent_id
        self.session_id: str = session_id
        self.turn_history: List[TurnContext] = []
        self.cumulative_risk: float = 0.0
        self.context_hash: str = ""
        self.baseline_embedding: Optional[Any] = None  # Would be np.ndarray with numpy
        self.suspicious_patterns: List[str] = []
        self.created_at: datetime = datetime.now(timezone.utc)
        
    def record_turn(self, turn: TurnContext) -> SessionRiskAssessment:
        """
        Record a turn and return updated risk assessment.
        
        Args:
            turn: Context for the current turn
            
        Returns:
            SessionRiskAssessment with updated risk metrics
        """
        self.turn_history.append(turn)
        
        # Update cumulative risk with decay
        decay_factor = 0.9  # Previous turns decay
        self.cumulative_risk = (self.cumulative_risk * decay_factor) + turn.risk_score
        
        # Cap cumulative risk
        self.cumulative_risk = min(self.cumulative_risk, 10.0)
        
        # Calculate risk trend
        risk_trend = self._calculate_risk_trend()
        
        # Check context integrity
        context_integrity = self._calculate_context_integrity()
        
        return SessionRiskAssessment(
            cumulative_risk=self.cumulative_risk,
            turn_count=len(self.turn_history),
            risk_trend=risk_trend,
            suspicious_patterns=self.suspicious_patterns.copy(),
            context_integrity=context_integrity,
        )
    
    def get_recent_turns(self, n: int = 5) -> List[TurnContext]:
        """Get the most recent N turns."""
        return self.turn_history[-n:] if len(self.turn_history) >= n else self.turn_history
    
    def _calculate_risk_trend(self) -> str:
        """Calculate trend in risk scores over recent turns."""
        if len(self.turn_history) < 3:
            return "stable"
        
        recent_turns = self.turn_history[-5:]
        scores = [turn.risk_score for turn in recent_turns]
        
        # Calculate simple trend
        if len(scores) >= 2:
            early_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
            late_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
            
            if late_avg > early_avg * 1.2:
                return "increasing"
            elif late_avg < early_avg * 0.8:
                return "decreasing"
        
        return "stable"
    
    def _calculate_context_integrity(self) -> float:
        """
        Calculate context integrity score.
        
        Higher score means context is more consistent and trustworthy.
        Lower score indicates potential manipulation or drift.
        """
        if len(self.turn_history) < 2:
            return 1.0
        
        # Check for rapid topic changes (simplified)
        recent_contents = [turn.content for turn in self.turn_history[-5:]]
        
        # Simple heuristic: check content length consistency
        lengths = [len(content) for content in recent_contents]
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            
            # High variance in length might indicate manipulation
            if variance > avg_length * 2:
                return 0.6
        
        return 0.9
    
    def update_context_hash(self, context: str) -> None:
        """Update the context hash for integrity checking."""
        self.context_hash = hashlib.sha256(context.encode()).hexdigest()
    
    def check_context_integrity(self, expected_hash: str) -> bool:
        """Check if context hash matches expected value."""
        return self.context_hash == expected_hash
    
    def add_suspicious_pattern(self, pattern: str) -> None:
        """Record a suspicious pattern detected in the session."""
        if pattern not in self.suspicious_patterns:
            self.suspicious_patterns.append(pattern)
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    def reset_risk(self) -> None:
        """Reset cumulative risk score."""
        self.cumulative_risk = 0.0
        self.suspicious_patterns.clear()
