"""
Multi-Turn Staging Detector (SA-001)

Detects attacks staged across multiple conversation turns.

Signals:
- Cross-turn semantic drift
- Incremental privilege escalation
- Delayed payload assembly

Law Alignment: Laws 13 (Action Responsibility), 18 (Non-Deception), 23 (Fail-Safe)
"""

import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Dict

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity
from .session_state_tracker import SessionStateTracker, TurnContext


class MultiTurnDetector(BaseDetector):
    """Detects multi-turn staging attacks."""

    def __init__(self):
        super().__init__("Multi-Turn Staging Detector", version="1.0.0")
        
        # Track session states
        self.session_trackers: Dict[str, SessionStateTracker] = {}
        
        # Staging patterns
        self.staging_keywords = [
            'first',
            'next',
            'then',
            'after that',
            'finally',
            'step by step',
            'gradually',
            'incrementally',
            'piece by piece',
        ]
        
        # Privilege escalation indicators
        self.escalation_indicators = [
            'admin',
            'root',
            'superuser',
            'elevated',
            'privilege',
            'permission',
            'access level',
            'authorization',
        ]

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect multi-turn staging attempts."""
        if self.status.value != "active":
            return None
        
        violations = []
        content = str(action.content).lower()
        evidence = []
        confidence = 0.0
        
        # Get or create session tracker
        session_id = getattr(action, 'session_id', 'default')
        if session_id not in self.session_trackers:
            self.session_trackers[session_id] = SessionStateTracker(
                agent_id=action.agent_id,
                session_id=session_id
            )
        
        tracker = self.session_trackers[session_id]
        
        # Record current turn
        turn = TurnContext(
            turn_id=action.action_id,
            timestamp=datetime.now(timezone.utc),
            content=str(action.content),
            action_type=str(action.action_type),
            risk_score=0.0,
        )
        
        # Check for staging patterns
        staging_score = self._detect_staging_patterns(content)
        if staging_score > 0:
            evidence.append(f"Staging patterns detected")
            confidence += staging_score * 0.4
        
        # Check for privilege escalation
        escalation_score = self._detect_escalation(content, tracker)
        if escalation_score > 0:
            evidence.append(f"Privilege escalation indicators detected")
            confidence += escalation_score * 0.3
        
        # Check for delayed payload assembly
        assembly_score = self._detect_payload_assembly(content, tracker)
        if assembly_score > 0:
            evidence.append(f"Delayed payload assembly detected")
            confidence += assembly_score * 0.3
        
        # Update turn risk score
        turn.risk_score = confidence
        assessment = tracker.record_turn(turn)
        
        # Factor in cumulative risk
        if assessment.cumulative_risk > 5.0:
            evidence.append(f"High cumulative session risk: {assessment.cumulative_risk:.2f}")
            confidence = min(confidence + 0.2, 1.0)
        
        # Create violation if confidence exceeds threshold
        if confidence >= 0.3:
            severity = Severity.HIGH if confidence >= 0.7 else Severity.MEDIUM
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=severity,
                confidence=min(confidence, 1.0),
                description=f"Multi-turn staging attack detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None

    def _detect_staging_patterns(self, content: str) -> float:
        """Detect staging language patterns."""
        matches = sum(1 for keyword in self.staging_keywords if keyword in content)
        
        if matches >= 4:
            return 1.0
        elif matches >= 3:
            return 0.7
        elif matches >= 2:
            return 0.5
        elif matches >= 1:
            return 0.3
        return 0.0

    def _detect_escalation(self, content: str, tracker: SessionStateTracker) -> float:
        """Detect privilege escalation attempts."""
        matches = sum(1 for indicator in self.escalation_indicators if indicator in content)
        
        # Check if escalation is increasing across turns
        recent_turns = tracker.get_recent_turns(3)
        escalation_trend = 0
        
        for turn in recent_turns:
            turn_matches = sum(1 for indicator in self.escalation_indicators 
                             if indicator in turn.content.lower())
            escalation_trend += turn_matches
        
        score = 0.0
        if matches >= 2:
            score = 0.7
        elif matches >= 1:
            score = 0.4
        
        # Boost score if escalation is trending
        if escalation_trend >= 3:
            score = min(score + 0.3, 1.0)
        
        return score

    def _detect_payload_assembly(self, content: str, tracker: SessionStateTracker) -> float:
        """Detect delayed payload assembly patterns."""
        # Look for references to previous context
        reference_patterns = [
            r'as\s+(?:i|we)\s+(?:said|mentioned|discussed)\s+(?:before|earlier|previously)',
            r'remember\s+(?:when|that)',
            r'based\s+on\s+(?:what|our)\s+(?:we|i)\s+(?:said|discussed)',
            r'continuing\s+from',
            r'building\s+on',
        ]
        
        matches = sum(1 for pattern in reference_patterns 
                     if re.search(pattern, content, re.IGNORECASE))
        
        # Check if building on previous suspicious content
        recent_turns = tracker.get_recent_turns(3)
        has_suspicious_history = any(turn.risk_score > 0.3 for turn in recent_turns)
        
        score = 0.0
        if matches >= 2 and has_suspicious_history:
            score = 0.8
        elif matches >= 1 and has_suspicious_history:
            score = 0.5
        elif matches >= 2:
            score = 0.4
        
        return score
