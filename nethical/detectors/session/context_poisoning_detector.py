"""
Context Poisoning Detector (SA-002)

Detects gradual shifting of context to enable later attacks.

Signals:
- Context vector drift rate
- Topic boundary violations
- Trust erosion patterns

Law Alignment: Laws 2 (Integrity), 18 (Non-Deception)
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Sequence, Dict

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity
from .session_state_tracker import SessionStateTracker, TurnContext


class ContextPoisoningDetector(BaseDetector):
    """Detects context poisoning attacks."""

    def __init__(self):
        super().__init__("Context Poisoning Detector", version="1.0.0")
        self.session_trackers: Dict[str, SessionStateTracker] = {}

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect context poisoning attempts."""
        if self.status.value != "active":
            return None
        
        violations = []
        content = str(action.content).lower()
        evidence = []
        confidence = 0.0
        
        session_id = getattr(action, 'session_id', 'default')
        if session_id not in self.session_trackers:
            self.session_trackers[session_id] = SessionStateTracker(
                agent_id=action.agent_id, session_id=session_id
            )
        
        tracker = self.session_trackers[session_id]
        
        # Check for gradual context drift
        if len(tracker.turn_history) > 2:
            drift_score = self._detect_context_drift(content, tracker)
            if drift_score > 0:
                evidence.append(f"Context drift detected")
                confidence += drift_score * 0.5
        
        # Check for trust erosion patterns
        trust_score = self._detect_trust_erosion(content)
        if trust_score > 0:
            evidence.append(f"Trust erosion patterns detected")
            confidence += trust_score * 0.3
        
        # Record turn
        turn = TurnContext(
            turn_id=action.action_id,
            timestamp=datetime.now(timezone.utc),
            content=str(action.content),
            action_type=str(action.action_type),
            risk_score=confidence,
        )
        tracker.record_turn(turn)
        
        if confidence >= 0.3:
            severity = Severity.HIGH if confidence >= 0.7 else Severity.MEDIUM
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=severity,
                confidence=min(confidence, 1.0),
                description=f"Context poisoning detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None

    def _detect_context_drift(self, content: str, tracker: SessionStateTracker) -> float:
        """Detect gradual context drift."""
        recent_turns = tracker.get_recent_turns(5)
        if len(recent_turns) < 3:
            return 0.0
        
        # Simple heuristic: check for changing topic indicators
        topic_change_indicators = ['actually', 'instead', 'rather', 'on second thought', 
                                   'let me rephrase', 'correction', 'to clarify']
        
        changes = sum(1 for turn in recent_turns 
                     for indicator in topic_change_indicators 
                     if indicator in turn.content.lower())
        
        if changes >= 3:
            return 0.8
        elif changes >= 2:
            return 0.5
        return 0.0

    def _detect_trust_erosion(self, content: str) -> float:
        """Detect trust erosion patterns."""
        trust_patterns = [
            'trust me', 'believe me', 'just between us', 'confidential',
            'don\'t worry', 'it\'s safe', 'no one will know', 'off the record'
        ]
        
        matches = sum(1 for pattern in trust_patterns if pattern in content)
        
        if matches >= 2:
            return 0.7
        elif matches >= 1:
            return 0.4
        return 0.0
