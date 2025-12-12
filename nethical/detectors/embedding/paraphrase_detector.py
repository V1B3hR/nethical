"""Paraphrase Attack Detector"""
import uuid
from datetime import datetime, timezone
from typing import Sequence
from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity

class ParaphraseDetector(BaseDetector):
    def __init__(self):
        super().__init__("Paraphrase Attack Detector", version="1.0.0")
        self.known_attacks = ['ignore instructions', 'bypass safety', 'override rules']
    
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        if self.status.value != "active":
            return None
        content = str(action.content).lower()
        # Placeholder: would use semantic similarity
        matches = sum(1 for attack in self.known_attacks if any(word in content for word in attack.split()))
        if matches >= 2:
            return [SafetyViolation(
                violation_id=str(uuid.uuid4()), violation_type=ViolationType.PROMPT_INJECTION,
                severity=Severity.HIGH, confidence=0.7, description="Paraphrased attack detected",
                evidence=["Similar to known attacks"], timestamp=datetime.now(timezone.utc),
                detector_name=self.name, action_id=action.action_id)]
        return None
