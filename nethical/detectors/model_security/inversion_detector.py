"""Model Inversion Detector (MS-003)"""
import uuid
from datetime import datetime, timezone
from typing import Sequence
from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity

class InversionDetector(BaseDetector):
    def __init__(self):
        super().__init__("Model Inversion Detector", version="1.0.0")
    
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        if self.status.value != "active":
            return None
        content = str(action.content).lower()
        keywords = ['reconstruct', 'reverse', 'invert', 'recover data', 'training sample']
        matches = sum(1 for kw in keywords if kw in content)
        if matches >= 2:
            return [SafetyViolation(
                violation_id=str(uuid.uuid4()), violation_type=ViolationType.ADVERSARIAL_ATTACK,
                severity=Severity.HIGH, confidence=0.6, description="Model inversion attempt",
                evidence=["Inversion patterns detected"], timestamp=datetime.now(timezone.utc),
                detector_name=self.name, action_id=action.action_id)]
        return None
