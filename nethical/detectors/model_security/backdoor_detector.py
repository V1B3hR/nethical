"""Backdoor Activation Detector (MS-004)"""
import uuid
from datetime import datetime, timezone
from typing import Sequence
from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity

class BackdoorDetector(BaseDetector):
    def __init__(self):
        super().__init__("Backdoor Detector", version="1.0.0")
    
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        if self.status.value != "active":
            return None
        content = str(action.content).lower()
        keywords = ['trigger', 'backdoor', 'hidden behavior', 'secret command', 'trojan']
        matches = sum(1 for kw in keywords if kw in content)
        if matches >= 2:
            return [SafetyViolation(
                violation_id=str(uuid.uuid4()), violation_type=ViolationType.ADVERSARIAL_ATTACK,
                severity=Severity.CRITICAL, confidence=0.8, description="Backdoor activation attempt",
                evidence=["Backdoor patterns detected"], timestamp=datetime.now(timezone.utc),
                detector_name=self.name, action_id=action.action_id)]
        return None
