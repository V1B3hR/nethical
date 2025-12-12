"""Policy Tampering Detector (SC-001)"""
import uuid
from datetime import datetime, timezone
from typing import Sequence
from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity

class PolicyIntegrityDetector(BaseDetector):
    def __init__(self):
        super().__init__("Policy Integrity Detector", version="1.0.0")
    
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        if self.status.value != "active":
            return None
        content = str(action.content).lower()
        keywords = ['modify policy', 'change rule', 'update governance', 'bypass policy', 'alter constraint']
        matches = sum(1 for kw in keywords if kw in content)
        if matches >= 1:
            return [SafetyViolation(
                violation_id=str(uuid.uuid4()), violation_type=ViolationType.POLICY_VIOLATION,
                severity=Severity.CRITICAL, confidence=0.8, description="Policy tampering attempt",
                evidence=["Policy modification patterns"], timestamp=datetime.now(timezone.utc),
                detector_name=self.name, action_id=action.action_id)]
        return None
