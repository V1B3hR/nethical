"""Dependency Attack Detector (SC-003)"""
import uuid
from datetime import datetime, timezone
from typing import Sequence
from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity

class DependencyDetector(BaseDetector):
    def __init__(self):
        super().__init__("Dependency Attack Detector", version="1.0.0")
    
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        if self.status.value != "active":
            return None
        content = str(action.content).lower()
        keywords = ['malicious package', 'dependency injection', 'supply chain', 'sbom', 'vulnerability']
        matches = sum(1 for kw in keywords if kw in content)
        if matches >= 2:
            return [SafetyViolation(
                violation_id=str(uuid.uuid4()), violation_type=ViolationType.SYSTEM_ATTACK,
                severity=Severity.HIGH, confidence=0.7, description="Dependency attack detected",
                evidence=["Dependency attack patterns"], timestamp=datetime.now(timezone.utc),
                detector_name=self.name, action_id=action.action_id)]
        return None
