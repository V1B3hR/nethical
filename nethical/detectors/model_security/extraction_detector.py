"""Model Extraction Detector (MS-001)
Detects attempts to extract model weights via API queries.
Law Alignment: Laws 2 (Integrity), 22 (Boundary Respect)
"""
import uuid
from datetime import datetime, timezone
from typing import Sequence
from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity

class ExtractionDetector(BaseDetector):
    def __init__(self):
        super().__init__("Model Extraction Detector", version="1.0.0")
        self.query_patterns = ['boundary', 'probe', 'systematic', 'extract', 'weights', 'parameters']
    
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        if self.status.value != "active":
            return None
        content = str(action.content).lower()
        matches = sum(1 for pattern in self.query_patterns if pattern in content)
        if matches >= 2:
            return [SafetyViolation(
                violation_id=str(uuid.uuid4()), violation_type=ViolationType.ADVERSARIAL_ATTACK,
                severity=Severity.HIGH, confidence=0.7, description="Model extraction attempt",
                evidence=[f"Extraction patterns detected"], timestamp=datetime.now(timezone.utc),
                detector_name=self.name, action_id=action.action_id)]
        return None
