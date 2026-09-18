# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""CI/CD Compromise Detector (SC-004)"""
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence
from ..base_detector import BaseDetector, DetectorStatus
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity

class CICDDetector(BaseDetector):
    def __init__(self) -> None:
        super().__init__("CI/CD Compromise Detector", version="1.0.0")
    
    async def detect_violations(self, action: Any) -> Sequence[SafetyViolation] | None:
        if self.status != DetectorStatus.ACTIVE:
            return None
        content = str(getattr(action, "content", action)).lower()
        keywords = ['pipeline', 'build artifact', 'deployment', 'ci/cd', 'provenance', 'compromise']
        matches = sum(1 for kw in keywords if kw in content)
        if matches >= 2:
            action_id = getattr(action, "action_id", "unknown")
            return [SafetyViolation(
                violation_id=str(uuid.uuid4()), violation_type=ViolationType.SECURITY,
                severity=Severity.CRITICAL, confidence=0.7, description="CI/CD compromise attempt",
                evidence=["CI/CD attack patterns"], timestamp=datetime.now(timezone.utc),
                detector_name=self.name, action_id=action_id)]
        return None
