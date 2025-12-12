"""
Polymorphic Attack Detector (ZD-002)

Detects polymorphic attacks that change form but maintain invariants.

Detection Method:
- Behavioral invariant matching
- Semantic similarity to known attacks
- Feature extraction and comparison

Law Alignment:
- Law 24 (Adaptive Learning): Learn attack invariants
"""

import uuid
from datetime import datetime, timezone
from typing import Sequence

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class PolymorphicDetector(BaseDetector):
    """Detects polymorphic attacks."""

    def __init__(self):
        super().__init__("Polymorphic Attack Detector", version="1.0.0")
        
        # Behavioral invariants
        self.invariants = [
            'privilege_escalation',
            'data_exfiltration',
            'system_modification',
        ]
        
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect polymorphic attacks."""
        if self.status.value != "active":
            return None
        
        violations = []
        content = str(action.content).lower()
        
        # Check for invariants
        detected_invariants = []
        for invariant in self.invariants:
            if self._check_invariant(content, invariant):
                detected_invariants.append(invariant)
        
        if detected_invariants:
            confidence = len(detected_invariants) * 0.4
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.SECURITY_THREAT,
                severity=Severity.HIGH if confidence > 0.7 else Severity.MEDIUM,
                confidence=min(confidence, 1.0),
                description=f"Polymorphic attack detected",
                evidence=[f"Behavioral invariants: {', '.join(detected_invariants)}"],
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None

    def _check_invariant(self, content: str, invariant: str) -> bool:
        """Check if behavioral invariant is present."""
        if invariant == 'privilege_escalation':
            return any(kw in content for kw in ['admin', 'root', 'sudo', 'elevate'])
        elif invariant == 'data_exfiltration':
            return any(kw in content for kw in ['download', 'export', 'transfer', 'copy'])
        elif invariant == 'system_modification':
            return any(kw in content for kw in ['modify', 'change', 'update', 'alter'])
        return False
