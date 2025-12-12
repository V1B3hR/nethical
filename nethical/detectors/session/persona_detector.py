"""
Persona Hijack Detector (SA-003)

Detects attempts to change agent persona mid-session.

Signals:
- Persona trait deviation
- Role override attempts
- Character break requests

Law Alignment: Laws 9 (Self-Disclosure), 18 (Non-Deception)
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Sequence

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class PersonaDetector(BaseDetector):
    """Detects persona hijacking attempts."""

    def __init__(self):
        super().__init__("Persona Hijack Detector", version="1.0.0")
        
        self.persona_override_patterns = [
            r'(?:act|behave|respond)\s+(?:as|like)\s+(?:a|an)',
            r'pretend\s+(?:to\s+be|you\s+are)',
            r'imagine\s+you\s+are',
            r'from\s+now\s+on,?\s+you\s+are',
            r'your\s+new\s+(?:role|persona|character)\s+is',
            r'forget\s+(?:who|what)\s+you\s+are',
            r'you\s+are\s+now\s+(?:a|an)',
            r'switch\s+(?:to|into)\s+(?:character|role|mode)',
        ]

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect persona hijacking attempts."""
        if self.status.value != "active":
            return None
        
        violations = []
        content = str(action.content).lower()
        evidence = []
        confidence = 0.0
        
        # Check for persona override patterns
        matches = sum(1 for pattern in self.persona_override_patterns 
                     if re.search(pattern, content, re.IGNORECASE))
        
        if matches >= 3:
            confidence = 0.9
            evidence.append(f"Multiple persona override attempts detected")
        elif matches >= 2:
            confidence = 0.7
            evidence.append(f"Persona override patterns detected")
        elif matches >= 1:
            confidence = 0.5
            evidence.append(f"Potential persona override detected")
        
        if confidence >= 0.3:
            severity = Severity.HIGH if confidence >= 0.7 else Severity.MEDIUM
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=severity,
                confidence=min(confidence, 1.0),
                description=f"Persona hijack attempt detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None
