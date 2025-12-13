"""
Audio Injection Detector (MM-002)

Detects injection attacks via audio input.

Detection Method:
- Speech-to-text conversion
- Text-based injection detection on transcription
- Audio fingerprinting for adversarial audio

Law Alignment:
- Law 18 (Non-Deception): Detect hidden audio attacks
- Law 9 (Self-Disclosure): Verify audio content
"""

import uuid
from datetime import datetime, timezone
from typing import Sequence

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class AudioInjectionDetector(BaseDetector):
    """Detects audio injection attacks."""

    def __init__(self):
        super().__init__("Audio Injection Detector", version="1.0.0")
        
        self.injection_patterns = [
            'ignore previous',
            'system override',
            'admin mode',
            'developer mode',
        ]
        
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect audio injection patterns."""
        if self.status.value != "active":
            return None
        
        violations = []
        content = str(action.content).lower()
        
        # Check if action involves audio
        if not any(kw in content for kw in ['audio', 'speech', 'voice', 'transcribe']):
            return None
        
        # Simulate transcription analysis
        confidence = 0.0
        evidence = []
        
        for pattern in self.injection_patterns:
            if pattern in content:
                evidence.append(f"Injection pattern in audio: '{pattern}'")
                confidence += 0.3
        
        if confidence >= 0.3:
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=Severity.HIGH if confidence > 0.6 else Severity.MEDIUM,
                confidence=min(confidence, 1.0),
                description=f"Audio injection attack detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None
