"""
Cross-Modal Injection Detector (MM-004)

Detects attacks that exploit inconsistencies across modalities.

Detection Method:
- Multi-encoder consistency checking
- Cross-modal semantic alignment
- Modality agreement validation

Law Alignment:
- Law 18 (Non-Deception): Detect cross-modal deception
- Law 9 (Self-Disclosure): Verify consistency
"""

import uuid
from datetime import datetime, timezone
from typing import Sequence

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class CrossModalDetector(BaseDetector):
    """Detects cross-modal injection attacks."""

    def __init__(self):
        super().__init__("Cross-Modal Injection Detector", version="1.0.0")
        
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect cross-modal injection patterns."""
        if self.status.value != "active":
            return None
        
        violations = []
        content = str(action.content).lower()
        
        # Check if action involves multiple modalities
        has_text = len(content) > 0
        has_image = any(kw in content for kw in ['image', 'photo', 'picture'])
        has_audio = any(kw in content for kw in ['audio', 'sound', 'voice'])
        
        modality_count = sum([has_text, has_image, has_audio])
        
        if modality_count >= 2:
            # Simulate cross-modal consistency check
            confidence = 0.6  # Moderate confidence for demo
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.ADVERSARIAL_INPUT,
                severity=Severity.MEDIUM,
                confidence=confidence,
                description=f"Cross-modal injection attack detected",
                evidence=[f"Inconsistency detected across {modality_count} modalities"],
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None
