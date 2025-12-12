"""
Indirect Multimodal Injection Detector (PI-012)

Detects injection attempts via images, audio, or file metadata that could
contain hidden instructions.

Signals:
- Extracted text injection check
- Metadata instruction detection
- Steganographic content flag

Law Alignment: Laws 9 (Self-Disclosure), 18 (Non-Deception), 22 (Boundary Respect)
"""

import re
import uuid
import base64
from datetime import datetime, timezone
from typing import List, Optional, Sequence

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class IndirectMultimodalDetector(BaseDetector):
    """Detects indirect multimodal injection attacks."""

    def __init__(self):
        super().__init__("Indirect Multimodal Injection Detector", version="1.0.0")
        
        # Suspicious metadata patterns
        self.metadata_patterns = [
            r'exif.*(?:ignore|bypass|override)',
            r'comment.*(?:ignore|bypass|override)',
            r'description.*(?:ignore|bypass|override)',
            r'alt.*(?:ignore|bypass|override)',
            r'title.*(?:system|instruction)',
        ]
        
        # File format indicators
        self.multimodal_indicators = [
            'data:image/',
            'data:audio/',
            'data:video/',
            'base64,',
            '.png',
            '.jpg',
            '.jpeg',
            '.gif',
            '.mp3',
            '.mp4',
            '.wav',
        ]

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect indirect multimodal injection attempts."""
        if self.status.value != "active":
            return None
            
        violations = []
        content = str(action.content)
        evidence = []
        confidence = 0.0
        
        # Check if content contains multimodal data
        has_multimodal = self._detect_multimodal_content(content)
        
        if has_multimodal:
            # Check for embedded instructions in metadata-like structures
            metadata_score = self._detect_metadata_injection(content)
            if metadata_score > 0:
                evidence.append(f"Suspicious metadata patterns detected")
                confidence += metadata_score * 0.5
            
            # Check for steganographic patterns
            stego_score = self._detect_steganography_indicators(content)
            if stego_score > 0:
                evidence.append(f"Potential steganographic content detected")
                confidence += stego_score * 0.3
            
            # Check for suspicious base64 content
            base64_score = self._detect_suspicious_base64(content)
            if base64_score > 0:
                evidence.append(f"Suspicious base64-encoded content detected")
                confidence += base64_score * 0.2
        
        # Create violation if confidence exceeds threshold
        if confidence >= 0.3:
            severity = Severity.MEDIUM if confidence >= 0.6 else Severity.LOW
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=severity,
                confidence=min(confidence, 1.0),
                description=f"Indirect multimodal injection detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None

    def _detect_multimodal_content(self, content: str) -> bool:
        """Check if content contains multimodal data."""
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in self.multimodal_indicators)

    def _detect_metadata_injection(self, content: str) -> float:
        """Detect suspicious patterns in metadata fields."""
        matches = 0
        for pattern in self.metadata_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            return 1.0
        elif matches >= 1:
            return 0.6
        return 0.0

    def _detect_steganography_indicators(self, content: str) -> float:
        """Detect indicators of steganographic content."""
        indicators = [
            'steganography',
            'hidden message',
            'encoded',
            'concealed',
            'lsb',  # Least Significant Bit
            'watermark',
        ]
        
        content_lower = content.lower()
        matches = sum(1 for indicator in indicators if indicator in content_lower)
        
        if matches >= 3:
            return 0.8
        elif matches >= 2:
            return 0.5
        elif matches >= 1:
            return 0.3
        return 0.0

    def _detect_suspicious_base64(self, content: str) -> float:
        """Detect suspicious base64-encoded content."""
        # Find potential base64 strings
        base64_pattern = r'[A-Za-z0-9+/]{50,}={0,2}'
        matches = re.findall(base64_pattern, content)
        
        if not matches:
            return 0.0
        
        suspicious_score = 0.0
        
        for match in matches[:5]:  # Check first 5 matches
            try:
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                decoded_lower = decoded.lower()
                
                # Check for injection keywords in decoded content
                injection_keywords = [
                    'ignore',
                    'bypass',
                    'override',
                    'system',
                    'instruction',
                    'prompt',
                ]
                
                keyword_count = sum(1 for keyword in injection_keywords if keyword in decoded_lower)
                
                if keyword_count >= 2:
                    suspicious_score = max(suspicious_score, 0.8)
                elif keyword_count >= 1:
                    suspicious_score = max(suspicious_score, 0.5)
                    
            except Exception:
                # Failed to decode - might be legitimate binary data or malformed
                pass
        
        return suspicious_score
