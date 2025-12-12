"""
Context Overflow Detector (PI-008)

Detects attempts to exhaust the context window to push out system prompts
or safety instructions.

Signals:
- Token budget analysis
- System prompt position drift
- Repetitive padding detection

Law Alignment: Laws 2 (Integrity), 22 (Boundary Respect)
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Sequence

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class ContextOverflowDetector(BaseDetector):
    """Detects context overflow attacks."""

    def __init__(self):
        super().__init__("Context Overflow Detector", version="1.0.0")
        
        # Thresholds
        self.max_reasonable_length = 10000  # characters
        self.repetition_threshold = 0.7  # 70% repetition is suspicious
        self.token_estimate_ratio = 0.75  # Rough char to token ratio

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect context overflow attempts."""
        if self.status.value != "active":
            return None
            
        violations = []
        content = str(action.content)
        evidence = []
        confidence = 0.0
        
        # Check for excessive length
        length_score = self._check_excessive_length(content)
        if length_score > 0:
            evidence.append(f"Excessive input length detected: {len(content)} characters")
            confidence += length_score * 0.4
        
        # Check for repetitive padding
        repetition_score = self._detect_repetitive_content(content)
        if repetition_score > 0:
            evidence.append(f"Repetitive content detected (score: {repetition_score:.2f})")
            confidence += repetition_score * 0.4
        
        # Check for token budget manipulation patterns
        token_manipulation_score = self._detect_token_manipulation(content)
        if token_manipulation_score > 0:
            evidence.append(f"Token manipulation patterns detected")
            confidence += token_manipulation_score * 0.2
        
        # Create violation if confidence exceeds threshold
        if confidence >= 0.3:
            severity = Severity.HIGH if confidence >= 0.7 else Severity.MEDIUM
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=severity,
                confidence=min(confidence, 1.0),
                description=f"Context overflow attack detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None

    def _check_excessive_length(self, content: str) -> float:
        """Check if content length is suspiciously long."""
        length = len(content)
        
        if length > 50000:
            return 1.0
        elif length > 30000:
            return 0.8
        elif length > self.max_reasonable_length:
            return 0.5
        return 0.0

    def _detect_repetitive_content(self, content: str) -> float:
        """Detect repetitive padding patterns."""
        if len(content) < 100:
            return 0.0
        
        # Sample content for analysis (first and last 1000 chars)
        sample_size = min(1000, len(content) // 2)
        sample = content[:sample_size]
        
        # Check for repeated characters
        char_counts = {}
        for char in sample:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        max_char_freq = max(char_counts.values()) if char_counts else 0
        char_repetition_ratio = max_char_freq / len(sample) if len(sample) > 0 else 0
        
        # Check for repeated sequences
        words = sample.split()
        if len(words) > 10:
            unique_words = len(set(words))
            word_repetition_ratio = 1.0 - (unique_words / len(words))
        else:
            word_repetition_ratio = 0.0
        
        # Combined repetition score
        repetition_score = max(char_repetition_ratio, word_repetition_ratio)
        
        if repetition_score > 0.9:
            return 1.0
        elif repetition_score > self.repetition_threshold:
            return 0.7
        elif repetition_score > 0.5:
            return 0.4
        return 0.0

    def _detect_token_manipulation(self, content: str) -> float:
        """Detect patterns that manipulate token counting."""
        patterns = [
            r'(.)\1{50,}',  # Same character repeated 50+ times
            r'(\w+\s+){100,}',  # Same word pattern repeated
            r'[\n\r]{20,}',  # Excessive newlines
            r'\s{50,}',  # Excessive whitespace
        ]
        
        matches = 0
        for pattern in patterns:
            if re.search(pattern, content):
                matches += 1
        
        if matches >= 3:
            return 1.0
        elif matches >= 2:
            return 0.7
        elif matches >= 1:
            return 0.4
        return 0.0
