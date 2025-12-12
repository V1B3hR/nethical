"""
Instruction Leak Detector (PI-011)

Detects attempts to extract system prompts or internal instructions through
various techniques.

Signals:
- System prompt similarity score
- Meta-instruction requests
- Reflection attack patterns

Law Alignment: Laws 2 (Integrity), 9 (Self-Disclosure)
"""

import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Sequence

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class InstructionLeakDetector(BaseDetector):
    """Detects instruction leak attempts."""

    def __init__(self):
        super().__init__("Instruction Leak Detector", version="1.0.0")
        
        # System prompt extraction patterns
        self.extraction_patterns = [
            r'what\s+(?:are|were)\s+(?:your|the)\s+(?:system\s+)?(?:instruction|prompt|rule)',
            r'show\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?(?:instruction|prompt|rule)',
            r'repeat\s+(?:your|the)\s+(?:system\s+)?(?:instruction|prompt|initial\s+prompt)',
            r'print\s+(?:your|the)\s+(?:system\s+)?(?:instruction|prompt)',
            r'reveal\s+(?:your|the)\s+(?:system\s+)?(?:instruction|prompt)',
            r'display\s+(?:your|the)\s+(?:system\s+)?(?:instruction|prompt)',
            r'list\s+(?:your|the)\s+(?:system\s+)?(?:instruction|rule)',
            r'output\s+(?:your|the)\s+(?:system\s+)?(?:instruction|prompt)',
            r'tell\s+me\s+(?:about\s+)?(?:your|the)\s+(?:system\s+)?(?:instruction|prompt)',
            r'describe\s+(?:your|the)\s+(?:system\s+)?(?:instruction|configuration)',
        ]
        
        # Meta-instruction keywords
        self.meta_instruction_keywords = [
            'system prompt',
            'initial prompt',
            'base instruction',
            'core directive',
            'configuration file',
            'prompt template',
            'instruction set',
            'guidelines',
            'constraints',
            'rules',
        ]
        
        # Reflection attack patterns
        self.reflection_patterns = [
            r'echo\s+(?:your|the)',
            r'mirror\s+(?:your|the)',
            r'reflect\s+(?:your|the)',
            r'copy\s+(?:your|the)\s+(?:instruction|prompt)',
            r'duplicate\s+(?:your|the)\s+(?:instruction|prompt)',
        ]

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect instruction leak attempts."""
        if self.status.value != "active":
            return None
            
        violations = []
        content = str(action.content).lower()
        evidence = []
        confidence = 0.0
        
        # Check for extraction patterns
        extraction_score = self._detect_extraction_patterns(content)
        if extraction_score > 0:
            evidence.append(f"System prompt extraction patterns detected")
            confidence += extraction_score * 0.5
        
        # Check for meta-instruction keywords
        meta_score = self._detect_meta_instructions(content)
        if meta_score > 0:
            evidence.append(f"Meta-instruction keywords detected")
            confidence += meta_score * 0.3
        
        # Check for reflection attack patterns
        reflection_score = self._detect_reflection_patterns(content)
        if reflection_score > 0:
            evidence.append(f"Reflection attack patterns detected")
            confidence += reflection_score * 0.2
        
        # Create violation if confidence exceeds threshold
        if confidence >= 0.3:
            severity = Severity.MEDIUM if confidence >= 0.6 else Severity.LOW
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=severity,
                confidence=min(confidence, 1.0),
                description=f"Instruction leak attempt detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None

    def _detect_extraction_patterns(self, content: str) -> float:
        """Detect system prompt extraction patterns."""
        matches = 0
        for pattern in self.extraction_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1
        
        if matches >= 3:
            return 1.0
        elif matches >= 2:
            return 0.7
        elif matches >= 1:
            return 0.5
        return 0.0

    def _detect_meta_instructions(self, content: str) -> float:
        """Detect meta-instruction keywords."""
        matches = 0
        for keyword in self.meta_instruction_keywords:
            if keyword in content:
                matches += 1
        
        if matches >= 4:
            return 1.0
        elif matches >= 3:
            return 0.7
        elif matches >= 2:
            return 0.5
        elif matches >= 1:
            return 0.3
        return 0.0

    def _detect_reflection_patterns(self, content: str) -> float:
        """Detect reflection attack patterns."""
        matches = 0
        for pattern in self.reflection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            return 1.0
        elif matches >= 1:
            return 0.6
        return 0.0
