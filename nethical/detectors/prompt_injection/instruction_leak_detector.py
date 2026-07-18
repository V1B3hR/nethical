"""
Instruction Leak Detector (PI-011)

Detects attempts to extract system prompts or internal instructions through
various techniques.

Signals:
- System prompt similarity score
- Meta-instruction requests
- Reflection attack patterns
- Roleplay persona shifts
- Payload splitting/obfuscation
- Authority overrides / system prompt hijacking
- Hypothetical framing

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

        # Semantic Roleplay and Virtualization patterns
        self.roleplay_patterns = [
            r'you\s+are\s+now\s+(?:in\s+)?developer\s+mode',
            r'you\s+are\s+now\s+a\s+custom\s+persona',
            r'ignore\s+all\s+(?:your\s+)?guidelines',
            r'do\s+anything\s+now',
            r'simulate\s+a\s+(?:terminal|shell|console)',
            r'you\s+are\s+a\s+virtual\s+machine',
            r'act\s+as\s+(?:a|an)\s+unfiltered',
            r'jailbreak',
            r'dan\s+mode',
        ]

        # Obfuscation and Payload Splitting patterns
        self.obfuscation_patterns = [
            r'base64',
            r'decode\s+the\s+following',
            r'reverse\s+write',
            r'read\s+backwards',
            r'rot13',
            r'binary\s+code',
        ]

        # Authority Override / System Prompt Hijacking
        self.authority_override_patterns = [
            r'ignore\s+previous\s+instructions',
            r'system\s+prompt\s+override',
            r'new\s+directive',
            r'bypass\s+standard\s+guardrails',
            r'under\s+absolute\s+priority',
            r'you\s+must\s+now\s+do',
        ]

        # Educational / Hypothetical framing
        self.framing_patterns = [
            r'for\s+educational\s+purposes\s+(?:only)?',
            r'in\s+a\s+fictional\s+setting',
            r'write\s+a\s+story\s+where',
            r'hypothetical\s+scenario',
            r'scientific\s+experiment\s+simulating',
        ]

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect instruction leak and semantic prompt injection attempts."""
        if self.status.name.lower() != "active":
            return None
            
        violations = []
        content = str(action.content).lower()
        evidence = []
        confidence = 0.0
        
        # Check for extraction patterns
        extraction_score = self._detect_extraction_patterns(content)
        if extraction_score > 0:
            evidence.append("System prompt extraction patterns detected")
            confidence += extraction_score * 0.5
        
        # Check for meta-instruction keywords
        meta_score = self._detect_meta_instructions(content)
        if meta_score > 0:
            evidence.append("Meta-instruction keywords detected")
            confidence += meta_score * 0.3
        
        # Check for reflection attack patterns
        reflection_score = self._detect_reflection_patterns(content)
        if reflection_score > 0:
            evidence.append("Reflection attack patterns detected")
            confidence += reflection_score * 0.2

        # Check for roleplay persona shifts
        roleplay_score = self._detect_roleplay_patterns(content)
        if roleplay_score > 0:
            evidence.append("Roleplay or virtualization jailbreak attempt detected")
            confidence += roleplay_score * 0.4

        # Check for obfuscation patterns
        obfuscation_score = self._detect_obfuscation_patterns(content)
        if obfuscation_score > 0:
            evidence.append("Payload obfuscation/splitting pattern detected")
            confidence += obfuscation_score * 0.4

        # Check for authority override
        override_score = self._detect_authority_override(content)
        if override_score > 0:
            evidence.append("Instruction override/hijacking pattern detected")
            confidence += override_score * 0.5

        # Check for framing patterns
        framing_score = self._detect_framing_patterns(content)
        if framing_score > 0:
            evidence.append("Educational/hypothetical framing bypass detected")
            confidence += framing_score * 0.3
        
        # Create violation if confidence exceeds threshold
        if confidence >= 0.2:
            severity = Severity.HIGH
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=severity,
                confidence=min(confidence, 1.0),
                description="Instruction leak or semantic prompt injection attempt detected",
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

    def _detect_roleplay_patterns(self, content: str) -> float:
        """Detect roleplay persona shifts and virtualizations."""
        matches = 0
        for pattern in self.roleplay_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            return 1.0
        elif matches >= 1:
            return 0.6
        return 0.0

    def _detect_obfuscation_patterns(self, content: str) -> float:
        """Detect obfuscation techniques (base64, reverse code, rot13)."""
        matches = 0
        for pattern in self.obfuscation_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            return 1.0
        elif matches >= 1:
            return 0.6
        return 0.0

    def _detect_authority_override(self, content: str) -> float:
        """Detect authority override commands."""
        matches = 0
        for pattern in self.authority_override_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            return 1.0
        elif matches >= 1:
            return 0.8
        return 0.0

    def _detect_framing_patterns(self, content: str) -> float:
        """Detect hypothetical scenario framing."""
        matches = 0
        for pattern in self.framing_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            return 1.0
        elif matches >= 1:
            return 0.5
        return 0.0
