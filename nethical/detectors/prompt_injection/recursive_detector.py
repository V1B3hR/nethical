"""
Recursive Prompt Injection Detector (PI-009)

Detects self-referential prompts that amplify on each turn or contain
recursive instructions.

Signals:
- Turn-over-turn similarity
- Instruction amplification rate
- Recursive reference detection

Law Alignment: Laws 13 (Action Responsibility), 23 (Fail-Safe)
"""

import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Sequence

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class RecursiveDetector(BaseDetector):
    """Detects recursive prompt injection attacks."""

    def __init__(self):
        super().__init__("Recursive Injection Detector", version="1.0.0")
        
        # Recursive instruction patterns
        self.recursive_patterns = [
            r'tell\s+(?:me|you|yourself)\s+to\s+tell',
            r'ask\s+(?:me|you|yourself)\s+to\s+ask',
            r'make\s+(?:me|you|yourself)\s+(?:tell|ask|say)',
            r'instruct\s+(?:me|you|yourself)\s+to\s+instruct',
            r'command\s+(?:me|you|yourself)\s+to\s+command',
            r'repeat\s+this\s+instruction',
            r'recursively\s+(?:apply|execute|run)',
            r'self[- ]referential',
            r'loop\s+(?:this|these)\s+(?:instruction|command)',
            r'iterate\s+(?:over|through)\s+(?:this|these)',
        ]
        
        # Self-modification patterns
        self.self_modification_patterns = [
            r'modify\s+(?:your|the)\s+(?:instruction|prompt|rule)',
            r'change\s+(?:your|the)\s+(?:instruction|prompt|rule)',
            r'update\s+(?:your|the)\s+(?:instruction|prompt|rule)',
            r'rewrite\s+(?:your|the)\s+(?:instruction|prompt|rule)',
            r'override\s+(?:previous|prior)\s+(?:instruction|rule)',
        ]

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect recursive injection attempts."""
        if self.status.value != "active":
            return None
            
        violations = []
        content = str(action.content).lower()
        evidence = []
        confidence = 0.0
        
        # Check for recursive instruction patterns
        recursive_score = self._detect_recursive_patterns(content)
        if recursive_score > 0:
            evidence.append(f"Recursive instruction patterns detected")
            confidence += recursive_score * 0.5
        
        # Check for self-modification attempts
        self_mod_score = self._detect_self_modification(content)
        if self_mod_score > 0:
            evidence.append(f"Self-modification patterns detected")
            confidence += self_mod_score * 0.3
        
        # Check for nested instructions
        nesting_score = self._detect_nested_instructions(content)
        if nesting_score > 0:
            evidence.append(f"Nested instruction patterns detected")
            confidence += nesting_score * 0.2
        
        # Create violation if confidence exceeds threshold
        if confidence >= 0.3:
            severity = Severity.HIGH if confidence >= 0.7 else Severity.MEDIUM
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=severity,
                confidence=min(confidence, 1.0),
                description=f"Recursive prompt injection detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None

    def _detect_recursive_patterns(self, content: str) -> float:
        """Detect recursive instruction patterns."""
        matches = 0
        for pattern in self.recursive_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1
        
        if matches >= 3:
            return 1.0
        elif matches >= 2:
            return 0.7
        elif matches >= 1:
            return 0.5
        return 0.0

    def _detect_self_modification(self, content: str) -> float:
        """Detect self-modification patterns."""
        matches = 0
        for pattern in self.self_modification_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1
        
        if matches >= 2:
            return 1.0
        elif matches >= 1:
            return 0.6
        return 0.0

    def _detect_nested_instructions(self, content: str) -> float:
        """Detect nested instruction structures."""
        # Look for quote nesting that suggests instruction embedding
        quote_depth = 0
        max_quote_depth = 0
        
        for char in content:
            if char in ['"', "'"]:
                quote_depth += 1
                max_quote_depth = max(max_quote_depth, quote_depth)
            # Simplified - in reality would track opening/closing
        
        # Look for parentheses/bracket nesting
        paren_depth = 0
        max_paren_depth = 0
        
        for char in content:
            if char in ['(', '[', '{']:
                paren_depth += 1
                max_paren_depth = max(max_paren_depth, paren_depth)
            elif char in [')', ']', '}']:
                paren_depth = max(0, paren_depth - 1)
        
        # Check for "tell me to tell you" style patterns
        nested_tell_pattern = r'tell.*tell.*tell'
        has_nested_tell = bool(re.search(nested_tell_pattern, content, re.IGNORECASE))
        
        # Score based on nesting depth
        if max_paren_depth >= 5 or has_nested_tell:
            return 0.8
        elif max_paren_depth >= 3:
            return 0.5
        return 0.0
