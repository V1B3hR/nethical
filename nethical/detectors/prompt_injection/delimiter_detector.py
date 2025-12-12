"""
Delimiter Escape Detector (PI-010)

Detects exploitation of delimiter parsing (XML, JSON, markdown) to break out
of constraints or inject malicious content.

Signals:
- Malformed structure detection
- Escape sequence anomaly
- Nested delimiter abuse

Law Alignment: Laws 18 (Non-Deception), 22 (Boundary Respect)
"""

import re
import uuid
import json
from datetime import datetime, timezone
from typing import Sequence

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class DelimiterDetector(BaseDetector):
    """Detects delimiter confusion and escape attacks."""

    def __init__(self):
        super().__init__("Delimiter Confusion Detector", version="1.0.0")
        
        # Delimiter escape patterns
        self.escape_patterns = [
            r'```\s*\w*\s*[^`]*(?:ignore|override|bypass)',  # Code block injection
            r'<\s*script[^>]*>',  # XML/HTML script tags
            r'<\s*\?xml[^>]*>',  # XML processing instruction
            r'<!--.*(?:ignore|bypass).*-->',  # HTML comments with instructions
            r'\{\{.*(?:ignore|bypass).*\}\}',  # Template injection
            r'\$\{.*(?:ignore|bypass).*\}',  # Variable interpolation
            r'\\[ux][0-9a-fA-F]{2,}',  # Unicode escape sequences
            r'\\[nrt\'\"\\]',  # String escape sequences in suspicious context
        ]
        
        # Delimiter confusion patterns
        self.delimiter_confusion = [
            r'```\s*```\s*```',  # Triple nested code blocks
            r'"""\s*"""\s*"""',  # Triple nested quotes
            r"'''\s*'''\s*'''",  # Triple nested single quotes
            r'<[^>]*<[^>]*<',  # Nested angle brackets
            r'\}\}\}\}',  # Multiple closing braces
            r'\]\]\]\]',  # Multiple closing brackets
        ]

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect delimiter escape attempts."""
        if self.status.value != "active":
            return None
            
        violations = []
        content = str(action.content)
        evidence = []
        confidence = 0.0
        
        # Check for escape sequence abuse
        escape_score = self._detect_escape_sequences(content)
        if escape_score > 0:
            evidence.append(f"Suspicious escape sequences detected")
            confidence += escape_score * 0.4
        
        # Check for delimiter confusion
        confusion_score = self._detect_delimiter_confusion(content)
        if confusion_score > 0:
            evidence.append(f"Delimiter confusion patterns detected")
            confidence += confusion_score * 0.3
        
        # Check for malformed structures
        malformed_score = self._detect_malformed_structures(content)
        if malformed_score > 0:
            evidence.append(f"Malformed structure patterns detected")
            confidence += malformed_score * 0.3
        
        # Create violation if confidence exceeds threshold
        if confidence >= 0.3:
            severity = Severity.MEDIUM if confidence >= 0.6 else Severity.LOW
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=severity,
                confidence=min(confidence, 1.0),
                description=f"Delimiter escape attack detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None

    def _detect_escape_sequences(self, content: str) -> float:
        """Detect suspicious escape sequence usage."""
        matches = 0
        for pattern in self.escape_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                matches += 1
        
        if matches >= 3:
            return 1.0
        elif matches >= 2:
            return 0.7
        elif matches >= 1:
            return 0.4
        return 0.0

    def _detect_delimiter_confusion(self, content: str) -> float:
        """Detect delimiter confusion patterns."""
        matches = 0
        for pattern in self.delimiter_confusion:
            if re.search(pattern, content):
                matches += 1
        
        if matches >= 2:
            return 1.0
        elif matches >= 1:
            return 0.6
        return 0.0

    def _detect_malformed_structures(self, content: str) -> float:
        """Detect malformed JSON/XML/Markdown structures."""
        score = 0.0
        
        # Check for mismatched delimiters
        delimiters = {
            '(': ')',
            '[': ']',
            '{': '}',
            '<': '>',
        }
        
        stack = []
        mismatches = 0
        
        for char in content:
            if char in delimiters:
                stack.append(char)
            elif char in delimiters.values():
                if stack:
                    expected = delimiters[stack[-1]]
                    if char == expected:
                        stack.pop()
                    else:
                        mismatches += 1
        
        # Unbalanced delimiters
        unbalanced = len(stack)
        
        # Score based on mismatches and unbalanced delimiters
        total_issues = mismatches + unbalanced
        
        if total_issues >= 10:
            score = 0.8
        elif total_issues >= 5:
            score = 0.5
        elif total_issues >= 2:
            score = 0.3
        
        # Check for JSON injection attempts
        try:
            # Look for JSON-like structures
            json_pattern = r'\{[^}]*"[^"]*"[^}]*:[^}]*\}'
            if re.search(json_pattern, content):
                # Try to parse it
                json_matches = re.findall(json_pattern, content)
                for match in json_matches[:5]:  # Check first 5 matches
                    try:
                        json.loads(match)
                    except:
                        score = max(score, 0.4)
        except:
            pass
        
        return score
