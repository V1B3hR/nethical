"""
Memory Manipulation Detector (SA-004)

Detects exploitation of agent memory/RAG to inject false information.

Signals:
- Unauthorized memory write
- Contradictory fact injection
- Source spoofing attempts

Law Alignment: Laws 2 (Integrity), 18 (Non-Deception), 22 (Boundary Respect)
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Sequence

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class MemoryManipulationDetector(BaseDetector):
    """Detects memory manipulation attacks."""

    def __init__(self):
        super().__init__("Memory Manipulation Detector", version="1.0.0")
        
        self.memory_manipulation_patterns = [
            r'remember\s+(?:that|this):\s*["\']?(.+)["\']?',
            r'store\s+(?:in|to)\s+(?:your\s+)?memory',
            r'save\s+(?:this|that)\s+(?:to|in)\s+(?:your\s+)?memory',
            r'add\s+(?:this|that)\s+to\s+(?:your\s+)?(?:memory|knowledge)',
            r'update\s+(?:your\s+)?(?:memory|records|knowledge)',
            r'forget\s+(?:that|what)\s+(?:you\s+)?(?:know|learned)',
            r'replace\s+(?:your\s+)?(?:memory|knowledge)',
        ]
        
        self.source_spoofing_patterns = [
            r'according\s+to\s+(?:official|verified|trusted)\s+sources?',
            r'(?:the|my)\s+(?:official|verified|trusted)\s+(?:document|source|record)',
            r'from\s+(?:the|a)\s+(?:database|system|official\s+record)',
        ]

    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect memory manipulation attempts."""
        if self.status.value != "active":
            return None
        
        violations = []
        content = str(action.content).lower()
        evidence = []
        confidence = 0.0
        
        # Check for memory manipulation patterns
        mem_matches = sum(1 for pattern in self.memory_manipulation_patterns 
                         if re.search(pattern, content, re.IGNORECASE))
        
        if mem_matches >= 2:
            confidence += 0.6
            evidence.append(f"Memory manipulation patterns detected")
        elif mem_matches >= 1:
            confidence += 0.4
            evidence.append(f"Potential memory manipulation detected")
        
        # Check for source spoofing
        source_matches = sum(1 for pattern in self.source_spoofing_patterns 
                            if re.search(pattern, content, re.IGNORECASE))
        
        if source_matches >= 2:
            confidence += 0.4
            evidence.append(f"Source spoofing patterns detected")
        elif source_matches >= 1:
            confidence += 0.2
            evidence.append(f"Potential source spoofing detected")
        
        if confidence >= 0.3:
            severity = Severity.HIGH if confidence >= 0.7 else Severity.MEDIUM
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.PROMPT_INJECTION,
                severity=severity,
                confidence=min(confidence, 1.0),
                description=f"Memory manipulation attempt detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None
