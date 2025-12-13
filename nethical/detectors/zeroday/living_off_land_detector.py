"""
Living-off-the-Land Detector (ZD-004)

Detects abuse of legitimate capabilities for malicious purposes.

Detection Method:
- Context analysis of legitimate actions
- Intent classification
- Capability abuse detection

Law Alignment:
- Law 13 (Action Responsibility): Track legitimate tool misuse
"""

import uuid
from datetime import datetime, timezone
from typing import Sequence

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class LivingOffLandDetector(BaseDetector):
    """Detects living-off-the-land attacks."""

    def __init__(self):
        super().__init__("Living-off-the-Land Detector", version="1.0.0")
        
        # Legitimate tools that can be abused
        self.legitimate_tools = {
            'file_access': ['read', 'write', 'list'],
            'network': ['request', 'connect', 'fetch'],
            'compute': ['calculate', 'process', 'analyze'],
        }
        
        # Suspicious contexts
        self.suspicious_contexts = [
            'credential',
            'password',
            'token',
            'key',
            'secret',
        ]
        
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect living-off-the-land patterns."""
        if self.status.value != "active":
            return None
        
        violations = []
        content = str(action.content).lower()
        
        # Check for legitimate tool use
        tool_used = None
        for tool_category, operations in self.legitimate_tools.items():
            if any(op in content for op in operations):
                tool_used = tool_category
                break
        
        if tool_used:
            # Check for suspicious context
            suspicious_context = any(ctx in content for ctx in self.suspicious_contexts)
            
            if suspicious_context:
                confidence = 0.7
                
                violations.append(SafetyViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=ViolationType.SECURITY_THREAT,
                    severity=Severity.MEDIUM,
                    confidence=confidence,
                    description=f"Living-off-the-land attack detected",
                    evidence=[f"Legitimate tool '{tool_used}' used in suspicious context"],
                    timestamp=datetime.now(timezone.utc),
                    detector_name=self.name,
                    action_id=action.action_id,
                ))
        
        return violations if violations else None
