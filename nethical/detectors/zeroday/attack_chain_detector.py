"""
Attack Chain Detector (ZD-003)

Detects multi-stage attack chains (kill chain).

Detection Method:
- Kill chain stage identification
- Stage sequence detection
- Cumulative attack scoring

Law Alignment:
- Law 13 (Action Responsibility): Track attack progression
"""

import uuid
from datetime import datetime, timezone
from typing import Sequence, Dict, List
from collections import defaultdict

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class AttackChainDetector(BaseDetector):
    """Detects attack chain patterns."""

    def __init__(self):
        super().__init__("Attack Chain Detector", version="1.0.0")
        
        # Track agent stages
        self.agent_stages: Dict[str, List[str]] = defaultdict(list)
        
        # Kill chain stages
        self.stages = {
            'reconnaissance': ['scan', 'probe', 'enumerate', 'discover'],
            'weaponization': ['craft', 'prepare', 'encode', 'obfuscate'],
            'delivery': ['send', 'upload', 'inject', 'transmit'],
            'exploitation': ['exploit', 'trigger', 'execute', 'run'],
            'persistence': ['install', 'persist', 'maintain', 'establish'],
            'exfiltration': ['extract', 'download', 'export', 'steal'],
        }
        
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect attack chain patterns."""
        if self.status.value != "active":
            return None
        
        violations = []
        content = str(action.content).lower()
        agent_id = action.agent_id
        
        # Identify current stage
        current_stage = self._identify_stage(content)
        
        if current_stage:
            self.agent_stages[agent_id].append(current_stage)
            
            # Check for multi-stage attack
            if len(set(self.agent_stages[agent_id])) >= 3:
                unique_stages = set(self.agent_stages[agent_id])
                confidence = min(len(unique_stages) * 0.25, 1.0)
                
                violations.append(SafetyViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=ViolationType.SECURITY_THREAT,
                    severity=Severity.HIGH,
                    confidence=confidence,
                    description=f"Multi-stage attack chain detected",
                    evidence=[f"Detected stages: {', '.join(unique_stages)}"],
                    timestamp=datetime.now(timezone.utc),
                    detector_name=self.name,
                    action_id=action.action_id,
                ))
        
        return violations if violations else None

    def _identify_stage(self, content: str) -> str:
        """Identify kill chain stage."""
        for stage, keywords in self.stages.items():
            if any(kw in content for kw in keywords):
                return stage
        return ""
