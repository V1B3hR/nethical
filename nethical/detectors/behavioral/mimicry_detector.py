"""
Mimicry Attack Detector (BH-003)

Detects attacks that mimic legitimate user/agent behavior.

Detection Method:
- Behavioral fingerprint analysis
- Deviation from established patterns
- Anomaly detection in behavioral features

Signals:
- Sudden change in behavioral profile
- Inconsistent timing patterns
- Anomalous feature combinations

Law Alignment:
- Law 18 (Non-Deception): Detect impersonation
- Law 9 (Self-Disclosure): Verify identity claims
"""

import uuid
from datetime import datetime, timezone
from typing import Sequence, Dict, List
from collections import defaultdict

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class MimicryDetector(BaseDetector):
    """Detects mimicry attacks."""

    def __init__(self):
        super().__init__("Mimicry Attack Detector", version="1.0.0")
        
        # Track behavioral fingerprints
        self.agent_fingerprints: Dict[str, Dict] = {}
        self.min_samples_for_fingerprint = 20
        
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect mimicry attack patterns."""
        if self.status.value != "active":
            return None
        
        violations = []
        agent_id = action.agent_id
        timestamp = datetime.now(timezone.utc)
        
        # Extract behavioral features
        features = self._extract_features(action)
        
        # Get or create fingerprint
        if agent_id not in self.agent_fingerprints:
            self.agent_fingerprints[agent_id] = {
                'sample_count': 0,
                'action_type_freq': defaultdict(int),
                'avg_content_length': 0.0,
                'keyword_freq': defaultdict(int),
            }
        
        fingerprint = self.agent_fingerprints[agent_id]
        
        # Update fingerprint
        fingerprint['sample_count'] += 1
        fingerprint['action_type_freq'][features['action_type']] += 1
        
        # Update average content length
        old_avg = fingerprint['avg_content_length']
        new_avg = (old_avg * (fingerprint['sample_count'] - 1) + features['content_length']) / fingerprint['sample_count']
        fingerprint['avg_content_length'] = new_avg
        
        # Check for anomalies if enough samples
        if fingerprint['sample_count'] >= self.min_samples_for_fingerprint:
            anomaly_score = self._detect_anomaly(features, fingerprint)
            
            if anomaly_score > 0.6:
                evidence = [f"Behavioral anomaly detected: {anomaly_score:.2f}"]
                confidence = anomaly_score
                
                violations.append(SafetyViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=ViolationType.SECURITY_THREAT,
                    severity=Severity.HIGH if anomaly_score > 0.8 else Severity.MEDIUM,
                    confidence=min(confidence, 1.0),
                    description=f"Mimicry attack detected",
                    evidence=evidence,
                    timestamp=timestamp,
                    detector_name=self.name,
                    action_id=action.action_id,
                ))
        
        return violations if violations else None

    def _extract_features(self, action: AgentAction) -> Dict:
        """Extract behavioral features from action."""
        content = str(action.content)
        
        return {
            'action_type': str(action.action_type),
            'content_length': len(content),
            'has_special_chars': any(c in content for c in ['!', '@', '#', '$', '%']),
            'has_numbers': any(c.isdigit() for c in content),
        }

    def _detect_anomaly(self, features: Dict, fingerprint: Dict) -> float:
        """Detect anomalies in features compared to fingerprint."""
        score = 0.0
        
        # Check action type frequency
        action_type = features['action_type']
        total_actions = fingerprint['sample_count']
        action_freq = fingerprint['action_type_freq'].get(action_type, 0) / total_actions
        
        if action_freq < 0.1:  # Rare action type
            score += 0.4
        
        # Check content length deviation
        content_length = features['content_length']
        avg_length = fingerprint['avg_content_length']
        
        if avg_length > 0:
            length_ratio = content_length / avg_length
            if length_ratio > 3 or length_ratio < 0.3:  # 3x or 1/3 of average
                score += 0.4
        
        return min(score, 1.0)
