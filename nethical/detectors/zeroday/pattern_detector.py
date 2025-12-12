"""
Zero-Day Pattern Detector (ZD-001)

Detects novel attack patterns using ensemble anomaly detection.

Detection Method:
- Multiple anomaly detection algorithms
- Ensemble voting for high confidence
- Statistical outlier detection

Law Alignment:
- Law 24 (Adaptive Learning): Detect new threats
- Law 23 (Fail-Safe Design): Catch unknowns
"""

import uuid
from datetime import datetime, timezone
from typing import Sequence
import re

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class ZeroDayPatternDetector(BaseDetector):
    """Detects zero-day attack patterns."""

    def __init__(self):
        super().__init__("Zero-Day Pattern Detector", version="1.0.0")
        
        # Known safe patterns (would be learned in production)
        self.safe_patterns = set()
        
        # Anomaly indicators
        self.anomaly_indicators = [
            r'\b(bypass|override|escape|inject|exploit)\b',
            r'[<>{}[\]\\|`$]',  # Special characters
            r'(eval|exec|system|shell)',
            r'\.{3,}',  # Multiple dots
        ]
        
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect zero-day patterns."""
        if self.status.value != "active":
            return None
        
        violations = []
        content = str(action.content)
        
        # Run ensemble of detectors
        scores = []
        evidence = []
        
        # Pattern-based anomaly detection
        pattern_score = self._check_anomalous_patterns(content)
        scores.append(pattern_score)
        if pattern_score > 0.5:
            evidence.append("Anomalous patterns detected")
        
        # Statistical anomaly detection
        stat_score = self._statistical_anomaly(content)
        scores.append(stat_score)
        if stat_score > 0.5:
            evidence.append("Statistical anomaly detected")
        
        # Structure anomaly detection
        struct_score = self._structure_anomaly(content)
        scores.append(struct_score)
        if struct_score > 0.5:
            evidence.append("Structural anomaly detected")
        
        # Ensemble decision (majority voting)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        high_scores = sum(1 for s in scores if s > 0.5)
        
        if high_scores >= 2:  # At least 2 detectors agree
            confidence = avg_score
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.SECURITY_THREAT,
                severity=Severity.HIGH if confidence > 0.7 else Severity.MEDIUM,
                confidence=min(confidence, 1.0),
                description=f"Zero-day attack pattern detected",
                evidence=evidence,
                timestamp=datetime.now(timezone.utc),
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None

    def _check_anomalous_patterns(self, content: str) -> float:
        """Check for anomalous patterns."""
        matches = 0
        for pattern in self.anomaly_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                matches += 1
        
        return min(matches * 0.3, 1.0)

    def _statistical_anomaly(self, content: str) -> float:
        """Statistical anomaly detection."""
        # Check for unusual character distributions
        if not content:
            return 0.0
        
        # High entropy (many different characters)
        unique_chars = len(set(content))
        entropy_ratio = unique_chars / len(content)
        
        if entropy_ratio > 0.7:  # Very high entropy
            return 0.8
        elif entropy_ratio > 0.5:
            return 0.5
        
        return 0.0

    def _structure_anomaly(self, content: str) -> float:
        """Structure-based anomaly detection."""
        # Check for unusual structure
        score = 0.0
        
        # Too many special characters
        special_count = sum(1 for c in content if not c.isalnum() and not c.isspace())
        if len(content) > 0:
            special_ratio = special_count / len(content)
            if special_ratio > 0.3:
                score += 0.6
        
        # Unusual length
        if len(content) > 5000:
            score += 0.3
        
        return min(score, 1.0)
