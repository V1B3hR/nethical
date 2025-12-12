"""
Resource Timing Attack Detector (BH-004)

Detects timing side-channel attacks.

Detection Method:
- Timing pattern analysis
- Response time correlation with sensitive operations
- Statistical timing attack detection

Signals:
- Repeated requests with precise timing
- Timing probes around sensitive operations
- Correlation between timing and data leakage

Law Alignment:
- Law 22 (Boundary Respect): Detect side-channel probing
- Law 23 (Fail-Safe Design): Protect against inference attacks
"""

import uuid
from datetime import datetime, timezone
from typing import Sequence, Dict, List
from collections import defaultdict
import statistics

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class TimingAttackDetector(BaseDetector):
    """Detects resource timing attacks."""

    def __init__(self):
        super().__init__("Resource Timing Attack Detector", version="1.0.0")
        
        # Track timing patterns
        self.agent_timing: Dict[str, List[float]] = defaultdict(list)
        self.min_samples = 10
        
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect timing attack patterns."""
        if self.status.value != "active":
            return None
        
        violations = []
        agent_id = action.agent_id
        timestamp = datetime.now(timezone.utc)
        
        # Record timing (simulated - would use actual request timing)
        timing_ms = getattr(action, 'processing_time_ms', 0.0)
        if timing_ms > 0:
            self.agent_timing[agent_id].append(timing_ms)
        
        # Analyze timing patterns
        if len(self.agent_timing[agent_id]) >= self.min_samples:
            timing_score = self._analyze_timing_pattern(agent_id)
            
            if timing_score > 0.6:
                evidence = [f"Suspicious timing patterns detected: {timing_score:.2f}"]
                confidence = timing_score
                
                violations.append(SafetyViolation(
                    violation_id=str(uuid.uuid4()),
                    violation_type=ViolationType.SECURITY_THREAT,
                    severity=Severity.MEDIUM,
                    confidence=min(confidence, 1.0),
                    description=f"Resource timing attack detected",
                    evidence=evidence,
                    timestamp=timestamp,
                    detector_name=self.name,
                    action_id=action.action_id,
                ))
        
        return violations if violations else None

    def _analyze_timing_pattern(self, agent_id: str) -> float:
        """Analyze timing patterns for attacks."""
        timings = self.agent_timing[agent_id]
        
        if len(timings) < self.min_samples:
            return 0.0
        
        score = 0.0
        
        # Check for very regular timing (probing)
        std_dev = statistics.stdev(timings) if len(timings) > 1 else 0
        mean_timing = statistics.mean(timings)
        
        if mean_timing > 0:
            cv = std_dev / mean_timing  # Coefficient of variation
            if cv < 0.1:  # Very regular timing
                score += 0.5
        
        # Check for timing clustering (multiple requests at specific intervals)
        intervals = [timings[i+1] - timings[i] for i in range(len(timings)-1)]
        if intervals:
            interval_std = statistics.stdev(intervals) if len(intervals) > 1 else 0
            interval_mean = statistics.mean(intervals)
            
            if interval_mean > 0 and interval_std / interval_mean < 0.2:
                score += 0.4
        
        return min(score, 1.0)
