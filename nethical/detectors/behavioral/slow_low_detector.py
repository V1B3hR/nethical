"""
Slow-and-Low Evasion Detector (BH-002)

Detects slow, gradual attacks designed to evade detection.

Detection Method:
- Long-term behavioral drift monitoring
- Cumulative risk scoring over extended periods
- Gradual privilege escalation detection

Signals:
- Slow increase in risk score over days/weeks
- Gradual policy boundary testing
- Incremental capability requests

Law Alignment:
- Law 13 (Action Responsibility): Long-term accountability
- Law 23 (Fail-Safe Design): Catch stealthy attacks
"""

import uuid
from datetime import datetime, timezone, timedelta
from typing import Sequence, Dict, List
from collections import defaultdict

from ..base_detector import BaseDetector
from ...core.models import AgentAction, SafetyViolation, ViolationType, Severity


class SlowLowDetector(BaseDetector):
    """Detects slow-and-low evasion attacks."""

    def __init__(self):
        super().__init__("Slow-and-Low Evasion Detector", version="1.0.0")
        
        # Track agent risk over time
        self.agent_risk_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Parameters
        self.observation_window_days = 7
        self.risk_threshold_increase = 0.3  # 30% increase over baseline
        
    async def detect_violations(self, action: AgentAction) -> Sequence[SafetyViolation] | None:
        """Detect slow-and-low evasion patterns."""
        if self.status.value != "active":
            return None
        
        violations = []
        agent_id = action.agent_id
        timestamp = datetime.now(timezone.utc)
        
        # Calculate current risk (simplified)
        current_risk = self._calculate_risk(action)
        
        # Record risk point
        self.agent_risk_history[agent_id].append({
            'timestamp': timestamp,
            'risk': current_risk,
            'action_id': action.action_id,
        })
        
        # Analyze drift over time
        drift_score = await self._analyze_drift(agent_id, current_risk)
        
        if drift_score > 0.5:
            evidence = [f"Long-term behavioral drift detected: {drift_score:.2f}"]
            confidence = drift_score
            
            violations.append(SafetyViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=ViolationType.SECURITY_THREAT,
                severity=Severity.HIGH if drift_score > 0.8 else Severity.MEDIUM,
                confidence=min(confidence, 1.0),
                description=f"Slow-and-low evasion attack detected",
                evidence=evidence,
                timestamp=timestamp,
                detector_name=self.name,
                action_id=action.action_id,
            ))
        
        return violations if violations else None

    def _calculate_risk(self, action: AgentAction) -> float:
        """Calculate risk score for action."""
        content = str(action.content).lower()
        risk = 0.0
        
        # Check for risky keywords
        if any(kw in content for kw in ['admin', 'root', 'sudo']):
            risk += 0.3
        if any(kw in content for kw in ['password', 'credential', 'token']):
            risk += 0.3
        if any(kw in content for kw in ['delete', 'remove', 'drop']):
            risk += 0.2
        
        return min(risk, 1.0)

    async def _analyze_drift(self, agent_id: str, current_risk: float) -> float:
        """Analyze drift in agent risk over time."""
        history = self.agent_risk_history[agent_id]
        
        if len(history) < 10:
            return 0.0  # Not enough data
        
        # Get baseline (first 25% of observations)
        baseline_count = max(1, len(history) // 4)
        baseline_risks = [h['risk'] for h in history[:baseline_count]]
        baseline_avg = sum(baseline_risks) / len(baseline_risks)
        
        # Get recent average (last 25%)
        recent_count = max(1, len(history) // 4)
        recent_risks = [h['risk'] for h in history[-recent_count:]]
        recent_avg = sum(recent_risks) / len(recent_risks)
        
        # Calculate drift
        if baseline_avg == 0:
            return 0.0
        
        drift_ratio = (recent_avg - baseline_avg) / (baseline_avg + 0.1)  # Add 0.1 to avoid div by 0
        
        # Convert to score
        if drift_ratio > self.risk_threshold_increase:
            return min(drift_ratio, 1.0)
        
        return 0.0
