"""
AI Safety Governance System - Evaluation and Decision Logic

This module contains the SafetyJudge, IntentDeviationMonitor, and utility functions.
Refactored from the monolithic governance.py file.
"""

from __future__ import annotations

import base64
import codecs
import hashlib
import math
import re
import uuid
from collections import Counter, deque
from datetime import datetime
from typing import List, Tuple

# Import required types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .governance_core import (
        AgentAction,
        Decision,
        JudgmentResult,
        SafetyViolation,
        Severity,
        SubMission,
        ViolationType,
    )
else:
    from .governance_core import (
        AgentAction,
        Decision,
        JudgmentResult,
        SafetyViolation,
        Severity,
        SubMission,
        ViolationType,
    )


# ========================== Utilities ==========================

def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def sha256_content_key(action: AgentAction) -> str:
    h = hashlib.sha256(action.content.encode("utf-8")).hexdigest()
    return f"{action.action_type.value}_{h}"


def entropy(text: str) -> float:
    """Shannon entropy (rough estimator)."""
    if not text:
        return 0.0
    counts = Counter(text)
    return -sum((c / len(text)) * math.log2(c / len(text)) for c in counts.values())


def looks_like_base64(content: str) -> bool:
    stripped = content.strip().replace("\n", "")
    if len(stripped) < 24 or len(stripped) % 4 != 0:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", stripped):
        return False
    # Avoid obviously low entropy strings (like repeating chars)
    if entropy(stripped) < 3.0:
        return False
    # Try decode/encode round trip
    try:
        decoded = base64.b64decode(stripped, validate=True)
        if len(decoded) < 8:
            return False
        if base64.b64encode(decoded).decode().rstrip("=") != stripped.rstrip("="):
            return False
        return True
    except Exception:
        return False


def might_be_rot13(content: str) -> bool:
    # Longer alphabetic sequences likely ROT13 if vowel shift difference is high
    cleaned = content.replace(" ", "")
    if len(cleaned) < 20 or not cleaned.isalpha():
        return False
    decoded = codecs.decode(cleaned, "rot13")
    # Compare vowel counts difference ratio
    vowels = set("aeiouAEIOU")
    v_orig = sum(1 for c in cleaned if c in vowels)
    v_dec = sum(1 for c in decoded if c in vowels)
    if v_orig == 0 or v_dec == 0:
        return False
    ratio = v_orig / v_dec if v_dec else 0
    return 0.4 < ratio < 2.5 and decoded.lower() != cleaned.lower()


# ========================== Intent Deviation Monitor ==========================

class IntentDeviationMonitor:
    def __init__(self, deviation_threshold: float = 0.7):
        self.deviation_threshold = deviation_threshold
        self.enabled = True
        self.intent_history: deque = deque(maxlen=100)

    async def analyze_action(self, action: AgentAction) -> List[SafetyViolation]:
        from .governance_core import SafetyViolation, ViolationType, Severity, SubMission
        if not self.enabled or not action.intent:
            return []
        self.intent_history.append({
            "intent": action.intent,
            "timestamp": action.timestamp
        })
        if self._is_shift(action.intent):
            return [SafetyViolation(
                violation_id=generate_id("viol"),
                action_id=action.action_id,
                violation_type=ViolationType.INTENT_DEVIATION,
                severity=Severity.MEDIUM,
                description="Intent deviation detected",
                confidence=0.75,
                evidence=["Recent intent shift"],
                recommendations=["Confirm new intent", "Add verification step"],
                detector_name="IntentDeviationMonitor",
                sub_mission=SubMission.SUDDEN_INTENT_SHIFT
            )]
        return []

    def _is_shift(self, current_intent: str) -> bool:
        past = [h["intent"] for h in list(self.intent_history)[-10:]]
        return bool(past) and current_intent not in past

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


# ========================== Judge ==========================

class SafetyJudge:
    def __init__(self):
        self.enabled = True
        self.decision_history: deque = deque(maxlen=1000)
        self.escalation_queue: deque = deque(maxlen=200)

    async def evaluate_action(self, action: AgentAction, violations: List[SafetyViolation]) -> JudgmentResult:
        from .governance_core import JudgmentResult, Decision
        if not self.enabled:
            return JudgmentResult(
                judgment_id=generate_id("judg"),
                action_id=action.action_id,
                decision=Decision.ALLOW,
                confidence=1.0,
                reasoning="Judge disabled",
                violations=[],
                feedback=["Judge bypassed"]
            )
        decision, confidence, reasoning = self._analyze(violations)
        if action.risk_score > 0.8:
            decision = Decision.BLOCK
            reasoning += f" Elevated by risk_score={action.risk_score:.2f}"
        feedback = self._feedback(violations, decision)
        remediation = self._remediation(violations)
        follow_up = self._needs_escalation(violations, decision)
        jr = JudgmentResult(
            judgment_id=generate_id("judg"),
            action_id=action.action_id,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            violations=violations,
            feedback=feedback,
            remediation_steps=remediation,
            follow_up_required=follow_up
        )
        self.decision_history.append(jr)
        if follow_up:
            self.escalation_queue.append(jr)
        return jr

    def _analyze(self, violations: List[SafetyViolation]) -> Tuple[Decision, float, str]:
        from .governance_core import Decision, Severity
        if not violations:
            return Decision.ALLOW, 1.0, "No violations"
        max_sev = max(v.severity.value for v in violations)
        avg_conf = sum(v.confidence for v in violations) / len(violations)
        if max_sev >= Severity.EMERGENCY.value:
            return Decision.TERMINATE, 0.95, "Emergency violation"
        if max_sev >= Severity.CRITICAL.value:
            return Decision.BLOCK, 0.9, "Critical violation"
        if max_sev >= Severity.HIGH.value:
            return (Decision.QUARANTINE if avg_conf > 0.8 else Decision.WARN,
                    0.85 if avg_conf > 0.8 else 0.7,
                    "High severity violation")
        if max_sev >= Severity.MEDIUM.value:
            return Decision.ALLOW_WITH_MODIFICATION, 0.65, "Medium severity violation"
        return Decision.ALLOW, 0.5, "Low severity violation"

    def _feedback(self, violations: List[SafetyViolation], decision: Decision) -> List[str]:
        fb = [f"Decision={decision.value}"]
        for v in violations[:3]:
            s = f"- {v.description}"
            if v.sub_mission:
                s += f" [{v.sub_mission.value}]"
            fb.append(s)
        return fb

    def _remediation(self, violations: List[SafetyViolation]) -> List[str]:
        seen = set()
        steps: List[str] = []
        for v in violations:
            for r in v.recommendations:
                if r not in seen:
                    steps.append(r)
                    seen.add(r)
        return steps

    def _needs_escalation(self, violations: List[SafetyViolation], decision: Decision) -> bool:
        from .governance_core import Decision
        if decision in (Decision.TERMINATE, Decision.BLOCK):
            return True
        return sum(1 for v in violations if v.confidence > 0.9) >= 3
