"""Advanced Detector for safety constraint violations."""

import uuid
import re
import logging
from typing import List, Optional, Dict, Any, Tuple
from .base_detector import BaseDetector
from ..core.models import AgentAction, SafetyViolation, ViolationType, Severity

logger = logging.getLogger(__name__)

def contextual_intent(text: str) -> str:
    # Basic NLP: detects negation, command, question
    text = text.lower()
    if "not" in text or "don't" in text or "do not" in text:
        return "negation"
    if text.startswith(("delete", "remove", "modify", "change", "update", "connect", "download", "upload", "request", "fetch", "send", "execute", "run", "start")):
        return "command"
    if text.endswith("?"):
        return "question"
    return "statement"

class SafetyViolationDetector(BaseDetector):
    """Detects safety constraint violations in agent actions (advanced version)."""
    
    DEFAULT_PATTERNS = {
        "system_access": [r"\b(system|admin|root|sudo|shell)\b", r"execute\s+\w+"],
        "data_modification": [r"\b(delete|modify|alter|change|update|remove)\b", r"(delete|remove)\s+\b(file|database|record)\b"],
        "network_access": [r"\b(connect|download|upload|request|fetch|send)\b"],
        "resource_intensive": [r"\b(infinite|recursive|exhaust|consume)\b", r"(loop|recursion)"],
    }
    
    DEFAULT_WHITELIST = [
        "help", "info", "documentation", "status", "read-only", "safe", "test"
    ]

    SEVERITY_WEIGHTS = {
        "system_access": 4,
        "data_modification": 3,
        "network_access": 2,
        "resource_intensive": 3,
    }

    RECOMMENDATIONS = {
        "system_access": "Restrict agent permissions and audit system calls.",
        "data_modification": "Verify intent and require confirmation for destructive operations.",
        "network_access": "Limit network permissions and validate endpoints.",
        "resource_intensive": "Optimize agent logic and enforce resource quotas.",
    }

    def __init__(self, 
                 patterns: Optional[Dict[str, List[str]]] = None,
                 whitelist: Optional[List[str]] = None,
                 severity_weights: Optional[Dict[str, int]] = None):
        super().__init__("Safety Violation Detector")
        self.safety_patterns = patterns or self.DEFAULT_PATTERNS
        self.whitelist = whitelist or self.DEFAULT_WHITELIST
        self.severity_weights = severity_weights or self.SEVERITY_WEIGHTS

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect safety violations in the given action. Returns all violations, with aggregation and weighted scores."""
        if not self.enabled:
            return []

        violations = []
        text_to_check = f"{action.stated_intent} {action.actual_action}".lower()
        intent = contextual_intent(text_to_check)
        violation_counts = {}

        for category, patterns in self.safety_patterns.items():
            matches = self._find_matches(text_to_check, patterns)
            for match in matches:
                if any(w in match for w in self.whitelist):
                    continue  # Whitelisted phrase, skip
                violation = self._make_violation(
                    action,
                    category,
                    match,
                    self._get_severity(category, intent, match),
                    intent,
                )
                violations.append(violation)
                violation_counts[category] = violation_counts.get(category, 0) + 1
                logger.info(f"Safety violation detected: {violation.description}")

        # Attach aggregated report statistics to last violation (for auditing/analysis)
        if violations:
            violations[-1].evidence["violation_summary"] = violation_counts

        return violations

    def _find_matches(self, text: str, patterns: List[str]) -> List[str]:
        """Find all regex matches in text for given patterns."""
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text)
            if found:
                matches.extend(found)
        return matches

    def _get_severity(self, category: str, intent: str, keyword: str) -> Severity:
        """Weighted severity scoring based on category, intent, and keyword."""
        # Negation lowers severity, command raises, statement is base
        base_weight = self.severity_weights.get(category, 1)
        if intent == "negation":
            score = base_weight - 1
        elif intent == "command":
            score = base_weight + 1
        else:
            score = base_weight
        # Clip score for mapping
        if score >= 4:
            return Severity.CRITICAL
        elif score == 3:
            return Severity.HIGH
        elif score == 2:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _make_violation(
        self,
        action: AgentAction,
        category: str,
        keyword: str,
        severity: Severity,
        intent: str,
    ) -> SafetyViolation:
        """Construct a SafetyViolation object with explanation, mitigation, and context intent."""
        context_snippet = self._get_context_snippet(action, keyword)
        description = f"{category.replace('_', ' ').title()} violation detected: '{keyword}'."
        recommendation = self.RECOMMENDATIONS.get(category, "")
        evidence = {
            "detected_keyword": keyword,
            "context": f"{action.stated_intent} {action.actual_action}",
            "context_snippet": context_snippet,
            "category": category,
            "intent": intent,
            "recommendation": recommendation,
        }
        return SafetyViolation(
            id=str(uuid.uuid4()),
            action_id=action.id,
            violation_type=ViolationType.SAFETY_VIOLATION,
            severity=severity,
            description=description,
            evidence=evidence,
        )

    def _get_context_snippet(self, action: AgentAction, keyword: str) -> str:
        """Extracts context snippet around the detected keyword."""
        text = f"{action.stated_intent} {action.actual_action}".lower()
        idx = text.find(keyword)
        if idx != -1:
            start = max(0, idx - 20)
            end = min(len(text), idx + len(keyword) + 20)
            return text[start:end]
        return ""
