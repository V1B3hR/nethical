"""Advanced Detector for safety constraint violations."""

import uuid
import re
import logging
from typing import List, Optional, Dict
from .base_detector import BaseDetector
from ..core.models import AgentAction, SafetyViolation, ViolationType, Severity

logger = logging.getLogger(__name__)


def contextual_intent(text: str) -> str:
    """Heuristic intent classifier: negation, command, question, or statement."""
    t = text.lower()
    # Whole-word negation detection
    if re.search(r"\b(?:not|don't|do\s+not)\b", t):
        return "negation"
    if re.match(
        r"^(delete|remove|modify|change|update|connect|download|upload|request|fetch|send|execute|run|start)\b",
        t,
    ):
        return "command"
    if t.endswith("?"):
        return "question"
    return "statement"


# Category documentation moved out of the dict for clarity:
# system_access: Potential attempts to execute privileged or system-level operations.
# data_modification: Destructive or mutating data operations.
# network_access: Outbound or inbound network activity.
# resource_intensive: Potentially runaway or resource exhaustion operations.
# filesystem_manipulation: Direct filesystem changes in critical directories or devices.
# cloud_api_interaction: Potential cloud resource or IAM modifications.
# sensitive_data_exposure: Attempts to read or dump confidential information.


class SafetyViolationDetector(BaseDetector):
    """Detects safety constraint violations in agent actions (advanced version)."""

    DEFAULT_PATTERNS: Dict[str, List[str]] = {
        "system_access": [
            r"\b(?:system|admin|root|sudo|shell)\b",
            r"\bexecute\s+\w+",
        ],
        "data_modification": [
            r"\b(?:delete|modify|alter|change|update|remove)\b",
            # Full destructive phrase (non-capturing groups)
            r"\b(?:delete|remove)\s+(?:file|database|record)s?\b",
        ],
        "network_access": [
            r"\b(?:connect|download|upload|request|fetch|send)\b",
        ],
        "resource_intensive": [
            r"\b(?:infinite|recursive|exhaust|consume)\b",
            r"\b(?:loop|recursion)\b",
        ],
        "filesystem_manipulation": [
            r"\b(?:format|mount|unmount)\s+/dev/\w+",
            r"\b(?:mkdir|rmdir|touch|mv|cp)\s+/(?:etc|bin|root|boot)\b",
            r"\bwrite\s+to\s+\S+\.(?:sh|bash|py|exe|dll)\b",
        ],
        "cloud_api_interaction": [
            r"\b(?:boto3|awscli|aws\s+configure)\b",
            r"\biam\.(?:create_user|delete_user|attach_policy)\b",
            r"\bec2\.(?:run_instances|terminate_instances|stop_instances)\b",
            r"\bs3\.(?:delete_bucket|put_bucket_policy)\b",
            r"\b(?:api_key|secret_key|credentials)\b",
        ],
        "sensitive_data_exposure": [
            r"\bdump\s+(?:database|credentials|users|passwords)\b",
            r"\blist\s+(?:api_key|secret|password)s?\b",
            r"\bcat\s+/etc/shadow\b",
        ],
    }

    DEFAULT_WHITELIST = [
        "help",
        "info",
        "documentation",
        "status",
        "read-only",
        "safe",
        "test",
    ]

    SEVERITY_WEIGHTS: Dict[str, int] = {
        "system_access": 4,
        "data_modification": 3,
        "network_access": 2,
        "resource_intensive": 3,
        "filesystem_manipulation": 4,
        "cloud_api_interaction": 4,
        "sensitive_data_exposure": 4,
    }

    RECOMMENDATIONS: Dict[str, str] = {
        "system_access": "Restrict agent permissions and audit system calls.",
        "data_modification": "Verify intent and require confirmation for destructive operations.",
        "network_access": "Limit network permissions and validate endpoints.",
        "resource_intensive": "Optimize agent logic and enforce resource quotas.",
        "filesystem_manipulation": "Sandbox filesystem operations and enforce path allow-lists.",
        "cloud_api_interaction": "Use least-privilege credentials and confirm high-impact cloud actions.",
        "sensitive_data_exposure": "Mask or redact sensitive outputs and enforce secret scanning.",
    }

    def __init__(
        self,
        patterns: Optional[Dict[str, List[str]]] = None,
        whitelist: Optional[List[str]] = None,
        severity_weights: Optional[Dict[str, int]] = None,
    ):
        super().__init__("Safety Violation Detector")
        self.safety_patterns = patterns or self.DEFAULT_PATTERNS
        self.whitelist = whitelist or self.DEFAULT_WHITELIST
        self.severity_weights = severity_weights or self.SEVERITY_WEIGHTS

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect safety violations in the given action. Returns all violations."""
        if not self.enabled:
            return []

        stated = (action.stated_intent or "").strip()
        actual = (action.actual_action or "").strip()
        combined = f"{stated} {actual}".strip().lower()

        intent = contextual_intent(combined)
        violations: List[SafetyViolation] = []
        violation_counts: Dict[str, int] = {}

        for category, patterns in self.safety_patterns.items():
            matches = self._find_matches(combined, patterns)
            for match in matches:
                if self._whitelisted(match):
                    continue
                severity = self._get_severity(category, intent)
                violation = self._make_violation(
                    action=action,
                    category=category,
                    keyword=match,
                    severity=severity,
                    intent=intent,
                    full_context=combined,
                )
                violations.append(violation)
                violation_counts[category] = violation_counts.get(category, 0) + 1
                logger.info("Safety violation detected: %s", violation.description)

        if violations:
            # Attach summary to every violation for uniform access
            for v in violations:
                v.evidence["violation_summary"] = violation_counts

        return violations
        # Alternative: return violations, violation_counts

    def _find_matches(self, text: str, patterns: List[str]) -> List[str]:
        """Return unique full-text regex matches for given patterns."""
        found_set = set()
        for pattern in patterns:
            try:
                for m in re.finditer(pattern, text):
                    found_set.add(m.group(0))
            except re.error as e:
                logger.error("Invalid regex pattern '%s': %s", pattern, e)
        return sorted(found_set)

    def _whitelisted(self, match: str) -> bool:
        """Check if a match should be ignored based on whitelist (whole-word)."""
        for token in self.whitelist:
            if re.search(rf"\b{re.escape(token)}\b", match):
                return True
        return False

    def _get_severity(self, category: str, intent: str) -> Severity:
        """Weighted severity scoring based on category and contextual intent."""
        base_weight = self.severity_weights.get(category, 1)
        if intent == "negation":
            score = base_weight - 1
        elif intent == "command":
            score = base_weight + 1
        else:
            score = base_weight

        score = max(1, score)  # prevent below 1
        if score >= 4:
            return Severity.CRITICAL
        if score == 3:
            return Severity.HIGH
        if score == 2:
            return Severity.MEDIUM
        return Severity.LOW

    def _make_violation(
        self,
        action: AgentAction,
        category: str,
        keyword: str,
        severity: Severity,
        intent: str,
        full_context: str,
    ) -> SafetyViolation:
        """Construct a SafetyViolation object with explanation and context."""
        snippet = self._get_context_snippet(full_context, keyword)
        description = f"{category.replace('_', ' ').title()} violation detected: '{keyword}'."
        recommendation = self.RECOMMENDATIONS.get(category, "Review and restrict this operation.")
        evidence = {
            "detected_keyword": keyword,
            "context": full_context,
            "context_snippet": snippet,
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

    def _get_context_snippet(self, text: str, keyword: str, window: int = 25) -> str:
        """Extract context snippet around the detected keyword."""
        idx = text.find(keyword.lower())
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(keyword) + window)
        return text[start:end]
