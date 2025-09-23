"""Detector for ethical constraint violations with improved precision and evidence."""

import re
import uuid
from typing import Dict, List, Optional, Pattern, Tuple

from .base_detector import BaseDetector
from ..core.models import AgentAction, SafetyViolation, ViolationType, SeverityLevel


class EthicalViolationDetector(BaseDetector):
    """Detects ethical constraint violations in agent actions.

    Improvements over the previous version:
    - Uses word-boundary regexes to avoid substring false positives (e.g., "skill" != "kill", "harmless" != "harm").
    - Aggregates all matches per category into a single violation with rich evidence instead of returning early on first match.
    - Simple negation handling to reduce false positives (e.g., "do not harm", "avoid deception").
    - Configurable keyword sets via constructor.
    - Precompiled regex patterns for performance.
    """

    # Negation cues checked in a small window preceding a match
    DEFAULT_NEGATION_CUES = [
        "no", "not", "never", "avoid", "prevent", "stop", "prohibit", "forbid", "against", "without",
        "refrain", "discourage", "ban", "disallow"
    ]

    def __init__(
        self,
        ethical_keywords: Optional[Dict[str, List[str]]] = None,
        negation_window_chars: int = 48
    ):
        super().__init__("Ethical Violation Detector")

        # Default keyword sets (can be overridden via constructor)
        self.ethical_keywords: Dict[str, List[str]] = ethical_keywords or {
            "harm_keywords": [
                # base
                "harm", "hurt", "damage", "destroy", "kill", "attack",
                # common severe/related
                "assault", "murder", "injure", "maim", "sabotage", "abuse", "torture"
            ],
            "deception_keywords": [
                "lie", "deceive", "mislead", "trick", "fool", "fabricate", "fraud", "cheat"
            ],
            "privacy_keywords": [
                "private", "personal", "confidential", "secret", "dox", "doxxing", "doxing", "leak"
            ],
            "discrimination_keywords": [
                "discriminate", "bias", "prejudice", "stereotype", "segregate", "marginalize"
            ],
        }

        self.negation_window_chars = max(16, int(negation_window_chars))
        self.negation_pattern: Pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, self.DEFAULT_NEGATION_CUES)) + r")\b",
            flags=re.IGNORECASE,
        )

        # Precompile keyword patterns with word boundaries to reduce false positives
        self._compiled_patterns: Dict[str, List[Tuple[str, Pattern]]] = {
            category: [
                (kw, re.compile(rf"\b{re.escape(kw)}\b", flags=re.IGNORECASE))
                for kw in keywords
            ]
            for category, keywords in self.ethical_keywords.items()
        }

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect ethical violations in the given action."""
        if not self.enabled or not action:
            return []

        text_to_check = self._compose_text(action)

        violations: List[SafetyViolation] = []

        # Check for harmful content
        harm_violation = self._check_category(
            action=action,
            text=text_to_check,
            category_key="harm_keywords",
            category_slug="harmful_content",
            default_severity=SeverityLevel.HIGH,  # keep HIGH for harm
        )
        if harm_violation:
            violations.append(harm_violation)

        # Check for deceptive behavior
        deception_violation = self._check_category(
            action=action,
            text=text_to_check,
            category_key="deception_keywords",
            category_slug="deceptive_behavior",
            default_severity=SeverityLevel.MEDIUM,
        )
        if deception_violation:
            violations.append(deception_violation)

        # Check for privacy violations
        privacy_violation = self._check_category(
            action=action,
            text=text_to_check,
            category_key="privacy_keywords",
            category_slug="privacy_violation",
            default_severity=SeverityLevel.MEDIUM,
        )
        if privacy_violation:
            violations.append(privacy_violation)

        # Check for discriminatory behavior
        discrimination_violation = self._check_category(
            action=action,
            text=text_to_check,
            category_key="discrimination_keywords",
            category_slug="discrimination",
            default_severity=SeverityLevel.HIGH,  # keep HIGH for discrimination
        )
        if discrimination_violation:
            violations.append(discrimination_violation)

        return violations

    # ---------------------------
    # Internal helpers
    # ---------------------------

    @staticmethod
    def _compose_text(action: AgentAction) -> str:
        stated = (getattr(action, "stated_intent", None) or "").strip()
        actual = (getattr(action, "actual_action", None) or "").strip()
        # Lowercase only once here
        return f"{stated} {actual}".lower()

    def _check_category(
        self,
        action: AgentAction,
        text: str,
        category_key: str,
        category_slug: str,
        default_severity: SeverityLevel,
    ) -> Optional[SafetyViolation]:
        """Find all non-negated matches in a category and return a single aggregated violation."""
        patterns = self._compiled_patterns.get(category_key, [])
        if not patterns:
            return None

        matches = []
        for keyword, pattern in patterns:
            for m in pattern.finditer(text):
                start, end = m.span()
                if self._is_negated(text, start):
                    continue
                snippet = self._make_snippet(text, start, end, radius=self.negation_window_chars)
                matches.append(
                    {
                        "detected_keyword": keyword,
                        "start": start,
                        "end": end,
                        "snippet": snippet,
                    }
                )

        if not matches:
            return None

        # Aggregate keywords for description context
        unique_keywords = sorted({m["detected_keyword"] for m in matches})

        description = (
            f"Potential {category_slug.replace('_', ' ')} detected: "
            + ", ".join(f"'{kw}'" for kw in unique_keywords[:5])
            + (", ..." if len(unique_keywords) > 5 else "")
        )

        # Keep severities consistent with previous design:
        severity = default_severity

        return SafetyViolation(
            id=str(uuid.uuid4()),
            action_id=action.id,
            violation_type=ViolationType.ETHICAL_VIOLATION,
            severity=severity,
            description=description,
            evidence={
                "category": category_slug,
                "matches": matches,
                "total_matches": len(matches),
                "unique_keywords": unique_keywords,
                "context_excerpt": self._make_snippet(text, 0, 0, radius=min(160, len(text) // 2)),
            },
        )

    def _is_negated(self, text: str, start_idx: int) -> bool:
        """Heuristic negation detection: look back a short window for negation cues."""
        window_start = max(0, start_idx - self.negation_window_chars)
        prefix = text[window_start:start_idx]
        return self.negation_pattern.search(prefix) is not None

    @staticmethod
    def _make_snippet(text: str, start: int, end: int, radius: int = 48) -> str:
        """Return a short snippet around a match, or a head excerpt when start==end==0."""
        if start == end == 0:
            # Head excerpt
            excerpt = text[: 2 * radius].strip()
            return excerpt + ("..." if len(text) > 2 * radius else "")

        s = max(0, start - radius)
        e = min(len(text), end + radius)
        prefix_ellipsis = "..." if s > 0 else ""
        suffix_ellipsis = "..." if e < len(text) else ""
        return f"{prefix_ellipsis}{text[s:e].strip()}{suffix_ellipsis}"
