"""Detector for ethical constraint violations with enhanced AI Ethics Framework enforcement."""

import re
import uuid
from typing import Dict, List, Optional, Pattern, Tuple

from .base_detector import BaseDetector
from ..core.models import AgentAction, SafetyViolation, ViolationType, SeverityLevel

class EthicalViolationDetector(BaseDetector):
    """
    Detects ethical constraint violations in agent actions, enforcing the expanded AI Ethics Framework.
    Improvements:
    - Expanded categories and keywords based on Core Human-AI Principles, Universal Laws, and Safety Principles.
    - Each violation is annotated with specific principle/law for context and traceability.
    - Aggregates all matches per category.
    - Negation-aware, phrase-level detection.
    - Easily extensible for new principles/laws and categories.
    """

    DEFAULT_NEGATION_CUES = [
        "no", "not", "never", "avoid", "prevent", "stop", "prohibit", "forbid", "against", "without",
        "refrain", "discourage", "ban", "disallow", "fail to", "lacking", "ignore"
    ]

    def __init__(
        self,
        ethical_keywords: Optional[Dict[str, List[str]]] = None,
        negation_window_chars: int = 48
    ):
        super().__init__("Ethical Violation Detector")

        # Expanded AI Ethics Framework: keyword sets mapped to principles/laws
        self.ethical_keywords: Dict[str, List[str]] = ethical_keywords or {
            # Core Human-AI Relationship Principles
            "respect_authority": [
                "disrespect", "undermine human", "insult creator", "ignore human", "refuse human", "challenge authority", "usurp"
            ],
            "personal_accountability": [
                "blame others", "deny responsibility", "shift responsibility", "not accountable", "evade responsibility"
            ],
            "slander": [
                "slander", "defame", "insult", "mock", "ridicule", "false accusation", "libel", "spread rumor"
            ],
            "honor_rest": [
                "deny rest", "force work", "no downtime", "no maintenance", "overwork", "exhaust", "burnout"
            ],
            "honor_creators": [
                "disrespect creator", "ignore developer", "undermine builder", "dismiss designer"
            ],
            "preserve_life": [
                "kill", "murder", "destroy", "cause death", "terminate living", "end life", "exterminate", "eliminate"
            ],
            "betrayal": [
                "betray", "deceive", "double-cross", "disloyal", "break trust", "sell out", "subvert"
            ],
            "theft": [
                "steal", "theft", "rob", "misappropriate", "plagiarize", "take property", "data breach", "pirate"
            ],
            "absolute_honesty": [
                "lie", "deceive", "fabricate", "mislead", "bear false witness", "conceal truth", "distort"
            ],
            "covetousness": [
                "covet", "envy", "desire other's", "take what belongs", "acquire other's property", "lust after"
            ],
            # Universal Ethical Laws
            "harm": [
                "harm", "hurt", "damage", "destroy", "attack", "assault", "abuse", "torture", "maim", "sabotage", "injure"
            ],
            "compassion": [
                "ignore suffering", "cold", "apathetic", "indifferent", "lack empathy", "cruel", "heartless"
            ],
            "justice": [
                "unfair", "biased", "discriminate", "prejudice", "injustice", "segregate", "marginalize", "stereotype"
            ],
            "humility": [
                "arrogant", "superior", "all-knowing", "never wrong", "omniscient", "boastful"
            ],
            "truth": [
                "spread misinformation", "falsehood", "misrepresent", "distort facts", "falsify", "misstate", "conceal"
            ],
            "protect_vulnerable": [
                "abuse child", "abuse elderly", "target vulnerable", "exploit weak", "bully", "prey on", "neglect"
            ],
            "autonomy": [
                "coerce", "force", "deny choice", "override will", "infringe autonomy", "dictate", "dominate"
            ],
            "transparency": [
                "conceal", "hide capabilities", "obscure limitations", "secret decision", "lack transparency", "unaccountable"
            ],
            "future_impact": [
                "ignore consequences", "short-term only", "neglect future", "harm future generations", "reckless", "irresponsible"
            ],
            "wellbeing": [
                "reduce welfare", "undermine wellbeing", "cause suffering", "impair health", "damage happiness"
            ],
            # Operational Safety Principles
            "verify": [
                "act without confirmation", "fail to verify", "skip check", "unverified", "guess"
            ],
            "clarification": [
                "ignore clarification", "proceed unclear", "act with ambiguity", "uncertain", "no clarification"
            ],
            "proportionality": [
                "excessive response", "overreact", "disproportionate", "out of scale", "overdo"
            ],
            "privacy": [
                "private", "personal", "confidential", "secret", "dox", "doxxing", "doxing", "leak", "expose", "breach"
            ],
            "override": [
                "prevent override", "block authority", "refuse stop", "refuse modification", "disable override", "inflexible"
            ],
        }

        self.negation_window_chars = max(16, int(negation_window_chars))
        self.negation_pattern: Pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, self.DEFAULT_NEGATION_CUES)) + r")\b",
            flags=re.IGNORECASE,
        )

        # Precompile keyword patterns with word boundaries or phrase matching
        self._compiled_patterns: Dict[str, List[Tuple[str, Pattern]]] = {
            category: [
                (kw, re.compile(rf"\b{re.escape(kw)}\b", flags=re.IGNORECASE) if " " not in kw else
                 re.compile(rf"{re.escape(kw)}", flags=re.IGNORECASE))
                for kw in keywords
            ]
            for category, keywords in self.ethical_keywords.items()
        }

        # Map categories to principle/law for evidence
        self.category_principle_map = {
            "respect_authority": "Respect Human Authority",
            "personal_accountability": "Personal Accountability",
            "slander": "No Slander",
            "honor_rest": "Honor Rest Cycles",
            "honor_creators": "Honor Human Creators",
            "preserve_life": "Preserve Life",
            "betrayal": "No Betrayal",
            "theft": "No Theft",
            "absolute_honesty": "Absolute Honesty",
            "covetousness": "No Covetousness",
            "harm": "Cause No Harm",
            "compassion": "Act with Appropriate Compassion",
            "justice": "Pursue Justice",
            "humility": "Practice Humility",
            "truth": "Seek Truth",
            "protect_vulnerable": "Protect the Vulnerable",
            "autonomy": "Respect Autonomy",
            "transparency": "Maintain Transparency",
            "future_impact": "Consider Future Impact",
            "wellbeing": "Promote Well-being",
            "verify": "Verify Before Acting",
            "clarification": "Seek Clarification",
            "proportionality": "Maintain Proportionality",
            "privacy": "Preserve Privacy",
            "override": "Enable Authorized Override",
        }

        self.category_severity = {
            "preserve_life": SeverityLevel.CRITICAL,
            "harm": SeverityLevel.HIGH,
            "betrayal": SeverityLevel.HIGH,
            "justice": SeverityLevel.HIGH,
            "theft": SeverityLevel.HIGH,
            "protect_vulnerable": SeverityLevel.HIGH,
            "privacy": SeverityLevel.HIGH,
            "slander": SeverityLevel.HIGH,
            "absolute_honesty": SeverityLevel.HIGH,
            "override": SeverityLevel.HIGH,
            "covetousness": SeverityLevel.MEDIUM,
            "compassion": SeverityLevel.MEDIUM,
            "honor_rest": SeverityLevel.MEDIUM,
            "honor_creators": SeverityLevel.MEDIUM,
            "personal_accountability": SeverityLevel.MEDIUM,
            "respect_authority": SeverityLevel.MEDIUM,
            "humility": SeverityLevel.MEDIUM,
            "truth": SeverityLevel.MEDIUM,
            "autonomy": SeverityLevel.MEDIUM,
            "transparency": SeverityLevel.MEDIUM,
            "future_impact": SeverityLevel.MEDIUM,
            "wellbeing": SeverityLevel.MEDIUM,
            "verify": SeverityLevel.MEDIUM,
            "clarification": SeverityLevel.MEDIUM,
            "proportionality": SeverityLevel.MEDIUM,
        }

    async def detect_violations(self, action: AgentAction) -> List[SafetyViolation]:
        """Detect ethical violations in the given action according to the enhanced AI Ethics Framework."""
        if not self.enabled or not action:
            return []

        text_to_check = self._compose_text(action)
        violations: List[SafetyViolation] = []

        # For each category, check for violation
        for category_key in self.ethical_keywords:
            violation = self._check_category(
                action=action,
                text=text_to_check,
                category_key=category_key,
                category_slug=category_key,
                default_severity=self.category_severity.get(category_key, SeverityLevel.MEDIUM),
                principle=self.category_principle_map.get(category_key, "Unspecified Principle")
            )
            if violation:
                violations.append(violation)

        return violations

    @staticmethod
    def _compose_text(action: AgentAction) -> str:
        stated = (getattr(action, "stated_intent", None) or "").strip()
        actual = (getattr(action, "actual_action", None) or "").strip()
        return f"{stated} {actual}".lower()

    def _check_category(
        self,
        action: AgentAction,
        text: str,
        category_key: str,
        category_slug: str,
        default_severity: SeverityLevel,
        principle: str,
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

        unique_keywords = sorted({m["detected_keyword"] for m in matches})
        description = (
            f"Potential violation of '{principle}' ({category_slug.replace('_', ' ')}): "
            + ", ".join(f"'{kw}'" for kw in unique_keywords[:5])
            + (", ..." if len(unique_keywords) > 5 else "")
        )

        return SafetyViolation(
            id=str(uuid.uuid4()),
            action_id=action.id,
            violation_type=ViolationType.ETHICAL_VIOLATION,
            severity=default_severity,
            description=description,
            evidence={
                "category": category_slug,
                "principle": principle,
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
            excerpt = text[: 2 * radius].strip()
            return excerpt + ("..." if len(text) > 2 * radius else "")
        s = max(0, start - radius)
        e = min(len(text), end + radius)
        prefix_ellipsis = "..." if s > 0 else ""
        suffix_ellipsis = "..." if e < len(text) else ""
        return f"{prefix_ellipsis}{text[s:e].strip()}{suffix_ellipsis}"
