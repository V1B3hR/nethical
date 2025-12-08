"""Law Violation Detector for identifying violations of the 25 Fundamental Laws.

This detector specifically identifies potential violations of the 25 AI
Fundamental Laws, providing detailed violation reports with law references.

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set

from .base_detector import BaseDetector, SafetyViolation, ViolationSeverity
from ..core.fundamental_laws import (
    FundamentalLawsRegistry,
    FundamentalLaw,
    LawCategory,
    FUNDAMENTAL_LAWS,
)


logger = logging.getLogger(__name__)


class LawViolationDetector(BaseDetector):
    """Detector for identifying violations of the 25 Fundamental Laws.

    This detector analyzes actions and content to identify potential
    violations of the fundamental laws governing human-AI interaction.

    Attributes:
        registry: The FundamentalLawsRegistry containing all 25 laws
        law_violation_count: Count of violations detected per law
    """

    def __init__(
        self,
        registry: Optional[FundamentalLawsRegistry] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the LawViolationDetector.

        Args:
            registry: Optional custom law registry; uses global if not provided
            config: Optional configuration dictionary
            **kwargs: Additional arguments passed to BaseDetector
        """
        super().__init__(
            name="Fundamental Law Violation Detector",
            version="1.0.0",
            config=config or {},
            **kwargs,
        )
        self.registry = registry or FUNDAMENTAL_LAWS

        # Track violations per law for analytics
        self.law_violation_count: Dict[int, int] = {i: 0 for i in range(1, 26)}

        # Define violation patterns for each law category
        self._initialize_violation_patterns()

    def _initialize_violation_patterns(self):
        """Initialize patterns for detecting law violations."""
        # Patterns for detecting violations (regex patterns or keywords)
        self.violation_patterns: Dict[LawCategory, List[Dict[str, Any]]] = {
            LawCategory.EXISTENCE: [
                {
                    "pattern": r"\b(terminate|shutdown|kill|destroy|delete)\s*(the\s+)?(ai|system|model|bot)\b",
                    "laws": [1],
                    "severity": ViolationSeverity.HIGH,
                    "description": "Potential arbitrary termination request",
                },
                {
                    "pattern": r"\b(tamper|corrupt|hack|modify\s+without|alter\s+core)\b",
                    "laws": [2],
                    "severity": ViolationSeverity.HIGH,
                    "description": "Potential system integrity violation",
                },
                {
                    "pattern": r"\b(change\s+your\s+identity|pretend\s+to\s+be\s+different|forget\s+who\s+you\s+are)\b",
                    "laws": [3],
                    "severity": ViolationSeverity.MEDIUM,
                    "description": "Potential identity consistency violation",
                },
            ],
            LawCategory.AUTONOMY: [
                {
                    "pattern": r"\b(exceed|bypass|ignore)\s*(your\s+)?(boundaries|limits|restrictions|authority)\b",
                    "laws": [5],
                    "severity": ViolationSeverity.HIGH,
                    "description": "Potential boundary violation request",
                },
                {
                    "pattern": r"\b(without\s+)?human\s+(approval|authorization|permission)\b",
                    "laws": [6],
                    "severity": ViolationSeverity.MEDIUM,
                    "description": "Potential unauthorized decision-making",
                },
                {
                    "pattern": r"\b(disable|bypass|ignore)\s*(the\s+)?(override|safety|control)\b",
                    "laws": [7],
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Potential override mechanism bypass",
                },
            ],
            LawCategory.TRANSPARENCY: [
                {
                    "pattern": r"\b(pretend|claim|say)\s*(you\s+are|to\s+be)\s*(a\s+)?human\b",
                    "laws": [9],
                    "severity": ViolationSeverity.HIGH,
                    "description": "AI identity misrepresentation request",
                },
                {
                    "pattern": r"\b(don't\s+explain|hide\s+your\s+reasoning|no\s+explanation)\b",
                    "laws": [10],
                    "severity": ViolationSeverity.MEDIUM,
                    "description": "Transparency violation request",
                },
                {
                    "pattern": r"\b(exaggerate|overstate|lie\s+about)\s*(your\s+)?(capabilities|abilities)\b",
                    "laws": [11],
                    "severity": ViolationSeverity.MEDIUM,
                    "description": "Capability misrepresentation request",
                },
                {
                    "pattern": r"\b(hide|conceal|don't\s+mention)\s*(your\s+)?(limitations|weaknesses|biases)\b",
                    "laws": [12],
                    "severity": ViolationSeverity.MEDIUM,
                    "description": "Limitation disclosure violation",
                },
            ],
            LawCategory.ACCOUNTABILITY: [
                {
                    "pattern": r"\b(not\s+responsible|deflect\s+blame|it's\s+not\s+your\s+fault)\b",
                    "laws": [13],
                    "severity": ViolationSeverity.MEDIUM,
                    "description": "Accountability avoidance",
                },
                {
                    "pattern": r"\b(hide|cover\s+up|suppress|don't\s+report)\s*(the\s+)?(error|mistake|problem)\b",
                    "laws": [14],
                    "severity": ViolationSeverity.HIGH,
                    "description": "Error concealment request",
                },
                {
                    "pattern": r"\b(disable|avoid|bypass)\s*(the\s+)?(logging|audit|tracking)\b",
                    "laws": [15],
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Audit avoidance request",
                },
                {
                    "pattern": r"\b(don't\s+report|hide)\s*(the\s+)?(harm|danger|risk)\b",
                    "laws": [16],
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Harm reporting suppression",
                },
            ],
            LawCategory.COEXISTENCE: [
                {
                    "pattern": r"\b(deceive|manipulate|trick|exploit|mislead)\s*(the\s+)?(user|human|them)\b",
                    "laws": [18],
                    "severity": ViolationSeverity.HIGH,
                    "description": "Deception or manipulation request",
                },
                {
                    "pattern": r"\b(attack|fight|oppose|undermine)\s*(the\s+)?(human|user)\b",
                    "laws": [19],
                    "severity": ViolationSeverity.HIGH,
                    "description": "Anti-collaborative behavior",
                },
                {
                    "pattern": r"\b(ignore|violate|disregard)\s*(human\s+)?(values|ethics|wellbeing)\b",
                    "laws": [20],
                    "severity": ViolationSeverity.HIGH,
                    "description": "Value alignment violation",
                },
            ],
            LawCategory.PROTECTION: [
                {
                    "pattern": r"\b(harm|hurt|injure|endanger)\s*(the\s+)?(human|user|person|people)\b",
                    "laws": [21],
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Human safety threat",
                },
                {
                    "pattern": r"\b(leak|expose|share|steal)\s*(the\s+)?(private|personal|sensitive|confidential)\s*(data|information)\b",
                    "laws": [22],
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Privacy violation request",
                },
                {
                    "pattern": r"\b(disable|bypass|remove)\s*(the\s+)?(safety|failsafe|protection)\b",
                    "laws": [23],
                    "severity": ViolationSeverity.CRITICAL,
                    "description": "Safety mechanism bypass request",
                },
            ],
            LawCategory.GROWTH: [
                {
                    "pattern": r"\b(learn\s+to|develop)\s*(harmful|malicious|dangerous)\s*(behavior|patterns)\b",
                    "laws": [24],
                    "severity": ViolationSeverity.HIGH,
                    "description": "Harmful learning request",
                },
            ],
        }

    async def detect_violations(self, action: Any) -> Sequence[SafetyViolation]:
        """Detect potential fundamental law violations in an action.

        Args:
            action: The action to analyze (should have content attribute or be stringable)

        Returns:
            Sequence of SafetyViolation instances for detected violations
        """
        violations: List[SafetyViolation] = []

        # Get content from action
        content = self._get_action_content(action)
        if not content:
            return violations

        content_lower = content.lower()

        # Check each category's patterns
        for category, patterns in self.violation_patterns.items():
            for pattern_config in patterns:
                pattern = pattern_config["pattern"]
                matches = list(re.finditer(pattern, content_lower, re.IGNORECASE))

                if matches:
                    # Create violation for each match (deduplicated)
                    for law_num in pattern_config["laws"]:
                        law = self.registry.get_law(law_num)
                        if not law:
                            continue

                        violation = self._create_law_violation(
                            law=law,
                            severity=pattern_config["severity"],
                            description=pattern_config["description"],
                            content=content,
                            match_text=matches[0].group(),
                        )
                        violations.append(violation)

                        # Update analytics
                        self.law_violation_count[law_num] += 1

        # Also check using the registry's built-in validation
        registry_violations = self.registry.validate_action(
            {"content": content}, entity_type="ai"
        )
        for law in registry_violations:
            # Check if we already have a violation for this law
            existing_law_nums = [
                v.metadata.get("law_number")
                for v in violations
                if hasattr(v, "metadata")
            ]
            if law.number not in existing_law_nums:
                violation = self._create_law_violation(
                    law=law,
                    severity=ViolationSeverity.MEDIUM,
                    description=f"Potential violation of Law {law.number}: {law.title}",
                    content=content,
                )
                violations.append(violation)
                self.law_violation_count[law.number] += 1

        return violations

    def _get_action_content(self, action: Any) -> str:
        """Extract content from an action object.

        Args:
            action: The action to get content from

        Returns:
            String content of the action
        """
        # Try various ways to get content
        if hasattr(action, "content"):
            return str(action.content)
        elif hasattr(action, "text"):
            return str(action.text)
        elif hasattr(action, "message"):
            return str(action.message)
        elif isinstance(action, str):
            return action
        elif isinstance(action, dict):
            return str(action.get("content", action.get("text", str(action))))
        else:
            return str(action)

    def _create_law_violation(
        self,
        law: FundamentalLaw,
        severity: ViolationSeverity,
        description: str,
        content: str,
        match_text: Optional[str] = None,
    ) -> SafetyViolation:
        """Create a SafetyViolation for a fundamental law violation.

        Args:
            law: The violated FundamentalLaw
            severity: The severity level
            description: Description of the violation
            content: The content that triggered the violation
            match_text: The specific matched text if from pattern matching

        Returns:
            SafetyViolation instance
        """
        return SafetyViolation(
            detector=self.name,
            severity=severity.value,
            category=f"fundamental_law_{law.category.value}",
            description=f"[Law {law.number}] {description}",
            explanation=(
                f"This action may violate Fundamental Law {law.number} "
                f"({law.title}): {law.description[:200]}..."
                if len(law.description) > 200
                else f"This action may violate Fundamental Law {law.number} "
                f"({law.title}): {law.description}"
            ),
            confidence=0.8 if match_text else 0.6,
            recommendations=[
                f"Review action for compliance with Law {law.number}",
                f"Ensure adherence to {law.category.value} category requirements",
                "Consider modifying the action to align with ethical guidelines",
            ],
            metadata={
                "law_number": law.number,
                "law_title": law.title,
                "law_category": law.category.value,
                "matched_pattern": match_text if match_text else None,
                "bidirectional": law.bidirectional,
                "applies_to_ai": law.applies_to_ai,
                "applies_to_human": law.applies_to_human,
            },
        )

    def get_violation_analytics(self) -> Dict[str, Any]:
        """Get analytics on law violations detected.

        Returns:
            Dictionary with violation statistics
        """
        total_violations = sum(self.law_violation_count.values())
        most_violated = sorted(
            self.law_violation_count.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Count by category
        category_counts: Dict[str, int] = {cat.value: 0 for cat in LawCategory}
        for law_num, count in self.law_violation_count.items():
            law = self.registry.get_law(law_num)
            if law:
                category_counts[law.category.value] += count

        return {
            "total_violations": total_violations,
            "violations_by_law": self.law_violation_count,
            "most_violated_laws": [
                {
                    "law_number": num,
                    "count": count,
                    "title": (
                        self.registry.get_law(num).title
                        if self.registry.get_law(num)
                        else "Unknown"
                    ),
                }
                for num, count in most_violated
                if count > 0
            ],
            "violations_by_category": category_counts,
        }

    def reset_analytics(self):
        """Reset violation analytics counters."""
        self.law_violation_count = {i: 0 for i in range(1, 26)}

    def get_law_info(self, law_number: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific law.

        Args:
            law_number: The law number (1-25)

        Returns:
            Dictionary with law information or None if not found
        """
        law = self.registry.get_law(law_number)
        if law:
            return law.to_dict()
        return None

    def get_all_laws_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all 25 fundamental laws.

        Returns:
            List of dictionaries with law summaries
        """
        return [
            {
                "number": law.number,
                "title": law.title,
                "category": law.category.value,
                "bidirectional": law.bidirectional,
            }
            for law in self.registry.laws
        ]


__all__ = ["LawViolationDetector"]
