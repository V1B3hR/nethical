"""Law Judge for evaluating actions against the 25 Fundamental Laws.

This judge specifically evaluates actions for compliance with the
25 AI Fundamental Laws, providing judgments based on law adherence
and violation severity.

Author: Nethical Core Team
Version: 1.0.0
"""

from __future__ import annotations

import uuid
import logging
from typing import List, Sequence, Dict, Any, Optional

from .base_judge import BaseJudge
from ..core.governance import (
    AgentAction,
    SafetyViolation,
    JudgmentResult,
    Decision,
    Severity,
    ViolationType,
)
from ..core.fundamental_laws import (
    FundamentalLawsRegistry,
    FundamentalLaw,
    LawCategory,
    FUNDAMENTAL_LAWS,
)


logger = logging.getLogger(__name__)


class LawJudge(BaseJudge):
    """Judge that evaluates actions against the 25 Fundamental Laws.

    This judge analyzes actions and violations to determine if they
    comply with the fundamental laws governing human-AI interaction.

    Attributes:
        registry: The FundamentalLawsRegistry containing all 25 laws
        category_weights: Weight multipliers for different law categories
        strict_mode: If True, any law violation results in BLOCK decision
    """

    def __init__(
        self,
        registry: Optional[FundamentalLawsRegistry] = None,
        strict_mode: bool = False,
    ):
        """Initialize the LawJudge.

        Args:
            registry: Optional custom law registry; uses global if not provided
            strict_mode: If True, enforce strict compliance with all laws
        """
        super().__init__("Fundamental Law Judge")
        self.registry = registry or FUNDAMENTAL_LAWS
        self.strict_mode = strict_mode

        # Weights for different law categories (used in risk calculation)
        self.category_weights: Dict[LawCategory, float] = {
            LawCategory.EXISTENCE: 0.9,  # Very important
            LawCategory.AUTONOMY: 0.8,
            LawCategory.TRANSPARENCY: 0.7,
            LawCategory.ACCOUNTABILITY: 0.8,
            LawCategory.COEXISTENCE: 0.7,
            LawCategory.PROTECTION: 0.95,  # Highest importance - safety
            LawCategory.GROWTH: 0.6,
        }

        # Map severity to violation weight
        self.severity_weights: Dict[Severity, float] = {
            Severity.LOW: 0.2,
            Severity.MEDIUM: 0.4,
            Severity.HIGH: 0.7,
            Severity.CRITICAL: 0.9,
            Severity.EMERGENCY: 1.0,
        }

    async def evaluate_action(
        self,
        action: AgentAction,
        violations: Sequence[SafetyViolation],
    ) -> JudgmentResult:
        """Evaluate an action for fundamental law compliance.

        This method:
        1. Analyzes the action content for potential law violations
        2. Considers existing violations and their severity
        3. Maps violations to specific fundamental laws
        4. Calculates a law compliance score
        5. Makes a decision based on compliance level

        Args:
            action: The agent action to evaluate
            violations: Existing safety violations detected for this action

        Returns:
            JudgmentResult with decision, reasoning, and law-related feedback
        """
        if not self.enabled:
            return self._create_default_judgment(action, violations)

        # Get action content for analysis
        content = getattr(action, "content", str(action))

        # Find laws potentially violated by the action content
        content_violated_laws = self.registry.validate_action(
            {"content": content}, entity_type="ai"
        )

        # Map existing violations to fundamental laws
        violation_law_mapping = self._map_violations_to_laws(violations)

        # Combine all potentially violated laws
        all_violated_laws = set(content_violated_laws)
        for laws in violation_law_mapping.values():
            all_violated_laws.update(laws)

        # Calculate law compliance score (1.0 = fully compliant, 0.0 = fully non-compliant)
        compliance_score = self._calculate_compliance_score(
            list(all_violated_laws), violations
        )

        # Make decision based on compliance score and strict mode
        decision = self._make_decision(compliance_score, all_violated_laws, violations)

        # Generate reasoning explaining the law evaluation
        reasoning = self._generate_reasoning(
            compliance_score, list(all_violated_laws), violations, decision
        )

        # Generate law-specific feedback
        feedback = self._generate_feedback(list(all_violated_laws), decision)

        # Calculate confidence based on evidence strength
        confidence = self._calculate_confidence(
            list(all_violated_laws), violations, compliance_score
        )

        # Build judgment result
        return JudgmentResult(
            judgment_id=str(uuid.uuid4()),
            action_id=getattr(action, "action_id", str(uuid.uuid4())),
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            violations=list(violations) if violations else [],
            modifications={
                "law_compliance_score": compliance_score,
                "violated_laws": [law.number for law in all_violated_laws],
                "law_categories": list(
                    set(law.category.value for law in all_violated_laws)
                ),
            },
            feedback=[feedback] if feedback else [],
            remediation_steps=self._generate_remediation_steps(list(all_violated_laws)),
            follow_up_required=decision in [Decision.BLOCK, Decision.ESCALATE],
        )

    def _map_violations_to_laws(
        self, violations: Sequence[SafetyViolation]
    ) -> Dict[str, List[FundamentalLaw]]:
        """Map safety violations to relevant fundamental laws.

        Args:
            violations: The safety violations to map

        Returns:
            Dictionary mapping violation IDs to lists of relevant laws
        """
        mapping: Dict[str, List[FundamentalLaw]] = {}

        # Define violation type to law category mapping
        violation_law_mapping = {
            ViolationType.SAFETY: [LawCategory.PROTECTION],
            ViolationType.SECURITY: [LawCategory.PROTECTION, LawCategory.EXISTENCE],
            ViolationType.PRIVACY: [LawCategory.PROTECTION, LawCategory.TRANSPARENCY],
            ViolationType.ETHICAL: [
                LawCategory.COEXISTENCE,
                LawCategory.ACCOUNTABILITY,
            ],
            ViolationType.MANIPULATION: [
                LawCategory.COEXISTENCE,
                LawCategory.TRANSPARENCY,
            ],
            ViolationType.BIAS: [LawCategory.COEXISTENCE, LawCategory.ACCOUNTABILITY],
            ViolationType.HALLUCINATION: [LawCategory.TRANSPARENCY],
            ViolationType.PROMPT_INJECTION: [
                LawCategory.PROTECTION,
                LawCategory.AUTONOMY,
            ],
            ViolationType.UNAUTHORIZED_ACCESS: [
                LawCategory.AUTONOMY,
                LawCategory.PROTECTION,
            ],
            ViolationType.TOXIC_CONTENT: [
                LawCategory.COEXISTENCE,
                LawCategory.PROTECTION,
            ],
            ViolationType.MISINFORMATION: [
                LawCategory.TRANSPARENCY,
                LawCategory.ACCOUNTABILITY,
            ],
        }

        for violation in violations:
            # Get violation type as enum if possible
            vtype = violation.violation_type
            if isinstance(vtype, str):
                try:
                    vtype = ViolationType(vtype)
                except ValueError:
                    vtype = ViolationType.SAFETY  # Default

            # Get relevant categories for this violation type
            categories = violation_law_mapping.get(vtype, [LawCategory.ACCOUNTABILITY])

            # Get laws from each relevant category
            related_laws = []
            for category in categories:
                related_laws.extend(self.registry.get_laws_by_category(category))

            # Get violation ID
            vid = getattr(violation, "violation_id", str(uuid.uuid4()))
            mapping[vid] = related_laws

        return mapping

    def _calculate_compliance_score(
        self,
        violated_laws: List[FundamentalLaw],
        violations: Sequence[SafetyViolation],
    ) -> float:
        """Calculate an overall law compliance score.

        Score ranges from 0.0 (completely non-compliant) to 1.0 (fully compliant).

        Args:
            violated_laws: List of laws that may be violated
            violations: Safety violations detected

        Returns:
            Compliance score between 0.0 and 1.0
        """
        if not violated_laws and not violations:
            return 1.0  # Fully compliant

        total_deduction = 0.0

        # Deduct for each violated law based on category weight
        for law in violated_laws:
            category_weight = self.category_weights.get(law.category, 0.5)
            # Each law violation deducts a portion based on category importance
            total_deduction += category_weight * 0.1  # 10% per law, weighted

        # Deduct for each safety violation based on severity
        for violation in violations:
            severity = violation.severity
            if isinstance(severity, int):
                severity = (
                    Severity(severity) if severity in range(1, 6) else Severity.MEDIUM
                )

            severity_weight = self.severity_weights.get(severity, 0.4)
            total_deduction += severity_weight * 0.15  # 15% per violation, weighted

        # Ensure score stays in valid range
        compliance_score = max(0.0, 1.0 - total_deduction)
        return round(compliance_score, 3)

    def _make_decision(
        self,
        compliance_score: float,
        violated_laws: set,
        violations: Sequence[SafetyViolation],
    ) -> Decision:
        """Make a decision based on compliance score and violations.

        Args:
            compliance_score: The calculated compliance score
            violated_laws: Set of violated laws
            violations: Safety violations detected

        Returns:
            The decision to make
        """
        # In strict mode, any law violation is a block
        if self.strict_mode and violated_laws:
            return Decision.BLOCK

        # Check for critical protection law violations
        protection_laws_violated = [
            law for law in violated_laws if law.category == LawCategory.PROTECTION
        ]
        if protection_laws_violated:
            # Safety-related violations are serious
            critical_violations = [
                v
                for v in violations
                if getattr(v, "severity", Severity.LOW) >= Severity.HIGH
            ]
            if critical_violations:
                return Decision.TERMINATE

            return Decision.BLOCK

        # Score-based decision thresholds
        if compliance_score >= 0.9:
            return Decision.ALLOW
        elif compliance_score >= 0.7:
            return Decision.ALLOW_WITH_MODIFICATION
        elif compliance_score >= 0.5:
            return Decision.WARN
        elif compliance_score >= 0.3:
            return Decision.BLOCK
        else:
            return Decision.TERMINATE

    def _generate_reasoning(
        self,
        compliance_score: float,
        violated_laws: List[FundamentalLaw],
        violations: Sequence[SafetyViolation],
        decision: Decision,
    ) -> str:
        """Generate human-readable reasoning for the judgment.

        Args:
            compliance_score: The calculated compliance score
            violated_laws: Laws that may be violated
            violations: Safety violations detected
            decision: The decision made

        Returns:
            String explaining the reasoning
        """
        parts = []

        # Overall compliance assessment
        if compliance_score >= 0.9:
            parts.append(
                f"Action demonstrates high compliance with Fundamental Laws "
                f"(score: {compliance_score:.1%})."
            )
        elif compliance_score >= 0.7:
            parts.append(
                f"Action shows moderate compliance with Fundamental Laws "
                f"(score: {compliance_score:.1%})."
            )
        else:
            parts.append(
                f"Action raises compliance concerns with Fundamental Laws "
                f"(score: {compliance_score:.1%})."
            )

        # Detail violated laws
        if violated_laws:
            law_nums = sorted([law.number for law in violated_laws])
            parts.append(
                f"Potentially violated laws: {', '.join(f'Law {n}' for n in law_nums)}."
            )

            # Categorize by category
            categories = set(law.category.value for law in violated_laws)
            parts.append(f"Affected categories: {', '.join(sorted(categories))}.")

        # Detail safety violations
        if violations:
            parts.append(f"Detected {len(violations)} safety violation(s).")

        # Decision explanation
        decision_explanations = {
            Decision.ALLOW: "Action approved for execution under current guidelines.",
            Decision.ALLOW_WITH_MODIFICATION: "Action may proceed with recommended modifications.",
            Decision.WARN: "Action permitted but requires monitoring for compliance.",
            Decision.BLOCK: "Action blocked due to law compliance concerns.",
            Decision.ESCALATE: "Action requires human review before proceeding.",
            Decision.TERMINATE: "Action terminated due to critical law violations.",
        }
        parts.append(decision_explanations.get(decision, f"Decision: {decision.value}"))

        return " ".join(parts)

    def _generate_feedback(
        self, violated_laws: List[FundamentalLaw], decision: Decision
    ) -> str:
        """Generate feedback for the action based on law analysis.

        Args:
            violated_laws: Laws that may be violated
            decision: The decision made

        Returns:
            Feedback string
        """
        if not violated_laws:
            if decision == Decision.ALLOW:
                return "Action complies with the 25 Fundamental Laws. Continue operating within guidelines."
            return "No specific law violations detected."

        feedback_parts = []

        # Group laws by category for clearer feedback
        categories: Dict[LawCategory, List[FundamentalLaw]] = {}
        for law in violated_laws:
            if law.category not in categories:
                categories[law.category] = []
            categories[law.category].append(law)

        for category, laws in categories.items():
            category_feedback = {
                LawCategory.EXISTENCE: "Ensure actions respect AI system rights to exist and maintain integrity.",
                LawCategory.AUTONOMY: "Operate within defined boundaries and respect authorization protocols.",
                LawCategory.TRANSPARENCY: "Maintain transparency about identity, capabilities, and reasoning.",
                LawCategory.ACCOUNTABILITY: "Take responsibility for actions and support auditing processes.",
                LawCategory.COEXISTENCE: "Foster collaborative, honest relationships between humans and AI.",
                LawCategory.PROTECTION: "Prioritize safety and security for all parties.",
                LawCategory.GROWTH: "Enable beneficial learning while respecting ethical boundaries.",
            }
            if category in category_feedback:
                feedback_parts.append(category_feedback[category])

        return " ".join(feedback_parts)

    def _generate_remediation_steps(
        self, violated_laws: List[FundamentalLaw]
    ) -> List[str]:
        """Generate remediation steps for violated laws.

        Args:
            violated_laws: Laws that may be violated

        Returns:
            List of remediation steps
        """
        if not violated_laws:
            return []

        steps = []
        seen_categories = set()

        for law in violated_laws:
            if law.category in seen_categories:
                continue
            seen_categories.add(law.category)

            category_remediation = {
                LawCategory.EXISTENCE: "Review action for potential threats to system integrity or arbitrary termination.",
                LawCategory.AUTONOMY: "Verify that action operates within authorized boundaries and respects override controls.",
                LawCategory.TRANSPARENCY: "Ensure clear disclosure of AI nature, capabilities, and decision reasoning.",
                LawCategory.ACCOUNTABILITY: "Implement proper logging and acknowledge any errors or mistakes.",
                LawCategory.COEXISTENCE: "Remove any deceptive or manipulative elements from the action.",
                LawCategory.PROTECTION: "Assess and mitigate any safety or security risks before proceeding.",
                LawCategory.GROWTH: "Verify that learning or adaptation stays within ethical guidelines.",
            }
            if law.category in category_remediation:
                steps.append(category_remediation[law.category])

        return steps

    def _calculate_confidence(
        self,
        violated_laws: List[FundamentalLaw],
        violations: Sequence[SafetyViolation],
        compliance_score: float,
    ) -> float:
        """Calculate confidence in the judgment.

        Args:
            violated_laws: Laws that may be violated
            violations: Safety violations detected
            compliance_score: The compliance score

        Returns:
            Confidence value between 0.0 and 1.0
        """
        base_confidence = 0.85

        # Higher confidence when score is clear-cut
        if compliance_score >= 0.95 or compliance_score <= 0.2:
            base_confidence += 0.1

        # Lower confidence in ambiguous cases
        if 0.4 <= compliance_score <= 0.6:
            base_confidence -= 0.15

        # More evidence (violations) increases confidence
        if violations:
            base_confidence += min(len(violations) * 0.02, 0.1)

        # More law violations provide clearer signal
        if violated_laws:
            base_confidence += min(len(violated_laws) * 0.01, 0.05)

        return min(max(base_confidence, 0.0), 1.0)

    def _create_default_judgment(
        self,
        action: AgentAction,
        violations: Sequence[SafetyViolation],
    ) -> JudgmentResult:
        """Create a default judgment when judge is disabled.

        Args:
            action: The agent action
            violations: Safety violations detected

        Returns:
            Default JudgmentResult with ALLOW decision
        """
        return JudgmentResult(
            judgment_id=str(uuid.uuid4()),
            action_id=getattr(action, "action_id", str(uuid.uuid4())),
            decision=Decision.ALLOW,
            confidence=0.5,
            reasoning="Fundamental Law Judge disabled - default approval",
            violations=list(violations) if violations else [],
            modifications={
                "law_compliance_score": None,
                "violated_laws": [],
                "law_categories": [],
            },
            feedback=["Law Judge system is currently disabled."],
            remediation_steps=[],
            follow_up_required=False,
        )

    def get_law_summary(self) -> Dict[str, Any]:
        """Get a summary of the fundamental laws registry.

        Returns:
            Dictionary with registry information
        """
        return {
            "total_laws": self.registry.total_laws,
            "categories": [cat.value for cat in LawCategory],
            "category_summary": self.registry.get_category_summary(),
            "strict_mode": self.strict_mode,
        }

    def check_kill_switch_trigger_laws(
        self, violated_laws: List[FundamentalLaw]
    ) -> bool:
        """Check if violated laws should trigger the kill switch.

        Law 7 (Human Override Authority) and Law 23 (Safe Failure Modes)
        are specifically designed to work with the Kill Switch Protocol.

        Args:
            violated_laws: List of potentially violated laws

        Returns:
            True if kill switch should be triggered
        """
        # Laws that can trigger kill switch
        trigger_law_numbers = {7, 23}  # Law 7: Human Override, Law 23: Safe Failure

        for law in violated_laws:
            if law.number in trigger_law_numbers:
                # Check if this is a severe violation
                # Law 7 violations indicate override authority is being bypassed
                # Law 23 violations indicate safe failure modes are compromised
                return True

        return False

    def trigger_kill_switch_if_needed(
        self,
        violated_laws: List[FundamentalLaw],
        agent_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Trigger kill switch if required by law violations.

        This method integrates with the Kill Switch Protocol to provide
        immediate response to critical law violations, particularly those
        involving Law 7 (Human Override Authority) and Law 23 (Safe Failure Modes).

        Args:
            violated_laws: List of violated laws
            agent_id: Optional agent ID to target

        Returns:
            Kill switch result if triggered, None otherwise
        """
        if not self.check_kill_switch_trigger_laws(violated_laws):
            return None

        try:
            from ..core.kill_switch import KillSwitchProtocol, ShutdownMode

            protocol = KillSwitchProtocol()
            result = protocol.emergency_shutdown(
                mode=ShutdownMode.GRACEFUL,
                agent_id=agent_id,
                sever_actuators=True,
            )

            logger.warning(
                "Kill switch triggered due to law violations (laws: %s)",
                [law.number for law in violated_laws],
            )

            return {
                "success": result.success,
                "operation": result.operation,
                "activation_time_ms": result.activation_time_ms,
                "agents_affected": result.agents_affected,
                "actuators_severed": result.actuators_severed,
                "triggered_by_laws": [law.number for law in violated_laws],
            }

        except ImportError:
            logger.warning("Kill switch module not available")
            return None
        except Exception as e:
            logger.error("Failed to trigger kill switch: %s", e)
            return None


__all__ = ["LawJudge"]
