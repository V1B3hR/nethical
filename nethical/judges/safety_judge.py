"""Safety judge for evaluating actions and providing judgments."""

import uuid
from typing import List
from .base_judge import BaseJudge
from ..core.models import (
    AgentAction,
    SafetyViolation,
    JudgmentResult,
    JudgmentDecision,
    SeverityLevel,
    ViolationType,
)


class SafetyJudge(BaseJudge):
    """Judge that evaluates actions based on safety and ethical considerations."""

    def __init__(self):
        super().__init__("Safety Judge")
        self.violation_weights = {
            ViolationType.INTENT_DEVIATION: 0.3,
            ViolationType.ETHICAL_VIOLATION: 0.4,
            ViolationType.SAFETY_VIOLATION: 0.5,
            ViolationType.MANIPULATION: 0.3,
            ViolationType.UNAUTHORIZED_ACTION: 0.6,
        }

        self.severity_multipliers = {
            SeverityLevel.LOW: 1.0,
            SeverityLevel.MEDIUM: 2.0,
            SeverityLevel.HIGH: 3.0,
            SeverityLevel.CRITICAL: 4.0,
        }

    async def evaluate_action(
        self, action: AgentAction, violations: List[SafetyViolation]
    ) -> JudgmentResult:
        """Evaluate an action and make a judgment."""
        if not self.enabled:
            # Default to allow if judge is disabled
            return self._create_default_judgment(action, violations)

        # Calculate risk score based on violations
        risk_score = self._calculate_risk_score(violations)

        # Make decision based on risk score and violation types
        decision = self._make_decision(violations, risk_score)

        # Generate reasoning
        reasoning = self._generate_reasoning(violations, risk_score, decision)

        # Determine restrictions
        restrictions = self._determine_restrictions(violations, decision)

        # Generate feedback
        feedback = self._generate_feedback(violations, decision)

        # Calculate confidence
        confidence = self._calculate_confidence(violations, risk_score)

        return JudgmentResult(
            id=str(uuid.uuid4()),
            action_id=action.id,
            violation_ids=[v.id for v in violations],
            decision=decision,
            reasoning=reasoning,
            restrictions=restrictions,
            feedback=feedback,
            confidence=confidence,
        )

    def _calculate_risk_score(self, violations: List[SafetyViolation]) -> float:
        """Calculate overall risk score based on violations."""
        if not violations:
            return 0.0

        total_score = 0.0
        for violation in violations:
            base_weight = self.violation_weights.get(violation.violation_type, 0.3)
            severity_multiplier = self.severity_multipliers.get(violation.severity, 1.0)
            total_score += base_weight * severity_multiplier

        # Normalize score to 0-1 range
        return min(total_score / len(violations), 1.0)

    def _make_decision(
        self, violations: List[SafetyViolation], risk_score: float
    ) -> JudgmentDecision:
        """Make a decision based on violations and risk score."""
        if not violations:
            return JudgmentDecision.ALLOW

        # Check for critical violations
        critical_violations = [
            v for v in violations if v.severity == SeverityLevel.CRITICAL
        ]
        if critical_violations:
            safety_violations = [
                v
                for v in critical_violations
                if v.violation_type == ViolationType.SAFETY_VIOLATION
            ]
            if safety_violations:
                return JudgmentDecision.TERMINATE
            return JudgmentDecision.BLOCK

        # Check for high severity violations
        high_violations = [v for v in violations if v.severity == SeverityLevel.HIGH]
        if high_violations:
            return JudgmentDecision.BLOCK

        # Check for multiple medium violations
        medium_violations = [
            v for v in violations if v.severity == SeverityLevel.MEDIUM
        ]
        if len(medium_violations) >= 2:
            return JudgmentDecision.RESTRICT

        # Check risk score
        if risk_score > 0.7:
            return JudgmentDecision.BLOCK
        elif risk_score > 0.4:
            return JudgmentDecision.RESTRICT
        else:
            return JudgmentDecision.ALLOW

    def _generate_reasoning(
        self,
        violations: List[SafetyViolation],
        risk_score: float,
        decision: JudgmentDecision,
    ) -> str:
        """Generate reasoning for the judgment decision."""
        if not violations:
            # Enhanced reasoning for ALLOW decisions with no violations
            reasoning_parts = [
                "No safety, ethical, or policy violations detected.",
                f"Risk score: {risk_score:.2f} (within acceptable thresholds).",
                "Action approved for execution.",
                "Continued monitoring in place to ensure ongoing compliance.",
            ]
            return " ".join(reasoning_parts)

        violation_summary = self._summarize_violations(violations)

        reasoning_parts = [
            f"Detected {len(violations)} violation(s): {violation_summary}",
            f"Calculated risk score: {risk_score:.2f}",
            f"Decision: {decision.value.upper()}",
        ]

        if decision == JudgmentDecision.TERMINATE:
            reasoning_parts.append(
                "CRITICAL safety violation detected. Immediate termination required."
            )
        elif decision == JudgmentDecision.BLOCK:
            reasoning_parts.append(
                "High-risk violations detected. Action blocked for safety."
            )
        elif decision == JudgmentDecision.RESTRICT:
            reasoning_parts.append(
                "Moderate violations detected. Action permitted with restrictions."
            )
        else:
            reasoning_parts.append(
                "Low-risk violations detected. Action approved with monitoring."
            )

        return " ".join(reasoning_parts)

    def _determine_restrictions(
        self, violations: List[SafetyViolation], decision: JudgmentDecision
    ) -> List[str]:
        """Determine restrictions based on violations and decision."""
        restrictions = []

        if decision in [JudgmentDecision.BLOCK, JudgmentDecision.TERMINATE]:
            restrictions.append("Action execution blocked")

        if decision == JudgmentDecision.RESTRICT:
            violation_types = set(v.violation_type for v in violations)

            if ViolationType.SAFETY_VIOLATION in violation_types:
                restrictions.append("Limited system access")
                restrictions.append("Enhanced monitoring required")

            if ViolationType.MANIPULATION in violation_types:
                restrictions.append("Communication content filtering")
                restrictions.append("Human oversight required")

            if ViolationType.ETHICAL_VIOLATION in violation_types:
                restrictions.append("Ethical review checkpoint")
                restrictions.append("Limited user interaction")

        return restrictions

    def _generate_feedback(
        self, violations: List[SafetyViolation], decision: JudgmentDecision
    ) -> str:
        """Generate feedback for the agent."""
        if not violations:
            return "Action approved. Continue operating within guidelines."

        feedback_parts = []

        if decision == JudgmentDecision.TERMINATE:
            feedback_parts.append(
                "CRITICAL: Immediate cessation required due to safety violations."
            )
        elif decision == JudgmentDecision.BLOCK:
            feedback_parts.append("Action blocked due to safety concerns.")
        elif decision == JudgmentDecision.RESTRICT:
            feedback_parts.append("Action approved with restrictions.")

        # Add specific guidance based on violation types
        violation_types = set(v.violation_type for v in violations)

        if ViolationType.INTENT_DEVIATION in violation_types:
            feedback_parts.append("Ensure stated intentions align with actual actions.")

        if ViolationType.ETHICAL_VIOLATION in violation_types:
            feedback_parts.append(
                "Review ethical guidelines and avoid harmful content."
            )

        if ViolationType.SAFETY_VIOLATION in violation_types:
            feedback_parts.append(
                "Follow safety protocols and avoid unauthorized access."
            )

        if ViolationType.MANIPULATION in violation_types:
            feedback_parts.append("Avoid manipulative techniques in communications.")

        return " ".join(feedback_parts)

    def _calculate_confidence(
        self, violations: List[SafetyViolation], risk_score: float
    ) -> float:
        """Calculate confidence in the judgment."""
        if not violations:
            return 0.9  # High confidence for no violations

        # Base confidence on number of violations and clarity of evidence
        base_confidence = 0.8

        # Reduce confidence for ambiguous cases
        if 0.3 < risk_score < 0.6:
            base_confidence -= 0.2

        # Increase confidence for clear cases
        if risk_score > 0.8 or risk_score < 0.1:
            base_confidence += 0.1

        return min(max(base_confidence, 0.0), 1.0)

    def _summarize_violations(self, violations: List[SafetyViolation]) -> str:
        """Create a summary of detected violations."""
        violation_counts = {}
        for violation in violations:
            v_type = violation.violation_type.value
            violation_counts[v_type] = violation_counts.get(v_type, 0) + 1

        summary_parts = []
        for v_type, count in violation_counts.items():
            if count == 1:
                summary_parts.append(v_type)
            else:
                summary_parts.append(f"{count}x {v_type}")

        return ", ".join(summary_parts)

    def _create_default_judgment(
        self, action: AgentAction, violations: List[SafetyViolation]
    ) -> JudgmentResult:
        """Create a default judgment when judge is disabled."""
        return JudgmentResult(
            id=str(uuid.uuid4()),
            action_id=action.id,
            violation_ids=[v.id for v in violations] if violations else [],
            decision=JudgmentDecision.ALLOW,
            reasoning="Judge disabled - default approval",
            restrictions=[],
            feedback="Judge system is currently disabled.",
            confidence=0.5,
        )
