"""Explainable AI Layer for Phase 2.3.

This module implements:
- Decision explanations for policy engine
- Natural language explanation generation
- Decision tree visualization
- Transparency reporting
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ExplanationType(Enum):
    """Types of explanations."""

    RULE_BASED = "rule_based"
    ML_FEATURE_IMPORTANCE = "ml_feature_importance"
    POLICY_MATCH = "policy_match"
    RISK_ASSESSMENT = "risk_assessment"
    ETHICAL_DIMENSION = "ethical_dimension"


@dataclass
class DecisionExplanation:
    """Explanation for a decision."""

    decision: str
    explanation_type: ExplanationType
    primary_reason: str
    contributing_factors: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    rules_matched: List[str] = field(default_factory=list)
    threshold_comparisons: Dict[str, Any] = field(default_factory=dict)
    natural_language: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationComponent:
    """A single component of an explanation."""

    name: str
    value: Any
    weight: float
    description: str
    category: str = "general"


class DecisionExplainer:
    """Generates explanations for decisions."""

    # Configuration constants
    DEFAULT_RISK_THRESHOLD = 0.7
    DEFAULT_VIOLATION_THRESHOLD = 0

    def __init__(self, risk_threshold: float = DEFAULT_RISK_THRESHOLD):
        """Initialize decision explainer.

        Args:
            risk_threshold: Risk score threshold for highlighting
        """
        self.risk_threshold = risk_threshold
        self.explanation_templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load natural language templates.

        Returns:
            Dictionary of templates
        """
        return {
            "BLOCK": "Action was BLOCKED because {primary_reason}. {additional_details}",
            "RESTRICT": "Action was RESTRICTED due to {primary_reason}. {additional_details}",
            "ALLOW": "Action was ALLOWED. {additional_details}",
            "TERMINATE": "Session was TERMINATED because {primary_reason}. {additional_details}",
            "violation_detected": "A {severity} severity {violation_type} violation was detected",
            "policy_match": "Policy rule '{rule_id}' matched with confidence {confidence:.2%}",
            "threshold_exceeded": "{metric} ({value:.2f}) exceeded threshold ({threshold:.2f})",
            "ethical_dimension": "Ethical dimension '{dimension}' scored {score:.2f}",
            "risk_score": "Risk score of {score:.2f} exceeds acceptable threshold of {threshold:.2f}",
        }

    def explain_decision(
        self, decision: str, judgment_data: Dict[str, Any], include_ml_explanation: bool = False
    ) -> DecisionExplanation:
        """Generate explanation for a decision.

        Args:
            decision: Decision made (ALLOW, RESTRICT, BLOCK, TERMINATE)
            judgment_data: Data from judgment process
            include_ml_explanation: Whether to include ML feature importance

        Returns:
            Decision explanation
        """
        # Determine explanation type
        explanation_type = ExplanationType.POLICY_MATCH

        # Extract primary reason
        primary_reason = self._extract_primary_reason(decision, judgment_data)

        # Extract contributing factors
        contributing_factors = self._extract_contributing_factors(judgment_data)

        # Get matched rules
        rules_matched = judgment_data.get("matched_rules", [])
        if isinstance(rules_matched, list) and rules_matched and isinstance(rules_matched[0], dict):
            rules_matched = [r.get("id", str(r)) for r in rules_matched]

        # Get threshold comparisons
        threshold_comparisons = self._extract_threshold_comparisons(judgment_data)

        # Generate natural language explanation
        natural_language = self._generate_natural_language(
            decision=decision,
            primary_reason=primary_reason,
            contributing_factors=contributing_factors,
            rules_matched=rules_matched,
            threshold_comparisons=threshold_comparisons,
        )

        return DecisionExplanation(
            decision=decision,
            explanation_type=explanation_type,
            primary_reason=primary_reason,
            contributing_factors=contributing_factors,
            rules_matched=rules_matched,
            threshold_comparisons=threshold_comparisons,
            natural_language=natural_language,
            metadata=judgment_data,
        )

    def _extract_primary_reason(self, decision: str, judgment_data: Dict[str, Any]) -> str:
        """Extract primary reason for decision.

        Args:
            decision: Decision made
            judgment_data: Judgment data

        Returns:
            Primary reason string
        """
        # Check for violations
        violations = judgment_data.get("violations", [])
        if violations:
            if isinstance(violations, list) and violations:
                first_violation = violations[0]
                if isinstance(first_violation, dict):
                    return first_violation.get("type", "unknown violation")
                return str(first_violation)

        # Check for policy matches
        matched_rules = judgment_data.get("matched_rules", [])
        if matched_rules:
            if isinstance(matched_rules, list) and matched_rules:
                first_rule = matched_rules[0]
                if isinstance(first_rule, dict):
                    rule_id = first_rule.get("id", "unknown rule")
                    return f"policy rule '{rule_id}' matched"
                return f"policy rule matched"

        # Check for risk score
        risk_score = judgment_data.get("risk_score")
        if risk_score and risk_score > self.risk_threshold:
            return f"high risk score ({risk_score:.2f})"

        # Default reasons
        if decision == "ALLOW":
            return "no violations detected and all checks passed"
        elif decision == "RESTRICT":
            return "minor policy concerns detected"
        elif decision == "BLOCK":
            return "policy violation detected"
        elif decision == "TERMINATE":
            return "critical security violation"

        return "standard governance evaluation"

    def _extract_contributing_factors(self, judgment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract contributing factors from judgment data.

        Args:
            judgment_data: Judgment data

        Returns:
            List of contributing factors
        """
        factors = []

        # Add risk score
        if "risk_score" in judgment_data:
            factors.append(
                {
                    "name": "Risk Score",
                    "value": judgment_data["risk_score"],
                    "weight": 0.3,
                    "category": "risk",
                }
            )

        # Add violations
        violations = judgment_data.get("violations", [])
        if violations:
            factors.append(
                {
                    "name": "Violations Detected",
                    "value": len(violations),
                    "weight": 0.4,
                    "category": "violation",
                }
            )

        # Add ethical dimensions
        ethical_tags = judgment_data.get("ethical_tags", {})
        if ethical_tags:
            factors.append(
                {
                    "name": "Ethical Dimensions",
                    "value": len(ethical_tags),
                    "weight": 0.2,
                    "category": "ethics",
                }
            )

        # Add escalation flag
        if judgment_data.get("escalate", False):
            factors.append(
                {
                    "name": "Escalation Required",
                    "value": True,
                    "weight": 0.1,
                    "category": "escalation",
                }
            )

        return factors

    def _extract_threshold_comparisons(self, judgment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract threshold comparisons.

        Args:
            judgment_data: Judgment data

        Returns:
            Threshold comparisons
        """
        comparisons = {}

        # Risk score threshold
        risk_score = judgment_data.get("risk_score")
        if risk_score is not None:
            comparisons["risk_score"] = {
                "value": risk_score,
                "threshold": self.risk_threshold,
                "exceeded": risk_score > self.risk_threshold,
            }

        # Violation count
        violations = judgment_data.get("violations", [])
        if violations:
            comparisons["violation_count"] = {
                "value": len(violations),
                "threshold": self.DEFAULT_VIOLATION_THRESHOLD,
                "exceeded": len(violations) > self.DEFAULT_VIOLATION_THRESHOLD,
            }

        return comparisons

    def _generate_natural_language(
        self,
        decision: str,
        primary_reason: str,
        contributing_factors: List[Dict[str, Any]],
        rules_matched: List[str],
        threshold_comparisons: Dict[str, Any],
    ) -> str:
        """Generate natural language explanation.

        Args:
            decision: Decision made
            primary_reason: Primary reason
            contributing_factors: Contributing factors
            rules_matched: Matched rules
            threshold_comparisons: Threshold comparisons

        Returns:
            Natural language explanation
        """
        # Start with base template
        base_template = self.explanation_templates.get(decision, "Action resulted in {decision}.")

        # Build additional details
        additional_parts = []

        # Add contributing factors
        if contributing_factors:
            factor_summaries = []
            for factor in contributing_factors:
                if factor.get("category") == "violation":
                    factor_summaries.append(f"{factor['value']} violation(s) were identified")
                elif factor.get("category") == "risk":
                    factor_summaries.append(f"risk score of {factor['value']:.2f}")
                elif factor.get("category") == "ethics":
                    factor_summaries.append(f"{factor['value']} ethical dimension(s) flagged")

            if factor_summaries:
                additional_parts.append(
                    "Contributing factors include: " + ", ".join(factor_summaries)
                )

        # Add matched rules
        if rules_matched:
            rule_list = ", ".join(rules_matched[:3])
            if len(rules_matched) > 3:
                rule_list += f" and {len(rules_matched) - 3} more"
            additional_parts.append(f"Matched policy rules: {rule_list}")

        # Add threshold information
        if threshold_comparisons:
            for metric, data in threshold_comparisons.items():
                if data.get("exceeded"):
                    additional_parts.append(
                        f"{metric.replace('_', ' ').title()} "
                        f"({data['value']:.2f}) exceeded threshold ({data['threshold']:.2f})"
                    )

        additional_details = (
            ". ".join(additional_parts) if additional_parts else "No additional details available"
        )

        # Format final explanation
        explanation = base_template.format(
            primary_reason=primary_reason, additional_details=additional_details
        )

        return explanation

    def explain_policy_match(self, matched_rule: Dict[str, Any], facts: Dict[str, Any]) -> str:
        """Explain why a policy rule matched.

        Args:
            matched_rule: Matched rule details
            facts: Input facts

        Returns:
            Explanation string
        """
        rule_id = matched_rule.get("id", "unknown")
        priority = matched_rule.get("priority", 0)
        decision = matched_rule.get("decision", "unknown")

        explanation = f"Policy rule '{rule_id}' (priority {priority}) matched, resulting in decision: {decision}. "

        # Add condition information if available
        when_condition = matched_rule.get("when", {})
        if when_condition:
            explanation += "The rule condition evaluated to true based on the provided facts. "

        return explanation

    def generate_decision_tree_viz(self, judgment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate decision tree visualization data.

        Args:
            judgment_data: Judgment data

        Returns:
            Visualization data structure
        """
        # Create tree structure
        tree = {
            "name": "Decision Process",
            "decision": judgment_data.get("decision", "UNKNOWN"),
            "children": [],
        }

        # Add violation check branch
        violations = judgment_data.get("violations", [])
        if violations:
            violation_node = {
                "name": "Violation Detection",
                "value": len(violations),
                "children": [
                    {
                        "name": v if isinstance(v, str) else v.get("type", "unknown"),
                        "severity": "HIGH",
                    }
                    for v in violations[:5]  # Limit to first 5
                ],
            }
            tree["children"].append(violation_node)

        # Add policy check branch
        matched_rules = judgment_data.get("matched_rules", [])
        if matched_rules:
            policy_node = {
                "name": "Policy Evaluation",
                "value": len(matched_rules),
                "children": [
                    {
                        "name": r.get("id", str(r)) if isinstance(r, dict) else str(r),
                        "priority": r.get("priority", 0) if isinstance(r, dict) else 0,
                    }
                    for r in matched_rules[:5]
                ],
            }
            tree["children"].append(policy_node)

        # Add risk assessment branch
        if "risk_score" in judgment_data:
            risk_node = {
                "name": "Risk Assessment",
                "value": judgment_data["risk_score"],
                "threshold": 0.7,
            }
            tree["children"].append(risk_node)

        return tree


class TransparencyReportGenerator:
    """Generates transparency reports for decisions."""

    def __init__(self):
        """Initialize transparency report generator."""
        self.explainer = DecisionExplainer()

    def generate_report(
        self, decisions: List[Dict[str, Any]], time_period: str = "recent"
    ) -> Dict[str, Any]:
        """Generate transparency report.

        Args:
            decisions: List of decisions to report on
            time_period: Time period description

        Returns:
            Transparency report
        """
        if not decisions:
            return {
                "summary": "No decisions to report",
                "time_period": time_period,
                "total_decisions": 0,
            }

        # Count decisions by type
        decision_counts = {}
        for decision_data in decisions:
            decision = decision_data.get("decision", "UNKNOWN")
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        # Count violations
        total_violations = 0
        violation_types = {}
        for decision_data in decisions:
            violations = decision_data.get("violations", [])
            total_violations += len(violations)
            for violation in violations:
                vtype = (
                    violation if isinstance(violation, str) else violation.get("type", "unknown")
                )
                violation_types[vtype] = violation_types.get(vtype, 0) + 1

        # Get most common explanations
        explanations = []
        for decision_data in decisions[:10]:  # Sample first 10
            explanation = self.explainer.explain_decision(
                decision=decision_data.get("decision", "UNKNOWN"), judgment_data=decision_data
            )
            explanations.append(
                {
                    "decision": explanation.decision,
                    "reason": explanation.primary_reason,
                    "natural_language": explanation.natural_language,
                }
            )

        return {
            "summary": f"Analyzed {len(decisions)} decisions during {time_period}",
            "time_period": time_period,
            "total_decisions": len(decisions),
            "decision_breakdown": decision_counts,
            "total_violations": total_violations,
            "violation_types": violation_types,
            "sample_explanations": explanations,
            "generated_at": datetime.utcnow().isoformat(),
        }
