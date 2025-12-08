"""Explainability API endpoints for Phase 2.3.

This module provides REST API endpoints for:
- Decision explanations
- Decision tree visualization
- Transparency reports
"""

from typing import Dict, Any, List
from ..core.explainability import DecisionExplainer, TransparencyReportGenerator
from .taxonomy_api import APIResponse


class ExplainabilityAPI:
    """REST API for explainability features."""

    def __init__(self):
        """Initialize explainability API."""
        self.explainer = DecisionExplainer()
        self.report_generator = TransparencyReportGenerator()

    def explain_decision_endpoint(
        self, decision: str, judgment_data: Dict[str, Any], include_ml: bool = False
    ) -> Dict[str, Any]:
        """API endpoint: Explain a decision.

        Args:
            decision: Decision made
            judgment_data: Judgment data
            include_ml: Include ML explanations

        Returns:
            Explanation response
        """
        try:
            explanation = self.explainer.explain_decision(
                decision=decision,
                judgment_data=judgment_data,
                include_ml_explanation=include_ml,
            )

            return APIResponse(
                success=True,
                data={
                    "decision": explanation.decision,
                    "explanation_type": explanation.explanation_type.value,
                    "primary_reason": explanation.primary_reason,
                    "contributing_factors": explanation.contributing_factors,
                    "rules_matched": explanation.rules_matched,
                    "threshold_comparisons": explanation.threshold_comparisons,
                    "natural_language": explanation.natural_language,
                    "confidence": explanation.confidence,
                    "timestamp": explanation.timestamp.isoformat(),
                },
                message="Decision explained successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to explain decision"
            ).to_dict()

    def explain_policy_match_endpoint(
        self, matched_rule: Dict[str, Any], facts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """API endpoint: Explain why a policy rule matched.

        Args:
            matched_rule: Matched rule details
            facts: Input facts

        Returns:
            Explanation response
        """
        try:
            explanation = self.explainer.explain_policy_match(matched_rule, facts)

            return APIResponse(
                success=True,
                data={
                    "rule_id": matched_rule.get("id", "unknown"),
                    "explanation": explanation,
                    "matched_rule": matched_rule,
                },
                message="Policy match explained successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to explain policy match"
            ).to_dict()

    def get_decision_tree_endpoint(
        self, judgment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """API endpoint: Get decision tree visualization.

        Args:
            judgment_data: Judgment data

        Returns:
            Decision tree visualization
        """
        try:
            tree = self.explainer.generate_decision_tree_viz(judgment_data)

            return APIResponse(
                success=True,
                data={"tree": tree, "format": "hierarchical"},
                message="Decision tree generated successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False, error=str(e), message="Failed to generate decision tree"
            ).to_dict()

    def generate_transparency_report_endpoint(
        self, decisions: List[Dict[str, Any]], time_period: str = "recent"
    ) -> Dict[str, Any]:
        """API endpoint: Generate transparency report.

        Args:
            decisions: List of decisions
            time_period: Time period description

        Returns:
            Transparency report
        """
        try:
            report = self.report_generator.generate_report(decisions, time_period)

            return APIResponse(
                success=True,
                data=report,
                message="Transparency report generated successfully",
            ).to_dict()

        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                message="Failed to generate transparency report",
            ).to_dict()
