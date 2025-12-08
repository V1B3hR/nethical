"""
Natural Language Generator - Converts technical decisions to human-readable text.

This module takes technical decision data and generates natural language
explanations that are easy for non-technical users to understand.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class NaturalLanguageExplanation:
    """Natural language explanation of a decision."""

    title: str
    summary: str
    detailed_explanation: str
    key_points: List[str]
    recommendations: List[str]


class NaturalLanguageGenerator:
    """
    Generates natural language explanations from technical decision data.

    This class converts technical governance decisions, risk scores, and
    policy violations into clear, human-readable explanations suitable
    for end users, reviewers, and auditors.
    """

    def __init__(self, language: str = "en", tone: str = "professional"):
        """
        Initialize the natural language generator.

        Args:
            language: Language code (currently only 'en' supported)
            tone: Tone of explanations ('professional', 'casual', 'technical')
        """
        self.language = language
        self.tone = tone
        self.templates = self._load_templates()

    def generate_explanation(
        self,
        decision: str,
        context: Dict[str, Any],
        components: List[Dict[str, Any]],
        reasoning_chain: List[str],
    ) -> NaturalLanguageExplanation:
        """
        Generate a natural language explanation.

        Args:
            decision: The decision that was made
            context: Context information
            components: Explanation components
            reasoning_chain: Step-by-step reasoning

        Returns:
            A NaturalLanguageExplanation with human-readable text
        """
        title = self._generate_title(decision, context)
        summary = self._generate_summary(decision, components)
        detailed = self._generate_detailed_explanation(
            decision, components, reasoning_chain
        )
        key_points = self._extract_key_points(components)
        recommendations = self._generate_recommendations(decision, components, context)

        return NaturalLanguageExplanation(
            title=title,
            summary=summary,
            detailed_explanation=detailed,
            key_points=key_points,
            recommendations=recommendations,
        )

    def _generate_title(self, decision: str, context: Dict[str, Any]) -> str:
        """Generate a clear title for the explanation."""
        decision_verbs = {
            "BLOCK": "Blocked",
            "ALLOW": "Allowed",
            "RESTRICT": "Restricted",
            "TERMINATE": "Terminated",
        }

        verb = decision_verbs.get(decision, decision)
        action_type = context.get("action_type", "request")

        return f"{verb}: {action_type.title()}"

    def _generate_summary(self, decision: str, components: List[Dict[str, Any]]) -> str:
        """Generate a one-sentence summary."""
        if decision == "BLOCK":
            return self._generate_block_summary(components)
        elif decision == "ALLOW":
            return self._generate_allow_summary(components)
        elif decision == "RESTRICT":
            return self._generate_restrict_summary(components)
        elif decision == "TERMINATE":
            return self._generate_terminate_summary(components)
        else:
            return f"Decision: {decision}"

    def _generate_block_summary(self, components: List[Dict[str, Any]]) -> str:
        """Generate summary for BLOCK decision."""
        if not components:
            return "This request was blocked to maintain safety standards."

        # Find the most significant component
        main_component = max(components, key=lambda c: c.get("weight", 0))
        comp_type = main_component.get("type", "unknown")

        if comp_type == "rule_based":
            rule_count = main_component.get("details", {}).get("count", 0)
            return f"This request was blocked because it violated {rule_count} safety rule(s)."
        elif comp_type == "risk_factors":
            risk_score = main_component.get("details", {}).get("total_risk", 0)
            return (
                f"This request was blocked due to high risk score ({risk_score:.2f})."
            )
        elif comp_type == "policy_match":
            return "This request was blocked by organization policies."
        else:
            return "This request was blocked to prevent potential harm."

    def _generate_allow_summary(self, components: List[Dict[str, Any]]) -> str:
        """Generate summary for ALLOW decision."""
        if not components:
            return "This request was allowed as it meets all safety requirements."

        return "This request was allowed with standard monitoring enabled."

    def _generate_restrict_summary(self, components: List[Dict[str, Any]]) -> str:
        """Generate summary for RESTRICT decision."""
        return "This request was allowed with restrictions to ensure safety."

    def _generate_terminate_summary(self, components: List[Dict[str, Any]]) -> str:
        """Generate summary for TERMINATE decision."""
        return (
            "This session was terminated immediately due to critical policy violations."
        )

    def _generate_detailed_explanation(
        self,
        decision: str,
        components: List[Dict[str, Any]],
        reasoning_chain: List[str],
    ) -> str:
        """Generate a detailed multi-paragraph explanation."""
        paragraphs = []

        # Introduction paragraph
        intro = self._generate_intro_paragraph(decision)
        paragraphs.append(intro)

        # Explain each component
        for component in components:
            component_text = self._explain_component(component)
            if component_text:
                paragraphs.append(component_text)

        # Reasoning process
        if reasoning_chain:
            reasoning_text = self._explain_reasoning(reasoning_chain)
            paragraphs.append(reasoning_text)

        return "\n\n".join(paragraphs)

    def _generate_intro_paragraph(self, decision: str) -> str:
        """Generate introduction paragraph."""
        intros = {
            "BLOCK": "This request was carefully evaluated and blocked to maintain safety and security standards. The decision was made based on multiple factors that indicated potential risks.",
            "ALLOW": "This request was evaluated and allowed to proceed. Our analysis indicates that it meets all safety requirements and poses minimal risk.",
            "RESTRICT": "This request was evaluated and partially allowed with certain restrictions in place. These restrictions help mitigate identified risks while allowing the core functionality.",
            "TERMINATE": "This session was immediately terminated due to critical policy violations. Immediate action was necessary to prevent potential harm.",
        }
        return intros.get(decision, f"This request received a decision of {decision}.")

    def _explain_component(self, component: Dict[str, Any]) -> str:
        """Explain a single component in natural language."""
        comp_type = component.get("type")
        details = component.get("details", {})

        if comp_type == "rule_based":
            return self._explain_rule_component(details)
        elif comp_type == "risk_factors":
            return self._explain_risk_component(details)
        elif comp_type == "policy_match":
            return self._explain_policy_component(details)
        else:
            return ""

    def _explain_rule_component(self, details: Dict[str, Any]) -> str:
        """Explain rule violations."""
        violated_rules = details.get("violated_rules", [])
        if not violated_rules:
            return ""

        text = f"The request violated {len(violated_rules)} safety rule(s):\n"

        for rule in violated_rules[:3]:  # Show top 3
            name = rule.get("rule_name", "unnamed")
            severity = rule.get("severity", "medium")
            desc = rule.get("description", "No description")
            text += f"\n- **{name}** (severity: {severity}): {desc}"

        if len(violated_rules) > 3:
            text += f"\n- ... and {len(violated_rules) - 3} more rule(s)"

        return text

    def _explain_risk_component(self, details: Dict[str, Any]) -> str:
        """Explain risk score breakdown."""
        risk_breakdown = details.get("risk_breakdown", {})
        total_risk = details.get("total_risk", 0)

        if not risk_breakdown:
            return f"The overall risk score was {total_risk:.2f}."

        text = f"The risk analysis produced a total score of {total_risk:.2f}, calculated from:\n"

        # Sort by score descending
        sorted_risks = sorted(
            risk_breakdown.items(), key=lambda x: x[1].get("score", 0), reverse=True
        )

        for category, data in sorted_risks[:3]:  # Top 3 risks
            score = data.get("score", 0)
            percentage = data.get("percentage", 0)
            text += f"\n- **{category.replace('_', ' ').title()}**: {score:.2f} ({percentage:.1f}% of total)"

        return text

    def _explain_policy_component(self, details: Dict[str, Any]) -> str:
        """Explain policy matches."""
        matched_policies = details.get("matched_policies", [])
        if not matched_policies:
            return ""

        policy_word = "policy" if len(matched_policies) == 1 else "policies"
        text = (
            f"The request matched {len(matched_policies)} organization {policy_word}:\n"
        )

        for policy in matched_policies[:3]:  # Top 3
            name = policy.get("policy_name", "unnamed")
            action = policy.get("action", "unknown")
            text += f"\n- **{name}**: {action}"

        return text

    def _explain_reasoning(self, reasoning_chain: List[str]) -> str:
        """Explain the reasoning process."""
        text = "**Decision Process:**\n"
        for i, step in enumerate(reasoning_chain, 1):
            text += f"\n{i}. {step}"
        return text

    def _extract_key_points(self, components: List[Dict[str, Any]]) -> List[str]:
        """Extract key points from components."""
        key_points = []

        for component in components:
            comp_type = component.get("type")
            weight = component.get("weight", 0)

            if weight < 0.3:  # Skip low-weight components
                continue

            if comp_type == "rule_based":
                count = component.get("details", {}).get("count", 0)
                key_points.append(f"Violated {count} safety rule(s)")
            elif comp_type == "risk_factors":
                risk = component.get("details", {}).get("total_risk", 0)
                key_points.append(f"Risk score: {risk:.2f}")
            elif comp_type == "policy_match":
                count = component.get("details", {}).get("count", 0)
                policy_word = "policy" if count == 1 else "policies"
                key_points.append(f"Matched {count} {policy_word}")

        return key_points

    def _generate_recommendations(
        self, decision: str, components: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if decision == "BLOCK":
            recommendations.append(
                "Review the violated rules and adjust your request accordingly"
            )
            recommendations.append(
                "Contact support if you believe this decision is incorrect"
            )

            # Check if there are specific high-severity violations
            for component in components:
                if component.get("type") == "rule_based":
                    violated_rules = component.get("details", {}).get(
                        "violated_rules", []
                    )
                    critical_rules = [
                        r for r in violated_rules if r.get("severity") == "critical"
                    ]
                    if critical_rules:
                        recommendations.append(
                            "Address critical security violations before retrying"
                        )

        elif decision == "ALLOW":
            recommendations.append("Proceed with your request as normal")
            recommendations.append(
                "All actions will be monitored for safety compliance"
            )

        elif decision == "RESTRICT":
            recommendations.append("Note the restrictions applied to this request")
            recommendations.append("Review limitations before proceeding")

        elif decision == "TERMINATE":
            recommendations.append("Review your organization's acceptable use policy")
            recommendations.append("Contact your administrator for account status")

        return recommendations

    def _load_templates(self) -> Dict[str, str]:
        """Load language templates."""
        # Future: Load from external files for multi-language support
        return {
            "decision_made": "A decision of {decision} was made",
            "rule_violation": "Rule violation detected: {rule}",
            "high_risk": "High risk detected in {category}",
            "policy_match": "Policy {policy} was triggered",
        }

    def to_markdown(self, explanation: NaturalLanguageExplanation) -> str:
        """Convert explanation to markdown format."""
        md = f"# {explanation.title}\n\n"
        md += f"## Summary\n\n{explanation.summary}\n\n"
        md += f"## Detailed Explanation\n\n{explanation.detailed_explanation}\n\n"

        if explanation.key_points:
            md += "## Key Points\n\n"
            for point in explanation.key_points:
                md += f"- {point}\n"
            md += "\n"

        if explanation.recommendations:
            md += "## Recommendations\n\n"
            for rec in explanation.recommendations:
                md += f"- {rec}\n"
            md += "\n"

        return md

    def to_html(self, explanation: NaturalLanguageExplanation) -> str:
        """Convert explanation to HTML format."""
        html = f"<h1>{explanation.title}</h1>\n"
        html += f"<h2>Summary</h2>\n<p>{explanation.summary}</p>\n"
        html += f"<h2>Detailed Explanation</h2>\n"

        # Convert paragraphs
        for para in explanation.detailed_explanation.split("\n\n"):
            html += f"<p>{para}</p>\n"

        if explanation.key_points:
            html += "<h2>Key Points</h2>\n<ul>\n"
            for point in explanation.key_points:
                html += f"<li>{point}</li>\n"
            html += "</ul>\n"

        if explanation.recommendations:
            html += "<h2>Recommendations</h2>\n<ul>\n"
            for rec in explanation.recommendations:
                html += f"<li>{rec}</li>\n"
            html += "</ul>\n"

        return html
