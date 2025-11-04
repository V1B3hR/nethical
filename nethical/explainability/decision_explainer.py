"""
Decision Explainer - Generates explanations for governance decisions.

This module provides detailed explanations for why a particular decision
was made by the governance system, including:
- Which rules or policies were triggered
- What factors contributed to the risk score
- Why a particular action was recommended
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ExplanationType(Enum):
    """Types of explanations that can be generated."""
    RULE_BASED = "rule_based"  # Explanation based on rule triggers
    RISK_FACTORS = "risk_factors"  # Breakdown of risk score components
    POLICY_MATCH = "policy_match"  # Which policies matched
    VIOLATION_DETAILS = "violation_details"  # Details about detected violations
    CONTEXT_ANALYSIS = "context_analysis"  # Contextual factors considered


@dataclass
class ExplanationComponent:
    """A single component of an explanation."""
    type: ExplanationType
    description: str
    weight: float  # Impact on final decision (0.0-1.0)
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class DecisionExplanation:
    """Complete explanation for a governance decision."""
    decision: str  # The decision that was made (ALLOW, BLOCK, etc.)
    summary: str  # High-level summary of why
    components: List[ExplanationComponent]
    confidence: float
    reasoning_chain: List[str]  # Step-by-step reasoning
    contributing_factors: Dict[str, float]  # Factor -> weight
    alternative_outcomes: Dict[str, str] = field(default_factory=dict)


class DecisionExplainer:
    """
    Generates explanations for governance decisions.
    
    This class analyzes the decision-making process and generates
    human-readable explanations that help users understand why
    a particular decision was made.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the decision explainer.
        
        Args:
            verbose: Whether to include detailed technical information
        """
        self.verbose = verbose
        self.explanation_templates = self._load_templates()
    
    def explain_decision(
        self,
        decision: str,
        context: Dict[str, Any],
        violated_rules: Optional[List[Dict[str, Any]]] = None,
        risk_scores: Optional[Dict[str, float]] = None,
        policy_matches: Optional[List[Dict[str, Any]]] = None
    ) -> DecisionExplanation:
        """
        Generate a comprehensive explanation for a decision.
        
        Args:
            decision: The decision that was made (ALLOW, BLOCK, etc.)
            context: Context information about the request
            violated_rules: List of rules that were violated
            risk_scores: Dictionary of risk score components
            policy_matches: List of policies that matched
            
        Returns:
            A DecisionExplanation object with complete explanation
        """
        components = []
        reasoning_chain = []
        contributing_factors = {}
        
        # Analyze rule violations
        if violated_rules:
            component, factors = self._explain_rule_violations(violated_rules)
            components.append(component)
            contributing_factors.update(factors)
            reasoning_chain.append(f"Detected {len(violated_rules)} rule violation(s)")
        
        # Analyze risk scores
        if risk_scores:
            component, factors = self._explain_risk_scores(risk_scores)
            components.append(component)
            contributing_factors.update(factors)
            reasoning_chain.append(f"Calculated risk score: {sum(risk_scores.values()):.2f}")
        
        # Analyze policy matches
        if policy_matches:
            component, factors = self._explain_policy_matches(policy_matches)
            components.append(component)
            contributing_factors.update(factors)
            reasoning_chain.append(f"Matched {len(policy_matches)} polic(y/ies)")
        
        # Generate summary
        summary = self._generate_summary(decision, components, context)
        reasoning_chain.append(f"Final decision: {decision}")
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(components)
        
        # Generate alternative outcomes
        alternatives = self._generate_alternatives(decision, contributing_factors)
        
        return DecisionExplanation(
            decision=decision,
            summary=summary,
            components=components,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            contributing_factors=contributing_factors,
            alternative_outcomes=alternatives
        )
    
    def _explain_rule_violations(
        self,
        violated_rules: List[Dict[str, Any]]
    ) -> tuple[ExplanationComponent, Dict[str, float]]:
        """Explain which rules were violated and why."""
        details = {
            "violated_rules": [
                {
                    "rule_name": rule.get("name", "unnamed"),
                    "severity": rule.get("severity", "medium"),
                    "description": rule.get("description", "No description")
                }
                for rule in violated_rules
            ],
            "count": len(violated_rules)
        }
        
        # Calculate weight based on severity
        severity_weights = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.3}
        total_weight = sum(
            severity_weights.get(rule.get("severity", "medium"), 0.5)
            for rule in violated_rules
        ) / max(len(violated_rules), 1)
        
        description = f"Violated {len(violated_rules)} rule(s)"
        if violated_rules:
            high_severity = [r for r in violated_rules if r.get("severity") in ["critical", "high"]]
            if high_severity:
                description += f", including {len(high_severity)} critical/high severity"
        
        component = ExplanationComponent(
            type=ExplanationType.RULE_BASED,
            description=description,
            weight=min(total_weight, 1.0),
            details=details,
            confidence=0.9
        )
        
        factors = {f"rule_{rule.get('name', i)}": severity_weights.get(rule.get("severity", "medium"), 0.5)
                   for i, rule in enumerate(violated_rules)}
        
        return component, factors
    
    def _explain_risk_scores(
        self,
        risk_scores: Dict[str, float]
    ) -> tuple[ExplanationComponent, Dict[str, float]]:
        """Explain the breakdown of risk scores."""
        total_risk = sum(risk_scores.values())
        
        details = {
            "risk_breakdown": {
                category: {
                    "score": score,
                    "percentage": (score / total_risk * 100) if total_risk > 0 else 0
                }
                for category, score in risk_scores.items()
            },
            "total_risk": total_risk
        }
        
        # Find highest risk categories
        sorted_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
        top_risks = [f"{cat} ({score:.2f})" for cat, score in sorted_risks[:3]]
        
        description = f"Risk score: {total_risk:.2f}"
        if top_risks:
            description += f". Top factors: {', '.join(top_risks)}"
        
        component = ExplanationComponent(
            type=ExplanationType.RISK_FACTORS,
            description=description,
            weight=min(total_risk / 10.0, 1.0),  # Normalize assuming max risk ~10
            details=details,
            confidence=0.85
        )
        
        return component, risk_scores
    
    def _explain_policy_matches(
        self,
        policy_matches: List[Dict[str, Any]]
    ) -> tuple[ExplanationComponent, Dict[str, float]]:
        """Explain which policies matched."""
        details = {
            "matched_policies": [
                {
                    "policy_name": policy.get("name", "unnamed"),
                    "action": policy.get("action", "unknown"),
                    "conditions": policy.get("conditions", [])
                }
                for policy in policy_matches
            ],
            "count": len(policy_matches)
        }
        
        description = f"Matched {len(policy_matches)} polic(y/ies)"
        if policy_matches:
            actions = [p.get("action") for p in policy_matches]
            description += f": {', '.join(set(actions))}"
        
        weight = min(len(policy_matches) * 0.3, 1.0)
        
        component = ExplanationComponent(
            type=ExplanationType.POLICY_MATCH,
            description=description,
            weight=weight,
            details=details,
            confidence=0.95
        )
        
        factors = {f"policy_{policy.get('name', i)}": 0.3 
                   for i, policy in enumerate(policy_matches)}
        
        return component, factors
    
    def _generate_summary(
        self,
        decision: str,
        components: List[ExplanationComponent],
        context: Dict[str, Any]
    ) -> str:
        """Generate a high-level summary of the decision."""
        if decision == "BLOCK":
            summary = "Request blocked due to safety concerns"
            if components:
                # Find highest weight component
                top_component = max(components, key=lambda c: c.weight)
                summary += f": {top_component.description.lower()}"
        elif decision == "ALLOW":
            summary = "Request allowed"
            if components:
                summary += " with monitoring"
        elif decision == "RESTRICT":
            summary = "Request restricted with limitations"
        elif decision == "TERMINATE":
            summary = "Session terminated due to critical violations"
        else:
            summary = f"Decision: {decision}"
        
        return summary
    
    def _calculate_confidence(self, components: List[ExplanationComponent]) -> float:
        """Calculate overall confidence in the explanation."""
        if not components:
            return 0.5
        
        # Average confidence weighted by component weight
        total_weight = sum(c.weight for c in components)
        if total_weight == 0:
            return 0.5
        
        weighted_confidence = sum(c.confidence * c.weight for c in components)
        return weighted_confidence / total_weight
    
    def _generate_alternatives(
        self,
        decision: str,
        contributing_factors: Dict[str, float]
    ) -> Dict[str, str]:
        """Generate alternative outcomes that could have occurred."""
        alternatives = {}
        
        total_weight = sum(contributing_factors.values())
        
        if decision == "BLOCK":
            if total_weight < 0.5:
                alternatives["RESTRICT"] = "If risk factors were slightly lower"
            if total_weight < 0.3:
                alternatives["ALLOW"] = "If no significant violations detected"
        elif decision == "ALLOW":
            if total_weight > 0.3:
                alternatives["RESTRICT"] = "If risk score was higher"
            if total_weight > 0.7:
                alternatives["BLOCK"] = "If violations were more severe"
        
        return alternatives
    
    def _load_templates(self) -> Dict[str, str]:
        """Load explanation templates."""
        return {
            "BLOCK": "Request blocked to prevent potential harm",
            "ALLOW": "Request allowed as it meets safety requirements",
            "RESTRICT": "Request partially allowed with restrictions",
            "TERMINATE": "Session ended due to critical policy violations"
        }
    
    def explain_to_json(self, explanation: DecisionExplanation) -> Dict[str, Any]:
        """Convert explanation to JSON-serializable format."""
        return {
            "decision": explanation.decision,
            "summary": explanation.summary,
            "confidence": explanation.confidence,
            "components": [
                {
                    "type": comp.type.value,
                    "description": comp.description,
                    "weight": comp.weight,
                    "confidence": comp.confidence,
                    "details": comp.details
                }
                for comp in explanation.components
            ],
            "reasoning_chain": explanation.reasoning_chain,
            "contributing_factors": explanation.contributing_factors,
            "alternative_outcomes": explanation.alternative_outcomes
        }
