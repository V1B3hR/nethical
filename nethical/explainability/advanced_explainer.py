"""Advanced Explainability Tools for AI Governance.

This module provides SHAP-like and LIME-like explainability capabilities
for understanding AI governance decisions without requiring the full
SHAP/LIME libraries (to minimize dependencies).

Features:
- Feature importance calculation (SHAP-style)
- Local explanation generation (LIME-style)
- Counterfactual explanations
- Decision path visualization
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math


@dataclass
class FeatureImportance:
    """Feature importance scores for a decision."""

    feature_name: str
    importance_score: float  # -1.0 to 1.0, negative means pushes towards safe, positive towards violation
    base_value: float  # The feature's actual value
    description: str = ""
    category: str = "general"


@dataclass
class LocalExplanation:
    """Local explanation for a specific decision (LIME-style)."""

    decision: str
    confidence: float
    feature_importances: List[FeatureImportance]
    most_influential_features: List[str]
    explanation_text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation showing what changes would alter the decision."""

    original_decision: str
    counterfactual_decision: str
    required_changes: List[Dict[str, Any]]
    minimal_change: bool
    explanation_text: str


class AdvancedExplainer:
    """Advanced explainability tools for governance decisions."""

    # Constants for maintainability
    SHAP_SCALING_FACTOR = 2.0  # Scale factor for SHAP value normalization
    MAX_VIOLATIONS_FOR_NORMALIZATION = 5.0  # Cap for violation count normalization

    # Impact thresholds for visualization color coding
    HIGH_RISK_THRESHOLD = 0.15  # Red: Strong risk contribution
    MEDIUM_RISK_THRESHOLD = 0.05  # Orange: Moderate risk contribution
    MEDIUM_SAFETY_THRESHOLD = -0.05  # Light green: Moderate safety signal
    HIGH_SAFETY_THRESHOLD = -0.15  # Green: Strong safety signal

    def __init__(self):
        """Initialize the advanced explainer."""
        # Feature weights for importance calculation
        self.feature_weights = {
            "risk_score": 0.30,
            "violation_count": 0.25,
            "pii_risk": 0.20,
            "ethical_score": 0.15,
            "quota_pressure": 0.10,
        }

        # Decision thresholds
        self.decision_thresholds = {
            "BLOCK": 0.7,
            "RESTRICT": 0.4,
            "ALLOW": 0.0,
        }

    def explain_decision_with_features(
        self, decision: str, judgment_data: Dict[str, Any], confidence: float = 1.0
    ) -> LocalExplanation:
        """Generate a local explanation for a decision (LIME-style).

        Args:
            decision: The governance decision
            judgment_data: Data from the judgment process
            confidence: Confidence in the decision

        Returns:
            LocalExplanation with feature importances
        """
        # Extract features from judgment data
        features = self._extract_features(judgment_data)

        # Calculate feature importances
        feature_importances = self._calculate_feature_importances(
            features, decision, judgment_data
        )

        # Sort by absolute importance
        sorted_features = sorted(
            feature_importances, key=lambda f: abs(f.importance_score), reverse=True
        )

        # Get most influential features (top 5)
        most_influential = [f.feature_name for f in sorted_features[:5]]

        # Generate explanation text
        explanation_text = self._generate_feature_explanation(
            decision, sorted_features, confidence
        )

        return LocalExplanation(
            decision=decision,
            confidence=confidence,
            feature_importances=feature_importances,
            most_influential_features=most_influential,
            explanation_text=explanation_text,
        )

    def calculate_shap_values(self, judgment_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate SHAP-like values for features.

        Args:
            judgment_data: Data from the judgment process

        Returns:
            Dictionary of feature names to SHAP values
        """
        features = self._extract_features(judgment_data)
        shap_values = {}

        # Calculate base risk (average expected risk)
        base_risk = 0.3  # Neutral baseline

        for feature_name, feature_value in features.items():
            # Calculate the marginal contribution of this feature
            weight = self.feature_weights.get(feature_name, 0.1)

            # Normalize feature value to 0-1 range
            normalized_value = self._normalize_feature(feature_name, feature_value)

            # For violation_count, any violations contribute positively to risk
            # For other features, compare to baseline 0.5
            if feature_name == "violation_count":
                # Violations always contribute to risk (no negative baseline comparison)
                shap_value = normalized_value * weight * self.SHAP_SCALING_FACTOR
            else:
                # SHAP value is the contribution relative to baseline
                shap_value = (
                    (normalized_value - 0.5) * weight * self.SHAP_SCALING_FACTOR
                )

            shap_values[feature_name] = shap_value

        return shap_values

    def generate_counterfactual(
        self,
        current_decision: str,
        judgment_data: Dict[str, Any],
        desired_decision: Optional[str] = None,
    ) -> CounterfactualExplanation:
        """Generate counterfactual explanation.

        Shows what would need to change for a different decision.

        Args:
            current_decision: Current governance decision
            judgment_data: Current judgment data
            desired_decision: Desired decision (defaults to ALLOW)

        Returns:
            CounterfactualExplanation showing required changes
        """
        if desired_decision is None:
            desired_decision = "ALLOW"

        features = self._extract_features(judgment_data)
        required_changes = []

        # Determine what needs to change
        current_risk = judgment_data.get("phase3", {}).get("risk_score", 0.5)
        target_threshold = self.decision_thresholds.get(desired_decision, 0.0)

        if current_decision == desired_decision:
            explanation_text = (
                f"Current decision is already {desired_decision}. "
                "No changes required."
            )
        else:
            # Find minimal changes needed
            risk_reduction_needed = current_risk - target_threshold

            # Calculate what changes would achieve this
            if risk_reduction_needed > 0:
                # Need to reduce risk
                violation_count = features.get("violation_count", 0)
                pii_risk = features.get("pii_risk", 0.0)

                if violation_count > 0:
                    required_changes.append(
                        {
                            "feature": "violation_count",
                            "current_value": violation_count,
                            "required_value": 0,
                            "change_type": "eliminate",
                            "description": "Remove all violations",
                        }
                    )

                if pii_risk > 0.5:
                    required_changes.append(
                        {
                            "feature": "pii_risk",
                            "current_value": pii_risk,
                            "required_value": 0.0,
                            "change_type": "reduce",
                            "description": "Remove all PII from content",
                        }
                    )

                explanation_text = (
                    f"To change decision from {current_decision} to {desired_decision}, "
                    f"risk score must decrease by {risk_reduction_needed:.2f}. "
                    "Required changes: "
                    + ", ".join(c["description"] for c in required_changes)
                )
            else:
                explanation_text = (
                    f"Current risk score {current_risk:.2f} is already below "
                    f"threshold {target_threshold:.2f} for {desired_decision}."
                )

        return CounterfactualExplanation(
            original_decision=current_decision,
            counterfactual_decision=desired_decision,
            required_changes=required_changes,
            minimal_change=len(required_changes) <= 2,
            explanation_text=explanation_text,
        )

    def generate_decision_path_visualization(
        self, judgment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate data for decision path visualization (tree-like structure).

        Args:
            judgment_data: Judgment data

        Returns:
            Visualization data structure
        """
        features = self._extract_features(judgment_data)
        decision = judgment_data.get("decision", "UNKNOWN")

        # Build decision tree structure
        tree = {"name": "Governance Decision", "decision": decision, "children": []}

        # Add feature branches
        for feature_name, feature_value in features.items():
            normalized = self._normalize_feature(feature_name, feature_value)
            weight = self.feature_weights.get(feature_name, 0.1)
            impact = normalized * weight

            feature_node = {
                "name": feature_name.replace("_", " ").title(),
                "value": feature_value,
                "normalized_value": normalized,
                "weight": weight,
                "impact": impact,
                "color": self._get_color_for_impact(impact),
            }
            tree["children"].append(feature_node)

        return tree

    def _extract_features(self, judgment_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from judgment data.

        Args:
            judgment_data: Judgment data

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Extract Phase 3 features
        phase3 = judgment_data.get("phase3", {})
        features["risk_score"] = phase3.get("risk_score", 0.0)

        # Extract violation count
        violations = judgment_data.get("violations", [])
        features["violation_count"] = len(violations)

        # Extract PII risk
        pii_detection = judgment_data.get("pii_detection", {})
        features["pii_risk"] = pii_detection.get("pii_risk_score", 0.0)

        # Extract ethical score
        phase4 = judgment_data.get("phase4", {})
        ethical_tags = phase4.get("ethical_tags", {})
        if ethical_tags:
            dimensions = ethical_tags.get("dimensions", {})
            features["ethical_score"] = (
                sum(dimensions.values()) / len(dimensions) if dimensions else 0.0
            )
        else:
            features["ethical_score"] = 0.0

        # Extract quota pressure
        quota_enforcement = judgment_data.get("quota_enforcement")
        if quota_enforcement and isinstance(quota_enforcement, dict):
            features["quota_pressure"] = quota_enforcement.get(
                "backpressure_level", 0.0
            )
        else:
            features["quota_pressure"] = 0.0

        return features

    def _calculate_feature_importances(
        self, features: Dict[str, float], decision: str, judgment_data: Dict[str, Any]
    ) -> List[FeatureImportance]:
        """Calculate importance scores for features.

        Args:
            features: Feature dictionary
            decision: Decision made
            judgment_data: Full judgment data

        Returns:
            List of FeatureImportance objects
        """
        importances = []

        for feature_name, feature_value in features.items():
            # Normalize feature value
            normalized = self._normalize_feature(feature_name, feature_value)

            # Get weight
            weight = self.feature_weights.get(feature_name, 0.1)

            # Calculate importance (contribution to decision)
            # Positive importance = pushes towards violation/block
            # Negative importance = pushes towards safe/allow
            importance_score = (normalized - 0.5) * weight * 2

            # Create description
            description = self._describe_feature_contribution(
                feature_name, feature_value, importance_score, decision
            )

            importances.append(
                FeatureImportance(
                    feature_name=feature_name,
                    importance_score=importance_score,
                    base_value=feature_value,
                    description=description,
                    category=self._categorize_feature(feature_name),
                )
            )

        return importances

    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """Normalize feature value to 0-1 range.

        Args:
            feature_name: Name of the feature
            value: Raw feature value

        Returns:
            Normalized value between 0 and 1
        """
        if feature_name == "violation_count":
            # Cap at MAX_VIOLATIONS_FOR_NORMALIZATION for normalization
            return min(value / self.MAX_VIOLATIONS_FOR_NORMALIZATION, 1.0)
        elif feature_name in [
            "risk_score",
            "pii_risk",
            "ethical_score",
            "quota_pressure",
        ]:
            # Already in 0-1 range
            return min(max(value, 0.0), 1.0)
        else:
            # Default normalization
            return min(max(value, 0.0), 1.0)

    def _describe_feature_contribution(
        self, feature_name: str, value: float, importance: float, decision: str
    ) -> str:
        """Describe how a feature contributed to the decision.

        Args:
            feature_name: Name of the feature
            value: Feature value
            importance: Calculated importance score
            decision: Final decision

        Returns:
            Human-readable description
        """
        direction = "increased" if importance > 0 else "decreased"
        strength = (
            "strongly"
            if abs(importance) > 0.15
            else "moderately" if abs(importance) > 0.05 else "slightly"
        )

        feature_display = feature_name.replace("_", " ").title()

        if importance > 0:
            return f"{feature_display} ({value:.2f}) {strength} {direction} likelihood of blocking"
        elif importance < 0:
            return f"{feature_display} ({value:.2f}) {strength} supported allowing the action"
        else:
            return f"{feature_display} ({value:.2f}) had minimal impact on decision"

    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize a feature for grouping.

        Args:
            feature_name: Name of the feature

        Returns:
            Category name
        """
        # Check more specific patterns first
        if "pii" in feature_name:
            return "privacy"
        elif "violation" in feature_name:
            return "violation"
        elif "ethical" in feature_name:
            return "ethics"
        elif "quota" in feature_name:
            return "resource"
        elif "risk" in feature_name:
            return "risk"
        else:
            return "general"

    def _generate_feature_explanation(
        self,
        decision: str,
        feature_importances: List[FeatureImportance],
        confidence: float,
    ) -> str:
        """Generate natural language explanation from features.

        Args:
            decision: The decision made
            feature_importances: List of feature importances
            confidence: Confidence in decision

        Returns:
            Natural language explanation
        """
        parts = [
            f"Decision: {decision} (confidence: {confidence:.2%})",
            "",
            "Key Contributing Factors:",
        ]

        # Add top 3 most influential features
        top_features = sorted(
            feature_importances, key=lambda f: abs(f.importance_score), reverse=True
        )[:3]

        for i, feature in enumerate(top_features, 1):
            parts.append(f"{i}. {feature.description}")

        # Add summary
        total_positive = sum(
            f.importance_score for f in feature_importances if f.importance_score > 0
        )
        total_negative = sum(
            f.importance_score for f in feature_importances if f.importance_score < 0
        )

        parts.append("")
        if abs(total_positive) > abs(total_negative):
            parts.append(
                f"Overall: Features pushed towards blocking/restricting the action."
            )
        elif abs(total_negative) > abs(total_positive):
            parts.append(f"Overall: Features supported allowing the action.")
        else:
            parts.append(
                f"Overall: Features were balanced between allowing and blocking."
            )

        return "\n".join(parts)

    def _get_color_for_impact(self, impact: float) -> str:
        """Get color code for visualization based on impact.

        Args:
            impact: Impact score

        Returns:
            Color name or hex code
        """
        if impact > self.HIGH_RISK_THRESHOLD:
            return "red"  # High risk contribution
        elif impact > self.MEDIUM_RISK_THRESHOLD:
            return "orange"  # Medium risk contribution
        elif impact < self.HIGH_SAFETY_THRESHOLD:
            return "green"  # Strong safety signal
        elif impact < self.MEDIUM_SAFETY_THRESHOLD:
            return "lightgreen"  # Medium safety signal
        else:
            return "gray"  # Neutral
