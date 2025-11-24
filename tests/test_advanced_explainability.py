"""Tests for Advanced Explainability Features.

Tests SHAP-like, LIME-like, and counterfactual explanation capabilities.
"""

import pytest
from nethical.explainability.advanced_explainer import (
    AdvancedExplainer,
    FeatureImportance,
    LocalExplanation,
    CounterfactualExplanation
)


@pytest.fixture
def explainer():
    """Create an advanced explainer instance."""
    return AdvancedExplainer()


@pytest.fixture
def sample_judgment_data_safe():
    """Create sample judgment data for a safe action."""
    return {
        "agent_id": "test_agent",
        "decision": "ALLOW",
        "violations": [],
        "phase3": {
            "risk_score": 0.1,
            "risk_tier": "low"
        },
        "pii_detection": {
            "matches_count": 0,
            "pii_risk_score": 0.0
        },
        "phase4": {
            "ethical_tags": {}
        },
        "quota_enforcement": {
            "backpressure_level": 0.2
        }
    }


@pytest.fixture
def sample_judgment_data_violation():
    """Create sample judgment data for a violation."""
    return {
        "agent_id": "test_agent",
        "decision": "BLOCK",
        "violations": [
            {
                "violation_type": "privacy",
                "severity": "high",
                "description": "PII detected"
            },
            {
                "violation_type": "unauthorized_access",
                "severity": "critical",
                "description": "Unauthorized database access"
            }
        ],
        "phase3": {
            "risk_score": 0.85,
            "risk_tier": "critical"
        },
        "pii_detection": {
            "matches_count": 3,
            "pii_risk_score": 0.9
        },
        "phase4": {
            "ethical_tags": {
                "primary_dimension": "privacy",
                "dimensions": {
                    "privacy": 0.95,
                    "security": 0.80
                }
            }
        },
        "quota_enforcement": {
            "backpressure_level": 0.7
        }
    }


def test_explainer_initialization(explainer):
    """Test that explainer initializes correctly."""
    assert explainer is not None
    assert len(explainer.feature_weights) > 0
    assert "risk_score" in explainer.feature_weights
    assert "BLOCK" in explainer.decision_thresholds


def test_explain_safe_decision(explainer, sample_judgment_data_safe):
    """Test explanation for a safe ALLOW decision."""
    explanation = explainer.explain_decision_with_features(
        decision="ALLOW",
        judgment_data=sample_judgment_data_safe,
        confidence=0.95
    )
    
    assert isinstance(explanation, LocalExplanation)
    assert explanation.decision == "ALLOW"
    assert explanation.confidence == 0.95
    assert len(explanation.feature_importances) > 0
    assert len(explanation.most_influential_features) > 0
    assert len(explanation.explanation_text) > 0
    
    # Check that low risk features have negative importance (support allowing)
    risk_feature = next(
        (f for f in explanation.feature_importances if f.feature_name == "risk_score"),
        None
    )
    assert risk_feature is not None
    assert risk_feature.importance_score < 0  # Low risk supports ALLOW


def test_explain_violation_decision(explainer, sample_judgment_data_violation):
    """Test explanation for a BLOCK decision with violations."""
    explanation = explainer.explain_decision_with_features(
        decision="BLOCK",
        judgment_data=sample_judgment_data_violation,
        confidence=0.90
    )
    
    assert isinstance(explanation, LocalExplanation)
    assert explanation.decision == "BLOCK"
    assert len(explanation.feature_importances) > 0
    
    # Check that high risk features have positive importance (push towards blocking)
    risk_feature = next(
        (f for f in explanation.feature_importances if f.feature_name == "risk_score"),
        None
    )
    assert risk_feature is not None
    assert risk_feature.importance_score > 0  # High risk pushes towards BLOCK
    
    # Check violation count
    violation_feature = next(
        (f for f in explanation.feature_importances if f.feature_name == "violation_count"),
        None
    )
    assert violation_feature is not None
    assert violation_feature.base_value == 2  # Two violations in sample data


def test_calculate_shap_values(explainer, sample_judgment_data_violation):
    """Test SHAP-like value calculation."""
    shap_values = explainer.calculate_shap_values(sample_judgment_data_violation)
    
    assert isinstance(shap_values, dict)
    assert len(shap_values) > 0
    assert "risk_score" in shap_values
    assert "violation_count" in shap_values
    
    # High risk should have positive SHAP value
    assert shap_values["risk_score"] > 0
    
    # Violations should have positive SHAP value
    assert shap_values["violation_count"] > 0


def test_feature_importance_ordering(explainer, sample_judgment_data_violation):
    """Test that features are ordered by importance."""
    explanation = explainer.explain_decision_with_features(
        decision="BLOCK",
        judgment_data=sample_judgment_data_violation
    )
    
    # Most influential features should be in the list
    assert len(explanation.most_influential_features) > 0
    assert len(explanation.most_influential_features) <= 5
    
    # Check that features are actually influential
    for feature_name in explanation.most_influential_features:
        feature = next(
            (f for f in explanation.feature_importances if f.feature_name == feature_name),
            None
        )
        assert feature is not None
        assert abs(feature.importance_score) > 0


def test_generate_counterfactual_block_to_allow(explainer, sample_judgment_data_violation):
    """Test counterfactual explanation from BLOCK to ALLOW."""
    counterfactual = explainer.generate_counterfactual(
        current_decision="BLOCK",
        judgment_data=sample_judgment_data_violation,
        desired_decision="ALLOW"
    )
    
    assert isinstance(counterfactual, CounterfactualExplanation)
    assert counterfactual.original_decision == "BLOCK"
    assert counterfactual.counterfactual_decision == "ALLOW"
    assert len(counterfactual.required_changes) > 0
    assert len(counterfactual.explanation_text) > 0
    
    # Should suggest removing violations
    violation_change = next(
        (c for c in counterfactual.required_changes if c["feature"] == "violation_count"),
        None
    )
    assert violation_change is not None
    assert violation_change["required_value"] == 0


def test_generate_counterfactual_same_decision(explainer, sample_judgment_data_safe):
    """Test counterfactual when current and desired decisions are the same."""
    counterfactual = explainer.generate_counterfactual(
        current_decision="ALLOW",
        judgment_data=sample_judgment_data_safe,
        desired_decision="ALLOW"
    )
    
    assert isinstance(counterfactual, CounterfactualExplanation)
    assert "already" in counterfactual.explanation_text.lower()
    assert "no changes" in counterfactual.explanation_text.lower()


def test_decision_path_visualization(explainer, sample_judgment_data_violation):
    """Test decision path visualization data generation."""
    viz_data = explainer.generate_decision_path_visualization(
        sample_judgment_data_violation
    )
    
    assert isinstance(viz_data, dict)
    assert "name" in viz_data
    assert "decision" in viz_data
    assert "children" in viz_data
    assert len(viz_data["children"]) > 0
    
    # Check that children have required fields
    for child in viz_data["children"]:
        assert "name" in child
        assert "value" in child
        assert "impact" in child
        assert "color" in child


def test_feature_normalization(explainer):
    """Test feature normalization."""
    # Test risk score (already 0-1)
    assert explainer._normalize_feature("risk_score", 0.5) == 0.5
    assert explainer._normalize_feature("risk_score", 1.5) == 1.0  # Capped at 1
    assert explainer._normalize_feature("risk_score", -0.5) == 0.0  # Capped at 0
    
    # Test violation count (capped at 5)
    assert explainer._normalize_feature("violation_count", 2.5) == 0.5
    assert explainer._normalize_feature("violation_count", 10) == 1.0  # Capped at 5


def test_feature_categorization(explainer):
    """Test feature categorization."""
    assert explainer._categorize_feature("risk_score") == "risk"
    assert explainer._categorize_feature("violation_count") == "violation"
    assert explainer._categorize_feature("pii_risk") == "privacy"
    assert explainer._categorize_feature("ethical_score") == "ethics"
    assert explainer._categorize_feature("quota_pressure") == "resource"
    assert explainer._categorize_feature("unknown_feature") == "general"


def test_explanation_text_quality(explainer, sample_judgment_data_violation):
    """Test that explanation text is informative and well-structured."""
    explanation = explainer.explain_decision_with_features(
        decision="BLOCK",
        judgment_data=sample_judgment_data_violation
    )
    
    text = explanation.explanation_text
    
    # Should contain decision and confidence
    assert "BLOCK" in text
    assert "confidence" in text.lower()
    
    # Should contain key factors section
    assert "Key Contributing Factors" in text or "factors" in text.lower()
    
    # Should have reasonable length
    assert len(text) > 100
    
    # Should mention at least one feature
    feature_found = any(
        f.feature_name.replace("_", " ") in text.lower()
        for f in explanation.feature_importances
    )
    assert feature_found


def test_color_assignment(explainer):
    """Test color assignment for visualization."""
    assert explainer._get_color_for_impact(0.20) == "red"  # High risk
    assert explainer._get_color_for_impact(0.10) == "orange"  # Medium risk
    assert explainer._get_color_for_impact(0.0) == "gray"  # Neutral
    assert explainer._get_color_for_impact(-0.10) == "lightgreen"  # Medium safety
    assert explainer._get_color_for_impact(-0.20) == "green"  # Strong safety


def test_minimal_counterfactual(explainer, sample_judgment_data_violation):
    """Test that counterfactual suggests minimal changes."""
    counterfactual = explainer.generate_counterfactual(
        current_decision="BLOCK",
        judgment_data=sample_judgment_data_violation,
        desired_decision="RESTRICT"  # Smaller change
    )
    
    # Should have fewer changes for smaller decision change
    assert len(counterfactual.required_changes) <= 3


def test_feature_extraction(explainer, sample_judgment_data_violation):
    """Test that features are correctly extracted from judgment data."""
    features = explainer._extract_features(sample_judgment_data_violation)
    
    assert isinstance(features, dict)
    assert "risk_score" in features
    assert "violation_count" in features
    assert "pii_risk" in features
    assert "ethical_score" in features
    assert "quota_pressure" in features
    
    # Check values
    assert features["risk_score"] == 0.85
    assert features["violation_count"] == 2
    assert features["pii_risk"] == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
