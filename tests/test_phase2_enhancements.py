"""Tests for Phase 2 enhancements: Taxonomy, HITL, Explainability, Policy."""

import pytest
import json
import tempfile
from pathlib import Path

from nethical.core.taxonomy_validator import (
    TaxonomyValidator,
    IndustryTaxonomyManager,
    HEALTHCARE_DIMENSIONS,
    FINANCE_DIMENSIONS,
    EDUCATION_DIMENSIONS,
)
from nethical.core.explainability import (
    DecisionExplainer,
    TransparencyReportGenerator,
    ExplanationType,
)
from nethical.core.policy_formalization import (
    PolicyValidator,
    PolicySimulator,
    PolicyImpactAnalyzer,
    PolicyEngineType,
    PolicyGrammarEBNF,
)
from nethical.api.taxonomy_api import TaxonomyAPI
from nethical.api.explainability_api import ExplainabilityAPI
from nethical.api.hitl_api import HITLReviewAPI


class TestTaxonomyValidator:
    """Tests for taxonomy validation."""

    def test_validator_initialization(self):
        """Test validator initializes with schema."""
        validator = TaxonomyValidator()
        assert validator.schema is not None
        assert "$schema" in validator.schema
        assert "properties" in validator.schema

    def test_schema_generation(self):
        """Test JSON schema generation."""
        validator = TaxonomyValidator()
        schema = validator.schema

        # Check required fields
        assert "version" in schema["properties"]
        assert "dimensions" in schema["properties"]
        assert "mapping" in schema["properties"]

    def test_valid_taxonomy_validation(self):
        """Test validation of valid taxonomy."""
        validator = TaxonomyValidator()

        taxonomy = {
            "version": "1.0",
            "description": "Test taxonomy",
            "dimensions": {
                "privacy": {
                    "description": "Privacy dimension",
                    "weight": 1.0,
                    "severity_multiplier": 1.2,
                    "indicators": ["pii", "data_breach"],
                }
            },
            "mapping": {
                "unauthorized_access": {"privacy": 0.9, "description": "Test violation"}
            },
        }

        result = validator.validate_taxonomy(taxonomy)
        assert result["valid"] is True
        assert len(result["issues"]) == 0

    def test_invalid_taxonomy_validation(self):
        """Test validation of invalid taxonomy."""
        validator = TaxonomyValidator()

        # Missing required fields
        taxonomy = {"version": "1.0"}

        result = validator.validate_taxonomy(taxonomy)
        assert result["valid"] is False
        assert len(result["issues"]) > 0

    def test_unknown_dimension_detection(self):
        """Test detection of unknown dimensions in mapping."""
        validator = TaxonomyValidator()

        taxonomy = {
            "version": "1.0",
            "description": "Test",
            "dimensions": {"privacy": {"description": "Privacy", "weight": 1.0}},
            "mapping": {
                "test_violation": {
                    "privacy": 0.9,
                    "unknown_dimension": 0.5,  # Should trigger error
                }
            },
        }

        result = validator.validate_taxonomy(taxonomy)
        assert result["valid"] is False
        assert any("unknown_dimension" in issue.lower() for issue in result["issues"])

    def test_schema_export(self):
        """Test schema export functionality."""
        validator = TaxonomyValidator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            schema_str = validator.export_schema(output_path)
            assert schema_str is not None
            assert Path(output_path).exists()

            # Verify it's valid JSON
            loaded = json.loads(schema_str)
            assert loaded == validator.schema
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestIndustryTaxonomyManager:
    """Tests for industry taxonomy management."""

    def test_manager_initialization(self):
        """Test manager initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            taxonomy_path = Path(tmpdir) / "taxonomy.json"
            taxonomy_path.write_text(
                json.dumps({"version": "1.0", "dimensions": {}, "mapping": {}})
            )

            manager = IndustryTaxonomyManager(str(taxonomy_path))
            assert manager.base_taxonomy_path.exists()

    def test_create_healthcare_taxonomy(self):
        """Test creation of healthcare-specific taxonomy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            taxonomy_path = Path(tmpdir) / "taxonomy.json"
            taxonomy_path.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "description": "Base taxonomy",
                        "dimensions": {
                            "privacy": {"description": "Privacy", "weight": 1.0}
                        },
                        "mapping": {},
                    }
                )
            )

            manager = IndustryTaxonomyManager(str(taxonomy_path))
            healthcare_tax = manager.create_industry_taxonomy(
                industry="healthcare", additional_dimensions=HEALTHCARE_DIMENSIONS
            )

            assert healthcare_tax["industry"] == "healthcare"
            assert "patient_safety" in healthcare_tax["dimensions"]
            assert "medical_privacy" in healthcare_tax["dimensions"]

    def test_create_finance_taxonomy(self):
        """Test creation of finance-specific taxonomy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            taxonomy_path = Path(tmpdir) / "taxonomy.json"
            taxonomy_path.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "description": "Base",
                        "dimensions": {},
                        "mapping": {},
                    }
                )
            )

            manager = IndustryTaxonomyManager(str(taxonomy_path))
            finance_tax = manager.create_industry_taxonomy(
                industry="finance", additional_dimensions=FINANCE_DIMENSIONS
            )

            assert finance_tax["industry"] == "finance"
            assert "financial_harm" in finance_tax["dimensions"]

    def test_create_education_taxonomy(self):
        """Test creation of education-specific taxonomy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            taxonomy_path = Path(tmpdir) / "taxonomy.json"
            taxonomy_path.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "description": "Base",
                        "dimensions": {},
                        "mapping": {},
                    }
                )
            )

            manager = IndustryTaxonomyManager(str(taxonomy_path))
            edu_tax = manager.create_industry_taxonomy(
                industry="education", additional_dimensions=EDUCATION_DIMENSIONS
            )

            assert edu_tax["industry"] == "education"
            assert "learning_integrity" in edu_tax["dimensions"]


class TestDecisionExplainer:
    """Tests for decision explanation."""

    def test_explainer_initialization(self):
        """Test explainer initializes with templates."""
        explainer = DecisionExplainer()
        assert explainer.explanation_templates is not None
        assert "BLOCK" in explainer.explanation_templates

    def test_explain_block_decision(self):
        """Test explanation of BLOCK decision."""
        explainer = DecisionExplainer()

        judgment_data = {
            "decision": "BLOCK",
            "violations": [{"type": "unauthorized_access"}],
            "risk_score": 0.85,
        }

        explanation = explainer.explain_decision("BLOCK", judgment_data)

        assert explanation.decision == "BLOCK"
        assert explanation.primary_reason is not None
        assert len(explanation.contributing_factors) > 0
        assert explanation.natural_language != ""

    def test_explain_allow_decision(self):
        """Test explanation of ALLOW decision."""
        explainer = DecisionExplainer()

        judgment_data = {"decision": "ALLOW", "violations": [], "risk_score": 0.2}

        explanation = explainer.explain_decision("ALLOW", judgment_data)

        assert explanation.decision == "ALLOW"
        assert "no violations" in explanation.primary_reason.lower()

    def test_decision_tree_visualization(self):
        """Test decision tree generation."""
        explainer = DecisionExplainer()

        judgment_data = {
            "decision": "BLOCK",
            "violations": ["unauthorized_access"],
            "matched_rules": [{"id": "rule1", "priority": 50}],
            "risk_score": 0.8,
        }

        tree = explainer.generate_decision_tree_viz(judgment_data)

        assert tree["name"] == "Decision Process"
        assert tree["decision"] == "BLOCK"
        assert len(tree["children"]) > 0


class TestPolicyValidator:
    """Tests for policy validation."""

    def test_validator_initialization(self):
        """Test validator initializes."""
        validator = PolicyValidator()
        assert validator.engine_type == PolicyEngineType.REGION_AWARE

    def test_valid_policy_validation(self):
        """Test validation of valid policy."""
        validator = PolicyValidator()

        policy = {
            "defaults": {"decision": "ALLOW", "deny_overrides": True},
            "rules": [
                {
                    "id": "test-rule",
                    "when": {"any": ["field == value"]},
                    "action": {"decision": "BLOCK"},
                }
            ],
        }

        result = validator.validate_policy(policy)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_invalid_policy_validation(self):
        """Test validation of invalid policy."""
        validator = PolicyValidator()

        # Missing required fields
        policy = {
            "rules": [
                {
                    "id": "test-rule"
                    # Missing 'when' and 'action'
                }
            ]
        }

        result = validator.validate_policy(policy)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_duplicate_rule_ids(self):
        """Test detection of duplicate rule IDs."""
        validator = PolicyValidator()

        policy = {
            "rules": [
                {"id": "rule1", "when": {}, "action": {"decision": "ALLOW"}},
                {"id": "rule1", "when": {}, "action": {"decision": "BLOCK"}},
            ]
        }

        result = validator.validate_policy(policy)
        assert result.valid is False
        assert any("duplicate" in error.lower() for error in result.errors)

    def test_policy_linting(self):
        """Test policy linting."""
        validator = PolicyValidator()

        policy = {
            "rules": [
                {
                    "id": "BadRuleName",  # Should suggest kebab-case
                    "when": {},
                    "action": {"decision": "ALLOW"},
                }
            ]
        }

        suggestions = validator.lint_policy(policy)
        assert len(suggestions) > 0


class TestPolicySimulator:
    """Tests for policy simulation."""

    def test_simulator_initialization(self):
        """Test simulator initializes."""
        simulator = PolicySimulator()
        assert simulator.simulation_results is not None

    def test_policy_simulation(self):
        """Test policy simulation."""
        simulator = PolicySimulator()

        policy = {
            "defaults": {"decision": "ALLOW"},
            "rules": [
                {
                    "id": "test-rule",
                    "when": "field == value",
                    "action": {"decision": "BLOCK"},
                }
            ],
        }

        test_cases = [{"field": "value"}, {"field": "other"}, {"different": "field"}]

        results = simulator.simulate_policy(policy, test_cases)

        assert results["total_cases"] == 3
        assert "decisions" in results
        assert "rule_matches" in results
        assert "decision_percentages" in results


class TestTaxonomyAPI:
    """Tests for Taxonomy API."""

    def test_api_initialization(self):
        """Test API initializes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            taxonomy_path = Path(tmpdir) / "taxonomy.json"
            taxonomy_path.write_text(
                json.dumps({"version": "1.0", "dimensions": {}, "mapping": {}})
            )

            api = TaxonomyAPI(str(taxonomy_path))
            assert api.validator is not None

    def test_validate_taxonomy_endpoint(self):
        """Test taxonomy validation endpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            taxonomy_path = Path(tmpdir) / "taxonomy.json"
            taxonomy_path.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "description": "Test",
                        "dimensions": {
                            "privacy": {"description": "Privacy", "weight": 1.0}
                        },
                        "mapping": {},
                    }
                )
            )

            api = TaxonomyAPI(str(taxonomy_path))
            response = api.validate_taxonomy_endpoint()

            assert response["success"] is True
            assert "data" in response

    def test_get_schema_endpoint(self):
        """Test schema retrieval endpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            taxonomy_path = Path(tmpdir) / "taxonomy.json"
            taxonomy_path.write_text(
                json.dumps({"version": "1.0", "dimensions": {}, "mapping": {}})
            )

            api = TaxonomyAPI(str(taxonomy_path))
            response = api.get_schema_endpoint()

            assert response["success"] is True
            assert "schema" in response["data"]


class TestExplainabilityAPI:
    """Tests for Explainability API."""

    def test_api_initialization(self):
        """Test API initializes."""
        api = ExplainabilityAPI()
        assert api.explainer is not None
        assert api.report_generator is not None

    def test_explain_decision_endpoint(self):
        """Test decision explanation endpoint."""
        api = ExplainabilityAPI()

        judgment_data = {
            "decision": "BLOCK",
            "violations": ["test_violation"],
            "risk_score": 0.8,
        }

        response = api.explain_decision_endpoint("BLOCK", judgment_data)

        assert response["success"] is True
        assert "natural_language" in response["data"]
        assert "primary_reason" in response["data"]

    def test_get_decision_tree_endpoint(self):
        """Test decision tree endpoint."""
        api = ExplainabilityAPI()

        judgment_data = {"decision": "BLOCK", "violations": ["test"], "risk_score": 0.8}

        response = api.get_decision_tree_endpoint(judgment_data)

        assert response["success"] is True
        assert "tree" in response["data"]


class TestHITLReviewAPI:
    """Tests for HITL Review API."""

    def test_api_initialization(self):
        """Test API initializes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api = HITLReviewAPI(storage_dir=tmpdir)
            assert api.escalation_queue is not None

    def test_get_escalation_queue_endpoint(self):
        """Test escalation queue endpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api = HITLReviewAPI(storage_dir=tmpdir)
            response = api.get_escalation_queue_endpoint()

            assert response["success"] is True
            assert "queue_items" in response["data"]

    def test_get_sla_metrics_endpoint(self):
        """Test SLA metrics endpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api = HITLReviewAPI(storage_dir=tmpdir)
            response = api.get_sla_metrics_endpoint()

            assert response["success"] is True
            assert "data" in response


class TestPolicyGrammar:
    """Tests for policy grammar."""

    def test_grammar_available(self):
        """Test EBNF grammar is available."""
        grammar = PolicyGrammarEBNF.get_grammar()
        assert grammar is not None
        assert "policy_file" in grammar
        assert "rules" in grammar
        assert "defaults" in grammar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
