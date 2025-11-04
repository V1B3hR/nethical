"""Tests for the explainability module."""

import pytest
from nethical.explainability import (
    DecisionExplainer,
    NaturalLanguageGenerator,
    TransparencyReportGenerator,
)
from datetime import datetime, timedelta


class TestDecisionExplainer:
    """Test cases for DecisionExplainer."""
    
    def test_explain_block_decision(self):
        """Test explanation for BLOCK decision."""
        explainer = DecisionExplainer()
        
        violated_rules = [
            {
                "name": "no_malicious_content",
                "severity": "critical",
                "description": "Prevents malicious content"
            }
        ]
        
        risk_scores = {
            "security": 0.8,
            "privacy": 0.6
        }
        
        explanation = explainer.explain_decision(
            decision="BLOCK",
            context={"agent_id": "test_agent"},
            violated_rules=violated_rules,
            risk_scores=risk_scores
        )
        
        assert explanation.decision == "BLOCK"
        assert "blocked" in explanation.summary.lower()
        assert len(explanation.components) > 0
        assert explanation.confidence > 0.0
    
    def test_explain_allow_decision(self):
        """Test explanation for ALLOW decision."""
        explainer = DecisionExplainer()
        
        explanation = explainer.explain_decision(
            decision="ALLOW",
            context={"agent_id": "test_agent"},
            risk_scores={"security": 0.1}
        )
        
        assert explanation.decision == "ALLOW"
        assert "allowed" in explanation.summary.lower()
    
    def test_json_serialization(self):
        """Test JSON serialization of explanation."""
        explainer = DecisionExplainer()
        
        explanation = explainer.explain_decision(
            decision="BLOCK",
            context={},
            violated_rules=[{"name": "test_rule", "severity": "high"}]
        )
        
        json_data = explainer.explain_to_json(explanation)
        
        assert "decision" in json_data
        assert "components" in json_data
        assert "confidence" in json_data


class TestNaturalLanguageGenerator:
    """Test cases for NaturalLanguageGenerator."""
    
    def test_generate_explanation(self):
        """Test natural language generation."""
        generator = NaturalLanguageGenerator()
        
        components = [
            {
                "type": "rule_based",
                "weight": 0.8,
                "details": {
                    "count": 2,
                    "violated_rules": [
                        {"rule_name": "test", "severity": "high", "description": "Test rule"}
                    ]
                }
            }
        ]
        
        explanation = generator.generate_explanation(
            decision="BLOCK",
            context={"action_type": "request"},
            components=components,
            reasoning_chain=["Step 1", "Step 2"]
        )
        
        assert explanation.title
        assert explanation.summary
        assert explanation.detailed_explanation
        assert len(explanation.key_points) > 0
    
    def test_markdown_conversion(self):
        """Test markdown conversion."""
        generator = NaturalLanguageGenerator()
        
        explanation = generator.generate_explanation(
            decision="ALLOW",
            context={},
            components=[],
            reasoning_chain=[]
        )
        
        markdown = generator.to_markdown(explanation)
        
        assert "# " in markdown  # Has headers
        assert "## " in markdown
    
    def test_html_conversion(self):
        """Test HTML conversion."""
        generator = NaturalLanguageGenerator()
        
        explanation = generator.generate_explanation(
            decision="RESTRICT",
            context={},
            components=[],
            reasoning_chain=[]
        )
        
        html = generator.to_html(explanation)
        
        assert "<h1>" in html
        assert "</h1>" in html


class TestTransparencyReportGenerator:
    """Test cases for TransparencyReportGenerator."""
    
    def test_generate_report(self):
        """Test report generation."""
        generator = TransparencyReportGenerator()
        
        # Create sample data
        now = datetime.now()
        decisions = [
            {
                "decision": "BLOCK",
                "timestamp": now.isoformat(),
                "agent_id": "agent1",
                "category": "security"
            },
            {
                "decision": "ALLOW",
                "timestamp": now.isoformat(),
                "agent_id": "agent2",
                "category": "normal"
            }
        ]
        
        violations = [
            {
                "type": "security",
                "severity": "high",
                "timestamp": now.isoformat()
            }
        ]
        
        policies = [
            {"name": "policy1"},
            {"name": "policy2"}
        ]
        
        report = generator.generate_report(
            decisions=decisions,
            violations=violations,
            policies=policies,
            period_days=7
        )
        
        assert report.report_id
        assert report.summary["total_decisions"] == 2
        assert report.summary["total_violations"] == 1
        assert len(report.key_insights) > 0
        assert len(report.recommendations) > 0
    
    def test_json_export(self):
        """Test JSON export."""
        generator = TransparencyReportGenerator()
        
        report = generator.generate_report(
            decisions=[],
            violations=[],
            policies=[],
            period_days=7
        )
        
        json_str = generator.to_json(report)
        
        assert "report_id" in json_str
        assert "summary" in json_str
    
    def test_markdown_export(self):
        """Test markdown export."""
        generator = TransparencyReportGenerator()
        
        report = generator.generate_report(
            decisions=[],
            violations=[],
            policies=[],
            period_days=7
        )
        
        markdown = generator.to_markdown(report)
        
        assert "# Transparency Report" in markdown
        assert "## Summary" in markdown
