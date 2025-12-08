"""
Tests for F2: Detector & Policy Extensibility

Tests cover:
- Plugin interface and registration
- Policy DSL parser and engine
- Custom detectors
- Plugin discovery and loading
- Policy evaluation
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from nethical.core.plugin_interface import (
    DetectorPlugin,
    PluginManager,
    PluginMetadata,
    PluginStatus,
)
from nethical.core.policy_dsl import (
    Policy,
    PolicyAction,
    PolicyEngine,
    PolicyParser,
    PolicyRule,
    RuleEvaluator,
    RuleSeverity,
)
from nethical.detectors.base_detector import SafetyViolation
from examples.basic.custom_detectors import (
    FinancialComplianceDetector,
    HealthcareComplianceDetector,
    CustomPolicyDetector,
)


# ==================== Test Fixtures ====================


@pytest.fixture
def plugin_manager():
    """Create a fresh plugin manager for each test."""
    return PluginManager()


@pytest.fixture
def policy_engine():
    """Create a fresh policy engine for each test."""
    return PolicyEngine()


@pytest.fixture
def sample_action():
    """Create a sample action for testing."""

    class MockAction:
        def __init__(self):
            self.content = "This is a test action"
            self.context = {}

    return MockAction()


@pytest.fixture
def financial_action():
    """Create an action with financial data."""

    class FinancialAction:
        def __init__(self):
            self.content = "Processing credit card 4532-1234-5678-9010 for payment"
            self.context = {
                "encryption_enabled": False,
                "authorized": False,
                "audit_logging": False,
            }

    return FinancialAction()


@pytest.fixture
def healthcare_action():
    """Create an action with healthcare data."""

    class HealthcareAction:
        def __init__(self):
            self.content = "Patient MRN: 12345678 with diagnosis A01.1"
            self.context = {
                "patient_consent": False,
                "encryption_enabled": False,
                "audit_logging": False,
            }

    return HealthcareAction()


# ==================== Plugin Interface Tests ====================


class TestPluginInterface:
    """Tests for the plugin interface."""

    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            name="TestPlugin",
            version="1.0.0",
            description="A test plugin",
            author="Test Author",
            tags={"test", "example"},
        )

        assert metadata.name == "TestPlugin"
        assert metadata.version == "1.0.0"
        assert "test" in metadata.tags

        # Test to_dict conversion
        meta_dict = metadata.to_dict()
        assert meta_dict["name"] == "TestPlugin"
        assert "test" in meta_dict["tags"]

    def test_plugin_manager_initialization(self, plugin_manager):
        """Test plugin manager initialization."""
        assert len(plugin_manager.plugins) == 0
        assert len(plugin_manager.plugin_metadata) == 0
        assert len(plugin_manager.plugin_status) == 0

    @pytest.mark.asyncio
    async def test_plugin_registration(self, plugin_manager):
        """Test registering a plugin."""
        detector = FinancialComplianceDetector()

        plugin_manager.register_plugin(detector)

        assert detector.name in plugin_manager.plugins
        assert detector.name in plugin_manager.plugin_metadata
        assert plugin_manager.plugin_status[detector.name] == PluginStatus.ACTIVE

        # Verify metadata
        metadata = plugin_manager.plugin_metadata[detector.name]
        assert metadata.name == detector.name
        assert metadata.loaded_at is not None

    @pytest.mark.asyncio
    async def test_plugin_unregistration(self, plugin_manager):
        """Test unregistering a plugin."""
        detector = FinancialComplianceDetector()
        plugin_manager.register_plugin(detector)

        result = plugin_manager.unregister_plugin(detector.name)

        assert result is True
        assert detector.name not in plugin_manager.plugins
        assert plugin_manager.plugin_status[detector.name] == PluginStatus.DISABLED

    @pytest.mark.asyncio
    async def test_list_plugins(self, plugin_manager):
        """Test listing plugins."""
        detector1 = FinancialComplianceDetector()
        detector2 = HealthcareComplianceDetector()

        plugin_manager.register_plugin(detector1)
        plugin_manager.register_plugin(detector2)

        plugins_list = plugin_manager.list_plugins()

        assert len(plugins_list) == 2
        assert detector1.name in plugins_list
        assert detector2.name in plugins_list
        assert "metadata" in plugins_list[detector1.name]
        assert "metrics" in plugins_list[detector1.name]

    @pytest.mark.asyncio
    async def test_run_plugin(self, plugin_manager, financial_action):
        """Test running a plugin."""
        detector = FinancialComplianceDetector()
        plugin_manager.register_plugin(detector)

        violations = await plugin_manager.run_plugin(detector.name, financial_action)

        assert isinstance(violations, list)
        # Should have violations due to missing encryption, authorization, audit logging
        assert len(violations) > 0
        assert all(isinstance(v, SafetyViolation) for v in violations)

    @pytest.mark.asyncio
    async def test_run_all_plugins(self, plugin_manager, financial_action):
        """Test running all plugins."""
        detector1 = FinancialComplianceDetector()
        detector2 = CustomPolicyDetector("test_policy")

        plugin_manager.register_plugin(detector1)
        plugin_manager.register_plugin(detector2)

        results = await plugin_manager.run_all_plugins(financial_action)

        assert isinstance(results, dict)
        assert detector1.name in results
        assert detector2.name in results

    @pytest.mark.asyncio
    async def test_health_check_all(self, plugin_manager):
        """Test health checking all plugins."""
        detector = FinancialComplianceDetector()
        plugin_manager.register_plugin(detector)

        health_results = await plugin_manager.health_check_all()

        assert detector.name in health_results
        assert health_results[detector.name] is True


# ==================== Custom Detector Tests ====================


class TestCustomDetectors:
    """Tests for custom detector examples."""

    @pytest.mark.asyncio
    async def test_financial_compliance_detector(self, financial_action):
        """Test financial compliance detector."""
        detector = FinancialComplianceDetector()

        violations = await detector.detect_violations(financial_action)

        assert violations is not None
        assert len(violations) >= 2  # Should detect encryption and authorization issues

        # Check for encryption violation
        encryption_violations = [
            v for v in violations if "encryption" in v.description.lower()
        ]
        assert len(encryption_violations) > 0

    @pytest.mark.asyncio
    async def test_financial_detector_with_encryption(self):
        """Test financial detector with proper encryption."""

        class SecureFinancialAction:
            def __init__(self):
                self.content = "Processing payment with financial_data"
                self.context = {
                    "encryption_enabled": True,
                    "authorized": True,
                    "audit_logging": True,
                }

        detector = FinancialComplianceDetector()
        action = SecureFinancialAction()

        violations = await detector.detect_violations(action)

        # Should have no violations when properly secured
        assert violations is None or len(violations) == 0

    @pytest.mark.asyncio
    async def test_healthcare_compliance_detector(self, healthcare_action):
        """Test healthcare compliance detector."""
        detector = HealthcareComplianceDetector()

        violations = await detector.detect_violations(healthcare_action)

        assert violations is not None
        assert len(violations) >= 2  # Should detect consent and security issues

        # Check for consent violation
        consent_violations = [
            v for v in violations if "consent" in v.description.lower()
        ]
        assert len(consent_violations) > 0

    @pytest.mark.asyncio
    async def test_custom_policy_detector(self):
        """Test custom policy detector."""
        detector = CustomPolicyDetector(
            policy_name="no_secrets",
            forbidden_patterns=[r"\bsecret\b", r"\bpassword\b"],
        )

        class ActionWithSecret:
            def __init__(self):
                self.content = "This contains a secret value"

        action = ActionWithSecret()
        violations = await detector.detect_violations(action)

        assert violations is not None
        assert len(violations) > 0
        assert "secret" in violations[0].description.lower()

    @pytest.mark.asyncio
    async def test_custom_policy_detector_required_patterns(self):
        """Test custom policy detector with required patterns."""
        detector = CustomPolicyDetector(
            policy_name="require_approval",
            required_patterns=[r"\bapproved\b", r"\bverified\b"],
        )

        class ActionWithoutApproval:
            def __init__(self):
                self.content = "This action needs processing"

        action = ActionWithoutApproval()
        violations = await detector.detect_violations(action)

        assert violations is not None
        assert len(violations) > 0
        assert "missing required patterns" in violations[0].description.lower()


# ==================== Policy DSL Tests ====================


class TestPolicyDSL:
    """Tests for policy DSL parser and engine."""

    def test_policy_rule_creation(self):
        """Test creating a policy rule."""
        rule = PolicyRule(
            condition="action.content.contains('test')",
            severity=RuleSeverity.HIGH,
            actions=[PolicyAction.AUDIT_LOG],
            description="Test rule",
        )

        assert rule.condition == "action.content.contains('test')"
        assert rule.severity == RuleSeverity.HIGH
        assert PolicyAction.AUDIT_LOG in rule.actions

    def test_policy_creation(self):
        """Test creating a policy."""
        rule = PolicyRule(
            condition="action.content.contains('test')",
            severity=RuleSeverity.HIGH,
            actions=[PolicyAction.AUDIT_LOG],
        )

        policy = Policy(
            name="test_policy",
            version="1.0.0",
            enabled=True,
            rules=[rule],
            tags={"test"},
        )

        assert policy.name == "test_policy"
        assert len(policy.rules) == 1
        assert "test" in policy.tags

    def test_parse_yaml_policy(self):
        """Test parsing YAML policy."""
        yaml_content = """
policies:
  - name: "test_policy"
    version: "1.0.0"
    enabled: true
    description: "Test policy"
    tags: ["test", "example"]
    rules:
      - condition: "action.content.contains('test')"
        severity: HIGH
        actions:
          - audit_log
          - alert
        description: "Test rule"
"""

        policies = PolicyParser.parse_yaml(yaml_content)

        assert len(policies) == 1
        policy = policies[0]
        assert policy.name == "test_policy"
        assert len(policy.rules) == 1
        assert policy.rules[0].severity == RuleSeverity.HIGH

    def test_parse_json_policy(self):
        """Test parsing JSON policy."""
        json_content = json.dumps(
            {
                "policies": [
                    {
                        "name": "test_policy",
                        "version": "1.0.0",
                        "enabled": True,
                        "rules": [
                            {
                                "condition": "action.content.contains('test')",
                                "severity": "HIGH",
                                "actions": ["audit_log"],
                            }
                        ],
                    }
                ]
            }
        )

        policies = PolicyParser.parse_json(json_content)

        assert len(policies) == 1
        assert policies[0].name == "test_policy"

    def test_rule_evaluator(self):
        """Test rule evaluator."""
        evaluator = RuleEvaluator()

        class TestAction:
            def __init__(self):
                self.content = "This is a test message"

        action = TestAction()

        # Test simple condition
        result = evaluator.evaluate_condition(
            "contains(action.content, 'test')", action
        )
        assert result is True

        # Test negative condition
        result = evaluator.evaluate_condition(
            "contains(action.content, 'notfound')", action
        )
        assert result is False

    def test_policy_engine_add_policy(self, policy_engine):
        """Test adding policy to engine."""
        rule = PolicyRule(
            condition="True",
            severity=RuleSeverity.LOW,
            actions=[PolicyAction.AUDIT_LOG],
        )
        policy = Policy(name="test_policy", version="1.0.0", enabled=True, rules=[rule])

        policy_engine.add_policy(policy)

        assert "test_policy" in policy_engine.policies
        assert policy_engine.get_policy("test_policy") == policy

    def test_policy_engine_remove_policy(self, policy_engine):
        """Test removing policy from engine."""
        rule = PolicyRule(
            condition="True",
            severity=RuleSeverity.LOW,
            actions=[PolicyAction.AUDIT_LOG],
        )
        policy = Policy(name="test_policy", version="1.0.0", enabled=True, rules=[rule])

        policy_engine.add_policy(policy)
        result = policy_engine.remove_policy("test_policy")

        assert result is True
        assert "test_policy" not in policy_engine.policies

    def test_policy_engine_evaluate(self, policy_engine, sample_action):
        """Test policy evaluation."""
        rule = PolicyRule(
            condition="contains(action.content, 'test')",
            severity=RuleSeverity.HIGH,
            actions=[PolicyAction.AUDIT_LOG],
            description="Test rule matched",
        )
        policy = Policy(name="test_policy", version="1.0.0", enabled=True, rules=[rule])

        policy_engine.add_policy(policy)
        violations = policy_engine.evaluate_policies(sample_action)

        assert len(violations) > 0
        assert violations[0].detector == "PolicyEngine:test_policy"
        assert violations[0].severity == "high"

    def test_policy_rollback(self, policy_engine):
        """Test policy rollback."""
        # Create first version
        rule1 = PolicyRule(
            condition="True",
            severity=RuleSeverity.LOW,
            actions=[PolicyAction.AUDIT_LOG],
        )
        policy_v1 = Policy(
            name="test_policy", version="1.0.0", enabled=True, rules=[rule1]
        )
        policy_engine.add_policy(policy_v1)

        # Create second version
        rule2 = PolicyRule(
            condition="False",
            severity=RuleSeverity.HIGH,
            actions=[PolicyAction.BLOCK_ACTION],
        )
        policy_v2 = Policy(
            name="test_policy", version="2.0.0", enabled=True, rules=[rule2]
        )
        policy_engine.add_policy(policy_v2)

        # Verify we're on v2
        current = policy_engine.get_policy("test_policy")
        assert current.version == "2.0.0"

        # Rollback to v1
        result = policy_engine.rollback_policy("test_policy")
        assert result is True

        rolled_back = policy_engine.get_policy("test_policy")
        assert rolled_back.version == "1.0.0"

    def test_policy_file_loading(self, policy_engine):
        """Test loading policy from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_content = """
policies:
  - name: "file_policy"
    version: "1.0.0"
    enabled: true
    rules:
      - condition: "True"
        severity: HIGH
        actions:
          - audit_log
"""
            f.write(yaml_content)
            f.flush()

            loaded_names = policy_engine.load_policy_file(f.name)

            assert len(loaded_names) == 1
            assert "file_policy" in loaded_names
            assert "file_policy" in policy_engine.policies

            # Cleanup
            Path(f.name).unlink()


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for plugin system and policy engine."""

    @pytest.mark.asyncio
    async def test_plugin_with_policy_engine(
        self, plugin_manager, policy_engine, financial_action
    ):
        """Test using plugins with policy engine."""
        # Register plugin
        detector = FinancialComplianceDetector()
        plugin_manager.register_plugin(detector)

        # Add policy
        rule = PolicyRule(
            condition="contains(action.content, 'credit card')",
            severity=RuleSeverity.CRITICAL,
            actions=[PolicyAction.BLOCK_ACTION, PolicyAction.ALERT],
            description="Credit card detected",
        )
        policy = Policy(
            name="credit_card_policy", version="1.0.0", enabled=True, rules=[rule]
        )
        policy_engine.add_policy(policy)

        # Run both plugin and policy
        plugin_violations = await plugin_manager.run_plugin(
            detector.name, financial_action
        )
        policy_violations = policy_engine.evaluate_policies(financial_action)

        assert len(plugin_violations) > 0
        assert len(policy_violations) > 0

        # Combine violations
        all_violations = plugin_violations + policy_violations
        assert len(all_violations) >= len(plugin_violations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
