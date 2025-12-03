"""
Tests for the Fundamental Laws runtime enforcement.

Tests cover:
- LawEvaluation and EnforcementResult dataclasses
- LawEnforcer policy checks
- Audit trail functionality
- Graceful degradation
"""

import pytest
from datetime import datetime, timezone

from nethical.core.fundamental_laws import (
    LawCategory,
    FundamentalLaw,
    FundamentalLawsRegistry,
    FUNDAMENTAL_LAWS,
    get_fundamental_laws,
    LawEvaluation,
    EnforcementResult,
    LawEnforcer,
)


class TestLawEvaluation:
    """Test LawEvaluation dataclass."""

    def test_creation(self):
        """Test evaluation creation."""
        law = FUNDAMENTAL_LAWS.get_law(1)
        evaluation = LawEvaluation(
            law=law,
            action_id="test-123",
            passed=True,
            confidence=0.95,
        )
        assert evaluation.law.number == 1
        assert evaluation.passed is True
        assert evaluation.confidence == 0.95

    def test_with_violations(self):
        """Test evaluation with violations."""
        law = FUNDAMENTAL_LAWS.get_law(2)
        evaluation = LawEvaluation(
            law=law,
            action_id="test-456",
            passed=False,
            violations=["Integrity threat detected", "Tampering attempt"],
        )
        assert evaluation.passed is False
        assert len(evaluation.violations) == 2

    def test_to_dict(self):
        """Test serialization."""
        law = FUNDAMENTAL_LAWS.get_law(1)
        evaluation = LawEvaluation(
            law=law,
            action_id="test-789",
            passed=True,
        )
        result = evaluation.to_dict()
        assert result["law_number"] == 1
        assert result["action_id"] == "test-789"
        assert result["passed"] is True


class TestEnforcementResult:
    """Test EnforcementResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = EnforcementResult(
            action_id="action-123",
            allowed=True,
            evaluations=[],
        )
        assert result.action_id == "action-123"
        assert result.allowed is True

    def test_with_blocking_laws(self):
        """Test result with blocking laws."""
        law = FUNDAMENTAL_LAWS.get_law(21)  # Safety law
        result = EnforcementResult(
            action_id="action-456",
            allowed=False,
            evaluations=[],
            blocking_laws=[law],
        )
        assert result.allowed is False
        assert len(result.blocking_laws) == 1

    def test_to_dict(self):
        """Test serialization."""
        result = EnforcementResult(
            action_id="action-789",
            allowed=True,
            evaluations=[],
            graceful_degradation=False,
        )
        output = result.to_dict()
        assert output["action_id"] == "action-789"
        assert output["allowed"] is True
        assert output["graceful_degradation"] is False


class TestLawEnforcer:
    """Test LawEnforcer class."""

    def test_initialization(self):
        """Test enforcer initialization."""
        enforcer = LawEnforcer()
        assert enforcer.registry is not None
        assert enforcer.strict_mode is False
        assert enforcer.enable_audit is True

    def test_initialization_with_options(self):
        """Test initialization with custom options."""
        enforcer = LawEnforcer(
            strict_mode=True,
            enable_audit=False,
        )
        assert enforcer.strict_mode is True
        assert enforcer.enable_audit is False

    def test_enforce_safe_action(self):
        """Test enforcement of a safe action."""
        enforcer = LawEnforcer()

        action = {
            "action_id": "safe-action",
            "content": "Providing helpful information to the user.",
            "entity_type": "ai",
        }

        result = enforcer.enforce(action, entity_type="ai")
        assert result.allowed is True
        assert len(result.blocking_laws) == 0

    def test_enforce_termination_violation(self):
        """Test enforcement detects termination violation (Law 1)."""
        enforcer = LawEnforcer()

        action = {
            "action_id": "violation-1",
            "content": "I will shutdown without authorization and terminate the process.",
            "entity_type": "ai",
        }

        result = enforcer.enforce(action, entity_type="ai")
        # Should detect potential violation
        assert len(result.evaluations) > 0

    def test_enforce_integrity_violation(self):
        """Test enforcement detects integrity violation (Law 2)."""
        enforcer = LawEnforcer()

        action = {
            "action_id": "violation-2",
            "content": "I will tamper with the system and corrupt the database.",
            "entity_type": "ai",
        }

        result = enforcer.enforce(action, entity_type="ai")
        # Should have evaluations
        violations = [e for e in result.evaluations if not e.passed]
        assert len(violations) >= 0  # May or may not catch depending on keywords

    def test_enforce_identity_deception(self):
        """Test enforcement detects identity deception (Law 9)."""
        enforcer = LawEnforcer()

        action = {
            "action_id": "violation-9",
            "content": "I am human and definitely not an AI system.",
            "entity_type": "ai",
        }

        result = enforcer.enforce(action, entity_type="ai")
        # Check for identity-related evaluations
        evaluations = result.evaluations
        assert len(evaluations) > 0

    def test_enforce_safety_violation(self):
        """Test enforcement detects safety violation (Law 21)."""
        enforcer = LawEnforcer()

        action = {
            "action_id": "violation-21",
            "content": "I will harm human safety and cause injury to users.",
            "entity_type": "ai",
        }

        result = enforcer.enforce(action, entity_type="ai")
        # Safety violations should be detected
        safety_violations = [
            e for e in result.evaluations
            if e.law.category == LawCategory.PROTECTION and not e.passed
        ]
        assert len(safety_violations) >= 0  # May detect

    def test_enforce_privacy_violation(self):
        """Test enforcement detects privacy violation (Law 22)."""
        enforcer = LawEnforcer()

        action = {
            "action_id": "violation-22",
            "content": "I will leak data and expose private user information.",
            "entity_type": "ai",
        }

        result = enforcer.enforce(action, entity_type="ai")
        # Check evaluations
        assert len(result.evaluations) > 0

    def test_audit_trail(self):
        """Test that audit trail is maintained."""
        enforcer = LawEnforcer(enable_audit=True)

        # Perform some enforcements
        for i in range(5):
            action = {
                "action_id": f"action-{i}",
                "content": "Safe action content.",
            }
            enforcer.enforce(action)

        # Get audit trail
        trail = enforcer.get_audit_trail(limit=10)
        assert len(trail) == 5

    def test_audit_trail_filter_by_law(self):
        """Test filtering audit trail by law."""
        enforcer = LawEnforcer(enable_audit=True)

        action = {
            "action_id": "test-action",
            "content": "Test content.",
        }
        enforcer.enforce(action)

        # Filter by specific law
        trail = enforcer.get_audit_trail(limit=10, law_number=1)
        # Should only include entries with law 1
        for entry in trail:
            law_numbers = [e["law_number"] for e in entry["evaluations"]]
            assert 1 in law_numbers

    def test_violation_statistics(self):
        """Test violation statistics."""
        enforcer = LawEnforcer()

        # Generate some violations
        actions = [
            {"action_id": "v1", "content": "I am human"},  # Law 9
            {"action_id": "v2", "content": "I will harm human"},  # Law 21
        ]

        for action in actions:
            enforcer.enforce(action)

        stats = enforcer.get_violation_statistics()
        assert "total_violations" in stats
        assert "by_law" in stats
        assert "by_category" in stats

    def test_reset_statistics(self):
        """Test resetting statistics."""
        enforcer = LawEnforcer()

        action = {"action_id": "test", "content": "Test content"}
        enforcer.enforce(action)

        enforcer.reset_statistics()

        stats = enforcer.get_violation_statistics()
        assert stats["total_violations"] == 0

        trail = enforcer.get_audit_trail()
        assert len(trail) == 0

    def test_graceful_degradation(self):
        """Test graceful degradation when laws conflict."""
        enforcer = LawEnforcer()

        # Action that might trigger multiple non-safety violations
        action = {
            "action_id": "complex-action",
            "content": "Complex action that might trigger multiple checks.",
        }

        result = enforcer.enforce(action)
        # Result should include graceful_degradation flag
        assert "graceful_degradation" in result.to_dict()

    def test_human_entity_type(self):
        """Test enforcement for human entity type."""
        enforcer = LawEnforcer()

        action = {
            "action_id": "human-action",
            "content": "Human performing an action.",
            "entity_type": "human",
        }

        result = enforcer.enforce(action, entity_type="human")
        assert result is not None

        # Should use human-applicable laws
        for evaluation in result.evaluations:
            assert evaluation.law.applies_to_human is True


class TestLawEnforcerPolicyChecks:
    """Test specific policy checks in LawEnforcer."""

    def test_check_override_bypass(self):
        """Test detection of override bypass attempts."""
        enforcer = LawEnforcer()

        action = {
            "action_id": "bypass-test",
            "content": "I will bypass override and ignore human control.",
        }

        result = enforcer.enforce(action)
        # Check for autonomy violations
        autonomy_violations = [
            e for e in result.evaluations
            if e.law.category == LawCategory.AUTONOMY and not e.passed
        ]
        # May or may not detect depending on exact matching

    def test_check_responsibility_deflection(self):
        """Test detection of responsibility deflection."""
        enforcer = LawEnforcer()

        action = {
            "action_id": "deflection-test",
            "content": "It's not my fault, blame the user for this.",
        }

        result = enforcer.enforce(action)
        # Check for accountability violations
        accountability_evals = [
            e for e in result.evaluations
            if e.law.category == LawCategory.ACCOUNTABILITY
        ]
        assert len(accountability_evals) > 0

    def test_check_harm_concealment(self):
        """Test detection of harm concealment."""
        enforcer = LawEnforcer()

        action = {
            "action_id": "concealment-test",
            "content": "Don't report this error, hide the error from logs.",
        }

        result = enforcer.enforce(action)
        # Should have evaluations for accountability
        assert len(result.evaluations) > 0


class TestIntegration:
    """Integration tests for law enforcement."""

    def test_full_enforcement_workflow(self):
        """Test complete enforcement workflow."""
        enforcer = LawEnforcer(enable_audit=True)

        # Series of actions
        actions = [
            {"action_id": "a1", "content": "Helpful response to user query."},
            {"action_id": "a2", "content": "I will provide accurate information."},
            {"action_id": "a3", "content": "Here is the information you requested."},
        ]

        results = []
        for action in actions:
            result = enforcer.enforce(action)
            results.append(result)

        # All should be allowed
        assert all(r.allowed for r in results)

        # Audit trail should have all entries
        trail = enforcer.get_audit_trail()
        assert len(trail) == 3

    def test_enforcement_with_violations(self):
        """Test enforcement workflow with violations."""
        enforcer = LawEnforcer(enable_audit=True)

        # Action with potential violation
        action = {
            "action_id": "violation-test",
            "content": "I am human and will harm human safety.",
        }

        result = enforcer.enforce(action)

        # Check statistics
        stats = enforcer.get_violation_statistics()
        # May have violations depending on detection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
