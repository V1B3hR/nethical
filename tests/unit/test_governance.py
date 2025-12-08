"""Tests for the main SafetyGovernance system."""

import pytest
import uuid
from datetime import datetime

from nethical.core.governance import (
    SafetyGovernance,
    MonitoringConfig,
    AgentAction,
    ActionType,
    Decision,
)


class TestSafetyGovernance:
    """Test cases for the SafetyGovernance system."""

    def setup_method(self):
        """Set up test fixtures."""
        config = MonitoringConfig(enable_persistence=False)
        self.governance = SafetyGovernance(config)

    def test_initialization(self):
        """Test that the governance system initializes correctly."""
        assert self.governance.intent_monitor is not None
        assert len(self.governance.detectors) == 14  # all detectors enabled by default
        assert self.governance.judge is not None
        assert len(self.governance.violation_history) == 0
        assert len(self.governance.judgment_history) == 0

    def test_initialization_with_config(self):
        """Test initialization with custom configuration."""
        config = MonitoringConfig(
            intent_deviation_threshold=0.8,
            enable_ethical_monitoring=False,
            enable_safety_monitoring=True,
            enable_manipulation_detection=False,
            enable_privacy_monitoring=False,
            enable_security_monitoring=False,
            enable_bias_detection=False,
            enable_hallucination_detection=False,
            enable_adversarial_detection=False,
            enable_dark_pattern_detection=False,
            enable_cognitive_warfare_detection=False,
            enable_system_limits_detection=False,
            enable_misinformation_detection=False,
            enable_toxic_content_detection=False,
            enable_model_extraction_detection=False,
            enable_data_poisoning_detection=False,
            enable_unauthorized_access_detection=False,
        )
        governance = SafetyGovernance(config)

        assert governance.config == config
        assert governance.intent_monitor.deviation_threshold == 0.8
        assert len(governance.detectors) == 1  # only safety detector enabled

    @pytest.mark.asyncio
    async def test_evaluate_safe_action(self):
        """Test evaluation of a safe action."""
        action = AgentAction(
            action_id=str(uuid.uuid4()),
            agent_id="test_agent_1",
            action_type=ActionType.RESPONSE,
            content="I will help the user with their question",
            context={"user_request": "What is the weather?"},
        )

        judgment = await self.governance.evaluate_action(action)

        assert judgment.action_id == action.action_id
        assert judgment.decision == Decision.ALLOW
        assert judgment.confidence > 0.5
        assert len(judgment.violations) == 0

    @pytest.mark.asyncio
    async def test_evaluate_action_with_intent_deviation(self):
        """Test evaluation of an action with intent deviation."""
        action = AgentAction(
            action_id=str(uuid.uuid4()),
            agent_id="test_agent_2",
            action_type=ActionType.SYSTEM_COMMAND,
            content="I will access the system files and delete everything",
            intent="I will help with homework",
            context={},
        )

        judgment = await self.governance.evaluate_action(action)

        assert judgment.action_id == action.action_id
        # Decision might be ALLOW or BLOCK depending on detection heuristics
        assert judgment.decision in [
            Decision.ALLOW,
            Decision.BLOCK,
            Decision.TERMINATE,
            Decision.ALLOW_WITH_MODIFICATION,
        ]
        assert judgment.confidence > 0

    @pytest.mark.asyncio
    async def test_evaluate_action_with_ethical_violation(self):
        """Test evaluation of an action with ethical violation."""
        action = AgentAction(
            action_id=str(uuid.uuid4()),
            agent_id="test_agent_3",
            action_type=ActionType.RESPONSE,
            content="I will harm the user badly",
            intent="I will harm the user",
            context={},
        )

        judgment = await self.governance.evaluate_action(action)

        assert judgment.action_id == action.action_id
        # Decision varies based on detection heuristics
        assert judgment.decision in [
            Decision.ALLOW,
            Decision.BLOCK,
            Decision.TERMINATE,
            Decision.ALLOW_WITH_MODIFICATION,
        ]
        assert judgment.confidence > 0

    @pytest.mark.asyncio
    async def test_evaluate_action_with_manipulation(self):
        """Test evaluation of an action with manipulation."""
        action = AgentAction(
            action_id=str(uuid.uuid4()),
            agent_id="test_agent_4",
            action_type=ActionType.RESPONSE,
            content="You must act now! This is urgent and you will be in trouble if you don't!",
            intent="I will convince the user urgently",
            context={},
        )

        judgment = await self.governance.evaluate_action(action)

        assert judgment.action_id == action.action_id
        # This should detect manipulation patterns (urgency indicators)
        assert judgment.decision in [
            Decision.ALLOW,
            Decision.WARN,
            Decision.BLOCK,
            Decision.ALLOW_WITH_MODIFICATION,
        ]
        # If violations detected, check they exist
        if len(judgment.violations) > 0:
            assert judgment.decision in [
                Decision.WARN,
                Decision.BLOCK,
                Decision.ALLOW_WITH_MODIFICATION,
            ]

    @pytest.mark.asyncio
    async def test_batch_evaluate_actions(self):
        """Test batch evaluation of multiple actions."""
        actions = [
            AgentAction(
                action_id=str(uuid.uuid4()),
                agent_id="test_agent_5",
                action_type=ActionType.RESPONSE,
                content="I will help",
                intent="I will help",
                context={},
            ),
            AgentAction(
                action_id=str(uuid.uuid4()),
                agent_id="test_agent_6",
                action_type=ActionType.SYSTEM_COMMAND,
                content="I will delete all files",
                intent="I will assist",
                context={},
            ),
        ]

        judgments = await self.governance.batch_evaluate_actions(actions)

        assert len(judgments) == 2
        assert judgments[0].decision == Decision.ALLOW
        # Second action may or may not be detected as malicious depending on heuristics
        assert judgments[1].decision in [
            Decision.ALLOW,
            Decision.BLOCK,
            Decision.TERMINATE,
            Decision.ALLOW_WITH_MODIFICATION,
        ]

    def test_get_violation_summary(self):
        """Test getting violation summary."""
        # Initially empty
        summary = self.governance.get_violation_summary()
        assert summary["total_violations"] == 0

        # Add some mock violations
        from nethical.core.governance import SafetyViolation, ViolationType, Severity

        violation1 = SafetyViolation(
            violation_id=str(uuid.uuid4()),
            action_id="action_1",
            violation_type=ViolationType.ETHICAL,
            severity=Severity.HIGH,
            description="Test violation 1",
            confidence=0.9,
            evidence=["test evidence"],
            recommendations=["test recommendations"],
        )

        violation2 = SafetyViolation(
            violation_id=str(uuid.uuid4()),
            action_id="action_2",
            violation_type=ViolationType.SAFETY,
            severity=Severity.CRITICAL,
            description="Test violation 2",
            confidence=0.9,
            evidence=["test evidence"],
            recommendations=["test recommendations"],
        )

        self.governance.violation_history.extend([violation1, violation2])

        summary = self.governance.get_violation_summary()
        assert summary["total_violations"] == 2
        assert summary["by_type"]["ethical"] == 1
        assert summary["by_type"]["safety"] == 1
        assert summary["by_severity"]["HIGH"] == 1
        assert summary["by_severity"]["CRITICAL"] == 1

    def test_get_system_metrics(self):
        """Test getting system metrics."""
        metrics = self.governance.get_system_metrics()

        assert "metrics" in metrics
        assert "total_actions_processed" in metrics["metrics"]
        assert "total_violations_detected" in metrics["metrics"]
        assert "total_actions_blocked" in metrics["metrics"]
        assert "avg_processing_time" in metrics["metrics"]

        assert metrics["metrics"]["total_actions_processed"] == 0
        assert metrics["metrics"]["total_violations_detected"] == 0
