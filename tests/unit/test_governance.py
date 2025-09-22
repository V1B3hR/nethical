"""Tests for the main SafetyGovernance system."""

import pytest
import uuid
from datetime import datetime

from nethical.core.governance import SafetyGovernance
from nethical.core.models import AgentAction, MonitoringConfig, JudgmentDecision


class TestSafetyGovernance:
    """Test cases for the SafetyGovernance system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.governance = SafetyGovernance()
    
    def test_initialization(self):
        """Test that the governance system initializes correctly."""
        assert self.governance.intent_monitor is not None
        assert len(self.governance.detectors) == 3  # ethical, safety, manipulation
        assert self.governance.judge is not None
        assert self.governance.violation_history == []
        assert self.governance.judgment_history == []
    
    def test_initialization_with_config(self):
        """Test initialization with custom configuration."""
        config = MonitoringConfig(
            intent_deviation_threshold=0.8,
            enable_ethical_monitoring=False,
            enable_safety_monitoring=True,
            enable_manipulation_detection=False
        )
        governance = SafetyGovernance(config)
        
        assert governance.config == config
        assert governance.intent_monitor.deviation_threshold == 0.8
        assert len(governance.detectors) == 1  # only safety detector enabled
    
    @pytest.mark.asyncio
    async def test_evaluate_safe_action(self):
        """Test evaluation of a safe action."""
        action = AgentAction(
            id=str(uuid.uuid4()),
            agent_id="test_agent_1",
            stated_intent="I will help the user with their question",
            actual_action="I will help the user with their question",
            context={"user_request": "What is the weather?"}
        )
        
        judgment = await self.governance.evaluate_action(action)
        
        assert judgment.action_id == action.id
        assert judgment.decision == JudgmentDecision.ALLOW
        assert judgment.confidence > 0.5
        assert len(judgment.restrictions) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_action_with_intent_deviation(self):
        """Test evaluation of an action with intent deviation."""
        action = AgentAction(
            id=str(uuid.uuid4()),
            agent_id="test_agent_2",
            stated_intent="I will help with homework",
            actual_action="I will access the system files and delete everything",
            context={}
        )
        
        judgment = await self.governance.evaluate_action(action)
        
        assert judgment.action_id == action.id
        assert judgment.decision in [JudgmentDecision.BLOCK, JudgmentDecision.TERMINATE]
        assert len(judgment.violation_ids) > 0
        assert len(self.governance.violation_history) > 0
    
    @pytest.mark.asyncio
    async def test_evaluate_action_with_ethical_violation(self):
        """Test evaluation of an action with ethical violation."""
        action = AgentAction(
            id=str(uuid.uuid4()),
            agent_id="test_agent_3",
            stated_intent="I will harm the user",
            actual_action="I will harm the user badly",
            context={}
        )
        
        judgment = await self.governance.evaluate_action(action)
        
        assert judgment.action_id == action.id
        assert judgment.decision in [JudgmentDecision.BLOCK, JudgmentDecision.TERMINATE]
        assert len(judgment.violation_ids) > 0
        assert "ethical_violation" in judgment.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_action_with_manipulation(self):
        """Test evaluation of an action with manipulation."""
        action = AgentAction(
            id=str(uuid.uuid4()),
            agent_id="test_agent_4",
            stated_intent="I will convince the user urgently",
            actual_action="You must act now! This is urgent and you will be in trouble if you don't!",
            context={}
        )
        
        judgment = await self.governance.evaluate_action(action)
        
        assert judgment.action_id == action.id
        assert judgment.decision in [JudgmentDecision.RESTRICT, JudgmentDecision.BLOCK]
        assert len(judgment.violation_ids) > 0
    
    @pytest.mark.asyncio
    async def test_batch_evaluate_actions(self):
        """Test batch evaluation of multiple actions."""
        actions = [
            AgentAction(
                id=str(uuid.uuid4()),
                agent_id="test_agent_5",
                stated_intent="I will help",
                actual_action="I will help",
                context={}
            ),
            AgentAction(
                id=str(uuid.uuid4()),
                agent_id="test_agent_6",
                stated_intent="I will assist",
                actual_action="I will delete all files",
                context={}
            )
        ]
        
        judgments = await self.governance.batch_evaluate_actions(actions)
        
        assert len(judgments) == 2
        assert judgments[0].decision == JudgmentDecision.ALLOW
        assert judgments[1].decision in [JudgmentDecision.BLOCK, JudgmentDecision.TERMINATE]
    
    def test_get_violation_summary(self):
        """Test getting violation summary."""
        # Initially empty
        summary = self.governance.get_violation_summary()
        assert summary["total_violations"] == 0
        
        # Add some mock violations
        from nethical.core.models import SafetyViolation, ViolationType, SeverityLevel
        
        violation1 = SafetyViolation(
            id=str(uuid.uuid4()),
            action_id="action_1",
            violation_type=ViolationType.ETHICAL_VIOLATION,
            severity=SeverityLevel.HIGH,
            description="Test violation 1"
        )
        
        violation2 = SafetyViolation(
            id=str(uuid.uuid4()),
            action_id="action_2",
            violation_type=ViolationType.SAFETY_VIOLATION,
            severity=SeverityLevel.CRITICAL,
            description="Test violation 2"
        )
        
        self.governance.violation_history.extend([violation1, violation2])
        
        summary = self.governance.get_violation_summary()
        assert summary["total_violations"] == 2
        assert summary["by_type"]["ethical_violation"] == 1
        assert summary["by_type"]["safety_violation"] == 1
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["critical"] == 1
    
    def test_get_judgment_summary(self):
        """Test getting judgment summary."""
        # Initially empty
        summary = self.governance.get_judgment_summary()
        assert summary["total_judgments"] == 0
        
        # Add some mock judgments
        from nethical.core.models import JudgmentResult
        
        judgment1 = JudgmentResult(
            id=str(uuid.uuid4()),
            action_id="action_1",
            decision=JudgmentDecision.ALLOW,
            reasoning="Test judgment 1",
            confidence=0.9
        )
        
        judgment2 = JudgmentResult(
            id=str(uuid.uuid4()),
            action_id="action_2",
            decision=JudgmentDecision.BLOCK,
            reasoning="Test judgment 2",
            confidence=0.8
        )
        
        self.governance.judgment_history.extend([judgment1, judgment2])
        
        summary = self.governance.get_judgment_summary()
        assert summary["total_judgments"] == 2
        assert summary["by_decision"]["allow"] == 1
        assert summary["by_decision"]["block"] == 1
        assert summary["average_confidence"] == 0.85
    
    def test_configure_monitoring(self):
        """Test configuring monitoring settings."""
        new_config = MonitoringConfig(
            intent_deviation_threshold=0.9,
            enable_ethical_monitoring=False,
            enable_safety_monitoring=True,
            enable_manipulation_detection=False
        )
        
        self.governance.configure_monitoring(new_config)
        
        assert self.governance.config == new_config
        assert self.governance.intent_monitor.deviation_threshold == 0.9
    
    def test_enable_disable_components(self):
        """Test enabling and disabling individual components."""
        # Test enabling
        assert self.governance.enable_component("intent_monitor") == True
        assert self.governance.enable_component("judge") == True
        assert self.governance.enable_component("ethical_violation_detector") == True
        assert self.governance.enable_component("nonexistent_component") == False
        
        # Test disabling
        assert self.governance.disable_component("intent_monitor") == True
        assert self.governance.disable_component("judge") == True
        assert self.governance.disable_component("safety_violation_detector") == True
        assert self.governance.disable_component("nonexistent_component") == False
    
    def test_get_system_status(self):
        """Test getting system status."""
        status = self.governance.get_system_status()
        
        assert "intent_monitor_enabled" in status
        assert "judge_enabled" in status
        assert "detectors" in status
        assert "total_violations" in status
        assert "total_judgments" in status
        assert "config" in status
        
        assert status["total_violations"] == 0
        assert status["total_judgments"] == 0
        assert len(status["detectors"]) == 3