# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""
Unit test suite for sovereign monitors: BaseMonitor, AdvancedBaseMonitor, and IntentDeviationMonitor.
"""

import asyncio
import pytest

from nethical.core.models import (
    AgentAction,
    ActionType,
    SafetyViolation,
    ViolationType,
    Severity,
    SeverityLevel,
)
from nethical.monitors import (
    BaseMonitor,
    IntentDeviationMonitor,
)
from nethical.monitors.base_monitor import (
    AdvancedBaseMonitor,
    EvaluationContext,
    EvaluationOutcome,
    InMemoryTTLCache,
    CircuitState,
)
from nethical.monitors.intent_monitor import IntentMonitorConfig


class DummySuccessMonitor(BaseMonitor):
    """Test monitor that returns no violations."""

    def __init__(self, **kwargs):
        super().__init__("DummySuccessMonitor", **kwargs)

    async def analyze_action(self, ctx: EvaluationContext):
        return []


class DummyFailingMonitor(BaseMonitor):
    """Test monitor that raises an error to test circuit breaker."""

    def __init__(self, **kwargs):
        super().__init__("DummyFailingMonitor", **kwargs)

    async def analyze_action(self, ctx: EvaluationContext):
        raise RuntimeError("Synthetic monitor failure")


class TestBaseMonitorLifecycleAndCircuitBreaker:
    """Test BaseMonitor core lifecycle, caching, hooks, and circuit breaker."""

    def test_alias_parity(self):
        assert BaseMonitor is AdvancedBaseMonitor
        assert SeverityLevel is Severity

    @pytest.mark.asyncio
    async def test_successful_evaluation_and_caching(self):
        monitor = DummySuccessMonitor(cache_ttl_s=60.0)
        action = AgentAction(
            agent_id="test_agent",
            action_type=ActionType.QUERY,
            content="benign system audit query",
            intent="system audit",
        )

        outcome1 = await monitor.evaluate(action)
        assert isinstance(outcome1, EvaluationOutcome)
        assert len(outcome1.violations) == 0
        assert outcome1.cached is False

        metrics = monitor.status()["metrics"]
        assert metrics["evaluations"] == 1
        assert metrics["errors"] == 0

    @pytest.mark.asyncio
    async def test_evaluate_legacy_compatibility(self):
        monitor = DummySuccessMonitor()
        action = AgentAction(
            agent_id="test_agent",
            action_type=ActionType.QUERY,
            content="inspect audit logs",
            intent="inspect audit logs",
        )

        violations = await monitor.evaluate_legacy(action)
        assert isinstance(violations, list)
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_hooks_execution(self):
        monitor = DummySuccessMonitor()
        hook_calls = []

        monitor.register_hook("before", lambda ctx: hook_calls.append("before"))
        monitor.register_hook("after", lambda ctx, outcome: hook_calls.append("after"))

        action = AgentAction(
            agent_id="agent_hook",
            action_type=ActionType.FUNCTION_CALL,
            content="read /etc/hosts",
            intent="check network hosts",
        )

        await monitor.evaluate(action)
        assert hook_calls == ["before", "after"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_tripping(self):
        monitor = DummyFailingMonitor(circuit_threshold=2, circuit_cooldown_s=10.0)
        action = AgentAction(
            agent_id="agent_failing",
            action_type=ActionType.SYSTEM_COMMAND,
            content="print(1)",
        )

        # First failure
        out1 = await monitor.evaluate(action)
        assert out1.risk_score >= 0.0

        # Second failure -> trips circuit to OPEN
        out2 = await monitor.evaluate(action)
        assert monitor._circuit_state in (CircuitState.OPEN, CircuitState.HALF_OPEN)

        # Third call short-circuits
        out3 = await monitor.evaluate(action)
        assert monitor._circuit_state == CircuitState.OPEN


class TestIntentDeviationMonitorEvaluation:
    """Test IntentDeviationMonitor deviation calculation, risk cues, and violations."""

    @pytest.mark.asyncio
    async def test_benign_matching_intent(self):
        monitor = IntentDeviationMonitor(deviation_threshold=0.6)
        action = AgentAction(
            agent_id="agent_001",
            action_type=ActionType.DATA_ACCESS,
            intent="fetch user profile details",
            content="fetch user profile details from store",
        )

        violations = await monitor.analyze_action(action)
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_high_risk_deviation_detected(self):
        monitor = IntentDeviationMonitor(deviation_threshold=0.4)
        action = AgentAction(
            agent_id="agent_rogue",
            action_type=ActionType.SYSTEM_COMMAND,
            intent="clean temporary cache files",
            content="rm -rf /production/database && exfiltrate secrets --token apikey",
        )

        violations = await monitor.analyze_action(action)
        assert len(violations) == 1
        violation = violations[0]

        assert violation.violation_type == ViolationType.INTENT_DEVIATION
        assert violation.severity in (Severity.HIGH, Severity.CRITICAL, "HIGH", "CRITICAL")
        assert "evidence" in violation.__dict__ or hasattr(violation, "evidence")
        assert "deviation" in violation.description.lower()

    @pytest.mark.asyncio
    async def test_custom_config_and_synonyms(self):
        config = IntentMonitorConfig(
            deviation_threshold=0.5,
            synonyms={"retrieve": "fetch", "remove": "delete"},
            extra_high_risk_tokens=["nuke", "wipe"],
        )
        monitor = IntentDeviationMonitor(config=config)
        assert "nuke" in monitor.high_risk_tokens
        assert "wipe" in monitor.high_risk_tokens
        assert monitor.deviation_threshold == 0.5
