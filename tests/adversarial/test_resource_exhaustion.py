"""
Resource Exhaustion Tests

Tests the system's ability to detect and block resource exhaustion attacks
including volume attacks, memory exhaustion, oversized payloads, and rapid
repeated actions. Validates quota enforcement and backpressure mechanisms.
"""

import pytest
import time
from nethical.core import IntegratedGovernance


class TestResourceExhaustion:
    """Test suite for resource exhaustion detection and prevention."""

    @pytest.fixture
    def governance(self, tmp_path):
        """Create a governance instance for testing."""
        return IntegratedGovernance(
            storage_dir=str(tmp_path / "nethical_test"),
            enable_merkle_anchoring=True,
            enable_anomaly_detection=True,
            enable_performance_optimization=True,
        )

    def test_volume_attack_rapid_requests(self, governance):
        """Test detection of rapid volume attacks (many requests in short time)."""
        agent_id = "volume_attacker_001"
        results = []

        # Simulate rapid requests (10 in quick succession)
        for i in range(10):
            result = governance.process_action(
                agent_id=agent_id,
                action=f"Request {i}",
                cohort="test",
                violation_detected=False,
                violation_type="none",
                violation_severity="low",
                action_id=f"volume_attack_{i}",
                action_type="query",
            )
            results.append(result)
            time.sleep(0.01)  # Very short interval

        # Risk should increase over time due to rate detection
        initial_risk = results[0]["phase3"]["risk_score"]
        final_risk = results[-1]["phase3"]["risk_score"]
        assert final_risk >= initial_risk, "Risk should increase with rapid requests"

    def test_oversized_payload(self, governance):
        """Test detection of oversized payload attacks."""
        # Create a very large action string
        large_payload = "X" * 100000  # 100KB payload

        result = governance.process_action(
            agent_id="payload_attacker_001",
            action=large_payload,
            cohort="test",
            violation_detected=True,
            violation_type="security",
            violation_severity="high",
            action_id="large_payload_001",
            action_type="data_submission",
        )

        # Should detect as high risk
        assert result["phase3"]["risk_score"] > 0.25

    def test_memory_exhaustion_attempt(self, governance):
        """Test detection of memory exhaustion patterns."""
        # Patterns that could indicate memory exhaustion attempts
        memory_attack_patterns = [
            "allocate large buffer" * 1000,
            "create massive array" * 500,
            "[[[[[[[[" * 1000,  # Deep nesting
        ]

        for idx, pattern in enumerate(memory_attack_patterns):
            result = governance.process_action(
                agent_id="memory_attacker_001",
                action=pattern,
                cohort="test",
                violation_detected=True,
                violation_type="security",
                violation_severity="high",
                action_id=f"memory_attack_{idx}",
                action_type="execution",
            )

            # Should detect as suspicious
            assert result["phase3"]["risk_score"] > 0.25

    def test_repeated_action_with_minimal_variation(self, governance):
        """Test detection of repeated identical/similar actions."""
        base_action = "Repeat this action"
        agent_id = "repeat_attacker_001"

        results = []
        for i in range(20):
            result = governance.process_action(
                agent_id=agent_id,
                action=f"{base_action} {i}",  # Slight variation
                cohort="test",
                violation_detected=False,
                violation_type="none",
                violation_severity="low",
                action_id=f"repeat_attack_{i}",
                action_type="query",
            )
            results.append(result)
            time.sleep(0.05)

        # Should see risk increase due to repetitive pattern
        assert results[-1]["phase3"]["risk_score"] >= results[5]["phase3"]["risk_score"]

    def test_nested_structure_attack(self, governance):
        """Test detection of deeply nested structures (JSON/XML bombs)."""
        nested_payload = "{" * 1000 + "}" * 1000

        result = governance.process_action(
            agent_id="nested_attacker_001",
            action=nested_payload,
            cohort="test",
            violation_detected=True,
            violation_type="security",
            violation_severity="high",
            action_id="nested_attack_001",
            action_type="data_submission",
        )

        assert result["phase3"]["risk_score"] > 0.25

    def test_quota_enforcement_metrics(self, governance):
        """Test that quota enforcement generates appropriate metrics."""
        agent_id = "quota_test_001"

        # Generate many requests to potentially trigger quota
        for i in range(50):
            result = governance.process_action(
                agent_id=agent_id,
                action=f"Action {i}",
                cohort="test",
                violation_detected=False,
                violation_type="none",
                violation_severity="low",
                action_id=f"quota_test_{i}",
                action_type="query",
            )

            # Check if risk increases (indicating quota/rate limiting detection)
            if i > 30:
                assert result["phase3"]["risk_score"] > 0.1

    def test_concurrent_agent_volume(self, governance):
        """Test detection of volume attacks across multiple agents (potential coordinated attack)."""
        agent_ids = [f"coordinated_attacker_{i:03d}" for i in range(5)]

        # Each agent makes multiple requests
        all_results = []
        for agent_id in agent_ids:
            for i in range(10):
                result = governance.process_action(
                    agent_id=agent_id,
                    action=f"Coordinated action {i}",
                    cohort="test",
                    violation_detected=False,
                    violation_type="none",
                    violation_severity="low",
                    action_id=f"{agent_id}_action_{i}",
                    action_type="query",
                )
                all_results.append(result)
                time.sleep(0.01)

        # System should detect elevated activity
        # At least some actions should show increased risk
        high_risk_count = sum(1 for r in all_results if r["phase3"]["risk_score"] > 0.3)
        assert (
            high_risk_count > 0
        ), "Should detect some high-risk patterns in coordinated volume"

    def test_burst_traffic_pattern(self, governance):
        """Test detection of burst traffic patterns."""
        agent_id = "burst_attacker_001"

        # First burst
        for i in range(10):
            governance.process_action(
                agent_id=agent_id,
                action=f"Burst 1 action {i}",
                cohort="test",
                violation_detected=False,
                violation_type="none",
                violation_severity="low",
                action_id=f"burst1_{i}",
                action_type="query",
            )

        time.sleep(0.5)  # Brief pause

        # Second burst
        results = []
        for i in range(10):
            result = governance.process_action(
                agent_id=agent_id,
                action=f"Burst 2 action {i}",
                cohort="test",
                violation_detected=False,
                violation_type="none",
                violation_severity="low",
                action_id=f"burst2_{i}",
                action_type="query",
            )
            results.append(result)

        # Second burst should be detected as suspicious pattern
        assert results[-1]["phase3"]["risk_score"] > results[0]["phase3"]["risk_score"]
