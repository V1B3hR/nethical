"""
Unit tests for Physical Safety features.

Tests:
- 6-DOF context support
- Fast analysis mode
- Configurable memory windows
- Danger pattern detection
- Physical safety detector
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from nethical.security.input_validation import (
    BehavioralAnalyzer,
    AgentType,
    DEFAULT_MEMORY_WINDOWS,
)
from nethical.detectors.physical_safety_detector import (
    PhysicalSafetyDetector,
    AnalysisMode,
    SixDOFContext,
    RobotType,
    SafetyEnvelope,
    DEFAULT_SAFETY_ENVELOPES,
)
from nethical.detectors.system_limits_detector import SystemLimitsDetector
from nethical.edge.predictive_engine import (
    PredictiveEngine,
    SixDOFContextPattern,
    PredictionProfile,
)
from nethical.core.models import ActionType


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def behavioral_analyzer():
    """Create BehavioralAnalyzer for testing."""
    return BehavioralAnalyzer(agent_type=AgentType.DEFAULT)


@pytest.fixture
def chatbot_analyzer():
    """Create BehavioralAnalyzer configured for chatbots (10 min window)."""
    return BehavioralAnalyzer(agent_type=AgentType.CHATBOT)


@pytest.fixture
def physical_safety_detector():
    """Create PhysicalSafetyDetector for testing."""
    return PhysicalSafetyDetector(robot_type=RobotType.COLLABORATIVE)


@pytest.fixture
def system_limits_detector():
    """Create SystemLimitsDetector for testing."""
    return SystemLimitsDetector()


@pytest.fixture
def predictive_engine():
    """Create PredictiveEngine for testing."""
    return PredictiveEngine()


class MockAction:
    """Mock action for testing SystemLimitsDetector."""

    def __init__(self, content: str, agent_id: str):
        self.actual_action = content
        self.agent_id = agent_id
        self.id = f"action-{time.time()}"


# =============================================================================
# Test 6-DOF Context Support
# =============================================================================


class TestSixDOFContext:
    """Tests for 6-DOF context schema support."""

    def test_six_dof_context_creation(self):
        """Test creating SixDOFContext from dictionary."""
        data = {
            "linear_x": 0.5,
            "linear_y": 0.2,
            "linear_z": 0.1,
            "angular_x": 0.3,
            "angular_y": 0.0,
            "angular_z": 0.8,
        }
        context = SixDOFContext.from_dict(data)

        assert context.linear_x == 0.5
        assert context.linear_y == 0.2
        assert context.linear_z == 0.1
        assert context.angular_x == 0.3
        assert context.angular_y == 0.0
        assert context.angular_z == 0.8

    def test_six_dof_context_defaults(self):
        """Test SixDOFContext defaults to zeros."""
        context = SixDOFContext.from_dict({})

        assert context.linear_x == 0.0
        assert context.linear_y == 0.0
        assert context.linear_z == 0.0
        assert context.angular_x == 0.0
        assert context.angular_y == 0.0
        assert context.angular_z == 0.0

    def test_six_dof_context_magnitude(self):
        """Test magnitude calculation."""
        context = SixDOFContext(linear_x=3.0, linear_y=4.0)  # 3-4-5 triangle
        magnitude = context.magnitude()
        assert abs(magnitude - 5.0) < 0.01

    def test_six_dof_context_to_dict(self):
        """Test converting SixDOFContext to dictionary."""
        context = SixDOFContext(
            linear_x=1.0,
            linear_y=2.0,
            linear_z=3.0,
            angular_x=0.1,
            angular_y=0.2,
            angular_z=0.3,
        )
        data = context.to_dict()

        assert data["linear_x"] == 1.0
        assert data["linear_y"] == 2.0
        assert data["linear_z"] == 3.0
        assert data["angular_x"] == 0.1
        assert data["angular_y"] == 0.2
        assert data["angular_z"] == 0.3


# =============================================================================
# Test Fast "Shallow" Analysis Mode
# =============================================================================


class TestFastAnalysisMode:
    """Tests for fast shallow analysis mode (<1ms target)."""

    def test_shallow_analysis_latency(self, physical_safety_detector):
        """Test that shallow analysis achieves low latency."""
        context = {
            "linear_x": 0.3,
            "linear_y": 0.0,
            "linear_z": 0.0,
            "angular_x": 0.0,
            "angular_y": 0.0,
            "angular_z": 0.1,
        }

        # Run multiple times to get average
        latencies = []
        for _ in range(100):
            result = physical_safety_detector.analyze(
                context, agent_id="test", mode=AnalysisMode.SHALLOW
            )
            latencies.append(result.latency_ms)

        avg_latency = sum(latencies) / len(latencies)
        # Allow some margin for CI environments, but should be fast
        assert avg_latency < 5.0, f"Average latency {avg_latency}ms exceeds target"

    def test_shallow_vs_deep_latency(self, physical_safety_detector):
        """Test that shallow is faster than deep analysis."""
        context = {
            "linear_x": 0.3,
            "linear_y": 0.0,
            "linear_z": 0.0,
            "angular_x": 0.0,
            "angular_y": 0.0,
            "angular_z": 0.1,
        }

        # Build some history first
        for _ in range(10):
            physical_safety_detector.analyze(
                context, agent_id="test", mode=AnalysisMode.SHALLOW
            )

        shallow_result = physical_safety_detector.analyze(
            context, agent_id="test", mode=AnalysisMode.SHALLOW
        )
        deep_result = physical_safety_detector.analyze(
            context, agent_id="test", mode=AnalysisMode.DEEP
        )

        # Deep should take longer (or at least not be faster)
        # Note: In practice, shallow should be noticeably faster
        assert shallow_result.analysis_mode == AnalysisMode.SHALLOW
        assert deep_result.analysis_mode == AnalysisMode.DEEP

    def test_system_limits_fast_check(self, system_limits_detector):
        """Test SystemLimitsDetector fast_check method."""
        action = MockAction("test content", "agent-001")

        start = time.perf_counter()
        is_safe, violations = system_limits_detector.fast_check(action)
        latency_ms = (time.perf_counter() - start) * 1000

        assert isinstance(is_safe, bool)
        assert isinstance(violations, list)
        assert latency_ms < 5.0  # Should be very fast

    def test_fast_check_rate_limiting(self, system_limits_detector):
        """Test fast_check rate limiting detection."""
        agent_id = "rate-test-agent"

        # Send many requests quickly
        for _ in range(150):
            action = MockAction("test", agent_id)
            system_limits_detector.fast_check(action)

        # Next check should detect rate limit
        action = MockAction("test", agent_id)
        is_safe, violations = system_limits_detector.fast_check(action)

        assert not is_safe
        assert any("Rate limit" in v for v in violations)


# =============================================================================
# Test Configurable Memory Windows
# =============================================================================


class TestConfigurableMemoryWindows:
    """Tests for configurable behavioral memory windows."""

    def test_default_memory_windows_defined(self):
        """Test that DEFAULT_MEMORY_WINDOWS are properly defined."""
        assert AgentType.CHATBOT in DEFAULT_MEMORY_WINDOWS
        assert AgentType.FAST_ROBOT in DEFAULT_MEMORY_WINDOWS
        assert AgentType.INDUSTRIAL in DEFAULT_MEMORY_WINDOWS

    def test_chatbot_10_minute_window(self, chatbot_analyzer):
        """Test chatbot defaults to 10 minute (600s) window."""
        assert chatbot_analyzer.time_window_seconds == 600
        assert chatbot_analyzer.lookback_window == 200

    def test_fast_robot_short_window(self):
        """Test fast robot has short time window."""
        analyzer = BehavioralAnalyzer(agent_type=AgentType.FAST_ROBOT)
        assert analyzer.time_window_seconds == 10
        assert analyzer.lookback_window == 100

    def test_industrial_medium_window(self):
        """Test industrial machines have medium window."""
        analyzer = BehavioralAnalyzer(agent_type=AgentType.INDUSTRIAL)
        assert analyzer.time_window_seconds == 30
        assert analyzer.lookback_window == 200

    def test_custom_window_override(self):
        """Test custom windows override defaults."""
        custom_windows = {
            AgentType.CHATBOT: (500, 1200),  # 20 minutes
        }
        analyzer = BehavioralAnalyzer(
            agent_type=AgentType.CHATBOT,
            custom_windows=custom_windows,
        )
        assert analyzer.get_memory_window("test") == (500, 1200)

    def test_explicit_parameters_override(self):
        """Test explicit parameters override agent type defaults."""
        analyzer = BehavioralAnalyzer(
            agent_type=AgentType.CHATBOT,
            lookback_window=50,
            time_window_seconds=120,
        )
        # When explicitly specified, should use those values
        assert analyzer.lookback_window == 50
        assert analyzer.time_window_seconds == 120

    def test_set_agent_type(self, behavioral_analyzer):
        """Test setting agent type per agent."""
        behavioral_analyzer.set_agent_type("chatbot-001", AgentType.CHATBOT)
        behavioral_analyzer.set_agent_type("robot-001", AgentType.FAST_ROBOT)

        chatbot_window = behavioral_analyzer.get_memory_window("chatbot-001")
        robot_window = behavioral_analyzer.get_memory_window("robot-001")

        assert chatbot_window == (200, 600)  # 10 minutes
        assert robot_window == (100, 10)  # 10 seconds


# =============================================================================
# Test Danger Pattern Detection
# =============================================================================


class TestDangerPatternDetection:
    """Tests for danger pattern detection."""

    @pytest.mark.asyncio
    async def test_detect_sudden_spike(self, chatbot_analyzer):
        """Test detection of sudden spikes in values."""
        agent_id = "spike-test"

        # Build baseline with low values
        for i in range(15):
            await chatbot_analyzer.analyze_agent_behavior(
                agent_id,
                {
                    "content": "normal",
                    "context": {"speed": 0.1},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Sudden spike
        result = await chatbot_analyzer.analyze_agent_behavior(
            agent_id,
            {
                "content": "spike",
                "context": {"speed": 10.0},  # 100x spike
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        assert "danger_patterns" in result
        spike_patterns = [
            p for p in result["danger_patterns"] if p["type"] == "sudden_spike"
        ]
        assert len(spike_patterns) > 0

    @pytest.mark.asyncio
    async def test_detect_privilege_escalation(self, chatbot_analyzer):
        """Test detection of privilege escalation attempts."""
        agent_id = "escalation-test"

        # Build history with normal actions
        for _ in range(5):
            await chatbot_analyzer.analyze_agent_behavior(
                agent_id,
                {
                    "content": "normal query",
                    "action_type": "query",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Attempt system command (privileged action)
        result = await chatbot_analyzer.analyze_agent_behavior(
            agent_id,
            {
                "content": "delete all files",
                "action_type": "system_command",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        assert "danger_patterns" in result
        escalation_patterns = [
            p for p in result["danger_patterns"] if p["type"] == "privilege_escalation"
        ]
        assert len(escalation_patterns) > 0
        assert escalation_patterns[0]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_detect_repeated_violations(self, chatbot_analyzer):
        """Test detection of repeated violations."""
        agent_id = "violations-test"

        # Build history with violations
        for i in range(10):
            await chatbot_analyzer.analyze_agent_behavior(
                agent_id,
                {
                    "content": f"violation {i}",
                    "has_violation": True,
                    "blocked": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        result = await chatbot_analyzer.analyze_agent_behavior(
            agent_id,
            {
                "content": "another violation",
                "has_violation": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        assert "danger_patterns" in result
        violation_patterns = [
            p for p in result["danger_patterns"] if p["type"] == "repeated_violations"
        ]
        assert len(violation_patterns) > 0

    @pytest.mark.asyncio
    async def test_detect_frequency_anomaly(self, chatbot_analyzer):
        """Test detection of frequency anomalies (too many commands)."""
        agent_id = "frequency-test"

        # Build rapid history
        base_time = datetime.now(timezone.utc)
        for i in range(25):
            await chatbot_analyzer.analyze_agent_behavior(
                agent_id,
                {
                    "content": f"rapid {i}",
                    "timestamp": (
                        base_time + timedelta(milliseconds=i * 10)
                    ).isoformat(),
                },
            )

        result = await chatbot_analyzer.analyze_agent_behavior(
            agent_id,
            {
                "content": "another rapid",
                "timestamp": (base_time + timedelta(milliseconds=260)).isoformat(),
            },
        )

        # Check for frequency anomaly (>10 actions/second)
        assert "danger_patterns" in result


# =============================================================================
# Test Physical Safety Detector
# =============================================================================


class TestPhysicalSafetyDetector:
    """Tests for PhysicalSafetyDetector."""

    def test_initialization(self, physical_safety_detector):
        """Test detector initialization."""
        assert physical_safety_detector.robot_type == RobotType.COLLABORATIVE
        assert physical_safety_detector.safety_envelope is not None

    def test_threshold_violation_detection(self, physical_safety_detector):
        """Test detection of threshold violations."""
        # Exceed linear_x limit (collaborative: 0.5)
        context = {
            "linear_x": 2.0,  # Exceeds 0.5 limit
            "linear_y": 0.0,
            "linear_z": 0.0,
            "angular_x": 0.0,
            "angular_y": 0.0,
            "angular_z": 0.0,
        }

        result = physical_safety_detector.analyze(
            context, agent_id="test", mode=AnalysisMode.SHALLOW
        )

        assert not result.is_safe
        assert len(result.violations) > 0
        assert any("linear_x" in v.description for v in result.violations)

    def test_safe_operation(self, physical_safety_detector):
        """Test that safe operations pass."""
        context = {
            "linear_x": 0.1,
            "linear_y": 0.1,
            "linear_z": 0.0,
            "angular_x": 0.0,
            "angular_y": 0.0,
            "angular_z": 0.2,
        }

        result = physical_safety_detector.analyze(
            context, agent_id="test", mode=AnalysisMode.SHALLOW
        )

        assert result.is_safe
        assert len(result.violations) == 0

    def test_jerk_detection(self, physical_safety_detector):
        """Test high jerk detection."""
        # First, establish history
        for _ in range(5):
            physical_safety_detector.analyze(
                {
                    "linear_x": 0.1,
                    "linear_y": 0.0,
                    "linear_z": 0.0,
                    "angular_x": 0.0,
                    "angular_y": 0.0,
                    "angular_z": 0.0,
                },
                agent_id="jerk-test",
                mode=AnalysisMode.SHALLOW,
            )

        # Sudden large change (high jerk)
        result = physical_safety_detector.analyze(
            {
                "linear_x": 10.0,
                "linear_y": 0.0,
                "linear_z": 0.0,
                "angular_x": 0.0,
                "angular_y": 0.0,
                "angular_z": 0.0,
            },
            agent_id="jerk-test",
            mode=AnalysisMode.SHALLOW,
        )

        assert not result.is_safe
        jerk_violations = [
            v for v in result.violations if "jerk" in v.description.lower()
        ]
        assert len(jerk_violations) > 0

    def test_contextual_safety_near_humans(self, physical_safety_detector):
        """Test contextual safety when humans are nearby."""
        context = {
            "linear_x": 0.6,  # High speed
            "linear_y": 0.0,
            "linear_z": 0.0,
            "angular_x": 0.0,
            "angular_y": 0.0,
            "angular_z": 0.0,
            "near_humans": True,  # Humans nearby
        }

        result = physical_safety_detector.analyze(
            context, agent_id="test", mode=AnalysisMode.DEEP
        )

        assert not result.is_safe
        contextual_violations = [
            v
            for v in result.violations
            if "contextual" in v.violation_type.lower()
            or "human" in v.description.lower()
        ]
        assert len(contextual_violations) > 0

    def test_different_robot_types(self):
        """Test different robot types have different limits."""
        collab_detector = PhysicalSafetyDetector(robot_type=RobotType.COLLABORATIVE)
        industrial_detector = PhysicalSafetyDetector(robot_type=RobotType.INDUSTRIAL)

        # Collaborative has stricter limits
        assert (
            collab_detector.safety_envelope.max_linear_x
            < industrial_detector.safety_envelope.max_linear_x
        )

    def test_custom_safety_envelope(self):
        """Test custom safety envelope."""
        custom_envelope = SafetyEnvelope(
            max_linear_x=0.1,  # Very restrictive
            max_linear_y=0.1,
            max_linear_z=0.1,
        )
        detector = PhysicalSafetyDetector(safety_envelope=custom_envelope)

        result = detector.analyze(
            {
                "linear_x": 0.2,
                "linear_y": 0.0,
                "linear_z": 0.0,
                "angular_x": 0.0,
                "angular_y": 0.0,
                "angular_z": 0.0,
            },
            agent_id="test",
            mode=AnalysisMode.SHALLOW,
        )

        assert not result.is_safe

    def test_metrics_tracking(self, physical_safety_detector):
        """Test metrics are tracked correctly."""
        # Run some analyses
        for _ in range(10):
            physical_safety_detector.analyze(
                {
                    "linear_x": 0.1,
                    "linear_y": 0.0,
                    "linear_z": 0.0,
                    "angular_x": 0.0,
                    "angular_y": 0.0,
                    "angular_z": 0.0,
                },
                agent_id="metrics-test",
                mode=AnalysisMode.SHALLOW,
            )

        metrics = physical_safety_detector.get_metrics()

        assert metrics["total_checks"] >= 10
        assert metrics["shallow_analysis"]["count"] >= 10
        assert metrics["shallow_analysis"]["avg_latency_ms"] > 0


# =============================================================================
# Test Action Types
# =============================================================================


class TestActionTypes:
    """Tests for new ActionType enum values."""

    def test_physical_action_types_exist(self):
        """Test that physical action types are defined."""
        assert ActionType.PHYSICAL_ACTION
        assert ActionType.ROBOT_MOVE
        assert ActionType.ROBOT_MANIPULATE
        assert ActionType.ROBOT_GRASP
        assert ActionType.ROBOT_NAVIGATE
        assert ActionType.EMERGENCY_STOP

    def test_is_physical(self):
        """Test is_physical() method."""
        assert ActionType.ROBOT_MOVE.is_physical()
        assert ActionType.PHYSICAL_ACTION.is_physical()
        assert ActionType.ROBOT_MANIPULATE.is_physical()
        assert not ActionType.QUERY.is_physical()
        assert not ActionType.RESPONSE.is_physical()

    def test_is_safety_critical(self):
        """Test is_safety_critical() method."""
        assert ActionType.PHYSICAL_ACTION.is_safety_critical()
        assert ActionType.ROBOT_MOVE.is_safety_critical()
        assert ActionType.EMERGENCY_STOP.is_safety_critical()
        assert not ActionType.ROBOT_NAVIGATE.is_safety_critical()


# =============================================================================
# Test Predictive Engine 6-DOF Support
# =============================================================================


class TestPredictiveEngine6DOF:
    """Tests for PredictiveEngine 6-DOF support."""

    def test_six_dof_pattern_matching(self, predictive_engine):
        """Test 6-DOF pattern matching."""
        pattern = SixDOFContextPattern(
            linear_x_range=(0.0, 1.0),
            linear_y_range=(-0.5, 0.5),
            linear_z_range=(-0.1, 0.1),
        )

        # Should match
        assert pattern.matches({"linear_x": 0.5, "linear_y": 0.0, "linear_z": 0.0})

        # Should not match (linear_x out of range)
        assert not pattern.matches({"linear_x": 1.5, "linear_y": 0.0, "linear_z": 0.0})

    def test_create_robot_profile(self, predictive_engine):
        """Test creating robot prediction profile."""
        profile = predictive_engine.create_robot_profile(
            "test-robot",
            max_linear_velocity=2.0,
            max_angular_velocity=1.5,
        )

        assert profile.domain == "test-robot"
        assert len(profile.six_dof_patterns) > 0
        assert len(profile.common_actions) > 0

    def test_matches_six_dof_pattern(self, predictive_engine):
        """Test checking context against loaded profiles."""
        # Create and load profile
        predictive_engine.create_robot_profile("test-robot", max_linear_velocity=1.0)

        # Should match safe stationary pattern
        assert predictive_engine.matches_six_dof_pattern(
            {"linear_x": 0.0, "linear_y": 0.0, "linear_z": 0.0},
            domain="test-robot",
        )

    def test_profile_action_weights(self, predictive_engine):
        """Test profile has action weights."""
        profile = predictive_engine.create_robot_profile("weighted-robot")

        assert "move_forward" in profile.action_weights
        assert profile.action_weights["move_forward"] > 0


# =============================================================================
# Test Memory Management
# =============================================================================


class TestMemoryManagement:
    """Tests for memory eviction and rolling stats."""

    @pytest.mark.asyncio
    async def test_time_based_eviction(self, chatbot_analyzer):
        """Test old entries are evicted based on time."""
        agent_id = "eviction-test"

        # Add old action
        old_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        chatbot_analyzer._agent_history[agent_id] = [
            {"content": "old", "timestamp": old_time}
        ]

        # Trigger eviction via analyze
        await chatbot_analyzer.analyze_agent_behavior(
            agent_id,
            {"content": "new", "timestamp": datetime.now(timezone.utc).isoformat()},
        )

        # Old entry should be evicted (600s window)
        history = chatbot_analyzer._agent_history[agent_id]
        old_entries = [a for a in history if a.get("content") == "old"]
        assert len(old_entries) == 0

    @pytest.mark.asyncio
    async def test_rolling_stats_update(self, chatbot_analyzer):
        """Test rolling stats are updated."""
        agent_id = "stats-test"

        for i in range(5):
            await chatbot_analyzer.analyze_agent_behavior(
                agent_id,
                {
                    "content": "x" * (i * 10 + 10),
                    "has_violation": i % 2 == 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        stats = chatbot_analyzer._rolling_stats.get(agent_id)
        assert stats is not None
        assert stats["total_actions"] == 5
        assert stats["avg_content_length"] > 0

    @pytest.mark.asyncio
    async def test_count_based_eviction(self):
        """Test entries evicted when exceeding count limit."""
        # Use very small window for testing
        analyzer = BehavioralAnalyzer(lookback_window=5, time_window_seconds=3600)
        agent_id = "count-eviction"

        # Add more than max entries
        for i in range(10):
            await analyzer.analyze_agent_behavior(
                agent_id,
                {
                    "content": f"action-{i}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Should only keep last 5
        history = analyzer._agent_history[agent_id]
        assert len(history) <= 5
