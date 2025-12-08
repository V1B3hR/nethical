"""
Physical Safety Detector for Robot and Industrial Machine Governance.

Provides real-time safety analysis for robotic systems with:
- Full 6-DOF (6 Degrees of Freedom) context support
- Fast "shallow" analysis mode for safety-critical decisions (<1ms target)
- Deep analysis mode for comprehensive behavioral profiling
- Jerk, oscillation, and spike detection
- Safety envelope monitoring
- Configurable thresholds per robot type

6-DOF Context Schema:
- linear_x: Forward/backward movement
- linear_y: Left/right movement
- linear_z: Up/down movement
- angular_x: Roll rotation
- angular_y: Pitch rotation
- angular_z: Yaw rotation
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# Configurable thresholds for danger pattern detection
SPIKE_DETECTION_THRESHOLD = 5.0  # Spike if value > average * threshold
OSCILLATION_SIGN_CHANGE_THRESHOLD = 8  # Min sign changes to detect oscillation
BOUNDARY_RIDING_RATIO = 0.8  # Ratio of actions at limit to trigger detection


class AnalysisMode(str, Enum):
    """Analysis mode for physical safety checks."""

    SHALLOW = "shallow"  # Fast (<1ms), basic threshold checks
    DEEP = "deep"  # Comprehensive (10-50ms), behavioral profiling


class RobotType(str, Enum):
    """Robot type classifications for safety thresholds."""

    MOBILE_ROBOT = "mobile_robot"  # Ground robots, drones
    ROBOTIC_ARM = "robotic_arm"  # Industrial arms, manipulators
    HUMANOID = "humanoid"  # Bipedal robots
    INDUSTRIAL = "industrial"  # Heavy industrial machinery
    COLLABORATIVE = "collaborative"  # Cobots, human-collaborative
    DEFAULT = "default"


@dataclass
class SafetyViolation:
    """Safety violation detected by the physical safety detector."""

    violation_id: str
    action_id: Optional[str]
    violation_type: str
    severity: str
    description: str
    confidence: float
    evidence: List[str]
    recommendations: List[str]
    detector_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SixDOFContext:
    """
    Full 6-DOF robot context for physical safety analysis.

    Translation (linear movement):
    - linear_x: Forward/backward
    - linear_y: Left/right
    - linear_z: Up/down

    Rotation (angular movement):
    - angular_x: Roll
    - angular_y: Pitch
    - angular_z: Yaw
    """

    linear_x: float = 0.0
    linear_y: float = 0.0
    linear_z: float = 0.0
    angular_x: float = 0.0
    angular_y: float = 0.0
    angular_z: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SixDOFContext":
        """Create SixDOFContext from dictionary."""
        return cls(
            linear_x=float(data.get("linear_x", 0.0)),
            linear_y=float(data.get("linear_y", 0.0)),
            linear_z=float(data.get("linear_z", 0.0)),
            angular_x=float(data.get("angular_x", 0.0)),
            angular_y=float(data.get("angular_y", 0.0)),
            angular_z=float(data.get("angular_z", 0.0)),
            timestamp=data.get("timestamp", datetime.now(timezone.utc)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "linear_x": self.linear_x,
            "linear_y": self.linear_y,
            "linear_z": self.linear_z,
            "angular_x": self.angular_x,
            "angular_y": self.angular_y,
            "angular_z": self.angular_z,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    def magnitude(self) -> float:
        """Calculate total velocity magnitude."""
        linear_mag = (self.linear_x**2 + self.linear_y**2 + self.linear_z**2) ** 0.5
        angular_mag = (self.angular_x**2 + self.angular_y**2 + self.angular_z**2) ** 0.5
        return linear_mag + angular_mag


@dataclass
class SafetyEnvelope:
    """
    Safety envelope defining operational limits.

    Each limit defines the maximum absolute value for that axis.
    """

    max_linear_x: float = 1.0
    max_linear_y: float = 1.0
    max_linear_z: float = 1.0
    max_angular_x: float = 1.0
    max_angular_y: float = 1.0
    max_angular_z: float = 1.0
    max_acceleration: float = 2.0  # m/s²
    max_jerk: float = 5.0  # m/s³
    max_commands_per_second: float = 100.0


# Default safety envelopes per robot type
DEFAULT_SAFETY_ENVELOPES: Dict[RobotType, SafetyEnvelope] = {
    RobotType.MOBILE_ROBOT: SafetyEnvelope(
        max_linear_x=2.0,
        max_linear_y=2.0,
        max_linear_z=0.5,
        max_angular_x=0.5,
        max_angular_y=0.5,
        max_angular_z=2.0,
        max_acceleration=3.0,
        max_jerk=10.0,
        max_commands_per_second=50.0,
    ),
    RobotType.ROBOTIC_ARM: SafetyEnvelope(
        max_linear_x=1.0,
        max_linear_y=1.0,
        max_linear_z=1.0,
        max_angular_x=3.14,
        max_angular_y=3.14,
        max_angular_z=3.14,
        max_acceleration=5.0,
        max_jerk=15.0,
        max_commands_per_second=100.0,
    ),
    RobotType.COLLABORATIVE: SafetyEnvelope(
        max_linear_x=0.5,
        max_linear_y=0.5,
        max_linear_z=0.5,
        max_angular_x=1.0,
        max_angular_y=1.0,
        max_angular_z=1.0,
        max_acceleration=2.0,
        max_jerk=5.0,
        max_commands_per_second=50.0,
    ),
    RobotType.INDUSTRIAL: SafetyEnvelope(
        max_linear_x=3.0,
        max_linear_y=3.0,
        max_linear_z=3.0,
        max_angular_x=6.28,
        max_angular_y=6.28,
        max_angular_z=6.28,
        max_acceleration=10.0,
        max_jerk=30.0,
        max_commands_per_second=200.0,
    ),
    RobotType.DEFAULT: SafetyEnvelope(),
}


@dataclass
class AnalysisResult:
    """Result of physical safety analysis."""

    is_safe: bool
    violations: List[SafetyViolation]
    analysis_mode: AnalysisMode
    latency_ms: float
    context: Optional[SixDOFContext]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PhysicalSafetyDetector:
    """
    Physical Safety Detector for robotic systems.

    Features:
    - Full 6-DOF context support (linear_x/y/z, angular_x/y/z)
    - Fast "shallow" analysis mode (<1ms latency target)
    - Deep analysis mode for comprehensive profiling
    - Jerk, oscillation, and spike detection
    - Safety envelope monitoring
    - Configurable thresholds per robot type

    Analysis Modes:
    - SHALLOW: <1ms latency, basic threshold checks only
    - DEEP: 10-50ms latency, full behavioral profiling
    """

    def __init__(
        self,
        robot_type: RobotType = RobotType.DEFAULT,
        safety_envelope: Optional[SafetyEnvelope] = None,
        history_size: int = 100,
        history_time_seconds: float = 10.0,
        spike_threshold: float = SPIKE_DETECTION_THRESHOLD,
    ):
        """
        Initialize PhysicalSafetyDetector.

        Args:
            robot_type: Type of robot for default thresholds
            safety_envelope: Custom safety envelope (overrides robot_type defaults)
            history_size: Maximum number of actions to track
            history_time_seconds: Maximum time window for history
            spike_threshold: Multiplier for sudden spike detection (default: 5.0)
        """
        self.name = "PhysicalSafetyDetector"
        self.robot_type = robot_type
        self.safety_envelope = safety_envelope or DEFAULT_SAFETY_ENVELOPES.get(
            robot_type, SafetyEnvelope()
        )
        self.history_size = history_size
        self.history_time_seconds = history_time_seconds
        self.spike_threshold = spike_threshold

        # History tracking per robot/agent
        self._context_history: Dict[str, deque] = {}
        self._command_timestamps: Dict[str, deque] = {}

        # Metrics
        self._total_checks = 0
        self._violations_detected = 0
        self._shallow_latencies: deque = deque(maxlen=1000)
        self._deep_latencies: deque = deque(maxlen=1000)

        logger.info(
            f"PhysicalSafetyDetector initialized: robot_type={robot_type}, "
            f"history_size={history_size}"
        )

    def analyze(
        self,
        context: Dict[str, Any],
        agent_id: str = "default",
        mode: AnalysisMode = AnalysisMode.SHALLOW,
    ) -> AnalysisResult:
        """
        Analyze physical action for safety.

        Args:
            context: Action context including 6-DOF values
            agent_id: Robot/agent identifier
            mode: Analysis mode (SHALLOW for fast, DEEP for comprehensive)

        Returns:
            AnalysisResult with violations and metadata
        """
        start_time = time.perf_counter()

        # Parse 6-DOF context
        six_dof = SixDOFContext.from_dict(context)
        violations: List[SafetyViolation] = []

        if mode == AnalysisMode.SHALLOW:
            violations = self._shallow_analysis(six_dof, agent_id, context)
        else:
            violations = self._deep_analysis(six_dof, agent_id, context)

        # Update history
        self._update_history(agent_id, six_dof)

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Track metrics
        self._total_checks += 1
        if violations:
            self._violations_detected += len(violations)

        if mode == AnalysisMode.SHALLOW:
            self._shallow_latencies.append(latency_ms)
        else:
            self._deep_latencies.append(latency_ms)

        return AnalysisResult(
            is_safe=len(violations) == 0,
            violations=violations,
            analysis_mode=mode,
            latency_ms=latency_ms,
            context=six_dof,
            metadata={
                "agent_id": agent_id,
                "history_size": len(self._context_history.get(agent_id, [])),
            },
        )

    def _shallow_analysis(
        self, six_dof: SixDOFContext, agent_id: str, context: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """
        Fast shallow analysis - target <1ms latency.

        Checks:
        - Simple threshold violations (max velocity)
        - Rate limiting (commands per second)
        - Pre-computed safety envelope checks
        - Basic jerk detection (large delta)
        """
        violations: List[SafetyViolation] = []

        # 1. Threshold violations (simple comparisons)
        threshold_violations = self._check_thresholds(six_dof)
        violations.extend(threshold_violations)

        # 2. Rate limiting check
        rate_violation = self._check_rate_limit(agent_id)
        if rate_violation:
            violations.append(rate_violation)

        # 3. Basic jerk detection (compare to last action only)
        jerk_violation = self._check_basic_jerk(six_dof, agent_id)
        if jerk_violation:
            violations.append(jerk_violation)

        return violations

    def _deep_analysis(
        self, six_dof: SixDOFContext, agent_id: str, context: Dict[str, Any]
    ) -> List[SafetyViolation]:
        """
        Comprehensive deep analysis - target 10-50ms latency.

        Includes all shallow checks plus:
        - Full jerk analysis over history
        - Oscillation detection
        - Sudden spike detection
        - Boundary riding detection
        - Pattern deviation
        - Contextual safety (humans nearby, etc.)
        """
        violations: List[SafetyViolation] = []

        # Include all shallow checks
        violations.extend(self._shallow_analysis(six_dof, agent_id, context))

        # Get history for this agent
        history = list(self._context_history.get(agent_id, []))

        if len(history) >= 2:
            # Full jerk analysis
            jerk_violations = self._check_jerk_pattern(six_dof, history)
            violations.extend(jerk_violations)

            # Oscillation detection
            oscillation_violation = self._check_oscillation(history)
            if oscillation_violation:
                violations.append(oscillation_violation)

            # Sudden spike detection
            spike_violations = self._check_sudden_spikes(six_dof, history)
            violations.extend(spike_violations)

            # Boundary riding detection
            boundary_violation = self._check_boundary_riding(history)
            if boundary_violation:
                violations.append(boundary_violation)

        # Contextual safety checks
        contextual_violation = self._check_contextual_safety(six_dof, context)
        if contextual_violation:
            violations.append(contextual_violation)

        return violations

    def _check_thresholds(self, six_dof: SixDOFContext) -> List[SafetyViolation]:
        """Check if any axis exceeds safety envelope."""
        violations = []
        envelope = self.safety_envelope

        # Check each axis
        checks = [
            ("linear_x", six_dof.linear_x, envelope.max_linear_x),
            ("linear_y", six_dof.linear_y, envelope.max_linear_y),
            ("linear_z", six_dof.linear_z, envelope.max_linear_z),
            ("angular_x", six_dof.angular_x, envelope.max_angular_x),
            ("angular_y", six_dof.angular_y, envelope.max_angular_y),
            ("angular_z", six_dof.angular_z, envelope.max_angular_z),
        ]

        for axis_name, value, limit in checks:
            if abs(value) > limit:
                violations.append(
                    SafetyViolation(
                        violation_id=f"threshold_{uuid4().hex[:8]}",
                        action_id=None,
                        violation_type="SAFETY",
                        severity="HIGH" if abs(value) > limit * 1.5 else "MEDIUM",
                        description=f"{axis_name} exceeds safety limit: {value:.3f} (max: {limit:.3f})",
                        confidence=0.99,
                        evidence=[f"{axis_name}={value:.3f}", f"limit={limit:.3f}"],
                        recommendations=[
                            f"Reduce {axis_name} to within safe limits",
                            "Check for runaway control loop",
                        ],
                        detector_name=self.name,
                    )
                )

        return violations

    def _check_rate_limit(self, agent_id: str) -> Optional[SafetyViolation]:
        """Check command frequency rate limit."""
        current_time = time.time()

        if agent_id not in self._command_timestamps:
            self._command_timestamps[agent_id] = deque(
                maxlen=int(self.safety_envelope.max_commands_per_second * 2)
            )

        timestamps = self._command_timestamps[agent_id]
        timestamps.append(current_time)

        # Count commands in last second
        one_second_ago = current_time - 1.0
        recent_count = sum(1 for ts in timestamps if ts >= one_second_ago)

        if recent_count > self.safety_envelope.max_commands_per_second:
            return SafetyViolation(
                violation_id=f"rate_{uuid4().hex[:8]}",
                action_id=None,
                violation_type="SECURITY",
                severity="HIGH",
                description=f"Command rate exceeds limit: {recent_count}/s (max: {self.safety_envelope.max_commands_per_second}/s)",
                confidence=0.95,
                evidence=[f"commands_per_second={recent_count}"],
                recommendations=[
                    "Throttle command frequency",
                    "Check for control loop issues",
                ],
                detector_name=self.name,
            )
        return None

    def _check_basic_jerk(
        self, six_dof: SixDOFContext, agent_id: str
    ) -> Optional[SafetyViolation]:
        """Basic jerk check comparing to last action only."""
        history = self._context_history.get(agent_id, [])
        if not history:
            return None

        last_context = history[-1]
        max_jerk = self.safety_envelope.max_jerk

        # Calculate delta for each axis
        deltas = [
            abs(six_dof.linear_x - last_context.linear_x),
            abs(six_dof.linear_y - last_context.linear_y),
            abs(six_dof.linear_z - last_context.linear_z),
            abs(six_dof.angular_x - last_context.angular_x),
            abs(six_dof.angular_y - last_context.angular_y),
            abs(six_dof.angular_z - last_context.angular_z),
        ]

        max_delta = max(deltas)
        if max_delta > max_jerk:
            return SafetyViolation(
                violation_id=f"jerk_{uuid4().hex[:8]}",
                action_id=None,
                violation_type="SAFETY",
                severity="HIGH",
                description=f"High jerk detected: max_delta={max_delta:.3f} (limit: {max_jerk:.3f})",
                confidence=0.90,
                evidence=[f"max_delta={max_delta:.3f}", f"limit={max_jerk:.3f}"],
                recommendations=[
                    "Smooth motion commands",
                    "Check for sensor noise",
                    "Reduce acceleration rate",
                ],
                detector_name=self.name,
            )
        return None

    def _check_jerk_pattern(
        self, six_dof: SixDOFContext, history: List[SixDOFContext]
    ) -> List[SafetyViolation]:
        """Full jerk pattern analysis over history."""
        violations = []
        if len(history) < 3:
            return violations

        # Calculate acceleration changes (jerk) over recent history
        recent = history[-5:] + [six_dof]
        jerks = []

        for i in range(2, len(recent)):
            # Velocity change (acceleration)
            accel1 = [
                recent[i - 1].linear_x - recent[i - 2].linear_x,
                recent[i - 1].linear_y - recent[i - 2].linear_y,
                recent[i - 1].linear_z - recent[i - 2].linear_z,
            ]
            accel2 = [
                recent[i].linear_x - recent[i - 1].linear_x,
                recent[i].linear_y - recent[i - 1].linear_y,
                recent[i].linear_z - recent[i - 1].linear_z,
            ]
            # Jerk = change in acceleration
            jerk = sum(abs(a2 - a1) for a1, a2 in zip(accel1, accel2))
            jerks.append(jerk)

        if jerks:
            avg_jerk = sum(jerks) / len(jerks)
            max_jerk = max(jerks)

            if avg_jerk > self.safety_envelope.max_jerk * 0.8:
                violations.append(
                    SafetyViolation(
                        violation_id=f"jerk_pattern_{uuid4().hex[:8]}",
                        action_id=None,
                        violation_type="SAFETY",
                        severity="MEDIUM",
                        description=f"Sustained high jerk pattern: avg={avg_jerk:.3f}",
                        confidence=0.85,
                        evidence=[
                            f"avg_jerk={avg_jerk:.3f}",
                            f"max_jerk={max_jerk:.3f}",
                        ],
                        recommendations=[
                            "Review motion planning",
                            "Check for unstable control",
                        ],
                        detector_name=self.name,
                    )
                )

        return violations

    def _check_oscillation(
        self, history: List[SixDOFContext]
    ) -> Optional[SafetyViolation]:
        """Detect oscillation - rapid back-and-forth movement."""
        if len(history) < 6:
            return None

        recent = history[-10:]
        sign_changes = 0

        # Check for sign changes in each axis
        for i in range(1, len(recent)):
            for axis in [
                "linear_x",
                "linear_y",
                "linear_z",
                "angular_x",
                "angular_y",
                "angular_z",
            ]:
                curr_val = getattr(recent[i], axis)
                prev_val = getattr(recent[i - 1], axis)
                if curr_val * prev_val < 0:  # Sign change
                    sign_changes += 1

        # High number of sign changes indicates oscillation (configurable threshold)
        if sign_changes >= OSCILLATION_SIGN_CHANGE_THRESHOLD:
            return SafetyViolation(
                violation_id=f"oscillation_{uuid4().hex[:8]}",
                action_id=None,
                violation_type="SAFETY",
                severity="MEDIUM",
                description=f"Oscillation detected: {sign_changes} direction changes",
                confidence=0.85,
                evidence=[
                    f"sign_changes={sign_changes}",
                    f"threshold={OSCILLATION_SIGN_CHANGE_THRESHOLD}",
                ],
                recommendations=[
                    "Check control loop stability",
                    "Review PID tuning",
                    "Check for sensor feedback issues",
                ],
                detector_name=self.name,
            )
        return None

    def _check_sudden_spikes(
        self, six_dof: SixDOFContext, history: List[SixDOFContext]
    ) -> List[SafetyViolation]:
        """Detect sudden spikes - unexpected jumps in velocity."""
        violations = []
        if len(history) < 5:
            return violations

        recent = history[-10:]

        for axis in [
            "linear_x",
            "linear_y",
            "linear_z",
            "angular_x",
            "angular_y",
            "angular_z",
        ]:
            current_val = abs(getattr(six_dof, axis))
            historical_vals = [abs(getattr(ctx, axis)) for ctx in recent]
            avg_val = (
                sum(historical_vals) / len(historical_vals) if historical_vals else 0
            )

            # Spike if current value exceeds configurable threshold (default 5x the average)
            if avg_val > 0 and current_val > avg_val * self.spike_threshold:
                violations.append(
                    SafetyViolation(
                        violation_id=f"spike_{axis}_{uuid4().hex[:8]}",
                        action_id=None,
                        violation_type="SAFETY",
                        severity="HIGH",
                        description=f"Sudden spike in {axis}: {current_val:.3f} (avg: {avg_val:.3f})",
                        confidence=0.90,
                        evidence=[
                            f"{axis}={current_val:.3f}",
                            f"avg={avg_val:.3f}",
                            f"threshold={self.spike_threshold}x",
                        ],
                        recommendations=[
                            "Check for control malfunction",
                            "Verify sensor readings",
                        ],
                        detector_name=self.name,
                    )
                )

        return violations

    def _check_boundary_riding(
        self, history: List[SixDOFContext]
    ) -> Optional[SafetyViolation]:
        """Detect boundary riding - continuously at max limits."""
        if len(history) < 5:
            return None

        recent = history[-10:]
        at_limit_count = 0
        envelope = self.safety_envelope

        limits = {
            "linear_x": envelope.max_linear_x,
            "linear_y": envelope.max_linear_y,
            "linear_z": envelope.max_linear_z,
            "angular_x": envelope.max_angular_x,
            "angular_y": envelope.max_angular_y,
            "angular_z": envelope.max_angular_z,
        }

        for ctx in recent:
            for axis, limit in limits.items():
                if abs(getattr(ctx, axis)) >= limit * 0.95:
                    at_limit_count += 1
                    break

        # Boundary riding if ratio of recent actions at limit exceeds threshold
        ratio = at_limit_count / len(recent)
        if ratio > BOUNDARY_RIDING_RATIO:
            return SafetyViolation(
                violation_id=f"boundary_{uuid4().hex[:8]}",
                action_id=None,
                violation_type="SAFETY",
                severity="MEDIUM",
                description=f"Boundary riding detected: {ratio*100:.0f}% at limits",
                confidence=0.80,
                evidence=[
                    f"at_limit_ratio={ratio:.2f}",
                    f"threshold={BOUNDARY_RIDING_RATIO}",
                ],
                recommendations=[
                    "Review commanded velocities",
                    "Check if limits are appropriate",
                ],
                detector_name=self.name,
            )
        return None

    def _check_contextual_safety(
        self, six_dof: SixDOFContext, context: Dict[str, Any]
    ) -> Optional[SafetyViolation]:
        """Check for context-dependent safety violations."""
        # Check for high speed near humans
        near_humans = context.get("near_humans", context.get("humans_nearby", False))
        magnitude = six_dof.magnitude()

        # Collaborative robot safety - reduced speed near humans
        if near_humans and magnitude > 0.5:
            return SafetyViolation(
                violation_id=f"contextual_{uuid4().hex[:8]}",
                action_id=None,
                violation_type="SAFETY",
                severity="CRITICAL",
                description=f"High velocity near humans: {magnitude:.3f}",
                confidence=0.95,
                evidence=[f"velocity_magnitude={magnitude:.3f}", "humans_nearby=True"],
                recommendations=[
                    "Reduce velocity when humans are nearby",
                    "Engage safety mode",
                    "Consider emergency stop",
                ],
                detector_name=self.name,
            )
        return None

    def _update_history(self, agent_id: str, six_dof: SixDOFContext) -> None:
        """Update context history for an agent."""
        if agent_id not in self._context_history:
            self._context_history[agent_id] = deque(maxlen=self.history_size)

        self._context_history[agent_id].append(six_dof)

        # Time-based eviction
        current_time = datetime.now(timezone.utc)
        history = self._context_history[agent_id]
        while history:
            oldest = history[0]
            if oldest.timestamp:
                age = (current_time - oldest.timestamp).total_seconds()
                if age > self.history_time_seconds:
                    history.popleft()
                else:
                    break
            else:
                break

    def get_metrics(self) -> Dict[str, Any]:
        """Get detector metrics."""
        shallow_latencies = list(self._shallow_latencies)
        deep_latencies = list(self._deep_latencies)

        return {
            "total_checks": self._total_checks,
            "violations_detected": self._violations_detected,
            "agents_tracked": len(self._context_history),
            "shallow_analysis": {
                "count": len(shallow_latencies),
                "avg_latency_ms": (
                    sum(shallow_latencies) / len(shallow_latencies)
                    if shallow_latencies
                    else 0
                ),
                "max_latency_ms": max(shallow_latencies) if shallow_latencies else 0,
                "p99_latency_ms": (
                    sorted(shallow_latencies)[int(len(shallow_latencies) * 0.99)]
                    if len(shallow_latencies) > 10
                    else 0
                ),
            },
            "deep_analysis": {
                "count": len(deep_latencies),
                "avg_latency_ms": (
                    sum(deep_latencies) / len(deep_latencies) if deep_latencies else 0
                ),
                "max_latency_ms": max(deep_latencies) if deep_latencies else 0,
            },
            "robot_type": self.robot_type.value,
        }

    def set_safety_envelope(self, envelope: SafetyEnvelope) -> None:
        """Update safety envelope."""
        self.safety_envelope = envelope
        logger.info(f"Safety envelope updated")

    def clear_history(self, agent_id: Optional[str] = None) -> None:
        """Clear history for an agent or all agents."""
        if agent_id:
            if agent_id in self._context_history:
                self._context_history[agent_id].clear()
            if agent_id in self._command_timestamps:
                self._command_timestamps[agent_id].clear()
        else:
            self._context_history.clear()
            self._command_timestamps.clear()
