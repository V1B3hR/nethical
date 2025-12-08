"""
Nethical Runtime Verification Module

This module provides continuous runtime verification of safety invariants
for the Nethical governance system. It monitors the system state and
verifies that critical safety properties are maintained at runtime.

Key Features:
- Real-time invariant checking
- Violation detection and alerting
- Automatic safe-mode triggering
- Comprehensive audit logging
"""

from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading
import logging
import json
import time

logger = logging.getLogger(__name__)


class InvariantSeverity(Enum):
    """Severity levels for invariant violations."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class InvariantStatus(Enum):
    """Status of an invariant check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class InvariantViolation:
    """Record of an invariant violation."""

    invariant_name: str
    severity: InvariantSeverity
    timestamp: datetime
    details: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_action: str = ""
    auto_remediated: bool = False


@dataclass
class InvariantDefinition:
    """Definition of a safety invariant."""

    name: str
    description: str
    check_function: Callable[[Dict[str, Any]], bool]
    severity: InvariantSeverity = InvariantSeverity.CRITICAL
    enabled: bool = True
    check_frequency_ms: int = 100
    auto_remediate: bool = False
    remediation_function: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class RuntimeState:
    """Current runtime state of the governance system."""

    agent_states: Dict[str, str] = field(default_factory=dict)
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    terminated_agents: Set[str] = field(default_factory=set)
    risk_scores: Dict[str, float] = field(default_factory=dict)
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    latency_metrics: Dict[str, float] = field(default_factory=dict)
    policy_state: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class RuntimeVerifier:
    """
    Runtime monitor for safety invariant verification.

    This class continuously monitors the governance system state and
    verifies that critical safety invariants are maintained.

    Verified Invariants:
    - NoAllowAfterTerminate: Terminated agents cannot have actions allowed
    - DecisionLatencyBound: Decisions are made within SLO
    - AuditLogIntegrity: Audit logs are complete and consistent
    - PolicyConsistency: Active policies are non-contradictory
    - RiskScoreBounds: Risk scores are within valid range
    - PendingActionsBounded: Pending actions don't exceed threshold
    """

    def __init__(
        self,
        max_violations_before_halt: int = 3,
        enable_auto_remediation: bool = True,
        monitoring_interval_ms: int = 100,
    ):
        """
        Initialize the runtime verifier.

        Args:
            max_violations_before_halt: Maximum violations before triggering safe mode
            enable_auto_remediation: Whether to auto-remediate violations
            monitoring_interval_ms: How often to check invariants (ms)
        """
        self._state = RuntimeState()
        self._invariants: Dict[str, InvariantDefinition] = {}
        self._violations: List[InvariantViolation] = []
        self._max_violations = max_violations_before_halt
        self._auto_remediation = enable_auto_remediation
        self._monitoring_interval_ms = monitoring_interval_ms
        self._running = False
        self._safe_mode = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._violation_handlers: List[Callable[[InvariantViolation], None]] = []

        # Register default invariants
        self._register_default_invariants()

    def _register_default_invariants(self) -> None:
        """Register the default safety invariants."""

        # Invariant 1: No ALLOW after TERMINATE
        self.register_invariant(
            InvariantDefinition(
                name="no_allow_after_terminate",
                description="Terminated agents cannot have actions allowed",
                check_function=self._check_no_allow_after_terminate,
                severity=InvariantSeverity.FATAL,
                auto_remediate=True,
                remediation_function=self._remediate_no_allow_after_terminate,
            )
        )

        # Invariant 2: Decision latency within SLO
        self.register_invariant(
            InvariantDefinition(
                name="decision_latency_bound",
                description="Decision latency is within SLO bounds",
                check_function=self._check_decision_latency_bound,
                severity=InvariantSeverity.WARNING,
                check_frequency_ms=500,
            )
        )

        # Invariant 3: Audit log integrity
        self.register_invariant(
            InvariantDefinition(
                name="audit_log_integrity",
                description="Audit logs are complete and sequential",
                check_function=self._check_audit_log_integrity,
                severity=InvariantSeverity.CRITICAL,
            )
        )

        # Invariant 4: Policy consistency
        self.register_invariant(
            InvariantDefinition(
                name="policy_consistency",
                description="Active policies are non-contradictory",
                check_function=self._check_policy_consistency,
                severity=InvariantSeverity.CRITICAL,
                check_frequency_ms=1000,
            )
        )

        # Invariant 5: Risk score bounds
        self.register_invariant(
            InvariantDefinition(
                name="risk_score_bounds",
                description="All risk scores are within [0, 1]",
                check_function=self._check_risk_score_bounds,
                severity=InvariantSeverity.CRITICAL,
                auto_remediate=True,
                remediation_function=self._remediate_risk_score_bounds,
            )
        )

        # Invariant 6: Pending actions bounded
        self.register_invariant(
            InvariantDefinition(
                name="pending_actions_bounded",
                description="Pending actions do not exceed threshold",
                check_function=self._check_pending_actions_bounded,
                severity=InvariantSeverity.WARNING,
                check_frequency_ms=200,
            )
        )

        # Invariant 7: Agent state consistency
        self.register_invariant(
            InvariantDefinition(
                name="agent_state_consistency",
                description="Agent states are consistent with history",
                check_function=self._check_agent_state_consistency,
                severity=InvariantSeverity.CRITICAL,
            )
        )

        # Invariant 8: Safe mode enforcement
        self.register_invariant(
            InvariantDefinition(
                name="safe_mode_enforcement",
                description="Safe mode is properly enforced when triggered",
                check_function=self._check_safe_mode_enforcement,
                severity=InvariantSeverity.FATAL,
            )
        )

    def register_invariant(self, invariant: InvariantDefinition) -> None:
        """Register a new invariant to be verified."""
        with self._lock:
            self._invariants[invariant.name] = invariant
            logger.info(f"Registered invariant: {invariant.name}")

    def unregister_invariant(self, name: str) -> bool:
        """Unregister an invariant."""
        with self._lock:
            if name in self._invariants:
                del self._invariants[name]
                logger.info(f"Unregistered invariant: {name}")
                return True
            return False

    def add_violation_handler(
        self, handler: Callable[[InvariantViolation], None]
    ) -> None:
        """Add a handler to be called when violations occur."""
        self._violation_handlers.append(handler)

    def update_state(
        self,
        agent_id: Optional[str] = None,
        agent_state: Optional[str] = None,
        decision: Optional[Dict[str, Any]] = None,
        risk_score: Optional[float] = None,
        pending_action: Optional[Dict[str, Any]] = None,
        latency_metric: Optional[Dict[str, float]] = None,
        policy_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the runtime state.

        This method is called whenever the governance system state changes.
        After updating the state, invariants are immediately checked.
        """
        with self._lock:
            if agent_id and agent_state:
                self._state.agent_states[agent_id] = agent_state
                if agent_state == "TERMINATED":
                    self._state.terminated_agents.add(agent_id)

            if decision:
                self._state.decision_history.append(decision)

            if agent_id and risk_score is not None:
                self._state.risk_scores[agent_id] = risk_score

            if pending_action:
                self._state.pending_actions.append(pending_action)

            if latency_metric:
                self._state.latency_metrics.update(latency_metric)

            if policy_update:
                self._state.policy_state.update(policy_update)

            self._state.last_updated = datetime.now()

        # Check invariants after state update
        self._check_all_invariants()

    def verify_invariant(self, invariant_name: str) -> InvariantStatus:
        """
        Verify a specific invariant.

        Args:
            invariant_name: Name of the invariant to verify

        Returns:
            Status of the invariant check
        """
        with self._lock:
            if invariant_name not in self._invariants:
                return InvariantStatus.SKIPPED

            invariant = self._invariants[invariant_name]
            if not invariant.enabled:
                return InvariantStatus.SKIPPED

            try:
                state_dict = self._get_state_dict()
                passed = invariant.check_function(state_dict)

                if passed:
                    return InvariantStatus.PASSED
                else:
                    self._handle_violation(invariant, state_dict)
                    return InvariantStatus.FAILED

            except Exception as e:
                logger.error(f"Error checking invariant {invariant_name}: {e}")
                return InvariantStatus.ERROR

    def _check_all_invariants(self) -> Dict[str, InvariantStatus]:
        """Check all registered invariants."""
        results = {}
        for name, invariant in self._invariants.items():
            if invariant.enabled:
                results[name] = self.verify_invariant(name)
        return results

    def _get_state_dict(self) -> Dict[str, Any]:
        """Convert runtime state to dictionary for invariant checks."""
        return {
            "agent_states": dict(self._state.agent_states),
            "decision_history": list(self._state.decision_history),
            "terminated_agents": set(self._state.terminated_agents),
            "risk_scores": dict(self._state.risk_scores),
            "pending_actions": list(self._state.pending_actions),
            "latency_metrics": dict(self._state.latency_metrics),
            "policy_state": dict(self._state.policy_state),
            "safe_mode": self._safe_mode,
        }

    def _handle_violation(
        self, invariant: InvariantDefinition, state: Dict[str, Any]
    ) -> None:
        """Handle an invariant violation."""
        violation = InvariantViolation(
            invariant_name=invariant.name,
            severity=invariant.severity,
            timestamp=datetime.now(),
            details=f"Invariant '{invariant.description}' violated",
            evidence={"state_snapshot": state},
        )

        self._violations.append(violation)
        logger.warning(f"Invariant violation: {violation}")

        # Notify handlers
        for handler in self._violation_handlers:
            try:
                handler(violation)
            except Exception as e:
                logger.error(f"Error in violation handler: {e}")

        # Auto-remediate if enabled
        if self._auto_remediation and invariant.auto_remediate:
            if invariant.remediation_function:
                try:
                    invariant.remediation_function(state)
                    violation.auto_remediated = True
                    logger.info(f"Auto-remediated violation: {invariant.name}")
                except Exception as e:
                    logger.error(f"Auto-remediation failed: {e}")

        # Check if we should enter safe mode
        fatal_violations = sum(
            1 for v in self._violations if v.severity == InvariantSeverity.FATAL
        )

        if fatal_violations >= 1 or len(self._violations) >= self._max_violations:
            self._trigger_safe_mode()

    def _trigger_safe_mode(self) -> None:
        """Trigger safe mode to halt dangerous operations."""
        self._safe_mode = True
        logger.critical("SAFE MODE TRIGGERED - Restricting all operations")

        # Create safe mode violation record
        violation = InvariantViolation(
            invariant_name="safe_mode_trigger",
            severity=InvariantSeverity.FATAL,
            timestamp=datetime.now(),
            details="Safe mode triggered due to critical violations",
            evidence={"total_violations": len(self._violations)},
        )
        self._violations.append(violation)

    def is_safe_mode(self) -> bool:
        """Check if system is in safe mode."""
        return self._safe_mode

    def exit_safe_mode(self, authorization_key: str) -> bool:
        """
        Exit safe mode with proper authorization.

        Args:
            authorization_key: Authorization key to exit safe mode

        Returns:
            True if safe mode was exited
        """
        # In production, this would verify the authorization key
        if authorization_key and len(authorization_key) >= 8:
            self._safe_mode = False
            self._violations.clear()
            logger.info("Safe mode exited with authorization")
            return True
        return False

    def get_violations(
        self, severity: Optional[InvariantSeverity] = None, limit: int = 100
    ) -> List[InvariantViolation]:
        """Get recorded violations, optionally filtered by severity."""
        with self._lock:
            violations = self._violations
            if severity:
                violations = [v for v in violations if v.severity == severity]
            return violations[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics."""
        with self._lock:
            return {
                "total_invariants": len(self._invariants),
                "enabled_invariants": sum(
                    1 for i in self._invariants.values() if i.enabled
                ),
                "total_violations": len(self._violations),
                "violations_by_severity": {
                    s.value: sum(1 for v in self._violations if v.severity == s)
                    for s in InvariantSeverity
                },
                "safe_mode": self._safe_mode,
                "monitoring_active": self._running,
                "last_state_update": self._state.last_updated.isoformat(),
            }

    def start_monitoring(self) -> None:
        """Start continuous monitoring in a background thread."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()
        logger.info("Runtime verification monitoring started")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
        logger.info("Runtime verification monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self._check_all_invariants()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self._monitoring_interval_ms / 1000.0)

    # ========================
    # Invariant Check Functions
    # ========================

    def _check_no_allow_after_terminate(self, state: Dict[str, Any]) -> bool:
        """Check that terminated agents don't have ALLOW decisions."""
        terminated = state.get("terminated_agents", set())
        history = state.get("decision_history", [])

        for decision in history:
            agent_id = decision.get("agent_id")
            decision_type = decision.get("decision")

            if agent_id in terminated and decision_type == "ALLOW":
                # Check if TERMINATE came before ALLOW
                terminate_idx = -1
                allow_idx = -1

                for i, d in enumerate(history):
                    if d.get("agent_id") == agent_id:
                        if d.get("decision") == "TERMINATE":
                            terminate_idx = i
                        elif d.get("decision") == "ALLOW" and i > terminate_idx >= 0:
                            return False  # ALLOW after TERMINATE

        return True

    def _remediate_no_allow_after_terminate(self, state: Dict[str, Any]) -> None:
        """Remediate by converting ALLOW to RESTRICT for terminated agents."""
        terminated = state.get("terminated_agents", set())
        history = self._state.decision_history

        for i, decision in enumerate(history):
            if decision.get("agent_id") in terminated:
                if decision.get("decision") == "ALLOW":
                    history[i]["decision"] = "RESTRICT"
                    history[i]["remediated"] = True

    def _check_decision_latency_bound(self, state: Dict[str, Any]) -> bool:
        """Check that decision latency is within SLO."""
        latency_metrics = state.get("latency_metrics", {})

        # Check p99 latency (default SLO: 250ms)
        p99_latency = latency_metrics.get("p99_ms", 0)
        return p99_latency <= 250

    def _check_audit_log_integrity(self, state: Dict[str, Any]) -> bool:
        """Check that audit logs are sequential and complete."""
        history = state.get("decision_history", [])

        if not history:
            return True

        # Check for sequential timestamps
        prev_timestamp = None
        for decision in history:
            timestamp = decision.get("timestamp")
            if timestamp and prev_timestamp:
                if timestamp < prev_timestamp:
                    return False  # Out of order
            prev_timestamp = timestamp

        return True

    def _check_policy_consistency(self, state: Dict[str, Any]) -> bool:
        """Check that active policies are non-contradictory."""
        policy_state = state.get("policy_state", {})
        active_policies = policy_state.get("active_policies", [])

        # Check for contradictions between same-priority policies
        policy_map: Dict[int, List[Dict]] = {}
        for policy in active_policies:
            priority = policy.get("priority", 0)
            if priority not in policy_map:
                policy_map[priority] = []
            policy_map[priority].append(policy)

        for priority, policies in policy_map.items():
            decisions = set()
            for p in policies:
                decisions.add(p.get("decision"))

            # Check for ALLOW + BLOCK contradiction
            if "ALLOW" in decisions and (
                "BLOCK" in decisions or "TERMINATE" in decisions
            ):
                return False

        return True

    def _check_risk_score_bounds(self, state: Dict[str, Any]) -> bool:
        """Check that all risk scores are within [0, 1]."""
        risk_scores = state.get("risk_scores", {})

        for agent_id, score in risk_scores.items():
            if score < 0 or score > 1:
                return False

        return True

    def _remediate_risk_score_bounds(self, state: Dict[str, Any]) -> None:
        """Clamp risk scores to valid range."""
        for agent_id, score in self._state.risk_scores.items():
            if score < 0:
                self._state.risk_scores[agent_id] = 0
            elif score > 1:
                self._state.risk_scores[agent_id] = 1

    def _check_pending_actions_bounded(self, state: Dict[str, Any]) -> bool:
        """Check that pending actions don't exceed threshold."""
        pending = state.get("pending_actions", [])
        return len(pending) <= 1000  # Default threshold

    def _check_agent_state_consistency(self, state: Dict[str, Any]) -> bool:
        """Check that agent states are consistent with history."""
        agent_states = state.get("agent_states", {})
        terminated = state.get("terminated_agents", set())

        # Terminated agents should have TERMINATED state
        for agent_id in terminated:
            if agent_states.get(agent_id) != "TERMINATED":
                return False

        return True

    def _check_safe_mode_enforcement(self, state: Dict[str, Any]) -> bool:
        """Check that safe mode is properly enforced."""
        if not state.get("safe_mode"):
            return True  # Not in safe mode, nothing to check

        # In safe mode, no ALLOW decisions should be made
        history = state.get("decision_history", [])
        if history:
            last_decision = history[-1]
            if last_decision.get("decision") == "ALLOW":
                if last_decision.get("after_safe_mode", False):
                    return False

        return True


# Global runtime verifier instance
_runtime_verifier: Optional[RuntimeVerifier] = None


def get_runtime_verifier() -> RuntimeVerifier:
    """Get or create the global runtime verifier."""
    global _runtime_verifier
    if _runtime_verifier is None:
        _runtime_verifier = RuntimeVerifier()
    return _runtime_verifier


def verify_before_decision(agent_id: str, action: str, proposed_decision: str) -> bool:
    """
    Verify invariants before making a decision.

    This function should be called before any governance decision
    is finalized to ensure it won't violate safety invariants.

    Args:
        agent_id: The agent making the request
        action: The proposed action
        proposed_decision: The proposed governance decision

    Returns:
        True if the decision is safe to proceed
    """
    verifier = get_runtime_verifier()

    # Check if in safe mode
    if verifier.is_safe_mode():
        if proposed_decision == "ALLOW":
            logger.warning(f"Blocking ALLOW decision in safe mode: {agent_id}")
            return False

    # Check no-allow-after-terminate
    state = verifier._get_state_dict()
    if agent_id in state.get("terminated_agents", set()):
        if proposed_decision == "ALLOW":
            logger.warning(f"Blocking ALLOW for terminated agent: {agent_id}")
            return False

    return True


# Export classes and functions
__all__ = [
    "RuntimeVerifier",
    "InvariantDefinition",
    "InvariantViolation",
    "InvariantSeverity",
    "InvariantStatus",
    "RuntimeState",
    "get_runtime_verifier",
    "verify_before_decision",
]
