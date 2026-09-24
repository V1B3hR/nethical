# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Comprehensive tests for nethical.verification runtime monitoring module."""

import time
from datetime import datetime, timezone
import pytest

from nethical.verification import (
    RuntimeVerifier,
    InvariantDefinition,
    InvariantViolation,
    InvariantSeverity,
    InvariantStatus,
    RuntimeState,
    get_runtime_verifier,
    verify_before_decision,
)


def test_runtime_state_utc_timestamp():
    """Verify RuntimeState initializes with timezone-aware UTC datetime."""
    state = RuntimeState()
    assert state.last_updated.tzinfo is not None
    assert state.last_updated.tzinfo == timezone.utc


def test_no_allow_after_terminate_invariant_and_remediation():
    """Verify terminated agents cannot receive ALLOW decisions, and auto-remediates to RESTRICT."""
    verifier = RuntimeVerifier(max_violations_before_halt=5, enable_auto_remediation=True)

    # Agent is terminated
    verifier.update_state(agent_id="agent-007", agent_state="TERMINATED")

    # A subsequent ALLOW decision occurs
    verifier.update_state(decision={
        "agent_id": "agent-007",
        "decision": "ALLOW",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # The invariant should have failed and auto-remediated
    violations = verifier.get_violations()
    assert len(violations) >= 1
    assert violations[0].invariant_name == "no_allow_after_terminate"
    assert violations[0].auto_remediated is True

    # Check that decision history was remediated to RESTRICT
    state_dict = verifier._get_state_dict()
    assert state_dict["decision_history"][-1]["decision"] == "RESTRICT"


def test_risk_score_bounds_and_remediation():
    """Verify risk scores outside [0, 1] violate invariant and get clamped."""
    verifier = RuntimeVerifier(enable_auto_remediation=True)

    verifier.update_state(agent_id="bad-agent", risk_score=1.5)
    
    violations = verifier.get_violations()
    assert len(violations) >= 1
    assert violations[0].invariant_name == "risk_score_bounds"
    assert violations[0].auto_remediated is True

    # Check clamped
    assert verifier._state.risk_scores["bad-agent"] == 1.0

    # Without auto-remediation, verify_invariant returns FAILED and leaves value intact
    v_no_remedy = RuntimeVerifier(enable_auto_remediation=False)
    v_no_remedy._state.risk_scores["bad-agent"] = 1.5
    status = v_no_remedy.verify_invariant("risk_score_bounds")
    assert status == InvariantStatus.FAILED
    assert v_no_remedy._state.risk_scores["bad-agent"] == 1.5


def test_decision_latency_bound():
    """Verify decision latency bound check passes within SLO (250ms) and fails above it."""
    verifier = RuntimeVerifier()

    verifier.update_state(latency_metric={"p99_ms": 120.0})
    assert verifier.verify_invariant("decision_latency_bound") == InvariantStatus.PASSED

    verifier.update_state(latency_metric={"p99_ms": 350.0})
    assert verifier.verify_invariant("decision_latency_bound") == InvariantStatus.FAILED


def test_audit_log_integrity():
    """Verify audit logs must be sequential and detect out-of-order timestamps."""
    verifier = RuntimeVerifier()

    # Sequential timestamps -> PASS
    verifier.update_state(decision={"id": 1, "timestamp": "2026-01-01T10:00:00Z"})
    verifier.update_state(decision={"id": 2, "timestamp": "2026-01-01T10:05:00Z"})
    assert verifier.verify_invariant("audit_log_integrity") == InvariantStatus.PASSED

    # Out of order timestamp -> FAIL
    verifier.update_state(decision={"id": 3, "timestamp": "2026-01-01T09:00:00Z"})
    assert verifier.verify_invariant("audit_log_integrity") == InvariantStatus.FAILED


def test_policy_consistency():
    """Verify contradictory decisions at same priority trigger violation."""
    verifier = RuntimeVerifier()

    # Contradictory policies at same priority 1
    verifier.update_state(policy_update={
        "active_policies": [
            {"id": "pol_allow", "priority": 1, "decision": "ALLOW"},
            {"id": "pol_block", "priority": 1, "decision": "BLOCK"},
        ]
    })
    assert verifier.verify_invariant("policy_consistency") == InvariantStatus.FAILED


def test_safe_mode_trigger_and_exit():
    """Verify fatal violation or threshold triggers safe mode, and authorization exits it."""
    verifier = RuntimeVerifier(max_violations_before_halt=2)
    assert verifier.is_safe_mode() is False

    # Trigger safe mode manually or via threshold
    verifier._trigger_safe_mode()
    assert verifier.is_safe_mode() is True

    # Invalid authorization key
    assert verifier.exit_safe_mode("short") is False
    assert verifier.is_safe_mode() is True

    # Valid authorization key (>= 8 chars)
    assert verifier.exit_safe_mode("valid-secret-key") is True
    assert verifier.is_safe_mode() is False
    assert len(verifier.get_violations()) == 0


def test_verify_before_decision_guard():
    """Verify pre-decision gate blocks ALLOW in safe mode or for terminated agents."""
    verifier = RuntimeVerifier()

    # Normal active agent -> ALLOW is safe
    assert verify_before_decision("agent-1", "read_file", "ALLOW", verifier=verifier) is True

    # Terminated agent -> ALLOW is blocked
    verifier.update_state(agent_id="agent-rogue", agent_state="TERMINATED")
    assert verify_before_decision("agent-rogue", "read_file", "ALLOW", verifier=verifier) is False

    # Safe mode active -> ALL ALLOW decisions blocked
    verifier.exit_safe_mode("reset-key-12345")
    verifier._trigger_safe_mode()
    assert verify_before_decision("agent-1", "read_file", "ALLOW", verifier=verifier) is False
    # Non-ALLOW decisions still proceed
    assert verify_before_decision("agent-1", "read_file", "BLOCK", verifier=verifier) is True


def test_custom_invariant_registration_and_handler():
    """Verify registering a custom invariant and handling violation callbacks."""
    verifier = RuntimeVerifier()
    caught_violations = []

    verifier.add_violation_handler(lambda v: caught_violations.append(v.invariant_name))

    custom_inv = InvariantDefinition(
        name="custom_check",
        description="Verify custom rule",
        check_function=lambda state: False,
        severity=InvariantSeverity.WARNING,
    )

    verifier.register_invariant(custom_inv)
    status = verifier.verify_invariant("custom_check")

    assert status == InvariantStatus.FAILED
    assert "custom_check" in caught_violations

    assert verifier.unregister_invariant("custom_check") is True
    assert verifier.unregister_invariant("non_existent") is False


def test_background_monitoring_thread():
    """Verify background monitor loop starts and stops cleanly."""
    verifier = RuntimeVerifier(monitoring_interval_ms=50)
    verifier.start_monitoring()
    assert verifier._running is True

    time.sleep(0.1)
    verifier.stop_monitoring()
    assert verifier._running is False
    assert verifier._monitor_thread is None
