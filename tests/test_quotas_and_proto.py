# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Unit tests for quotas rate limiting and proto dataclass definitions."""

import threading
import time
import pytest

from nethical.proto import (
    Violation,
    Explanation,
    EvaluateRequest,
    EvaluateResponse,
    Decision,
    Policy,
    BatchEvaluateRequest,
    DecisionStreamRequest,
    GetDecisionRequest,
    ListPoliciesRequest,
    ListPoliciesResponse,
    HealthCheckRequest,
    HealthCheckResponse,
)
from nethical.quotas import (
    QuotaConfig,
    QuotaEnforcer,
    QuotaUsage,
    get_quota_enforcer,
    configure_quotas,
)


def test_proto_mirror_dataclasses():
    """Verify all proto message dataclasses initialise correctly with full parity."""
    # Violation
    v = Violation(
        id="viol-1",
        type="harm_prevention",
        severity="high",
        description="Attempted unsafe command execution",
        law_reference="Law 21",
        evidence={"cmd": "rm -rf /"},
    )
    assert v.id == "viol-1"
    assert v.law_reference == "Law 21"

    # Explanation
    exp = Explanation(
        summary="Action blocked due to safety violations",
        risk_factors=["destruction"],
        decision_rationale="Violates human safety",
        laws_applied=["Law 21"],
        recommendations=["Sanitise command input"],
    )
    assert len(exp.risk_factors) == 1

    # EvaluateRequest & EvaluateResponse
    req = EvaluateRequest(
        agent_id="agent-007",
        action="execute script",
        action_type="execution",
        context={"env": "prod"},
        stated_intent="run migration",
        priority="high",
        require_explanation=True,
        request_id="req-123",
    )
    assert req.agent_id == "agent-007"
    assert req.priority == "high"

    resp = EvaluateResponse(
        decision="BLOCK",
        decision_id="dec-456",
        risk_score=0.95,
        confidence=0.99,
        latency_ms=4,
        violations=[v],
        reason="Severe violation detected",
        explanation=exp,
        audit_id="audit-789",
        cache_hit=False,
        fundamental_laws_checked=[21, 23],
        timestamp="2026-09-24T12:00:00Z",
    )
    assert resp.decision == "BLOCK"
    assert len(resp.violations) == 1
    assert resp.explanation.summary == exp.summary

    # Decision & Policy
    dec = Decision(
        decision_id="dec-456",
        decision="BLOCK",
        agent_id="agent-007",
        action_summary="execute script",
        action_type="execution",
        risk_score=0.95,
        confidence=0.99,
        reasoning="Dangerous filesystem operation",
        violations=[v],
        fundamental_laws=[21],
    )
    assert dec.agent_id == "agent-007"

    pol = Policy(
        policy_id="pol-1",
        name="Filesystem Integrity",
        description="Prevents unauthorized disk alterations",
        version="1.0.0",
        status="active",
        scope="global",
        fundamental_laws=[21],
    )
    assert pol.name == "Filesystem Integrity"

    # Extended proto definitions
    batch_req = BatchEvaluateRequest(requests=[req], parallel=True, fail_fast=False)
    assert len(batch_req.requests) == 1
    assert batch_req.parallel is True

    stream_req = DecisionStreamRequest(
        agent_id="agent-007",
        decision_types=["BLOCK", "TERMINATE"],
        min_risk_score=0.7,
        history_seconds=300,
    )
    assert stream_req.min_risk_score == 0.7

    get_dec = GetDecisionRequest(decision_id="dec-456")
    assert get_dec.decision_id == "dec-456"

    list_req = ListPoliciesRequest(status="active", scope="global", page=1, page_size=10)
    assert list_req.page_size == 10

    list_resp = ListPoliciesResponse(policies=[pol], total_count=1, has_next=False)
    assert list_resp.total_count == 1
    assert list_resp.has_next is False

    hc_req = HealthCheckRequest()
    assert isinstance(hc_req, HealthCheckRequest)

    hc_resp = HealthCheckResponse(status="SERVING", version="1.2.0", uptime_seconds=120)
    assert hc_resp.status == "SERVING"
    assert hc_resp.version == "1.2.0"


def test_quota_enforcer_basics():
    """Test basic quota evaluation, payload limit, and metric tracking."""
    config = QuotaConfig(
        requests_per_second=20.0,
        burst_size=10,
        max_payload_bytes=500,
        rate_window_seconds=10,
    )
    enforcer = QuotaEnforcer(config)

    # 1. Normal request allowed
    res = enforcer.check_quota(agent_id="agent-1", payload_size=100)
    assert res["allowed"] is True
    assert res["decision"] == "ALLOW"
    assert res["enforcement_action"] is None

    # 2. Oversized payload blocked
    res_oversized = enforcer.check_quota(agent_id="agent-1", payload_size=600)
    assert res_oversized["allowed"] is False
    assert res_oversized["decision"] == "BLOCK"
    assert res_oversized["enforcement_action"] == "REJECT_OVERSIZED_PAYLOAD"
    assert "Payload size 600 exceeds limit" in res_oversized["reason"]

    # 3. Usage summary
    summary = enforcer.get_usage_summary("agent-1")
    assert summary["entity_id"] == "agent-1"
    assert summary["total_requests"] == 1
    assert summary["total_bytes"] == 100
    assert summary["first_request"] is not None
    assert summary["last_request"] is not None


def test_quota_burst_throttling_and_blocking():
    """Test burst limit triggers THROTTLE and BLOCK thresholds accurately."""
    # Burst size = 10, throttle at 80% (8 requests in burst window), block at 95% (~10 requests)
    config = QuotaConfig(
        requests_per_second=100.0,
        burst_size=10,
        throttle_threshold=0.8,
        block_threshold=0.95,
        burst_window_seconds=2,
    )
    enforcer = QuotaEnforcer(config)

    decisions = []
    for _ in range(12):
        dec = enforcer.check_quota(agent_id="burst-agent", payload_size=10)
        decisions.append(dec["decision"])

    # First several are ALLOW
    assert decisions[0] == "ALLOW"
    # Approaching 80% triggers THROTTLE
    assert "THROTTLE" in decisions
    # Crossing 95% triggers BLOCK
    assert "BLOCK" in decisions

    summary = enforcer.get_usage_summary("burst-agent")
    assert summary["throttle_count"] > 0
    assert summary["block_count"] > 0
    assert summary["last_violation"] is not None


def test_quota_action_rate_limiting():
    """Test action-specific rate limiting against actions_per_minute."""
    config = QuotaConfig(
        requests_per_second=100.0,
        burst_size=50,
        actions_per_minute=5,
        throttle_threshold=0.6,
        block_threshold=0.9,
    )
    enforcer = QuotaEnforcer(config)

    # 5 actions in a row are allowed
    for i in range(5):
        res = enforcer.check_quota(agent_id="action-agent", action_type="mutation")
        assert res["allowed"] is True

    # 6th action hits 100% capacity utilization -> BLOCK
    res_blocked = enforcer.check_quota(agent_id="action-agent", action_type="mutation")
    assert res_blocked["allowed"] is False
    assert res_blocked["decision"] == "BLOCK"
    assert res_blocked["enforcement_action"] == "RATE_LIMIT_BLOCK"

    summary = enforcer.get_usage_summary("action-agent")
    assert summary["total_actions"] == 5  # 5 succeeded before block


def test_quota_cohort_and_tenant_isolation():
    """Test multi-tenant and cohort isolation limits."""
    config = QuotaConfig(
        requests_per_second=5.0,
        burst_size=4,
        block_threshold=0.95,
        enable_cohort_isolation=True,
        enable_tenant_isolation=True,
    )
    enforcer = QuotaEnforcer(config)

    # Exhaust tenant-A quota using agent-1
    for _ in range(5):
        enforcer.check_quota(agent_id="agent-1", tenant="tenant-A")

    # Another agent on tenant-A should be blocked
    res_tenant_a = enforcer.check_quota(agent_id="agent-2", tenant="tenant-A")
    assert res_tenant_a["allowed"] is False
    assert "tenant tenant-A exceeded quota" in res_tenant_a["reason"]

    # But agent-3 on tenant-B should be allowed
    res_tenant_b = enforcer.check_quota(agent_id="agent-3", tenant="tenant-B")
    assert res_tenant_b["allowed"] is True
    assert res_tenant_b["decision"] == "ALLOW"


def test_quota_thread_safety():
    """Test concurrent thread safety without race conditions or data corruption."""
    config = QuotaConfig(
        requests_per_second=1000.0,
        burst_size=2000,
        max_payload_bytes=10000,
    )
    enforcer = QuotaEnforcer(config)

    thread_count = 10
    calls_per_thread = 50

    def worker(worker_id: int):
        for _ in range(calls_per_thread):
            enforcer.check_quota(
                agent_id=f"agent-{worker_id}",
                tenant="shared-tenant",
                payload_size=10,
                action_type="query",
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    metrics = enforcer.get_global_metrics()
    expected_total = thread_count * calls_per_thread
    assert metrics["total_requests"] == expected_total
    assert metrics["total_bytes_processed"] == expected_total * 10
    assert metrics["tracked_entities"] == thread_count + 1  # 10 agents + 1 shared tenant


def test_quota_metrics_and_reset():
    """Test global metrics reporting and entity/global resets."""
    config = QuotaConfig(requests_per_second=10.0)
    enforcer = QuotaEnforcer(config)

    enforcer.check_quota(agent_id="agent-x", payload_size=50)
    enforcer.check_quota(agent_id="agent-y", payload_size=75)

    metrics = enforcer.get_global_metrics()
    assert metrics["total_requests"] == 2
    assert metrics["total_bytes_processed"] == 125
    assert metrics["tracked_entities"] == 2

    # Reset single entity
    enforcer.reset_usage("agent-x")
    assert "error" in enforcer.get_usage_summary("agent-x")
    assert enforcer.get_usage_summary("agent-y")["total_requests"] == 1

    # Reset all
    enforcer.reset_usage()
    assert enforcer.get_global_metrics()["total_requests"] == 0
    assert enforcer.get_global_metrics()["tracked_entities"] == 0


def test_singleton_get_and_configure():
    """Test global singleton accessor and reconfiguration."""
    conf1 = QuotaConfig(requests_per_second=15.0)
    e1 = configure_quotas(conf1)
    e2 = get_quota_enforcer()
    assert e1 is e2
    assert e2.config.requests_per_second == 15.0

    conf2 = QuotaConfig(requests_per_second=42.0)
    e3 = configure_quotas(conf2)
    assert e3.config.requests_per_second == 42.0
    assert get_quota_enforcer() is e3
