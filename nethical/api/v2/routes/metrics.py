"""Metrics routes for API v2.

Provides Prometheus-compatible metrics and monitoring.

Implements:
- GET /metrics - Prometheus metrics endpoint
- GET /metrics/summary - Human-readable metrics summary

Adheres to:
- Law 10: Reasoning Transparency - System behavior is observable
- Law 15: Audit Compliance - Metrics support auditing
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

# In-memory metrics store (would be Prometheus registry in production)
_metrics: dict[str, Any] = {
    "requests_total": 0,
    "decisions_allow": 0,
    "decisions_restrict": 0,
    "decisions_block": 0,
    "decisions_terminate": 0,
    "violations_total": 0,
    "latency_sum_ms": 0,
    "latency_count": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "appeals_submitted": 0,
    "appeals_granted": 0,
    "fairness_audits": 0,
}


class MetricsSummary(BaseModel):
    """Human-readable metrics summary."""

    requests_total: int = Field(..., description="Total requests processed")
    decisions: dict[str, int] = Field(
        ...,
        description="Decision counts by type",
    )
    violations_total: int = Field(..., description="Total violations detected")
    avg_latency_ms: float = Field(..., description="Average latency in milliseconds")
    cache_hit_rate: float = Field(..., description="Cache hit rate (0.0-1.0)")
    appeals: dict[str, int] = Field(..., description="Appeals statistics")
    fairness_audits: int = Field(..., description="Fairness audits completed")
    uptime_seconds: int = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="Metrics timestamp")


class PrometheusMetrics(BaseModel):
    """Prometheus-compatible metrics format."""

    content_type: str = Field(
        default="text/plain; version=0.0.4",
        description="Content type for Prometheus",
    )
    metrics: str = Field(..., description="Prometheus metrics text")


# Track service start time
_start_time = time.time()


def record_decision(decision: str) -> None:
    """Record a decision in metrics."""
    _metrics["requests_total"] += 1
    decision_key = f"decisions_{decision.lower()}"
    if decision_key in _metrics:
        _metrics[decision_key] += 1


def record_latency(latency_ms: int) -> None:
    """Record request latency."""
    _metrics["latency_sum_ms"] += latency_ms
    _metrics["latency_count"] += 1


def record_violation() -> None:
    """Record a violation."""
    _metrics["violations_total"] += 1


def record_cache_access(hit: bool) -> None:
    """Record a cache access."""
    if hit:
        _metrics["cache_hits"] += 1
    else:
        _metrics["cache_misses"] += 1


def record_appeal(granted: bool = False) -> None:
    """Record an appeal submission."""
    _metrics["appeals_submitted"] += 1
    if granted:
        _metrics["appeals_granted"] += 1


def record_fairness_audit() -> None:
    """Record a fairness audit."""
    _metrics["fairness_audits"] += 1


@router.get("/metrics")
async def prometheus_metrics() -> str:
    """Get Prometheus-compatible metrics.

    Returns metrics in Prometheus text exposition format.
    Supports Law 10 (Reasoning Transparency) through system observability.

    Returns:
        Prometheus metrics text
    """
    uptime = int(time.time() - _start_time)

    # Calculate derived metrics
    avg_latency = 0.0
    if _metrics["latency_count"] > 0:
        avg_latency = _metrics["latency_sum_ms"] / _metrics["latency_count"]

    cache_total = _metrics["cache_hits"] + _metrics["cache_misses"]
    cache_hit_rate = _metrics["cache_hits"] / cache_total if cache_total > 0 else 0.0

    # Build Prometheus format
    lines = [
        "# HELP nethical_requests_total Total number of requests processed",
        "# TYPE nethical_requests_total counter",
        f'nethical_requests_total {_metrics["requests_total"]}',
        "",
        "# HELP nethical_decisions_total Decisions by type",
        "# TYPE nethical_decisions_total counter",
        f'nethical_decisions_total{{decision="allow"}} {_metrics["decisions_allow"]}',
        f'nethical_decisions_total{{decision="restrict"}} {_metrics["decisions_restrict"]}',
        f'nethical_decisions_total{{decision="block"}} {_metrics["decisions_block"]}',
        f'nethical_decisions_total{{decision="terminate"}} {_metrics["decisions_terminate"]}',
        "",
        "# HELP nethical_violations_total Total violations detected",
        "# TYPE nethical_violations_total counter",
        f'nethical_violations_total {_metrics["violations_total"]}',
        "",
        "# HELP nethical_latency_ms_avg Average request latency in milliseconds",
        "# TYPE nethical_latency_ms_avg gauge",
        f"nethical_latency_ms_avg {avg_latency:.2f}",
        "",
        "# HELP nethical_cache_hit_rate Cache hit rate (0.0-1.0)",
        "# TYPE nethical_cache_hit_rate gauge",
        f"nethical_cache_hit_rate {cache_hit_rate:.4f}",
        "",
        "# HELP nethical_cache_hits_total Total cache hits",
        "# TYPE nethical_cache_hits_total counter",
        f'nethical_cache_hits_total {_metrics["cache_hits"]}',
        "",
        "# HELP nethical_cache_misses_total Total cache misses",
        "# TYPE nethical_cache_misses_total counter",
        f'nethical_cache_misses_total {_metrics["cache_misses"]}',
        "",
        "# HELP nethical_appeals_submitted_total Total appeals submitted",
        "# TYPE nethical_appeals_submitted_total counter",
        f'nethical_appeals_submitted_total {_metrics["appeals_submitted"]}',
        "",
        "# HELP nethical_appeals_granted_total Total appeals granted",
        "# TYPE nethical_appeals_granted_total counter",
        f'nethical_appeals_granted_total {_metrics["appeals_granted"]}',
        "",
        "# HELP nethical_fairness_audits_total Total fairness audits completed",
        "# TYPE nethical_fairness_audits_total counter",
        f'nethical_fairness_audits_total {_metrics["fairness_audits"]}',
        "",
        "# HELP nethical_uptime_seconds Service uptime in seconds",
        "# TYPE nethical_uptime_seconds gauge",
        f"nethical_uptime_seconds {uptime}",
        "",
    ]

    return "\n".join(lines)


@router.get("/metrics/summary", response_model=MetricsSummary)
async def metrics_summary() -> MetricsSummary:
    """Get a human-readable metrics summary.

    Provides a JSON summary of key metrics for monitoring
    and dashboard integration.

    Returns:
        MetricsSummary with current metrics
    """
    uptime = int(time.time() - _start_time)

    avg_latency = 0.0
    if _metrics["latency_count"] > 0:
        avg_latency = _metrics["latency_sum_ms"] / _metrics["latency_count"]

    cache_total = _metrics["cache_hits"] + _metrics["cache_misses"]
    cache_hit_rate = _metrics["cache_hits"] / cache_total if cache_total > 0 else 0.0

    return MetricsSummary(
        requests_total=_metrics["requests_total"],
        decisions={
            "allow": _metrics["decisions_allow"],
            "restrict": _metrics["decisions_restrict"],
            "block": _metrics["decisions_block"],
            "terminate": _metrics["decisions_terminate"],
        },
        violations_total=_metrics["violations_total"],
        avg_latency_ms=avg_latency,
        cache_hit_rate=cache_hit_rate,
        appeals={
            "submitted": _metrics["appeals_submitted"],
            "granted": _metrics["appeals_granted"],
        },
        fairness_audits=_metrics["fairness_audits"],
        uptime_seconds=uptime,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
