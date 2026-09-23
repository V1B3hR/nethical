"""
Appeals Metrics Collector

Collects and computes appeals processing metrics including volume,
resolution times, and outcome distribution for the governance dashboard.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _ensure_utc(dt: Optional[datetime]) -> datetime:
    """Ensure datetime has UTC timezone."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class Appeal:
    """Record of an appeal."""
    appeal_id: str
    decision_id: str
    filed_at: datetime
    resolved_at: Optional[datetime]
    outcome: Optional[str]  # "upheld", "overturned", "modified", "withdrawn"
    resolution_hours: Optional[float]


class AppealsMetricsCollector:
    """
    Appeals Metrics Collector

    Tracks appeals processing metrics for sovereign governance dashboard monitoring.
    Optimised with bounded double-ended queues for O(1) ingestion.
    """

    def __init__(self, max_appeals: int = 10000) -> None:
        """Initialise appeals metrics collector."""
        self._max_appeals: int = max_appeals
        self._appeals: deque[Appeal] = deque(maxlen=self._max_appeals)

    def record_appeal(
        self,
        appeal_id: str,
        decision_id: str,
        filed_at: Optional[datetime] = None,
    ) -> None:
        """
        Record a new appeal.

        Args:
            appeal_id: Appeal identifier
            decision_id: Original decision ID
            filed_at: Filing timestamp (UTC normalised)
        """
        appeal = Appeal(
            appeal_id=appeal_id,
            decision_id=decision_id,
            filed_at=_ensure_utc(filed_at),
            resolved_at=None,
            outcome=None,
            resolution_hours=None,
        )
        self._appeals.append(appeal)

    def resolve_appeal(
        self,
        appeal_id: str,
        outcome: str,
        resolved_at: Optional[datetime] = None,
    ) -> None:
        """
        Mark an appeal as resolved.

        Args:
            appeal_id: Appeal identifier
            outcome: Resolution outcome
            resolved_at: Resolution timestamp
        """
        resolved_ts = _ensure_utc(resolved_at)
        for appeal in self._appeals:
            if appeal.appeal_id == appeal_id:
                appeal.resolved_at = resolved_ts
                appeal.outcome = outcome
                appeal.resolution_hours = (
                    (resolved_ts - appeal.filed_at).total_seconds() / 3600.0
                )
                break

    def get_volume_metrics(self) -> Dict[str, Any]:
        """
        Get appeal volume metrics.

        Returns:
            Volume statistics dictionary
        """
        total_appeals = len(self._appeals)
        pending_appeals = sum(1 for a in self._appeals if a.resolved_at is None)
        resolved_appeals = total_appeals - pending_appeals

        # Appeals per day
        if self._appeals:
            oldest = min(a.filed_at for a in self._appeals)
            days = max(1, (datetime.now(timezone.utc) - oldest).days)
            appeals_per_day = total_appeals / float(days)
        else:
            appeals_per_day = 0.0

        return {
            "total_appeals": total_appeals,
            "pending_appeals": pending_appeals,
            "resolved_appeals": resolved_appeals,
            "appeals_per_day": appeals_per_day,
        }

    def get_resolution_metrics(self) -> Dict[str, Any]:
        """
        Get appeal resolution time metrics.

        Returns:
            Resolution time statistics dictionary
        """
        resolved = [a for a in self._appeals if a.resolution_hours is not None]

        if not resolved:
            return {
                "median_hours": 0.0,
                "p95_hours": 0.0,
                "p99_hours": 0.0,
                "slo_target_hours": 72,
                "slo_compliance_rate": 1.0,
                "sample_size": 0,
            }

        resolution_times = sorted(a.resolution_hours for a in resolved)
        n = len(resolution_times)

        median = resolution_times[n // 2]
        p95 = resolution_times[min(n - 1, int(n * 0.95))]
        p99 = resolution_times[min(n - 1, int(n * 0.99))]

        # Check SLO compliance (72 hours target)
        slo_target = 72
        within_slo = sum(1 for t in resolution_times if t <= slo_target)
        slo_compliance_rate = within_slo / float(n)

        return {
            "median_hours": median,
            "p95_hours": p95,
            "p99_hours": p99,
            "slo_target_hours": slo_target,
            "slo_compliance_rate": slo_compliance_rate,
            "sample_size": n,
            "status": "healthy" if median <= slo_target else "warning",
        }

    def get_outcome_distribution(self) -> Dict[str, Any]:
        """
        Get distribution of appeal outcomes.

        Returns:
            Outcome distribution dictionary
        """
        resolved = [a for a in self._appeals if a.outcome is not None]

        if not resolved:
            return {
                "total": 0,
                "distribution": {},
            }

        outcomes = Counter(a.outcome for a in resolved)
        total = len(resolved)

        distribution = {
            outcome: {
                "count": count,
                "percentage": (count / float(total)) * 100.0,
            }
            for outcome, count in outcomes.items()
        }

        return {
            "total": total,
            "distribution": distribution,
        }
