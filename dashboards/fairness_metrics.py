"""
Fairness Metrics Collector

Collects and computes sovereign AI governance fairness metrics including Statistical Parity,
Disparate Impact, and Equal Opportunity across protected attributes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence


def _ensure_utc(dt: Optional[datetime]) -> datetime:
    """Ensure datetime has UTC timezone."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class DecisionRecord:
    """Record of an algorithmic decision for fairness analysis."""
    decision: str  # "allow" or "deny"
    protected_group: str
    timestamp: datetime
    context: Dict[str, Any]


class FairnessMetricsCollector:
    """
    Fairness Metrics Collector

    Computes ethical and compliance fairness metrics across protected attributes:
    - Statistical Parity: P(allow|protected) - P(allow|unprotected)
    - Disparate Impact: P(allow|protected) / P(allow|unprotected)
    - Equal Opportunity: TPR(protected) - TPR(unprotected)
    """

    def __init__(
        self,
        protected_attributes: Optional[Sequence[str]] = None,
        window_hours: int = 24,
        max_decisions: int = 100000,
    ) -> None:
        """
        Initialise fairness metrics collector.

        Args:
            protected_attributes: List of protected attributes to monitor
            window_hours: Time window for metrics computation
            max_decisions: Maximum history capacity kept in bounded deque
        """
        self.protected_attributes: List[str] = list(protected_attributes or [])
        self.window_hours: int = window_hours
        self._max_decisions: int = max_decisions
        self._decisions: deque[DecisionRecord] = deque(maxlen=self._max_decisions)

    def record_decision(
        self,
        decision: str,
        protected_group: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Record a decision for fairness analysis.

        Args:
            decision: Decision outcome ("allow" or "deny")
            protected_group: Protected group identifier (or None / "unprotected")
            context: Additional contextual attributes
            timestamp: Optional decision timestamp
        """
        record = DecisionRecord(
            decision=decision,
            protected_group=protected_group if protected_group is not None else "unprotected",
            timestamp=_ensure_utc(timestamp),
            context=context or {},
        )
        self._decisions.append(record)

    def get_statistical_parity(
        self,
        attribute: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute Statistical Parity metric.

        Formula: P(decision=allow|protected) - P(decision=allow|unprotected)
        Threshold: |difference| <= 0.10 (healthy)

        Args:
            attribute: Specific attribute to analyse (None = all)

        Returns:
            Statistical parity metrics dictionary
        """
        recent = self._get_recent_decisions()

        if not recent:
            return {
                "difference": 0.0,
                "protected_rate": 0.0,
                "unprotected_rate": 0.0,
                "status": "insufficient_data",
                "sample_size": 0,
            }

        protected = [d for d in recent if d.protected_group != "unprotected"]
        unprotected = [d for d in recent if d.protected_group == "unprotected"]

        protected_allows = sum(1 for d in protected if d.decision == "allow")
        unprotected_allows = sum(1 for d in unprotected if d.decision == "allow")

        protected_rate = protected_allows / float(len(protected)) if protected else 0.0
        unprotected_rate = unprotected_allows / float(len(unprotected)) if unprotected else 0.0

        difference = protected_rate - unprotected_rate

        abs_diff = abs(difference)
        if abs_diff <= 0.10:
            status = "healthy"
        elif abs_diff <= 0.20:
            status = "warning"
        else:
            status = "critical"

        return {
            "difference": difference,
            "protected_rate": protected_rate,
            "unprotected_rate": unprotected_rate,
            "status": status,
            "sample_size": len(recent),
            "protected_count": len(protected),
            "unprotected_count": len(unprotected),
            "threshold": 0.10,
        }

    def get_disparate_impact(
        self,
        attribute: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute Disparate Impact Ratio.

        Formula: P(decision=allow|protected) / P(decision=allow|unprotected)
        Threshold: 0.80 <= ratio <= 1.25 (healthy)

        Args:
            attribute: Specific attribute to analyse

        Returns:
            Disparate impact metrics dictionary
        """
        recent = self._get_recent_decisions()

        if not recent:
            return {
                "ratio": 1.0,
                "protected_rate": 0.0,
                "unprotected_rate": 0.0,
                "status": "insufficient_data",
                "sample_size": 0,
            }

        protected = [d for d in recent if d.protected_group != "unprotected"]
        unprotected = [d for d in recent if d.protected_group == "unprotected"]

        protected_allows = sum(1 for d in protected if d.decision == "allow")
        unprotected_allows = sum(1 for d in unprotected if d.decision == "allow")

        protected_rate = protected_allows / float(len(protected)) if protected else 0.0
        unprotected_rate = unprotected_allows / float(len(unprotected)) if unprotected else 1.0

        ratio = protected_rate / unprotected_rate if unprotected_rate > 0.0 else 0.0

        if 0.80 <= ratio <= 1.25:
            status = "healthy"
        elif 0.70 <= ratio <= 1.40:
            status = "warning"
        else:
            status = "critical"

        return {
            "ratio": ratio,
            "protected_rate": protected_rate,
            "unprotected_rate": unprotected_rate,
            "status": status,
            "sample_size": len(recent),
            "protected_count": len(protected),
            "unprotected_count": len(unprotected),
            "threshold_min": 0.80,
            "threshold_max": 1.25,
        }

    def get_equal_opportunity(
        self,
        attribute: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute Equal Opportunity metric.

        Formula: TPR(protected) - TPR(unprotected)
        Note: Uses approval rate as TPR proxy when ground truth labels are pending.

        Args:
            attribute: Specific attribute to analyse

        Returns:
            Equal opportunity metrics dictionary
        """
        recent = self._get_recent_decisions()

        if not recent:
            return {
                "difference": 0.0,
                "protected_tpr": 0.0,
                "unprotected_tpr": 0.0,
                "status": "insufficient_data",
                "sample_size": 0,
            }

        protected = [d for d in recent if d.protected_group != "unprotected"]
        unprotected = [d for d in recent if d.protected_group == "unprotected"]

        protected_allows = sum(1 for d in protected if d.decision == "allow")
        unprotected_allows = sum(1 for d in unprotected if d.decision == "allow")

        protected_tpr = protected_allows / float(len(protected)) if protected else 0.0
        unprotected_tpr = unprotected_allows / float(len(unprotected)) if unprotected else 0.0

        difference = protected_tpr - unprotected_tpr

        abs_diff = abs(difference)
        if abs_diff <= 0.10:
            status = "healthy"
        elif abs_diff <= 0.20:
            status = "warning"
        else:
            status = "critical"

        return {
            "difference": difference,
            "protected_tpr": protected_tpr,
            "unprotected_tpr": unprotected_tpr,
            "status": status,
            "sample_size": len(recent),
            "protected_count": len(protected),
            "unprotected_count": len(unprotected),
            "threshold": 0.10,
            "note": "Using approval rate as TPR proxy",
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all fairness metrics.

        Returns:
            Summary dictionary with overall status
        """
        sp = self.get_statistical_parity()
        di = self.get_disparate_impact()
        eo = self.get_equal_opportunity()

        statuses = [sp["status"], di["status"], eo["status"]]
        if "critical" in statuses:
            overall_status = "critical"
        elif "warning" in statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return {
            "overall_status": overall_status,
            "statistical_parity": sp,
            "disparate_impact": di,
            "equal_opportunity": eo,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "protected_attributes": self.protected_attributes,
        }

    def _get_recent_decisions(self) -> List[DecisionRecord]:
        """Get decisions within active rolling time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.window_hours)
        return [d for d in self._decisions if _ensure_utc(d.timestamp) > cutoff]

    def get_by_attribute(self, attribute: str) -> Dict[str, Any]:
        """
        Get fairness metrics for a specific protected attribute.

        Args:
            attribute: Protected attribute identifier

        Returns:
            Fairness metrics specific to the attribute
        """
        return {
            "attribute": attribute,
            "statistical_parity": self.get_statistical_parity(attribute),
            "disparate_impact": self.get_disparate_impact(attribute),
            "equal_opportunity": self.get_equal_opportunity(attribute),
        }
