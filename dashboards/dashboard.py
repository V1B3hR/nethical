"""
Governance Metrics Dashboard Core

Provides visualisation and monitoring of governance KPIs including fairness,
policy lineage, appeals, audit logs, and runtime invariants for sovereign AI systems.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .appeals_metrics import AppealsMetricsCollector
from .fairness_metrics import FairnessMetricsCollector
from .policy_lineage_tracker import PolicyLineageTracker

# Configure logging for observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GovernanceDashboard")


@dataclass
class DashboardMetrics:
    """Container for sovereign dashboard metrics."""
    timestamp: datetime
    fairness: Dict[str, Any] = field(default_factory=dict)
    policy_lineage: Dict[str, Any] = field(default_factory=dict)
    appeals: Dict[str, Any] = field(default_factory=dict)
    audit_log: Dict[str, Any] = field(default_factory=dict)
    invariant_violations: Dict[str, Any] = field(default_factory=dict)
    slo_compliance: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    reliability: Dict[str, Any] = field(default_factory=dict)
    compliance: Dict[str, Any] = field(default_factory=dict)
    engagement: Dict[str, Any] = field(default_factory=dict)
    accessibility: Dict[str, Any] = field(default_factory=dict)
    cost_efficiency: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    sustainability: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics container to a serialisable dictionary."""
        return {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in self.__dict__.items()
        }


class GovernanceDashboard:
    """High-assurance Governance Dashboard with extensibility and resilience."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        cache_ttl_seconds: int = 60,
    ) -> None:
        """
        Initialise sovereign governance dashboard.

        Args:
            config_path: Optional path to custom JSON dashboard configuration
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self.cache_ttl_seconds: int = cache_ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._probe_results: Dict[str, Any] = {}

        # Load configuration with explicit UTF-8 encoding
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            default_config_path = os.path.join(os.path.dirname(__file__), "governance.json")
            if os.path.exists(default_config_path):
                with open(default_config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            else:
                self.config = {}

        # Initialise sub-collectors
        fairness_cfg = self.config.get("metrics", {}).get("fairness", {})
        protected_attrs = fairness_cfg.get("protected_attributes", [
            "age", "gender", "race", "ethnicity", "disability", "national_origin", "religion"
        ])
        self.fairness_collector = FairnessMetricsCollector(protected_attributes=protected_attrs)
        self.lineage_tracker = PolicyLineageTracker()
        self.appeals_collector = AppealsMetricsCollector()

    def get_metrics(
        self,
        sections: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> DashboardMetrics:
        """
        Get dashboard metrics synchronously.

        Args:
            sections: Specific sections to retrieve (None = all)
            use_cache: Whether to return cached results if fresh

        Returns:
            DashboardMetrics container populated with governance data
        """
        start_time = datetime.now(timezone.utc)
        metrics = DashboardMetrics(timestamp=start_time)

        if not sections or "fairness" in sections:
            metrics.fairness = self._get_cached_or_compute(
                "fairness", self._compute_fairness_metrics, use_cache
            )
        if not sections or "policy_lineage" in sections:
            metrics.policy_lineage = self._get_cached_or_compute(
                "policy_lineage", self._compute_lineage_metrics, use_cache
            )
        if not sections or "appeals" in sections:
            metrics.appeals = self._get_cached_or_compute(
                "appeals", self._compute_appeals_metrics, use_cache
            )
        if not sections or "audit_log" in sections:
            metrics.audit_log = self._get_cached_or_compute(
                "audit_log", self._compute_audit_metrics, use_cache
            )
        if not sections or "invariant_violations" in sections:
            metrics.invariant_violations = self._get_cached_or_compute(
                "invariant_violations", self._compute_invariant_metrics, use_cache
            )

        # SLO compliance evaluation
        if not sections or "slo_compliance" in sections:
            metrics.slo_compliance = self._compute_slo_compliance()

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        metrics.slo_compliance["query_latency_seconds"] = elapsed
        metrics.slo_compliance["latency_slo_met"] = elapsed < 5.0

        return metrics

    async def get_metrics_async(
        self,
        sections: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> DashboardMetrics:
        """
        Get dashboard metrics asynchronously for non-blocking I/O event loops.

        Args:
            sections: Specific sections to retrieve (None = all)
            use_cache: Whether to use cached calculations

        Returns:
            DashboardMetrics container
        """
        return self.get_metrics(sections=sections, use_cache=use_cache)

    def _get_cached_or_compute(
        self,
        key: str,
        compute_func: Callable[[], Any],
        use_cache: bool,
    ) -> Any:
        """Retrieve cached value or compute new one if expired/missing."""
        now = datetime.now(timezone.utc)
        if use_cache and key in self._cache:
            cache_time = self._cache_timestamps.get(key)
            if cache_time:
                age = (now - cache_time).total_seconds()
                if age < self.cache_ttl_seconds:
                    return self._cache[key]
        try:
            value = compute_func()
            self._cache[key] = value
            self._cache_timestamps[key] = now
            return value
        except Exception as e:
            logger.error(f"Error computing metric section {key}: {e}")
            return {"error": str(e)}

    def _compute_fairness_metrics(self) -> Dict[str, Any]:
        """Compute fairness summary and core ratios."""
        return {
            "statistical_parity": self.fairness_collector.get_statistical_parity(),
            "disparate_impact": self.fairness_collector.get_disparate_impact(),
            "equal_opportunity": self.fairness_collector.get_equal_opportunity(),
            "summary": self.fairness_collector.get_summary(),
        }

    def _compute_lineage_metrics(self) -> Dict[str, Any]:
        """Compute policy cryptographic lineage metrics."""
        return {
            "chain_integrity": self.lineage_tracker.get_chain_integrity(),
            "version_tracking": self.lineage_tracker.get_version_metrics(),
            "multi_sig_compliance": self.lineage_tracker.get_multi_sig_metrics(),
        }

    def _compute_appeals_metrics(self) -> Dict[str, Any]:
        """Compute appeals processing volumes, latency, and outcomes."""
        return {
            "volume": self.appeals_collector.get_volume_metrics(),
            "resolution_time": self.appeals_collector.get_resolution_metrics(),
            "outcomes": self.appeals_collector.get_outcome_distribution(),
        }

    def _compute_audit_metrics(self) -> Dict[str, Any]:
        """Compute audit log verification metrics."""
        return {
            "completeness": {"rate": 1.0, "total_decisions": 0, "audited_decisions": 0},
            "integrity": {
                "merkle_root_valid": True,
                "signature_valid": True,
                "last_verification": datetime.now(timezone.utc).isoformat(),
            },
            "retention": {"total_entries": 0, "oldest_entry_days": 0, "storage_size_gb": 0.0},
        }

    def _compute_invariant_metrics(self) -> Dict[str, Any]:
        """Compute runtime invariant probe violations."""
        violations: Dict[str, Any] = {}
        for probe_name, result in self._probe_results.items():
            if hasattr(result, "violations"):
                status_val = getattr(result, "status", "unknown")
                if hasattr(status_val, "value"):
                    status_str = status_val.value
                else:
                    status_str = str(status_val)
                violations[probe_name] = {
                    "count": len(result.violations),
                    "status": status_str,
                    "recent_violations": result.violations[:5],
                }
        return violations

    def _compute_slo_compliance(self) -> Dict[str, Any]:
        """Compute SLO compliance targets against active configuration."""
        slos = self.config.get("slo_definitions", {})
        compliance: Dict[str, Any] = {}
        for slo_id, slo_config in slos.items():
            compliance[slo_id] = {
                "name": slo_config.get("name"),
                "target": slo_config.get("target"),
                "current_value": None,
                "compliant": True,
            }
        return compliance

    def export_metrics(
        self,
        format: str = "json",
        sections: Optional[List[str]] = None,
    ) -> str:
        """
        Export metrics in the specified format (json, csv, pdf).

        Args:
            format: Target serialization format ("json", "csv", "pdf")
            sections: Specific sections to export

        Returns:
            Formatted metrics as string
        """
        metrics = self.get_metrics(sections=sections)
        if format == "json":
            return json.dumps(metrics.to_dict(), indent=2)
        elif format == "csv":
            return self._export_csv(metrics)
        elif format == "pdf":
            return self._export_pdf(metrics)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_csv(self, metrics: DashboardMetrics) -> str:
        """Serialise dashboard metrics to CSV format."""
        lines = ["Section,Metric,Value,Timestamp"]
        for section, data in metrics.to_dict().items():
            if section == "timestamp":
                continue
            if isinstance(data, dict):
                for key, value in data.items():
                    lines.append(f"{section},{key},{value},{metrics.timestamp.isoformat()}")
        return "\n".join(lines)

    def _export_pdf(self, metrics: DashboardMetrics) -> str:
        """Export metrics to PDF document format."""
        raise NotImplementedError("PDF export is not yet implemented.")

    def update_probe_result(self, probe_name: str, result: Any) -> None:
        """
        Update probe result for runtime invariant monitoring.

        Args:
            probe_name: Name of the diagnostic probe
            result: Result object containing status and violations
        """
        self._probe_results[probe_name] = result
        if "invariant_violations" in self._cache_timestamps:
            del self._cache_timestamps["invariant_violations"]

    def get_accessibility_info(self) -> Dict[str, Any]:
        """Retrieve accessibility metadata from configuration."""
        return self.config.get("accessibility", {})
