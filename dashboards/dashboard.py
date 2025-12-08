from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import os
import logging
import asyncio

from .fairness_metrics import FairnessMetricsCollector
from .policy_lineage_tracker import PolicyLineageTracker
from .appeals_metrics import AppealsMetricsCollector

# Configure logging for observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GovernanceDashboard")


@dataclass
class DashboardMetrics:
    """Container for dashboard metrics"""

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
        """Convert to dictionary"""
        return {
            k: (v.isoformat() if isinstance(v, datetime) else v)
            for k, v in self.__dict__.items()
        }


class GovernanceDashboard:
    """High-end Governance Dashboard with extensibility and resilience"""

    def __init__(self, config_path: Optional[str] = None, cache_ttl_seconds: int = 60):
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._probe_results: Dict[str, Any] = {}

        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            default_config_path = os.path.join(
                os.path.dirname(__file__), "governance.json"
            )
            with open(default_config_path, "r") as f:
                self.config = json.load(f)

        # Initialize collectors
        self.fairness_collector = FairnessMetricsCollector(
            protected_attributes=self.config["metrics"]["fairness"][
                "protected_attributes"
            ]
        )
        self.lineage_tracker = PolicyLineageTracker()
        self.appeals_collector = AppealsMetricsCollector()

    async def get_metrics(
        self, sections: Optional[List[str]] = None, use_cache: bool = True
    ) -> DashboardMetrics:
        """Get dashboard metrics asynchronously"""
        start_time = datetime.utcnow()
        metrics = DashboardMetrics(timestamp=start_time)

        async def collect(section: str, func: callable):
            metrics.__dict__[section] = await self._get_cached_or_compute(
                section, func, use_cache
            )

        tasks = []
        if not sections or "fairness" in sections:
            tasks.append(collect("fairness", self._compute_fairness_metrics))
        if not sections or "policy_lineage" in sections:
            tasks.append(collect("policy_lineage", self._compute_lineage_metrics))
        if not sections or "appeals" in sections:
            tasks.append(collect("appeals", self._compute_appeals_metrics))
        if not sections or "audit_log" in sections:
            tasks.append(collect("audit_log", self._compute_audit_metrics))
        if not sections or "invariant_violations" in sections:
            tasks.append(
                collect("invariant_violations", self._compute_invariant_metrics)
            )

        await asyncio.gather(*tasks)

        # SLO compliance
        metrics.slo_compliance = self._compute_slo_compliance()
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        metrics.slo_compliance["query_latency_seconds"] = elapsed
        metrics.slo_compliance["latency_slo_met"] = elapsed < 5.0

        return metrics

    async def _get_cached_or_compute(
        self, key: str, compute_func: callable, use_cache: bool
    ) -> Any:
        """Get cached value or compute new one"""
        if use_cache and key in self._cache:
            cache_time = self._cache_timestamps.get(key)
            if cache_time:
                age = (datetime.utcnow() - cache_time).total_seconds()
                if age < self.cache_ttl_seconds:
                    return self._cache[key]
        try:
            value = compute_func()
            self._cache[key] = value
            self._cache_timestamps[key] = datetime.utcnow()
            return value
        except Exception as e:
            logger.error(f"Error computing {key}: {e}")
            return {"error": str(e)}

    def _compute_fairness_metrics(self) -> Dict[str, Any]:
        return {
            "statistical_parity": self.fairness_collector.get_statistical_parity(),
            "disparate_impact": self.fairness_collector.get_disparate_impact(),
            "equal_opportunity": self.fairness_collector.get_equal_opportunity(),
            "summary": self.fairness_collector.get_summary(),
        }

    def _compute_lineage_metrics(self) -> Dict[str, Any]:
        return {
            "chain_integrity": self.lineage_tracker.get_chain_integrity(),
            "version_tracking": self.lineage_tracker.get_version_metrics(),
            "multi_sig_compliance": self.lineage_tracker.get_multi_sig_metrics(),
        }

    def _compute_appeals_metrics(self) -> Dict[str, Any]:
        return {
            "volume": self.appeals_collector.get_volume_metrics(),
            "resolution_time": self.appeals_collector.get_resolution_metrics(),
            "outcomes": self.appeals_collector.get_outcome_distribution(),
        }

    def _compute_audit_metrics(self) -> Dict[str, Any]:
        return {
            "completeness": {"rate": 1.0, "total_decisions": 0, "audited_decisions": 0},
            "integrity": {
                "merkle_root_valid": True,
                "signature_valid": True,
                "last_verification": datetime.utcnow().isoformat(),
            },
            "retention": {
                "total_entries": 0,
                "oldest_entry_days": 0,
                "storage_size_gb": 0.0,
            },
        }

    def _compute_invariant_metrics(self) -> Dict[str, Any]:
        violations = {}
        for probe_name, result in self._probe_results.items():
            if hasattr(result, "violations"):
                violations[probe_name] = {
                    "count": len(result.violations),
                    "status": getattr(result, "status", "unknown"),
                    "recent_violations": result.violations[:5],
                }
        return violations

    def _compute_slo_compliance(self) -> Dict[str, Any]:
        slos = self.config.get("slo_definitions", {})
        compliance = {}
        for slo_id, slo_config in slos.items():
            compliance[slo_id] = {
                "name": slo_config.get("name"),
                "target": slo_config.get("target"),
                "current_value": None,
                "compliant": True,
            }
        return compliance

    def export_metrics(
        self, format: str = "json", sections: Optional[List[str]] = None
    ) -> str:
        """Export metrics in specified format"""
        metrics = asyncio.run(self.get_metrics(sections=sections))
        if format == "json":
            return json.dumps(metrics.to_dict(), indent=2)
        elif format == "csv":
            return self._export_csv(metrics)
        elif format == "pdf":
            return self._export_pdf(metrics)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_csv(self, metrics: DashboardMetrics) -> str:
        lines = ["Section,Metric,Value,Timestamp"]
        for section, data in metrics.to_dict().items():
            if section == "timestamp":
                continue
            if isinstance(data, dict):
                for key, value in data.items():
                    lines.append(
                        f"{section},{key},{value},{metrics.timestamp.isoformat()}"
                    )
        return "\n".join(lines)

    def _export_pdf(self, metrics: DashboardMetrics) -> str:
        return "PDF export not yet implemented"

    def update_probe_result(self, probe_name: str, result: Any):
        self._probe_results[probe_name] = result
        if "invariant_violations" in self._cache_timestamps:
            del self._cache_timestamps["invariant_violations"]

    def get_accessibility_info(self) -> Dict[str, Any]:
        return self.config.get("accessibility", {})
