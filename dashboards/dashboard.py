"""
Main Governance Dashboard Implementation

Aggregates metrics from all subsystems and provides unified API for
dashboard queries with <5s latency requirement.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
import os

from .fairness_metrics import FairnessMetricsCollector
from .policy_lineage_tracker import PolicyLineageTracker
from .appeals_metrics import AppealsMetricsCollector


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

    # 🔑 New important categories
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
            "timestamp": self.timestamp.isoformat(),
            "fairness": self.fairness,
            "policy_lineage": self.policy_lineage,
            "appeals": self.appeals,
            "audit_log": self.audit_log,
            "invariant_violations": self.invariant_violations,
            "slo_compliance": self.slo_compliance,
            "security": self.security,
            "performance": self.performance,
            "reliability": self.reliability,
            "compliance": self.compliance,
            "engagement": self.engagement,
            "accessibility": self.accessibility,
            "cost_efficiency": self.cost_efficiency,
            "risk": self.risk,
            "sustainability": self.sustainability,
        }



class GovernanceDashboard:
    """
    Main Governance Dashboard
    
    Provides unified interface for querying governance metrics with
    <5s latency SLO compliance.
    
    Features:
    - Real-time metrics aggregation
    - Caching for performance
    - WCAG 2.1 AA accessibility
    - Multiple export formats
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        cache_ttl_seconds: int = 60,
    ):
        """
        Initialize governance dashboard.
        
        Args:
            config_path: Path to governance.json configuration
            cache_ttl_seconds: Cache TTL for metrics
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Load default config
            default_config_path = os.path.join(
                os.path.dirname(__file__),
                "governance.json"
            )
            with open(default_config_path, 'r') as f:
                self.config = json.load(f)
        
        # Initialize metrics collectors
        self.fairness_collector = FairnessMetricsCollector(
            protected_attributes=self.config["metrics"]["fairness"]["protected_attributes"]
        )
        self.lineage_tracker = PolicyLineageTracker()
        self.appeals_collector = AppealsMetricsCollector()
        
        self._probe_results: Dict[str, Any] = {}
    
    def get_metrics(
        self,
        sections: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> DashboardMetrics:
        """
        Get dashboard metrics.
        
        Args:
            sections: Specific sections to retrieve (None = all)
            use_cache: Whether to use cached data
        
        Returns:
            DashboardMetrics containing requested data
        """
        start_time = datetime.utcnow()
        
        metrics = DashboardMetrics(timestamp=start_time)
        
        # Get fairness metrics
        if not sections or "fairness" in sections:
            metrics.fairness = self._get_cached_or_compute(
                "fairness",
                self._compute_fairness_metrics,
                use_cache
            )
        
        # Get policy lineage metrics
        if not sections or "policy_lineage" in sections:
            metrics.policy_lineage = self._get_cached_or_compute(
                "policy_lineage",
                self._compute_lineage_metrics,
                use_cache
            )
        
        # Get appeals metrics
        if not sections or "appeals" in sections:
            metrics.appeals = self._get_cached_or_compute(
                "appeals",
                self._compute_appeals_metrics,
                use_cache
            )
        
        # Get audit log metrics
        if not sections or "audit_log" in sections:
            metrics.audit_log = self._get_cached_or_compute(
                "audit_log",
                self._compute_audit_metrics,
                use_cache
            )
        
        # Get invariant violation metrics
        if not sections or "invariant_violations" in sections:
            metrics.invariant_violations = self._get_cached_or_compute(
                "invariant_violations",
                self._compute_invariant_metrics,
                use_cache
            )
        
        # Get SLO compliance
        if not sections or "slo_compliance" in sections:
            metrics.slo_compliance = self._compute_slo_compliance()
        
        # Ensure <5s latency SLO
        elapsed = (datetime.utcnow() - start_time).total_seconds()
        metrics.slo_compliance["query_latency_seconds"] = elapsed
        metrics.slo_compliance["latency_slo_met"] = elapsed < 5.0
        
        return metrics
    
    def update_probe_result(self, probe_name: str, result: Any):
        """Update probe result for dashboard display"""
        self._probe_results[probe_name] = result
        # Invalidate cache for invariant violations
        if "invariant_violations" in self._cache_timestamps:
            del self._cache_timestamps["invariant_violations"]
    
    def _get_cached_or_compute(
        self,
        key: str,
        compute_func: callable,
        use_cache: bool,
    ) -> Any:
        """Get cached value or compute new one"""
        if use_cache and key in self._cache:
            cache_time = self._cache_timestamps.get(key)
            if cache_time:
                age = (datetime.utcnow() - cache_time).total_seconds()
                if age < self.cache_ttl_seconds:
                    return self._cache[key]
        
        # Compute new value
        value = compute_func()
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.utcnow()
        return value
    
    def _compute_fairness_metrics(self) -> Dict[str, Any]:
        """Compute fairness metrics"""
        return {
            "statistical_parity": self.fairness_collector.get_statistical_parity(),
            "disparate_impact": self.fairness_collector.get_disparate_impact(),
            "equal_opportunity": self.fairness_collector.get_equal_opportunity(),
            "summary": self.fairness_collector.get_summary(),
        }
    
    def _compute_lineage_metrics(self) -> Dict[str, Any]:
        """Compute policy lineage metrics"""
        return {
            "chain_integrity": self.lineage_tracker.get_chain_integrity(),
            "version_tracking": self.lineage_tracker.get_version_metrics(),
            "multi_sig_compliance": self.lineage_tracker.get_multi_sig_metrics(),
        }
    
    def _compute_appeals_metrics(self) -> Dict[str, Any]:
        """Compute appeals metrics"""
        return {
            "volume": self.appeals_collector.get_volume_metrics(),
            "resolution_time": self.appeals_collector.get_resolution_metrics(),
            "outcomes": self.appeals_collector.get_outcome_distribution(),
        }
    
    def _compute_audit_metrics(self) -> Dict[str, Any]:
        """Compute audit log metrics"""
        # Placeholder - would integrate with actual audit service
        return {
            "completeness": {
                "rate": 1.0,
                "total_decisions": 0,
                "audited_decisions": 0,
            },
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
        """Compute invariant violation metrics"""
        violations = {}
        
        for probe_name, result in self._probe_results.items():
            if hasattr(result, 'violations'):
                violations[probe_name] = {
                    "count": len(result.violations),
                    "status": result.status.value if hasattr(result, 'status') else "unknown",
                    "recent_violations": result.violations[:5],
                }
        
        return violations
    
    def _compute_slo_compliance(self) -> Dict[str, Any]:
        """Compute SLO compliance metrics"""
        slos = self.config.get("slo_definitions", {})
        
        compliance = {}
        for slo_id, slo_config in slos.items():
            # Placeholder - would compute actual compliance
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
        Export metrics in specified format.
        
        Args:
            format: Export format (json, csv, pdf)
            sections: Sections to export
        
        Returns:
            Exported data as string
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
        """Export metrics as CSV"""
        lines = ["Section,Metric,Value,Timestamp"]
        
        for section, data in metrics.to_dict().items():
            if section == "timestamp":
                continue
            
            if isinstance(data, dict):
                for key, value in data.items():
                    lines.append(f"{section},{key},{value},{metrics.timestamp.isoformat()}")
        
        return "\n".join(lines)
    
    def _export_pdf(self, metrics: DashboardMetrics) -> str:
        """Export metrics as PDF (placeholder)"""
        # Would use a PDF library like reportlab
        return "PDF export not yet implemented"
    
    def get_accessibility_info(self) -> Dict[str, Any]:
        """Get accessibility information"""
        return self.config.get("accessibility", {})
