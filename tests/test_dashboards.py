"""
Comprehensive Unit and Integration Tests for Sovereign Governance Dashboards

Tests cover:
- GovernanceDashboard (sync/async retrieval, caching, probes, exports)
- FairnessMetricsCollector (statistical parity, disparate impact, equal opportunity, bounding)
- PolicyLineageTracker (cryptographic chain validation, tamper detection, multi-sig compliance)
- AppealsMetricsCollector (bounded deques, latency percentiles, SLO compliance, outcome distributions)
- Integrity and validation of dashboard JSON templates
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import pytest

from dashboards import (
    GovernanceDashboard,
    DashboardMetrics,
    FairnessMetricsCollector,
    PolicyLineageTracker,
    AppealsMetricsCollector,
)
from probes import ProbeResult, ProbeStatus


class TestGovernanceDashboardComprehensive:
    """Rigorous tests for GovernanceDashboard core operations."""

    def test_initialisation_default_config(self) -> None:
        dashboard = GovernanceDashboard()
        assert dashboard.cache_ttl_seconds == 60
        assert "metrics" in dashboard.config
        assert "fairness" in dashboard.config["metrics"]

    def test_initialisation_custom_config(self, tmp_path: Path) -> None:
        custom_cfg = {
            "metrics": {
                "fairness": {
                    "protected_attributes": ["veteran_status", "clearance_level"]
                }
            },
            "slo_definitions": {
                "custom_slo": {"name": "Test SLO", "target": 99.9}
            }
        }
        cfg_file = tmp_path / "custom_governance.json"
        cfg_file.write_text(json.dumps(custom_cfg), encoding="utf-8")

        dashboard = GovernanceDashboard(config_path=str(cfg_file))
        assert dashboard.fairness_collector.protected_attributes == ["veteran_status", "clearance_level"]
        metrics = dashboard.get_metrics()
        assert "custom_slo" in metrics.slo_compliance
        assert metrics.slo_compliance["custom_slo"]["target"] == 99.9

    def test_initialisation_nonexistent_config_fallback(self) -> None:
        dashboard = GovernanceDashboard(config_path="nonexistent_config_file_path.json")
        assert "metrics" in dashboard.config

    @pytest.mark.asyncio
    async def test_get_metrics_async(self) -> None:
        dashboard = GovernanceDashboard()
        metrics = await dashboard.get_metrics_async(sections=["fairness", "slo_compliance"])
        assert isinstance(metrics, DashboardMetrics)
        assert metrics.fairness is not None
        assert metrics.slo_compliance["latency_slo_met"] is True

    def test_caching_and_ttl_expiration(self) -> None:
        dashboard = GovernanceDashboard(cache_ttl_seconds=1)
        m1 = dashboard.get_metrics(sections=["audit_log"], use_cache=True)
        assert "audit_log" in dashboard._cache

        # Immediate repeat should hit cache
        m2 = dashboard.get_metrics(sections=["audit_log"], use_cache=True)
        assert m1.audit_log == m2.audit_log

        # Simulate TTL expiration by manually modifying cache timestamp
        dashboard._cache_timestamps["audit_log"] = datetime.now(timezone.utc) - timedelta(seconds=2)
        m3 = dashboard.get_metrics(sections=["audit_log"], use_cache=True)
        assert m3.audit_log is not None

    def test_probe_results_update_and_cache_invalidation(self) -> None:
        dashboard = GovernanceDashboard()
        # Query invariant violations to populate cache
        dashboard.get_metrics(sections=["invariant_violations"])
        assert "invariant_violations" in dashboard._cache

        probe_res = ProbeResult(
            probe_name="quantum_entropy_probe",
            status=ProbeStatus.WARNING,
            timestamp=datetime.now(timezone.utc),
            message="Entropy degradation detected",
            violations=["entropy_sub_threshold"],
        )
        dashboard.update_probe_result("quantum_entropy_probe", probe_res)
        assert "invariant_violations" not in dashboard._cache_timestamps

        metrics = dashboard.get_metrics(sections=["invariant_violations"])
        violations = metrics.invariant_violations
        assert "quantum_entropy_probe" in violations
        assert violations["quantum_entropy_probe"]["count"] == 1
        assert violations["quantum_entropy_probe"]["status"] == "warning"

    def test_export_formats_and_error_handling(self) -> None:
        dashboard = GovernanceDashboard()

        # JSON export
        json_out = dashboard.export_metrics(format="json")
        parsed = json.loads(json_out)
        assert "timestamp" in parsed
        assert "fairness" in parsed

        # CSV export
        csv_out = dashboard.export_metrics(format="csv")
        assert csv_out.startswith("Section,Metric,Value,Timestamp")
        assert "fairness" in csv_out

        # PDF export raises NotImplementedError
        with pytest.raises(NotImplementedError, match="PDF export is not yet implemented"):
            dashboard.export_metrics(format="pdf")

        # Unsupported format raises ValueError
        with pytest.raises(ValueError, match="Unsupported export format"):
            dashboard.export_metrics(format="xml")

    def test_compute_error_resilience(self) -> None:
        dashboard = GovernanceDashboard()

        def faulty_compute():
            raise RuntimeError("Database connection timeout")

        res = dashboard._get_cached_or_compute("faulty_section", faulty_compute, use_cache=False)
        assert "error" in res
        assert "Database connection timeout" in res["error"]

    def test_to_dict_serialisation(self) -> None:
        ts = datetime.now(timezone.utc)
        metrics = DashboardMetrics(timestamp=ts)
        d = metrics.to_dict()
        assert d["timestamp"] == ts.isoformat()
        assert isinstance(d["fairness"], dict)


class TestFairnessMetricsCollectorComprehensive:
    """Exhaustive tests for fairness metrics and edge cases."""

    def test_empty_collector_metrics(self) -> None:
        collector = FairnessMetricsCollector(protected_attributes=["age", "gender"])
        sp = collector.get_statistical_parity()
        di = collector.get_disparate_impact()
        eo = collector.get_equal_opportunity()
        summary = collector.get_summary()

        assert sp["status"] == "insufficient_data"
        assert di["status"] == "insufficient_data"
        assert eo["status"] == "insufficient_data"
        assert summary["overall_status"] == "healthy"

    def test_bounded_deque_capacity(self) -> None:
        collector = FairnessMetricsCollector(max_decisions=5)
        for i in range(10):
            collector.record_decision("allow", protected_group=f"group_{i}")

        assert len(collector._decisions) == 5
        # The oldest items (group_0 .. group_4) were automatically dropped
        assert collector._decisions[0].protected_group == "group_5"
        assert collector._decisions[-1].protected_group == "group_9"

    def test_disparate_impact_zero_unprotected_guard(self) -> None:
        collector = FairnessMetricsCollector()
        # Record only protected decisions
        collector.record_decision("allow", protected_group="protected_A")
        di = collector.get_disparate_impact()
        assert di["protected_rate"] == 1.0
        assert di["unprotected_rate"] == 1.0  # default guard when no unprotected
        assert di["ratio"] == 1.0

    def test_time_window_filtering(self) -> None:
        collector = FairnessMetricsCollector(window_hours=2)
        old_time = datetime.now(timezone.utc) - timedelta(hours=3)
        recent_time = datetime.now(timezone.utc) - timedelta(minutes=30)

        collector.record_decision("allow", protected_group="female", timestamp=old_time)
        collector.record_decision("deny", protected_group="female", timestamp=recent_time)

        recent = collector._get_recent_decisions()
        assert len(recent) == 1
        assert recent[0].decision == "deny"

    def test_naive_datetime_handling(self) -> None:
        collector = FairnessMetricsCollector()
        naive_dt = datetime(2026, 1, 1, 12, 0, 0)
        # Should not raise TypeError: can't subtract offset-naive and offset-aware datetimes
        collector.record_decision("allow", protected_group="female", timestamp=naive_dt)
        recent = collector._get_recent_decisions()
        # Old timestamp correctly excluded by rolling window
        assert len(recent) == 0

    def test_get_by_attribute_interface(self) -> None:
        collector = FairnessMetricsCollector(protected_attributes=["age"])
        collector.record_decision("allow", protected_group="age_under_25")
        attr_metrics = collector.get_by_attribute("age")
        assert attr_metrics["attribute"] == "age"
        assert "statistical_parity" in attr_metrics
        assert "disparate_impact" in attr_metrics


class TestPolicyLineageTrackerComprehensive:
    """Tests for policy history, SHA-256 chain verification, and multi-sig."""

    def test_single_version_chain(self) -> None:
        tracker = PolicyLineageTracker()
        tracker.record_policy_version(
            policy_id="pol_root",
            version=1,
            content="DEFENSE_POLICY_ALPHA",
            parent_hash=None,
            signatures=[{"signer_id": "auditor_1", "signature": "sig_a"}],
            author="secops_lead",
        )
        integrity = tracker.get_chain_integrity()
        assert integrity["status"] == "healthy"
        assert integrity["integrity_rate"] == 1.0

    def test_multi_step_valid_chain(self) -> None:
        tracker = PolicyLineageTracker()
        tracker.record_policy_version("pol_dual", 1, "v1_spec", None, [{"signer_id": "u1", "signature": "s1"}], "auth1")
        h1 = tracker._policies["pol_dual"][0].content_hash

        tracker.record_policy_version("pol_dual", 2, "v2_spec", h1, [{"signer_id": "u1", "signature": "s1"}], "auth1")
        h2 = tracker._policies["pol_dual"][1].content_hash

        tracker.record_policy_version("pol_dual", 3, "v3_spec", h2, [{"signer_id": "u1", "signature": "s1"}], "auth1")

        integrity = tracker.get_chain_integrity()
        assert integrity["verified_chains"] == 1
        assert integrity["broken_chains"] == 0
        assert integrity["status"] == "healthy"

    def test_tampered_parent_hash_broken_chain(self) -> None:
        tracker = PolicyLineageTracker()
        tracker.record_policy_version("pol_tamper", 1, "v1", None, [], "auth")
        # Provide forged parent hash
        tracker.record_policy_version("pol_tamper", 2, "v2", "forged_parent_hash_xyz", [], "auth")

        integrity = tracker.get_chain_integrity()
        assert integrity["verified_chains"] == 0
        assert integrity["broken_chains"] == 1
        assert integrity["status"] == "critical"
        assert integrity["integrity_rate"] == 0.0

    def test_multi_sig_threshold_metrics(self) -> None:
        tracker = PolicyLineageTracker()
        # Single signature
        tracker.record_policy_version("pol_sec", 1, "content", None, [{"signer_id": "u1", "signature": "s1"}], "auth")
        m_2sig = tracker.get_multi_sig_metrics(min_required=2)
        assert m_2sig["status"] == "warning"
        assert m_2sig["compliance_rate"] == 0.0

        m_1sig = tracker.get_multi_sig_metrics(min_required=1)
        assert m_1sig["status"] == "healthy"
        assert m_1sig["compliance_rate"] == 1.0


class TestAppealsMetricsCollectorComprehensive:
    """Tests for appeals throughput, SLA calculation, and percentiles."""

    def test_empty_collector_defaults(self) -> None:
        collector = AppealsMetricsCollector()
        vol = collector.get_volume_metrics()
        res = collector.get_resolution_metrics()
        dist = collector.get_outcome_distribution()

        assert vol["total_appeals"] == 0
        assert vol["appeals_per_day"] == 0.0
        assert res["sample_size"] == 0
        assert res["slo_compliance_rate"] == 1.0
        assert dist["total"] == 0

    def test_bounded_deque_capacity(self) -> None:
        collector = AppealsMetricsCollector(max_appeals=3)
        for i in range(5):
            collector.record_appeal(f"appeal_{i}", f"dec_{i}")

        assert len(collector._appeals) == 3
        assert collector._appeals[0].appeal_id == "appeal_2"
        assert collector._appeals[-1].appeal_id == "appeal_4"

    def test_slo_compliance_and_percentiles(self) -> None:
        collector = AppealsMetricsCollector()
        now = datetime.now(timezone.utc)

        # 8 fast appeals (10 hours resolution)
        for i in range(8):
            collector.record_appeal(f"fast_{i}", f"d_{i}", filed_at=now - timedelta(hours=10))
            collector.resolve_appeal(f"fast_{i}", "upheld", resolved_at=now)

        # 2 slow appeals (100 hours resolution > 72h SLO)
        for i in range(2):
            collector.record_appeal(f"slow_{i}", f"d_slow_{i}", filed_at=now - timedelta(hours=100))
            collector.resolve_appeal(f"slow_{i}", "overturned", resolved_at=now)

        res = collector.get_resolution_metrics()
        assert res["sample_size"] == 10
        assert res["median_hours"] == 10.0
        assert res["slo_compliance_rate"] == 0.80  # 8 out of 10
        assert res["status"] == "healthy"  # median is 10 <= 72h SLO target

    def test_outcome_distribution_multi_category(self) -> None:
        collector = AppealsMetricsCollector()
        collector.record_appeal("a1", "d1")
        collector.resolve_appeal("a1", "upheld")
        collector.record_appeal("a2", "d2")
        collector.resolve_appeal("a2", "modified")
        collector.record_appeal("a3", "d3")
        collector.resolve_appeal("a3", "withdrawn")

        dist = collector.get_outcome_distribution()
        assert dist["total"] == 3
        assert "upheld" in dist["distribution"]
        assert "modified" in dist["distribution"]
        assert "withdrawn" in dist["distribution"]
        assert pytest.approx(dist["distribution"]["upheld"]["percentage"], 0.1) == 33.33


class TestDashboardJsonIntegrity:
    """Verifies that all dashboard JSON templates in dashboards/ are valid and well-formed."""

    @pytest.mark.parametrize("filename", [
        "governance.json",
        "detection_intelligence.json",
        "latency_slo.json",
        "security.json",
    ])
    def test_json_template_schema_validity(self, filename: str) -> None:
        path = Path("dashboards") / filename
        assert path.exists(), f"Missing dashboard JSON: {filename}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict), f"JSON root must be an object: {filename}"
        assert len(data) > 0, f"Dashboard configuration cannot be empty: {filename}"
