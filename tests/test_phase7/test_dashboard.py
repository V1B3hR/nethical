"""
Tests for Phase 7 Governance Dashboard

Tests dashboard metrics, fairness tracking, policy lineage, and appeals.
"""

import pytest
import json
from datetime import datetime, timedelta
from dashboards import (
    GovernanceDashboard,
    DashboardMetrics,
    FairnessMetricsCollector,
    PolicyLineageTracker,
    AppealsMetricsCollector,
)


class TestFairnessMetricsCollector:
    """Test fairness metrics collector"""
    
    def test_creation(self):
        """Test creating fairness collector"""
        collector = FairnessMetricsCollector(
            protected_attributes=["age", "gender"],
            window_hours=24
        )
        assert len(collector.protected_attributes) == 2
        assert collector.window_hours == 24
    
    def test_record_decision(self):
        """Test recording decisions"""
        collector = FairnessMetricsCollector(protected_attributes=["gender"])
        
        collector.record_decision("allow", "female")
        collector.record_decision("allow", "male")
        collector.record_decision("deny", "female")
        
        assert len(collector._decisions) == 3
    
    def test_statistical_parity_balanced(self):
        """Test statistical parity with balanced outcomes"""
        collector = FairnessMetricsCollector(protected_attributes=["gender"])
        
        # Balanced decisions
        for _ in range(50):
            collector.record_decision("allow", "female")
            collector.record_decision("allow", None)
        
        for _ in range(50):
            collector.record_decision("deny", "female")
            collector.record_decision("deny", None)
        
        sp = collector.get_statistical_parity()
        assert sp['status'] == "healthy"
        assert abs(sp['difference']) <= 0.10
    
    def test_statistical_parity_biased(self):
        """Test statistical parity with biased outcomes"""
        collector = FairnessMetricsCollector(protected_attributes=["gender"])
        
        # Biased decisions
        for _ in range(80):
            collector.record_decision("allow", None)  # Unprotected
        
        for _ in range(20):
            collector.record_decision("allow", "female")  # Protected
        
        for _ in range(20):
            collector.record_decision("deny", None)
        
        for _ in range(80):
            collector.record_decision("deny", "female")
        
        sp = collector.get_statistical_parity()
        assert sp['status'] in ["warning", "critical"]
        assert abs(sp['difference']) > 0.10
    
    def test_disparate_impact_balanced(self):
        """Test disparate impact with balanced outcomes"""
        collector = FairnessMetricsCollector(protected_attributes=["gender"])
        
        # Balanced approvals
        for _ in range(80):
            collector.record_decision("allow", "female")
            collector.record_decision("allow", None)
        
        for _ in range(20):
            collector.record_decision("deny", "female")
            collector.record_decision("deny", None)
        
        di = collector.get_disparate_impact()
        assert di['status'] == "healthy"
        assert 0.80 <= di['ratio'] <= 1.25
    
    def test_get_summary(self):
        """Test getting fairness summary"""
        collector = FairnessMetricsCollector(protected_attributes=["gender"])
        
        # Add some data
        for _ in range(50):
            collector.record_decision("allow", "female")
            collector.record_decision("allow", None)
        
        summary = collector.get_summary()
        assert 'overall_status' in summary
        assert 'statistical_parity' in summary
        assert 'disparate_impact' in summary
        assert 'equal_opportunity' in summary


class TestPolicyLineageTracker:
    """Test policy lineage tracker"""
    
    def test_creation(self):
        """Test creating lineage tracker"""
        tracker = PolicyLineageTracker()
        assert len(tracker._policies) == 0
    
    def test_record_policy_version(self):
        """Test recording policy version"""
        tracker = PolicyLineageTracker()
        
        tracker.record_policy_version(
            policy_id="pol_001",
            version=1,
            content="policy v1 content",
            parent_hash=None,
            signatures=[{"signer_id": "user1", "signature": "sig1"}],
            author="admin"
        )
        
        assert "pol_001" in tracker._policies
        assert len(tracker._policies["pol_001"]) == 1
    
    def test_get_chain_integrity_valid(self):
        """Test chain integrity with valid chain"""
        tracker = PolicyLineageTracker()
        
        # Create valid hash chain
        tracker.record_policy_version(
            policy_id="pol_001",
            version=1,
            content="v1",
            parent_hash=None,
            signatures=[{"signer_id": "user1", "signature": "sig1"},
                       {"signer_id": "user2", "signature": "sig2"}],
            author="admin"
        )
        
        metrics = tracker.get_chain_integrity()
        assert metrics['status'] == "healthy"
        assert metrics['broken_chains'] == 0
    
    def test_get_version_metrics(self):
        """Test version tracking metrics"""
        tracker = PolicyLineageTracker()
        
        # Add multiple versions
        tracker.record_policy_version(
            "pol_001", 1, "v1", None,
            [{"signer_id": "user1", "signature": "sig1"}], "admin"
        )
        tracker.record_policy_version(
            "pol_001", 2, "v2", "hash1",
            [{"signer_id": "user1", "signature": "sig1"}], "admin"
        )
        tracker.record_policy_version(
            "pol_002", 1, "v1", None,
            [{"signer_id": "user1", "signature": "sig1"}], "admin"
        )
        
        metrics = tracker.get_version_metrics()
        assert metrics['total_versions'] == 3
        assert metrics['active_policies'] == 2
    
    def test_get_multi_sig_metrics(self):
        """Test multi-signature compliance metrics"""
        tracker = PolicyLineageTracker()
        
        # Add versions with proper signatures
        tracker.record_policy_version(
            "pol_001", 1, "v1", None,
            [{"signer_id": "user1", "signature": "sig1"},
             {"signer_id": "user2", "signature": "sig2"}],
            "admin"
        )
        
        metrics = tracker.get_multi_sig_metrics()
        assert metrics['total_changes'] == 1
        assert metrics['properly_signed'] == 1
        assert metrics['compliance_rate'] == 1.0


class TestAppealsMetricsCollector:
    """Test appeals metrics collector"""
    
    def test_creation(self):
        """Test creating appeals collector"""
        collector = AppealsMetricsCollector()
        assert len(collector._appeals) == 0
    
    def test_record_appeal(self):
        """Test recording appeal"""
        collector = AppealsMetricsCollector()
        
        collector.record_appeal("app_001", "dec_001")
        assert len(collector._appeals) == 1
    
    def test_resolve_appeal(self):
        """Test resolving appeal"""
        collector = AppealsMetricsCollector()
        
        collector.record_appeal("app_001", "dec_001")
        collector.resolve_appeal("app_001", "upheld")
        
        appeal = collector._appeals[0]
        assert appeal.outcome == "upheld"
        assert appeal.resolved_at is not None
        assert appeal.resolution_hours is not None
    
    def test_get_volume_metrics(self):
        """Test volume metrics"""
        collector = AppealsMetricsCollector()
        
        # Record some appeals
        collector.record_appeal("app_001", "dec_001")
        collector.record_appeal("app_002", "dec_002")
        collector.resolve_appeal("app_001", "upheld")
        
        metrics = collector.get_volume_metrics()
        assert metrics['total_appeals'] == 2
        assert metrics['pending_appeals'] == 1
        assert metrics['resolved_appeals'] == 1
    
    def test_get_resolution_metrics(self):
        """Test resolution time metrics"""
        collector = AppealsMetricsCollector()
        
        # Record and resolve appeals
        now = datetime.utcnow()
        
        collector.record_appeal("app_001", "dec_001", filed_at=now - timedelta(hours=48))
        collector.resolve_appeal("app_001", "upheld", resolved_at=now)
        
        collector.record_appeal("app_002", "dec_002", filed_at=now - timedelta(hours=24))
        collector.resolve_appeal("app_002", "overturned", resolved_at=now)
        
        metrics = collector.get_resolution_metrics()
        assert metrics['sample_size'] == 2
        assert metrics['median_hours'] > 0
        assert 'slo_compliance_rate' in metrics
    
    def test_get_outcome_distribution(self):
        """Test outcome distribution"""
        collector = AppealsMetricsCollector()
        
        # Create appeals with different outcomes
        for i, outcome in enumerate(["upheld", "overturned", "upheld", "modified"]):
            collector.record_appeal(f"app_{i}", f"dec_{i}")
            collector.resolve_appeal(f"app_{i}", outcome)
        
        distribution = collector.get_outcome_distribution()
        assert distribution['total'] == 4
        assert 'upheld' in distribution['distribution']
        assert distribution['distribution']['upheld']['count'] == 2


class TestGovernanceDashboard:
    """Test governance dashboard"""
    
    def test_creation(self):
        """Test creating dashboard"""
        dashboard = GovernanceDashboard(cache_ttl_seconds=60)
        assert dashboard.cache_ttl_seconds == 60
        assert len(dashboard.config) > 0
    
    def test_get_metrics_all_sections(self):
        """Test getting all metrics"""
        dashboard = GovernanceDashboard()
        metrics = dashboard.get_metrics()
        
        assert isinstance(metrics, DashboardMetrics)
        assert metrics.fairness is not None
        assert metrics.policy_lineage is not None
        assert metrics.appeals is not None
        assert metrics.audit_log is not None
        assert metrics.slo_compliance is not None
    
    def test_get_metrics_specific_sections(self):
        """Test getting specific sections"""
        dashboard = GovernanceDashboard()
        metrics = dashboard.get_metrics(sections=["fairness"])
        
        assert metrics.fairness is not None
        assert metrics.policy_lineage == {}
        assert metrics.appeals == {}
    
    def test_get_metrics_latency_slo(self):
        """Test dashboard query latency SLO"""
        dashboard = GovernanceDashboard()
        metrics = dashboard.get_metrics()
        
        # Check that query completed within 5s SLO
        latency = metrics.slo_compliance['query_latency_seconds']
        assert latency < 5.0
        assert metrics.slo_compliance['latency_slo_met'] is True
    
    def test_caching(self):
        """Test metric caching"""
        dashboard = GovernanceDashboard(cache_ttl_seconds=60)
        
        # First query (cache miss)
        metrics1 = dashboard.get_metrics(sections=["fairness"])
        
        # Second query (cache hit)
        metrics2 = dashboard.get_metrics(sections=["fairness"])
        
        # Should be cached
        assert "fairness" in dashboard._cache
    
    def test_cache_bypass(self):
        """Test bypassing cache"""
        dashboard = GovernanceDashboard()
        
        # Query with cache
        metrics1 = dashboard.get_metrics(sections=["fairness"], use_cache=True)
        
        # Query without cache
        metrics2 = dashboard.get_metrics(sections=["fairness"], use_cache=False)
        
        # Both should succeed
        assert metrics1.fairness is not None
        assert metrics2.fairness is not None
    
    def test_export_json(self):
        """Test JSON export"""
        dashboard = GovernanceDashboard()
        
        json_data = dashboard.export_metrics(format="json")
        assert isinstance(json_data, str)
        
        # Should be valid JSON
        parsed = json.loads(json_data)
        assert 'timestamp' in parsed
    
    def test_export_csv(self):
        """Test CSV export"""
        dashboard = GovernanceDashboard()
        
        csv_data = dashboard.export_metrics(format="csv")
        assert isinstance(csv_data, str)
        assert 'Section,Metric,Value,Timestamp' in csv_data
    
    def test_update_probe_result(self):
        """Test updating probe results"""
        dashboard = GovernanceDashboard()
        
        from probes import ProbeResult, ProbeStatus
        
        result = ProbeResult(
            probe_name="test-probe",
            status=ProbeStatus.HEALTHY,
            timestamp=datetime.utcnow(),
            message="Test",
            violations=[]
        )
        
        dashboard.update_probe_result("test-probe", result)
        assert "test-probe" in dashboard._probe_results
    
    def test_get_accessibility_info(self):
        """Test getting accessibility information"""
        dashboard = GovernanceDashboard()
        
        accessibility = dashboard.get_accessibility_info()
        assert 'wcag_version' in accessibility
        assert accessibility['wcag_version'] == "2.1"
        assert accessibility['conformance_level'] == "AA"
    
    def test_record_fairness_decision(self):
        """Test recording fairness decision"""
        dashboard = GovernanceDashboard()
        
        dashboard.fairness_collector.record_decision(
            decision="allow",
            protected_group="female",
            context={"age": 30}
        )
        
        assert len(dashboard.fairness_collector._decisions) == 1
    
    def test_record_policy_version(self):
        """Test recording policy version"""
        dashboard = GovernanceDashboard()
        
        dashboard.lineage_tracker.record_policy_version(
            policy_id="pol_001",
            version=1,
            content="test policy",
            parent_hash=None,
            signatures=[{"signer_id": "user1", "signature": "sig1"}],
            author="admin"
        )
        
        assert "pol_001" in dashboard.lineage_tracker._policies
    
    def test_record_appeal(self):
        """Test recording appeal"""
        dashboard = GovernanceDashboard()
        
        dashboard.appeals_collector.record_appeal("app_001", "dec_001")
        assert len(dashboard.appeals_collector._appeals) == 1


class TestDashboardIntegration:
    """Integration tests for dashboard"""
    
    def test_end_to_end_fairness_workflow(self):
        """Test complete fairness workflow"""
        dashboard = GovernanceDashboard()
        
        # Record decisions
        for _ in range(50):
            dashboard.fairness_collector.record_decision("allow", "female")
            dashboard.fairness_collector.record_decision("allow", None)
        
        for _ in range(50):
            dashboard.fairness_collector.record_decision("deny", "female")
            dashboard.fairness_collector.record_decision("deny", None)
        
        # Get metrics
        metrics = dashboard.get_metrics(sections=["fairness"])
        
        # Verify fairness is balanced
        sp = metrics.fairness['statistical_parity']
        assert sp['status'] == "healthy"
        assert abs(sp['difference']) <= 0.10
    
    def test_end_to_end_appeals_workflow(self):
        """Test complete appeals workflow"""
        dashboard = GovernanceDashboard()
        
        # File appeals
        for i in range(5):
            dashboard.appeals_collector.record_appeal(f"app_{i}", f"dec_{i}")
        
        # Resolve some appeals
        dashboard.appeals_collector.resolve_appeal("app_0", "upheld")
        dashboard.appeals_collector.resolve_appeal("app_1", "overturned")
        
        # Get metrics
        metrics = dashboard.get_metrics(sections=["appeals"])
        
        # Verify counts
        volume = metrics.appeals['volume']
        assert volume['total_appeals'] == 5
        assert volume['pending_appeals'] == 3
        assert volume['resolved_appeals'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
