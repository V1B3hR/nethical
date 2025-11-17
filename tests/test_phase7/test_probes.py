"""
Tests for Phase 7 Runtime Probes

Tests all probe implementations including invariant probes,
governance probes, and performance probes.
"""

import pytest
from datetime import datetime, timedelta
from probes import (
    BaseProbe,
    ProbeResult,
    ProbeStatus,
    DeterminismProbe,
    TerminationProbe,
    AcyclicityProbe,
    AuditCompletenessProbe,
    NonRepudiationProbe,
    MultiSigProbe,
    PolicyLineageProbe,
    DataMinimizationProbe,
    TenantIsolationProbe,
    LatencyProbe,
    ThroughputProbe,
    ResourceUtilizationProbe,
)


class TestBaseProbe:
    """Test base probe infrastructure"""
    
    def test_probe_creation(self):
        """Test creating a base probe"""
        class TestProbe(BaseProbe):
            def check(self):
                return ProbeResult(
                    probe_name=self.name,
                    status=ProbeStatus.HEALTHY,
                    timestamp=datetime.utcnow(),
                    message="Test probe check"
                )
        
        probe = TestProbe(name="test-probe", check_interval_seconds=60)
        assert probe.name == "test-probe"
        assert probe.check_interval_seconds == 60
        assert probe._consecutive_failures == 0
    
    def test_probe_run(self):
        """Test running a probe check"""
        class TestProbe(BaseProbe):
            def check(self):
                return ProbeResult(
                    probe_name=self.name,
                    status=ProbeStatus.HEALTHY,
                    timestamp=datetime.utcnow(),
                    message="Test check"
                )
        
        probe = TestProbe(name="test-probe")
        result = probe.run()
        
        assert result.probe_name == "test-probe"
        assert result.status == ProbeStatus.HEALTHY
        assert len(probe.get_history()) == 1
    
    def test_probe_consecutive_failures(self):
        """Test consecutive failure tracking"""
        class FailingProbe(BaseProbe):
            def check(self):
                return ProbeResult(
                    probe_name=self.name,
                    status=ProbeStatus.CRITICAL,
                    timestamp=datetime.utcnow(),
                    message="Failure"
                )
        
        probe = FailingProbe(name="failing-probe", alert_threshold=3)
        
        # First failure
        probe.run()
        assert probe._consecutive_failures == 1
        assert not probe.should_alert()
        
        # Second failure
        probe.run()
        assert probe._consecutive_failures == 2
        assert not probe.should_alert()
        
        # Third failure - should trigger alert
        probe.run()
        assert probe._consecutive_failures == 3
        assert probe.should_alert()
    
    def test_probe_metrics(self):
        """Test probe metrics aggregation"""
        class TestProbe(BaseProbe):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.check_count = 0
            
            def check(self):
                self.check_count += 1
                status = ProbeStatus.HEALTHY if self.check_count % 2 == 0 else ProbeStatus.WARNING
                return ProbeResult(
                    probe_name=self.name,
                    status=status,
                    timestamp=datetime.utcnow(),
                    message=f"Check {self.check_count}"
                )
        
        probe = TestProbe(name="test-probe")
        
        # Run multiple checks
        for _ in range(10):
            probe.run()
        
        metrics = probe.get_metrics()
        assert metrics['total_checks'] == 10
        assert metrics['healthy_count'] == 5
        assert metrics['warning_count'] == 5
        assert metrics['health_rate'] == 0.5


class TestDeterminismProbe:
    """Test determinism probe"""
    
    def test_determinism_probe_creation(self):
        """Test creating determinism probe"""
        probe = DeterminismProbe(
            evaluation_service=None,
            check_interval_seconds=300,
            sample_size=10
        )
        assert probe.name == "P-DET-Determinism"
        assert probe.sample_size == 10
    
    def test_add_test_case(self):
        """Test adding test cases"""
        probe = DeterminismProbe(evaluation_service=None)
        
        probe.add_test_case("pol_001", {"action": "read"})
        probe.add_test_case("pol_002", {"action": "write"})
        
        assert len(probe._test_cases) == 2
    
    def test_determinism_check_no_cases(self):
        """Test check with no test cases"""
        probe = DeterminismProbe(evaluation_service=None)
        result = probe.run()
        
        assert result.status == ProbeStatus.WARNING
        assert "No test cases" in result.message


class TestTerminationProbe:
    """Test termination probe"""
    
    def test_termination_probe_creation(self):
        """Test creating termination probe"""
        probe = TerminationProbe(max_evaluation_time_ms=5000)
        assert probe.name == "P-TERM-Termination"
        assert probe.max_evaluation_time_ms == 5000
    
    def test_record_evaluation(self):
        """Test recording evaluations"""
        probe = TerminationProbe()
        
        probe.record_evaluation("pol_001", duration_ms=100.0, completed=True)
        probe.record_evaluation("pol_002", duration_ms=200.0, completed=True)
        
        assert len(probe._recent_evaluations) == 2
    
    def test_termination_check_healthy(self):
        """Test check with all evaluations within bounds"""
        probe = TerminationProbe(max_evaluation_time_ms=1000)
        
        # Record healthy evaluations
        for i in range(10):
            probe.record_evaluation(f"pol_{i}", duration_ms=500.0, completed=True)
        
        result = probe.run()
        assert result.status == ProbeStatus.HEALTHY
        assert result.metrics['timeout_violations'] == 0
        assert result.metrics['termination_rate'] == 1.0
    
    def test_termination_check_violations(self):
        """Test check with timeout violations"""
        probe = TerminationProbe(max_evaluation_time_ms=1000)
        
        # Record some violations
        probe.record_evaluation("pol_001", duration_ms=1500.0, completed=True)
        probe.record_evaluation("pol_002", duration_ms=500.0, completed=True)
        probe.record_evaluation("pol_003", duration_ms=2000.0, completed=True)
        
        result = probe.run()
        assert result.metrics['timeout_violations'] == 2
        assert len(result.violations) > 0


class TestAcyclicityProbe:
    """Test acyclicity probe"""
    
    def test_acyclicity_probe_creation(self):
        """Test creating acyclicity probe"""
        policy_graph = {"pol_001": ["pol_002"], "pol_002": []}
        probe = AcyclicityProbe(policy_graph=policy_graph)
        assert probe.name == "P-ACYCLIC-Acyclicity"
    
    def test_acyclic_graph(self):
        """Test acyclic policy graph"""
        policy_graph = {
            "pol_001": ["pol_002", "pol_003"],
            "pol_002": ["pol_004"],
            "pol_003": ["pol_004"],
            "pol_004": []
        }
        
        probe = AcyclicityProbe(policy_graph=policy_graph)
        result = probe.run()
        
        assert result.status == ProbeStatus.HEALTHY
        assert result.metrics['cycles_found'] == 0
    
    def test_cyclic_graph(self):
        """Test cyclic policy graph"""
        policy_graph = {
            "pol_001": ["pol_002"],
            "pol_002": ["pol_003"],
            "pol_003": ["pol_001"]  # Cycle!
        }
        
        probe = AcyclicityProbe(policy_graph=policy_graph)
        result = probe.run()
        
        assert result.status == ProbeStatus.CRITICAL
        assert result.metrics['cycles_found'] > 0
        assert len(result.violations) > 0


class TestMultiSigProbe:
    """Test multi-signature probe"""
    
    def test_multi_sig_probe_creation(self):
        """Test creating multi-sig probe"""
        probe = MultiSigProbe(policy_service=None, min_signatures=2)
        assert probe.name == "P-MULTI-SIG-MultiSignature"
        assert probe.min_signatures == 2
    
    def test_set_authorized_signers(self):
        """Test setting authorized signers"""
        probe = MultiSigProbe(policy_service=None)
        signers = ["user1", "user2", "user3"]
        
        probe.set_authorized_signers(signers)
        assert len(probe._authorized_signers) == 3


class TestDataMinimizationProbe:
    """Test data minimization probe"""
    
    def test_data_min_probe_creation(self):
        """Test creating data minimization probe"""
        allowed = {"field1", "field2"}
        probe = DataMinimizationProbe(allowed_fields=allowed)
        assert probe.name == "P-DATA-MIN-DataMinimization"
        assert probe.allowed_fields == allowed
    
    def test_record_authorized_access(self):
        """Test recording authorized access"""
        probe = DataMinimizationProbe(
            allowed_fields={"action_type", "resource_type"}
        )
        
        # Record authorized access
        probe.record_access({"action_type", "resource_type"})
        
        result = probe.run()
        assert result.status == ProbeStatus.HEALTHY
        assert result.metrics['unauthorized_count'] == 0
    
    def test_record_unauthorized_access(self):
        """Test recording unauthorized access"""
        probe = DataMinimizationProbe(
            allowed_fields={"action_type", "resource_type"}
        )
        
        # Record unauthorized access
        probe.record_access({"action_type", "sensitive_data"})
        
        result = probe.run()
        assert result.metrics['unauthorized_count'] == 1
        assert len(result.violations) > 0


class TestTenantIsolationProbe:
    """Test tenant isolation probe"""
    
    def test_tenant_iso_probe_creation(self):
        """Test creating tenant isolation probe"""
        probe = TenantIsolationProbe()
        assert probe.name == "P-TENANT-ISO-TenantIsolation"
    
    def test_record_isolated_access(self):
        """Test recording isolated access"""
        probe = TenantIsolationProbe()
        
        # Record same-tenant access
        probe.record_access("tenant_123", "tenant_123")
        probe.record_access("tenant_456", "tenant_456")
        
        result = probe.run()
        assert result.status == ProbeStatus.HEALTHY
        assert result.metrics['cross_tenant_count'] == 0
    
    def test_record_cross_tenant_access(self):
        """Test recording cross-tenant access"""
        probe = TenantIsolationProbe()
        
        # Record cross-tenant access
        probe.record_access("tenant_123", "tenant_456")
        
        result = probe.run()
        assert result.metrics['cross_tenant_count'] == 1
        assert len(result.violations) > 0


class TestLatencyProbe:
    """Test latency probe"""
    
    def test_latency_probe_creation(self):
        """Test creating latency probe"""
        probe = LatencyProbe(p95_target_ms=100.0, p99_target_ms=500.0)
        assert probe.name == "Latency-Monitor"
        assert probe.p95_target_ms == 100.0
    
    def test_record_latency(self):
        """Test recording latency samples"""
        probe = LatencyProbe()
        
        probe.record_latency(50.0)
        probe.record_latency(75.0)
        probe.record_latency(100.0)
        
        assert len(probe._latency_samples) == 3
    
    def test_latency_within_slo(self):
        """Test latency within SLO"""
        probe = LatencyProbe(p95_target_ms=100.0)
        
        # Record samples under SLO
        for _ in range(100):
            probe.record_latency(50.0)
        
        result = probe.run()
        assert result.status == ProbeStatus.HEALTHY
        assert result.metrics['slo_compliance'] is True


class TestThroughputProbe:
    """Test throughput probe"""
    
    def test_throughput_probe_creation(self):
        """Test creating throughput probe"""
        probe = ThroughputProbe(target_rps=1000.0)
        assert probe.name == "Throughput-Monitor"
        assert probe.target_rps == 1000.0
    
    def test_record_requests(self):
        """Test recording requests"""
        probe = ThroughputProbe()
        
        probe.record_request()
        probe.record_request()
        probe.record_request()
        
        assert len(probe._request_timestamps) == 3


class TestResourceUtilizationProbe:
    """Test resource utilization probe"""
    
    def test_resource_probe_creation(self):
        """Test creating resource utilization probe"""
        probe = ResourceUtilizationProbe(
            cpu_threshold_percent=80.0,
            memory_threshold_percent=85.0
        )
        assert probe.name == "ResourceUtilization-Monitor"
        assert probe.cpu_threshold == 80.0
        assert probe.memory_threshold == 85.0
    
    def test_resource_probe_check(self):
        """Test resource utilization check"""
        probe = ResourceUtilizationProbe()
        result = probe.run()
        
        # Should return some metrics
        assert 'cpu_percent' in result.metrics
        assert 'memory_percent' in result.metrics
        assert 'disk_percent' in result.metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
