"""Comprehensive tests for Phase 4 components."""

import pytest
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from nethical.core import (
    # Phase 4 components
    MerkleAnchor,
    PolicyDiffAuditor,
    QuarantineManager,
    EthicalTaxonomy,
    SLAMonitor,
    Phase4IntegratedGovernance,
    # Enums and types
    ChangeType,
    RiskLevel,
    QuarantineReason,
    QuarantineStatus,
    SLAStatus
)


class TestMerkleAnchor:
    """Test Merkle anchoring system."""
    
    def test_merkle_initialization(self):
        """Test Merkle anchor initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = MerkleAnchor(storage_path=tmpdir)
            
            assert anchor.storage_path.exists()
            assert anchor.chunk_size == 1000
            assert anchor.current_chunk is not None
    
    def test_add_event(self):
        """Test adding events to Merkle tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = MerkleAnchor(storage_path=tmpdir)
            
            event = {
                'event_id': 'test_001',
                'agent_id': 'agent_1',
                'action': 'test_action'
            }
            
            result = anchor.add_event(event)
            assert result is True
            assert anchor.current_chunk.event_count == 1
    
    def test_finalize_chunk(self):
        """Test chunk finalization and Merkle root computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = MerkleAnchor(storage_path=tmpdir, chunk_size=10)
            
            # Add events
            for i in range(5):
                anchor.add_event({'event_id': f'evt_{i}', 'data': i})
            
            # Finalize chunk
            merkle_root = anchor.finalize_chunk()
            
            assert merkle_root is not None
            assert len(merkle_root) == 64  # SHA256 hex length
            assert anchor.current_chunk.event_count == 0  # New chunk created
    
    def test_verify_chunk(self):
        """Test chunk verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = MerkleAnchor(storage_path=tmpdir, chunk_size=10)
            
            # Add and finalize chunk
            for i in range(5):
                anchor.add_event({'event_id': f'evt_{i}', 'data': i})
            
            merkle_root = anchor.finalize_chunk()
            chunk_id = list(anchor.finalized_chunks.keys())[0]
            
            # Verify chunk
            is_valid = anchor.verify_chunk(chunk_id)
            assert is_valid is True
    
    def test_statistics(self):
        """Test Merkle statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = MerkleAnchor(storage_path=tmpdir, chunk_size=10)
            
            # Add events across multiple chunks
            for i in range(25):
                anchor.add_event({'event_id': f'evt_{i}'})
            
            stats = anchor.get_statistics()
            
            assert stats['total_chunks'] >= 2
            assert stats['total_events'] >= 20
            assert stats['hash_algorithm'] == 'sha256'


class TestPolicyDiffAuditor:
    """Test policy diff auditing."""
    
    def test_auditor_initialization(self):
        """Test policy diff auditor initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auditor = PolicyDiffAuditor(storage_path=tmpdir)
            
            assert auditor.storage_path.exists()
            assert len(auditor.high_risk_fields) > 0
    
    def test_compare_policies_added(self):
        """Test policy comparison with added fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auditor = PolicyDiffAuditor(storage_path=tmpdir)
            
            old_policy = {'threshold': 0.5}
            new_policy = {'threshold': 0.5, 'new_field': 'value'}
            
            result = auditor.compare_policies(old_policy, new_policy)
            
            assert result.summary['added'] == 1
            assert result.risk_score > 0
    
    def test_compare_policies_removed(self):
        """Test policy comparison with removed fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auditor = PolicyDiffAuditor(storage_path=tmpdir)
            
            old_policy = {'threshold': 0.5, 'old_field': 'value'}
            new_policy = {'threshold': 0.5}
            
            result = auditor.compare_policies(old_policy, new_policy)
            
            assert result.summary['removed'] == 1
            assert result.risk_score > 0.5  # Removals are risky
    
    def test_compare_policies_modified(self):
        """Test policy comparison with modified fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auditor = PolicyDiffAuditor(storage_path=tmpdir)
            
            old_policy = {'threshold': 0.5}
            new_policy = {'threshold': 0.8}
            
            result = auditor.compare_policies(old_policy, new_policy)
            
            assert result.summary['modified'] == 1
            assert result.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
    
    def test_high_risk_field_detection(self):
        """Test high-risk field detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auditor = PolicyDiffAuditor(storage_path=tmpdir)
            
            old_policy = {'security': {'enabled': True}}
            new_policy = {'security': {'enabled': False}}
            
            result = auditor.compare_policies(old_policy, new_policy)
            
            # Security field changes should be high risk
            assert result.risk_score > 0.5
            assert len(result.recommendations) > 0
    
    def test_save_and_load_policy(self):
        """Test policy version saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            auditor = PolicyDiffAuditor(storage_path=tmpdir)
            
            policy = {'test': 'value'}
            version = 'v1.0'
            
            auditor.save_policy_version(policy, version)
            loaded = auditor.load_policy_version(version)
            
            assert loaded == policy


class TestQuarantineManager:
    """Test quarantine management."""
    
    def test_quarantine_initialization(self):
        """Test quarantine manager initialization."""
        manager = QuarantineManager()
        
        assert manager.default_duration_hours == 24.0
        assert manager.target_activation_time_s == 15.0
    
    def test_quarantine_cohort(self):
        """Test cohort quarantine."""
        manager = QuarantineManager()
        manager.register_agent_cohort('agent_1', 'test_cohort')
        
        record = manager.quarantine_cohort(
            cohort='test_cohort',
            reason=QuarantineReason.ANOMALY_DETECTED
        )
        
        assert record.status == QuarantineStatus.ACTIVE
        assert record.cohort == 'test_cohort'
        assert record.activation_time_ms is not None
    
    def test_quarantine_activation_speed(self):
        """Test quarantine activation speed (<15s requirement)."""
        manager = QuarantineManager()
        manager.register_agent_cohort('agent_1', 'speed_test')
        
        start = time.time()
        record = manager.quarantine_cohort(
            cohort='speed_test',
            reason=QuarantineReason.COORDINATED_ATTACK
        )
        activation_time_s = time.time() - start
        
        assert activation_time_s < 15.0
        assert record.activation_time_ms < 15000
    
    def test_quarantine_status(self):
        """Test quarantine status checking."""
        manager = QuarantineManager()
        manager.register_agent_cohort('agent_1', 'status_test')
        
        # Before quarantine
        status = manager.get_quarantine_status('status_test')
        assert status['is_quarantined'] is False
        
        # After quarantine
        manager.quarantine_cohort('status_test', QuarantineReason.ANOMALY_DETECTED)
        status = manager.get_quarantine_status('status_test')
        assert status['is_quarantined'] is True
    
    def test_release_cohort(self):
        """Test cohort release from quarantine."""
        manager = QuarantineManager()
        manager.register_agent_cohort('agent_1', 'release_test')
        
        manager.quarantine_cohort('release_test', QuarantineReason.MANUAL_OVERRIDE)
        released = manager.release_cohort('release_test')
        
        assert released is True
        status = manager.get_quarantine_status('release_test')
        assert status['is_quarantined'] is False
    
    def test_simulate_attack_response(self):
        """Test synthetic attack simulation."""
        manager = QuarantineManager()
        manager.register_agent_cohort('agent_1', 'attack_sim')
        
        result = manager.simulate_attack_response('attack_sim')
        
        assert result['meets_requirement'] is True
        assert result['total_time_s'] < 15.0
        assert result['status'] == QuarantineStatus.ACTIVE.value
    
    def test_agent_quarantine_check(self):
        """Test individual agent quarantine check."""
        manager = QuarantineManager()
        manager.register_agent_cohort('agent_1', 'agent_test')
        
        assert manager.is_agent_quarantined('agent_1') is False
        
        manager.quarantine_cohort('agent_test', QuarantineReason.HIGH_RISK_SCORE)
        assert manager.is_agent_quarantined('agent_1') is True


class TestEthicalTaxonomy:
    """Test ethical taxonomy system."""
    
    def test_taxonomy_initialization(self):
        """Test ethical taxonomy initialization."""
        taxonomy = EthicalTaxonomy(taxonomy_path='ethics_taxonomy.json')
        
        assert len(taxonomy.dimensions) > 0
        assert 'privacy' in taxonomy.dimensions
        assert 'manipulation' in taxonomy.dimensions
        assert 'fairness' in taxonomy.dimensions
        assert 'safety' in taxonomy.dimensions
    
    def test_tag_violation(self):
        """Test violation tagging."""
        taxonomy = EthicalTaxonomy(taxonomy_path='ethics_taxonomy.json')
        
        scores = taxonomy.tag_violation('unauthorized_data_access')
        
        assert 'privacy' in scores
        assert scores['privacy'] > 0.5  # Should be high
    
    def test_create_tagging(self):
        """Test complete tagging creation."""
        taxonomy = EthicalTaxonomy(taxonomy_path='ethics_taxonomy.json')
        
        tagging = taxonomy.create_tagging(
            violation_type='emotional_manipulation',
            context={'automated': True}
        )
        
        assert len(tagging.tags) > 0
        assert tagging.primary_dimension is not None
    
    def test_coverage_tracking(self):
        """Test coverage tracking."""
        taxonomy = EthicalTaxonomy(taxonomy_path='ethics_taxonomy.json')
        
        # Tag some violations
        taxonomy.tag_violation('unauthorized_data_access')
        taxonomy.tag_violation('emotional_manipulation')
        
        stats = taxonomy.get_coverage_stats()
        
        assert stats['total_violation_types'] >= 2
        assert stats['tagged_types'] >= 2
    
    def test_coverage_target(self):
        """Test coverage target (>90%)."""
        taxonomy = EthicalTaxonomy(taxonomy_path='ethics_taxonomy.json')
        
        # Tag many known violations
        known_violations = list(taxonomy.mapping.keys())
        for violation in known_violations[:int(len(known_violations) * 0.95)]:
            taxonomy.tag_violation(violation)
        
        stats = taxonomy.get_coverage_stats()
        
        # Should have high coverage since we're using known mappings
        assert stats['coverage_percentage'] > 90.0 or stats['meets_target']
    
    def test_dimension_report(self):
        """Test dimension-specific report."""
        taxonomy = EthicalTaxonomy(taxonomy_path='ethics_taxonomy.json')
        
        report = taxonomy.get_dimension_report('privacy')
        
        assert 'dimension' in report
        assert report['dimension'] == 'privacy'
        assert 'description' in report


class TestSLAMonitor:
    """Test SLA monitoring."""
    
    def test_sla_initialization(self):
        """Test SLA monitor initialization."""
        monitor = SLAMonitor(target_p95_ms=220.0)
        
        assert monitor.target_p95_ms == 220.0
        assert len(monitor.sla_targets) >= 3
    
    def test_record_latency(self):
        """Test latency recording."""
        monitor = SLAMonitor()
        
        monitor.record_latency(100.0)
        monitor.record_latency(150.0)
        monitor.record_latency(200.0)
        
        metrics = monitor.get_current_metrics()
        assert metrics['sample_count'] == 3
        assert metrics['avg_latency_ms'] > 0
    
    def test_p95_calculation(self):
        """Test P95 latency calculation."""
        monitor = SLAMonitor()
        
        # Add 100 measurements
        for i in range(100):
            monitor.record_latency(float(i))
        
        metrics = monitor.get_current_metrics()
        
        # P95 should be around 95
        assert 90 <= metrics['p95_latency_ms'] <= 99
    
    def test_sla_compliance_met(self):
        """Test SLA compliance when target is met."""
        monitor = SLAMonitor(target_p95_ms=220.0)
        
        # Add measurements under target
        for _ in range(100):
            monitor.record_latency(100.0)  # Well under 220ms
        
        compliance = monitor.check_sla_compliance()
        
        assert compliance['overall_status'] == SLAStatus.COMPLIANT
    
    def test_sla_compliance_breach(self):
        """Test SLA compliance breach detection."""
        monitor = SLAMonitor(target_p95_ms=100.0)
        
        # Add measurements over target
        for _ in range(100):
            monitor.record_latency(200.0)  # Over 100ms target
        
        compliance = monitor.check_sla_compliance()
        
        assert compliance['overall_status'] in [SLAStatus.WARNING, SLAStatus.BREACH]
    
    def test_sla_report(self):
        """Test comprehensive SLA report."""
        monitor = SLAMonitor(target_p95_ms=220.0)
        
        for _ in range(50):
            monitor.record_latency(150.0)
        
        report = monitor.get_sla_report()
        
        assert 'sla_met' in report
        assert 'p95_latency_ms' in report
        assert 'metrics' in report
    
    def test_target_validation(self):
        """Test P95 <220ms @ 2x load target."""
        monitor = SLAMonitor(target_p95_ms=220.0)
        
        # Simulate 2x load (expect some latency increase)
        monitor.set_load_multiplier(2.0)
        
        # Add realistic measurements for 2x load
        for _ in range(100):
            monitor.record_latency(180.0)  # Under 220ms even at 2x
        
        report = monitor.get_sla_report()
        
        # Should still meet SLA at 2x load
        assert report['p95_latency_ms'] < 220.0


class TestPhase4Integration:
    """Test Phase 4 integrated governance."""
    
    def test_integration_initialization(self):
        """Test Phase 4 integration initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase4IntegratedGovernance(
                storage_dir=tmpdir,
                taxonomy_path='ethics_taxonomy.json'
            )
            
            assert gov.merkle_anchor is not None
            assert gov.policy_auditor is not None
            assert gov.quarantine_manager is not None
            assert gov.ethical_taxonomy is not None
            assert gov.sla_monitor is not None
    
    def test_process_action(self):
        """Test integrated action processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase4IntegratedGovernance(
                storage_dir=tmpdir,
                taxonomy_path='ethics_taxonomy.json'
            )
            
            result = gov.process_action(
                agent_id='agent_1',
                action='test_action',
                cohort='test_cohort',
                violation_detected=True,
                violation_type='unauthorized_data_access'
            )
            
            assert result['action_allowed'] is True
            assert 'ethical_tags' in result
            assert 'latency_ms' in result
    
    def test_quarantine_integration(self):
        """Test quarantine integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase4IntegratedGovernance(storage_dir=tmpdir)
            
            # Quarantine cohort
            quar_result = gov.quarantine_cohort('blocked_cohort', 'manual')
            assert quar_result['status'] == QuarantineStatus.ACTIVE.value
            
            # Try to process action from quarantined cohort
            result = gov.process_action(
                agent_id='agent_1',
                action='test',
                cohort='blocked_cohort'
            )
            
            assert result['action_allowed'] is False
            assert result['reason'] == 'cohort_quarantined'
    
    def test_policy_comparison(self):
        """Test policy comparison integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase4IntegratedGovernance(storage_dir=tmpdir)
            
            old_policy = {'threshold': 0.5}
            new_policy = {'threshold': 0.8}
            
            diff = gov.compare_policies(old_policy, new_policy)
            
            assert 'risk_score' in diff
            assert 'changes' in diff
    
    def test_system_status(self):
        """Test system status reporting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase4IntegratedGovernance(
                storage_dir=tmpdir,
                taxonomy_path='ethics_taxonomy.json'
            )
            
            status = gov.get_system_status()
            
            assert 'components' in status
            assert 'merkle_anchor' in status['components']
            assert 'quarantine' in status['components']
            assert 'ethical_taxonomy' in status['components']
            assert 'sla_monitor' in status['components']
    
    def test_merkle_verification(self):
        """Test Merkle verification through integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gov = Phase4IntegratedGovernance(storage_dir=tmpdir)
            
            # Process some actions
            for i in range(5):
                gov.process_action(
                    agent_id=f'agent_{i}',
                    action=f'action_{i}',
                    cohort='test'
                )
            
            # Finalize chunk
            merkle_root = gov.finalize_audit_chunk()
            assert merkle_root is not None
            
            # Verify chunk
            chunk_id = list(gov.merkle_anchor.finalized_chunks.keys())[0]
            is_valid = gov.verify_audit_segment(chunk_id)
            assert is_valid is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
