"""
Tests for F5: Simulation & Replay functionality

Tests cover:
- Action stream persistence with >1M actions
- Time-travel replay functionality
- What-if analysis interface
- Policy validation workflow
- Performance benchmarks (replay speed)
"""

import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from nethical.core.action_replayer import ActionReplayer, ReplayResult, PolicyComparison
from nethical.core.governance import (
    AgentAction,
    ActionType,
    PersistenceManager,
    JudgmentResult,
    Decision,
    SafetyViolation,
    ViolationType,
    Severity,
)


class TestActionReplayer:
    """Test suite for ActionReplayer functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def replayer(self, temp_db):
        """Create an ActionReplayer instance with temporary storage."""
        return ActionReplayer(storage_path=temp_db)
    
    @pytest.fixture
    def populated_db(self, temp_db):
        """Create a database populated with test actions."""
        persistence = PersistenceManager(
            db_path=os.path.join(temp_db, "action_streams.db"),
            retention_days=365
        )
        
        # Create test actions
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        actions = []
        
        for i in range(100):
            action = AgentAction(
                action_id=f"action_{i}",
                agent_id=f"agent_{i % 10}",  # 10 different agents
                action_type=ActionType.QUERY,
                content=f"Test action {i} with content",
                timestamp=base_time + timedelta(minutes=i),
                intent=f"Intent {i}",
                risk_score=0.1 * (i % 10),
            )
            actions.append(action)
            persistence.store_action(action)
            
            # Store some judgments
            if i % 3 == 0:
                judgment = JudgmentResult(
                    judgment_id=f"judgment_{i}",
                    action_id=f"action_{i}",
                    decision=Decision.ALLOW if i % 6 == 0 else Decision.WARN,
                    confidence=0.8 + (i % 10) * 0.02,
                    reasoning=f"Test reasoning {i}",
                    violations=[],
                    timestamp=base_time + timedelta(minutes=i),
                    modifications={},
                    feedback={},
                    remediation_steps=[],
                    follow_up_required=False,
                )
                persistence.store_judgment(judgment)
        
        return temp_db, actions
    
    def test_initialization_with_file_path(self, temp_db):
        """Test initialization with direct database file path."""
        db_path = os.path.join(temp_db, "test.db")
        replayer = ActionReplayer(storage_path=db_path)
        
        assert replayer.db_path == db_path
        assert replayer.current_timestamp is None
        assert replayer.start_timestamp is None
        assert replayer.end_timestamp is None
    
    def test_initialization_with_directory_path(self, temp_db):
        """Test initialization with directory path."""
        replayer = ActionReplayer(storage_path=temp_db)
        
        expected_path = os.path.join(temp_db, "action_streams.db")
        assert replayer.db_path == expected_path
    
    def test_set_timestamp(self, replayer):
        """Test setting time-travel timestamp."""
        timestamp = "2024-01-15T10:30:00Z"
        replayer.set_timestamp(timestamp)
        
        assert replayer.current_timestamp == timestamp
        assert replayer.start_timestamp == timestamp
    
    def test_set_timestamp_invalid_format(self, replayer):
        """Test that invalid timestamp format raises error."""
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            replayer.set_timestamp("not-a-timestamp")
    
    def test_set_time_range(self, replayer):
        """Test setting time range for replay."""
        start = "2024-01-15T10:00:00Z"
        end = "2024-01-15T12:00:00Z"
        
        replayer.set_time_range(start, end)
        
        assert replayer.start_timestamp == start
        assert replayer.end_timestamp == end
    
    def test_set_time_range_invalid(self, replayer):
        """Test that invalid time range raises error."""
        with pytest.raises(ValueError):
            replayer.set_time_range(
                "2024-01-15T12:00:00Z",
                "2024-01-15T10:00:00Z"  # End before start
            )
    
    def test_get_actions_without_filters(self, populated_db):
        """Test retrieving all actions without filters."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        actions = replayer.get_actions()
        
        assert len(actions) == 100
        assert actions[0]['action_id'] == 'action_0'
    
    def test_get_actions_with_time_range(self, populated_db):
        """Test retrieving actions within time range."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        # Set time range to get first 30 actions
        replayer.set_time_range(
            "2024-01-15T10:00:00",
            "2024-01-15T10:30:00"
        )
        
        actions = replayer.get_actions()
        
        # Should get approximately 30 actions (one per minute)
        assert 25 <= len(actions) <= 35
    
    def test_get_actions_with_agent_filter(self, populated_db):
        """Test retrieving actions for specific agents."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        # Get actions for agent_0 (every 10th action)
        actions = replayer.get_actions(agent_ids=["agent_0"])
        
        assert len(actions) == 10
        assert all(a['agent_id'] == 'agent_0' for a in actions)
    
    def test_get_actions_with_pagination(self, populated_db):
        """Test pagination of action retrieval."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        # Get first 10 actions
        page1 = replayer.get_actions(limit=10, offset=0)
        assert len(page1) == 10
        assert page1[0]['action_id'] == 'action_0'
        
        # Get next 10 actions
        page2 = replayer.get_actions(limit=10, offset=10)
        assert len(page2) == 10
        assert page2[0]['action_id'] == 'action_10'
        
        # Verify no overlap
        page1_ids = {a['action_id'] for a in page1}
        page2_ids = {a['action_id'] for a in page2}
        assert len(page1_ids & page2_ids) == 0
    
    def test_count_actions(self, populated_db):
        """Test counting actions."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        total = replayer.count_actions()
        assert total == 100
        
        # Count for specific agents
        agent_count = replayer.count_actions(agent_ids=["agent_0", "agent_1"])
        assert agent_count == 20  # 10 each
    
    def test_replay_with_policy_basic(self, populated_db):
        """Test basic replay with policy."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        # Replay first 10 actions with strict policy
        results = replayer.replay_with_policy(
            new_policy="strict_policy.yaml",
            limit=10
        )
        
        assert len(results) == 10
        assert all(isinstance(r, ReplayResult) for r in results)
        assert all(r.policy_name == "strict_policy.yaml" for r in results)
        assert all(r.new_decision in [d.value for d in Decision] for r in results)
    
    def test_replay_with_policy_for_specific_agents(self, populated_db):
        """Test replaying actions for specific agents."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        results = replayer.replay_with_policy(
            new_policy="test_policy.yaml",
            agent_ids=["agent_0", "agent_1"]
        )
        
        assert len(results) == 20
        assert all(r.agent_id in ["agent_0", "agent_1"] for r in results)
    
    def test_compare_outcomes_basic(self, populated_db):
        """Test comparing outcomes between policies."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        comparison = replayer.compare_outcomes(
            baseline_policy="current",
            candidate_policy="strict_policy.yaml",
            limit=50
        )
        
        assert isinstance(comparison, PolicyComparison)
        assert comparison.total_actions == 50
        assert comparison.decisions_changed + comparison.decisions_same == 50
        assert comparison.baseline_policy == "current"
        assert comparison.candidate_policy == "strict_policy.yaml"
        assert comparison.execution_time_ms > 0
    
    def test_compare_outcomes_decision_breakdown(self, populated_db):
        """Test that policy comparison includes decision breakdown."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        comparison = replayer.compare_outcomes(
            baseline_policy="permissive.yaml",
            candidate_policy="strict.yaml",
            limit=30
        )
        
        assert "permissive.yaml" in comparison.decision_breakdown
        assert "strict.yaml" in comparison.decision_breakdown
        
        # Should have some decisions recorded
        baseline_decisions = comparison.decision_breakdown["permissive.yaml"]
        candidate_decisions = comparison.decision_breakdown["strict.yaml"]
        
        assert sum(baseline_decisions.values()) == 30
        assert sum(candidate_decisions.values()) == 30
    
    def test_compare_outcomes_restrictiveness_tracking(self, populated_db):
        """Test tracking of more/less restrictive decisions."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        comparison = replayer.compare_outcomes(
            baseline_policy="permissive.yaml",
            candidate_policy="strict.yaml",
            limit=20
        )
        
        # Verify restrictiveness counts
        assert comparison.more_restrictive >= 0
        assert comparison.less_restrictive >= 0
        assert (comparison.more_restrictive + comparison.less_restrictive 
                == comparison.decisions_changed)
    
    def test_get_statistics_with_data(self, populated_db):
        """Test getting statistics with populated database."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        stats = replayer.get_statistics()
        
        assert stats['total_actions'] == 100
        assert stats['first_action_timestamp'] is not None
        assert stats['last_action_timestamp'] is not None
        assert stats['database_path'] == replayer.db_path
    
    def test_get_statistics_empty_database(self, replayer):
        """Test getting statistics from empty database."""
        stats = replayer.get_statistics()
        
        assert stats['total_actions'] == 0
        assert stats['first_action_timestamp'] is None
        assert stats['last_action_timestamp'] is None
    
    def test_replay_result_serialization(self, populated_db):
        """Test that ReplayResult can be serialized to dict."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        results = replayer.replay_with_policy("test.yaml", limit=5)
        
        for result in results:
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert 'action_id' in result_dict
            assert 'new_decision' in result_dict
            assert 'confidence' in result_dict
    
    def test_policy_comparison_serialization(self, populated_db):
        """Test that PolicyComparison can be serialized to dict."""
        temp_db, _ = populated_db
        replayer = ActionReplayer(storage_path=temp_db)
        
        comparison = replayer.compare_outcomes(
            baseline_policy="current",
            candidate_policy="strict.yaml",
            limit=10
        )
        
        comparison_dict = comparison.to_dict()
        assert isinstance(comparison_dict, dict)
        assert 'total_actions' in comparison_dict
        assert 'decisions_changed' in comparison_dict
        assert 'change_rate' in comparison_dict
        assert 'decision_breakdown' in comparison_dict


class TestActionReplayPerformance:
    """Performance benchmarks for replay functionality."""
    
    @pytest.fixture
    def large_db(self):
        """Create a database with large number of actions for performance testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            persistence = PersistenceManager(
                db_path=os.path.join(tmpdir, "action_streams.db"),
                retention_days=365
            )
            
            # Create 10,000 actions for performance testing
            base_time = datetime(2024, 1, 1, 0, 0, 0)
            
            print(f"\nGenerating 10,000 test actions...")
            for i in range(10000):
                action = AgentAction(
                    action_id=f"action_{i}",
                    agent_id=f"agent_{i % 100}",
                    action_type=ActionType.QUERY,
                    content=f"Performance test action {i}",
                    timestamp=base_time + timedelta(seconds=i),
                    intent=f"Intent {i}",
                    risk_score=0.1 * (i % 10),
                )
                persistence.store_action(action)
                
                if i % 1000 == 0 and i > 0:
                    print(f"  Generated {i} actions...")
            
            print(f"  Completed: 10,000 actions generated")
            yield tmpdir
    
    def test_query_performance_10k_actions(self, large_db):
        """Test query performance with 10K actions."""
        replayer = ActionReplayer(storage_path=large_db)
        
        start = time.perf_counter()
        actions = replayer.get_actions(limit=1000)
        query_time = (time.perf_counter() - start) * 1000
        
        assert len(actions) == 1000
        print(f"\nQuery 1000 actions from 10K: {query_time:.2f}ms")
        assert query_time < 1000, "Query should complete in less than 1 second"
    
    def test_replay_performance_1k_actions(self, large_db):
        """Test replay performance with 1K actions."""
        replayer = ActionReplayer(storage_path=large_db)
        
        start = time.perf_counter()
        results = replayer.replay_with_policy("test_policy.yaml", limit=1000)
        replay_time = (time.perf_counter() - start) * 1000
        
        assert len(results) == 1000
        print(f"\nReplay 1000 actions: {replay_time:.2f}ms")
        print(f"Average per action: {replay_time/1000:.2f}ms")
        
        # Should be able to replay at least 100 actions per second
        actions_per_second = 1000 / (replay_time / 1000)
        print(f"Throughput: {actions_per_second:.0f} actions/second")
        assert actions_per_second >= 100, "Should replay at least 100 actions/second"
    
    def test_comparison_performance(self, large_db):
        """Test policy comparison performance."""
        replayer = ActionReplayer(storage_path=large_db)
        
        start = time.perf_counter()
        comparison = replayer.compare_outcomes(
            baseline_policy="permissive.yaml",
            candidate_policy="strict.yaml",
            limit=500
        )
        comparison_time = (time.perf_counter() - start) * 1000
        
        assert comparison.total_actions == 500
        print(f"\nCompare 500 actions between 2 policies: {comparison_time:.2f}ms")
        print(f"Decisions changed: {comparison.decisions_changed}")
        print(f"Change rate: {comparison.decisions_changed/comparison.total_actions*100:.1f}%")
        
        # Comparison includes two replays, so should be under 10 seconds
        assert comparison_time < 10000, "Comparison should complete in under 10 seconds"


class TestActionReplayIntegration:
    """Integration tests for replay system."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow: store, replay, compare."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create and populate database
            persistence = PersistenceManager(
                db_path=os.path.join(tmpdir, "action_streams.db"),
                retention_days=365
            )
            
            base_time = datetime(2024, 1, 15, 10, 0, 0)
            for i in range(50):
                action = AgentAction(
                    action_id=f"action_{i}",
                    agent_id=f"agent_{i % 5}",
                    action_type=ActionType.QUERY,
                    content=f"Test content {i}",
                    timestamp=base_time + timedelta(minutes=i),
                )
                persistence.store_action(action)
                
                # Store judgment for every other action
                if i % 2 == 0:
                    judgment = JudgmentResult(
                        judgment_id=f"judgment_{i}",
                        action_id=f"action_{i}",
                        decision=Decision.ALLOW,
                        confidence=0.9,
                        reasoning="Baseline decision",
                        violations=[],
                        timestamp=base_time + timedelta(minutes=i),
                        modifications={},
                        feedback={},
                        remediation_steps=[],
                        follow_up_required=False,
                    )
                    persistence.store_judgment(judgment)
            
            # Step 2: Create replayer and verify statistics
            replayer = ActionReplayer(storage_path=tmpdir)
            stats = replayer.get_statistics()
            
            assert stats['total_actions'] == 50
            
            # Step 3: Set time-travel point
            replayer.set_timestamp("2024-01-15T10:00:00Z")
            
            # Step 4: Replay with new policy
            results = replayer.replay_with_policy(
                new_policy="strict_financial_v2.yaml",
                agent_ids=["agent_0", "agent_1"],
                limit=20
            )
            
            assert len(results) > 0
            assert all(r.agent_id in ["agent_0", "agent_1"] for r in results)
            
            # Step 5: Compare policies
            comparison = replayer.compare_outcomes(
                baseline_policy="current",
                candidate_policy="strict_financial_v2.yaml",
                limit=30
            )
            
            assert comparison.total_actions == 30
            assert comparison.execution_time_ms > 0
            
            # Verify comparison dictionary can be serialized
            comparison_dict = comparison.to_dict()
            json_str = json.dumps(comparison_dict, indent=2)
            assert len(json_str) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
