"""
Performance Regression Tests

These tests ensure that performance doesn't regress below established benchmarks.
Run with: pytest tests/performance/test_performance_regression.py -v
"""

import pytest
import asyncio
from nethical.core.governance import SafetyGovernance, MonitoringConfig, AgentAction, ActionType
from nethical.performanceprofiling import PerformanceProfiler


# Initialize profiler
profiler = PerformanceProfiler(results_dir="tests/performance/results")


@pytest.fixture
def governance():
    """Create governance instance for testing"""
    config = MonitoringConfig(enable_persistence=False)
    return SafetyGovernance(config)


@pytest.fixture
def sample_action():
    """Create sample action for testing"""
    return AgentAction(
        action_id="test_001",
        agent_id="test_agent",
        action_type=ActionType.QUERY,
        content="This is a test query with some content for evaluation"
    )


class TestGovernancePerformance:
    """Performance tests for governance system"""
    
    def test_evaluate_action_performance(self, governance, sample_action):
        """Test performance of action evaluation"""
        # Benchmark: evaluate_action should complete in reasonable time
        stats = profiler.benchmark_function(
            lambda: asyncio.run(governance.evaluate_action(sample_action)),
            iterations=50
        )
        
        # Assert mean time is under 100ms
        assert stats['mean_ms'] < 100, f"evaluate_action too slow: {stats['mean_ms']:.2f}ms"
        
        # Assert p95 is under 150ms
        assert stats['p95_ms'] < 150, f"evaluate_action p95 too slow: {stats['p95_ms']:.2f}ms"
        
        print(f"✓ evaluate_action performance: mean={stats['mean_ms']:.2f}ms, p95={stats['p95_ms']:.2f}ms")
    
    def test_batch_evaluate_performance(self, governance):
        """Test performance of batch evaluation"""
        actions = [
            AgentAction(
                action_id=f"test_{i}",
                agent_id="test_agent",
                action_type=ActionType.QUERY,
                content=f"Test query {i}"
            )
            for i in range(10)
        ]
        
        stats = profiler.benchmark_function(
            lambda: asyncio.run(governance.batch_evaluate_actions(actions)),
            iterations=10
        )
        
        # Batch of 10 should complete in under 500ms
        assert stats['mean_ms'] < 500, f"batch_evaluate too slow: {stats['mean_ms']:.2f}ms"
        
        print(f"✓ batch_evaluate performance: mean={stats['mean_ms']:.2f}ms for 10 actions")
    
    def test_detector_initialization_performance(self):
        """Test performance of governance initialization"""
        stats = profiler.benchmark_function(
            lambda: SafetyGovernance(MonitoringConfig(enable_persistence=False)),
            iterations=20
        )
        
        # Initialization should be fast
        assert stats['mean_ms'] < 50, f"Initialization too slow: {stats['mean_ms']:.2f}ms"
        
        print(f"✓ Initialization performance: mean={stats['mean_ms']:.2f}ms")


class TestProfilerBaselines:
    """Test and set performance baselines"""
    
    def test_set_baselines(self, governance, sample_action):
        """Set baseline performance metrics for future regression testing"""
        # Measure evaluate_action baseline
        stats = profiler.benchmark_function(
            lambda: asyncio.run(governance.evaluate_action(sample_action)),
            iterations=100
        )
        
        # Set benchmark with 20% tolerance
        profiler.set_benchmark(
            name="evaluate_action",
            baseline_ms=stats['median_ms'],
            tolerance_pct=20.0
        )
        
        print(f"✓ Set baseline for evaluate_action: {stats['median_ms']:.2f}ms")
        
        # Measure initialization baseline
        init_stats = profiler.benchmark_function(
            lambda: SafetyGovernance(MonitoringConfig(enable_persistence=False)),
            iterations=50
        )
        
        profiler.set_benchmark(
            name="governance_init",
            baseline_ms=init_stats['median_ms'],
            tolerance_pct=20.0
        )
        
        print(f"✓ Set baseline for governance_init: {init_stats['median_ms']:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
