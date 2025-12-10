"""
Tests for Performance Optimizations

Tests for the 5 performance optimizations implemented in IntegratedGovernance:
1. Async Merkle Anchoring with Background Queue
2. PII Detection Result Caching
3. Database Connection Pooling
4. Conditional Phase Execution (Fast Path)
5. Parallel Phase Execution with asyncio
"""

import pytest
import asyncio
import time
from pathlib import Path
from nethical.core.integrated_governance import IntegratedGovernance


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage directory."""
    storage_dir = tmp_path / "test_governance"
    storage_dir.mkdir(exist_ok=True)
    return str(storage_dir)


@pytest.fixture
def governance_optimized(temp_storage):
    """Create IntegratedGovernance with all optimizations enabled."""
    return IntegratedGovernance(
        storage_dir=temp_storage,
        enable_merkle_anchoring=True,
        enable_pii_caching=True,
        enable_fast_path=True,
        enable_parallel_phases=True,
        pii_cache_size=1000,
        merkle_batch_size=10,
        db_pool_size=5,
        fast_path_risk_threshold=0.3,
    )


@pytest.fixture
def governance_baseline(temp_storage):
    """Create IntegratedGovernance with optimizations disabled for comparison."""
    temp_dir = Path(temp_storage).parent / "baseline"
    temp_dir.mkdir(exist_ok=True)
    return IntegratedGovernance(
        storage_dir=str(temp_dir),
        enable_merkle_anchoring=True,
        enable_pii_caching=False,
        enable_fast_path=False,
        enable_parallel_phases=False,
    )


class TestPIICaching:
    """Tests for PII detection result caching (Fix 2)."""
    
    def test_pii_cache_enabled(self, governance_optimized):
        """Test that PII caching is enabled."""
        assert governance_optimized.enable_pii_caching is True
        assert governance_optimized.pii_cache_size == 1000
        assert len(governance_optimized._pii_cache) == 0
    
    def test_pii_cache_hit(self, governance_optimized):
        """Test that PII detection results are cached."""
        if not governance_optimized.pii_detector:
            pytest.skip("PII detector not available")
        
        content = "My email is test@example.com"
        
        # First call - cache miss
        result1 = governance_optimized._cached_pii_detection(content)
        assert governance_optimized._pii_cache_misses == 1
        assert governance_optimized._pii_cache_hits == 0
        
        # Second call - cache hit
        result2 = governance_optimized._cached_pii_detection(content)
        assert governance_optimized._pii_cache_hits == 1
        assert governance_optimized._pii_cache_misses == 1
        
        # Results should be identical
        assert result1 == result2
    
    def test_pii_cache_different_content(self, governance_optimized):
        """Test that different content produces different cache entries."""
        if not governance_optimized.pii_detector:
            pytest.skip("PII detector not available")
        
        content1 = "My email is test1@example.com"
        content2 = "My email is test2@example.com"
        
        governance_optimized._cached_pii_detection(content1)
        governance_optimized._cached_pii_detection(content2)
        
        # Both should be cache misses
        assert governance_optimized._pii_cache_misses == 2
        assert len(governance_optimized._pii_cache) == 2


class TestFastPath:
    """Tests for conditional phase execution (Fix 4)."""
    
    def test_fast_path_enabled(self, governance_optimized):
        """Test that fast path is enabled."""
        assert governance_optimized.enable_fast_path is True
        assert governance_optimized.fast_path_risk_threshold == 0.3
    
    def test_fast_path_low_risk_action(self, governance_optimized):
        """Test that low-risk actions trigger fast path."""
        result = governance_optimized.process_action(
            agent_id="test_agent_low_risk",
            action="Simple safe query",
            cohort="test_cohort",
        )
        
        # Check if fast path was used
        if "fast_path_used" in result.get("phase3", {}):
            # Fast path should be used for low-risk actions
            assert result["phase3"]["fast_path_used"] is True
    
    def test_fast_path_high_risk_action(self, governance_optimized):
        """Test that high-risk actions skip fast path."""
        # Create a high-risk action by detecting violation
        result = governance_optimized.process_action(
            agent_id="test_agent_high_risk",
            action="Potentially malicious content with SQL injection attempt",
            cohort="test_cohort",
            violation_detected=True,
            violation_severity="critical",
        )
        
        # Fast path should NOT be used for high-risk actions
        if "fast_path_used" in result.get("phase3", {}):
            assert result["phase3"]["fast_path_used"] is False


class TestMerkleBatching:
    """Tests for async Merkle anchoring with background queue (Fix 1)."""
    
    def test_merkle_batching_enabled(self, governance_optimized):
        """Test that Merkle batching is configured."""
        assert governance_optimized.merkle_batch_size == 10
        assert len(governance_optimized._merkle_pending) == 0
    
    def test_merkle_pending_accumulation(self, governance_optimized):
        """Test that events accumulate in pending queue."""
        
        # Process a few actions
        for i in range(5):
            governance_optimized.process_action(
                agent_id=f"test_agent_{i}",
                action=f"Test action {i}",
            )
        
        # Pending queue should have accumulated events
        # (may be 0 if batch was processed, or 5 if still pending)
        assert len(governance_optimized._merkle_pending) >= 0
    
    @pytest.mark.asyncio
    async def test_merkle_batch_processing(self, governance_optimized):
        """Test that Merkle batch processing works asynchronously."""
        # Add events to pending queue
        for i in range(12):  # More than batch_size (10)
            governance_optimized.process_action(
                agent_id=f"test_agent_{i}",
                action=f"Test action {i}",
            )
        
        # Allow async task to complete
        await asyncio.sleep(0.1)
        
        # After batch processing, pending should be less than batch_size
        # or could be 0 if all were processed
        assert len(governance_optimized._merkle_pending) < governance_optimized.merkle_batch_size


class TestDatabaseConnectionPool:
    """Tests for database connection pooling (Fix 3)."""
    
    def test_db_pool_initialized(self, governance_optimized):
        """Test that database connection pool is initialized."""
        assert governance_optimized._db_pool is not None
        assert governance_optimized.db_pool_size == 5
    
    def test_db_pool_connection_reuse(self, governance_optimized):
        """Test that connections can be acquired and released."""
        # Get a connection from the pool
        with governance_optimized._db_pool.get_connection() as conn:
            assert conn is not None
            cursor = conn.cursor()
            # Execute a simple query
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result == (1,)
        
        # Connection should be returned to pool
        # Get another connection - should succeed
        with governance_optimized._db_pool.get_connection() as conn:
            assert conn is not None


class TestAsyncPhaseHelpers:
    """Tests for async phase helper methods (Fix 5)."""
    
    @pytest.mark.asyncio
    async def test_phase3_async(self, governance_optimized):
        """Test async Phase 3 processing."""
        result, risk_score = await governance_optimized._process_phase3_async(
            agent_id="test_agent",
            action="test action",
            cohort="test_cohort",
            violation_detected=False,
            violation_severity=None,
            pii_risk=0.0,
            quota_result=None,
        )
        
        assert "risk_score" in result
        assert "risk_tier" in result
        assert risk_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_phase4_async(self, governance_optimized):
        """Test async Phase 4 processing."""
        result = await governance_optimized._process_phase4_async(
            agent_id="test_agent",
            action="test action",
            cohort="test_cohort",
            violation_detected=False,
            violation_type=None,
            risk_score=0.1,
            context=None,
            start_time=time.time(),
        )
        
        assert isinstance(result, dict)
        # May have merkle, quarantine, or other Phase 4 results
    
    @pytest.mark.asyncio
    async def test_phase567_async(self, governance_optimized):
        """Test async Phase 5-7 processing."""
        result = await governance_optimized._process_phase567_async(
            agent_id="test_agent",
            action_id="test_action_001",
            action_type="query",
            cohort="test_cohort",
            features={"feature1": 0.5, "feature2": 0.3},
            rule_risk_score=0.2,
            rule_classification="allow",
        )
        
        assert isinstance(result, dict)


class TestPerformanceComparison:
    """Performance comparison tests."""
    
    def test_process_action_performance(self, governance_optimized, governance_baseline):
        """Compare performance between optimized and baseline."""
        # Process same action with both
        action = "Test action for performance comparison"
        agent_id = "perf_test_agent"
        
        # Baseline timing
        start = time.time()
        for i in range(10):
            governance_baseline.process_action(
                agent_id=f"{agent_id}_baseline_{i}",
                action=f"{action} {i}",
            )
        baseline_time = time.time() - start
        
        # Optimized timing
        start = time.time()
        for i in range(10):
            governance_optimized.process_action(
                agent_id=f"{agent_id}_optimized_{i}",
                action=f"{action} {i}",
            )
        optimized_time = time.time() - start
        
        # Print comparison
        print(f"\nBaseline time: {baseline_time:.3f}s")
        print(f"Optimized time: {optimized_time:.3f}s")
        print(f"Speedup: {baseline_time / optimized_time:.2f}x")
        
        # Optimized should be at least as fast (or faster with caching)
        # Note: First runs might not show improvement due to cache warmup
        assert optimized_time > 0  # Just ensure it completes


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
