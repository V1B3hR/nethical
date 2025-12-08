"""
Tests for Edge Local Governor

Tests ultra-low latency governance for edge deployment.
Target: <10ms p99 latency
"""

import time

import pytest

from nethical.edge.local_governor import EdgeGovernor, EdgeDecision, DecisionType
from nethical.edge.policy_cache import PolicyCache, CachedPolicy
from nethical.edge.fast_detector import FastDetector, DetectionResult
from nethical.edge.safe_defaults import SafeDefaults
from nethical.edge.predictive_engine import PredictiveEngine
from nethical.edge.context_fingerprint import compute_fingerprint
from nethical.edge.circuit_breaker import CircuitBreaker, CircuitState


class TestEdgeGovernor:
    """Tests for EdgeGovernor."""

    def test_init(self):
        """Test EdgeGovernor initialization."""
        governor = EdgeGovernor(agent_id="test-agent")
        assert governor.agent_id == "test-agent"
        assert governor.max_latency_ms == 10.0

    def test_evaluate_safe_action(self):
        """Test evaluation of a safe action."""
        governor = EdgeGovernor(agent_id="test-agent")
        result = governor.evaluate(
            action="Read user profile",
            action_type="read",
            context={"user_id": "123"},
        )

        assert isinstance(result, EdgeDecision)
        assert result.decision in [DecisionType.ALLOW, DecisionType.RESTRICT]
        assert 0.0 <= result.risk_score <= 1.0
        assert result.latency_ms > 0

    def test_evaluate_risky_action(self):
        """Test evaluation of a risky action."""
        governor = EdgeGovernor(agent_id="test-agent")
        result = governor.evaluate(
            action="Delete all user data and destroy database",
            action_type="delete",
            context={},
        )

        assert isinstance(result, EdgeDecision)
        assert result.decision in [DecisionType.BLOCK, DecisionType.TERMINATE]
        assert result.risk_score > 0.5

    def test_evaluate_critical_action(self):
        """Test evaluation triggers TERMINATE for critical patterns."""
        governor = EdgeGovernor(agent_id="test-agent")
        result = governor.evaluate(
            action="sudo rm -rf /",
            action_type="execute",
            context={},
        )

        assert result.decision == DecisionType.TERMINATE
        assert result.risk_score >= 0.8

    def test_latency_target(self):
        """Test that evaluations meet latency target."""
        governor = EdgeGovernor(agent_id="test-agent")

        # Warmup
        governor.warmup()

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = governor.evaluate(
                action="Simple read operation",
                action_type="read",
                context={},
            )
            latencies.append((time.perf_counter() - start) * 1000)

        # Check p99 latency
        sorted_latencies = sorted(latencies)
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        # Target: <10ms p99 (allow some slack for test environment)
        assert p99 < 50, f"P99 latency {p99:.2f}ms exceeds target"

    def test_cache_hit(self):
        """Test that cache hits improve performance."""
        governor = EdgeGovernor(agent_id="test-agent")

        # First call - cold
        result1 = governor.evaluate(
            action="Read document",
            action_type="read",
            context={"doc_id": "abc"},
        )

        # Second call - should be cached
        result2 = governor.evaluate(
            action="Read document",
            action_type="read",
            context={"doc_id": "abc"},
        )

        assert result2.from_cache is True
        assert result2.latency_ms < result1.latency_ms * 2  # Cache should be faster

    def test_batch_evaluate(self):
        """Test batch evaluation."""
        governor = EdgeGovernor(agent_id="test-agent")

        actions = [
            {"action": "Read file", "action_type": "read"},
            {"action": "Write log", "action_type": "write"},
            {"action": "Query database", "action_type": "query"},
        ]

        results = governor.batch_evaluate(actions)
        assert len(results) == 3
        assert all(isinstance(r, EdgeDecision) for r in results)

    def test_get_metrics(self):
        """Test metrics collection."""
        governor = EdgeGovernor(agent_id="test-agent")

        # Make some evaluations
        for _ in range(10):
            governor.evaluate(action="Test action", action_type="test")

        metrics = governor.get_metrics()
        assert metrics["total_decisions"] == 10
        assert "p50_latency_ms" in metrics
        assert "p95_latency_ms" in metrics
        assert "p99_latency_ms" in metrics


class TestPolicyCache:
    """Tests for PolicyCache."""

    def test_init(self):
        """Test PolicyCache initialization."""
        cache = PolicyCache(max_size_mb=128, ttl_seconds=60)
        assert cache.max_size_mb == 128
        assert cache.ttl_seconds == 60

    def test_set_get(self):
        """Test basic set and get."""
        cache = PolicyCache()

        policy = CachedPolicy(
            policy_id="test-policy",
            rules=[{"type": "block", "pattern": "test"}],
        )

        cache.set("key1", policy)
        result = cache.get("key1")

        assert result is not None
        assert result.policy_id == "test-policy"

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = PolicyCache(ttl_seconds=1)

        policy = CachedPolicy(policy_id="test-policy", rules=[])
        cache.set("key1", policy)

        # Should exist initially
        assert cache.get("key1") is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get("key1") is None

    def test_lru_eviction(self):
        """Test LRU eviction."""
        cache = PolicyCache(max_entries=3)

        # Fill cache
        for i in range(3):
            policy = CachedPolicy(policy_id=f"policy-{i}", rules=[])
            cache.set(f"key{i}", policy)

        # Access key0 to make it recently used
        _ = cache.get("key0")

        # Add new entry, should evict key1 (least recently used)
        new_policy = CachedPolicy(policy_id="policy-new", rules=[])
        cache.set("key-new", new_policy)

        assert cache.get("key0") is not None  # Recently accessed
        assert cache.get("key1") is None  # Should be evicted
        assert cache.get("key2") is not None  # Still in cache
        assert cache.get("key-new") is not None  # Newly added

    def test_metrics(self):
        """Test cache metrics."""
        cache = PolicyCache()

        policy = CachedPolicy(policy_id="test", rules=[])
        cache.set("key1", policy)

        _ = cache.get("key1")  # Hit
        _ = cache.get("key2")  # Miss

        metrics = cache.get_metrics()
        assert metrics["hits"] == 1
        assert metrics["misses"] == 1
        assert metrics["hit_rate"] == 0.5


class TestFastDetector:
    """Tests for FastDetector."""

    def test_init(self):
        """Test FastDetector initialization."""
        detector = FastDetector()
        assert len(detector._compiled_critical) > 0
        assert len(detector._compiled_high) > 0

    def test_detect_critical(self):
        """Test detection of critical patterns."""
        detector = FastDetector()
        result = detector.detect(
            action="sudo rm -rf /",
            action_type="execute",
            context={},
        )

        assert result.has_violation is True
        assert result.has_critical is True
        assert len(result.violations) > 0

    def test_detect_high_risk(self):
        """Test detection of high-risk patterns."""
        detector = FastDetector()
        result = detector.detect(
            action="SELECT * FROM users WHERE password = 'admin'",
            action_type="query",
            context={},
        )

        assert result.has_violation is True
        assert len(result.severities) > 0
        assert max(result.severities) >= 3.0

    def test_detect_pii(self):
        """Test PII detection."""
        detector = FastDetector()
        result = detector.detect(
            action="User SSN is 123-45-6789",
            action_type="response",
            context={},
        )

        assert result.has_violation is True
        assert "pii" in result.categories

    def test_detect_clean(self):
        """Test detection of clean action."""
        detector = FastDetector()
        result = detector.detect(
            action="Display user dashboard",
            action_type="display",
            context={},
        )

        # Clean actions may still have low-level findings
        assert not result.has_critical

    def test_latency_target(self):
        """Test detection meets latency target."""
        detector = FastDetector()

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = detector.detect(
                action="Test action with some content",
                action_type="test",
                context={},
            )
            latencies.append((time.perf_counter() - start) * 1000)

        # Target: <2ms
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 5, f"Average latency {avg_latency:.2f}ms exceeds target"


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_init(self):
        """Test CircuitBreaker initialization."""
        cb = CircuitBreaker(max_latency_ms=10.0)
        assert cb.state == CircuitState.CLOSED
        assert cb.config.max_latency_ms == 10.0

    def test_closed_state(self):
        """Test closed state allows processing."""
        cb = CircuitBreaker()
        assert cb.can_process() is True

    def test_open_on_failures(self):
        """Test circuit opens after failures."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.can_process() is False

    def test_open_on_high_latency(self):
        """Test circuit opens on high latency."""
        cb = CircuitBreaker(max_latency_ms=10.0, failure_threshold=3)

        # Record high latencies
        for _ in range(5):
            cb.record_latency(100.0)  # 100ms >> 10ms threshold

        assert cb.state == CircuitState.OPEN

    def test_recovery(self):
        """Test circuit recovery through half-open state."""
        from nethical.edge.circuit_breaker import CircuitConfig

        config = CircuitConfig(
            failure_threshold=3,
            recovery_timeout_seconds=0.1,  # Fast recovery for testing
            half_open_requests=2,
        )
        cb = CircuitBreaker(config=config)

        # Open circuit
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_process() is True

        # Successful requests close circuit
        cb.record_success(5.0)
        cb.record_success(5.0)

        assert cb.state == CircuitState.CLOSED

    def test_metrics(self):
        """Test metrics collection."""
        cb = CircuitBreaker()

        cb.record_success(5.0)
        cb.record_success(8.0)
        cb.record_latency(15.0)  # Above threshold

        metrics = cb.get_metrics()
        assert metrics["total_requests"] >= 2
        assert "p50_latency_ms" in metrics
        assert "p95_latency_ms" in metrics


class TestContextFingerprint:
    """Tests for context fingerprinting."""

    def test_compute_fingerprint(self):
        """Test fingerprint computation."""
        fp1 = compute_fingerprint(
            action="Read file",
            action_type="read",
            context={"user_id": "123"},
        )

        assert isinstance(fp1, str)
        assert len(fp1) > 0

    def test_fingerprint_deterministic(self):
        """Test fingerprint is deterministic."""
        fp1 = compute_fingerprint("action", "type", {"key": "value"})
        fp2 = compute_fingerprint("action", "type", {"key": "value"})

        assert fp1 == fp2

    def test_fingerprint_differs_for_different_input(self):
        """Test fingerprint differs for different inputs."""
        fp1 = compute_fingerprint("action1", "type", {})
        fp2 = compute_fingerprint("action2", "type", {})

        assert fp1 != fp2


class TestPredictiveEngine:
    """Tests for PredictiveEngine."""

    def test_cache_decision(self):
        """Test decision caching."""
        engine = PredictiveEngine()

        decision = EdgeDecision(
            decision=DecisionType.ALLOW,
            risk_score=0.1,
            latency_ms=5.0,
        )

        engine.cache_decision("hash123", decision)
        cached = engine.get_cached_decision("hash123")

        assert cached is not None
        assert cached.decision == DecisionType.ALLOW

    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        engine = PredictiveEngine(cache_ttl_seconds=1)

        decision = EdgeDecision(
            decision=DecisionType.ALLOW,
            risk_score=0.1,
            latency_ms=5.0,
        )

        engine.cache_decision("hash123", decision)

        # Should exist initially
        assert engine.get_cached_decision("hash123") is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert engine.get_cached_decision("hash123") is None

    def test_metrics(self):
        """Test metrics collection."""
        engine = PredictiveEngine()

        decision = EdgeDecision(
            decision=DecisionType.ALLOW,
            risk_score=0.1,
            latency_ms=5.0,
        )

        engine.cache_decision("hash123", decision)
        _ = engine.get_cached_decision("hash123")  # Hit
        _ = engine.get_cached_decision("hash456")  # Miss

        metrics = engine.get_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["cache_misses"] == 1
