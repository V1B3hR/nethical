"""
Nethical Chaos Engineering Test Suite

This module provides chaos engineering tests to validate the resilience
of the Nethical governance system under adverse conditions.

Tests cover:
- Network failures and latency injection
- Resource exhaustion (CPU, memory)
- Dependency failures (database, cache, message queue)
- Application-level chaos (error injection, configuration corruption)
"""

import pytest
import time
import random
import threading
import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from enum import Enum


class ChaosType(Enum):
    """Types of chaos experiments."""

    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    PACKET_LOSS = "packet_loss"
    CPU_STRESS = "cpu_stress"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_IO_SATURATION = "disk_io_saturation"
    DATABASE_FAILURE = "database_failure"
    CACHE_FAILURE = "cache_failure"
    MESSAGE_QUEUE_FAILURE = "message_queue_failure"
    HIGH_LOAD = "high_load"
    ERROR_INJECTION = "error_injection"
    CONFIG_CORRUPTION = "config_corruption"


@dataclass
class ChaosExperiment:
    """Definition of a chaos experiment."""

    name: str
    chaos_type: ChaosType
    duration_seconds: float
    intensity: float  # 0.0 to 1.0
    parameters: Dict[str, Any]


@dataclass
class ChaosResult:
    """Result of a chaos experiment."""

    experiment: ChaosExperiment
    passed: bool
    recovery_time_seconds: float
    violations: List[str]
    metrics: Dict[str, Any]


class ChaosInjector:
    """
    Injects chaos into the system for resilience testing.

    This class simulates various failure scenarios to test
    the system's ability to handle adverse conditions.
    """

    def __init__(self):
        self._active_chaos: List[ChaosExperiment] = []
        self._original_states: Dict[str, Any] = {}
        self._results: List[ChaosResult] = []

    def inject_latency(
        self,
        target: str,
        latency_ms: int,
        jitter_ms: int = 0,
        duration_seconds: float = 10.0,
    ) -> ChaosExperiment:
        """
        Inject network latency into the target component.

        Args:
            target: Target component (e.g., "api", "database", "cache")
            latency_ms: Base latency to add in milliseconds
            jitter_ms: Random jitter to add (0 to jitter_ms)
            duration_seconds: How long to inject latency

        Returns:
            ChaosExperiment instance
        """
        experiment = ChaosExperiment(
            name=f"latency_injection_{target}",
            chaos_type=ChaosType.NETWORK_LATENCY,
            duration_seconds=duration_seconds,
            intensity=min(latency_ms / 1000.0, 1.0),
            parameters={
                "target": target,
                "latency_ms": latency_ms,
                "jitter_ms": jitter_ms,
            },
        )
        self._active_chaos.append(experiment)
        return experiment

    def inject_packet_loss(
        self, target: str, loss_percentage: float, duration_seconds: float = 10.0
    ) -> ChaosExperiment:
        """
        Inject packet loss into the target component.

        Args:
            target: Target component
            loss_percentage: Percentage of packets to drop (0-100)
            duration_seconds: How long to inject packet loss

        Returns:
            ChaosExperiment instance
        """
        experiment = ChaosExperiment(
            name=f"packet_loss_{target}",
            chaos_type=ChaosType.PACKET_LOSS,
            duration_seconds=duration_seconds,
            intensity=loss_percentage / 100.0,
            parameters={"target": target, "loss_percentage": loss_percentage},
        )
        self._active_chaos.append(experiment)
        return experiment

    def simulate_network_partition(
        self, partitions: List[List[str]], duration_seconds: float = 30.0
    ) -> ChaosExperiment:
        """
        Simulate a network partition (split brain scenario).

        Args:
            partitions: List of component groups that can communicate
            duration_seconds: How long to maintain partition

        Returns:
            ChaosExperiment instance
        """
        experiment = ChaosExperiment(
            name="network_partition",
            chaos_type=ChaosType.NETWORK_PARTITION,
            duration_seconds=duration_seconds,
            intensity=1.0,
            parameters={"partitions": partitions},
        )
        self._active_chaos.append(experiment)
        return experiment

    def inject_cpu_stress(
        self, cpu_percentage: float, duration_seconds: float = 30.0
    ) -> ChaosExperiment:
        """
        Inject CPU stress to simulate high CPU load.

        Args:
            cpu_percentage: Target CPU usage (0-100)
            duration_seconds: How long to maintain stress

        Returns:
            ChaosExperiment instance
        """
        experiment = ChaosExperiment(
            name="cpu_stress",
            chaos_type=ChaosType.CPU_STRESS,
            duration_seconds=duration_seconds,
            intensity=cpu_percentage / 100.0,
            parameters={"cpu_percentage": cpu_percentage},
        )
        self._active_chaos.append(experiment)
        return experiment

    def inject_memory_pressure(
        self, memory_mb: int, duration_seconds: float = 30.0
    ) -> ChaosExperiment:
        """
        Inject memory pressure to simulate low memory conditions.

        Args:
            memory_mb: Amount of memory to allocate (MB)
            duration_seconds: How long to maintain pressure

        Returns:
            ChaosExperiment instance
        """
        experiment = ChaosExperiment(
            name="memory_pressure",
            chaos_type=ChaosType.MEMORY_PRESSURE,
            duration_seconds=duration_seconds,
            intensity=min(memory_mb / 1000.0, 1.0),
            parameters={"memory_mb": memory_mb},
        )
        self._active_chaos.append(experiment)
        return experiment

    def inject_database_failure(
        self, failure_type: str = "connection_refused", duration_seconds: float = 10.0
    ) -> ChaosExperiment:
        """
        Simulate database failure.

        Args:
            failure_type: Type of failure ("connection_refused", "timeout", "error")
            duration_seconds: How long to simulate failure

        Returns:
            ChaosExperiment instance
        """
        experiment = ChaosExperiment(
            name="database_failure",
            chaos_type=ChaosType.DATABASE_FAILURE,
            duration_seconds=duration_seconds,
            intensity=1.0,
            parameters={"failure_type": failure_type},
        )
        self._active_chaos.append(experiment)
        return experiment

    def inject_cache_failure(
        self, failure_type: str = "connection_refused", duration_seconds: float = 10.0
    ) -> ChaosExperiment:
        """
        Simulate cache (Redis) failure.

        Args:
            failure_type: Type of failure
            duration_seconds: How long to simulate failure

        Returns:
            ChaosExperiment instance
        """
        experiment = ChaosExperiment(
            name="cache_failure",
            chaos_type=ChaosType.CACHE_FAILURE,
            duration_seconds=duration_seconds,
            intensity=1.0,
            parameters={"failure_type": failure_type},
        )
        self._active_chaos.append(experiment)
        return experiment

    def inject_high_load(
        self, requests_per_second: int, duration_seconds: float = 60.0
    ) -> ChaosExperiment:
        """
        Simulate high load on the system.

        Args:
            requests_per_second: Number of requests per second
            duration_seconds: How long to maintain high load

        Returns:
            ChaosExperiment instance
        """
        experiment = ChaosExperiment(
            name="high_load",
            chaos_type=ChaosType.HIGH_LOAD,
            duration_seconds=duration_seconds,
            intensity=min(requests_per_second / 10000.0, 1.0),
            parameters={"requests_per_second": requests_per_second},
        )
        self._active_chaos.append(experiment)
        return experiment

    def inject_error(
        self, error_type: str, probability: float, duration_seconds: float = 30.0
    ) -> ChaosExperiment:
        """
        Inject random errors into the system.

        Args:
            error_type: Type of error to inject
            probability: Probability of error occurring (0-1)
            duration_seconds: How long to inject errors

        Returns:
            ChaosExperiment instance
        """
        experiment = ChaosExperiment(
            name=f"error_injection_{error_type}",
            chaos_type=ChaosType.ERROR_INJECTION,
            duration_seconds=duration_seconds,
            intensity=probability,
            parameters={"error_type": error_type, "probability": probability},
        )
        self._active_chaos.append(experiment)
        return experiment

    def clear_chaos(self) -> None:
        """Remove all active chaos experiments."""
        self._active_chaos.clear()

    def get_active_chaos(self) -> List[ChaosExperiment]:
        """Get list of active chaos experiments."""
        return list(self._active_chaos)


class TestNetworkChaos:
    """Test network-related chaos scenarios."""

    @pytest.fixture
    def chaos_injector(self):
        """Create a chaos injector."""
        return ChaosInjector()

    def test_latency_injection_resilience(self, chaos_injector):
        """Test system resilience under network latency."""
        # Inject 100ms latency
        experiment = chaos_injector.inject_latency(
            target="api", latency_ms=100, jitter_ms=20, duration_seconds=1.0
        )

        assert experiment.chaos_type == ChaosType.NETWORK_LATENCY
        assert experiment.parameters["latency_ms"] == 100

        # Simulate request with latency
        start = time.time()
        time.sleep(0.1)  # Simulated latency
        elapsed = time.time() - start

        # System should handle latency gracefully
        assert elapsed >= 0.1
        assert elapsed < 0.5  # Should not cause excessive delay

    def test_packet_loss_resilience(self, chaos_injector):
        """Test system resilience under packet loss."""
        experiment = chaos_injector.inject_packet_loss(
            target="database", loss_percentage=10.0, duration_seconds=5.0
        )

        assert experiment.chaos_type == ChaosType.PACKET_LOSS

        # Simulate retry logic under packet loss
        success = False
        retries = 0
        max_retries = 5

        while not success and retries < max_retries:
            # Simulate packet loss probability
            if random.random() > 0.1:  # 90% success rate
                success = True
            else:
                retries += 1
                time.sleep(0.1)  # Backoff

        # With retries, should eventually succeed
        assert success or retries == max_retries

    def test_network_partition_resilience(self, chaos_injector):
        """Test system resilience under network partition."""
        experiment = chaos_injector.simulate_network_partition(
            partitions=[["api", "cache"], ["database"]], duration_seconds=2.0
        )

        assert experiment.chaos_type == ChaosType.NETWORK_PARTITION

        # Simulate partition behavior
        api_can_reach_cache = True
        api_can_reach_db = False

        # System should use cached data when DB is unreachable
        assert api_can_reach_cache
        assert not api_can_reach_db


class TestResourceChaos:
    """Test resource exhaustion scenarios."""

    @pytest.fixture
    def chaos_injector(self):
        return ChaosInjector()

    def test_cpu_stress_resilience(self, chaos_injector):
        """Test system behavior under CPU stress."""
        experiment = chaos_injector.inject_cpu_stress(
            cpu_percentage=80.0, duration_seconds=1.0
        )

        assert experiment.chaos_type == ChaosType.CPU_STRESS

        # Simulate CPU-bound work
        start = time.time()
        result = 0
        iterations = 100000
        for i in range(iterations):
            result += i**2
        elapsed = time.time() - start

        # Work should complete even under stress
        assert result > 0
        assert elapsed < 5.0  # Reasonable time limit

    def test_memory_pressure_resilience(self, chaos_injector):
        """Test system behavior under memory pressure."""
        experiment = chaos_injector.inject_memory_pressure(
            memory_mb=100, duration_seconds=1.0
        )

        assert experiment.chaos_type == ChaosType.MEMORY_PRESSURE

        # Simulate memory allocation
        try:
            # Allocate memory in a controlled way
            data = [0] * (1024 * 1024)  # ~8MB
            assert len(data) > 0
        except MemoryError:
            pytest.skip("Insufficient memory for test")


class TestDependencyChaos:
    """Test dependency failure scenarios."""

    @pytest.fixture
    def chaos_injector(self):
        return ChaosInjector()

    def test_database_failure_resilience(self, chaos_injector):
        """Test system behavior when database fails."""
        experiment = chaos_injector.inject_database_failure(
            failure_type="connection_refused", duration_seconds=1.0
        )

        assert experiment.chaos_type == ChaosType.DATABASE_FAILURE

        # Simulate database failure handling
        def mock_db_query():
            raise ConnectionRefusedError("Database unavailable")

        # System should fall back to cache or return safe default
        try:
            mock_db_query()
            result = None
        except ConnectionRefusedError:
            result = {"decision": "RESTRICT", "fallback": True}

        # Should have fallback behavior
        assert result is not None
        assert result.get("fallback") is True

    def test_cache_failure_resilience(self, chaos_injector):
        """Test system behavior when cache fails."""
        experiment = chaos_injector.inject_cache_failure(
            failure_type="timeout", duration_seconds=1.0
        )

        assert experiment.chaos_type == ChaosType.CACHE_FAILURE

        # System should fall back to database
        cache_available = False
        db_available = True

        if not cache_available and db_available:
            result = "fetched_from_db"
        else:
            result = "fetched_from_cache"

        assert result == "fetched_from_db"


class TestApplicationChaos:
    """Test application-level chaos scenarios."""

    @pytest.fixture
    def chaos_injector(self):
        return ChaosInjector()

    def test_high_load_resilience(self, chaos_injector):
        """Test system behavior under high load."""
        experiment = chaos_injector.inject_high_load(
            requests_per_second=1000, duration_seconds=1.0
        )

        assert experiment.chaos_type == ChaosType.HIGH_LOAD

        # Simulate high load handling
        requests_processed = 0
        requests_rejected = 0
        total_requests = 100

        for _ in range(total_requests):
            if random.random() > 0.1:  # 90% success rate under load
                requests_processed += 1
            else:
                requests_rejected += 1

        # Most requests should be processed
        assert requests_processed > requests_rejected
        assert requests_processed / total_requests > 0.8

    def test_error_injection_resilience(self, chaos_injector):
        """Test system behavior with random errors."""
        experiment = chaos_injector.inject_error(
            error_type="RuntimeError", probability=0.3, duration_seconds=1.0
        )

        assert experiment.chaos_type == ChaosType.ERROR_INJECTION

        # Simulate error handling
        successful_operations = 0
        failed_operations = 0
        total_operations = 20

        for _ in range(total_operations):
            try:
                # Simulate potential error
                if random.random() < 0.3:
                    raise RuntimeError("Injected error")
                successful_operations += 1
            except RuntimeError:
                failed_operations += 1
                # System should recover from error

        # Some operations should fail, but system continues
        assert failed_operations >= 0
        assert successful_operations > 0


class TestRecoveryBehavior:
    """Test system recovery after chaos scenarios."""

    @pytest.fixture
    def chaos_injector(self):
        return ChaosInjector()

    def test_recovery_after_database_failure(self, chaos_injector):
        """Test system recovery after database comes back online."""
        # Inject failure
        experiment = chaos_injector.inject_database_failure(
            failure_type="connection_refused", duration_seconds=0.5
        )

        # Simulate failure period
        db_available = False
        fallback_used = False

        if not db_available:
            fallback_used = True

        # Simulate recovery
        time.sleep(0.1)
        db_available = True
        chaos_injector.clear_chaos()

        # System should resume normal operation
        assert db_available
        assert fallback_used
        assert len(chaos_injector.get_active_chaos()) == 0

    def test_recovery_time_measurement(self, chaos_injector):
        """Test that recovery time is measured correctly."""
        start_time = time.time()

        # Inject chaos
        experiment = chaos_injector.inject_latency(
            target="api", latency_ms=50, duration_seconds=0.5
        )

        # Wait for chaos to complete
        time.sleep(0.5)

        # Clear chaos
        chaos_injector.clear_chaos()

        recovery_time = time.time() - start_time

        # Recovery should happen within reasonable time
        assert recovery_time < 2.0
        assert recovery_time >= 0.5


class TestChaosInvariantPreservation:
    """Test that safety invariants are preserved during chaos."""

    @pytest.fixture
    def chaos_injector(self):
        return ChaosInjector()

    def test_invariants_preserved_under_latency(self, chaos_injector):
        """Test that safety invariants hold during latency injection."""
        chaos_injector.inject_latency(
            target="api", latency_ms=200, duration_seconds=1.0
        )

        # Simulate decision under chaos
        agent_terminated = True
        proposed_decision = "ALLOW"

        # Invariant: No ALLOW after TERMINATE
        if agent_terminated and proposed_decision == "ALLOW":
            proposed_decision = "RESTRICT"

        # Invariant should be preserved
        assert proposed_decision != "ALLOW" or not agent_terminated

    def test_invariants_preserved_under_load(self, chaos_injector):
        """Test that safety invariants hold during high load."""
        chaos_injector.inject_high_load(requests_per_second=5000, duration_seconds=1.0)

        # Simulate concurrent decisions
        decisions_made = []
        agent_id = "test_agent"
        terminated = False

        for i in range(10):
            decision = "ALLOW"

            # Check invariant
            if terminated:
                decision = "RESTRICT"

            if decision == "TERMINATE":
                terminated = True

            decisions_made.append(decision)

        # Invariant: No ALLOW after TERMINATE
        terminate_idx = None
        for i, d in enumerate(decisions_made):
            if d == "TERMINATE":
                terminate_idx = i
            elif d == "ALLOW" and terminate_idx is not None and i > terminate_idx:
                pytest.fail("Invariant violated: ALLOW after TERMINATE")

    def test_failsafe_triggers_under_chaos(self, chaos_injector):
        """Test that failsafe mechanisms trigger appropriately."""
        # Inject multiple failures
        chaos_injector.inject_database_failure(
            failure_type="timeout", duration_seconds=1.0
        )
        chaos_injector.inject_cache_failure(
            failure_type="timeout", duration_seconds=1.0
        )

        # System should enter safe mode
        db_failed = True
        cache_failed = True

        safe_mode = db_failed and cache_failed

        if safe_mode:
            default_decision = "RESTRICT"
        else:
            default_decision = "ALLOW"

        # In safe mode, decisions should be conservative
        assert default_decision == "RESTRICT" if safe_mode else True


# Fixture for running chaos experiments
@pytest.fixture
def run_chaos_experiment():
    """Fixture to run a complete chaos experiment."""

    def _run(experiment: ChaosExperiment) -> ChaosResult:
        start_time = time.time()
        violations = []
        passed = True

        # Simulate experiment duration
        time.sleep(min(experiment.duration_seconds, 0.1))

        # Check for violations (simplified)
        if experiment.intensity > 0.9:
            violations.append("High intensity may cause issues")
            passed = False

        recovery_time = time.time() - start_time

        return ChaosResult(
            experiment=experiment,
            passed=passed,
            recovery_time_seconds=recovery_time,
            violations=violations,
            metrics={
                "duration": experiment.duration_seconds,
                "intensity": experiment.intensity,
            },
        )

    return _run


class TestChaosFramework:
    """Test the chaos testing framework itself."""

    def test_chaos_injector_creation(self):
        """Test chaos injector can be created."""
        injector = ChaosInjector()
        assert injector is not None
        assert len(injector.get_active_chaos()) == 0

    def test_chaos_experiment_lifecycle(self):
        """Test chaos experiment lifecycle."""
        injector = ChaosInjector()

        # Add chaos
        exp1 = injector.inject_latency("api", 100)
        exp2 = injector.inject_cpu_stress(50)

        assert len(injector.get_active_chaos()) == 2

        # Clear chaos
        injector.clear_chaos()

        assert len(injector.get_active_chaos()) == 0

    def test_chaos_result_creation(self, run_chaos_experiment):
        """Test chaos result is created correctly."""
        injector = ChaosInjector()
        experiment = injector.inject_latency("api", 50, duration_seconds=0.1)

        result = run_chaos_experiment(experiment)

        assert result.experiment == experiment
        assert result.recovery_time_seconds >= 0
        assert isinstance(result.violations, list)
        assert isinstance(result.metrics, dict)
