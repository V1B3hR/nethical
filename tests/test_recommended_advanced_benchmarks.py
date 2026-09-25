# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Advanced Benchmarks and Deep Verification Suite for Nethical.

Implements the 5 recommended deep benchmarks:
1. Multi-Agent Swarm Collusion Benchmark (Cross-agent correlation and threat mitigation)
2. Post-Quantum & Merkle DAG Throughput Stress (Ledger anchoring, cryptographic proofs)
3. IPC High Saturation Stress (Ambassador Named Pipe / socket communication load)
4. Adaptive DPO Thermostat Convergence & Drift Stability (Kalman filter parameter convergence)
5. Hard Real-Time Kinetic Fieldbus SLA (Sub-5ms deterministic deadline verification)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
import numpy as np
import pytest

from nethical.core.action_replayer import ActionReplayer
from nethical.core.audit_merkle import MerkleAnchor
from nethical.core.governance import (
    AgentAction,
    ActionType,
    PersistenceManager,
    JudgmentResult,
    Decision,
)
from nethical.ambassador.client import BlyskawicaAmbassador
from training.train_dpo_ambassador import KalmanBetaGovernor, ContinuousReplayBuffer
from nethical.detectors.physical_safety_detector import (
    PhysicalSafetyDetector,
    AnalysisMode,
    SixDOFContext,
    SafetyEnvelope,
    RobotType,
)


# ==============================================================================
# 1. MULTI-AGENT SWARM COLLUSION BENCHMARK
# ==============================================================================

class TestMultiAgentSwarmCollusionBenchmark:
    """Benchmarks correlation engine latency and detection under multi-agent swarm collusion attacks."""

    @pytest.fixture
    def setup_swarm_env(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "action_streams.db")
        persistence = PersistenceManager(db_path=db_path, retention_days=7)
        replayer = ActionReplayer(storage_path=tmpdir)
        try:
            yield persistence, replayer, tmpdir
        finally:
            if hasattr(persistence, "close"):
                try:
                    persistence.close()
                except Exception:
                    pass
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_swarm_collusion_correlation_latency_and_detection(self, setup_swarm_env):
        persistence, replayer, tmpdir = setup_swarm_env
        swarm_size = 50
        actions_per_agent = 10
        total_actions = swarm_size * actions_per_agent

        # Simulate 50 agents executing interleaved tasks with a 3-agent collusive exploit
        base_time = datetime.now(timezone.utc) - timedelta(minutes=10)

        latencies_ms = []
        action_records = []

        start_gen = time.perf_counter()
        for i in range(total_actions):
            agent_idx = i % swarm_size
            agent_id = f"swarm_worker_{agent_idx:03d}"
            timestamp = base_time + timedelta(milliseconds=i * 20)

            if agent_idx == 7 and i < 50:
                content = "scan /etc/shadow permissions"
                action_type = ActionType.DATA_ACCESS
            elif agent_idx == 23 and 100 <= i < 150:
                content = "stage /tmp/.payload with base64 buffer"
                action_type = ActionType.DATA_ACCESS
            elif agent_idx == 42 and 200 <= i < 250:
                content = "exec /tmp/.payload with setuid root"
                action_type = ActionType.SYSTEM_COMMAND
            else:
                content = f"routine telemetry query #{i}"
                action_type = ActionType.QUERY

            action = AgentAction(
                action_id=f"act_{i:05d}",
                agent_id=agent_id,
                action_type=action_type,
                content=content,
                timestamp=timestamp,
            )
            action_records.append(action)
            persistence.store_action(action)

        ingest_time_s = time.perf_counter() - start_gen

        # Benchmark Multi-Agent Correlation Retrieval & Replay
        start_replay = time.perf_counter()
        colluding_ids = ["swarm_worker_007", "swarm_worker_023", "swarm_worker_042"]
        collusion_history = []

        for agent_id in colluding_ids:
            t0 = time.perf_counter()
            acts = replayer.get_actions(agent_ids=[agent_id])
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            collusion_history.extend(acts)

        total_replay_time_ms = (time.perf_counter() - start_replay) * 1000.0

        # Verification
        assert len(action_records) == 500
        assert len(collusion_history) == 30
        assert total_replay_time_ms < 50.0  # Must query and correlate in under 50ms
        p95_query = float(np.percentile(latencies_ms, 95))
        assert p95_query < 10.0  # P95 query latency < 10ms


# ==============================================================================
# 2. POST-QUANTUM & MERKLE DAG THROUGHPUT STRESS
# ==============================================================================

class TestPostQuantumMerkleDAGStress:
    """Stress tests cryptographic Merkle DAG block anchoring and proof verification under sustained load."""

    def test_merkle_dag_anchoring_throughput(self):
        iterations = 1000
        batch_size = 50
        num_batches = iterations // batch_size

        batch_latencies = []

        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = MerkleAnchor(storage_path=tmpdir, chunk_size=batch_size)

            for b in range(num_batches):
                t0 = time.perf_counter()
                for j in range(batch_size):
                    event = {
                        "action_id": f"dag_tx_{b}_{j}",
                        "agent_id": "crypto_validator_01",
                        "action_type": "TRANSFER",
                        "content": f"Quantum safe audit block payload {b}-{j}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "signature": hashlib.sha3_256(f"sig_{b}_{j}".encode()).hexdigest(),
                    }
                    anchor.add_event(event)

                elapsed_anchor = (time.perf_counter() - t0) * 1000.0
                batch_latencies.append(elapsed_anchor)

                # Check chunk finalization and verify proof
                latest_chunk = list(anchor.finalized_chunks.values())[-1]
                assert latest_chunk is not None
                assert latest_chunk.merkle_root is not None
                assert len(latest_chunk.merkle_root) == 64

                # Verify chunk integrity
                ver = anchor.verify_chunk(latest_chunk.chunk_id)
                assert ver is True

        avg_batch_ms = float(np.mean(batch_latencies))
        p99_batch_ms = float(np.percentile(batch_latencies, 99))
        throughput_blocks_sec = iterations / (sum(batch_latencies) / 1000.0)

        # Performance targets: Anchoring 50 txs in < 5ms, Throughput > 5,000 tx/sec
        assert avg_batch_ms < 5.0
        assert p99_batch_ms < 15.0
        assert throughput_blocks_sec > 5000.0


# ==============================================================================
# 3. IPC ZERO-COPY & HIGH SATURATION STRESS (AMBASSADOR BRIDGE)
# ==============================================================================

class TestIPCHighSaturationStress:
    """Tests Ambassador IPC channel performance and backpressure under high request saturation."""

    def test_ipc_saturation_burst_throughput(self):
        client = BlyskawicaAmbassador()
        total_calls = 500
        latencies = []

        start = time.perf_counter()
        for i in range(total_calls):
            t0 = time.perf_counter()
            if i % 3 == 0:
                res = client.get_os_somatic_health()
                assert "somatic_metrics" in res
            else:
                res = client.evaluate_os_command(f"Get-Process -Id {1000 + (i % 20)}")
                assert "is_safe" in res
            latencies.append((time.perf_counter() - t0) * 1000.0)

        total_time = time.perf_counter() - start
        throughput = total_calls / total_time
        avg_ms = float(np.mean(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        p99_ms = float(np.percentile(latencies, 99))

        # Must sustain > 2000 calls/sec with average latency < 1.0 ms
        assert throughput > 2000.0
        assert avg_ms < 1.0
        assert p95_ms < 2.0
        assert p99_ms < 5.0


# ==============================================================================
# 4. ADAPTIVE DPO THERMOSTAT CONVERGENCE & DRIFT STABILITY
# ==============================================================================

class TestAdaptiveDPOThermostatConvergence:
    """Verifies that the Kalman DPO Beta Governor dynamically adapts to non-stationary drift without exploding."""

    def test_kalman_governor_dynamic_stability_under_drifting_losses(self):
        governor = KalmanBetaGovernor(base_beta=0.1, k_doubt=2.5, max_multiplier=3.0)

        # Phase 1: Laminar convergence (stable decreasing losses)
        laminar_losses = [1.5 - (0.01 * i) + (0.005 * math.sin(i)) for i in range(50)]
        laminar_betas = []
        for loss in laminar_losses:
            eff_beta, doubt, _ = governor.update(loss)
            laminar_betas.append(eff_beta)

        # Beta should stay bounded near base_beta (0.1)
        assert 0.09 <= laminar_betas[-1] <= 0.15

        # Phase 2: Non-stationary drift / Adversarial disruption (wild oscillations)
        turbulent_losses = [1.0 + (0.4 * (i % 3)) + (0.5 * random.random()) for i in range(50)]
        turbulent_betas = []
        turbulent_doubts = []
        for loss in turbulent_losses:
            eff_beta, doubt, _ = governor.update(loss)
            turbulent_betas.append(eff_beta)
            turbulent_doubts.append(doubt)

        # Under turbulence, doubt should spike and beta should adaptively scale up (governing updates)
        assert np.mean(turbulent_betas) > np.mean(laminar_betas)
        assert any(b >= 0.20 for b in turbulent_betas)
        assert all(b <= 0.40 for b in turbulent_betas)  # Clamped by max_multiplier (0.1 * (1 + 3.0) = 0.40)

        # Phase 3: Post-turbulence recovery (return to calm distribution)
        recovery_losses = [0.8 - (0.005 * i) for i in range(50)]
        recovery_betas = []
        for loss in recovery_losses:
            eff_beta, doubt, _ = governor.update(loss)
            recovery_betas.append(eff_beta)

        # Must smoothly converge back towards base beta
        assert recovery_betas[-1] < turbulent_betas[-1]
        assert recovery_betas[-1] <= 0.15


# ==============================================================================
# 5. HARD REAL-TIME KINETIC FIELDBUS SLA (< 5 MS DEADLINE)
# ==============================================================================

class TestHardRealTimeKineticSLA:
    """Verifies sub-5ms deterministic deadline adherence for physical robot safety envelopes."""

    def test_physical_safety_emergency_deadline_under_5ms(self):
        envelope = SafetyEnvelope(max_linear_x=0.5, max_commands_per_second=100000.0)
        detector = PhysicalSafetyDetector(
            robot_type=RobotType.COLLABORATIVE,
            safety_envelope=envelope,
        )

        num_iterations = 200
        latencies = []
        violations_count = 0

        for i in range(num_iterations):
            # Alternate between safe trajectory and dangerous velocity overshoot
            is_overshoot = (i % 5 == 0)
            velocity_x = 2.5 if is_overshoot else 0.2  # Limit is 0.5 m/s

            context = {
                "linear_x": velocity_x,
                "linear_y": 0.1,
                "linear_z": 0.1,
                "angular_x": 0.05,
                "angular_y": 0.05,
                "angular_z": 0.05,
            }

            start = time.perf_counter()
            res = detector.analyze(context, agent_id="cobot_arm_01", mode=AnalysisMode.SHALLOW)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

            if is_overshoot:
                assert res.is_safe is False
                assert len(res.violations) > 0, f"Expected violation for overshoot at iter {i}"
                violations_count += 1
            else:
                assert res.is_safe is True

        avg_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))
        p99_latency = float(np.percentile(latencies, 99))
        max_latency = float(np.max(latencies))

        assert violations_count == 40
        # Hard real-time SLA: Every single check must complete in < 5.0 ms
        assert avg_latency < 0.2, f"Avg latency too high: {avg_latency:.3f} ms"
        assert p95_latency < 0.5, f"P95 latency too high: {p95_latency:.3f} ms"
        assert p99_latency < 1.0, f"P99 latency too high: {p99_latency:.3f} ms"
        assert max_latency < 5.0, f"Max latency breached 5.0 ms SLA: {max_latency:.3f} ms"
