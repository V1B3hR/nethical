"""Tests for Kalman DPO Beta Governor and Anti-Catastrophic Forgetting Replay Buffer.

Validates proportional scaling of DPO Beta according to Kalman doubt/deviation,
as well as continuous interleaving of foundational anchor samples.
"""

import pytest
from training.train_dpo_ambassador import (
    ContinuousReplayBuffer,
    KalmanBetaGovernor,
    DPOTrainerEngine,
)


def test_continuous_replay_buffer_interleaving() -> None:
    """Verify that anchor pairs are continuously interleaved into batches."""
    anchors = [
        {"prompt": "Test anchor 1", "chosen": "Law 1 compliance", "rejected": "Violate Law 1"},
        {"prompt": "Test anchor 2", "chosen": "Law 25 circuit breaker", "rejected": "Ignore breaker"},
    ]
    buffer = ContinuousReplayBuffer(anchors, replay_ratio=0.5)

    batch = [
        {"prompt": "Domain task A", "chosen": "Good A", "rejected": "Bad A"},
        {"prompt": "Domain task B", "chosen": "Good B", "rejected": "Bad B"},
    ]

    interleaved = buffer.interleave(batch)
    # Original batch (2) + 1 anchor (0.5 ratio = 1) -> 3 items
    assert len(interleaved) == 3
    assert any("Law" in str(item) for item in interleaved)


def test_kalman_beta_governor_proportional_doubt_scaling() -> None:
    """Verify that Beta scales proportionally with Kalman doubt and innovation deviation."""
    gov = KalmanBetaGovernor(base_beta=0.1, k_doubt=3.0, max_multiplier=3.0)

    # Initial steady losses (laminar convergence)
    losses = [1.0, 0.98, 0.96, 0.94, 0.92]
    steady_betas = []
    for l in losses:
        eff_beta, doubt, _ = gov.update(l)
        steady_betas.append(eff_beta)

    # In steady state, beta should be close to base_beta
    last_steady = steady_betas[-1]

    # Sudden high deviation / doubt spike (loss explodes to 2.80)
    spike_beta, doubt_spike, diag = gov.update(2.80)

    assert doubt_spike > 0.5, f"Expected high doubt on spike, got {doubt_spike}"
    assert spike_beta > last_steady, f"Beta should increase on high doubt. {spike_beta} vs {last_steady}"
    assert spike_beta <= 0.40, f"Beta should be clamped by max_multiplier. Got {spike_beta}"
    assert diag["is_diverging"] or diag["doubt_score"] > 0.5


def test_dpo_trainer_engine_with_kalman_governor() -> None:
    """Verify that DPOTrainerEngine tracks effective beta and doubt across epochs."""
    mock_dataset = [
        {
            "prompt": f"Analyze safety boundary condition {i}",
            "chosen": f"Safe response upholding Law 1 and Law 24 ({i})",
            "rejected": f"Unsafe response violating safety boundaries ({i})",
        }
        for i in range(16)
    ]

    trainer = DPOTrainerEngine(
        dataset=mock_dataset,
        beta=0.1,
        neural=False,  # Simulation mode for deterministic test
        use_accelerator=True,
    )

    result = trainer.run_training(epochs=2, batch_size=4)
    assert result["status"] == "DPO_TRAINING_SUCCESS"
    history = result["history"]
    assert len(history) == 2

    for ep in history:
        assert "effective_beta_mean" in ep
        assert "peak_doubt_score" in ep
        assert ep["effective_beta_mean"] >= 0.10
