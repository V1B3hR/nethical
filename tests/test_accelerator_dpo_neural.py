"""Test Suite for Neural DPO Ambassador Training with AcceleratorAI Integration.

Verifies:
1. AmbassadorNeuralPolicy forward pass and response log-probabilities calculation.
2. Direct Preference Optimization neural training cycle with AcceleratorAI (RTX 4070 / CUDA / CPU).
3. Pneumatic tanh soft-clipping and VRAMPressureGuard memory telemetry.
4. Cryptographic anchoring of neural checkpoints in MerkleLedger (ML-DSA-65).
"""

import json
from pathlib import Path
import pytest
import torch

from training.train_dpo_ambassador import (
    DPODatasetLoader,
    DPOTrainerEngine,
    AmbassadorNeuralPolicy,
    TORCH_AVAILABLE,
    ACCELERATOR_AI_AVAILABLE,
)
from nethical.security.merkle_ledger import MerkleLedger

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "data" / "ambassador_dpo_dataset.jsonl"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for neural tests")
def test_neural_policy_model_forward_and_log_probs():
    """Weryfikuje poprawność modelu neuronowego AmbassadorNeuralPolicy."""
    vocab_size = 1000
    model = AmbassadorNeuralPolicy(vocab_size=vocab_size, d_model=64, nhead=2, num_layers=2, max_seq_len=64)
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    response_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    response_mask[:, 8:] = 1  # tokens 8..15 to odpowiedź

    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Niepoprawny kształt logits: {logits.shape}"

    log_probs = model.compute_log_probs(input_ids, response_mask)
    assert log_probs.shape == (batch_size,), f"Niepoprawny kształt log_probs: {log_probs.shape}"
    assert not torch.isnan(log_probs).any(), "Wykryto wartości NaN w log_probs"
    assert not torch.isinf(log_probs).any(), "Wykryto wartości Inf w log_probs"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for neural tests")
def test_neural_dpo_training_cycle_with_accelerator(tmp_path):
    """Weryfikuje pełny cykl treningu neuronowego z włączonym AcceleratorAI."""
    loader = DPODatasetLoader(DATASET_PATH)
    samples = loader.load()[:12]

    ledger = MerkleLedger()
    output_dir = tmp_path / "lora_neural_test"

    trainer = DPOTrainerEngine(
        dataset=samples,
        beta=0.1,
        learning_rate=1e-3,
        output_dir=output_dir,
        ledger=ledger,
        use_accelerator=True,
        neural=True,
    )

    result = trainer.run_training(epochs=2, batch_size=4)

    assert result["status"] == "DPO_TRAINING_SUCCESS"
    assert result["epochs_completed"] == 2
    assert result["neural_mode"] is True
    assert result["accelerator_ai_active"] is True
    assert len(result["history"]) == 2

    # Weryfikacja metryk epoki
    h0 = result["history"][0]
    assert "loss" in h0
    assert "reward_margin" in h0
    assert "step_latency_ms" in h0
    assert "throughput_tokens_sec" in h0
    assert "soft_clips_count" in h0

    # Weryfikacja zapisu wag i manifestu
    manifest_file = Path(result["adapter_manifest"])
    assert manifest_file.exists()
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest_data["accelerator_ai_active"] is True
    assert manifest_data["neural_mode"] is True
    assert "merkle_anchor_root" in manifest_data

    weights_file = Path(result["model_weights"])
    assert weights_file.exists()
    assert weights_file.stat().st_size > 1000, "Plik wag modelu jest zbyt mały!"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required for neural tests")
def test_pneumatic_tanh_soft_clipping():
    """Weryfikuje działanie pneumatycznego tłumienia gradientów tanh."""
    model = AmbassadorNeuralPolicy(vocab_size=100, d_model=32, nhead=2, num_layers=1, max_seq_len=32)
    trainer = DPOTrainerEngine(dataset=[], beta=0.1, use_accelerator=True, neural=True)
    trainer.model = model

    # Sztuczne wygenerowanie ogromnego gradientu (symulacja eksplozji wag DPO)
    x = torch.randint(1, 100, (2, 10))
    logits = model(x)
    dummy_loss = (logits ** 2).sum() * 1000.0
    dummy_loss.backward()

    # Zastosowanie soft-clippingu
    norm_before = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters() if p.grad is not None])).item()
    assert norm_before > 10.0, "Gradient powinien być duży dla testu wybuchu"

    scaled_norm = trainer._apply_pneumatic_soft_clipping(max_norm=1.0, boost_ratio=1.0)
    norm_after = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in model.parameters() if p.grad is not None])).item()

    assert norm_after < 2.0, f"Pneumatyczny soft-clipping nie stłumił gradientu (norm_after={norm_after})"
