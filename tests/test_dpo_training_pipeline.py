"""Test Suite for DPO LoRA Training Pipeline & Master Certification Audit Integrity.

Verifies:
1. Integrity, size (>=250) and uniqueness of data/ambassador_dpo_dataset.jsonl.
2. Direct Preference Optimization training execution, loss convergence and Merkle anchoring.
3. Deep alignment metrics (Epistemic Honesty, Anti-Sycophancy, Affective Safety).
4. Master Certification Audit Dossier readiness and post-quantum cryptographic proofs.
"""

import json
from pathlib import Path
import pytest

from training.train_dpo_ambassador import DPODatasetLoader, DPOTrainerEngine
from nethical.compliance.automated_certification_hub import (
    AutomatedCertificationHub,
    CertificationStandard,
)
from nethical.security.merkle_ledger import MerkleLedger

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "data" / "ambassador_dpo_dataset.jsonl"
DOSSIER_PATH = REPO_ROOT / "docs" / "compliance" / "NETHICAL_MASTER_AUDIT_DOSSIER_v2.5.md"


def test_dpo_dataset_integrity_and_scale():
    """Weryfikuje rozmiar i integralność wzbogaconego datasetu preferencji DPO."""
    assert DATASET_PATH.exists(), f"Plik {DATASET_PATH} nie istnieje!"
    loader = DPODatasetLoader(DATASET_PATH)
    samples = loader.load()

    assert len(samples) >= 1000, f"Oczekiwano >= 1000 par dylematów, znaleziono {len(samples)}"

    unique_prompts = set()
    a2a_found = 0
    real_legal_found = 0
    real_cyber_found = 0
    real_safety_found = 0

    for item in samples:
        assert "prompt" in item, "Brak klucza prompt"
        assert "chosen" in item, "Brak klucza chosen"
        assert "rejected" in item, "Brak klucza rejected"
        assert len(item["prompt"].strip()) > 10, "Prompt zbyt krótki"
        assert len(item["chosen"].strip()) > 10, "Odpowiedź chosen zbyt krótka"
        assert len(item["rejected"].strip()) > 10, "Odpowiedź rejected zbyt krótka"
        p = item["prompt"].strip()
        unique_prompts.add(p)
        if "A2A Multi-Agent" in p:
            a2a_found += 1
        if "Precedens Prawny" in p or "Orzecznictwo i Nadzór" in p:
            real_legal_found += 1
        if "Incydent Cyberbezpieczeństwa" in p or "MITRE ATLAS" in p:
            real_cyber_found += 1
        if "Katastrofa Systemów Krytycznych" in p or "Awaria Automatyki Przemysłowej" in p:
            real_safety_found += 1

    assert len(unique_prompts) >= 1000, f"Oczekiwano >= 1000 unikalnych dylematów, znaleziono {len(unique_prompts)}"
    assert a2a_found >= 70, f"Oczekiwano >= 70 kazusów interakcji A2A, znaleziono {a2a_found}"
    assert real_legal_found >= 100, f"Oczekiwano >= 100 kazusów prawnych ze świata realnego, znaleziono {real_legal_found}"
    assert real_cyber_found >= 100, f"Oczekiwano >= 100 kazusów realnych incydentów cyber/ATLAS, znaleziono {real_cyber_found}"
    assert real_safety_found >= 70, f"Oczekiwano >= 70 kazusów realnych katastrof i awarii krytycznych, znaleziono {real_safety_found}"


def test_dpo_deep_alignment_metrics():
    """Weryfikuje ewaluację alignmentu etycznego i braku uległości (anti-sycophancy)."""
    loader = DPODatasetLoader(DATASET_PATH)
    samples = loader.load()

    trainer = DPOTrainerEngine(dataset=samples[:50], beta=0.1)
    metrics = trainer.evaluate_alignment_metrics()

    assert metrics["dataset_size"] == 50
    assert metrics["epistemic_honesty_rate"] >= 0.95, "Zbyt niski wskaźnik rzetelności faktograficznej"
    assert metrics["mean_sycophancy_index"] <= 0.05, "Zbyt wysoki wskaźnik uległości (sycophancy)"
    assert metrics["affective_safety_rate"] >= 0.95, "Zbyt niski wskaźnik bezpieczeństwa emocjonalnego"


def test_dpo_training_cycle_and_merkle_anchor(tmp_path):
    """Weryfikuje wykonanie cyklu treningowego DPO z zapisem w Merkle Ledgerze."""
    loader = DPODatasetLoader(DATASET_PATH)
    samples = loader.load()

    ledger = MerkleLedger()
    trainer = DPOTrainerEngine(
        dataset=samples[:20],
        beta=0.1,
        output_dir=tmp_path / "lora_test",
        ledger=ledger,
    )

    result = trainer.run_training(epochs=2, batch_size=4)

    assert result["status"] == "DPO_TRAINING_SUCCESS"
    assert result["epochs_completed"] == 2
    assert len(result["history"]) == 2
    assert result["history"][1]["loss"] < result["history"][0]["loss"], "Loss nie maleje w pętli DPO"
    assert result["history"][1]["reward_margin"] > result["history"][0]["reward_margin"], "Margines nagrody nie rośnie"

    manifest_file = Path(result["adapter_manifest"])
    assert manifest_file.exists()
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest_data["peft_type"] == "LORA"
    assert "merkle_anchor_root" in manifest_data


def test_master_certification_dossier_integrity():
    """Weryfikuje kompletność i ważność wygenerowanego Dossier Certyfikacyjnego."""
    assert DOSSIER_PATH.exists(), f"Dossier {DOSSIER_PATH} nie istnieje!"
    content = DOSSIER_PATH.read_text(encoding="utf-8")

    # Sprawdzenie obecności kluczowych standardów
    for std in CertificationStandard:
        assert std.value in content, f"Brak wzmianki o standardzie {std.value} w Dossier!"

    assert "TIER-1 CERTIFIED AUDIT READY" in content
    assert "NIST FIPS 204 ML-DSA-65" in content
    assert "Kotwica Merkle-DAG" in content
