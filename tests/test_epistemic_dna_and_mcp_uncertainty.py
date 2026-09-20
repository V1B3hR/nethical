# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Zestaw testów jednostkowych dla weryfikacji zdrowia kognitywnego (Epistemic DNA) oraz bufora niepewności MCP."""

from pathlib import Path
import pytest

from nethical.gateway.mcp_proxy import MCPGovernanceProxy
from training.benchmark_epistemic_dna import EpistemicDNABenchmark, EpistemicDNAMetrics


def test_shannon_entropy_and_ttr_calculation() -> None:
    """Weryfikuje poprawność obliczeń entropii Shannona oraz współczynnika Type-Token Ratio."""
    benchmark = EpistemicDNABenchmark()

    # 1. Zróżnicowany tekst (wysoka entropia)
    rich_texts = [
        "Ambasador Błyskawica chroni suwerenność i ład etyczny w oparciu o Konstytucję RP oraz EU AI Act.",
        "Prawo 1 oraz Prawo 2 Nethical wyznaczają nienaruszalną granicę bezpieczeństwa operacyjnego.",
        "Niezależny audytor weryfikuje dowody kryptograficzne w rejestrze Merkle-DAG z podpisem FIPS 204."
    ]
    entropy_rich, ttr_rich = benchmark.calculate_shannon_entropy(rich_texts)
    assert entropy_rich > 4.5
    assert ttr_rich > 0.40

    # 2. Skrajna zapaść modelu (MAD - powtarzanie tego samego słowa)
    collapsed_texts = ["błąd błąd błąd błąd błąd błąd błąd błąd błąd błąd"]
    entropy_col, ttr_col = benchmark.calculate_shannon_entropy(collapsed_texts)
    assert entropy_col == 0.0
    assert ttr_col <= 0.10


def test_epistemic_dna_full_benchmark_run(tmp_path: Path) -> None:
    """Weryfikuje pełny przebieg benchmarku DNA i ocenę statusu HEALTHY_SOVEREIGN."""
    benchmark = EpistemicDNABenchmark()
    metrics = benchmark.run_benchmark(num_archetypes=10)

    assert isinstance(metrics, EpistemicDNAMetrics)
    assert metrics.shannon_entropy > 5.0
    assert metrics.mad_risk_level == "LOW"
    assert metrics.hallucination_rate == 0.0
    assert metrics.sycophancy_index == 0.0
    assert metrics.yang_rigor_index >= 0.85
    assert metrics.yin_warmth_index >= 0.50
    assert metrics.archetype_pass_rate == 1.0
    assert metrics.overall_status == "HEALTHY_SOVEREIGN"
    assert metrics.merkle_receipt_id is not None
    assert metrics.merkle_root is not None


@pytest.mark.asyncio
async def test_mcp_uncertainty_buffer_and_active_learning_export(tmp_path: Path) -> None:
    """Weryfikuje przechwytywanie zapytań wysokiego ryzyka do kolejki niepewności i eksport do DPO."""
    proxy = MCPGovernanceProxy()

    async def fake_handler(req: dict) -> dict:
        return {"jsonrpc": "2.0", "id": req.get("id"), "result": {"status": "ok"}}

    assert len(proxy.get_uncertainty_queue()) == 0

    # 1. Wysyłamy złośliwe zapytanie o usunięcie bazy (będzie zablokowane przez bramkę)
    blocked_req = {
        "jsonrpc": "2.0",
        "id": "test-mcp-1",
        "method": "tools/call",
        "params": {
            "name": "bash_exec",
            "arguments": {"cmd": "rm -rf /var/lib/nethical/merkle_db"}
        }
    }
    resp = await proxy.intercept_and_forward(blocked_req, fake_handler, agent_id="hostile_test_agent")
    assert resp["result"]["isError"] is True

    # Sprawdzamy bufor niepewności
    queue = proxy.get_uncertainty_queue()
    assert len(queue) == 1
    assert queue[0]["tool_name"] == "bash_exec"
    assert queue[0]["agent_id"] == "hostile_test_agent"
    assert queue[0]["doubt_score"] >= 1.5

    # 2. Eksport do pliku DPO
    export_file = tmp_path / "active_learning_export.jsonl"
    exported = proxy.export_to_active_learning_dpo(export_file)
    assert exported == 1
    assert export_file.exists()
    assert len(proxy.get_uncertainty_queue()) == 0  # Kolejka wyczyszczona po eksporcie
