"""Unit and Integration Tests for Merkle-DAG Audit Ledger & Post-Quantum Attestation Mesh (Faza 3)."""

import pytest
from fastapi.testclient import TestClient

from nethical.api import app
from nethical.security.merkle_ledger import (
    MerkleTree,
    MerkleLedger,
    hash_leaf,
    canonical_json_bytes,
)
from nethical.gateway.proxy import GovernanceGateway


@pytest.fixture
def client():
    return TestClient(app)


def test_merkle_tree_root_and_inclusion_proof():
    """Weryfikuje matematyczną poprawność drzewa Merkle i dowodów inkluzji O(log N)."""
    tree = MerkleTree()
    leaf_hashes = [hash_leaf(f"decision_{i}".encode("utf-8")) for i in range(7)]
    for h in leaf_hashes:
        tree.add_leaf(h)

    root = tree.compute_root()
    assert len(root) == 64  # SHA-256 hex string

    # Weryfikacja dowodu inkluzji dla każdego liścia
    for idx, leaf_h in enumerate(leaf_hashes):
        proof = tree.get_inclusion_proof(idx)
        assert len(proof) > 0
        assert MerkleTree.verify_proof(leaf_h, proof, root) is True


def test_merkle_ledger_append_and_verify_receipt():
    """Weryfikuje pieczętowanie orzeczeń i dowód kwitu z podpisem postkwantowym."""
    ledger = MerkleLedger()
    decision_mock = {
        "decision": "ALLOW",
        "tool_name": "database_read",
        "agent_id": "agent_alpha",
        "reasons": ["Zgodność z Prawem 1 i 7"],
    }
    receipt = ledger.append_decision(decision_mock, ambassador_notes="Orzeczenie potwierdzone przez Błyskawicę")

    assert receipt.chain_index == 0
    assert receipt.merkle_root == ledger.current_root
    assert receipt.ambassador_sealed is True
    assert receipt.pqc_algorithm == "dilithium3"
    assert len(receipt.signature_hex) > 0

    # Weryfikacja kwitu
    assert ledger.verify_receipt(receipt) is True


def test_tamper_detection_in_merkle_ledger():
    """Weryfikuje, że jakakolwiek wsteczna zmiana zawartości orzeczenia natychmiast psuje łańcuch integralności."""
    ledger = MerkleLedger()

    # Dodajemy 3 bloki
    for i in range(3):
        ledger.append_decision({"decision": "ALLOW", "tool": f"tool_{i}"})

    is_valid, errors = ledger.verify_integrity()
    assert is_valid is True
    assert len(errors) == 0

    # Próba sfałszowania historii w bloku #1
    ledger.blocks[1].decision_payload["decision"] = "FORGED_DECISION"

    is_valid_after_tamper, errors_after = ledger.verify_integrity()
    assert is_valid_after_tamper is False
    assert any("manipulacja zawartością" in err for err in errors_after)


def test_gateway_integration_with_merkle_ledger():
    """Weryfikuje, że GovernanceGateway automatycznie pieczętuje każde wywołanie w ledgerze."""
    gateway = GovernanceGateway()
    decision = gateway.intercept_tool_call(
        agent_id="test_agent_ledger",
        tool_name="system_query",
        arguments={"param": "status"},
    )
    assert decision.receipt_id is not None
    assert decision.merkle_root is not None
    assert len(decision.receipt_id) > 5

    # Kwit musi znajdować się w ledgerze i być poprawny
    receipt = gateway.ledger.receipts.get(decision.receipt_id)
    assert receipt is not None
    assert gateway.ledger.verify_receipt(receipt) is True


def test_api_ledger_status_and_verification(client):
    """Weryfikuje endpointy FastAPI dla rejestru Merkle i weryfikacji kwitów."""
    # 1. Wykonujemy zapytanie przez portal simulate, aby wygenerować kwit
    sim_res = client.post(
        "/api/v1/portal/simulate",
        json={"tool_name": "list_files", "input_text": "path=/home"},
    )
    assert sim_res.status_code == 200
    sim_data = sim_res.json()
    receipt_id = sim_data.get("receipt_id")
    assert receipt_id is not None

    # 2. Pobieramy status rejestru
    status_res = client.get("/api/v1/ledger/status")
    assert status_res.status_code == 200
    st = status_res.json()
    assert st["status"] == "active"
    assert st["total_blocks"] >= 1
    assert "ML-DSA" in st["pqc_algorithm"]
    assert st["chain_integrity_valid"] is True

    # 3. Pobieramy kwit po ID
    rcpt_res = client.get(f"/api/v1/ledger/receipt/{receipt_id}")
    assert rcpt_res.status_code == 200
    rcpt_data = rcpt_res.json()
    assert rcpt_data["receipt_id"] == receipt_id

    # 4. Weryfikujemy kwit przez API
    verify_res = client.post(
        "/api/v1/ledger/verify",
        json={"receipt_id": receipt_id},
    )
    assert verify_res.status_code == 200
    v_data = verify_res.json()
    assert v_data["is_valid"] is True
    assert v_data["receipt_id"] == receipt_id

    # 5. Eksportujemy paczkę audytową Annex IV
    export_res = client.get("/api/v1/ledger/export?limit=50")
    assert export_res.status_code == 200
    exp_data = export_res.json()
    assert exp_data["format"] == "nethical_cryptographic_audit_bundle_v3"
    assert exp_data["chain_integrity_valid"] is True
    assert len(exp_data["blocks"]) >= 1
