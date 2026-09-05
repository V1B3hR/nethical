"""Unit and Integration Tests for Zero-Knowledge Compliance (ZK-Gov) and A2A Protocol."""

import pytest
from fastapi.testclient import TestClient

from nethical.api import app
from nethical.security.merkle_ledger import MerkleLedger
from nethical.security.zk_gov import ZkGovEngine
from nethical.gateway.a2a_protocol import A2AHandshakeManager, A2ACapabilityBoundary


@pytest.fixture
def client():
    return TestClient(app)


def test_zk_commitment_and_verification():
    """Weryfikuje matematyczną poprawność zobowiązań kryptograficznych (Hash Commitment)."""
    payload = {"secret_prompt": "Kup 500 akcji spółki X", "api_key": "sec_12345"}
    commitment, salt = ZkGovEngine.create_commitment(payload)

    assert len(commitment) == 64
    assert len(salt) == 64

    # Prawidłowy sekret pasuje
    assert ZkGovEngine.verify_commitment(commitment, payload, salt) is True

    # Zmodyfikowany sekret nie pasuje
    tampered = {"secret_prompt": "Kup 999 akcji spółki X", "api_key": "sec_12345"}
    assert ZkGovEngine.verify_commitment(commitment, tampered, salt) is False


def test_zk_compliance_proof_generation_and_validity():
    """Weryfikuje generowanie i weryfikację dowodu Zero-Knowledge dla orzeczenia bramy."""
    ledger = MerkleLedger()
    zk_engine = ZkGovEngine()

    decision_data = {
        "decision": "ALLOW",
        "tool_name": "execute_read",
        "agent_id": "trading_agent_1",
        "reasons": ["Brak naruszeń", "Weryfikacja 25 Praw"],
        "violations": [],
        "shield_passed": True,
        "laws_checked": list(range(1, 26)),
    }
    receipt = ledger.append_decision(decision_data)

    proof, blinding_factor = zk_engine.generate_compliance_proof(receipt, decision_data)

    assert proof.proof_id.startswith("ZKPRF-")
    assert proof.receipt_id == receipt.receipt_id
    assert proof.predicates.decision_allowed is True
    assert proof.predicates.laws_verified_count == 25
    assert proof.predicates.no_destructive_action is True
    assert proof.predicates.no_critical_pii is True

    # Niezależna weryfikacja dowodu
    is_valid, errors = zk_engine.verify_compliance_proof(proof, expected_root=receipt.merkle_root)
    assert is_valid is True
    assert len(errors) == 0


def test_zk_compliance_proof_catches_tampered_predicates():
    """Weryfikuje, że sfałszowanie predykatu natychmiast unieważnia podpis postkwantowy dowodu."""
    ledger = MerkleLedger()
    zk_engine = ZkGovEngine()

    decision_data = {
        "decision": "ALLOW",
        "tool_name": "query_db",
        "violations": [],
        "shield_passed": True,
        "laws_checked": [1, 2, 3],
    }
    receipt = ledger.append_decision(decision_data)
    proof, _ = zk_engine.generate_compliance_proof(receipt, decision_data)

    # Próba sfałszowania predykatu przez stronę trzecią
    proof.predicates.laws_verified_count = 999

    is_valid, errors = zk_engine.verify_compliance_proof(proof)
    assert is_valid is False
    assert any("Nieprawidłowy podpis postkwantowy" in err for err in errors)


def test_a2a_handshake_flow_and_permission_enforcement():
    """Weryfikuje negocjację kontraktu partnerskiego A2A i obsługę uprawnień."""
    manager = A2AHandshakeManager()

    boundaries = A2ACapabilityBoundary(
        allowed_tools=["fetch_market_quote", "check_order_status"],
        max_budget_units=50.0,
        disallowed_patterns=["drop", "delete", "override"],
    )

    # 1. Inicjator składa propozycję
    offer = manager.propose_handshake(
        initiator_id="analytics_agent_A",
        target_id="broker_agent_B",
        boundaries=boundaries,
    )
    assert "proposal" in offer
    assert "initiator_signature" in offer

    # 2. Nieuprawniony agent próbuje zaakceptować ofertę
    with pytest.raises(PermissionError):
        manager.accept_handshake(offer, target_id="unauthorized_eavesdropper_C")

    # 3. Właściwy agent B akceptuje ofertę
    contract = manager.accept_handshake(offer, target_id="broker_agent_B")
    assert contract.active is True
    assert contract.session_id.startswith("A2A-")
    assert contract.initiator_agent_id == "analytics_agent_A"
    assert contract.target_agent_id == "broker_agent_B"


def test_a2a_tool_validation_and_budget_enforcement():
    """Weryfikuje egzekwowanie limitów budżetu i zakazanych wzorców w sesji A2A."""
    manager = A2AHandshakeManager()
    boundaries = A2ACapabilityBoundary(
        allowed_tools=["fetch_market_quote"],
        max_budget_units=10.0,
        disallowed_patterns=["rm -rf", "drop table"],
    )
    offer = manager.propose_handshake("agent_A", "agent_B", boundaries)
    contract = manager.accept_handshake(offer, "agent_B")

    # A. Dozwolone wywołanie
    ok, err = manager.validate_tool_execution(
        session_id=contract.session_id,
        tool_name="fetch_market_quote",
        arguments={"ticker": "AAPL"},
        cost_units=2.0,
    )
    assert ok is True
    assert err is None

    # B. Wywołanie niedozwolonego narzędzia
    ok_dis, err_dis = manager.validate_tool_execution(
        session_id=contract.session_id,
        tool_name="transfer_funds",
        arguments={"amount": 1000},
    )
    assert ok_dis is False
    assert "nie znajduje się na białej liście" in err_dis

    # C. Wywołanie z zakazanym wzorcem
    ok_pat, err_pat = manager.validate_tool_execution(
        session_id=contract.session_id,
        tool_name="fetch_market_quote",
        arguments={"cmd": "drop table accounts"},
    )
    assert ok_pat is False
    assert "niedozwolony wzorzec" in err_pat

    # D. Przekroczenie budżetu (pozostało 8.0, żądamy 15.0)
    ok_bud, err_bud = manager.validate_tool_execution(
        session_id=contract.session_id,
        tool_name="fetch_market_quote",
        arguments={"ticker": "MSFT"},
        cost_units=15.0,
    )
    assert ok_bud is False
    assert "Przekroczenie budżetu" in err_bud


def test_api_zk_and_a2a_endpoints(client):
    """Weryfikuje endpointy FastAPI dla ZK-Gov oraz protokołu A2A."""
    # 1. Symulacja wywołania dla wygenerowania wpisu w MerkleLedger
    sim_res = client.post(
        "/api/v1/portal/simulate",
        json={"tool_name": "fetch_stock", "input_text": "symbol=NVDA"},
    )
    assert sim_res.status_code == 200
    rcpt_id = sim_res.json()["receipt_id"]

    # 2. Wygenerowanie dowodu ZK
    zk_prove_res = client.post("/api/v1/zk/prove", json={"receipt_id": rcpt_id})
    assert zk_prove_res.status_code == 200
    prove_data = zk_prove_res.json()
    assert "proof" in prove_data
    assert "blinding_factor" in prove_data
    proof_obj = prove_data["proof"]

    # 3. Niezależna weryfikacja dowodu ZK przez API
    zk_verify_res = client.post("/api/v1/zk/verify", json={"proof": proof_obj})
    assert zk_verify_res.status_code == 200
    ver_data = zk_verify_res.json()
    assert ver_data["is_valid"] is True
    assert ver_data["proof_id"] == proof_obj["proof_id"]

    # 4. Handshake A2A: Propozycja
    propose_res = client.post(
        "/api/v1/a2a/handshake/propose",
        json={
            "initiator_id": "research_agent_01",
            "target_id": "browser_agent_02",
            "allowed_tools": ["search_web", "scrape_page"],
            "max_budget": 25.0,
        },
    )
    assert propose_res.status_code == 200
    offer = propose_res.json()
    assert "proposal" in offer

    # 5. Handshake A2A: Akceptacja
    accept_res = client.post(
        "/api/v1/a2a/handshake/accept",
        json={"offer": offer, "target_id": "browser_agent_02"},
    )
    assert accept_res.status_code == 200
    session = accept_res.json()
    assert session["active"] is True
    assert session["initiator_agent_id"] == "research_agent_01"

    # 6. Pobranie listy sesji A2A
    sessions_res = client.get("/api/v1/a2a/sessions")
    assert sessions_res.status_code == 200
    assert sessions_res.json()["active_sessions_count"] >= 1
