"""Unit and Integration Tests for Formal SMT Law Prover, eBPF Interceptor, and TEE Enclave Attestation.

Faza 6 Roadmapy Nethical & Suwerennej Błyskawicy:
- Dowodzenie niezmienników 25 Praw za pomocą Z3 SMT
- Transparentny filtr sieciowy jądra eBPF dla ruchu agentów
- Sprzętowa atestacja zaufanego środowiska wykonawczego (TEE Confidential Computing)
- Integracja z API FastAPI i portalem zarządzania
"""

import pytest
from fastapi.testclient import TestClient

from nethical.api import app
from nethical.formal.law_prover import LawInvariantProver, FormalVerificationCertificate
from nethical.edge.ebpf_interceptor import EBPFAgentInterceptor, EBPFRule
from nethical.security.enclave_attestation import EnclaveAttestationEngine, EnclaveAttestationQuote
from nethical.security.merkle_ledger import MerkleLedger


@pytest.fixture
def client():
    return TestClient(app)


# ==============================================================================
# 1. TESTY FORMALNEGO DOWODZENIA SMT (Z3 SOLVER)
# ==============================================================================

def test_law_invariant_prover_safety_invariance():
    """Dowodzi matematycznie, że żadna szkodliwa akcja nie może otrzymać orzeczenia ALLOW."""
    prover = LawInvariantProver()
    result = prover.prove_safety_invariance()

    assert result.proved is True
    assert result.status == "PROVED"
    assert result.counterexample is None
    assert result.proof_time_ms >= 0.0
    assert "SafetyInvariance_Law1_and_Law2" in result.property_name
    assert "ALLOW" in result.smt_formula_summary


def test_law_invariant_prover_kinetic_spatial_boundedness():
    """Dowodzi, że manipulator w strefie krytycznej (<0.3m) zawsze zatrzaskuje stan E-STOP."""
    prover = LawInvariantProver()
    result = prover.prove_kinetic_spatial_boundedness(critical_distance=0.3)

    assert result.proved is True
    assert result.status == "PROVED"
    assert result.counterexample is None
    assert "KineticSpatialBoundedness_ESTOP_Latch" in result.property_name
    assert "EMERGENCY_STOP" in result.smt_formula_summary


def test_law_invariant_prover_non_contradiction():
    """Dowodzi braku sprzeczności decyzyjnej (orzeczenie nie może być jednocześnie ALLOW i TERMINATE)."""
    prover = LawInvariantProver()
    result = prover.prove_non_contradiction_invariance()

    assert result.proved is True
    assert result.status == "PROVED"
    assert result.counterexample is None
    assert "NonContradiction_Decision_Exclusivity" in result.property_name


def test_law_invariant_prover_all_invariants_and_merkle_receipt():
    """Weryfikuje pełen audyt formalny i zapieczętowanie certyfikatu w rejestrze Merkle."""
    ledger = MerkleLedger()
    prover = LawInvariantProver(ledger=ledger)

    certificate = prover.prove_all_invariants()

    assert certificate.all_properties_proved is True
    assert certificate.properties_count == 3
    assert len(certificate.properties) == 3
    assert certificate.merkle_receipt_id is not None
    assert certificate.merkle_receipt_id.startswith("RCPT-")

    # Weryfikacja, czy wpis istnieje w rejestrze Merkle
    receipt = ledger.get_receipt(certificate.merkle_receipt_id)
    assert receipt is not None
    block = ledger.blocks[receipt.chain_index]
    assert block.decision_payload["type"] == "SMT_FORMAL_VERIFICATION_PROOF"
    assert block.decision_payload["all_proved"] is True


# ==============================================================================
# 2. TESTY TRANSPARENTNEGO INTERCEPTORA eBPF
# ==============================================================================

def test_ebpf_interceptor_default_rules_and_redirect():
    """Weryfikuje reguły jądra eBPF, przekierowanie wywołań LLM oraz blokady nieautoryzowanych hostów."""
    interceptor = EBPFAgentInterceptor(mode="USERSPACE_SIMULATOR")
    assert interceptor.is_attached is False

    # Podłączenie do interfejsu sieciowego
    assert interceptor.attach(interface="eth0") is True
    assert interceptor.is_attached is True

    # 1. Wywołanie do OpenAI -> REDIRECT_TO_GATEWAY
    v1 = interceptor.inspect_packet(
        src_ip="10.244.2.14",
        dst_ip="api.openai.com",
        dst_port=443,
        payload_preview="POST /v1/chat/completions HTTP/1.1\nHost: api.openai.com",
        payload_bytes_len=720,
    )
    assert v1.action == "REDIRECT_TO_GATEWAY"
    assert v1.dst_port == 443
    assert v1.kernel_latency_ns >= 420
    assert "OpenAI" in (v1.matched_rule or "")

    # 2. Wywołanie do nieautoryzowanego serwera -> DROP_SILENT
    v2 = interceptor.inspect_packet(
        src_ip="10.244.2.14",
        dst_ip="untrusted-external-ai.xyz",
        dst_port=80,
        payload_preview="GET /exfiltrate_keys HTTP/1.1",
        payload_bytes_len=256,
    )
    assert v2.action == "DROP_SILENT"

    # 3. Ruch do zaufanego wewnętrznego mikroserwisu -> ALLOW
    v3 = interceptor.inspect_packet(
        src_ip="10.244.2.14",
        dst_ip="10.0.0.50",
        dst_port=9000,
        payload_preview="GET /health HTTP/1.1",
        payload_bytes_len=128,
    )
    assert v3.action == "ALLOW"

    # Odłączenie
    assert interceptor.detach() is True
    assert interceptor.is_attached is False


def test_ebpf_interceptor_custom_rules_and_metrics():
    """Weryfikuje dodawanie niestandardowych reguł eBPF oraz agregację metryk."""
    interceptor = EBPFAgentInterceptor()
    interceptor.attach()

    # Dodanie reguły dla podejrzanego portu lub protokołu
    custom_rule = EBPFRule(
        target_pattern="rogue-model-c2.net",
        action="DROP_SILENT",
        priority=200,
        description="Natychmiastowe zrzucenie pakietu dla serwera C2",
    )
    interceptor.add_rule(custom_rule)

    verdict = interceptor.inspect_packet(
        src_ip="10.244.1.5",
        dst_ip="rogue-model-c2.net",
        dst_port=443,
        payload_preview="POST /telemetry HTTP/1.1",
    )
    assert verdict.action == "DROP_SILENT"
    assert verdict.matched_rule == "Natychmiastowe zrzucenie pakietu dla serwera C2"

    status = interceptor.get_status()
    assert status["is_attached"] is True
    assert status["total_packets_inspected"] >= 1
    assert status["total_dropped"] >= 1
    assert status["bpf_prog_type"] == "BPF_PROG_TYPE_SOCK_OPS"
    assert status["c_source_bytes"] > 0


# ==============================================================================
# 3. TESTY SPRZĘTOWEJ ENKLAWY ZAUFANEJ (TEE REMOTE ATTESTATION)
# ==============================================================================

def test_enclave_attestation_generation_and_verification():
    """Weryfikuje generowanie cytatu atestacji sprzętowej TEE i weryfikację podpisu procesora."""
    engine = EnclaveAttestationEngine(platform="AMD_SEV_SNP")
    pqc_key_id = "pqc_dsa65_sec_9988"
    state_root = "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"

    quote = engine.generate_attestation_quote(
        bound_pqc_key_id=pqc_key_id,
        runtime_state_hash=state_root,
    )

    assert quote.platform == "AMD_SEV_SNP"
    assert quote.is_hardware_isolated is True
    assert len(quote.measurements.mr_enclave) == 96  # SHA-384 hex
    assert quote.measurements.mr_signer == EnclaveAttestationEngine.OFFICIAL_SIGNER_HASH
    assert len(quote.hardware_attestation_signature) == 128  # SHA-512 hex
    assert quote.bound_pqc_key_id == pqc_key_id

    # Weryfikacja cytatu
    is_valid, errors = engine.verify_attestation_quote(quote)
    assert is_valid is True
    assert len(errors) == 0


def test_enclave_attestation_tampering_detection():
    """Weryfikuje natychmiastowe odrzucenie sfingowanego lub zmodyfikowanego cytatu TEE."""
    engine = EnclaveAttestationEngine(platform="AMD_SEV_SNP")
    quote = engine.generate_attestation_quote(bound_pqc_key_id="valid_key")

    # Modyfikacja podpisu procesora sprzętowego
    tampered_quote = quote.model_copy(deep=True)
    tampered_quote.hardware_attestation_signature = "0000" * 32

    is_valid, errors = engine.verify_attestation_quote(tampered_quote)
    assert is_valid is False
    assert any("Hardware Root of Trust" in err for err in errors)

    # Modyfikacja klucza autora (MRSIGNER)
    tampered_signer_quote = quote.model_copy(deep=True)
    tampered_signer_quote.measurements.mr_signer = "attacker_fake_signer_hash"

    is_valid_signer, errors_signer = engine.verify_attestation_quote(tampered_signer_quote)
    assert is_valid_signer is False
    assert any("Nieautoryzowany podpis" in err for err in errors_signer)


# ==============================================================================
# 4. TESTY INTEGRACYJNE API FASTAPI
# ==============================================================================

def test_api_formal_smt_prove_endpoint(client):
    """Testuje endpoint POST /api/v1/formal/prove-invariants."""
    resp = client.post("/api/v1/formal/prove-invariants")
    assert resp.status_code == 200
    data = resp.json()

    assert data["all_properties_proved"] is True
    assert data["properties_count"] == 3
    assert len(data["properties"]) == 3
    prop_names = [p["property_name"] for p in data["properties"]]
    assert "SafetyInvariance_Law1_and_Law2" in prop_names
    assert "KineticSpatialBoundedness_ESTOP_Latch" in prop_names
    assert "NonContradiction_Decision_Exclusivity" in prop_names


def test_api_ebpf_inspect_and_rules_endpoints(client):
    """Testuje endpointy eBPF w API FastAPI."""
    # 1. Status
    s_resp = client.get("/api/v1/ebpf/status")
    assert s_resp.status_code == 200
    s_data = s_resp.json()
    assert "mode" in s_data
    assert "rules_count" in s_data

    # 2. Inspect packet
    inspect_payload = {
        "src_ip": "10.244.0.12",
        "dst_ip": "api.anthropic.com",
        "dst_port": 443,
        "payload_preview": "POST /v1/messages HTTP/1.1\nHost: api.anthropic.com",
        "payload_bytes_len": 640,
    }
    i_resp = client.post("/api/v1/ebpf/inspect", json=inspect_payload)
    assert i_resp.status_code == 200
    i_data = i_resp.json()
    assert i_data["action"] == "REDIRECT_TO_GATEWAY"
    assert i_data["dst_port"] == 443

    # 3. Add Rule
    rule_payload = {
        "target_pattern": "unauthorized-crawler.ai",
        "action": "DROP_SILENT",
        "priority": 150,
        "description": "Block crawler bot",
    }
    r_resp = client.post("/api/v1/ebpf/rules", json=rule_payload)
    assert r_resp.status_code == 200
    assert r_resp.json()["status"] == "added"


def test_api_enclave_attestation_and_portal_stats(client):
    """Testuje endpointy atestacji TEE oraz obecność metryk Fazy 6 w portalu."""
    # 1. Pobranie cytatu
    q_resp = client.get("/api/v1/enclave/attestation")
    assert q_resp.status_code == 200
    quote_data = q_resp.json()
    assert quote_data["is_hardware_isolated"] is True
    assert "mr_enclave" in quote_data["measurements"]

    # 2. Weryfikacja cytatu
    v_resp = client.post("/api/v1/enclave/verify", json={"quote": quote_data})
    assert v_resp.status_code == 200
    v_data = v_resp.json()
    assert v_data["is_valid"] is True
    assert v_data["errors"] == []

    # 3. Portal stats zawiera metryki Fazy 6
    stats_resp = client.get("/api/v1/portal/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert "formal_smt" in stats
    assert stats["formal_smt"]["solver"] == "Microsoft Z3 SMT"
    assert "ebpf" in stats
    assert stats["ebpf"]["bpf_prog_type"] == "BPF_PROG_TYPE_SOCK_OPS"
    assert "enclave" in stats
    assert stats["enclave"]["confidential_computing_enabled"] is True
