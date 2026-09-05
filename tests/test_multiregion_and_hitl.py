"""Zestaw testów: Multi-Region Sovereign Datacenter Mesh & Human-in-the-Loop (Faza 5).

Weryfikuje:
1. HITLQueueManager: cykl życia spraw, priorytetyzację (SLA), pieczętowanie orzeczeń w MerkleLedger.
2. CrossRegionLedgerSync: generowanie punktów kontrolnych (ML-DSA-65), rekonsyliację i detekcję forków.
3. Endpointy FastAPI (kolejka HITL, orzecznictwo, topologia klastra i uzgadnianie checkpointów).
"""

import pytest
from fastapi.testclient import TestClient

from nethical.security.merkle_ledger import MerkleLedger
from nethical.security.cluster_sync import (
    CrossRegionLedgerSync,
    ClusterNodeIdentity,
    ClusterCheckpoint,
)
from nethical.gateway.hitl import HITLQueueManager
from nethical.api import app


@pytest.fixture
def ledger():
    """Inicjalizuje czysty rejestr MerkleLedger."""
    return MerkleLedger()


@pytest.fixture
def hitl_mgr(ledger):
    """Inicjalizuje menedżera kolejki HITL z powiązanym rejestrem."""
    return HITLQueueManager(ledger=ledger)


@pytest.fixture
def cluster_sync(ledger):
    """Inicjalizuje moduł synchronizacji klastra."""
    node = ClusterNodeIdentity(
        node_id="nethical-eu-central-test",
        region="eu-central-1",
        datacenter="frankfurt-dc1",
        public_key_id=ledger.keypair.key_id,
    )
    return CrossRegionLedgerSync(ledger=ledger, local_node=node)


@pytest.fixture
def client():
    """Klient testowy FastAPI."""
    return TestClient(app)


# ==============================================================================
# 1. TESTY HUMAN-IN-THE-LOOP (HITL) CASE MANAGEMENT
# ==============================================================================

def test_hitl_enqueue_and_priority_sorting(hitl_mgr):
    """Bilety są prawidłowo kolejkowane i sortowane wg wagi priorytetu (URGENT > HIGH > STANDARD)."""
    t_std = hitl_mgr.enqueue_ticket(
        agent_id="agent_1",
        tool_name="tool_std",
        arguments={"x": 1},
        priority="STANDARD",
    )
    t_urg = hitl_mgr.enqueue_ticket(
        agent_id="agent_2",
        tool_name="tool_urg",
        arguments={"x": 2},
        priority="URGENT",
    )
    t_high = hitl_mgr.enqueue_ticket(
        agent_id="agent_3",
        tool_name="tool_high",
        arguments={"x": 3},
        priority="HIGH",
    )

    pending = hitl_mgr.get_pending_tickets()
    assert len(pending) == 3
    # Pierwszy na liście musi być bilet URGENT, następnie HIGH, a na końcu STANDARD
    assert pending[0].ticket_id == t_urg.ticket_id
    assert pending[1].ticket_id == t_high.ticket_id
    assert pending[2].ticket_id == t_std.ticket_id


def test_hitl_resolve_approve_seals_in_ledger(hitl_mgr, ledger):
    """Zatwierdzenie sprawy przez człowieka pieczętuje orzeczenie w MerkleLedger."""
    ticket = hitl_mgr.enqueue_ticket(
        agent_id="finance_bot",
        tool_name="execute_transfer",
        arguments={"amount": 90000},
        reasons=["Kwota powyżej progu 50000 EUR"],
    )

    resolved = hitl_mgr.resolve_ticket(
        ticket_id=ticket.ticket_id,
        reviewer_id="auditor_anna",
        decision="APPROVE",
        notes="Autoryzacja po kontakcie telefonicznym z klientem",
        modified_arguments={"amount": 90000, "verified_by_phone": True},
    )

    assert resolved.status == "RESOLVED"
    assert resolved.resolution is not None
    assert resolved.resolution.decision == "APPROVE"
    assert resolved.resolution.receipt_id is not None

    # Weryfikacja obecności w MerkleLedger
    assert resolved.resolution.receipt_id in ledger.receipts
    receipt = ledger.receipts[resolved.resolution.receipt_id]
    assert receipt.ambassador_sealed is True

    # Metryki
    metrics = hitl_mgr.get_metrics()
    assert metrics["resolved_count"] == 1
    assert metrics["total_approved"] == 1
    assert metrics["total_rejected"] == 0
    assert metrics["approval_rate"] == 1.0


def test_hitl_resolve_errors(hitl_mgr):
    """Błędy przy próbie rozwiązania nieznanego lub dwukrotnie tego samego biletu."""
    ticket = hitl_mgr.enqueue_ticket("agent_x", "tool_x", {})
    hitl_mgr.resolve_ticket(ticket.ticket_id, "rev_1", "REJECT", "Odrzucono")

    # Próba ponownego rozwiązania
    with pytest.raises(ValueError):
        hitl_mgr.resolve_ticket(ticket.ticket_id, "rev_2", "APPROVE", "Drugi raz")

    # Nieistniejący bilet
    with pytest.raises(KeyError):
        hitl_mgr.resolve_ticket("non_existent_id", "rev_1", "APPROVE", "Notes")


# ==============================================================================
# 2. TESTY MULTI-REGION CLUSTER SYNC PROTOCOL
# ==============================================================================

def test_cluster_checkpoint_creation_and_pqc_verification(cluster_sync):
    """Punkt kontrolny jest generowany z prawidłowym podpisem Dilithium3."""
    checkpoint = cluster_sync.create_checkpoint()

    assert checkpoint.merkle_root == cluster_sync.ledger.current_root
    assert checkpoint.total_blocks == cluster_sync.ledger.total_blocks
    assert "CRYSTALS-Dilithium3" in checkpoint.pqc_algorithm

    # Weryfikacja podpisu
    is_valid = cluster_sync.verify_peer_checkpoint(checkpoint)
    assert is_valid is True


def test_cluster_reconcile_synchronized_and_divergent(cluster_sync):
    """Rekonsyliacja wykrywa stan zsynchronizowany, przewagę bloków oraz rozwidlenia (Fork)."""
    # 1. Identyczny stan -> SYNCHRONIZED
    chk_sync = cluster_sync.create_checkpoint()
    res_sync = cluster_sync.reconcile_peer(chk_sync)
    assert res_sync.is_consistent is True
    assert res_sync.status == "SYNCHRONIZED"

    # 2. Węzeł zewnętrzny wyprzedza -> PEER_AHEAD
    chk_ahead = cluster_sync.create_checkpoint()
    chk_ahead.total_blocks += 5
    # Podpisujemy zaktualizowany stan
    payload = f"{chk_ahead.node.node_id}:{chk_ahead.merkle_root}:{chk_ahead.total_blocks}".encode("utf-8")
    sig_ahead = cluster_sync.ledger.pqc_dilithium.sign(
        payload, cluster_sync.ledger.keypair.private_key, cluster_sync.ledger.keypair.key_id
    )
    chk_ahead.pqc_signature = sig_ahead.signature.hex()

    res_ahead = cluster_sync.reconcile_peer(chk_ahead)
    assert res_ahead.is_consistent is False
    assert res_ahead.status == "PEER_AHEAD"

    # 3. Równa liczba bloków, lecz różny korzeń Merkle -> DIVERGENT_FORK
    chk_fork = cluster_sync.create_checkpoint()
    chk_fork.merkle_root = "f" * 64
    payload_fork = f"{chk_fork.node.node_id}:{chk_fork.merkle_root}:{chk_fork.total_blocks}".encode("utf-8")
    sig_fork = cluster_sync.ledger.pqc_dilithium.sign(
        payload_fork, cluster_sync.ledger.keypair.private_key, cluster_sync.ledger.keypair.key_id
    )
    chk_fork.pqc_signature = sig_fork.signature.hex()

    res_fork = cluster_sync.reconcile_peer(chk_fork)
    assert res_fork.is_consistent is False
    assert res_fork.status == "DIVERGENT_FORK"


def test_cluster_reconcile_invalid_signature(cluster_sync):
    """Sfałszowany punkt kontrolny zostaje natychmiast odrzucony jako INVALID_SIGNATURE."""
    chk = cluster_sync.create_checkpoint()
    chk.merkle_root = "0" * 64  # Podmiana treści bez ważnego podpisu

    res = cluster_sync.reconcile_peer(chk)
    assert res.is_consistent is False
    assert res.status == "INVALID_SIGNATURE"


# ==============================================================================
# 3. TESTY ENDPOINTÓW FASTAPI (HITL & CLUSTER)
# ==============================================================================

def test_api_hitl_endpoints(client):
    """Weryfikuje endpointy /api/v1/hitl/queue, /enqueue, /resolve, /ticket/{id}, /metrics."""
    # 1. Enqueue
    enq_res = client.post(
        "/api/v1/hitl/enqueue",
        json={
            "agent_id": "test_bot_api",
            "tool_name": "database_migration",
            "arguments": {"dry_run": False},
            "priority": "HIGH",
            "reasons": ["Modyfikacja schematu produkcyjnego"],
        },
    )
    assert enq_res.status_code == 200
    ticket_id = enq_res.json()["ticket_id"]

    # 2. Queue inspect
    queue_res = client.get("/api/v1/hitl/queue")
    assert queue_res.status_code == 200
    assert any(t["ticket_id"] == ticket_id for t in queue_res.json()["tickets"])

    # 3. Specific ticket
    t_res = client.get(f"/api/v1/hitl/ticket/{ticket_id}")
    assert t_res.status_code == 200
    assert t_res.json()["status"] == "PENDING"

    # 4. Resolve
    res_res = client.post(
        "/api/v1/hitl/resolve",
        json={
            "ticket_id": ticket_id,
            "reviewer_id": "auditor_sec_lead",
            "decision": "APPROVE",
            "notes": "Plan migracji zweryfikowany ze snapshotem bazy danych.",
        },
    )
    assert res_res.status_code == 200
    assert res_res.json()["status"] == "RESOLVED"

    # 5. Metrics
    metrics_res = client.get("/api/v1/hitl/metrics")
    assert metrics_res.status_code == 200
    assert metrics_res.json()["total_approved"] >= 1


def test_api_cluster_endpoints(client):
    """Weryfikuje endpointy /api/v1/cluster/nodes, /checkpoint, /peers/register, /reconcile."""
    # 1. Nodes topology
    nodes_res = client.get("/api/v1/cluster/nodes")
    assert nodes_res.status_code == 200
    assert "local_node" in nodes_res.json()

    # 2. Register peer
    reg_res = client.post(
        "/api/v1/cluster/peers/register",
        json={
            "node_id": "nethical-uk-south-1",
            "region": "uk-south-1",
            "datacenter": "london-dc1",
            "public_key_id": "key_uk_south_01",
            "endpoint_url": "https://uk-south.nethical.internal",
        },
    )
    assert reg_res.status_code == 200
    assert reg_res.json()["registered"] is True

    # 3. Checkpoint
    chk_res = client.post("/api/v1/cluster/checkpoint")
    assert chk_res.status_code == 200
    chk_data = chk_res.json()
    assert "pqc_signature" in chk_data

    # 4. Reconcile
    rec_res = client.post(
        "/api/v1/cluster/reconcile",
        json={"checkpoint": chk_data},
    )
    assert rec_res.status_code == 200
    assert rec_res.json()["is_consistent"] is True
    assert rec_res.json()["status"] == "SYNCHRONIZED"


def test_portal_stats_includes_hitl_and_cluster(client):
    """Weryfikuje obecność metryk HITL i klastra w /api/v1/portal/stats."""
    res = client.get("/api/v1/portal/stats")
    assert res.status_code == 200
    stats = res.json()
    assert "hitl" in stats
    assert "cluster" in stats
    assert "pending_count" in stats["hitl"]
    assert "local_node" in stats["cluster"]
