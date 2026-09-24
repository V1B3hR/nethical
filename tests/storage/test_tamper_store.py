# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Unit and integration tests for TamperStore and MerkleAppender."""

import concurrent.futures
import pytest

from nethical.storage import (
    TamperStore,
    TamperEvidentOfflineStore,
    Event,
    Anchor,
    MerkleAppender,
    TamperStoreError,
)


def test_tamper_store_alias():
    """Verify TamperStore is an exact alias for TamperEvidentOfflineStore."""
    assert TamperStore is TamperEvidentOfflineStore


def test_merkle_appender_basics():
    """Test core Merkle tree operations with empty, single, and multiple leaves."""
    tree = MerkleAppender(algorithm="sha256")
    assert tree.size == 0
    assert tree.root() is None

    # Single leaf
    leaf0 = tree.add_leaf(b"leaf-0")
    assert tree.size == 1
    assert tree.root() == leaf0
    assert tree.leaves == [leaf0]

    # Two leaves
    leaf1 = tree.add_leaf(b"leaf-1")
    assert tree.size == 2
    root2 = tree.root()
    assert root2 is not None and root2 != leaf0 and root2 != leaf1

    # Three leaves (odd number)
    leaf2 = tree.add_leaf(b"leaf-2")
    assert tree.size == 3
    root3 = tree.root()
    assert root3 is not None

    # Verify proofs for each leaf
    for i in range(3):
        proof = tree.prove(i)
        assert tree.verify(tree.leaves[i], proof, expected_root=root3) == root3

    # Verification failure on bad root
    with pytest.raises(TamperStoreError, match="Proof verification failed"):
        tree.verify(tree.leaves[0], tree.prove(0), expected_root="badroot" * 8)

    # Out of range index
    with pytest.raises(TamperStoreError, match="Leaf index out of range"):
        tree.prove(99)
    with pytest.raises(TamperStoreError, match="Leaf index out of range"):
        tree.prove(-1)


def test_tamper_store_append_and_proofs():
    """Test event appending, prev_root chaining, and proof generation."""
    store = TamperStore(digest="sha256")
    assert store.size() == 0
    assert store.root() is None

    # Append first event
    leaf1 = store.append_event({"action": "policy_evaluated", "decision": "ALLOW"}, correlation_id="cid-001")
    assert store.size() == 1
    assert store.root() == leaf1

    ev1 = store.get_event(1)
    assert ev1.seq == 1
    assert ev1.correlation_id == "cid-001"
    assert ev1.prev_root is None
    assert ev1.leaf == leaf1
    assert ev1.payload["decision"] == "ALLOW"

    # Append second event
    leaf2 = store.append_event({"action": "alert_dispatched", "severity": "HIGH"}, correlation_id="cid-002")
    assert store.size() == 2
    root2 = store.root()
    assert root2 is not None

    ev2 = store.get_event(2)
    assert ev2.seq == 2
    assert ev2.correlation_id == "cid-002"
    assert ev2.prev_root == leaf1

    # Append arbitrary bytes
    leaf3 = store.append_bytes(b"signed_policy_binary_blob", correlation_id="cid-003")
    assert store.size() == 3
    ev3 = store.get_event(3)
    assert "_raw_b64" in ev3.payload
    assert ev3.prev_root == root2

    # Verify full store integrity
    assert store.verify_all() is True

    # Test inclusion proofs for all 3 events
    for seq in (1, 2, 3):
        proof_bundle = store.prove(seq)
        assert proof_bundle["seq"] == seq
        assert proof_bundle["root"] == store.root()
        assert TamperStore.verify_proof(proof_bundle) is True

    # Out of range seq
    with pytest.raises(TamperStoreError, match="Sequence out of range"):
        store.get_event(0)
    with pytest.raises(TamperStoreError, match="Sequence out of range"):
        store.get_event(4)


def test_tamper_store_anchoring_and_snapshot():
    """Test anchoring receipts and snapshot summaries."""
    store = TamperStore(tsa_url="https://tsa.sovereign.local/timestamp")
    store.append_event({"event": "auth_login"})
    root_before = store.root()

    ok, root = store.flush_to_remote(anchor=True, anchor_type="tsa", receipt="rfc3161_sample_token")
    assert ok is True
    assert root == root_before

    snap = store.snapshot()
    assert snap["schema_version"] == 1
    assert snap["events"] == 1
    assert snap["last_seq"] == 1
    assert snap["merkle_root"] == root
    assert len(snap["anchors"]) == 1
    assert snap["anchors"][0]["receipt"] == "rfc3161_sample_token"
    assert snap["anchors"][0]["url"] == "https://tsa.sovereign.local/timestamp"


def test_tamper_store_export_import_roundtrip():
    """Test export, serialization, and import with full verification."""
    store = TamperStore(digest="sha256")
    store.append_event({"k1": "v1"}, correlation_id="c1")
    store.append_event({"k2": "v2"}, correlation_id="c2")
    store.append_event({"k3": "v3"}, correlation_id="c3")
    store.flush_to_remote(anchor=True, anchor_type="custom", receipt="anchor_receipt_xyz")

    records = store.export_records()
    assert len(records) == 1 + 3 + 1  # 1 header + 3 events + 1 anchor

    # Import into a new store
    imported_store = TamperStore.import_records(records)
    assert imported_store.size() == 3
    assert imported_store.root() == store.root()
    assert imported_store.verify_all() is True
    assert len(imported_store._anchors) == 1
    assert imported_store._anchors[0].receipt == "anchor_receipt_xyz"

    # Verify generator input handling
    generator_records = (r for r in records)
    gen_imported_store = TamperStore.import_records(generator_records)
    assert gen_imported_store.size() == 3
    assert gen_imported_store.root() == store.root()


def test_tamper_store_import_detects_tampering():
    """Verify that tampering with event payloads, leaves, sequences, or prev_roots fails import."""
    store = TamperStore()
    store.append_event({"key": "val1"})
    store.append_event({"key": "val2"})
    records = store.export_records()

    # 1. Tampering with event payload without changing leaf
    tampered_records_payload = [dict(r) for r in records]
    tampered_records_payload[1]["payload"] = {"key": "tampered_value"}
    with pytest.raises(TamperStoreError, match="leaf mismatch"):
        TamperStore.import_records(tampered_records_payload)

    # 2. Tampering with sequence number
    tampered_records_seq = [dict(r) for r in records]
    tampered_records_seq[2]["seq"] = 99
    with pytest.raises(TamperStoreError, match="Event sequence mismatch"):
        TamperStore.import_records(tampered_records_seq)

    # 3. Tampering with prev_root
    tampered_records_prev = [dict(r) for r in records]
    tampered_records_prev[2]["prev_root"] = "corrupted_prev_root"
    with pytest.raises(TamperStoreError, match="prev_root mismatch"):
        TamperStore.import_records(tampered_records_prev)

    # 4. Unknown record type
    tampered_records_bad_type = [dict(r) for r in records]
    tampered_records_bad_type.append({"type": "invalid_type"})
    with pytest.raises(TamperStoreError, match="Unknown record type"):
        TamperStore.import_records(tampered_records_bad_type)


def test_tamper_store_thread_safety():
    """Verify thread-safety when appending events concurrently."""
    store = TamperStore()
    total_events = 50

    def worker(i: int):
        return store.append_event({"worker": i, "data": f"thread-task-{i}"}, correlation_id=f"cid-{i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(total_events)]
        leaves = [f.result() for f in futures]

    assert len(leaves) == total_events
    assert store.size() == total_events
    assert store.verify_all() is True
