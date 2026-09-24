# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Nethical Contributors

"""Comprehensive unit and integration tests for nethical.sync module."""

import pytest

from nethical.sync import (
    VectorClock,
    HybridLogicalClock,
    EventOrder,
    GCounter,
    PNCounter,
    LWWRegister,
    ORSet,
    MVRegister,
    PolicyCRDT,
    PolicyState,
    PolicyDelta,
    CRDTMergeResult,
    AntiEntropyProtocol,
    SyncSession,
    SyncState,
    MerkleTree,
    DigestNode,
)
from nethical.sync.crdt import PolicyStatus


def test_vector_clock_causality_and_merging():
    """Verify vector clocks accurately track causal ordering and concurrent events."""
    vc1 = VectorClock(node_id="eu-central")
    vc2 = VectorClock(node_id="us-east")

    # Local increments
    vc1.increment()
    vc2.increment()

    # Both incremented independently -> concurrent
    assert vc1.compare(vc2) == EventOrder.CONCURRENT
    assert not (vc1 < vc2)
    assert not (vc1 > vc2)

    # Simulate message from vc1 to vc2
    vc1.send_event()
    vc2.receive_event(vc1)

    # vc1 happened-before vc2
    assert vc1 < vc2
    assert vc2 > vc1
    assert vc1.compare(vc2) == EventOrder.BEFORE
    assert vc2.compare(vc1) == EventOrder.AFTER

    # Roundtrip serialization
    d = vc2.to_dict()
    restored = VectorClock.from_dict(d)
    assert restored == vc2
    assert restored.clock == vc2.clock


def test_hybrid_logical_clock_monotonous_and_drift():
    """Verify HLC monotonicity, timestamp bit packing, and comparison."""
    hlc_a = HybridLogicalClock(node_id="node_a")
    hlc_b = HybridLogicalClock(node_id="node_b")

    t1_p, t1_l = hlc_a.now()
    t2_p, t2_l = hlc_a.now()
    assert (t2_p > t1_p) or (t2_p == t1_p and t2_l > t1_l)

    # Bit-packed 64-bit integer
    packed = hlc_a.timestamp()
    unpacked = HybridLogicalClock.from_timestamp(packed, node_id="node_a")
    assert unpacked.physical == hlc_a.physical
    assert unpacked.logical == hlc_a.logical

    # Receive from remote
    hlc_b.receive(hlc_a.physical, hlc_a.logical)
    assert hlc_b >= hlc_a


def test_gc_and_pn_counters():
    """Verify GCounter (grow-only) and PNCounter (positive-negative) convergence."""
    # GCounter
    g1 = GCounter(node_id="node1")
    g2 = GCounter(node_id="node2")

    g1.increment(10)
    g2.increment(5)
    g1.merge(g2)
    assert g1.value() == 15

    with pytest.raises(ValueError, match="can only be incremented"):
        g1.increment(-1)

    # Anonymous GCounter
    g_anon = GCounter()
    g_anon.increment(3)
    assert g_anon.value() == 3

    # PNCounter
    pn1 = PNCounter(node_id="node1")
    pn2 = PNCounter(node_id="node2")

    pn1.increment(20)
    pn1.decrement(5)
    assert pn1.value() == 15

    pn2.decrement(3)
    pn1.merge(pn2)
    assert pn1.value() == 12

    # Roundtrip serialization
    pn_restored = PNCounter.from_dict(pn1.to_dict())
    assert pn_restored.value() == 12


def test_lww_register_convergence():
    """Verify LWWRegister resolves concurrent writes by latest timestamp."""
    reg1 = LWWRegister[str](node_id="node_1")
    reg2 = LWWRegister[str](node_id="node_2")

    reg1.set("val_1")
    time_hlc = HybridLogicalClock(physical=reg1.timestamp.physical + 1000, logical=0, node_id="node_2")
    reg2.value = "val_2"
    reg2.timestamp = time_hlc

    # reg2 is newer than reg1
    reg1.merge(reg2)
    assert reg1.get() == "val_2"


def test_or_set_add_wins():
    """Verify OR-Set allows concurrent add/remove where add wins."""
    s1 = ORSet[str](node_id="node1")
    s2 = ORSet[str](node_id="node2")

    s1.add("policy_a")
    s1.add("policy_b")
    s2.merge(s1)
    assert s2.contains("policy_a")
    assert s2.contains("policy_b")

    # s1 removes policy_a, but s2 concurrently adds policy_a again
    s1.remove("policy_a")
    s2.add("policy_a")

    s1.merge(s2)
    # Add wins!
    assert s1.contains("policy_a")
    assert s1.contains("policy_b")


def test_mv_register_conflicts():
    """Verify MVRegister preserves concurrent values until causal resolution."""
    mv1 = MVRegister[str](node_id="node1")
    mv2 = MVRegister[str](node_id="node2")

    mv1.set("policy_v1")
    mv2.set("policy_v2")

    mv1.merge(mv2)
    assert mv1.has_conflict() is True
    assert set(mv1.get()) == {"policy_v1", "policy_v2"}

    # Resolving conflict with newer write
    mv1.set("policy_resolved")
    assert mv1.has_conflict() is False
    assert mv1.get_single() == "policy_resolved"


def test_policy_crdt_lifecycle_and_multiregion_merge():
    """Verify PolicyCRDT handles addition, deprecation, deletion, and cross-region merge."""
    crdt_eu = PolicyCRDT(node_id="eu-region")
    crdt_us = PolicyCRDT(node_id="us-region")

    p1 = PolicyState(
        policy_id="pol-001",
        content={"action": "block_pii", "level": "high"},
        status=PolicyStatus.ACTIVE,
    )
    p2 = PolicyState(
        policy_id="pol-002",
        content={"action": "rate_limit", "max_qps": 100},
        status=PolicyStatus.ACTIVE,
    )

    crdt_eu.add_policy(p1)
    crdt_us.add_policy(p2)

    # Initial states before merge
    assert crdt_eu.get_policy("pol-001") is not None
    assert crdt_eu.get_policy("pol-002") is None
    assert len(crdt_eu.get_active_policies()) == 1

    # Merge US state into EU
    merge_result = crdt_eu.merge(crdt_us)
    assert "pol-002" in merge_result.new_policies
    assert len(crdt_eu.get_active_policies()) == 2

    # Deprecate pol-001 in EU
    crdt_eu.deprecate_policy("pol-001")
    assert crdt_eu.get_policy("pol-001").status == PolicyStatus.DEPRECATED
    active_ids = [p.policy_id for p in crdt_eu.get_active_policies()]
    assert "pol-001" not in active_ids
    assert "pol-002" in active_ids

    # Soft delete pol-002 in EU
    crdt_eu.delete_policy("pol-002")
    assert crdt_eu.get_policy("pol-002").status == PolicyStatus.DELETED
    assert len(crdt_eu.get_active_policies()) == 0

    # Serialization roundtrip
    data = crdt_eu.to_dict()
    restored = PolicyCRDT.from_dict(data)
    assert restored.node_id == "eu-region"
    assert restored.get_policy("pol-001").status == PolicyStatus.DEPRECATED


def test_merkle_tree_and_anti_entropy():
    """Verify MerkleTree differential sync and AntiEntropyProtocol delta exchange."""
    crdt1 = PolicyCRDT(node_id="cluster-1")
    crdt2 = PolicyCRDT(node_id="cluster-2")

    crdt1.add_policy(PolicyState("P1", {"rule": 1}, status=PolicyStatus.ACTIVE))
    crdt1.add_policy(PolicyState("P2", {"rule": 2}, status=PolicyStatus.ACTIVE))
    crdt2.add_policy(PolicyState("P1", {"rule": 1}, status=PolicyStatus.ACTIVE))
    crdt2.add_policy(PolicyState("P3", {"rule": 3}, status=PolicyStatus.ACTIVE))

    ae1 = AntiEntropyProtocol(crdt=crdt1, node_id="cluster-1")
    ae2 = AntiEntropyProtocol(crdt=crdt2, node_id="cluster-2")

    ae1.rebuild_merkle_tree()
    ae2.rebuild_merkle_tree()

    digest1 = ae1.get_digest()
    digest2 = ae2.get_digest()
    assert digest1 != digest2

    # Detect differences using Merkle trees
    diffs = ae1.merkle_tree.get_differences(ae2.merkle_tree)
    assert "P2" in diffs  # In cluster-1 but not cluster-2
    assert "P3" in diffs  # In cluster-2 but not cluster-1
    assert "P1" not in diffs  # Same in both

    # Run Anti-Entropy exchange
    async def run_sync():
        session = await ae1.start_sync("cluster-2", digest2)
        assert session.state == SyncState.EXCHANGING_DIGESTS

        # Generate deltas from cluster-1 for cluster-2
        deltas = ae1.get_deltas_for_peer(ae2.merkle_tree, session)
        assert any(d.policy_id == "P2" for d in deltas)

        # Apply deltas to cluster-2
        result = ae2.apply_deltas(deltas, session)
        assert "P2" in result.new_policies
        assert crdt2.get_policy("P2") is not None

        ae1.complete_sync(session, success=True)
        assert session.state == SyncState.COMPLETED
        stats = ae1.get_statistics()
        assert stats["successful_syncs"] == 1

    import asyncio
    asyncio.run(run_sync())
