"""
Synchronization Module for Multi-Region Deployment

Provides Conflict-Free Replicated Data Types (CRDTs) for consistent
multi-region policy state synchronization without coordination.

Components:
- PolicyCRDT: CRDT for policy state synchronization
- VectorClock: Vector clocks for causal ordering
- AntiEntropy: Background sync for eventual consistency
- SyncProtocol: Edge-to-cloud synchronization protocol

Guarantees:
- Eventual consistency
- Conflict-free merging
- Offline support
- No coordination required
"""

from .vector_clock import VectorClock, HybridLogicalClock, EventOrder
from .crdt import (
    GCounter,
    PNCounter,
    LWWRegister,
    ORSet,
    MVRegister,
    PolicyCRDT,
    PolicyState,
    PolicyDelta,
    CRDTMergeResult,
)
from .anti_entropy import (
    AntiEntropyProtocol,
    SyncSession,
    SyncState,
    MerkleTree,
    DigestNode,
)

__all__ = [
    # Vector Clocks
    "VectorClock",
    "HybridLogicalClock",
    "EventOrder",
    # CRDTs
    "GCounter",
    "PNCounter",
    "LWWRegister",
    "ORSet",
    "MVRegister",
    "PolicyCRDT",
    "PolicyState",
    "PolicyDelta",
    "CRDTMergeResult",
    # Anti-Entropy
    "AntiEntropyProtocol",
    "SyncSession",
    "SyncState",
    "MerkleTree",
    "DigestNode",
]
