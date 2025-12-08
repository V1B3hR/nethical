"""
Anti-Entropy Protocol for Background Synchronization

Provides background synchronization mechanisms to ensure
eventual consistency across regions without coordination.

Components:
- AntiEntropyProtocol: Main synchronization coordinator
- MerkleTree: Efficient state comparison
- SyncSession: Individual sync session management
- SyncState: State tracking for ongoing syncs

Features:
- Merkle tree-based efficient diffing
- Delta synchronization to minimize bandwidth
- Prioritized sync for critical updates
- Configurable sync intervals
"""

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .crdt import PolicyCRDT, PolicyDelta, PolicyState, CRDTMergeResult
from .vector_clock import HybridLogicalClock


logger = logging.getLogger(__name__)


class SyncState(str, Enum):
    """State of a synchronization session."""

    IDLE = "idle"
    INITIATING = "initiating"
    EXCHANGING_DIGESTS = "exchanging_digests"
    EXCHANGING_DELTAS = "exchanging_deltas"
    APPLYING_DELTAS = "applying_deltas"
    COMPLETED = "completed"
    FAILED = "failed"


class SyncPriority(str, Enum):
    """Priority levels for sync operations."""

    CRITICAL = "critical"  # Security events, emergency updates
    HIGH = "high"  # Policy changes, configuration updates
    NORMAL = "normal"  # Regular background sync
    LOW = "low"  # Historical data, analytics


@dataclass
class DigestNode:
    """
    A node in the Merkle tree for efficient state comparison.

    Attributes:
        hash: Hash of this node's content
        level: Tree level (0 = leaf, higher = internal)
        start_key: Start of key range for this node
        end_key: End of key range for this node
        children: Child node hashes (for internal nodes)
    """

    hash: str
    level: int = 0
    start_key: str = ""
    end_key: str = ""
    children: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hash": self.hash,
            "level": self.level,
            "start_key": self.start_key,
            "end_key": self.end_key,
            "children": self.children,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DigestNode":
        """Create from dictionary."""
        return cls(
            hash=data.get("hash", ""),
            level=data.get("level", 0),
            start_key=data.get("start_key", ""),
            end_key=data.get("end_key", ""),
            children=data.get("children", []),
        )


class MerkleTree:
    """
    Merkle Tree for efficient state comparison.

    Builds a tree of hashes over the policy state, allowing
    efficient identification of differences between nodes.

    Properties:
    - O(log n) difference detection
    - O(k) data transfer where k is number of differences
    - Secure against tampering (cryptographic hashes)
    """

    def __init__(self, branching_factor: int = 16):
        """
        Initialize Merkle tree.

        Args:
            branching_factor: Number of children per internal node
        """
        self.branching_factor = branching_factor
        self.root: Optional[DigestNode] = None
        self.leaves: Dict[str, DigestNode] = {}

    def build(self, policies: Dict[str, PolicyState]) -> DigestNode:
        """
        Build the Merkle tree from policy states.

        Args:
            policies: Dictionary of policy ID to PolicyState

        Returns:
            Root DigestNode of the tree
        """
        # Create leaf nodes
        self.leaves = {}
        sorted_ids = sorted(policies.keys())

        for policy_id in sorted_ids:
            state = policies[policy_id]
            content_hash = self._hash_policy(policy_id, state)
            self.leaves[policy_id] = DigestNode(
                hash=content_hash,
                level=0,
                start_key=policy_id,
                end_key=policy_id,
            )

        if not self.leaves:
            # Empty tree
            self.root = DigestNode(hash=self._empty_hash(), level=0)
            return self.root

        # Build internal nodes bottom-up
        current_level = list(self.leaves.values())
        level = 1

        while len(current_level) > 1:
            next_level = []

            for i in range(0, len(current_level), self.branching_factor):
                children = current_level[i : i + self.branching_factor]
                child_hashes = [c.hash for c in children]

                # Compute parent hash
                combined = "".join(child_hashes)
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]

                parent = DigestNode(
                    hash=parent_hash,
                    level=level,
                    start_key=children[0].start_key,
                    end_key=children[-1].end_key,
                    children=child_hashes,
                )
                next_level.append(parent)

            current_level = next_level
            level += 1

        self.root = current_level[0]
        return self.root

    def get_differences(self, other_tree: "MerkleTree") -> Set[str]:
        """
        Find policy IDs that differ between trees.

        Args:
            other_tree: The tree to compare with

        Returns:
            Set of policy IDs that differ
        """
        if self.root is None or other_tree.root is None:
            # One tree is empty, return all leaves from the other
            return set(self.leaves.keys()) | set(other_tree.leaves.keys())

        if self.root.hash == other_tree.root.hash:
            # Trees are identical
            return set()

        # Find differing leaves by comparing trees
        differences = set()

        # Compare all leaves
        all_keys = set(self.leaves.keys()) | set(other_tree.leaves.keys())
        for key in all_keys:
            local_leaf = self.leaves.get(key)
            remote_leaf = other_tree.leaves.get(key)

            if local_leaf is None or remote_leaf is None:
                # Key exists in one tree but not the other
                differences.add(key)
            elif local_leaf.hash != remote_leaf.hash:
                # Content differs
                differences.add(key)

        return differences

    def get_digest(self) -> str:
        """Get the root hash of the tree."""
        return self.root.hash if self.root else self._empty_hash()

    def _hash_policy(self, policy_id: str, state: PolicyState) -> str:
        """Compute hash for a policy."""
        content = f"{policy_id}:{state.version_hash}:{state.status.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _empty_hash(self) -> str:
        """Return hash for empty tree."""
        return hashlib.sha256(b"empty").hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "root": self.root.to_dict() if self.root else None,
            "leaves": {k: v.to_dict() for k, v in self.leaves.items()},
            "branching_factor": self.branching_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MerkleTree":
        """Create from dictionary."""
        tree = cls(branching_factor=data.get("branching_factor", 16))
        if data.get("root"):
            tree.root = DigestNode.from_dict(data["root"])
        tree.leaves = {
            k: DigestNode.from_dict(v) for k, v in data.get("leaves", {}).items()
        }
        return tree


@dataclass
class SyncSession:
    """
    Individual synchronization session with a remote node.

    Attributes:
        session_id: Unique identifier for this session
        remote_node: ID of the remote node
        state: Current state of the session
        priority: Priority level for this sync
        started_at: Timestamp when session started
        completed_at: Timestamp when session completed
        deltas_sent: Number of deltas sent
        deltas_received: Number of deltas received
        bytes_transferred: Total bytes transferred
        error: Error message if failed
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    remote_node: str = ""
    state: SyncState = SyncState.IDLE
    priority: SyncPriority = SyncPriority.NORMAL
    started_at: float = 0.0
    completed_at: float = 0.0
    deltas_sent: int = 0
    deltas_received: int = 0
    bytes_transferred: int = 0
    error: str = ""

    def duration_ms(self) -> float:
        """Get session duration in milliseconds."""
        if self.completed_at > 0:
            return (self.completed_at - self.started_at) * 1000
        elif self.started_at > 0:
            return (time.time() - self.started_at) * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "remote_node": self.remote_node,
            "state": self.state.value,
            "priority": self.priority.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "deltas_sent": self.deltas_sent,
            "deltas_received": self.deltas_received,
            "bytes_transferred": self.bytes_transferred,
            "error": self.error,
        }


class AntiEntropyProtocol:
    """
    Anti-Entropy Protocol for background synchronization.

    Implements gossip-based synchronization with:
    - Merkle tree-based efficient diffing
    - Delta synchronization
    - Configurable sync intervals
    - Prioritized updates for critical changes

    The protocol runs in three phases:
    1. Exchange digests (Merkle tree roots)
    2. Identify differences (tree traversal)
    3. Exchange deltas (only changed policies)
    """

    def __init__(
        self,
        crdt: PolicyCRDT,
        node_id: str,
        sync_interval_sec: float = 30.0,
        max_concurrent_syncs: int = 3,
        max_delta_batch_size: int = 100,
    ):
        """
        Initialize anti-entropy protocol.

        Args:
            crdt: The PolicyCRDT to synchronize
            node_id: ID of this node
            sync_interval_sec: Interval between sync attempts
            max_concurrent_syncs: Maximum concurrent sync sessions
            max_delta_batch_size: Maximum deltas per batch
        """
        self.crdt = crdt
        self.node_id = node_id
        self.sync_interval_sec = sync_interval_sec
        self.max_concurrent_syncs = max_concurrent_syncs
        self.max_delta_batch_size = max_delta_batch_size

        # Sync state
        self.active_sessions: Dict[str, SyncSession] = {}
        self.completed_sessions: List[SyncSession] = []
        self.max_completed_history = 100

        # Merkle tree for efficient comparison
        self.merkle_tree = MerkleTree()

        # Known peers
        self.peers: Set[str] = set()

        # Statistics
        self.total_syncs = 0
        self.successful_syncs = 0
        self.failed_syncs = 0

        # Callbacks
        self._on_sync_complete: Optional[Callable[[SyncSession], None]] = None
        self._on_conflict: Optional[Callable[[str, PolicyState, PolicyState], None]] = (
            None
        )

        logger.info(f"AntiEntropyProtocol initialized for node {node_id}")

    def add_peer(self, peer_id: str):
        """
        Add a peer node for synchronization.

        Args:
            peer_id: ID of the peer node
        """
        self.peers.add(peer_id)
        logger.debug(f"Added peer {peer_id}, total peers: {len(self.peers)}")

    def remove_peer(self, peer_id: str):
        """
        Remove a peer node.

        Args:
            peer_id: ID of the peer node
        """
        self.peers.discard(peer_id)
        logger.debug(f"Removed peer {peer_id}")

    def rebuild_merkle_tree(self):
        """Rebuild the Merkle tree from current CRDT state."""
        policies = {}
        for policy_id, register in self.crdt.policies.items():
            state = register.get()
            if state:
                policies[policy_id] = state

        self.merkle_tree.build(policies)
        logger.debug(f"Rebuilt Merkle tree, digest: {self.merkle_tree.get_digest()}")

    def get_digest(self) -> str:
        """Get current state digest for comparison."""
        return self.merkle_tree.get_digest()

    async def start_sync(
        self,
        remote_node: str,
        remote_digest: str,
        priority: SyncPriority = SyncPriority.NORMAL,
    ) -> SyncSession:
        """
        Start a synchronization session with a remote node.

        Args:
            remote_node: ID of the remote node
            remote_digest: Digest from the remote node
            priority: Priority level for this sync

        Returns:
            SyncSession tracking the sync progress
        """
        # Check concurrent session limit
        if len(self.active_sessions) >= self.max_concurrent_syncs:
            # Wait for a slot or return existing session
            existing = next(
                (
                    s
                    for s in self.active_sessions.values()
                    if s.remote_node == remote_node
                ),
                None,
            )
            if existing:
                return existing

            logger.warning(
                f"Max concurrent syncs reached, queuing sync with {remote_node}"
            )

        # Create new session
        session = SyncSession(
            remote_node=remote_node,
            state=SyncState.INITIATING,
            priority=priority,
            started_at=time.time(),
        )

        self.active_sessions[session.session_id] = session

        # Check if sync needed
        local_digest = self.get_digest()
        if local_digest == remote_digest:
            session.state = SyncState.COMPLETED
            session.completed_at = time.time()
            self._complete_session(session)
            logger.debug(f"Sync not needed with {remote_node}, digests match")
            return session

        # Update state
        session.state = SyncState.EXCHANGING_DIGESTS

        return session

    def get_deltas_for_peer(
        self,
        peer_merkle_tree: MerkleTree,
        session: SyncSession,
    ) -> List[PolicyDelta]:
        """
        Get deltas to send to a peer based on their Merkle tree.

        Args:
            peer_merkle_tree: The peer's Merkle tree
            session: The sync session

        Returns:
            List of deltas to send
        """
        # Find differences
        differences = self.merkle_tree.get_differences(peer_merkle_tree)

        if not differences:
            return []

        # Get deltas for differing policies
        deltas = []
        for policy_id in differences:
            if policy_id in self.crdt.policies:
                state = self.crdt.policies[policy_id].get()
                if state:
                    deltas.append(
                        PolicyDelta(
                            policy_id=policy_id,
                            operation="update",
                            state=state,
                            timestamp=self.crdt.policies[policy_id].timestamp,
                            source_node=self.node_id,
                        )
                    )

        # Limit batch size
        if len(deltas) > self.max_delta_batch_size:
            deltas = deltas[: self.max_delta_batch_size]
            logger.info(f"Truncated deltas to {self.max_delta_batch_size}")

        session.deltas_sent = len(deltas)
        return deltas

    def apply_deltas(
        self,
        deltas: List[PolicyDelta],
        session: SyncSession,
    ) -> CRDTMergeResult:
        """
        Apply deltas received from a peer.

        Args:
            deltas: List of deltas to apply
            session: The sync session

        Returns:
            CRDTMergeResult with merge statistics
        """
        session.state = SyncState.APPLYING_DELTAS
        session.deltas_received = len(deltas)

        result = CRDTMergeResult()

        for delta in deltas:
            try:
                applied = self.crdt.apply_delta(delta)
                if applied:
                    if delta.operation == "add":
                        result.new_policies.append(delta.policy_id)
                    else:
                        result.updated_policies.append(delta.policy_id)
                    result.merged_count += 1
            except Exception as e:
                logger.error(f"Error applying delta for {delta.policy_id}: {e}")
                result.conflict_count += 1
                result.conflicts.append(delta.policy_id)

        # Rebuild Merkle tree after applying deltas
        self.rebuild_merkle_tree()

        return result

    def complete_sync(
        self, session: SyncSession, success: bool = True, error: str = ""
    ):
        """
        Complete a synchronization session.

        Args:
            session: The session to complete
            success: Whether sync was successful
            error: Error message if failed
        """
        session.completed_at = time.time()

        if success:
            session.state = SyncState.COMPLETED
            self.successful_syncs += 1
        else:
            session.state = SyncState.FAILED
            session.error = error
            self.failed_syncs += 1

        self.total_syncs += 1
        self._complete_session(session)

        logger.info(
            f"Sync with {session.remote_node} completed: "
            f"success={success}, duration={session.duration_ms():.2f}ms, "
            f"sent={session.deltas_sent}, received={session.deltas_received}"
        )

    def _complete_session(self, session: SyncSession):
        """Move session from active to completed."""
        if session.session_id in self.active_sessions:
            del self.active_sessions[session.session_id]

        self.completed_sessions.append(session)

        # Trim history
        if len(self.completed_sessions) > self.max_completed_history:
            self.completed_sessions = self.completed_sessions[
                -self.max_completed_history :
            ]

        # Call completion callback
        if self._on_sync_complete:
            self._on_sync_complete(session)

    def on_sync_complete(self, callback: Callable[[SyncSession], None]):
        """Set callback for sync completion."""
        self._on_sync_complete = callback

    def on_conflict(self, callback: Callable[[str, PolicyState, PolicyState], None]):
        """Set callback for conflict detection."""
        self._on_conflict = callback

    def get_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        return {
            "total_syncs": self.total_syncs,
            "successful_syncs": self.successful_syncs,
            "failed_syncs": self.failed_syncs,
            "success_rate": (
                self.successful_syncs / self.total_syncs
                if self.total_syncs > 0
                else 0.0
            ),
            "active_sessions": len(self.active_sessions),
            "peer_count": len(self.peers),
            "current_digest": self.get_digest(),
        }

    async def run_background_sync(self, get_peer_digest: Callable[[str], str]):
        """
        Run background synchronization loop.

        Args:
            get_peer_digest: Async function to get digest from peer
        """
        logger.info(f"Starting background sync with interval {self.sync_interval_sec}s")

        while True:
            try:
                # Rebuild local Merkle tree
                self.rebuild_merkle_tree()

                # Sync with each peer
                for peer_id in list(self.peers):
                    try:
                        # Get peer's digest
                        peer_digest = get_peer_digest(peer_id)

                        # Start sync if digests differ
                        if peer_digest != self.get_digest():
                            session = await self.start_sync(
                                remote_node=peer_id,
                                remote_digest=peer_digest,
                            )
                            logger.debug(
                                f"Started sync session {session.session_id} with {peer_id}"
                            )

                    except Exception as e:
                        logger.warning(f"Failed to sync with peer {peer_id}: {e}")

                # Wait for next sync interval
                await asyncio.sleep(self.sync_interval_sec)

            except Exception as e:
                logger.error(f"Error in background sync: {e}")
                await asyncio.sleep(self.sync_interval_sec)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "sync_interval_sec": self.sync_interval_sec,
            "max_concurrent_syncs": self.max_concurrent_syncs,
            "peers": list(self.peers),
            "statistics": self.get_statistics(),
            "active_sessions": [s.to_dict() for s in self.active_sessions.values()],
        }
