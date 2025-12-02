"""
Conflict-Free Replicated Data Types (CRDTs) for Multi-Region Sync

Provides CRDT implementations for consistent policy state
synchronization across multiple regions without coordination.

Implemented CRDTs:
- GCounter: Grow-only counter
- PNCounter: Positive-negative counter
- LWWRegister: Last-Writer-Wins register
- ORSet: Observed-Remove Set
- MVRegister: Multi-Value Register
- PolicyCRDT: Specialized CRDT for policy state

Guarantees:
- Eventual consistency
- Conflict-free merging
- Offline support
- No coordination required
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union

from .vector_clock import VectorClock, HybridLogicalClock, EventOrder


logger = logging.getLogger(__name__)

T = TypeVar("T")


class PolicyStatus(str, Enum):
    """Status of a policy in the CRDT."""

    QUARANTINE = "quarantine"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


@dataclass
class GCounter:
    """
    Grow-only Counter (G-Counter) CRDT.

    Each node maintains its own counter that can only be incremented.
    The total value is the sum of all node counters.

    Properties:
    - Monotonically increasing
    - Commutative and associative merge
    - Eventually consistent

    Attributes:
        counts: Dictionary mapping node IDs to their counts
        node_id: ID of the local node
    """

    counts: Dict[str, int] = field(default_factory=dict)
    node_id: str = ""

    def increment(self, amount: int = 1) -> "GCounter":
        """
        Increment the local counter.

        Args:
            amount: Amount to increment (must be positive)

        Returns:
            Self for method chaining
        """
        if amount < 0:
            raise ValueError("GCounter can only be incremented")
        if self.node_id:
            self.counts[self.node_id] = self.counts.get(self.node_id, 0) + amount
        return self

    def value(self) -> int:
        """Get the total counter value."""
        return sum(self.counts.values())

    def merge(self, other: "GCounter") -> "GCounter":
        """
        Merge with another G-Counter.

        Takes the maximum for each node ID.

        Args:
            other: The GCounter to merge with

        Returns:
            Self for method chaining
        """
        for node_id, count in other.counts.items():
            self.counts[node_id] = max(self.counts.get(node_id, 0), count)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"counts": dict(self.counts), "node_id": self.node_id}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GCounter":
        """Create from dictionary."""
        return cls(
            counts=dict(data.get("counts", {})),
            node_id=data.get("node_id", ""),
        )


@dataclass
class PNCounter:
    """
    Positive-Negative Counter (PN-Counter) CRDT.

    Combines two G-Counters: one for increments, one for decrements.
    The value is the difference between the two.

    Attributes:
        increments: G-Counter for positive changes
        decrements: G-Counter for negative changes
        node_id: ID of the local node
    """

    increments: GCounter = field(default_factory=GCounter)
    decrements: GCounter = field(default_factory=GCounter)
    node_id: str = ""

    def __post_init__(self):
        """Initialize with node ID."""
        if self.node_id:
            self.increments.node_id = self.node_id
            self.decrements.node_id = self.node_id

    def increment(self, amount: int = 1) -> "PNCounter":
        """Increment the counter."""
        if amount < 0:
            raise ValueError("Use decrement for negative values")
        self.increments.increment(amount)
        return self

    def decrement(self, amount: int = 1) -> "PNCounter":
        """Decrement the counter."""
        if amount < 0:
            raise ValueError("Use increment for negative values")
        self.decrements.increment(amount)
        return self

    def value(self) -> int:
        """Get the current counter value."""
        return self.increments.value() - self.decrements.value()

    def merge(self, other: "PNCounter") -> "PNCounter":
        """Merge with another PN-Counter."""
        self.increments.merge(other.increments)
        self.decrements.merge(other.decrements)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "increments": self.increments.to_dict(),
            "decrements": self.decrements.to_dict(),
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PNCounter":
        """Create from dictionary."""
        return cls(
            increments=GCounter.from_dict(data.get("increments", {})),
            decrements=GCounter.from_dict(data.get("decrements", {})),
            node_id=data.get("node_id", ""),
        )


@dataclass
class LWWRegister(Generic[T]):
    """
    Last-Writer-Wins Register (LWW-Register) CRDT.

    Uses timestamps to determine which write wins in case of concurrent updates.
    The write with the highest timestamp wins.

    Attributes:
        value: The current value
        timestamp: HLC timestamp of the last write
        node_id: ID of the local node
    """

    value: Optional[T] = None
    timestamp: HybridLogicalClock = field(default_factory=HybridLogicalClock)
    node_id: str = ""

    def __post_init__(self):
        """Initialize with node ID."""
        if self.node_id:
            self.timestamp.node_id = self.node_id

    def set(self, value: T) -> "LWWRegister[T]":
        """
        Set a new value.

        Args:
            value: The new value to set

        Returns:
            Self for method chaining
        """
        self.value = value
        self.timestamp.now()
        return self

    def get(self) -> Optional[T]:
        """Get the current value."""
        return self.value

    def merge(self, other: "LWWRegister[T]") -> "LWWRegister[T]":
        """
        Merge with another LWW-Register.

        The register with the higher timestamp wins.

        Args:
            other: The LWWRegister to merge with

        Returns:
            Self for method chaining
        """
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = HybridLogicalClock(
                physical=other.timestamp.physical,
                logical=other.timestamp.logical,
                node_id=self.node_id,
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.to_dict(),
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LWWRegister":
        """Create from dictionary."""
        return cls(
            value=data.get("value"),
            timestamp=HybridLogicalClock.from_dict(data.get("timestamp", {})),
            node_id=data.get("node_id", ""),
        )


@dataclass
class ORSet(Generic[T]):
    """
    Observed-Remove Set (OR-Set) CRDT.

    Also known as Add-Wins Set. Allows adding and removing elements
    while ensuring that concurrent add and remove operations are handled correctly.
    Add operations "win" over concurrent remove operations.

    Each element is tagged with a unique ID to track its presence.

    Attributes:
        elements: Dictionary mapping element values to their unique tags
        tombstones: Set of removed element tags
        node_id: ID of the local node
    """

    elements: Dict[T, Set[str]] = field(default_factory=dict)
    tombstones: Set[str] = field(default_factory=set)
    node_id: str = ""

    def add(self, element: T) -> "ORSet[T]":
        """
        Add an element to the set.

        Args:
            element: The element to add

        Returns:
            Self for method chaining
        """
        # Generate a unique tag for this add operation
        tag = f"{self.node_id}:{uuid.uuid4().hex[:12]}"

        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(tag)

        return self

    def remove(self, element: T) -> "ORSet[T]":
        """
        Remove an element from the set.

        Args:
            element: The element to remove

        Returns:
            Self for method chaining
        """
        if element in self.elements:
            # Add all current tags to tombstones
            self.tombstones.update(self.elements[element])
            del self.elements[element]

        return self

    def contains(self, element: T) -> bool:
        """Check if an element is in the set."""
        if element not in self.elements:
            return False

        # Check if any tag is not tombstoned
        live_tags = self.elements[element] - self.tombstones
        return len(live_tags) > 0

    def to_set(self) -> Set[T]:
        """Get the current set of elements."""
        result = set()
        for element, tags in self.elements.items():
            live_tags = tags - self.tombstones
            if live_tags:
                result.add(element)
        return result

    def merge(self, other: "ORSet[T]") -> "ORSet[T]":
        """
        Merge with another OR-Set.

        Args:
            other: The ORSet to merge with

        Returns:
            Self for method chaining
        """
        # Merge elements (union of all tags)
        for element, tags in other.elements.items():
            if element not in self.elements:
                self.elements[element] = set()
            self.elements[element].update(tags)

        # Merge tombstones
        self.tombstones.update(other.tombstones)

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Convert hashable elements to strings for JSON
        elements_serialized = {
            str(k): list(v) for k, v in self.elements.items()
        }
        return {
            "elements": elements_serialized,
            "tombstones": list(self.tombstones),
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ORSet":
        """Create from dictionary."""
        elements = {
            k: set(v) for k, v in data.get("elements", {}).items()
        }
        return cls(
            elements=elements,
            tombstones=set(data.get("tombstones", [])),
            node_id=data.get("node_id", ""),
        )


@dataclass
class MVRegister(Generic[T]):
    """
    Multi-Value Register (MV-Register) CRDT.

    Preserves all concurrent writes as multiple values.
    Useful when you want to see all conflicting values.

    Attributes:
        values: List of (value, vector_clock) pairs
        node_id: ID of the local node
    """

    values: List[Tuple[T, VectorClock]] = field(default_factory=list)
    node_id: str = ""

    def set(self, value: T, clock: Optional[VectorClock] = None) -> "MVRegister[T]":
        """
        Set a new value.

        Args:
            value: The new value to set
            clock: Optional vector clock (uses new clock if not provided)

        Returns:
            Self for method chaining
        """
        if clock is None:
            # Create a new clock based on existing values
            clock = VectorClock(node_id=self.node_id)
            for _, vc in self.values:
                clock.merge(vc)
            clock.increment()

        # Remove all values that this new value supersedes
        new_values = []
        for existing_value, existing_clock in self.values:
            order = existing_clock.compare(clock)
            if order == EventOrder.CONCURRENT:
                # Keep concurrent values
                new_values.append((existing_value, existing_clock))
            elif order == EventOrder.AFTER:
                # Existing value is newer, keep it
                new_values.append((existing_value, existing_clock))
            # Otherwise, the new value supersedes it

        # Add the new value
        new_values.append((value, clock))
        self.values = new_values

        return self

    def get(self) -> List[T]:
        """Get all current values (may be multiple if concurrent writes)."""
        return [v for v, _ in self.values]

    def get_single(self) -> Optional[T]:
        """Get a single value (arbitrary if multiple concurrent values)."""
        if self.values:
            return self.values[0][0]
        return None

    def has_conflict(self) -> bool:
        """Check if there are conflicting concurrent values."""
        return len(self.values) > 1

    def merge(self, other: "MVRegister[T]") -> "MVRegister[T]":
        """
        Merge with another MV-Register.

        Args:
            other: The MVRegister to merge with

        Returns:
            Self for method chaining
        """
        all_values = self.values + other.values

        # Keep only values that are not superseded by any other value
        result_values = []
        for value, clock in all_values:
            is_superseded = False
            for other_value, other_clock in all_values:
                if value == other_value and clock == other_clock:
                    continue
                if clock.compare(other_clock) == EventOrder.BEFORE:
                    is_superseded = True
                    break

            if not is_superseded:
                # Check if already in result
                already_present = any(
                    v == value and c == clock for v, c in result_values
                )
                if not already_present:
                    result_values.append((value, clock))

        self.values = result_values
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "values": [(v, c.to_dict()) for v, c in self.values],
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MVRegister":
        """Create from dictionary."""
        values = [
            (v, VectorClock.from_dict(c)) for v, c in data.get("values", [])
        ]
        return cls(values=values, node_id=data.get("node_id", ""))


@dataclass
class PolicyState:
    """
    State of a single policy in the CRDT.

    Attributes:
        policy_id: Unique identifier for the policy
        content: Policy content (rules, conditions, etc.)
        version_hash: Hash of the content for quick comparison
        status: Current status of the policy
        metadata: Additional metadata
    """

    policy_id: str
    content: Dict[str, Any]
    version_hash: str = ""
    status: PolicyStatus = PolicyStatus.QUARANTINE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute version hash if not provided."""
        if not self.version_hash:
            self.version_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute a hash of the policy content."""
        import json
        content_str = json.dumps(self.content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "policy_id": self.policy_id,
            "content": self.content,
            "version_hash": self.version_hash,
            "status": self.status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyState":
        """Create from dictionary."""
        return cls(
            policy_id=data.get("policy_id", ""),
            content=data.get("content", {}),
            version_hash=data.get("version_hash", ""),
            status=PolicyStatus(data.get("status", "quarantine")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PolicyDelta:
    """
    Delta update for policy synchronization.

    Represents a change to be applied to a policy.

    Attributes:
        policy_id: ID of the policy being updated
        operation: Type of operation (add, update, deprecate, delete)
        state: New state (for add/update operations)
        timestamp: HLC timestamp of the change
        source_node: Node that originated the change
    """

    policy_id: str
    operation: str  # "add", "update", "deprecate", "delete"
    state: Optional[PolicyState] = None
    timestamp: HybridLogicalClock = field(default_factory=HybridLogicalClock)
    source_node: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "policy_id": self.policy_id,
            "operation": self.operation,
            "state": self.state.to_dict() if self.state else None,
            "timestamp": self.timestamp.to_dict(),
            "source_node": self.source_node,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyDelta":
        """Create from dictionary."""
        state = None
        if data.get("state"):
            state = PolicyState.from_dict(data["state"])
        return cls(
            policy_id=data.get("policy_id", ""),
            operation=data.get("operation", ""),
            state=state,
            timestamp=HybridLogicalClock.from_dict(data.get("timestamp", {})),
            source_node=data.get("source_node", ""),
        )


@dataclass
class CRDTMergeResult:
    """
    Result of a CRDT merge operation.

    Attributes:
        merged_count: Number of items merged
        conflict_count: Number of conflicts detected
        conflicts: List of conflicting policy IDs
        new_policies: List of newly added policy IDs
        updated_policies: List of updated policy IDs
    """

    merged_count: int = 0
    conflict_count: int = 0
    conflicts: List[str] = field(default_factory=list)
    new_policies: List[str] = field(default_factory=list)
    updated_policies: List[str] = field(default_factory=list)


@dataclass
class PolicyCRDT:
    """
    CRDT for policy state synchronization.

    Combines multiple CRDT types to provide:
    - Policy versioning with LWW semantics
    - Active policy set with OR-Set semantics
    - Conflict detection with MV-Register
    - Efficient delta synchronization

    Guarantees:
    - Eventual consistency across all regions
    - Conflict-free merging
    - Offline support with sync
    - No coordination required

    Attributes:
        policies: Dictionary of policy ID to LWW-Register
        active_policies: OR-Set of active policy IDs
        version_history: Dictionary of policy ID to version counter
        clock: Hybrid logical clock for this node
        node_id: ID of this node
    """

    policies: Dict[str, LWWRegister[PolicyState]] = field(default_factory=dict)
    active_policies: ORSet = field(default_factory=ORSet)
    version_history: Dict[str, PNCounter] = field(default_factory=dict)
    clock: HybridLogicalClock = field(default_factory=HybridLogicalClock)
    node_id: str = ""

    def __post_init__(self):
        """Initialize with node ID."""
        if self.node_id:
            self.clock.node_id = self.node_id
            self.active_policies.node_id = self.node_id

    def add_policy(self, state: PolicyState) -> PolicyDelta:
        """
        Add a new policy or update existing.

        Args:
            state: The policy state to add

        Returns:
            PolicyDelta representing the change
        """
        # Update timestamp
        self.clock.now()

        # Create or update the LWW register
        if state.policy_id not in self.policies:
            self.policies[state.policy_id] = LWWRegister(node_id=self.node_id)
            self.version_history[state.policy_id] = PNCounter(node_id=self.node_id)

        self.policies[state.policy_id].set(state)
        self.version_history[state.policy_id].increment()

        # Add to active policies if status is active
        if state.status == PolicyStatus.ACTIVE:
            self.active_policies.add(state.policy_id)

        logger.debug(f"Added policy {state.policy_id} with status {state.status}")

        return PolicyDelta(
            policy_id=state.policy_id,
            operation="add" if state.policy_id not in self.policies else "update",
            state=state,
            timestamp=HybridLogicalClock(
                physical=self.clock.physical,
                logical=self.clock.logical,
                node_id=self.node_id,
            ),
            source_node=self.node_id,
        )

    def update_policy(self, policy_id: str, new_content: Dict[str, Any]) -> Optional[PolicyDelta]:
        """
        Update an existing policy's content.

        Args:
            policy_id: ID of the policy to update
            new_content: New content for the policy

        Returns:
            PolicyDelta representing the change, or None if policy not found
        """
        if policy_id not in self.policies:
            logger.warning(f"Policy {policy_id} not found for update")
            return None

        current = self.policies[policy_id].get()
        if current is None:
            return None

        # Create new state with updated content
        new_state = PolicyState(
            policy_id=policy_id,
            content=new_content,
            status=current.status,
            metadata=current.metadata,
        )

        return self.add_policy(new_state)

    def deprecate_policy(self, policy_id: str) -> Optional[PolicyDelta]:
        """
        Deprecate a policy (mark as deprecated but keep in system).

        Args:
            policy_id: ID of the policy to deprecate

        Returns:
            PolicyDelta representing the change
        """
        if policy_id not in self.policies:
            return None

        current = self.policies[policy_id].get()
        if current is None:
            return None

        # Update status
        new_state = PolicyState(
            policy_id=policy_id,
            content=current.content,
            version_hash=current.version_hash,
            status=PolicyStatus.DEPRECATED,
            metadata=current.metadata,
        )

        self.policies[policy_id].set(new_state)
        self.active_policies.remove(policy_id)

        self.clock.now()

        return PolicyDelta(
            policy_id=policy_id,
            operation="deprecate",
            state=new_state,
            timestamp=HybridLogicalClock(
                physical=self.clock.physical,
                logical=self.clock.logical,
                node_id=self.node_id,
            ),
            source_node=self.node_id,
        )

    def delete_policy(self, policy_id: str) -> Optional[PolicyDelta]:
        """
        Delete a policy (soft delete).

        Args:
            policy_id: ID of the policy to delete

        Returns:
            PolicyDelta representing the change
        """
        if policy_id not in self.policies:
            return None

        current = self.policies[policy_id].get()
        if current is None:
            return None

        # Mark as deleted
        new_state = PolicyState(
            policy_id=policy_id,
            content={},  # Clear content
            status=PolicyStatus.DELETED,
            metadata={"deleted_at": self.clock.physical},
        )

        self.policies[policy_id].set(new_state)
        self.active_policies.remove(policy_id)

        self.clock.now()

        return PolicyDelta(
            policy_id=policy_id,
            operation="delete",
            state=new_state,
            timestamp=HybridLogicalClock(
                physical=self.clock.physical,
                logical=self.clock.logical,
                node_id=self.node_id,
            ),
            source_node=self.node_id,
        )

    def get_policy(self, policy_id: str) -> Optional[PolicyState]:
        """
        Get a policy by ID.

        Args:
            policy_id: ID of the policy

        Returns:
            PolicyState if found, None otherwise
        """
        if policy_id not in self.policies:
            return None
        return self.policies[policy_id].get()

    def get_active_policies(self) -> List[PolicyState]:
        """Get all active policies."""
        result = []
        for policy_id in self.active_policies.to_set():
            state = self.get_policy(policy_id)
            if state and state.status == PolicyStatus.ACTIVE:
                result.append(state)
        return result

    def get_all_policies(self) -> List[PolicyState]:
        """Get all policies (including inactive)."""
        result = []
        for policy_id, register in self.policies.items():
            state = register.get()
            if state:
                result.append(state)
        return result

    def merge(self, other: "PolicyCRDT") -> CRDTMergeResult:
        """
        Merge with another PolicyCRDT.

        This is the core CRDT merge operation that ensures eventual consistency.

        Args:
            other: The PolicyCRDT to merge with

        Returns:
            CRDTMergeResult with merge statistics
        """
        result = CRDTMergeResult()

        # Merge policies (LWW semantics)
        for policy_id, other_register in other.policies.items():
            if policy_id not in self.policies:
                # New policy from remote
                self.policies[policy_id] = LWWRegister(node_id=self.node_id)
                self.policies[policy_id].merge(other_register)
                result.new_policies.append(policy_id)
                result.merged_count += 1
            else:
                # Existing policy - merge with LWW
                local_state = self.policies[policy_id].get()
                other_state = other_register.get()

                self.policies[policy_id].merge(other_register)
                merged_state = self.policies[policy_id].get()

                # Check if value changed
                if merged_state and local_state and merged_state.version_hash != local_state.version_hash:
                    result.updated_policies.append(policy_id)
                    result.merged_count += 1

        # Merge active policies (OR-Set semantics)
        self.active_policies.merge(other.active_policies)

        # Merge version history
        for policy_id, other_counter in other.version_history.items():
            if policy_id not in self.version_history:
                self.version_history[policy_id] = PNCounter(node_id=self.node_id)
            self.version_history[policy_id].merge(other_counter)

        # Update local clock
        self.clock.receive(other.clock.physical, other.clock.logical)

        logger.info(
            f"Merged with remote: {result.merged_count} merged, "
            f"{len(result.new_policies)} new, {len(result.updated_policies)} updated"
        )

        return result

    def apply_delta(self, delta: PolicyDelta) -> bool:
        """
        Apply a delta update from another node.

        Args:
            delta: The delta to apply

        Returns:
            True if applied successfully
        """
        # Update clock based on delta timestamp
        self.clock.receive(delta.timestamp.physical, delta.timestamp.logical)

        if delta.operation in ("add", "update"):
            if delta.state:
                self.add_policy(delta.state)
                return True
        elif delta.operation == "deprecate":
            self.deprecate_policy(delta.policy_id)
            return True
        elif delta.operation == "delete":
            self.delete_policy(delta.policy_id)
            return True

        return False

    def get_deltas_since(self, since_timestamp: int) -> List[PolicyDelta]:
        """
        Get all deltas since a given timestamp.

        Args:
            since_timestamp: Combined 64-bit HLC timestamp

        Returns:
            List of PolicyDelta changes
        """
        deltas = []

        for policy_id, register in self.policies.items():
            state = register.get()
            if state and register.timestamp.timestamp() > since_timestamp:
                deltas.append(PolicyDelta(
                    policy_id=policy_id,
                    operation="update",
                    state=state,
                    timestamp=register.timestamp,
                    source_node=self.node_id,
                ))

        return deltas

    def compute_digest(self) -> str:
        """
        Compute a digest of the current state for comparison.

        Returns:
            Hex digest of the state
        """
        import json

        state_dict = {
            "policies": sorted([
                (pid, reg.timestamp.timestamp())
                for pid, reg in self.policies.items()
            ]),
            "active": sorted(list(self.active_policies.to_set())),
        }

        state_str = json.dumps(state_dict, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "policies": {
                pid: reg.to_dict() for pid, reg in self.policies.items()
            },
            "active_policies": self.active_policies.to_dict(),
            "version_history": {
                pid: counter.to_dict() for pid, counter in self.version_history.items()
            },
            "clock": self.clock.to_dict(),
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyCRDT":
        """Create from dictionary."""
        policies = {}
        for pid, reg_data in data.get("policies", {}).items():
            reg = LWWRegister.from_dict(reg_data)
            if reg.value and isinstance(reg.value, dict):
                reg.value = PolicyState.from_dict(reg.value)
            policies[pid] = reg

        version_history = {
            pid: PNCounter.from_dict(counter_data)
            for pid, counter_data in data.get("version_history", {}).items()
        }

        return cls(
            policies=policies,
            active_policies=ORSet.from_dict(data.get("active_policies", {})),
            version_history=version_history,
            clock=HybridLogicalClock.from_dict(data.get("clock", {})),
            node_id=data.get("node_id", ""),
        )
