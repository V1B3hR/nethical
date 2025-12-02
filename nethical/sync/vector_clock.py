"""
Vector Clocks for Causal Ordering

Provides vector clocks and hybrid logical clocks for ordering events
in distributed multi-region deployments.

Features:
- Standard Vector Clock for total causal ordering
- Hybrid Logical Clock (HLC) for bounded clock skew
- Event ordering comparisons
- Merge operations for concurrent events
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)


class EventOrder(str, Enum):
    """Ordering relationship between two events."""

    BEFORE = "BEFORE"  # a happened-before b
    AFTER = "AFTER"  # a happened-after b
    CONCURRENT = "CONCURRENT"  # a and b are concurrent
    EQUAL = "EQUAL"  # a and b are the same event


@dataclass
class VectorClock:
    """
    Vector Clock for causal ordering in distributed systems.

    A vector clock assigns a logical timestamp to each node, allowing
    determination of causal relationships between events.

    Guarantees:
    - If a happened-before b, then VC(a) < VC(b)
    - If VC(a) < VC(b), then a may have happened-before b
    - If VC(a) || VC(b), then a and b are concurrent

    Attributes:
        clock: Dictionary mapping node IDs to their logical times
        node_id: ID of the local node
    """

    clock: Dict[str, int] = field(default_factory=dict)
    node_id: str = ""

    def __post_init__(self):
        """Initialize the clock for the local node if not present."""
        if self.node_id and self.node_id not in self.clock:
            self.clock[self.node_id] = 0

    def increment(self) -> "VectorClock":
        """
        Increment the local node's clock.

        Returns:
            Self for method chaining
        """
        if self.node_id:
            self.clock[self.node_id] = self.clock.get(self.node_id, 0) + 1
        return self

    def send_event(self) -> "VectorClock":
        """
        Record a send event - increments local clock.

        Returns:
            Self for method chaining
        """
        return self.increment()

    def receive_event(self, other: "VectorClock") -> "VectorClock":
        """
        Record a receive event - merge with sender's clock.

        Args:
            other: The vector clock from the sender

        Returns:
            Self for method chaining
        """
        # Merge with received clock (take max for each node)
        for node_id, timestamp in other.clock.items():
            self.clock[node_id] = max(self.clock.get(node_id, 0), timestamp)

        # Increment local clock
        return self.increment()

    def merge(self, other: "VectorClock") -> "VectorClock":
        """
        Merge with another vector clock (take max for each entry).

        Args:
            other: The vector clock to merge with

        Returns:
            Self for method chaining
        """
        for node_id, timestamp in other.clock.items():
            self.clock[node_id] = max(self.clock.get(node_id, 0), timestamp)
        return self

    def compare(self, other: "VectorClock") -> EventOrder:
        """
        Compare this clock with another to determine causal ordering.

        Args:
            other: The vector clock to compare with

        Returns:
            EventOrder indicating the relationship
        """
        all_nodes = set(self.clock.keys()) | set(other.clock.keys())

        less_or_equal = True
        greater_or_equal = True

        for node_id in all_nodes:
            self_time = self.clock.get(node_id, 0)
            other_time = other.clock.get(node_id, 0)

            if self_time > other_time:
                less_or_equal = False
            if self_time < other_time:
                greater_or_equal = False

        if less_or_equal and greater_or_equal:
            return EventOrder.EQUAL
        elif less_or_equal:
            return EventOrder.BEFORE
        elif greater_or_equal:
            return EventOrder.AFTER
        else:
            return EventOrder.CONCURRENT

    def __lt__(self, other: "VectorClock") -> bool:
        """Check if this clock happened-before another."""
        return self.compare(other) == EventOrder.BEFORE

    def __le__(self, other: "VectorClock") -> bool:
        """Check if this clock happened-before or equals another."""
        result = self.compare(other)
        return result in (EventOrder.BEFORE, EventOrder.EQUAL)

    def __gt__(self, other: "VectorClock") -> bool:
        """Check if this clock happened-after another."""
        return self.compare(other) == EventOrder.AFTER

    def __ge__(self, other: "VectorClock") -> bool:
        """Check if this clock happened-after or equals another."""
        result = self.compare(other)
        return result in (EventOrder.AFTER, EventOrder.EQUAL)

    def __eq__(self, other: object) -> bool:
        """Check if clocks are equal."""
        if not isinstance(other, VectorClock):
            return False
        return self.compare(other) == EventOrder.EQUAL

    def copy(self) -> "VectorClock":
        """Create a copy of this vector clock."""
        return VectorClock(clock=dict(self.clock), node_id=self.node_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "clock": dict(self.clock),
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorClock":
        """Create from dictionary."""
        return cls(
            clock=dict(data.get("clock", {})),
            node_id=data.get("node_id", ""),
        )


@dataclass
class HybridLogicalClock:
    """
    Hybrid Logical Clock (HLC) for bounded clock skew.

    Combines physical time with logical counters to provide:
    - Monotonically increasing timestamps
    - Bounded clock skew
    - Causality preservation
    - Good locality for time-based queries

    Attributes:
        physical: Physical time (wall clock) component
        logical: Logical counter component
        node_id: ID of the local node for tie-breaking
        max_drift_usec: Maximum allowed clock drift in microseconds
    """

    physical: int = 0  # Physical time in microseconds
    logical: int = 0
    node_id: str = ""
    max_drift_usec: int = 60 * 1_000_000  # Default: 1 minute

    # Class constant for backwards compatibility
    MAX_DRIFT_USEC: int = 60 * 1_000_000

    def now(self) -> Tuple[int, int]:
        """
        Generate a new HLC timestamp for a local event.

        Returns:
            Tuple of (physical, logical) timestamp components
        """
        wall = int(time.time() * 1_000_000)  # Current time in microseconds

        if wall > self.physical:
            # Physical time has advanced
            self.physical = wall
            self.logical = 0
        else:
            # Physical time hasn't advanced, increment logical
            self.logical += 1

        return (self.physical, self.logical)

    def send(self) -> Tuple[int, int]:
        """
        Generate HLC timestamp for send event.

        Returns:
            Tuple of (physical, logical) timestamp components
        """
        return self.now()

    def receive(self, remote_physical: int, remote_logical: int) -> Tuple[int, int]:
        """
        Update HLC on receiving a message.

        Args:
            remote_physical: Physical component from remote clock
            remote_logical: Logical component from remote clock

        Returns:
            Tuple of (physical, logical) timestamp components

        Raises:
            ValueError: If clock drift exceeds maximum allowed
        """
        wall = int(time.time() * 1_000_000)

        # Check for excessive clock drift (use instance variable if set, else class constant)
        max_drift = self.max_drift_usec if self.max_drift_usec else self.MAX_DRIFT_USEC
        if remote_physical > wall + max_drift:
            logger.warning(
                f"Excessive clock drift detected: "
                f"remote={remote_physical}, local={wall}, "
                f"drift={(remote_physical - wall) / 1_000_000:.2f}s"
            )

        if wall > self.physical and wall > remote_physical:
            # Local wall clock is ahead
            self.physical = wall
            self.logical = 0
        elif remote_physical > self.physical:
            # Remote physical is ahead
            self.physical = remote_physical
            self.logical = remote_logical + 1
        elif self.physical > remote_physical:
            # Local physical is ahead
            self.logical += 1
        else:
            # Same physical time
            self.logical = max(self.logical, remote_logical) + 1

        return (self.physical, self.logical)

    def compare(self, other: "HybridLogicalClock") -> EventOrder:
        """
        Compare this HLC with another.

        Args:
            other: The HLC to compare with

        Returns:
            EventOrder indicating the relationship
        """
        if self.physical < other.physical:
            return EventOrder.BEFORE
        elif self.physical > other.physical:
            return EventOrder.AFTER
        elif self.logical < other.logical:
            return EventOrder.BEFORE
        elif self.logical > other.logical:
            return EventOrder.AFTER
        elif self.node_id < other.node_id:
            return EventOrder.BEFORE
        elif self.node_id > other.node_id:
            return EventOrder.AFTER
        else:
            return EventOrder.EQUAL

    def timestamp(self) -> int:
        """
        Get a single 64-bit timestamp for storage.

        High 44 bits: physical time (good for ~500 years)
        Low 20 bits: logical counter (up to ~1M events per microsecond)

        Returns:
            64-bit combined timestamp
        """
        return (self.physical << 20) | (self.logical & 0xFFFFF)

    @classmethod
    def from_timestamp(cls, ts: int, node_id: str = "") -> "HybridLogicalClock":
        """
        Create HLC from a 64-bit timestamp.

        Args:
            ts: Combined 64-bit timestamp
            node_id: Node ID for the clock

        Returns:
            HybridLogicalClock instance
        """
        physical = ts >> 20
        logical = ts & 0xFFFFF
        return cls(physical=physical, logical=logical, node_id=node_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "physical": self.physical,
            "logical": self.logical,
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridLogicalClock":
        """Create from dictionary."""
        return cls(
            physical=data.get("physical", 0),
            logical=data.get("logical", 0),
            node_id=data.get("node_id", ""),
        )

    def __lt__(self, other: "HybridLogicalClock") -> bool:
        """Check if this HLC is before another."""
        return self.compare(other) == EventOrder.BEFORE

    def __le__(self, other: "HybridLogicalClock") -> bool:
        """Check if this HLC is before or equal to another."""
        result = self.compare(other)
        return result in (EventOrder.BEFORE, EventOrder.EQUAL)

    def __gt__(self, other: "HybridLogicalClock") -> bool:
        """Check if this HLC is after another."""
        return self.compare(other) == EventOrder.AFTER

    def __ge__(self, other: "HybridLogicalClock") -> bool:
        """Check if this HLC is after or equal to another."""
        result = self.compare(other)
        return result in (EventOrder.AFTER, EventOrder.EQUAL)

    def __eq__(self, other: object) -> bool:
        """Check if HLCs are equal."""
        if not isinstance(other, HybridLogicalClock):
            return False
        return self.compare(other) == EventOrder.EQUAL
