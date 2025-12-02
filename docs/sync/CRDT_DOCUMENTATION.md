# CRDT Synchronization Documentation

This document describes the Conflict-Free Replicated Data Type (CRDT) implementation
used for multi-region policy state synchronization in Nethical.

## Overview

Nethical uses CRDTs to maintain consistent policy state across multiple regions
without requiring coordination. This enables:

- **Eventual Consistency**: All regions converge to the same state
- **Offline Support**: Edge devices operate independently when disconnected
- **No Coordination**: No locks, no consensus protocols, no single point of failure
- **Automatic Conflict Resolution**: Concurrent updates merge deterministically

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   US-EAST-1     │     │   EU-WEST-1     │     │   AP-SOUTH-1    │
│                 │     │                 │     │                 │
│  ┌───────────┐  │     │  ┌───────────┐  │     │  ┌───────────┐  │
│  │ PolicyCRDT│  │     │  │ PolicyCRDT│  │     │  │ PolicyCRDT│  │
│  │           │◄─┼─────┼─►│           │◄─┼─────┼─►│           │  │
│  └───────────┘  │     │  └───────────┘  │     │  └───────────┘  │
│        │        │     │        │        │     │        │        │
│        ▼        │     │        ▼        │     │        ▼        │
│  ┌───────────┐  │     │  ┌───────────┐  │     │  ┌───────────┐  │
│  │ Anti-     │  │     │  │ Anti-     │  │     │  │ Anti-     │  │
│  │ Entropy   │◄─┼─────┼─►│ Entropy   │◄─┼─────┼─►│ Entropy   │  │
│  └───────────┘  │     │  └───────────┘  │     │  └───────────┘  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
           ▲                     ▲                     ▲
           │                     │                     │
           ▼                     ▼                     ▼
    ┌─────────────────────────────────────────────────────────┐
    │                     Edge Devices                         │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
    │  │ Edge    │  │ Edge    │  │ Edge    │  │ Edge    │     │
    │  │ CRDT    │  │ CRDT    │  │ CRDT    │  │ CRDT    │     │
    │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │
    └─────────────────────────────────────────────────────────┘
```

## CRDT Types

### GCounter (Grow-only Counter)

A counter that can only be incremented. Used for:
- Tracking total decisions made
- Counting policy activations
- Monotonic sequence numbers

```python
from nethical.sync import GCounter

counter = GCounter(node_id="node-1")
counter.increment(5)  # Add 5
print(counter.value())  # 5

# Merge with another node's counter
other = GCounter(node_id="node-2")
other.increment(3)
counter.merge(other)
print(counter.value())  # 8
```

### PNCounter (Positive-Negative Counter)

A counter that supports both increment and decrement.

```python
from nethical.sync import PNCounter

counter = PNCounter(node_id="node-1")
counter.increment(10)
counter.decrement(3)
print(counter.value())  # 7
```

### LWWRegister (Last-Writer-Wins Register)

A register where the most recent write wins based on timestamps.
Uses Hybrid Logical Clocks for ordering.

```python
from nethical.sync import LWWRegister

reg = LWWRegister(node_id="node-1")
reg.set("value-1")
print(reg.get())  # "value-1"

# Concurrent writes resolve by timestamp
other = LWWRegister(node_id="node-2")
other.set("value-2")  # Happens later
reg.merge(other)
print(reg.get())  # "value-2" (later timestamp wins)
```

### ORSet (Observed-Remove Set)

A set that supports both add and remove operations.
Add wins over concurrent remove.

```python
from nethical.sync import ORSet

policies = ORSet(node_id="node-1")
policies.add("policy-a")
policies.add("policy-b")
policies.remove("policy-a")

print(policies.to_set())  # {"policy-b"}
```

### MVRegister (Multi-Value Register)

Preserves all concurrent writes as multiple values.

```python
from nethical.sync import MVRegister

reg = MVRegister(node_id="node-1")
reg.set("value-1")

# Concurrent update from another node
other = MVRegister(node_id="node-2")
other.set("value-2")

reg.merge(other)
print(reg.get())  # ["value-1", "value-2"]
print(reg.has_conflict())  # True
```

## PolicyCRDT

The main CRDT for policy synchronization combines multiple primitive CRDTs:

```python
from nethical.sync import PolicyCRDT, PolicyState, PolicyStatus

# Initialize on each region
crdt = PolicyCRDT(node_id="us-east-1")

# Add a policy
state = PolicyState(
    policy_id="safety-rules-001",
    content={"rules": [...]},
    status=PolicyStatus.ACTIVE,
)
delta = crdt.add_policy(state)

# Update a policy
crdt.update_policy("safety-rules-001", {"rules": [...]})

# Deprecate a policy
crdt.deprecate_policy("safety-rules-001")

# Get active policies
active = crdt.get_active_policies()

# Merge with another region
other_crdt = PolicyCRDT(node_id="eu-west-1")
result = crdt.merge(other_crdt)
print(f"Merged {result.merged_count} policies")
print(f"New: {result.new_policies}")
print(f"Updated: {result.updated_policies}")
```

## Vector Clocks

Used for causal ordering of events.

```python
from nethical.sync import VectorClock, EventOrder

# Each node maintains its own clock
clock_a = VectorClock(node_id="node-a")
clock_b = VectorClock(node_id="node-b")

# Increment on local events
clock_a.increment()

# Update on receiving a message
clock_b.receive_event(clock_a)

# Compare clocks
order = clock_a.compare(clock_b)
if order == EventOrder.BEFORE:
    print("A happened before B")
elif order == EventOrder.CONCURRENT:
    print("A and B are concurrent")
```

## Hybrid Logical Clocks

Combines physical time with logical counters for:
- Bounded clock skew
- Good locality for queries
- Causality preservation

```python
from nethical.sync import HybridLogicalClock

hlc = HybridLogicalClock(node_id="node-1")

# Generate timestamp for local event
physical, logical = hlc.now()

# Generate timestamp for send event
physical, logical = hlc.send()

# Update on receiving a message
physical, logical = hlc.receive(remote_physical, remote_logical)

# Get combined 64-bit timestamp for storage
timestamp = hlc.timestamp()
```

## Anti-Entropy Protocol

Background synchronization to ensure eventual consistency.

```python
from nethical.sync import AntiEntropyProtocol, PolicyCRDT

# Initialize
crdt = PolicyCRDT(node_id="node-1")
anti_entropy = AntiEntropyProtocol(
    crdt=crdt,
    node_id="node-1",
    sync_interval_sec=30,
)

# Add peers
anti_entropy.add_peer("node-2")
anti_entropy.add_peer("node-3")

# Rebuild Merkle tree after changes
anti_entropy.rebuild_merkle_tree()

# Get digest for comparison
digest = anti_entropy.get_digest()

# Start sync session
session = await anti_entropy.start_sync(
    remote_node="node-2",
    remote_digest=remote_digest,
)

# Get deltas to send
deltas = anti_entropy.get_deltas_for_peer(peer_merkle_tree, session)

# Apply received deltas
result = anti_entropy.apply_deltas(deltas, session)

# Complete session
anti_entropy.complete_sync(session, success=True)

# Get statistics
stats = anti_entropy.get_statistics()
```

## Merkle Tree

Efficient state comparison using hash trees.

```python
from nethical.sync import MerkleTree

tree = MerkleTree(branching_factor=16)

# Build from policies
policies = {
    "policy-1": PolicyState(...),
    "policy-2": PolicyState(...),
}
root = tree.build(policies)

# Get digest
digest = tree.get_digest()

# Find differences with another tree
differences = tree.get_differences(other_tree)
```

## Configuration

Configure CRDT synchronization in `config/sync.yaml`:

```yaml
crdt:
  policy:
    conflict_resolution: "lww"
    max_delta_batch: 100
    merkle_branching: 16
  
  vector_clock:
    max_drift_sec: 60

anti_entropy:
  interval_sec: 30
  max_concurrent: 3
  max_session_sec: 60
```

## Best Practices

1. **Use appropriate CRDT types**:
   - Counters for monotonic values → GCounter
   - Counters with decrement → PNCounter
   - Single values with clear ordering → LWWRegister
   - Sets with add/remove → ORSet
   - Values where conflicts matter → MVRegister

2. **Handle conflicts explicitly**:
   - LWWRegister: Latest timestamp wins
   - ORSet: Add wins over concurrent remove
   - MVRegister: Preserve all concurrent values

3. **Monitor sync health**:
   - Track sync latency
   - Alert on excessive drift
   - Monitor conflict rates

4. **Optimize delta transfer**:
   - Use Merkle trees for efficient diffing
   - Batch deltas to reduce round trips
   - Compress payloads

## Performance Considerations

| Operation | Complexity | Typical Latency |
|-----------|------------|-----------------|
| GCounter increment | O(1) | <0.01ms |
| GCounter merge | O(n) nodes | <0.1ms |
| LWWRegister set | O(1) | <0.01ms |
| ORSet add | O(1) | <0.01ms |
| ORSet merge | O(n) elements | <1ms |
| PolicyCRDT merge | O(p) policies | <10ms |
| Merkle tree build | O(n log n) | <100ms |
| Merkle tree diff | O(log n + d) | <10ms |

Where:
- n = number of nodes/elements
- p = number of policies
- d = number of differences
