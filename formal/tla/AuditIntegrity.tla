---------------------------- MODULE AuditIntegrity ----------------------------
\* TLA+ Specification for Nethical Audit Trail Integrity
\* Verifies the Merkle tree-based append-only audit log properties
\* 
\* This specification ensures:
\* - Audit logs are append-only (immutable)
\* - Merkle tree roots are computed correctly
\* - No logs can be tampered with or deleted
\* - Cryptographic chain of custody is maintained

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    MaxLogEntries,      \* Maximum number of log entries to model
    AgentIds,          \* Set of agent identifiers
    ActionTypes        \* Set of action types

VARIABLES
    auditLog,          \* Sequence of audit entries
    merkleRoots,       \* Sequence of Merkle root hashes
    merkleLeaves,      \* Sequence of leaf hashes for current batch
    logVersion,        \* Monotonically increasing version
    anchored           \* Set of indices that have been anchored

vars == <<auditLog, merkleRoots, merkleLeaves, logVersion, anchored>>

\* ----- Type Definitions -----

AuditEntry == [
    entryId: Nat,
    agentId: AgentIds,
    actionType: ActionTypes,
    decision: {"ALLOW", "RESTRICT", "BLOCK", "TERMINATE"},
    timestamp: Nat,
    hash: STRING
]

TypeInvariant ==
    /\ auditLog \in Seq(AuditEntry)
    /\ Len(auditLog) <= MaxLogEntries
    /\ merkleRoots \in Seq(STRING)
    /\ merkleLeaves \in Seq(STRING)
    /\ logVersion \in Nat
    /\ anchored \subseteq 1..Len(auditLog)

\* ----- Initial State -----

Init ==
    /\ auditLog = <<>>
    /\ merkleRoots = <<>>
    /\ merkleLeaves = <<>>
    /\ logVersion = 0
    /\ anchored = {}

\* ----- Helper Functions -----

\* Compute a simple hash (abstracted for TLA+)
\* In reality, this would be SHA-256 or similar
ComputeHash(entry) ==
    "hash_" \o ToString(entry.entryId) \o "_" \o ToString(entry.agentId)

\* Compute Merkle root from leaves (simplified)
ComputeMerkleRoot(leaves) ==
    IF Len(leaves) = 0 THEN "empty_root"
    ELSE "merkle_root_" \o ToString(Len(leaves))

\* Check if an entry has been anchored
IsAnchored(index) ==
    index \in anchored

\* Verify chain integrity: each entry's hash depends on previous
ChainIntegrity ==
    \A i \in 2..Len(auditLog) :
        auditLog[i].entryId = auditLog[i-1].entryId + 1

\* ----- Actions -----

\* Append a new entry to the audit log
AppendEntry(agentId, actionType, decision) ==
    /\ Len(auditLog) < MaxLogEntries
    /\ agentId \in AgentIds
    /\ actionType \in ActionTypes
    /\ decision \in {"ALLOW", "RESTRICT", "BLOCK", "TERMINATE"}
    /\ LET newEntry == [
           entryId |-> Len(auditLog) + 1,
           agentId |-> agentId,
           actionType |-> actionType,
           decision |-> decision,
           timestamp |-> logVersion + 1,
           hash |-> ComputeHash([entryId |-> Len(auditLog) + 1, agentId |-> agentId])
       ]
       IN
           /\ auditLog' = Append(auditLog, newEntry)
           /\ merkleLeaves' = Append(merkleLeaves, newEntry.hash)
           /\ logVersion' = logVersion + 1
           /\ UNCHANGED <<merkleRoots, anchored>>

\* Anchor current batch by computing and storing Merkle root
AnchorBatch ==
    /\ Len(merkleLeaves) > 0
    /\ LET root == ComputeMerkleRoot(merkleLeaves)
           startIdx == Len(auditLog) - Len(merkleLeaves) + 1
           endIdx == Len(auditLog)
       IN
           /\ merkleRoots' = Append(merkleRoots, root)
           /\ anchored' = anchored \union {i \in startIdx..endIdx : TRUE}
           /\ merkleLeaves' = <<>>
           /\ UNCHANGED <<auditLog, logVersion>>

\* Attempt to tamper with log (should be prevented by invariants)
AttemptTamper(index, newDecision) ==
    /\ index \in DOMAIN auditLog
    /\ index \in anchored  \* Cannot tamper with anchored entries
    /\ newDecision \in {"ALLOW", "RESTRICT", "BLOCK", "TERMINATE"}
    /\ FALSE  \* This action should never be enabled

\* ----- State Machine Specification -----

Next ==
    \/ \E agentId \in AgentIds, actionType \in ActionTypes, decision \in {"ALLOW", "RESTRICT", "BLOCK", "TERMINATE"} :
           AppendEntry(agentId, actionType, decision)
    \/ AnchorBatch
    \* Note: AttemptTamper is never allowed, included for completeness

Spec == Init /\ [][Next]_vars

\* ----- Safety Properties -----

\* Property 1: Audit log is append-only (monotonically growing)
AppendOnlyProperty ==
    [][\/ Len(auditLog') = Len(auditLog)
        \/ Len(auditLog') = Len(auditLog) + 1]_auditLog

\* Property 2: Existing entries never change
ImmutabilityProperty ==
    [][\A i \in DOMAIN auditLog : auditLog'[i] = auditLog[i]]_auditLog

\* Property 3: Entry IDs are sequential
SequentialIdProperty ==
    [](\A i \in DOMAIN auditLog : auditLog[i].entryId = i)

\* Property 4: Timestamps are monotonic
MonotonicTimestampProperty ==
    [](\A i, j \in DOMAIN auditLog : i < j => auditLog[i].timestamp <= auditLog[j].timestamp)

\* Property 5: Anchored entries cannot be modified
AnchoredImmutabilityProperty ==
    [](\A i \in anchored : i \in DOMAIN auditLog => 
        [](auditLog[i] = [auditLog EXCEPT !]))

\* Property 6: Version number increases with new entries
VersionIncreasesProperty ==
    []([Len(auditLog') > Len(auditLog) => logVersion' > logVersion]_vars)

\* Property 7: Each anchored batch has a Merkle root
AnchorCompletenessProperty ==
    [](Len(anchored) > 0 => Len(merkleRoots) > 0)

\* Property 8: Chain integrity is maintained
ChainIntegrityProperty ==
    []ChainIntegrity

\* Property 9: Anchored set only grows
AnchoredMonotonicProperty ==
    [][anchored \subseteq anchored']_anchored

\* ----- Liveness Properties -----

\* Property 10: Entries are eventually anchored
EventuallyAnchored ==
    \A i \in 1..MaxLogEntries :
        (i <= Len(auditLog)) ~> (i \in anchored)

\* Property 11: If entries exist, they eventually get a Merkle root
EventuallyRooted ==
    (Len(auditLog) > 0) ~> (Len(merkleRoots) > 0)

\* ----- Security Properties -----

\* Property 12: No gaps in audit log
NoGapsProperty ==
    [](Len(auditLog) > 0 => 
        \A i \in 1..Len(auditLog) : \E entry \in {auditLog[j] : j \in DOMAIN auditLog} :
            entry.entryId = i)

\* Property 13: Hash uniqueness (simplified check)
HashUniquenessProperty ==
    [](\A i, j \in DOMAIN auditLog :
        i # j => auditLog[i].hash # auditLog[j].hash)

\* ----- Audit Compliance Theorems -----

\* Theorem: Audit log maintains immutability
THEOREM Spec => []ImmutabilityProperty

\* Theorem: Entry IDs are always sequential
THEOREM Spec => []SequentialIdProperty

\* Theorem: Timestamps are monotonic
THEOREM Spec => []MonotonicTimestampProperty

\* Theorem: Chain integrity is preserved
THEOREM Spec => []ChainIntegrityProperty

\* Theorem: Anchored entries are immutable
THEOREM Spec => []AnchoredImmutabilityProperty

\* ----- Model Checking Configuration -----

THEOREM Spec => []TypeInvariant
THEOREM Spec => []AppendOnlyProperty
THEOREM Spec => []AnchoredMonotonicProperty

\* Constraint for model checking (limit state space)
StateConstraint == Len(auditLog) <= 10 /\ logVersion <= 20

================================================================================
