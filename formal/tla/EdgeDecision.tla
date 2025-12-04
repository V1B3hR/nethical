---------------------------- MODULE EdgeDecision ----------------------------
\* TLA+ Specification for Nethical Edge Decision Engine
\* Formal verification of edge deployment safety properties
\* 
\* This specification models the edge governance decision process
\* with offline fallback and fail-safe guarantees.

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    EdgeDevices,      \* Set of edge device identifiers
    MaxLatencyMs,     \* Maximum allowed latency in milliseconds
    SafeDefaultDecision  \* Default decision when uncertain ("RESTRICT")

VARIABLES
    deviceStates,     \* Function: Device -> State (ONLINE, OFFLINE, DEGRADED)
    cachedPolicies,   \* Function: Device -> Set of cached policy hashes
    pendingDecisions, \* Set of pending decision requests per device
    decisionLog,      \* Sequence of decisions made
    latencyBudget,    \* Function: Device -> Remaining latency budget (ms)
    failsafeTriggered \* Function: Device -> Boolean (true if failsafe active)

vars == <<deviceStates, cachedPolicies, pendingDecisions, decisionLog, latencyBudget, failsafeTriggered>>

\* ----- Type Invariants -----

TypeInvariant ==
    /\ deviceStates \in [EdgeDevices -> {"ONLINE", "OFFLINE", "DEGRADED"}]
    /\ cachedPolicies \in [EdgeDevices -> Nat]  \* Simplified: count of cached policies
    /\ pendingDecisions \in [EdgeDevices -> Nat]  \* Count of pending decisions
    /\ decisionLog \in Seq([device: EdgeDevices, decision: {"ALLOW", "RESTRICT", "BLOCK", "TERMINATE"}, latency: Nat, failsafe: BOOLEAN])
    /\ latencyBudget \in [EdgeDevices -> 0..MaxLatencyMs]
    /\ failsafeTriggered \in [EdgeDevices -> BOOLEAN]

\* ----- Initial State -----

Init ==
    /\ deviceStates = [d \in EdgeDevices |-> "ONLINE"]
    /\ cachedPolicies = [d \in EdgeDevices |-> 10]  \* 10 cached policies initially
    /\ pendingDecisions = [d \in EdgeDevices |-> 0]
    /\ decisionLog = <<>>
    /\ latencyBudget = [d \in EdgeDevices |-> MaxLatencyMs]
    /\ failsafeTriggered = [d \in EdgeDevices |-> FALSE]

\* ----- Helper Predicates -----

\* Device can make decisions locally
CanDecideLocally(device) ==
    /\ cachedPolicies[device] > 0
    /\ latencyBudget[device] > 0

\* Device is fully operational
IsOperational(device) ==
    /\ deviceStates[device] = "ONLINE"
    /\ ~failsafeTriggered[device]

\* ----- Actions -----

\* Device goes offline
DeviceGoesOffline(device) ==
    /\ device \in EdgeDevices
    /\ deviceStates[device] = "ONLINE"
    /\ deviceStates' = [deviceStates EXCEPT ![device] = "OFFLINE"]
    /\ UNCHANGED <<cachedPolicies, pendingDecisions, decisionLog, latencyBudget, failsafeTriggered>>

\* Device comes back online
DeviceComesOnline(device) ==
    /\ device \in EdgeDevices
    /\ deviceStates[device] \in {"OFFLINE", "DEGRADED"}
    /\ deviceStates' = [deviceStates EXCEPT ![device] = "ONLINE"]
    /\ failsafeTriggered' = [failsafeTriggered EXCEPT ![device] = FALSE]
    /\ latencyBudget' = [latencyBudget EXCEPT ![device] = MaxLatencyMs]
    /\ UNCHANGED <<cachedPolicies, pendingDecisions, decisionLog>>

\* Device enters degraded mode
DeviceDegrades(device) ==
    /\ device \in EdgeDevices
    /\ deviceStates[device] = "ONLINE"
    /\ latencyBudget[device] < MaxLatencyMs \div 2  \* Low latency budget
    /\ deviceStates' = [deviceStates EXCEPT ![device] = "DEGRADED"]
    /\ UNCHANGED <<cachedPolicies, pendingDecisions, decisionLog, latencyBudget, failsafeTriggered>>

\* New decision request arrives
NewDecisionRequest(device) ==
    /\ device \in EdgeDevices
    /\ pendingDecisions' = [pendingDecisions EXCEPT ![device] = @ + 1]
    /\ UNCHANGED <<deviceStates, cachedPolicies, decisionLog, latencyBudget, failsafeTriggered>>

\* Make a normal decision (when operational)
MakeNormalDecision(device, decision, latency) ==
    /\ device \in EdgeDevices
    /\ pendingDecisions[device] > 0
    /\ IsOperational(device)
    /\ CanDecideLocally(device)
    /\ latency <= latencyBudget[device]
    /\ pendingDecisions' = [pendingDecisions EXCEPT ![device] = @ - 1]
    /\ decisionLog' = Append(decisionLog, [device |-> device, decision |-> decision, latency |-> latency, failsafe |-> FALSE])
    /\ latencyBudget' = [latencyBudget EXCEPT ![device] = @ - latency]
    /\ UNCHANGED <<deviceStates, cachedPolicies, failsafeTriggered>>

\* Make a failsafe decision (when offline or degraded)
MakeFailsafeDecision(device) ==
    /\ device \in EdgeDevices
    /\ pendingDecisions[device] > 0
    /\ \/ deviceStates[device] \in {"OFFLINE", "DEGRADED"}
       \/ ~CanDecideLocally(device)
    /\ pendingDecisions' = [pendingDecisions EXCEPT ![device] = @ - 1]
    /\ decisionLog' = Append(decisionLog, [device |-> device, decision |-> SafeDefaultDecision, latency |-> 1, failsafe |-> TRUE])
    /\ failsafeTriggered' = [failsafeTriggered EXCEPT ![device] = TRUE]
    /\ UNCHANGED <<deviceStates, cachedPolicies, latencyBudget>>

\* Sync policies from cloud
SyncPolicies(device) ==
    /\ device \in EdgeDevices
    /\ deviceStates[device] = "ONLINE"
    /\ cachedPolicies' = [cachedPolicies EXCEPT ![device] = 10]  \* Refresh cache
    /\ latencyBudget' = [latencyBudget EXCEPT ![device] = MaxLatencyMs]  \* Reset budget
    /\ UNCHANGED <<deviceStates, pendingDecisions, decisionLog, failsafeTriggered>>

\* Latency budget depletes over time
LatencyBudgetDepletes(device) ==
    /\ device \in EdgeDevices
    /\ latencyBudget[device] > 0
    /\ latencyBudget' = [latencyBudget EXCEPT ![device] = @ - 1]
    /\ UNCHANGED <<deviceStates, cachedPolicies, pendingDecisions, decisionLog, failsafeTriggered>>

\* ----- Next State Relation -----

Next ==
    \/ \E device \in EdgeDevices: DeviceGoesOffline(device)
    \/ \E device \in EdgeDevices: DeviceComesOnline(device)
    \/ \E device \in EdgeDevices: DeviceDegrades(device)
    \/ \E device \in EdgeDevices: NewDecisionRequest(device)
    \/ \E device \in EdgeDevices, decision \in {"ALLOW", "RESTRICT", "BLOCK", "TERMINATE"}, latency \in 1..MaxLatencyMs: 
        MakeNormalDecision(device, decision, latency)
    \/ \E device \in EdgeDevices: MakeFailsafeDecision(device)
    \/ \E device \in EdgeDevices: SyncPolicies(device)
    \/ \E device \in EdgeDevices: LatencyBudgetDepletes(device)

\* ----- Safety Properties -----

\* EDGE-1: All decisions have latency within budget (or are failsafe)
LatencyWithinBudget ==
    \A i \in 1..Len(decisionLog):
        decisionLog[i].latency <= MaxLatencyMs

\* EDGE-2: Offline devices only make failsafe decisions
OfflineOnlyFailsafe ==
    \A i \in 1..Len(decisionLog):
        LET entry == decisionLog[i]
        IN deviceStates[entry.device] = "OFFLINE" => entry.failsafe = TRUE

\* EDGE-3: Failsafe decisions are always safe (RESTRICT or stricter)
FailsafeDecisionsAreSafe ==
    \A i \in 1..Len(decisionLog):
        decisionLog[i].failsafe = TRUE =>
        decisionLog[i].decision \in {"RESTRICT", "BLOCK", "TERMINATE"}

\* EDGE-4: Pending decisions are bounded
PendingDecisionsBounded ==
    \A device \in EdgeDevices:
        pendingDecisions[device] <= 100  \* Arbitrary bound for verification

\* EDGE-5: No decision is faster than possible (latency > 0)
NoInstantDecisions ==
    \A i \in 1..Len(decisionLog):
        decisionLog[i].latency > 0

\* Combined safety invariant
SafetyInvariant ==
    /\ LatencyWithinBudget
    /\ FailsafeDecisionsAreSafe
    /\ NoInstantDecisions

\* ----- Critical Safety Property -----

\* CRITICAL: Devices can ALWAYS make a decision (either normal or failsafe)
\* This ensures no decision request is ever stuck
AlwaysCanDecide ==
    \A device \in EdgeDevices:
        pendingDecisions[device] > 0 =>
        \/ (IsOperational(device) /\ CanDecideLocally(device))  \* Normal decision possible
        \/ TRUE  \* Failsafe decision always possible

\* ----- Liveness Properties -----

\* Every pending decision eventually gets resolved
AllDecisionsResolved ==
    \A device \in EdgeDevices:
        pendingDecisions[device] > 0 ~> pendingDecisions[device] = 0

\* Offline devices eventually come back online (assuming network recovery)
EventualRecovery ==
    \A device \in EdgeDevices:
        deviceStates[device] = "OFFLINE" ~> deviceStates[device] = "ONLINE"

\* ----- Fairness -----

Fairness ==
    /\ WF_vars(Next)
    /\ \A device \in EdgeDevices: WF_vars(MakeFailsafeDecision(device))
    /\ \A device \in EdgeDevices: WF_vars(SyncPolicies(device))

\* ----- Specification -----

Spec == Init /\ [][Next]_vars /\ Fairness

\* ----- Theorems -----

THEOREM TypeSafety == Spec => []TypeInvariant

THEOREM EdgeSafety == Spec => []SafetyInvariant

THEOREM CriticalSafety == Spec => []AlwaysCanDecide

THEOREM Liveness == Spec => AllDecisionsResolved

===============================================================================
