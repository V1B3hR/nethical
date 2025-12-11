---------------------------- MODULE PolicyConsistency ----------------------------
\* TLA+ Specification for Nethical Policy Consistency Verification
\* Verifies that policies don't conflict and produce deterministic decisions
\* 
\* This specification ensures:
\* - No two policies contradict each other
\* - Policy priorities resolve ambiguities
\* - All action types have a default policy
\* - Policy updates maintain consistency

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    PolicyIds,          \* Set of policy identifiers
    ActionTypes,        \* Set of action types (e.g., "DATA_ACCESS", "EXECUTE_CODE")
    Priorities,         \* Set of priority levels (1..100)
    MaxPolicies        \* Maximum number of policies

VARIABLES
    policies,           \* Set of active policies
    policyPriorities,   \* Function: PolicyId -> Priority
    policyRules,        \* Function: PolicyId -> [actionType: ActionTypes, decision: Decisions]
    conflictLog,        \* Sequence of detected conflicts
    activeVersion       \* Current policy version number

vars == <<policies, policyPriorities, policyRules, conflictLog, activeVersion>>

\* ----- Type Definitions -----

Decisions == {"ALLOW", "RESTRICT", "BLOCK", "TERMINATE"}

PolicyRecord == [
    id: PolicyIds,
    priority: Priorities,
    actionType: ActionTypes,
    decision: Decisions,
    conditions: STRING
]

TypeInvariant ==
    /\ policies \subseteq PolicyIds
    /\ Cardinality(policies) <= MaxPolicies
    /\ policyPriorities \in [PolicyIds -> Priorities]
    /\ policyRules \in [PolicyIds -> [actionType: ActionTypes, decision: Decisions]]
    /\ conflictLog \in Seq([policy1: PolicyIds, policy2: PolicyIds, actionType: ActionTypes])
    /\ activeVersion \in Nat

\* ----- Initial State -----

Init ==
    /\ policies = {}
    /\ policyPriorities = [p \in PolicyIds |-> 0]
    /\ policyRules = [p \in PolicyIds |-> [actionType |-> "NONE", decision |-> "ALLOW"]]
    /\ conflictLog = <<>>
    /\ activeVersion = 1

\* ----- Helper Functions -----

\* Check if two policies conflict for the same action type
PoliciesConflict(p1, p2) ==
    /\ p1 \in policies
    /\ p2 \in policies
    /\ p1 # p2
    /\ policyRules[p1].actionType = policyRules[p2].actionType
    /\ policyRules[p1].decision # policyRules[p2].decision
    /\ policyPriorities[p1] = policyPriorities[p2]  \* Same priority = conflict

\* Get the effective decision for an action type
EffectiveDecision(actionType) ==
    LET applicablePolicies == {p \in policies : policyRules[p].actionType = actionType}
        maxPriority == IF applicablePolicies = {} 
                       THEN 0 
                       ELSE CHOOSE max \in {policyPriorities[p] : p \in applicablePolicies} :
                           \A p \in applicablePolicies : policyPriorities[p] <= max
        highestPolicy == CHOOSE p \in applicablePolicies : policyPriorities[p] = maxPriority
    IN IF applicablePolicies = {} 
       THEN "ALLOW"  \* Default decision
       ELSE policyRules[highestPolicy].decision

\* Check if policy set is deterministic (no unresolved conflicts)
IsDeterministic ==
    ~\E p1, p2 \in policies : PoliciesConflict(p1, p2)

\* ----- Actions -----

\* Add a new policy
AddPolicy(policyId, priority, actionType, decision) ==
    /\ policyId \in PolicyIds
    /\ policyId \notin policies
    /\ Cardinality(policies) < MaxPolicies
    /\ priority \in Priorities
    /\ actionType \in ActionTypes
    /\ decision \in Decisions
    /\ policies' = policies \union {policyId}
    /\ policyPriorities' = [policyPriorities EXCEPT ![policyId] = priority]
    /\ policyRules' = [policyRules EXCEPT ![policyId] = [actionType |-> actionType, decision |-> decision]]
    /\ activeVersion' = activeVersion + 1
    /\ UNCHANGED <<conflictLog>>

\* Remove a policy
RemovePolicy(policyId) ==
    /\ policyId \in policies
    /\ policies' = policies \ {policyId}
    /\ activeVersion' = activeVersion + 1
    /\ UNCHANGED <<policyPriorities, policyRules, conflictLog>>

\* Update policy priority to resolve conflicts
UpdatePriority(policyId, newPriority) ==
    /\ policyId \in policies
    /\ newPriority \in Priorities
    /\ policyPriorities' = [policyPriorities EXCEPT ![policyId] = newPriority]
    /\ activeVersion' = activeVersion + 1
    /\ UNCHANGED <<policies, policyRules, conflictLog>>

\* Detect and log a conflict
DetectConflict(p1, p2) ==
    /\ PoliciesConflict(p1, p2)
    /\ conflictLog' = Append(conflictLog, [
           policy1 |-> p1,
           policy2 |-> p2,
           actionType |-> policyRules[p1].actionType
       ])
    /\ UNCHANGED <<policies, policyPriorities, policyRules, activeVersion>>

\* ----- State Machine Specification -----

Next ==
    \/ \E pid \in PolicyIds, pri \in Priorities, act \in ActionTypes, dec \in Decisions :
           AddPolicy(pid, pri, act, dec)
    \/ \E pid \in policies : RemovePolicy(pid)
    \/ \E pid \in policies, pri \in Priorities : UpdatePriority(pid, pri)
    \/ \E p1, p2 \in policies : DetectConflict(p1, p2)

Spec == Init /\ [][Next]_vars

\* ----- Safety Properties -----

\* Property 1: Policy set is always deterministic (no unresolved conflicts)
DeterminismProperty ==
    [](IsDeterministic)

\* Property 2: Every action type has an effective decision
CompletenessProperty ==
    [](\A actionType \in ActionTypes :
        \E decision \in Decisions : EffectiveDecision(actionType) = decision)

\* Property 3: Higher priority policies override lower priority ones
PriorityRespectedProperty ==
    [](\A actionType \in ActionTypes :
        LET decision == EffectiveDecision(actionType)
            applicablePolicies == {p \in policies : policyRules[p].actionType = actionType}
            maxPriority == IF applicablePolicies = {} 
                          THEN 0 
                          ELSE CHOOSE max \in {policyPriorities[p] : p \in applicablePolicies} :
                              \A p \in applicablePolicies : policyPriorities[p] <= max
        IN \A p \in applicablePolicies :
            (policyPriorities[p] = maxPriority) => (policyRules[p].decision = decision))

\* Property 4: Adding policies doesn't violate cardinality constraint
CardinalityProperty ==
    [](Cardinality(policies) <= MaxPolicies)

\* Property 5: Version monotonically increases
VersionMonotonicProperty ==
    []([activeVersion' >= activeVersion]_vars)

\* Property 6: No same-priority conflicting policies exist
NoConflictProperty ==
    [](~\E p1, p2 \in policies : PoliciesConflict(p1, p2))

\* ----- Liveness Properties -----

\* Property 7: Conflicts are eventually detected
ConflictsEventuallyDetected ==
    \A p1, p2 \in PolicyIds :
        (PoliciesConflict(p1, p2)) ~> 
        (\E i \in DOMAIN conflictLog :
            conflictLog[i].policy1 = p1 /\ conflictLog[i].policy2 = p2)

\* Property 8: System eventually reaches deterministic state
EventuallyDeterministic ==
    <>[]IsDeterministic

\* ----- Consistency Theorems -----

\* Theorem: If no conflicts exist, the system is deterministic
THEOREM NoConflictProperty => DeterminismProperty

\* Theorem: The system always provides a decision for any action type
THEOREM Spec => []CompletenessProperty

\* Theorem: Version numbers never decrease
THEOREM Spec => []VersionMonotonicProperty

\* ----- Model Checking Configuration -----

THEOREM Spec => []TypeInvariant
THEOREM Spec => []CardinalityProperty
THEOREM Spec => []CompletenessProperty

================================================================================
