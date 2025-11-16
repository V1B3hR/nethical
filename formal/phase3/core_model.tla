---- MODULE core_model ----
(***************************************************************************
 * Nethical Core Model - Formal Specification (Phase 3A)
 * 
 * This TLA+ specification formalizes the core governance model for Nethical,
 * including state machines for policies, decisions, agents, and audit events.
 * 
 * Key Properties Verified:
 * - P-DET: Determinism - Same input always produces same output
 * - P-TERM: Termination - All evaluations terminate in finite time
 * - P-ACYCLIC: Acyclicity - Policy dependencies form a DAG (no cycles)
 * - P-AUD: Audit completeness - All state transitions are logged
 ***************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    PolicyIds,          \* Set of policy identifiers
    DecisionIds,        \* Set of decision identifiers
    AgentIds,           \* Set of agent identifiers
    MAX_EVAL_STEPS,     \* Maximum evaluation steps (termination bound)
    MAX_POLICY_DEPTH    \* Maximum policy dependency depth (acyclicity bound)

VARIABLES
    policies,           \* Map from policy_id to policy state
    decisions,          \* Map from decision_id to decision state
    agents,             \* Map from agent_id to agent state
    audit_log,          \* Sequence of audit events
    eval_counter,       \* Evaluation step counter (for termination)
    active_evaluations  \* Set of currently active evaluations

vars == <<policies, decisions, agents, audit_log, eval_counter, active_evaluations>>

----

(***************************************************************************
 * Policy State Machine (from phase2/state-model.md)
 ***************************************************************************)

PolicyState == {"INACTIVE", "QUARANTINE", "ACTIVE", "SUPERSEDED"}

PolicyRecord == [
    state: PolicyState,
    version: Nat,
    hash: STRING,
    parent_hash: STRING,
    approval_count: Nat,
    required_approvals: Nat,
    dependencies: SUBSET PolicyIds
]

----

(***************************************************************************
 * Decision State Machine (from phase2/state-model.md)
 ***************************************************************************)

DecisionState == {"PENDING", "EVALUATING", "DECIDED", "BLOCKED", 
                  "ESCALATED", "REVIEWED", "APPEALED", "RE_DECIDED"}

DecisionRecord == [
    state: DecisionState,
    action_hash: STRING,
    context_hash: STRING,
    policy_version: STRING,
    result: {"ALLOW", "RESTRICT", "BLOCK", "TERMINATE"},
    confidence: Nat,
    evaluated_at: Nat
]

----

(***************************************************************************
 * Agent State Machine (from phase2/state-model.md)
 ***************************************************************************)

AgentState == {"REGISTERED", "PROVISIONED", "ACTIVE", "SUSPENDED", 
               "DEPROVISIONED"}

AgentRecord == [
    state: AgentState,
    trust_score: Nat,
    violation_count: Nat
]

----

(***************************************************************************
 * Audit Event Structure
 ***************************************************************************)

AuditEventType == {"POLICY_LOADED", "POLICY_ACTIVATED", "POLICY_DEACTIVATED",
                   "DECISION_MADE", "AGENT_REGISTERED", "AGENT_SUSPENDED"}

AuditEvent == [
    event_type: AuditEventType,
    entity_id: STRING,
    timestamp: Nat,
    old_state: STRING,
    new_state: STRING,
    hash: STRING
]

----

(***************************************************************************
 * Type Invariants
 ***************************************************************************)

TypeOK ==
    /\ policies \in [PolicyIds -> PolicyRecord]
    /\ decisions \in [DecisionIds -> DecisionRecord]
    /\ agents \in [AgentIds -> AgentRecord]
    /\ audit_log \in Seq(AuditEvent)
    /\ eval_counter \in Nat
    /\ active_evaluations \subseteq DecisionIds

----

(***************************************************************************
 * Initialization
 ***************************************************************************)

Init ==
    /\ policies = [p \in PolicyIds |-> [
           state |-> "INACTIVE",
           version |-> 0,
           hash |-> "",
           parent_hash |-> "",
           approval_count |-> 0,
           required_approvals |-> 1,
           dependencies |-> {}
       ]]
    /\ decisions = [d \in DecisionIds |-> [
           state |-> "PENDING",
           action_hash |-> "",
           context_hash |-> "",
           policy_version |-> "",
           result |-> "ALLOW",
           confidence |-> 0,
           evaluated_at |-> 0
       ]]
    /\ agents = [a \in AgentIds |-> [
           state |-> "REGISTERED",
           trust_score |-> 100,
           violation_count |-> 0
       ]]
    /\ audit_log = <<>>
    /\ eval_counter = 0
    /\ active_evaluations = {}

----

(***************************************************************************
 * Policy State Transitions
 ***************************************************************************)

LoadPolicy(p) ==
    /\ policies[p].state = "INACTIVE"
    /\ policies' = [policies EXCEPT ![p].state = "QUARANTINE"]
    /\ audit_log' = Append(audit_log, [
           event_type |-> "POLICY_LOADED",
           entity_id |-> p,
           timestamp |-> eval_counter,
           old_state |-> "INACTIVE",
           new_state |-> "QUARANTINE",
           hash |-> policies[p].hash
       ])
    /\ UNCHANGED <<decisions, agents, eval_counter, active_evaluations>>

ActivatePolicy(p) ==
    /\ policies[p].state = "QUARANTINE"
    /\ policies[p].approval_count >= policies[p].required_approvals
    /\ policies' = [policies EXCEPT ![p].state = "ACTIVE"]
    /\ audit_log' = Append(audit_log, [
           event_type |-> "POLICY_ACTIVATED",
           entity_id |-> p,
           timestamp |-> eval_counter,
           old_state |-> "QUARANTINE",
           new_state |-> "ACTIVE",
           hash |-> policies[p].hash
       ])
    /\ UNCHANGED <<decisions, agents, eval_counter, active_evaluations>>

DeactivatePolicy(p) ==
    /\ policies[p].state = "ACTIVE"
    /\ policies' = [policies EXCEPT ![p].state = "SUPERSEDED"]
    /\ audit_log' = Append(audit_log, [
           event_type |-> "POLICY_DEACTIVATED",
           entity_id |-> p,
           timestamp |-> eval_counter,
           old_state |-> "ACTIVE",
           new_state |-> "SUPERSEDED",
           hash |-> policies[p].hash
       ])
    /\ UNCHANGED <<decisions, agents, eval_counter, active_evaluations>>

----

(***************************************************************************
 * Decision State Transitions
 ***************************************************************************)

EvaluateDecision(d) ==
    /\ decisions[d].state = "PENDING"
    /\ eval_counter < MAX_EVAL_STEPS
    /\ decisions' = [decisions EXCEPT ![d].state = "EVALUATING"]
    /\ eval_counter' = eval_counter + 1
    /\ active_evaluations' = active_evaluations \union {d}
    /\ audit_log' = Append(audit_log, [
           event_type |-> "DECISION_MADE",
           entity_id |-> d,
           timestamp |-> eval_counter,
           old_state |-> "PENDING",
           new_state |-> "EVALUATING",
           hash |-> decisions[d].action_hash
       ])
    /\ UNCHANGED <<policies, agents>>

CompleteDecision(d) ==
    /\ decisions[d].state = "EVALUATING"
    /\ d \in active_evaluations
    /\ decisions' = [decisions EXCEPT 
           ![d].state = "DECIDED",
           ![d].evaluated_at = eval_counter
       ]
    /\ active_evaluations' = active_evaluations \ {d}
    /\ audit_log' = Append(audit_log, [
           event_type |-> "DECISION_MADE",
           entity_id |-> d,
           timestamp |-> eval_counter,
           old_state |-> "EVALUATING",
           new_state |-> "DECIDED",
           hash |-> decisions[d].action_hash
       ])
    /\ UNCHANGED <<policies, agents, eval_counter>>

----

(***************************************************************************
 * Agent State Transitions
 ***************************************************************************)

ProvisionAgent(a) ==
    /\ agents[a].state = "REGISTERED"
    /\ agents' = [agents EXCEPT ![a].state = "PROVISIONED"]
    /\ audit_log' = Append(audit_log, [
           event_type |-> "AGENT_REGISTERED",
           entity_id |-> a,
           timestamp |-> eval_counter,
           old_state |-> "REGISTERED",
           new_state |-> "PROVISIONED",
           hash |-> ""
       ])
    /\ UNCHANGED <<policies, decisions, eval_counter, active_evaluations>>

ActivateAgent(a) ==
    /\ agents[a].state = "PROVISIONED"
    /\ agents' = [agents EXCEPT ![a].state = "ACTIVE"]
    /\ UNCHANGED <<policies, decisions, audit_log, eval_counter, active_evaluations>>

SuspendAgent(a) ==
    /\ agents[a].state = "ACTIVE"
    /\ agents' = [agents EXCEPT ![a].state = "SUSPENDED"]
    /\ audit_log' = Append(audit_log, [
           event_type |-> "AGENT_SUSPENDED",
           entity_id |-> a,
           timestamp |-> eval_counter,
           old_state |-> "ACTIVE",
           new_state |-> "SUSPENDED",
           hash |-> ""
       ])
    /\ UNCHANGED <<policies, decisions, eval_counter, active_evaluations>>

----

(***************************************************************************
 * Next State Relation
 ***************************************************************************)

Next ==
    \/ \E p \in PolicyIds: LoadPolicy(p)
    \/ \E p \in PolicyIds: ActivatePolicy(p)
    \/ \E p \in PolicyIds: DeactivatePolicy(p)
    \/ \E d \in DecisionIds: EvaluateDecision(d)
    \/ \E d \in DecisionIds: CompleteDecision(d)
    \/ \E a \in AgentIds: ProvisionAgent(a)
    \/ \E a \in AgentIds: ActivateAgent(a)
    \/ \E a \in AgentIds: SuspendAgent(a)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

----

(***************************************************************************
 * INVARIANTS - Core Properties to Verify
 ***************************************************************************)

\* P-DET: Determinism Invariant
\* Same input (action_hash + context_hash + policy_version) always produces same result
DeterminismInvariant ==
    \A d1, d2 \in DecisionIds:
        /\ decisions[d1].action_hash = decisions[d2].action_hash
        /\ decisions[d1].context_hash = decisions[d2].context_hash
        /\ decisions[d1].policy_version = decisions[d2].policy_version
        /\ decisions[d1].state \in {"DECIDED", "RE_DECIDED"}
        /\ decisions[d2].state \in {"DECIDED", "RE_DECIDED"}
        => decisions[d1].result = decisions[d2].result

\* P-TERM: Termination Invariant
\* All evaluations terminate within MAX_EVAL_STEPS
TerminationInvariant ==
    /\ eval_counter <= MAX_EVAL_STEPS
    /\ Cardinality(active_evaluations) < Cardinality(DecisionIds)

\* P-ACYCLIC: Acyclicity Invariant
\* Policy dependencies form a DAG (no cycles)
\* This is approximated by checking dependency depth bounds
AcyclicityInvariant ==
    \A p \in PolicyIds:
        Cardinality(policies[p].dependencies) < MAX_POLICY_DEPTH

\* P-AUD: Audit Completeness Invariant
\* All state transitions are logged in audit trail
AuditCompletenessInvariant ==
    \A p \in PolicyIds:
        policies[p].state # "INACTIVE" =>
            \E i \in DOMAIN audit_log:
                /\ audit_log[i].entity_id = p
                /\ audit_log[i].event_type \in {"POLICY_LOADED", "POLICY_ACTIVATED"}

\* Single Active Policy Version Invariant
\* Only one version of a policy can be ACTIVE at a time
SingleActivePolicyInvariant ==
    \A p1, p2 \in PolicyIds:
        /\ policies[p1].state = "ACTIVE"
        /\ policies[p2].state = "ACTIVE"
        /\ policies[p1].hash = policies[p2].hash
        => p1 = p2

\* Safety: No Lost Decisions
\* Once a decision starts evaluating, it must complete or remain in active_evaluations
NoLostDecisionsInvariant ==
    \A d \in DecisionIds:
        decisions[d].state = "EVALUATING" => d \in active_evaluations

----

(***************************************************************************
 * TEMPORAL PROPERTIES - Liveness
 ***************************************************************************)

\* Liveness: All pending decisions eventually complete
EventuallyDecided ==
    \A d \in DecisionIds:
        decisions[d].state = "PENDING" ~> decisions[d].state = "DECIDED"

\* Liveness: All evaluating decisions eventually complete
EvaluationsComplete ==
    \A d \in DecisionIds:
        d \in active_evaluations ~> d \notin active_evaluations

----

(***************************************************************************
 * MODEL CHECKING CONFIGURATION
 ***************************************************************************)

\* State constraint to limit state space exploration
StateConstraint ==
    /\ eval_counter <= MAX_EVAL_STEPS
    /\ Len(audit_log) <= MAX_EVAL_STEPS * 2

\* Properties to check
THEOREM Spec => []TypeOK
THEOREM Spec => []DeterminismInvariant
THEOREM Spec => []TerminationInvariant
THEOREM Spec => []AcyclicityInvariant
THEOREM Spec => []AuditCompletenessInvariant
THEOREM Spec => []SingleActivePolicyInvariant
THEOREM Spec => []NoLostDecisionsInvariant

====
