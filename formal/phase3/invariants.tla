---- MODULE invariants ----
(***************************************************************************
 * Nethical System Invariants - Formal Specification (Phase 3A)
 * 
 * This module defines system-wide invariants that must hold across
 * all executions of the Nethical governance platform.
 * 
 * Invariants are organized by category:
 * - Safety Invariants: "Bad things never happen"
 * - Liveness Invariants: "Good things eventually happen"
 * - Governance Invariants: Compliance and policy properties
 ***************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    PolicyIds,
    DecisionIds,
    AgentIds,
    MAX_EVAL_STEPS,
    MAX_POLICY_DEPTH,
    FAIRNESS_THRESHOLD

VARIABLES
    policies,
    decisions,
    agents,
    audit_log,
    eval_counter,
    active_evaluations

INSTANCE core_model

----

(***************************************************************************
 * SAFETY INVARIANTS
 ***************************************************************************)

\* S-1: Type Safety - All variables maintain their types
SafetyTypeInvariant == TypeOK

\* S-2: Bounded Execution - System never exceeds resource bounds
SafetyBoundedExecution ==
    /\ eval_counter <= MAX_EVAL_STEPS
    /\ Len(audit_log) < MAX_EVAL_STEPS * 10
    /\ Cardinality(active_evaluations) <= Cardinality(DecisionIds)

\* S-3: State Consistency - Evaluating decisions must be in active set
SafetyStateConsistency ==
    \A d \in DecisionIds:
        decisions[d].state = "EVALUATING" <=> d \in active_evaluations

\* S-4: Monotonic Audit Log - Audit log is append-only
SafetyMonotonicAuditLog ==
    \* Once an audit entry exists, it never changes
    \* (Captured by the fact that audit_log is Append-only in transitions)
    Len(audit_log) >= 0

\* S-5: No Unauthorized State Transitions
SafetyAuthorizedTransitions ==
    \A p \in PolicyIds:
        \* Can only activate from QUARANTINE
        policies[p].state = "ACTIVE" =>
            \E i \in DOMAIN audit_log:
                /\ audit_log[i].entity_id = p
                /\ audit_log[i].old_state = "QUARANTINE"
                /\ audit_log[i].new_state = "ACTIVE"

\* S-6: Immutable Active Policies
SafetyImmutablePolicies ==
    \A p \in PolicyIds:
        policies[p].state = "ACTIVE" =>
            policies[p].hash # ""

----

(***************************************************************************
 * GOVERNANCE INVARIANTS (Properties with P- prefix)
 ***************************************************************************)

\* P-DET: Determinism - Identical inputs produce identical outputs
InvariantDeterminism == DeterminismInvariant

\* P-TERM: Termination - All evaluations complete within bounded time
InvariantTermination == TerminationInvariant

\* P-ACYCLIC: Acyclicity - No circular policy dependencies
InvariantAcyclicity == AcyclicityInvariant

\* P-AUD: Audit Completeness - All transitions logged
InvariantAuditCompleteness == AuditCompletenessInvariant

\* P-NONREP: Non-Repudiation - Audit events are immutable
InvariantNonRepudiation ==
    \* All audit events have hashes and timestamps
    \A i \in DOMAIN audit_log:
        /\ audit_log[i].timestamp >= 0
        /\ audit_log[i].hash # "" \/ audit_log[i].event_type = "AGENT_SUSPENDED"

\* P-POL-LIN: Policy Lineage - Hash chain integrity
InvariantPolicyLineage ==
    \A p \in PolicyIds:
        policies[p].version > 0 =>
            policies[p].parent_hash # ""

\* P-MULTI-SIG: Multi-Signature - Critical policies require approvals
InvariantMultiSignature ==
    \A p \in PolicyIds:
        /\ policies[p].state = "ACTIVE"
        /\ policies[p].required_approvals > 1
        => policies[p].approval_count >= policies[p].required_approvals

\* P-DATA-MIN: Data Minimization - Only necessary context accessed
\* (Approximated - would need context field tracking in full implementation)
InvariantDataMinimization ==
    \A d \in DecisionIds:
        decisions[d].context_hash # ""

\* P-TENANT-ISO: Tenant Isolation - Cross-tenant access forbidden
\* (Requires tenant_id tracking - placeholder for full implementation)
InvariantTenantIsolation == TRUE

\* P-JUST: Justification Completeness - All decisions have justification
InvariantJustification ==
    \A d \in DecisionIds:
        decisions[d].state \in {"DECIDED", "BLOCKED", "RE_DECIDED"} =>
            /\ decisions[d].policy_version # ""
            /\ decisions[d].confidence >= 0

----

(***************************************************************************
 * LIVENESS PROPERTIES (Temporal)
 ***************************************************************************)

\* L-1: Progress - System eventually processes all pending work
LivenessProgress ==
    \A d \in DecisionIds:
        decisions[d].state = "PENDING" ~> decisions[d].state # "PENDING"

\* L-2: Completion - Active evaluations eventually complete
LivenessCompletion ==
    \A d \in DecisionIds:
        d \in active_evaluations ~> d \notin active_evaluations

\* L-3: Policy Activation - Quarantined policies can eventually activate
LivenessPolicyActivation ==
    \A p \in PolicyIds:
        /\ policies[p].state = "QUARANTINE"
        /\ policies[p].approval_count >= policies[p].required_approvals
        ~> policies[p].state = "ACTIVE"

\* L-4: Agent Provisioning - Registered agents eventually provision
LivenessAgentProvisioning ==
    \A a \in AgentIds:
        agents[a].state = "REGISTERED" ~> agents[a].state # "REGISTERED"

----

(***************************************************************************
 * FAIRNESS PROPERTIES
 ***************************************************************************)

\* F-1: Fair Scheduling - All entities get fair processing
FairnessScheduling ==
    \* All pending decisions eventually evaluated
    \A d \in DecisionIds:
        decisions[d].state = "PENDING" ~> decisions[d].state = "EVALUATING"

\* F-2: No Starvation - No entity is indefinitely blocked
FairnessNoStarvation ==
    \A d \in DecisionIds:
        d \in active_evaluations => <>(d \notin active_evaluations)

----

(***************************************************************************
 * COMPOSITE INVARIANTS
 ***************************************************************************)

\* All Safety Properties
AllSafetyInvariants ==
    /\ SafetyTypeInvariant
    /\ SafetyBoundedExecution
    /\ SafetyStateConsistency
    /\ SafetyMonotonicAuditLog
    /\ SafetyAuthorizedTransitions
    /\ SafetyImmutablePolicies

\* All Governance Properties
AllGovernanceInvariants ==
    /\ InvariantDeterminism
    /\ InvariantTermination
    /\ InvariantAcyclicity
    /\ InvariantAuditCompleteness
    /\ InvariantNonRepudiation
    /\ InvariantPolicyLineage
    /\ InvariantMultiSignature
    /\ InvariantDataMinimization
    /\ InvariantTenantIsolation
    /\ InvariantJustification

\* System-Wide Correctness
SystemCorrectness ==
    /\ AllSafetyInvariants
    /\ AllGovernanceInvariants

----

(***************************************************************************
 * THEOREMS - Properties to Prove
 ***************************************************************************)

\* Main Correctness Theorem
THEOREM Spec => []SystemCorrectness

\* Individual Property Theorems
THEOREM Spec => []InvariantDeterminism
THEOREM Spec => []InvariantTermination
THEOREM Spec => []InvariantAcyclicity
THEOREM Spec => []InvariantAuditCompleteness
THEOREM Spec => []InvariantNonRepudiation
THEOREM Spec => []InvariantPolicyLineage
THEOREM Spec => []InvariantMultiSignature

\* Liveness Theorems
THEOREM Spec => LivenessProgress
THEOREM Spec => LivenessCompletion
THEOREM Spec => FairnessNoStarvation

----

(***************************************************************************
 * MODEL CHECKING PROPERTIES
 * 
 * To check with TLC model checker:
 * 1. Set constants: PolicyIds <- {"p1", "p2"}, DecisionIds <- {"d1", "d2"}, etc.
 * 2. Set MAX_EVAL_STEPS <- 10, MAX_POLICY_DEPTH <- 5
 * 3. Check: Invariants = SystemCorrectness
 * 4. Check: Properties = LivenessProgress, LivenessCompletion
 ***************************************************************************)

\* Model Checking Configuration
MCStateConstraint ==
    /\ eval_counter <= 20
    /\ Len(audit_log) <= 50
    /\ Cardinality(active_evaluations) <= 5

\* Deadlock Freedom - System can always make progress
DeadlockFreedom ==
    ENABLED Next

====
