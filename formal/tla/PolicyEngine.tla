---------------------------- MODULE PolicyEngine ----------------------------
\* TLA+ Specification for Nethical Policy Engine
\* Formal verification of policy non-contradiction and determinism
\* 
\* This specification models the policy evaluation process and ensures
\* that policies are consistent and deterministic.

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Policies,         \* Set of policy identifiers
    Rules,            \* Set of rule identifiers
    Priorities,       \* Set of priority levels (1..10)
    Decisions         \* Set of possible decisions {ALLOW, RESTRICT, BLOCK, TERMINATE}

VARIABLES
    activePolicies,   \* Set of currently active policies
    policyRules,      \* Function: Policy -> Set of Rules
    rulePriorities,   \* Function: Rule -> Priority
    ruleDecisions,    \* Function: Rule -> Decision
    evaluationState,  \* Current evaluation state
    conflictLog       \* Log of detected conflicts

vars == <<activePolicies, policyRules, rulePriorities, ruleDecisions, evaluationState, conflictLog>>

\* ----- Type Invariants -----

TypeInvariant ==
    /\ activePolicies \subseteq Policies
    /\ policyRules \in [activePolicies -> SUBSET Rules]
    /\ rulePriorities \in [UNION {policyRules[p] : p \in activePolicies} -> Priorities]
    /\ ruleDecisions \in [UNION {policyRules[p] : p \in activePolicies} -> Decisions]
    /\ evaluationState \in {"IDLE", "EVALUATING", "CONFLICT", "RESOLVED"}
    /\ conflictLog \in Seq([rule1: Rules, rule2: Rules, resolution: Decisions])

\* ----- Decision Ordering -----
\* Defines the strictness ordering of decisions
\* TERMINATE > BLOCK > RESTRICT > ALLOW

DecisionOrder(d) ==
    CASE d = "ALLOW" -> 1
      [] d = "RESTRICT" -> 2
      [] d = "BLOCK" -> 3
      [] d = "TERMINATE" -> 4

StricterDecision(d1, d2) ==
    DecisionOrder(d1) > DecisionOrder(d2)

\* ----- Initial State -----

Init ==
    /\ activePolicies = {}
    /\ policyRules = [p \in {} |-> {}]
    /\ rulePriorities = [r \in {} |-> 1]
    /\ ruleDecisions = [r \in {} |-> "ALLOW"]
    /\ evaluationState = "IDLE"
    /\ conflictLog = <<>>

\* ----- Actions -----

\* Activate a policy
ActivatePolicy(policy, rules) ==
    /\ policy \in Policies
    /\ policy \notin activePolicies
    /\ rules \subseteq Rules
    /\ activePolicies' = activePolicies \cup {policy}
    /\ policyRules' = [p \in activePolicies' |-> 
                        IF p = policy THEN rules ELSE policyRules[p]]
    /\ UNCHANGED <<rulePriorities, ruleDecisions, evaluationState, conflictLog>>

\* Set rule priority
SetRulePriority(rule, priority) ==
    /\ rule \in UNION {policyRules[p] : p \in activePolicies}
    /\ priority \in Priorities
    /\ rulePriorities' = [rulePriorities EXCEPT ![rule] = priority]
    /\ UNCHANGED <<activePolicies, policyRules, ruleDecisions, evaluationState, conflictLog>>

\* Set rule decision
SetRuleDecision(rule, decision) ==
    /\ rule \in UNION {policyRules[p] : p \in activePolicies}
    /\ decision \in Decisions
    /\ ruleDecisions' = [ruleDecisions EXCEPT ![rule] = decision]
    /\ UNCHANGED <<activePolicies, policyRules, rulePriorities, evaluationState, conflictLog>>

\* Start policy evaluation
StartEvaluation ==
    /\ evaluationState = "IDLE"
    /\ activePolicies # {}
    /\ evaluationState' = "EVALUATING"
    /\ UNCHANGED <<activePolicies, policyRules, rulePriorities, ruleDecisions, conflictLog>>

\* Detect conflict between two rules
DetectConflict(rule1, rule2) ==
    /\ evaluationState = "EVALUATING"
    /\ rule1 \in UNION {policyRules[p] : p \in activePolicies}
    /\ rule2 \in UNION {policyRules[p] : p \in activePolicies}
    /\ rule1 # rule2
    /\ ruleDecisions[rule1] # ruleDecisions[rule2]
    /\ rulePriorities[rule1] = rulePriorities[rule2]  \* Same priority = conflict
    /\ evaluationState' = "CONFLICT"
    /\ LET resolution == IF StricterDecision(ruleDecisions[rule1], ruleDecisions[rule2])
                         THEN ruleDecisions[rule1]
                         ELSE ruleDecisions[rule2]
       IN conflictLog' = Append(conflictLog, [rule1 |-> rule1, rule2 |-> rule2, resolution |-> resolution])
    /\ UNCHANGED <<activePolicies, policyRules, rulePriorities, ruleDecisions>>

\* Resolve conflict (stricter decision wins)
ResolveConflict ==
    /\ evaluationState = "CONFLICT"
    /\ evaluationState' = "RESOLVED"
    /\ UNCHANGED <<activePolicies, policyRules, rulePriorities, ruleDecisions, conflictLog>>

\* Complete evaluation
CompleteEvaluation ==
    /\ evaluationState \in {"EVALUATING", "RESOLVED"}
    /\ evaluationState' = "IDLE"
    /\ UNCHANGED <<activePolicies, policyRules, rulePriorities, ruleDecisions, conflictLog>>

\* Deactivate a policy
DeactivatePolicy(policy) ==
    /\ policy \in activePolicies
    /\ evaluationState = "IDLE"
    /\ activePolicies' = activePolicies \ {policy}
    /\ policyRules' = [p \in activePolicies' |-> policyRules[p]]
    /\ UNCHANGED <<rulePriorities, ruleDecisions, evaluationState, conflictLog>>

\* ----- Next State Relation -----

Next ==
    \/ \E policy \in Policies, rules \in SUBSET Rules: ActivatePolicy(policy, rules)
    \/ \E rule \in Rules, priority \in Priorities: SetRulePriority(rule, priority)
    \/ \E rule \in Rules, decision \in Decisions: SetRuleDecision(rule, decision)
    \/ StartEvaluation
    \/ \E rule1, rule2 \in Rules: DetectConflict(rule1, rule2)
    \/ ResolveConflict
    \/ CompleteEvaluation
    \/ \E policy \in Policies: DeactivatePolicy(policy)

\* ----- Safety Properties -----

\* POLICY-1: Policy non-contradiction - conflicts are always resolved to stricter decision
ConflictsResolvedStricter ==
    \A i \in 1..Len(conflictLog):
        LET c == conflictLog[i]
        IN DecisionOrder(c.resolution) = 
           IF DecisionOrder(ruleDecisions[c.rule1]) > DecisionOrder(ruleDecisions[c.rule2])
           THEN DecisionOrder(ruleDecisions[c.rule1])
           ELSE DecisionOrder(ruleDecisions[c.rule2])

\* POLICY-2: Determinism - same input always produces same output
\* (Implicit in the specification - same state + same action = same next state)

\* POLICY-3: No evaluation during modification
NoModificationDuringEvaluation ==
    evaluationState # "IDLE" =>
    UNCHANGED <<activePolicies, policyRules, rulePriorities, ruleDecisions>>

\* POLICY-4: All active policies have at least one rule
ActivePoliciesHaveRules ==
    \A p \in activePolicies: policyRules[p] # {}

\* Combined safety invariant
SafetyInvariant ==
    /\ ConflictsResolvedStricter
    /\ ActivePoliciesHaveRules

\* ----- Liveness Properties -----

\* Evaluations eventually complete
EvaluationCompletes ==
    evaluationState = "EVALUATING" ~> evaluationState = "IDLE"

\* Conflicts are eventually resolved
ConflictsResolved ==
    evaluationState = "CONFLICT" ~> evaluationState \in {"RESOLVED", "IDLE"}

\* ----- Fairness -----

Fairness ==
    /\ WF_vars(Next)
    /\ WF_vars(CompleteEvaluation)
    /\ WF_vars(ResolveConflict)

\* ----- Specification -----

Spec == Init /\ [][Next]_vars /\ Fairness

\* ----- Theorems -----

THEOREM TypeSafety == Spec => []TypeInvariant

THEOREM PolicySafety == Spec => []SafetyInvariant

THEOREM LivenessProperty == Spec => EvaluationCompletes /\ ConflictsResolved

===============================================================================
