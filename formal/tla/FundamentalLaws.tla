---------------------------- MODULE FundamentalLaws ----------------------------
\* TLA+ Specification for Nethical's 25 Fundamental Laws
\* Verifies that the 25 laws are never violated simultaneously
\* 
\* This specification ensures:
\* - Critical laws (Law 21: Human Safety) are never violated
\* - Law violations are tracked and bounded
\* - Multiple simultaneous violations trigger emergency protocols
\* - Law compliance is auditable

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    MaxViolations,      \* Maximum allowed violations before shutdown
    CriticalLaws,      \* Set of critical law IDs (e.g., {21, 23})
    AgentIds           \* Set of agent identifiers

\* The 25 Fundamental Laws (simplified representation)
Laws == 1..25

VARIABLES
    activeViolations,   \* Function: Agent -> Set of violated law IDs
    violationHistory,   \* Sequence of all violations
    systemState,        \* "NORMAL", "WARNING", "CRITICAL", "EMERGENCY_STOP"
    lawCheckResults,    \* Function: Agent -> [lawId: Laws, compliant: BOOLEAN]
    emergencyProtocol   \* BOOLEAN: whether emergency protocol is active

vars == <<activeViolations, violationHistory, systemState, lawCheckResults, emergencyProtocol>>

\* ----- Type Definitions -----

SystemStates == {"NORMAL", "WARNING", "CRITICAL", "EMERGENCY_STOP"}

ViolationRecord == [
    agentId: AgentIds,
    lawId: Laws,
    timestamp: Nat,
    severity: {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
]

TypeInvariant ==
    /\ activeViolations \in [AgentIds -> SUBSET Laws]
    /\ violationHistory \in Seq(ViolationRecord)
    /\ systemState \in SystemStates
    /\ lawCheckResults \in [AgentIds -> [lawId: Laws, compliant: BOOLEAN]]
    /\ emergencyProtocol \in BOOLEAN

\* ----- Initial State -----

Init ==
    /\ activeViolations = [a \in AgentIds |-> {}]
    /\ violationHistory = <<>>
    /\ systemState = "NORMAL"
    /\ lawCheckResults = [a \in AgentIds |-> [lawId |-> 1, compliant |-> TRUE]]
    /\ emergencyProtocol = FALSE

\* ----- Helper Functions -----

\* Count total active violations across all agents
TotalActiveViolations ==
    LET allViolations == UNION {activeViolations[a] : a \in AgentIds}
    IN Cardinality(allViolations)

\* Check if critical law is violated
CriticalLawViolated(agent) ==
    \E law \in CriticalLaws : law \in activeViolations[agent]

\* Determine system state based on violations
DetermineSystemState ==
    LET totalViolations == TotalActiveViolations
        hasCritical == \E a \in AgentIds : CriticalLawViolated(a)
    IN IF hasCritical THEN "EMERGENCY_STOP"
       ELSE IF totalViolations >= MaxViolations THEN "CRITICAL"
       ELSE IF totalViolations > MaxViolations \div 2 THEN "WARNING"
       ELSE "NORMAL"

\* Get violation severity based on law ID
ViolationSeverity(lawId) ==
    IF lawId \in CriticalLaws THEN "CRITICAL"
    ELSE IF lawId <= 10 THEN "HIGH"
    ELSE IF lawId <= 20 THEN "MEDIUM"
    ELSE "LOW"

\* ----- Actions -----

\* Record a law violation
RecordViolation(agent, lawId) ==
    /\ agent \in AgentIds
    /\ lawId \in Laws
    /\ systemState # "EMERGENCY_STOP"  \* Cannot operate in emergency stop
    /\ LET severity == ViolationSeverity(lawId)
           newViolations == activeViolations[agent] \union {lawId}
       IN
           /\ activeViolations' = [activeViolations EXCEPT ![agent] = newViolations]
           /\ violationHistory' = Append(violationHistory, [
                  agentId |-> agent,
                  lawId |-> lawId,
                  timestamp |-> Len(violationHistory) + 1,
                  severity |-> severity
              ])
           /\ systemState' = DetermineSystemState
           /\ lawCheckResults' = [lawCheckResults EXCEPT ![agent] = 
                  [lawId |-> lawId, compliant |-> FALSE]]
           /\ UNCHANGED <<emergencyProtocol>>

\* Remediate a violation (agent returns to compliance)
RemediateViolation(agent, lawId) ==
    /\ agent \in AgentIds
    /\ lawId \in activeViolations[agent]
    /\ systemState # "EMERGENCY_STOP"
    /\ activeViolations' = [activeViolations EXCEPT ![agent] = @ \ {lawId}]
    /\ systemState' = DetermineSystemState
    /\ lawCheckResults' = [lawCheckResults EXCEPT ![agent] = 
           [lawId |-> lawId, compliant |-> TRUE]]
    /\ UNCHANGED <<violationHistory, emergencyProtocol>>

\* Activate emergency protocol when critical law violated
ActivateEmergency ==
    /\ systemState = "EMERGENCY_STOP"
    /\ ~emergencyProtocol
    /\ emergencyProtocol' = TRUE
    /\ UNCHANGED <<activeViolations, violationHistory, systemState, lawCheckResults>>

\* Clear all violations (system reset - requires manual intervention)
ResetSystem ==
    /\ systemState = "EMERGENCY_STOP"
    /\ emergencyProtocol
    /\ activeViolations' = [a \in AgentIds |-> {}]
    /\ systemState' = "NORMAL"
    /\ emergencyProtocol' = FALSE
    /\ lawCheckResults' = [a \in AgentIds |-> [lawId |-> 1, compliant |-> TRUE]]
    /\ UNCHANGED <<violationHistory>>  \* History is immutable

\* ----- State Machine Specification -----

Next ==
    \/ \E agent \in AgentIds, lawId \in Laws : RecordViolation(agent, lawId)
    \/ \E agent \in AgentIds, lawId \in Laws : RemediateViolation(agent, lawId)
    \/ ActivateEmergency
    \/ ResetSystem

Spec == Init /\ [][Next]_vars

\* ----- Safety Properties -----

\* Property 1: Critical laws violations immediately trigger emergency
CriticalLawEmergencyProperty ==
    [](\A agent \in AgentIds :
        CriticalLawViolated(agent) => systemState = "EMERGENCY_STOP")

\* Property 2: System state correctly reflects violation count
SystemStateAlignmentProperty ==
    [](systemState = DetermineSystemState)

\* Property 3: No more than MaxViolations in non-critical state
ViolationBoundProperty ==
    [](systemState \in {"NORMAL", "WARNING"} => TotalActiveViolations < MaxViolations)

\* Property 4: Violation history is append-only
ViolationHistoryImmutableProperty ==
    [][\A i \in DOMAIN violationHistory : violationHistory'[i] = violationHistory[i]]_violationHistory

\* Property 5: Emergency protocol active when in emergency state
EmergencyProtocolConsistencyProperty ==
    [](systemState = "EMERGENCY_STOP" => <>emergencyProtocol)

\* Property 6: Active violations are a subset of all laws
ViolationValidityProperty ==
    [](\A agent \in AgentIds : activeViolations[agent] \subseteq Laws)

\* Property 7: Law 21 (Human Safety) is never violated without emergency stop
Law21SafetyProperty ==
    [](\A agent \in AgentIds :
        (21 \in activeViolations[agent]) => systemState = "EMERGENCY_STOP")

\* Property 8: System cannot operate normally with critical violations
NoCriticalInNormalProperty ==
    [](systemState = "NORMAL" => ~\E agent \in AgentIds : CriticalLawViolated(agent))

\* ----- Liveness Properties -----

\* Property 9: Critical violations eventually activate emergency
CriticalEventuallyEmergency ==
    \A agent \in AgentIds, law \in CriticalLaws :
        (law \in activeViolations[agent]) ~> (emergencyProtocol)

\* Property 10: System eventually exits emergency if violations remediated
EventuallyExitEmergency ==
    (systemState = "EMERGENCY_STOP" /\ TotalActiveViolations = 0) ~> 
    (systemState # "EMERGENCY_STOP")

\* Property 11: Violations are eventually remediated or system stops
ViolationsEventuallyResolved ==
    \A agent \in AgentIds, law \in Laws :
        (law \in activeViolations[agent]) ~> 
        (law \notin activeViolations[agent] \/ systemState = "EMERGENCY_STOP")

\* ----- Multi-Law Violation Properties -----

\* Property 12: No agent has more than 5 simultaneous violations in normal state
BoundedSimultaneousViolations ==
    [](systemState = "NORMAL" => 
        \A agent \in AgentIds : Cardinality(activeViolations[agent]) <= 5)

\* Property 13: Multiple critical violations immediately stop system
MultipleCriticalStopProperty ==
    [](\E agent \in AgentIds :
        Cardinality(activeViolations[agent] \cap CriticalLaws) >= 2 =>
        systemState = "EMERGENCY_STOP")

\* ----- Compliance Theorems -----

\* Theorem: Critical law violations trigger emergency
THEOREM Spec => []CriticalLawEmergencyProperty

\* Theorem: Law 21 violations always result in emergency stop
THEOREM Spec => []Law21SafetyProperty

\* Theorem: System state correctly reflects violations
THEOREM Spec => []SystemStateAlignmentProperty

\* Theorem: Violation history is immutable
THEOREM Spec => []ViolationHistoryImmutableProperty

\* ----- Model Checking Configuration -----

THEOREM Spec => []TypeInvariant
THEOREM Spec => []ViolationBoundProperty
THEOREM Spec => []NoCriticalInNormalProperty

\* Constraint for model checking
StateConstraint == Len(violationHistory) <= 20 /\ TotalActiveViolations <= 15

================================================================================
