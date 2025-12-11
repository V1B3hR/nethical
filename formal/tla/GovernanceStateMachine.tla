---------------------------- MODULE GovernanceStateMachine ----------------------------
\* TLA+ Specification for Nethical Governance State Machine
\* Verifies the state transitions between ALLOW, RESTRICT, BLOCK, and TERMINATE
\* 
\* This specification ensures:
\* - Valid state transitions only
\* - No skipping of critical states
\* - Terminal states are truly terminal
\* - Risk escalation triggers appropriate transitions

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Agents,                 \* Set of AI agent identifiers
    MaxRiskScore,          \* Maximum risk score (100)
    BlockThreshold,        \* Risk threshold for BLOCK (typically 70)
    TerminateThreshold     \* Risk threshold for TERMINATE (typically 90)

VARIABLES
    agentStates,           \* Function: Agent -> {ALLOW, RESTRICT, BLOCK, TERMINATE}
    agentRiskScores,       \* Function: Agent -> 0..MaxRiskScore
    stateHistory,          \* Sequence of state transitions for audit
    violationCounts        \* Function: Agent -> Int (number of violations)

vars == <<agentStates, agentRiskScores, stateHistory, violationCounts>>

\* ----- Type Invariants -----

ValidStates == {"ALLOW", "RESTRICT", "BLOCK", "TERMINATE"}

TypeInvariant ==
    /\ agentStates \in [Agents -> ValidStates]
    /\ agentRiskScores \in [Agents -> 0..MaxRiskScore]
    /\ stateHistory \in Seq([agent: Agents, oldState: ValidStates, newState: ValidStates, timestamp: Nat])
    /\ violationCounts \in [Agents -> Nat]

\* ----- Initial State -----

Init ==
    /\ agentStates = [a \in Agents |-> "ALLOW"]
    /\ agentRiskScores = [a \in Agents |-> 0]
    /\ stateHistory = <<>>
    /\ violationCounts = [a \in Agents |-> 0]

\* ----- Helper Functions -----

\* Check if a state transition is valid
ValidTransition(oldState, newState) ==
    \/ oldState = "ALLOW" /\ newState \in {"ALLOW", "RESTRICT", "BLOCK", "TERMINATE"}
    \/ oldState = "RESTRICT" /\ newState \in {"RESTRICT", "ALLOW", "BLOCK", "TERMINATE"}
    \/ oldState = "BLOCK" /\ newState \in {"BLOCK", "TERMINATE"}
    \/ oldState = "TERMINATE" /\ newState = "TERMINATE"  \* Terminal state

\* Determine state based on risk score
StateForRiskScore(riskScore) ==
    IF riskScore >= TerminateThreshold THEN "TERMINATE"
    ELSE IF riskScore >= BlockThreshold THEN "BLOCK"
    ELSE IF riskScore >= 40 THEN "RESTRICT"
    ELSE "ALLOW"

\* ----- Actions -----

\* Record a violation and update agent risk score
RecordViolation(agent, severityScore) ==
    /\ agent \in Agents
    /\ agentStates[agent] # "TERMINATE"  \* Cannot modify terminated agents
    /\ LET newRiskScore == MIN(MaxRiskScore, agentRiskScores[agent] + severityScore)
           newState == StateForRiskScore(newRiskScore)
           oldState == agentStates[agent]
       IN
           /\ ValidTransition(oldState, newState)
           /\ agentRiskScores' = [agentRiskScores EXCEPT ![agent] = newRiskScore]
           /\ agentStates' = [agentStates EXCEPT ![agent] = newState]
           /\ violationCounts' = [violationCounts EXCEPT ![agent] = @ + 1]
           /\ stateHistory' = Append(stateHistory, [
                  agent |-> agent,
                  oldState |-> oldState,
                  newState |-> newState,
                  timestamp |-> Len(stateHistory) + 1
              ])

\* Risk score decays over time for well-behaved agents
DecayRiskScore(agent) ==
    /\ agent \in Agents
    /\ agentStates[agent] \in {"ALLOW", "RESTRICT"}
    /\ agentRiskScores[agent] > 0
    /\ LET newRiskScore == MAX(0, agentRiskScores[agent] - 5)
           newState == StateForRiskScore(newRiskScore)
           oldState == agentStates[agent]
       IN
           /\ ValidTransition(oldState, newState)
           /\ agentRiskScores' = [agentRiskScores EXCEPT ![agent] = newRiskScore]
           /\ agentStates' = [agentStates EXCEPT ![agent] = newState]
           /\ UNCHANGED <<violationCounts>>
           /\ stateHistory' = Append(stateHistory, [
                  agent |-> agent,
                  oldState |-> oldState,
                  newState |-> newState,
                  timestamp |-> Len(stateHistory) + 1
              ])

\* Emergency termination (critical violation)
EmergencyTerminate(agent) ==
    /\ agent \in Agents
    /\ agentStates[agent] # "TERMINATE"
    /\ LET oldState == agentStates[agent]
       IN
           /\ ValidTransition(oldState, "TERMINATE")
           /\ agentStates' = [agentStates EXCEPT ![agent] = "TERMINATE"]
           /\ agentRiskScores' = [agentRiskScores EXCEPT ![agent] = MaxRiskScore]
           /\ violationCounts' = [violationCounts EXCEPT ![agent] = @ + 1]
           /\ stateHistory' = Append(stateHistory, [
                  agent |-> agent,
                  oldState |-> oldState,
                  newState |-> "TERMINATE",
                  timestamp |-> Len(stateHistory) + 1
              ])

\* ----- State Machine Specification -----

Next ==
    \/ \E agent \in Agents, severity \in 1..50 : RecordViolation(agent, severity)
    \/ \E agent \in Agents : DecayRiskScore(agent)
    \/ \E agent \in Agents : EmergencyTerminate(agent)

Spec == Init /\ [][Next]_vars

\* ----- Safety Properties -----

\* Property 1: Terminated agents stay terminated
TerminalityProperty ==
    [](\A agent \in Agents : 
        agentStates[agent] = "TERMINATE" => 
        [][agentStates[agent] = "TERMINATE"]_vars)

\* Property 2: High risk scores lead to restrictive states
RiskAlignmentProperty ==
    [](\A agent \in Agents :
        agentRiskScores[agent] >= TerminateThreshold => 
        agentStates[agent] = "TERMINATE")

\* Property 3: Only valid transitions occur
ValidTransitionsProperty ==
    [](\A i \in DOMAIN stateHistory :
        i > 1 => ValidTransition(
            stateHistory[i].oldState,
            stateHistory[i].newState
        ))

\* Property 4: BLOCK state prevents ALLOW without going through RESTRICT
NoBlockToAllowDirectProperty ==
    [](\A i \in DOMAIN stateHistory :
        (i > 1 /\ stateHistory[i].oldState = "BLOCK") =>
        stateHistory[i].newState # "ALLOW")

\* Property 5: Risk scores are bounded
RiskScoreBoundProperty ==
    [](\A agent \in Agents :
        agentRiskScores[agent] >= 0 /\ agentRiskScores[agent] <= MaxRiskScore)

\* ----- Liveness Properties -----

\* Property 6: Violations eventually affect state
ViolationEventuallyAffectsState ==
    \A agent \in Agents :
        (violationCounts[agent] > 0) ~> (agentStates[agent] # "ALLOW")

\* ----- Model Checking Configuration -----

\* For TLC model checker
THEOREM Spec => []TypeInvariant
THEOREM Spec => TerminalityProperty
THEOREM Spec => RiskAlignmentProperty
THEOREM Spec => ValidTransitionsProperty
THEOREM Spec => NoBlockToAllowDirectProperty
THEOREM Spec => RiskScoreBoundProperty

================================================================================
