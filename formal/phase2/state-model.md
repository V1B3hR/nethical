# State Model

## Overview
This document defines the state spaces and state machines for key entities in the Nethical governance platform. Clear state modeling is essential for deterministic behavior (P-DET) and formal verification (Phase 3).

---

## State Spaces

### Policy State Machine

```
                    ┌──────────────┐
                    │   INACTIVE   │ (Initial state: policy defined but not loaded)
                    └──────┬───────┘
                           │ load() [validation passes]
                           ▼
                    ┌──────────────┐
                    │  QUARANTINE  │ (Loaded but not enforced; testing mode)
                    └──────┬───────┘
                           │ activate() [multi-sig approval if required]
                           ▼
                    ┌──────────────┐
                    │    ACTIVE    │ (Enforced in policy evaluation)
                    └──────┬───────┘
                           │ deactivate() or supersede()
                           ▼
                    ┌──────────────┐
                    │   INACTIVE   │ (Deactivated; superseded by new version)
                    └──────────────┘
```

**States**:
- **INACTIVE**: Policy exists in repository but not loaded into runtime
- **QUARANTINE**: Policy loaded and validated, but decisions logged as "would-have" only; not enforced
- **ACTIVE**: Policy enforced in production evaluation
- **INACTIVE** (after deactivation): Policy no longer enforced; historical only

**Invariants**:
- Only one version of a policy_id can be ACTIVE at a time (lineage property)
- Transition from QUARANTINE to ACTIVE requires multi-sig approval if policy is critical
- Once ACTIVE, a policy version cannot be modified (immutability; only deactivated)
- Deactivation is append-only (policy_lineage chain records deactivation event)

---

### Decision State Machine

```
                    ┌──────────────┐
                    │   PENDING    │ (Action received; evaluation queued)
                    └──────┬───────┘
                           │ evaluate()
                           ▼
                    ┌──────────────┐
                    │  EVALUATING  │ (Policy evaluation in progress)
                    └──────┬───────┘
                           │ judge()
                           ▼
          ┌─────────────────┴─────────────────┐
          │                                    │
          ▼                                    ▼
   ┌─────────────┐                      ┌─────────────┐
   │   DECIDED   │ (ALLOW/RESTRICT)     │   BLOCKED   │ (BLOCK/TERMINATE)
   └──────┬──────┘                      └──────┬──────┘
          │                                    │
          │ [auto-escalate trigger]            │ [auto-escalate trigger]
          └────────────────┬───────────────────┘
                           ▼
                    ┌──────────────┐
                    │  ESCALATED   │ (Sent to human review queue)
                    └──────┬───────┘
                           │ human_review()
                           ▼
                    ┌──────────────┐
                    │  REVIEWED    │ (Human feedback recorded)
                    └──────┬───────┘
                           │ [optional: appeal filed]
                           ▼
                    ┌──────────────┐
                    │   APPEALED   │ (Contested; re-evaluation requested)
                    └──────┬───────┘
                           │ re-evaluate()
                           ▼
                    ┌──────────────┐
                    │  RE-DECIDED  │ (Appeal outcome determined)
                    └──────────────┘
```

**States**:
- **PENDING**: Action received; queued for evaluation
- **EVALUATING**: Policy evaluation in progress; risk scores being computed
- **DECIDED**: Final judgment made (ALLOW or RESTRICT); action may proceed with conditions
- **BLOCKED**: Final judgment made (BLOCK or TERMINATE); action prevented
- **ESCALATED**: Sent to human review queue (high risk, low confidence, or policy-triggered)
- **REVIEWED**: Human reviewer provided feedback or override
- **APPEALED**: Data subject or agent operator filed appeal
- **RE-DECIDED**: Appeal re-evaluation completed

**Invariants**:
- All decisions eventually reach terminal state (DECIDED, BLOCKED, REVIEWED, or RE-DECIDED)
- Escalation SLA: Median time from ESCALATED to REVIEWED < 72h (R-014)
- Appeal re-evaluation must be deterministic (P-APPEAL): same context → same outcome

---

### Audit Event State Machine

```
                    ┌──────────────┐
                    │   BUFFERED   │ (Event created; in memory buffer)
                    └──────┬───────┘
                           │ flush()
                           ▼
                    ┌──────────────┐
                    │   PERSISTED  │ (Written to append-only log)
                    └──────┬───────┘
                           │ merkle_anchor()
                           ▼
                    ┌──────────────┐
                    │   ANCHORED   │ (Included in Merkle tree; root hash computed)
                    └──────┬───────┘
                           │ [optional: external timestamping]
                           ▼
                    ┌──────────────┐
                    │ TIMESTAMPED  │ (External timestamp proof received)
                    └──────────────┘
```

**States**:
- **BUFFERED**: Event logged in memory; not yet durable
- **PERSISTED**: Written to storage; durable but not yet Merkle-anchored
- **ANCHORED**: Included in Merkle tree; tamper-evident
- **TIMESTAMPED**: External timestamping proof obtained (strongest non-repudiation)

**Invariants**:
- Events never transition backward (append-only; P-AUD)
- Merkle root hash computed periodically (e.g., every 1000 events or 60 seconds)
- Tampering detection: Any modification to PERSISTED or ANCHORED event invalidates Merkle path

---

### Agent State Machine

```
                    ┌──────────────┐
                    │  REGISTERED  │ (Agent enrolled; assigned agent_id)
                    └──────┬───────┘
                           │ first_action()
                           ▼
                    ┌──────────────┐
                    │    ACTIVE    │ (Agent performing actions; monitored)
                    └──────┬───────┘
                           │ [violation threshold exceeded or TERMINATE judgment]
                           ▼
                    ┌──────────────┐
                    │  SUSPENDED   │ (Agent temporarily blocked; under review)
                    └──────┬───────┘
                           │ review_complete() [reinstate or terminate]
                           ├──────────────┐
                           ▼              ▼
                    ┌─────────────┐ ┌─────────────┐
                    │   ACTIVE    │ │ TERMINATED  │ (Agent permanently disabled)
                    └─────────────┘ └─────────────┘
```

**States**:
- **REGISTERED**: Agent enrolled in system; metadata recorded
- **ACTIVE**: Agent operational; actions monitored and evaluated
- **SUSPENDED**: Agent temporarily blocked due to violations or critical judgment
- **TERMINATED**: Agent permanently disabled; no further actions allowed

**Invariants**:
- TERMINATED is terminal state (no resurrection without re-registration as new agent)
- Suspension SLA: Review completed within 24h for high-priority agents

---

### Fairness Metric State Machine

```
                    ┌──────────────┐
                    │   PENDING    │ (Metric computation scheduled)
                    └──────┬───────┘
                           │ compute()
                           ▼
                    ┌──────────────┐
                    │  COMPUTING   │ (Batch processing in progress)
                    └──────┬───────┘
                           │ complete()
                           ▼
          ┌─────────────────┴─────────────────┐
          │                                    │
          ▼                                    ▼
   ┌─────────────┐                      ┌─────────────┐
   │ COMPLIANT   │ (Within threshold)   │ NON-COMPLIANT│ (Threshold breached)
   └─────────────┘                      └──────┬──────┘
                                               │ alert()
                                               ▼
                                        ┌──────────────┐
                                        │  ESCALATED   │ (Governance review triggered)
                                        └──────┬───────┘
                                               │ recalibrate() or accept_risk()
                                               ▼
                                        ┌──────────────┐
                                        │  RESOLVED    │ (Action taken)
                                        └──────────────┘
```

**States**:
- **PENDING**: Metric computation scheduled (e.g., monthly batch job)
- **COMPUTING**: Statistical analysis in progress
- **COMPLIANT**: Metric within acceptable threshold (e.g., SP diff ≤ 0.10)
- **NON-COMPLIANT**: Threshold breached; requires governance action
- **ESCALATED**: Governance team notified; review in progress
- **RESOLVED**: Recalibration or risk acceptance documented

**Invariants**:
- Metrics recomputed on fixed schedule (monthly) or trigger (significant data volume change)
- Non-compliance requires documented resolution (no silent acceptance)

---

## State Transition Guards & Actions

### Policy: QUARANTINE → ACTIVE

**Guards** (preconditions):
- Validation passed (no cycles, schema valid, dependencies resolved)
- Multi-sig approval obtained if policy criticality ≥ HIGH
- No conflicting ACTIVE policy version for same policy_id

**Actions**:
- Update policy status to ACTIVE
- Deactivate previous version (if any) for same policy_id
- Append policy activation event to audit log
- Notify operators of activation

---

### Decision: EVALUATING → DECIDED / BLOCKED

**Guards**:
- All policies evaluated (no pending dependencies)
- Risk score computed
- Judgment logic executed

**Actions**:
- Set final judgment (ALLOW, RESTRICT, BLOCK, TERMINATE)
- Generate justification with contributing factors
- Compute confidence score
- Append decision event to audit log
- [If auto-escalate trigger met] Transition to ESCALATED state

**Auto-Escalate Triggers**:
- Risk score > escalation_threshold (e.g., 0.9)
- Confidence < confidence_threshold (e.g., 0.6)
- Judgment = BLOCK or TERMINATE
- Policy flag: always_escalate = true

---

### Audit Event: PERSISTED → ANCHORED

**Guards**:
- Batch size reached (e.g., 1000 events) OR time elapsed (e.g., 60 seconds)
- All events in batch persisted

**Actions**:
- Construct Merkle tree over event batch
- Compute Merkle root hash
- Store root hash with timestamp and batch metadata
- Update event records with Merkle path
- [Optional] Submit root hash to external timestamping service

---

### Agent: ACTIVE → SUSPENDED

**Guards**:
- Violation count in time window > threshold (e.g., 10 BLOCK judgments in 1 hour)
- Single TERMINATE judgment received
- Manual suspension by admin

**Actions**:
- Set agent status to SUSPENDED
- Block all future actions from agent
- Append suspension event to audit log
- Notify agent operator and governance team
- Create review ticket with SLA

---

### Fairness Metric: NON-COMPLIANT → ESCALATED

**Guards**:
- Metric value outside acceptable threshold
- Sample size sufficient for statistical significance

**Actions**:
- Append fairness alert event to audit log
- Notify governance team (email, dashboard, incident ticket)
- Create escalation ticket with SLA (e.g., resolution within 7 days)
- Flag affected policies for review

---

## Composite States & Parallel Regions

### Multi-Tenant State Isolation

Each tenant has an independent state space for:
- Active policies (tenant-specific policy sets)
- Agent states (tenant agents do not affect each other)
- Fairness metrics (computed per tenant)

**Invariant (P-TENANT-ISO)**: State transitions in tenant A cannot affect tenant B's state.

---

### Concurrent Evaluation Pipelines

Multiple actions can be evaluated concurrently (subject to thread safety and resource limits):
- Each action has independent PENDING → EVALUATING → DECIDED state progression
- Shared state (policies, agent history) accessed read-only or with locks
- Audit log writes coordinated via database transactions

---

## State Persistence & Recovery

### Persistent State
- **Policies**: Database with versioned schema (ACID transactions)
- **Decisions**: Audit log (append-only; eventually consistent acceptable)
- **Agent metadata**: Database (ACID transactions for status changes)
- **Fairness metrics**: Database (batch updates; eventual consistency)

### Ephemeral State
- In-memory buffers (audit events, action queues)
- Caches (policies, agent history)
- Transient evaluation state (risk scores, intermediate results)

### Recovery Scenarios
- **Crash during evaluation**: Pending actions re-queued; idempotent evaluation ensures determinism
- **Audit log flush failure**: Buffered events retained; retry until success or alert
- **Policy activation interrupted**: ACID transaction ensures atomicity (either fully activated or rolled back)

---

## State Invariants Summary

| Invariant | Description | Enforcement |
|-----------|-------------|-------------|
| **I-POL-01** | At most one ACTIVE version per policy_id | Database unique constraint + validation |
| **I-POL-02** | Policy versions immutable once ACTIVE | Append-only storage; no UPDATE operations |
| **I-DEC-01** | All decisions eventually reach terminal state | Timeout handling + SLA monitoring |
| **I-DEC-02** | Deterministic re-evaluation (P-APPEAL) | Stored context + topological policy sort |
| **I-AUD-01** | Audit events append-only (P-AUD) | Storage layer constraint |
| **I-AUD-02** | Merkle anchoring detects tampering | Daily verification runs |
| **I-AGT-01** | TERMINATED agents remain terminal | Status change validation |
| **I-FAIR-01** | Non-compliant fairness requires resolution | Escalation workflow enforced |
| **I-TENANT-01** | Tenant state isolated (P-TENANT-ISO) | Database row-level security + app logic |

---

## Formal Verification (Phase 3A Preview)

Phase 3A will formalize these state machines in TLA+ and prove:
- **Safety**: Invalid state transitions impossible (e.g., QUARANTINE → ACTIVE without approval)
- **Liveness**: All decisions eventually reach terminal state (no infinite loops)
- **Invariants**: I-POL-01, I-AUD-01, etc. hold in all reachable states

---

## Related Documents
- overview.md: High-level system architecture
- transitions.md: Detailed transition specifications and algorithms
- api-contracts.md: API operations that trigger state transitions
- policy_lineage.md: Policy version chain and lineage integrity
- requirements.md: Requirements mapped to state invariants

---

**Status**: ✅ Phase 2A Deliverable - COMPLETE  
**Last Updated**: 2025-11-16  
**Owner**: Tech Lead / Formal Methods Engineer
