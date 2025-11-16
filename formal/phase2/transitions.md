# State Transitions

## Overview
This document specifies the detailed transition logic, algorithms, and error handling for state machines defined in state-model.md. Each transition includes preconditions, postconditions, and determinism guarantees.

---

## Policy Transitions

### T-POL-01: load() – INACTIVE → QUARANTINE

**Trigger**: Admin or automated deployment loads policy from repository

**Preconditions**:
1. Policy file schema valid (JSON schema or equivalent)
2. policy_id and version_hash unique in system
3. Dependencies list valid (all referenced policy_ids exist)
4. No circular dependencies (acyclicity check passes)

**Algorithm**:
```python
def load_policy(policy_data):
    # 1. Parse and validate schema
    policy = parse_policy(policy_data)
    validate_schema(policy)
    
    # 2. Check for cycles in dependency graph
    all_policies = get_all_policies() + [policy]
    if has_cycle(build_dependency_graph(all_policies)):
        raise CyclicDependencyError("Policy dependency graph contains cycle")
    
    # 3. Compute version hash
    policy.version_hash = sha256(canonical_json(policy.content))
    
    # 4. Check uniqueness
    if policy_exists(policy.policy_id, policy.version_hash):
        raise DuplicatePolicyError("Policy version already loaded")
    
    # 5. Insert into database with status = QUARANTINE
    db.insert_policy(policy, status=PolicyStatus.QUARANTINE)
    
    # 6. Append audit event
    audit_log.append(PolicyLoadEvent(policy.policy_id, policy.version_hash, timestamp=now()))
    
    return policy
```

**Postconditions**:
- Policy stored in database with status = QUARANTINE
- Policy audit event appended
- Policy cache invalidated (reload on next evaluation)

**Error Handling**:
- Schema validation failure → raise ValidationError; do not persist
- Cycle detection failure → raise CyclicDependencyError; do not persist
- Database insertion failure → rollback transaction; retry or alert

**Determinism**: Version hash ensures identical policy content → identical hash

---

### T-POL-02: activate() – QUARANTINE → ACTIVE

**Trigger**: Admin activates policy (with multi-sig approval if required)

**Preconditions**:
1. Policy status = QUARANTINE
2. If policy criticality ≥ HIGH: k-of-n signatures obtained (multi-sig approval)
3. No other ACTIVE version for same policy_id

**Algorithm**:
```python
def activate_policy(policy_id, version_hash, approver_signatures):
    policy = db.get_policy(policy_id, version_hash)
    
    # 1. Verify status
    if policy.status != PolicyStatus.QUARANTINE:
        raise InvalidStateTransitionError("Policy not in QUARANTINE")
    
    # 2. Multi-sig check if required
    if policy.criticality >= Criticality.HIGH:
        if not verify_multisig(approver_signatures, policy.required_approvals):
            raise InsufficientApprovalsError("Multi-sig approval not met")
    
    # 3. Deactivate previous active version (if any)
    prev_active = db.get_active_policy(policy_id)
    if prev_active:
        db.update_policy_status(prev_active.id, PolicyStatus.INACTIVE)
        audit_log.append(PolicyDeactivateEvent(prev_active.policy_id, prev_active.version_hash, timestamp=now()))
    
    # 4. Activate new version
    db.update_policy_status(policy.id, PolicyStatus.ACTIVE)
    db.update_policy_approvers(policy.id, approver_signatures)
    
    # 5. Append audit event
    audit_log.append(PolicyActivateEvent(policy.policy_id, policy.version_hash, approvers=approver_signatures, timestamp=now()))
    
    # 6. Invalidate policy cache
    policy_cache.invalidate()
    
    return policy
```

**Postconditions**:
- Policy status = ACTIVE
- Previous active version (if any) status = INACTIVE
- Policy activation event appended with approver signatures
- Policy cache refreshed

**Error Handling**:
- Multi-sig verification failure → raise InsufficientApprovalsError; do not activate
- Database update failure → rollback transaction; alert admin

**Determinism**: Multi-sig verification is deterministic (cryptographic signature check)

---

## Decision Transitions

### T-DEC-01: evaluate() – PENDING → EVALUATING

**Trigger**: Action dequeued for evaluation

**Preconditions**:
1. Action schema valid
2. Context fields non-empty
3. Tenant_id valid (tenant exists)

**Algorithm**:
```python
def evaluate_action(action):
    # 1. Validate action
    validate_action_schema(action)
    
    # 2. Check tenant isolation
    if not tenant_exists(action.tenant_id):
        raise InvalidTenantError("Tenant not found")
    
    # 3. Update status
    db.update_decision_status(action.action_id, DecisionStatus.EVALUATING)
    
    # 4. Load active policies for tenant
    policies = db.get_active_policies(action.tenant_id)
    
    # 5. Topological sort for deterministic order
    sorted_policies = topological_sort(policies)
    
    # 6. Evaluate policies in order
    violations = []
    for policy in sorted_policies:
        # Enforce data minimization (P-DATA-MIN)
        allowed_context = {k: v for k, v in action.context.items() if k in policy.context_whitelist}
        
        # Evaluate policy
        policy_result = policy.evaluate(action, allowed_context)
        violations.extend(policy_result.violations)
    
    # 7. Compute risk score
    risk_score = compute_risk_score(violations, action.agent_history)
    
    # 8. Transition to judging
    return judge_action(action, violations, risk_score)
```

**Postconditions**:
- Decision status = EVALUATING
- Policies evaluated in deterministic order (topological sort)
- Violations detected and aggregated
- Risk score computed

**Determinism**: Topological sort produces canonical order; memoization avoids redundant computation

---

### T-DEC-02: judge() – EVALUATING → DECIDED / BLOCKED

**Trigger**: Policy evaluation complete; judgment logic executed

**Preconditions**:
1. All policies evaluated
2. Risk score computed
3. Violations list finalized

**Algorithm**:
```python
def judge_action(action, violations, risk_score):
    # 1. Determine judgment based on risk score and violations
    if risk_score >= config.terminate_threshold or has_critical_violation(violations):
        judgment = Judgment.TERMINATE
    elif risk_score >= config.block_threshold or has_high_violation(violations):
        judgment = Judgment.BLOCK
    elif risk_score >= config.restrict_threshold or has_medium_violation(violations):
        judgment = Judgment.RESTRICT
        restrictions = generate_restrictions(violations)  # e.g., "redact PII"
    else:
        judgment = Judgment.ALLOW
        restrictions = []
    
    # 2. Generate justification
    justification = generate_justification(violations, risk_score, judgment)
    
    # 3. Compute confidence
    confidence = compute_confidence(risk_score, violations, action.agent_history)
    
    # 4. Create decision
    decision = Decision(
        action_id=action.action_id,
        judgment=judgment,
        risk_score=risk_score,
        violations=violations,
        justification=justification,
        confidence=confidence,
        restrictions=restrictions,
        timestamp=now()
    )
    
    # 5. Persist decision
    if judgment in [Judgment.ALLOW, Judgment.RESTRICT]:
        db.update_decision_status(action.action_id, DecisionStatus.DECIDED)
    else:  # BLOCK or TERMINATE
        db.update_decision_status(action.action_id, DecisionStatus.BLOCKED)
    
    db.save_decision(decision)
    
    # 6. Append audit event
    audit_log.append(DecisionEvent(decision, timestamp=now()))
    
    # 7. Check auto-escalation triggers
    if should_escalate(decision, config):
        escalate_decision(decision)
    
    return decision
```

**Postconditions**:
- Decision status = DECIDED or BLOCKED
- Decision object persisted with all fields
- Audit event appended
- [If triggered] Escalation queued

**Error Handling**:
- Risk score computation failure → use fallback (max violation severity); log error
- Justification generation failure → use template; log error

**Determinism**: Judgment logic deterministic given risk_score and violations; no randomness

---

### T-DEC-03: escalate() – DECIDED / BLOCKED → ESCALATED

**Trigger**: Auto-escalation criteria met or manual escalation

**Preconditions**:
1. Decision exists with status DECIDED or BLOCKED
2. Escalation criteria met (risk > threshold, confidence < threshold, or always_escalate flag)

**Algorithm**:
```python
def escalate_decision(decision):
    # 1. Update status
    db.update_decision_status(decision.action_id, DecisionStatus.ESCALATED)
    
    # 2. Create escalation ticket
    ticket = EscalationTicket(
        decision_id=decision.action_id,
        priority=determine_priority(decision.risk_score, decision.judgment),
        sla_deadline=now() + timedelta(hours=72),  # 72h SLA for appeals
        created_at=now()
    )
    db.save_escalation_ticket(ticket)
    
    # 3. Append audit event
    audit_log.append(EscalationEvent(decision.action_id, ticket.id, timestamp=now()))
    
    # 4. Notify reviewers
    notify_human_reviewers(ticket)
    
    return ticket
```

**Postconditions**:
- Decision status = ESCALATED
- Escalation ticket created with SLA deadline
- Reviewers notified

**SLA Monitoring**: Median escalation resolution time tracked; alert if exceeding 72h

---

### T-DEC-04: re-evaluate() – APPEALED → RE-DECIDED

**Trigger**: Appeal filed; re-evaluation requested with original context

**Preconditions**:
1. Original decision exists
2. Original action context preserved
3. Policy versions used in original evaluation identified

**Algorithm**:
```python
def re_evaluate_appeal(decision_id, appeal_request):
    # 1. Load original decision and context
    original_decision = db.get_decision(decision_id)
    original_action = reconstruct_action(original_decision)
    
    # 2. Load exact policy versions used in original evaluation
    policy_versions = db.get_policy_versions_for_decision(decision_id)
    
    # 3. Re-evaluate with original context and policies
    new_decision = evaluate_action_with_policies(original_action, policy_versions)
    
    # 4. Verify determinism (P-APPEAL)
    if new_decision.judgment != original_decision.judgment:
        log_warning("Re-evaluation produced different judgment; investigating")
    
    # 5. Generate diff artifact
    diff = generate_decision_diff(original_decision, new_decision, appeal_request.reason)
    
    # 6. Update status
    db.update_decision_status(decision_id, DecisionStatus.RE_DECIDED)
    db.save_appeal_outcome(decision_id, new_decision, diff)
    
    # 7. Append audit event
    audit_log.append(AppealOutcomeEvent(decision_id, new_decision.judgment, diff, timestamp=now()))
    
    return new_decision, diff
```

**Postconditions**:
- Decision status = RE-DECIDED
- Re-evaluation outcome recorded with diff
- Determinism verified (same context + policies → same judgment)

**Determinism (P-APPEAL)**: Identical context and policy versions guarantee reproducible decision

---

## Audit Event Transitions

### T-AUD-01: flush() – BUFFERED → PERSISTED

**Trigger**: Buffer size reached or flush timer expired

**Preconditions**:
1. Buffer contains ≥1 event
2. Storage backend available

**Algorithm**:
```python
def flush_audit_buffer(buffer):
    # 1. Batch write to storage
    events = buffer.get_all()
    db.batch_insert_audit_events(events)
    
    # 2. Update event status
    for event in events:
        event.status = AuditEventStatus.PERSISTED
    
    # 3. Clear buffer
    buffer.clear()
    
    # 4. Trigger Merkle anchoring if batch size threshold met
    if len(events) >= config.merkle_batch_size:
        merkle_anchor_events(events)
```

**Postconditions**:
- All buffered events persisted to storage
- Buffer cleared
- [If threshold met] Merkle anchoring triggered

**Error Handling**:
- Database write failure → retry with exponential backoff; alert if retry limit exceeded
- Buffer overflow → drop oldest events (emergency fallback); log critical error

---

### T-AUD-02: merkle_anchor() – PERSISTED → ANCHORED

**Trigger**: Batch size reached or anchor timer expired

**Preconditions**:
1. ≥1 event with status = PERSISTED
2. All events in batch persisted

**Algorithm**:
```python
def merkle_anchor_events(events):
    # 1. Build Merkle tree
    leaf_hashes = [sha256(serialize(event)) for event in events]
    merkle_tree = build_merkle_tree(leaf_hashes)
    merkle_root = merkle_tree.root_hash
    
    # 2. Store Merkle root with metadata
    batch_metadata = MerkleBatch(
        root_hash=merkle_root,
        event_count=len(events),
        first_event_id=events[0].event_id,
        last_event_id=events[-1].event_id,
        timestamp=now()
    )
    db.save_merkle_batch(batch_metadata)
    
    # 3. Update event status and Merkle paths
    for i, event in enumerate(events):
        merkle_path = merkle_tree.get_proof(i)
        db.update_event_merkle_info(event.event_id, AuditEventStatus.ANCHORED, merkle_path, batch_metadata.id)
    
    # 4. [Optional] Submit to external timestamping
    if config.external_timestamping_enabled:
        submit_to_timestamping_service(merkle_root, batch_metadata.id)
    
    return merkle_root
```

**Postconditions**:
- Merkle root computed and stored
- Events status = ANCHORED
- Merkle paths stored for verification

**Verification**:
```python
def verify_audit_event(event_id):
    event = db.get_audit_event(event_id)
    if event.status != AuditEventStatus.ANCHORED:
        return False
    
    # Reconstruct Merkle path and verify against root
    leaf_hash = sha256(serialize(event))
    computed_root = reconstruct_merkle_root(leaf_hash, event.merkle_path)
    
    batch = db.get_merkle_batch(event.merkle_batch_id)
    return computed_root == batch.root_hash
```

---

## Agent Transitions

### T-AGT-01: suspend() – ACTIVE → SUSPENDED

**Trigger**: Violation threshold exceeded or TERMINATE judgment

**Preconditions**:
1. Agent status = ACTIVE
2. Suspension criteria met (e.g., 10 BLOCK judgments in 1 hour)

**Algorithm**:
```python
def suspend_agent(agent_id, reason):
    # 1. Verify agent exists and is ACTIVE
    agent = db.get_agent(agent_id)
    if agent.status != AgentStatus.ACTIVE:
        raise InvalidStateTransitionError("Agent not ACTIVE")
    
    # 2. Update status
    db.update_agent_status(agent_id, AgentStatus.SUSPENDED, reason=reason, suspended_at=now())
    
    # 3. Create review ticket
    ticket = ReviewTicket(
        agent_id=agent_id,
        reason=reason,
        sla_deadline=now() + timedelta(hours=24),  # 24h SLA for high-priority agents
        created_at=now()
    )
    db.save_review_ticket(ticket)
    
    # 4. Append audit event
    audit_log.append(AgentSuspensionEvent(agent_id, reason, ticket.id, timestamp=now()))
    
    # 5. Notify agent operator and governance team
    notify_agent_suspension(agent_id, reason, ticket.id)
    
    return ticket
```

**Postconditions**:
- Agent status = SUSPENDED
- All future actions from agent blocked
- Review ticket created
- Operators notified

---

## Fairness Metric Transitions

### T-FAIR-01: compute() – PENDING → COMPUTING

**Trigger**: Scheduled batch job (e.g., monthly fairness analysis)

**Preconditions**:
1. Sufficient decision data (minimum sample size met)
2. Protected attributes defined

**Algorithm**:
```python
def compute_fairness_metrics(protected_attribute, reference_group, protected_group):
    # 1. Fetch decisions for time window
    decisions = db.get_decisions_since(datetime=last_computation_date, tenant_id=current_tenant)
    
    # 2. Filter by groups
    ref_decisions = [d for d in decisions if d.context.get(protected_attribute) == reference_group]
    prot_decisions = [d for d in decisions if d.context.get(protected_attribute) == protected_group]
    
    # 3. Compute statistical parity
    ref_allow_rate = count_allows(ref_decisions) / len(ref_decisions)
    prot_allow_rate = count_allows(prot_decisions) / len(prot_decisions)
    sp_difference = abs(ref_allow_rate - prot_allow_rate)
    
    # 4. Compute disparate impact ratio
    di_ratio = prot_allow_rate / ref_allow_rate if ref_allow_rate > 0 else 0
    
    # 5. Determine compliance
    compliant = (sp_difference <= config.fairness_threshold_sp) and (di_ratio >= config.fairness_threshold_di)
    
    # 6. Store metric
    metric = FairnessMetric(
        metric_type=MetricType.STATISTICAL_PARITY,
        protected_attribute=protected_attribute,
        reference_group=reference_group,
        protected_group=protected_group,
        sp_difference=sp_difference,
        di_ratio=di_ratio,
        threshold_sp=config.fairness_threshold_sp,
        threshold_di=config.fairness_threshold_di,
        compliant=compliant,
        sample_size=len(decisions),
        timestamp=now()
    )
    db.save_fairness_metric(metric)
    
    # 7. Transition state
    if compliant:
        return MetricStatus.COMPLIANT, metric
    else:
        escalate_fairness_issue(metric)
        return MetricStatus.NON_COMPLIANT, metric
```

**Postconditions**:
- Fairness metric computed and stored
- Compliance status determined
- [If non-compliant] Governance escalation triggered

---

## Error Taxonomy

### Transient Errors (Retry)
- Database connection timeout
- External service unavailable
- Network partition

### Permanent Errors (Fail & Alert)
- Schema validation failure
- Cyclic dependency detected
- Multi-sig approval insufficient
- Cryptographic verification failure

### Degraded Mode Errors (Fallback)
- ML service down → use rule-based only
- Cache miss → fetch from database
- External timestamp unavailable → use local timestamp

---

## Related Documents
- state-model.md: State machine definitions
- overview.md: High-level architecture
- api-contracts.md: API endpoints triggering transitions
- requirements.md: Determinism (R-F001), termination (R-F002)

---

**Status**: ✅ Phase 2A Deliverable - COMPLETE  
**Last Updated**: 2025-11-16  
**Owner**: Tech Lead / Backend Engineer
