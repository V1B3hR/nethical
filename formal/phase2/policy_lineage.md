# Policy Lifecycle & Lineage

## Overview
This document specifies the policy versioning, approval workflow, and lineage tracking system for Nethical. Policy lineage ensures auditability, non-repudiation, and contestability by maintaining a cryptographically verifiable chain of policy versions.

---

## Policy Versioning

### Version Identification
Each policy version is uniquely identified by:
- **policy_id**: Logical policy identifier (e.g., `safety.unauthorized_access`)
- **version_hash**: SHA-256 hash of canonical JSON representation of policy content

**Determinism**: Identical policy content → identical version_hash (P-POL-LIN)

### Canonical JSON
To ensure hash consistency, policy content is canonicalized before hashing:
1. Sort object keys alphabetically
2. Remove whitespace
3. Use UTF-8 encoding
4. No trailing commas

**Example**:
```json
{
  "context_whitelist": ["user_input", "environment"],
  "dependencies": ["policy_base_001"],
  "policy_id": "safety.unauthorized_access",
  "rules": [...]
}
```

**Version Hash**: `sha256(canonical_json(policy)) = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

---

## Policy Lifecycle States

### State Diagram
```
DRAFT → QUARANTINE → ACTIVE → INACTIVE
   ↑         ↓           ↓
   └─────────┴───────────┘ (revision loop)
```

**States**:
1. **DRAFT**: Policy being authored; not yet submitted for approval
2. **QUARANTINE**: Policy loaded into system; validation passed; not enforced (testing mode)
3. **ACTIVE**: Policy enforced in production evaluation
4. **INACTIVE**: Policy deactivated; superseded by new version or explicitly retired

---

## Approval Workflow

### Approval Requirements by Criticality

| Criticality | Required Approvals | Approvers |
|-------------|-------------------|-----------|
| **LOW** | 1 signature | Any policy admin |
| **MEDIUM** | 2 signatures | Policy admin + peer reviewer |
| **HIGH** | 3 signatures | Policy admin + peer reviewer + governance lead |
| **CRITICAL** | 4 signatures | Policy admin + peer reviewer + governance lead + security lead |

### Multi-Signature Process

#### 1. Policy Submission
- Author submits policy to repository
- System computes version_hash
- Policy enters QUARANTINE state
- Approval request created with required signature count

#### 2. Signature Collection
Each approver:
1. Reviews policy content
2. Generates signature: `sign(version_hash || approver_id || timestamp, private_key)`
3. Submits signature to system

**Signature Verification**:
```python
def verify_signature(version_hash, approver_id, timestamp, signature, public_key):
    message = version_hash + approver_id + str(timestamp)
    return crypto.verify(message, signature, public_key)
```

#### 3. Activation
Once k-of-n signatures collected:
1. System verifies all signatures
2. Deactivates previous ACTIVE version (if any) for same policy_id
3. Activates new version (QUARANTINE → ACTIVE)
4. Appends activation event to lineage chain

---

## Lineage Chain Structure

### Chain Format
Policy lineage is a linked list of version records, each pointing to its predecessor:

```
Version 1 (HEAD) ──previous_hash──> Version 2 ──previous_hash──> Version 3 ──> ... ──> NULL
```

### Version Record Schema
```json
{
  "policy_id": "string",
  "version_hash": "string (SHA-256 of policy content)",
  "previous_version_hash": "string (SHA-256 of previous version, or null if first)",
  "status": "ACTIVE | INACTIVE",
  "activated_at": "ISO 8601 datetime",
  "deactivated_at": "ISO 8601 datetime (null if ACTIVE)",
  "approvers": [
    {
      "approver_id": "string",
      "signature": "base64 cryptographic signature",
      "timestamp": "ISO 8601 datetime"
    }
  ],
  "activation_event_id": "string (references audit log event)",
  "lineage_chain_hash": "string (hash of this record + previous_lineage_chain_hash)"
}
```

### Lineage Chain Hash
```python
def compute_lineage_chain_hash(version_record, previous_chain_hash):
    record_hash = sha256(canonical_json(version_record))
    if previous_chain_hash:
        return sha256(record_hash + previous_chain_hash)
    else:
        return record_hash  # First version in chain
```

**Property (P-POL-LIN)**: Any modification to historical version record breaks chain hash verification.

---

## Lineage Verification

### Verify Single Version
```python
def verify_version_record(version_record):
    # 1. Verify version_hash matches content
    computed_hash = sha256(canonical_json(version_record.content))
    if computed_hash != version_record.version_hash:
        return False, "Version hash mismatch"
    
    # 2. Verify approver signatures
    for approver in version_record.approvers:
        if not verify_signature(version_record.version_hash, approver.approver_id, approver.timestamp, approver.signature, approver.public_key):
            return False, f"Invalid signature from {approver.approver_id}"
    
    # 3. Verify lineage_chain_hash
    prev_record = db.get_policy_version(version_record.previous_version_hash)
    prev_chain_hash = prev_record.lineage_chain_hash if prev_record else None
    computed_chain_hash = compute_lineage_chain_hash(version_record, prev_chain_hash)
    if computed_chain_hash != version_record.lineage_chain_hash:
        return False, "Lineage chain hash mismatch"
    
    return True, "Verified"
```

### Verify Entire Chain
```python
def verify_policy_lineage(policy_id):
    current = db.get_active_policy_version(policy_id)
    while current:
        verified, msg = verify_version_record(current)
        if not verified:
            return False, msg, current.version_hash
        
        if current.previous_version_hash:
            current = db.get_policy_version(current.previous_version_hash)
        else:
            break  # Reached genesis version
    
    return True, "Full lineage verified", None
```

---

## Policy Diff Auditing

### Diff Computation
When activating a new policy version, compute diff against previous version:

**Algorithm**:
```python
def compute_policy_diff(version_new, version_old):
    diff = {
        "additions": [],
        "deletions": [],
        "modifications": []
    }
    
    # Rule-level diff
    old_rules = {r['rule_id']: r for r in version_old.content.rules}
    new_rules = {r['rule_id']: r for r in version_new.content.rules}
    
    # Additions
    for rule_id in set(new_rules.keys()) - set(old_rules.keys()):
        diff["additions"].append(new_rules[rule_id])
    
    # Deletions
    for rule_id in set(old_rules.keys()) - set(new_rules.keys()):
        diff["deletions"].append(old_rules[rule_id])
    
    # Modifications
    for rule_id in set(old_rules.keys()) & set(new_rules.keys()):
        if old_rules[rule_id] != new_rules[rule_id]:
            diff["modifications"].append({
                "rule_id": rule_id,
                "old": old_rules[rule_id],
                "new": new_rules[rule_id]
            })
    
    return diff
```

### Diff Storage
Store diff alongside version record for audit purposes:
```json
{
  "diff_id": "string",
  "policy_id": "string",
  "from_version_hash": "string",
  "to_version_hash": "string",
  "diff": {
    "additions": [...],
    "deletions": [...],
    "modifications": [...]
  },
  "summary": "Added 2 rules, deleted 1 rule, modified 3 rules",
  "timestamp": "ISO 8601 datetime"
}
```

---

## External Timestamping (Optional)

For strongest non-repudiation, submit lineage_chain_hash to external timestamping service:

### Timestamping Flow
1. Compute lineage_chain_hash for new version
2. Submit hash to timestamping authority (e.g., RFC 3161 compliant service, blockchain anchor)
3. Receive timestamp proof (signed timestamp + hash)
4. Store timestamp proof with version record

### Timestamping Service Integration
```python
def submit_to_timestamping(lineage_chain_hash, version_record_id):
    # Example: RFC 3161 Time-Stamp Protocol
    tsp_request = create_tsp_request(lineage_chain_hash)
    tsp_response = http_post(config.timestamping_service_url, tsp_request)
    
    timestamp_proof = parse_tsp_response(tsp_response)
    db.save_timestamp_proof(version_record_id, timestamp_proof)
    
    return timestamp_proof
```

**Benefits**:
- External proof that version existed at specific time
- Strengthens non-repudiation (cannot backdate lineage)
- Compliant with legal/regulatory requirements for timestamping

---

## Quarantine Mode Testing

### Purpose
Quarantine mode allows testing new policy versions without production impact.

### Behavior
- Policy loaded with status = QUARANTINE
- During evaluation, policy executed but results logged as "would-have" only
- Actual decision uses only ACTIVE policies
- Comparison report generated (QUARANTINE vs ACTIVE outcomes)

### Comparison Metrics
```json
{
  "quarantine_test_id": "string",
  "policy_id": "string",
  "quarantine_version_hash": "string",
  "test_period": "start_datetime to end_datetime",
  "actions_evaluated": 10000,
  "agreement_rate": 0.95,
  "disagreements": [
    {
      "action_id": "string",
      "active_judgment": "ALLOW",
      "quarantine_judgment": "BLOCK",
      "reason": "New rule detected violation"
    }
  ],
  "false_positive_rate": 0.02,
  "false_negative_rate": 0.03
}
```

### Promotion Criteria
Promote from QUARANTINE to ACTIVE if:
1. Agreement rate > 95% with existing ACTIVE policies
2. False positive rate < 5%
3. False negative rate < 5%
4. No critical disagreements (ALLOW vs TERMINATE)
5. Multi-sig approval obtained

---

## Policy Rollback

### Emergency Rollback Procedure
If critical issue discovered in ACTIVE policy:
1. Deactivate problematic version (ACTIVE → INACTIVE)
2. Activate previous version (INACTIVE → ACTIVE) — use previous_version_hash from lineage
3. Append rollback event to audit log
4. Investigate root cause

**Rollback Constraints**:
- Requires emergency authorization (2 signatures from governance + security leads)
- Rollback event signed and timestamped
- Post-incident review mandatory

---

## Policy Retirement

### Permanent Retirement
For policies no longer needed:
1. Deactivate policy (ACTIVE → INACTIVE)
2. Mark as RETIRED in lineage
3. Append retirement event to audit log
4. Do not delete (preserve for historical audit)

**RETIRED policies**:
- Cannot be reactivated (immutable retirement)
- Preserved for contestability (past decisions may reference retired policies)
- Excluded from future evaluations

---

## Lineage Dashboard

### Key Metrics
- Total policy versions tracked
- Active policies per tenant
- Average approval time (submission → activation)
- Rollback frequency
- Lineage chain verification success rate (100% target)

### Visualizations
- Policy version timeline (Gantt chart)
- Approval workflow status (funnel chart)
- Lineage chain graph (directed acyclic graph visualization)
- Diff heatmap (additions/deletions/modifications over time)

---

## Success Criteria (Phase 2B)

Phase 2B (Policy Lifecycle & Lineage) is complete when:
1. ✅ Approval workflow designed with multi-sig requirements
2. ✅ Lineage chain structure specified (hash chain design)
3. ✅ Verification algorithms documented
4. ✅ Policy diff auditing process defined
5. ✅ Quarantine mode testing procedure specified
6. ✅ Emergency rollback process documented
7. ✅ Integration with Phase 3B Merkle audit design

---

## Related Documents
- state-model.md: Policy state machine
- transitions.md: Policy activation transitions
- api-contracts.md: Policy management API
- overview.md: System architecture
- requirements.md: R-F006 (Policy Lineage Tracking)

---

**Status**: ✅ Phase 2B Deliverable - COMPLETE  
**Last Updated**: 2025-11-16  
**Owner**: Backend Engineer / Governance Lead
