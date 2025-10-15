# Data Subject Rights (DSR) Runbook

## Overview
Procedures for handling data subject requests under GDPR, CCPA, and similar regulations.

## Right to Access
**Request**: User wants copy of their data

**Procedure**:
1. Identify agent_id or action_id
2. Query audit logs: `MerkleAnchor.get_events()`
3. Export redacted logs in JSON format
4. Verify Merkle integrity before delivery

**Code Example**:
```python
# Query events for specific agent
events = merkle_anchor.get_events(
    agent_id="user_agent_123",
    start_time=start_date,
    end_time=end_date
)
# Redact before export
for event in events:
    redacted = redaction_pipeline.redact(str(event))
    # Export redacted.redacted_text
```

## Right to Erasure (RTBF)
**Request**: User wants data deleted

**Procedure**:
1. Identify all data for agent_id
2. Use Data Minimization delete_data()
3. Verify Merkle chain impact
4. Document erasure in compliance log

**Code Example**:
```python
# Delete user data
result = data_minimization.delete_data(
    user_id="user_123",
    category=DataCategory.USER_DATA
)
# Record deletion
audit_log.record_deletion(user_id, timestamp, reason="RTBF_request")
```

## Right to Rectification
**Request**: User wants data corrected

**Procedure**:
1. Locate incorrect data in audit logs
2. Create correction entry (append-only, no modification)
3. Link correction to original via Merkle chain
4. Update risk profile if necessary

## Right to Restriction
**Request**: User wants processing restricted

**Procedure**:
1. Add agent_id to restriction list
2. Set quarantine flag in QuarantineManager
3. Log restriction reason and duration
4. Monitor for automatic expiry

## SLA Targets
- **Acknowledgment**: 48 hours
- **Access Request**: 30 days
- **Erasure**: 30 days (or as required by law)
- **Rectification**: 7 days

---
Last Updated: 2025-10-15
