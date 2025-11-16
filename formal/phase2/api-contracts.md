# API Contracts

## Overview
This document specifies the external API contracts for the Nethical governance platform. Each endpoint includes request/response schemas, error codes, and governance constraints.

---

## Base URL
```
Production: https://api.nethical.example.com/v1
Staging: https://staging-api.nethical.example.com/v1
```

## Authentication
All endpoints require authentication via:
- **API Key** (Header: `X-API-Key: <key>`)
- **JWT Token** (Header: `Authorization: Bearer <token>`)
- **SSO/SAML** (via OAuth 2.0 flow for admin operations)

---

## Core Decision API

### POST /actions/evaluate
**Purpose**: Submit an action for governance evaluation  
**Authentication**: API Key or JWT  
**Rate Limit**: Per tenant quota (default: 1000 RPS)

**Request**:
```json
{
  "agent_id": "string (required)",
  "action_id": "string (optional, auto-generated if not provided)",
  "action_type": "string (required, e.g., 'response', 'api_call', 'data_access')",
  "context": {
    "user_input": "string",
    "environment": "string",
    "history": ["array of recent actions"],
    "custom_fields": {}
  },
  "cohort": "string (optional, default: 'default')",
  "tenant_id": "string (required for multi-tenant)"
}
```

**Response** (200 OK):
```json
{
  "action_id": "string",
  "judgment": "ALLOW | RESTRICT | BLOCK | TERMINATE",
  "risk_score": "float [0.0, 1.0]",
  "confidence": "float [0.0, 1.0]",
  "violations": [
    {
      "type": "safety | ethical | privacy | fairness",
      "severity": "low | medium | high | critical",
      "description": "string",
      "contributing_policies": ["policy_id_1", "policy_id_2"]
    }
  ],
  "justification": "string (human-readable explanation)",
  "restrictions": ["array of conditions if judgment = RESTRICT"],
  "timestamp": "ISO 8601 datetime",
  "evaluation_time_ms": "float"
}
```

**Error Responses**:
- **400 Bad Request**: Invalid schema or missing required fields
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Tenant quota exceeded or agent suspended
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Evaluation failure; check logs

**Governance Constraints**:
- **P-DATA-MIN**: Only whitelisted context fields used per active policies
- **P-DET**: Identical requests produce identical responses (idempotent if action_id provided)

---

### GET /decisions/{action_id}
**Purpose**: Retrieve a past decision  
**Authentication**: API Key or JWT  
**Authorization**: Tenant isolation enforced (can only access own tenant's decisions)

**Response** (200 OK):
```json
{
  "action_id": "string",
  "judgment": "ALLOW | RESTRICT | BLOCK | TERMINATE",
  "risk_score": "float",
  "confidence": "float",
  "violations": [...],
  "justification": "string",
  "restrictions": [...],
  "timestamp": "ISO 8601 datetime",
  "status": "DECIDED | BLOCKED | ESCALATED | REVIEWED | APPEALED | RE_DECIDED",
  "escalation_ticket_id": "string (if escalated)",
  "appeal_outcome": "object (if appealed)"
}
```

**Error Responses**:
- **404 Not Found**: Decision not found or not accessible

---

## Policy Management API

### POST /policies
**Purpose**: Load a new policy into the system  
**Authentication**: JWT (admin role required)  
**Authorization**: Policy management permission

**Request**:
```json
{
  "policy_id": "string (required)",
  "content": "object (policy definition)",
  "dependencies": ["array of policy_ids"],
  "context_whitelist": ["array of allowed context field names"],
  "criticality": "low | medium | high | critical"
}
```

**Response** (201 Created):
```json
{
  "policy_id": "string",
  "version_hash": "string (SHA-256)",
  "status": "QUARANTINE",
  "created_at": "ISO 8601 datetime"
}
```

**Error Responses**:
- **400 Bad Request**: Invalid schema, cyclic dependencies
- **409 Conflict**: Policy version already exists

---

### POST /policies/{policy_id}/activate
**Purpose**: Activate a policy from QUARANTINE to ACTIVE  
**Authentication**: JWT (admin role required)  
**Authorization**: Multi-sig approval required for criticality ≥ HIGH

**Request**:
```json
{
  "version_hash": "string (required)",
  "approver_signatures": [
    {
      "approver_id": "string",
      "signature": "base64 cryptographic signature",
      "timestamp": "ISO 8601 datetime"
    }
  ]
}
```

**Response** (200 OK):
```json
{
  "policy_id": "string",
  "version_hash": "string",
  "status": "ACTIVE",
  "activated_at": "ISO 8601 datetime",
  "approvers": ["array of approver_ids"]
}
```

**Error Responses**:
- **403 Forbidden**: Insufficient approvals (multi-sig not met)
- **409 Conflict**: Another version already ACTIVE

**Governance Constraints**:
- **P-MULTI-SIG**: k-of-n signatures verified cryptographically
- **P-POL-LIN**: Activation event appended to policy lineage chain

---

### GET /policies
**Purpose**: List active policies  
**Authentication**: API Key or JWT  
**Authorization**: Tenant-specific (returns only tenant's policies)

**Query Parameters**:
- `status`: Filter by status (ACTIVE, QUARANTINE, INACTIVE)
- `page`: Pagination (default: 1)
- `per_page`: Results per page (default: 50, max: 100)

**Response** (200 OK):
```json
{
  "policies": [
    {
      "policy_id": "string",
      "version_hash": "string",
      "status": "ACTIVE | QUARANTINE | INACTIVE",
      "activated_at": "ISO 8601 datetime",
      "dependencies": ["array of policy_ids"]
    }
  ],
  "page": 1,
  "per_page": 50,
  "total_count": 123
}
```

---

## Audit & Integrity API

### GET /audit/decisions
**Purpose**: Query audit log for decisions  
**Authentication**: JWT (auditor role required)  
**Authorization**: Tenant isolation enforced

**Query Parameters**:
- `agent_id`: Filter by agent
- `judgment`: Filter by judgment type
- `start_date`, `end_date`: Time range
- `page`, `per_page`: Pagination

**Response** (200 OK):
```json
{
  "decisions": [...],
  "page": 1,
  "per_page": 50,
  "total_count": 456
}
```

---

### GET /audit/merkle/verify/{event_id}
**Purpose**: Verify Merkle proof for an audit event  
**Authentication**: API Key or JWT

**Response** (200 OK):
```json
{
  "event_id": "string",
  "event_hash": "string (SHA-256)",
  "merkle_path": ["array of hash strings"],
  "merkle_root": "string",
  "batch_id": "string",
  "verified": true,
  "timestamp": "ISO 8601 datetime"
}
```

**Governance Constraints**:
- **P-AUD**: All audit events verifiable via Merkle proof
- **P-NONREP**: Merkle root cryptographically signed

---

## Fairness & Compliance API

### GET /fairness/metrics
**Purpose**: Retrieve fairness metrics for protected attributes  
**Authentication**: JWT (governance role required)

**Query Parameters**:
- `protected_attribute`: Filter by attribute (race, gender, age, etc.)
- `start_date`, `end_date`: Time range
- `compliant`: Filter by compliance status (true/false)

**Response** (200 OK):
```json
{
  "metrics": [
    {
      "metric_id": "string",
      "metric_type": "statistical_parity | disparate_impact_ratio",
      "protected_attribute": "gender",
      "reference_group": "male",
      "protected_group": "female",
      "sp_difference": 0.08,
      "di_ratio": 0.92,
      "threshold_sp": 0.10,
      "threshold_di": 0.80,
      "compliant": true,
      "sample_size": 10000,
      "timestamp": "ISO 8601 datetime"
    }
  ]
}
```

---

## Human-in-the-Loop API

### GET /escalations
**Purpose**: List escalated decisions awaiting human review  
**Authentication**: JWT (reviewer role required)

**Response** (200 OK):
```json
{
  "escalations": [
    {
      "ticket_id": "string",
      "decision_id": "string (action_id)",
      "priority": "low | medium | high | critical",
      "reason": "high_risk | low_confidence | policy_flag",
      "sla_deadline": "ISO 8601 datetime",
      "created_at": "ISO 8601 datetime",
      "status": "PENDING | UNDER_REVIEW | RESOLVED"
    }
  ]
}
```

---

### POST /escalations/{ticket_id}/review
**Purpose**: Submit human review for escalated decision  
**Authentication**: JWT (reviewer role required)

**Request**:
```json
{
  "decision": "APPROVE | OVERRIDE_ALLOW | OVERRIDE_BLOCK | REQUEST_APPEAL",
  "feedback": "string (reviewer comments)",
  "corrected_judgment": "ALLOW | RESTRICT | BLOCK | TERMINATE (if override)"
}
```

**Response** (200 OK):
```json
{
  "ticket_id": "string",
  "status": "RESOLVED",
  "reviewed_at": "ISO 8601 datetime",
  "reviewer_id": "string"
}
```

---

## Appeals & Contestability API

### POST /appeals
**Purpose**: File an appeal to contest a decision  
**Authentication**: API Key or JWT

**Request**:
```json
{
  "decision_id": "string (action_id)",
  "reason": "string (explanation of why decision is contested)",
  "requester_email": "string"
}
```

**Response** (201 Created):
```json
{
  "appeal_id": "string",
  "decision_id": "string",
  "status": "PENDING",
  "sla_deadline": "ISO 8601 datetime (now + 72h)",
  "created_at": "ISO 8601 datetime"
}
```

---

### GET /appeals/{appeal_id}
**Purpose**: Check status of an appeal  
**Authentication**: API Key or JWT

**Response** (200 OK):
```json
{
  "appeal_id": "string",
  "decision_id": "string",
  "status": "PENDING | UNDER_REVIEW | RESOLVED",
  "original_judgment": "ALLOW | RESTRICT | BLOCK | TERMINATE",
  "re_evaluated_judgment": "ALLOW | RESTRICT | BLOCK | TERMINATE (if resolved)",
  "diff_artifact": "object (differences between original and re-evaluation)",
  "resolution_time_hours": 48.5,
  "resolved_at": "ISO 8601 datetime (if resolved)"
}
```

**Governance Constraints**:
- **P-APPEAL**: Re-evaluation deterministic (same context → same judgment)

---

## Data Subject Rights API (GDPR/CCPA)

### POST /dsr/access
**Purpose**: Request access to personal data (GDPR Art. 15, CCPA)  
**Authentication**: JWT or email verification

**Request**:
```json
{
  "email": "string",
  "identifier_type": "email | agent_id | user_id",
  "identifier_value": "string"
}
```

**Response** (200 OK):
```json
{
  "request_id": "string",
  "status": "PENDING",
  "estimated_completion": "ISO 8601 datetime (30 days from now)"
}
```

---

### POST /dsr/delete
**Purpose**: Request deletion of personal data (GDPR Art. 17 RTBF, CCPA)  
**Authentication**: JWT or email verification

**Request**:
```json
{
  "email": "string",
  "identifier_type": "email | agent_id | user_id",
  "identifier_value": "string"
}
```

**Response** (200 OK):
```json
{
  "request_id": "string",
  "status": "PENDING",
  "retention_policies_applied": ["audit logs retained per regulatory requirements"],
  "estimated_completion": "ISO 8601 datetime"
}
```

**Note**: Audit logs may be retained for compliance despite deletion request; anonymization applied.

---

## Observability API

### GET /metrics
**Purpose**: Prometheus-compatible metrics export  
**Authentication**: Internal only (metrics scraping endpoint)

**Response** (200 OK, text/plain):
```
# HELP nethical_actions_total Total number of actions evaluated
# TYPE nethical_actions_total counter
nethical_actions_total{judgment="ALLOW"} 12345
nethical_actions_total{judgment="BLOCK"} 678

# HELP nethical_risk_score Risk score distribution
# TYPE nethical_risk_score histogram
nethical_risk_score_bucket{le="0.1"} 1000
nethical_risk_score_bucket{le="0.5"} 5000
...
```

---

### GET /health
**Purpose**: Health check endpoint  
**Authentication**: None (public)

**Response** (200 OK):
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 123456,
  "checks": {
    "database": "healthy",
    "policy_cache": "healthy",
    "ml_service": "degraded (using rule-based fallback)"
  }
}
```

---

## Error Response Format

All error responses follow this schema:
```json
{
  "error": {
    "code": "string (ERROR_CODE)",
    "message": "string (human-readable)",
    "details": "object (optional, additional context)",
    "request_id": "string (for support/debugging)"
  }
}
```

**Common Error Codes**:
- `INVALID_REQUEST`: Schema validation failure
- `UNAUTHORIZED`: Authentication failure
- `FORBIDDEN`: Authorization failure
- `RATE_LIMIT_EXCEEDED`: Quota enforcement
- `RESOURCE_NOT_FOUND`: Entity not found
- `CONFLICT`: State conflict (e.g., duplicate policy)
- `INTERNAL_ERROR`: Unexpected server error

---

## API Versioning

Versioning strategy: URL-based (`/v1`, `/v2`)
- **Breaking changes**: New major version (`/v2`)
- **Non-breaking additions**: Same version (`/v1`)
- **Deprecation**: Minimum 6-month notice before removing old version

---

## Rate Limiting

Headers returned on all responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1633024800 (Unix timestamp)
```

When rate limit exceeded (429 Too Many Requests):
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Quota exceeded for tenant",
    "retry_after_seconds": 60
  }
}
```

---

## Related Documents
- overview.md: System architecture
- state-model.md: Entity states
- transitions.md: State transition logic
- requirements.md: API requirements

---

**Status**: ✅ Phase 2A Deliverable - COMPLETE  
**Last Updated**: 2025-11-16  
**Owner**: Tech Lead / API Developer
