# Nethical Data Flow Documentation

**Version**: 1.0.0  
**Last Updated**: 2025-12-03

---

## Overview

This document describes the data flows within Nethical, including what data is processed, how it moves through the system, and what is logged for audit purposes.

---

## Data Flow Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI AGENT / AI SYSTEM                               │
│                    (Autonomous Vehicle, Robot, AI Platform)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Action Request
                                      │ (action_type, context)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NETHICAL GOVERNANCE ENGINE                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Policy    │  │    Risk     │  │  Detector   │  │   Decision  │         │
│  │   Engine    │  │  Scoring    │  │  Framework  │  │   Engine    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                     ┌────────────────┼────────────────┐
                     ▼                ▼                ▼
              ┌───────────┐    ┌───────────┐    ┌───────────┐
              │  Audit    │    │  Metrics  │    │   Cache   │
              │   Log     │    │ (Prometheus)│   │  (Redis)  │
              └───────────┘    └───────────┘    └───────────┘
                     │                │                │
                     ▼                ▼                ▼
              ┌───────────┐    ┌───────────┐    ┌───────────┐
              │ PostgreSQL│    │  Grafana  │    │  L2/L3    │
              │ + Timescale│   │ Dashboard │    │  Cache    │
              └───────────┘    └───────────┘    └───────────┘
```

---

## What Data Flows Through Nethical

### Input Data (What We Receive)

| Data Element | Description | Example | Personal Data? |
|--------------|-------------|---------|----------------|
| `agent_id` | Unique AI agent identifier | `"agent-12345"` | No |
| `action_type` | Category of AI action | `"code_generation"` | No |
| `action` | Brief action description | `"Generate SQL query"` | No |
| `context` | Action context metadata | `{"domain": "finance"}` | No |
| `timestamp` | When action occurred | `2025-12-03T10:30:00Z` | No |

### Output Data (What We Return)

| Data Element | Description | Example | Personal Data? |
|--------------|-------------|---------|----------------|
| `decision` | Governance decision | `"ALLOW"` / `"BLOCK"` | No |
| `risk_score` | Risk assessment (0.0-1.0) | `0.23` | No |
| `violations` | Policy violations detected | `["pii_detected"]` | No |
| `explanation` | Decision explanation | `"Action within policy"` | No |
| `latency_ms` | Processing time | `5` | No |

---

## What is Logged

### Governance Decision Log

Every governance decision is logged with the following structure:

```json
{
  "event_id": "uuid-v4",
  "timestamp": "2025-12-03T10:30:00.123Z",
  "agent_id": "agent-12345",
  "action_type": "code_generation",
  "decision": "ALLOW",
  "risk_score": 0.23,
  "latency_ms": 5,
  "policies_evaluated": ["code-safety-policy", "pii-protection"],
  "violations": [],
  "fundamental_laws_checked": [1, 2, 3, 8, 15, 22],
  "merkle_root": "sha256:abc123..."
}
```

### What Each Field Represents

| Field | Purpose | Why Logged |
|-------|---------|------------|
| `event_id` | Unique identifier | Traceability |
| `timestamp` | When decision made | Audit timeline |
| `agent_id` | Which AI agent | Agent accountability |
| `action_type` | Action category | Pattern analysis |
| `decision` | Governance outcome | Core audit record |
| `risk_score` | Risk assessment | Safety monitoring |
| `latency_ms` | Performance | SLO compliance |
| `policies_evaluated` | Policies applied | Policy traceability |
| `violations` | Issues detected | Compliance reporting |
| `fundamental_laws_checked` | Laws verified | Ethical compliance |
| `merkle_root` | Cryptographic anchor | Tamper detection |

---

## What is NOT Logged

The following data is explicitly NOT collected or logged:

| Data Type | Why Not Logged | Privacy Impact |
|-----------|----------------|----------------|
| **User Identity** | Not relevant to AI governance | Protects user privacy |
| **Personal Data** | Not needed for decisions | GDPR compliance |
| **Conversation Content** | Only action metadata needed | Content privacy |
| **File Contents** | Not relevant to governance | Data minimization |
| **IP Addresses** | Not needed for AI governance | Location privacy |
| **User Credentials** | Security risk, not needed | Credential protection |
| **Browsing History** | Not relevant | User privacy |
| **Keystrokes** | Not relevant | User privacy |

### Exception: Required by Policy

In rare cases, additional data may be logged if:
1. Explicitly required by organizational policy
2. Required for regulatory compliance
3. Approved by DPO/legal

In such cases:
- Data minimization still applies
- Retention limits enforced
- Access strictly controlled
- Clear documentation required

---

## Data Flow by Component

### 1. Edge Governance Flow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  AI Agent      │────▶│ Edge Governor  │────▶│ Local Cache    │
│ (On Device)    │     │ (< 10ms)       │     │ (L1 Memory)    │
└────────────────┘     └────────────────┘     └────────────────┘
                              │
                              │ Sync (async)
                              ▼
                       ┌────────────────┐
                       │ Cloud Backend  │
                       │ (Audit Log)    │
                       └────────────────┘
```

**Data at Edge**:
- Local policy cache
- Recent decisions (ephemeral)
- Predictive pre-computations

**Data Synced to Cloud**:
- Governance decisions (for audit)
- Policy updates (from cloud)
- Metrics (for monitoring)

---

### 2. Cloud API Flow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  AI Platform   │────▶│  API Gateway   │────▶│ Load Balancer  │
│  (Enterprise)  │     │ (Rate Limit)   │     │                │
└────────────────┘     └────────────────┘     └────────────────┘
                                                      │
                              ┌────────────────────────┤
                              ▼                        ▼
                       ┌────────────────┐     ┌────────────────┐
                       │ Nethical API   │     │  L2 Cache      │
                       │ (Governance)   │     │  (Redis)       │
                       └────────────────┘     └────────────────┘
                              │
                              ▼
                       ┌────────────────┐
                       │  PostgreSQL    │
                       │ (Audit Trail)  │
                       └────────────────┘
```

**Data Flow**:
1. Request arrives with action metadata
2. Rate limiting and authentication applied
3. Cache checked for recent similar decisions
4. Policy evaluation performed
5. Decision logged to audit trail
6. Response returned with decision

---

### 3. Multi-Region Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    Global Load Balancer                         │
└────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌───────────┐        ┌───────────┐        ┌───────────┐
   │  US-EAST  │        │  EU-WEST  │        │ AP-SOUTH  │
   │  Region   │        │  Region   │        │  Region   │
   └───────────┘        └───────────┘        └───────────┘
         │                    │                    │
         │    ┌───────────────┼───────────────┐    │
         │    │               ▼               │    │
         │    │     ┌────────────────┐       │    │
         └────┼────▶│  Policy Sync   │◀──────┼────┘
              │     │  (CRDT-based)  │       │
              │     └────────────────┘       │
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                       ┌────────────────┐
                       │  Global Audit  │
                       │    Stream      │
                       └────────────────┘
```

**Data Residency**:
- EU data stays in EU regions
- US data stays in US regions
- Policies replicated globally (anonymized)
- Audit logs follow data residency rules

---

## Data Retention Policies

### Retention Schedule

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LIFECYCLE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Hot Storage ─────────▶ Warm Storage ─────────▶ Cold Storage    │
│  (Active)              (Archive)               (Compliance)      │
│                                                                  │
│  ┌─────────┐           ┌─────────┐             ┌─────────┐      │
│  │ 0-90    │           │ 90-365  │             │ 1-7     │      │
│  │ days    │           │ days    │             │ years   │      │
│  └─────────┘           └─────────┘             └─────────┘      │
│                                                                  │
│  Real-time             Queryable               Audit-only        │
│  Dashboard             Analytics               Compliance         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### By Data Type

| Data Category | Hot | Warm | Cold | Total Retention |
|---------------|-----|------|------|-----------------|
| Governance Decisions | 90 days | 275 days | 6 years | 7 years |
| Policy Evaluations | 30 days | 335 days | - | 1 year |
| Performance Metrics | 30 days | 60 days | - | 90 days |
| Security Events | 90 days | 275 days | 6 years | 7 years |
| Debug Logs | 7 days | 23 days | - | 30 days |

---

## Data Security in Transit and at Rest

### Encryption Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ENCRYPTION FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    TLS 1.3    ┌─────────┐    TLS 1.3    ┌─────────┐│
│  │ Client  │──────────────▶│   API   │──────────────▶│ Database││
│  └─────────┘               └─────────┘               └─────────┘│
│                                                         │        │
│                                                    ┌────▼────┐   │
│                                                    │ AES-256 │   │
│                                                    │ At Rest │   │
│                                                    └─────────┘   │
│                                                         │        │
│                                                    ┌────▼────┐   │
│                                                    │   HSM   │   │
│                                                    │  Keys   │   │
│                                                    └─────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Protection by Layer

| Layer | Protection Method | Standard |
|-------|-------------------|----------|
| Transport | TLS 1.3 | NIST guidelines |
| Application | JWT/API keys | OAuth 2.0 |
| Database | AES-256-GCM | FIPS 140-2 |
| Key Management | HSM | PKCS#11 |
| Audit Logs | Merkle Tree | SHA-256 |

---

## Access Control Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ACCESS CONTROL FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │  User   │───▶│  Auth   │───▶│  RBAC   │───▶│ Access  │       │
│  │ Request │    │  (MFA)  │    │ Check   │    │ Granted │       │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│                      │              │              │             │
│                      ▼              ▼              ▼             │
│                 ┌─────────┐   ┌─────────┐   ┌─────────┐         │
│                 │ Audit   │   │ Audit   │   │ Audit   │         │
│                 │ Log     │   │ Log     │   │ Log     │         │
│                 └─────────┘   └─────────┘   └─────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

All access attempts are logged, regardless of success or failure.

---

## Data Subject Request Flow

For GDPR data subject requests:

```
┌─────────────────────────────────────────────────────────────────┐
│                  DATA SUBJECT REQUEST FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │  DSR    │───▶│ Verify  │───▶│ Process │───▶│ Respond │       │
│  │ Received│    │ Identity│    │ Request │    │ (30 days)│      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│                                     │                            │
│                      ┌──────────────┼──────────────┐             │
│                      ▼              ▼              ▼             │
│                 ┌─────────┐   ┌─────────┐   ┌─────────┐         │
│                 │ Access  │   │ Rectify │   │ Erasure │         │
│                 │ Export  │   │ Correct │   │ Delete  │         │
│                 └─────────┘   └─────────┘   └─────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

### Data We Process

| Category | Examples | Purpose |
|----------|----------|---------|
| AI Actions | Action types, decisions | Governance |
| Risk Data | Scores, violations | Safety |
| Audit Data | Timestamps, IDs | Compliance |
| Metrics | Latency, throughput | Operations |

### Data We Don't Process

| Category | Why Not |
|----------|---------|
| Personal Data | Not relevant to AI governance |
| User Content | Privacy protection |
| Credentials | Security risk |
| Location Data | Not needed |

---

## Related Documents

- [Privacy Policy](../../PRIVACY.md) - Data practices overview
- [Deployment Scope](DEPLOYMENT_SCOPE.md) - Appropriate use guidance
- [Audit Response Guide](AUDIT_RESPONSE_GUIDE.md) - Auditor FAQ
- [Architecture](../../ARCHITECTURE.md) - System design

---

**"Transparent data flows, protected privacy."**
