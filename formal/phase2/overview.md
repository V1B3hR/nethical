# System Overview

## Purpose
This document provides a high-level architectural overview of the Nethical governance platform, describing its core components, data flows, and operational context.

---

## System Mission

Nethical serves as a **real-time governance and safety layer** for AI agent operations. It continuously monitors agent actions, evaluates them against policies, and makes graduated enforcement decisions (ALLOW, RESTRICT, BLOCK, TERMINATE) to ensure:

- **Safety**: Prevention of harmful, dangerous, or unauthorized behaviors
- **Ethics**: Compliance with ethical principles and societal norms
- **Fairness**: Equitable treatment across population groups
- **Privacy**: Protection of personally identifiable information
- **Transparency**: Auditable, explainable, and contestable decisions
- **Compliance**: Adherence to regulatory requirements (GDPR, CCPA, EU AI Act, etc.)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AI Agents                                │
│  (Chatbots, Decision Systems, Recommendation Engines, etc.)      │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Actions
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Nethical Governance Platform                    │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │              Ingestion & Preprocessing                       │ │
│ │  • Action validation      • PII detection & redaction        │ │
│ │  • Context enrichment     • Tenant isolation check           │ │
│ └────────────────────────┬───────────────────────────────────┬─┘ │
│                          │                                   │   │
│                          ▼                                   │   │
│ ┌────────────────────────────────────────────────────────────┐  │
│ │           Policy Evaluation Engine                          │  │
│ │  • Topological sort of policies (determinism)              │  │
│ │  • Data minimization enforcement (context whitelisting)    │  │
│ │  • Rule-based safety/ethical violation detection           │  │
│ │  • Risk score computation                                   │  │
│ └────────────────────────┬───────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│ ┌────────────────────────────────────────────────────────────┐  │
│ │           ML & Anomaly Detection (Optional)                 │  │
│ │  • Shadow mode ML classifier                                │  │
│ │  • Blended risk scoring (rule + ML)                         │  │
│ │  • Behavioral anomaly detection                             │  │
│ │  • Drift monitoring (PSI, KL divergence)                    │  │
│ └────────────────────────┬───────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│ ┌────────────────────────────────────────────────────────────┐  │
│ │              Safety Judge & Decision                        │  │
│ │  • Graduated judgment: ALLOW / RESTRICT / BLOCK / TERMINATE │  │
│ │  • Justification generation                                 │  │
│ │  • Confidence scoring                                       │  │
│ └────────────────────────┬───────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│ ┌────────────────────────────────────────────────────────────┐  │
│ │           Audit & Integrity Layer                           │  │
│ │  • Merkle-anchored audit log (tamper-evident)              │  │
│ │  • Policy lineage tracking (version hash chain)            │  │
│ │  • Fairness metrics aggregation                             │  │
│ │  • SLA monitoring                                           │  │
│ └────────────────────────┬───────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│ ┌────────────────────────────────────────────────────────────┐  │
│ │         Human-in-the-Loop & Optimization                    │  │
│ │  • Escalation queue for high-risk / low-confidence          │  │
│ │  • Human feedback collection                                │  │
│ │  • Multi-objective optimization (thresholds, weights)       │  │
│ │  • A/B testing and promotion gates                          │  │
│ └────────────────────────┬───────────────────────────────────┘  │
│                          │                                       │
└──────────────────────────┼───────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────────┐
        │     External Integrations & Outputs      │
        │  • Observability (OTEL, Prometheus)      │
        │  • Audit portal (decision explorer)      │
        │  • Appeals CLI (contestability)          │
        │  • Alerting & notifications              │
        │  • Data subject rights API               │
        └──────────────────────────────────────────┘
```

---

## Core Components

### 1. Ingestion & Preprocessing
**Purpose**: Receive and prepare actions for evaluation  
**Responsibilities**:
- Validate action schema and required fields
- Detect and redact PII (10+ types) per configured policy
- Enrich context with agent history, cohort info, tenant metadata
- Enforce tenant isolation boundaries
- Apply quota/rate limiting checks

**Key Properties**:
- **P-DATA-MIN**: Only whitelisted fields passed to policy evaluation
- **P-TENANT-ISO**: Tenant data strictly segregated

---

### 2. Policy Evaluation Engine
**Purpose**: Deterministically evaluate actions against loaded policies  
**Responsibilities**:
- Load and validate policy dependency graph (acyclicity check)
- Topologically sort policies for deterministic evaluation order
- Execute policies in sorted order with memoization
- Compute rule-based risk scores
- Detect safety violations (unauthorized access, data modification, resource abuse)
- Detect ethical violations (harmful content, deception, manipulation, discrimination)

**Key Properties**:
- **P-DET**: Deterministic evaluation (same input → same output)
- **P-TERM**: Terminating evaluation (bounded depth, acyclic graph)
- **P-DATA-MIN**: Context field access restricted per policy whitelist

**Algorithms**:
- Topological sort (Kahn's algorithm or DFS-based)
- Cycle detection (DFS with back-edge detection)
- Risk score aggregation (weighted sum, max, threshold-based)

---

### 3. ML & Anomaly Detection (Optional)
**Purpose**: Enhance rule-based decisions with machine learning and anomaly detection  
**Responsibilities**:
- Shadow mode: Log ML predictions alongside rule-based decisions for validation
- Blended mode: Combine ML risk scores with rule-based scores
- Anomaly detection: Identify unusual action patterns (sequence, behavioral, statistical)
- Drift monitoring: Track distribution changes (PSI, KL divergence)

**Key Properties**:
- Shadow mode ensures ML does not impact production decisions until validated
- Promotion gates require sufficient accuracy/agreement before ML enforcement

**Models**:
- ML Classifier: Binary/multi-class risk prediction
- Anomaly Detector: Isolation Forest, Autoencoders, Statistical thresholds
- Drift Detector: PSI, KL divergence, statistical tests

---

### 4. Safety Judge & Decision
**Purpose**: Make final enforcement decisions with graduated responses  
**Responsibilities**:
- Map risk scores and violations to judgment levels (ALLOW, RESTRICT, BLOCK, TERMINATE)
- Generate human-readable justifications with contributing factors
- Compute decision confidence based on agreement, history, thresholds
- Apply restrictions (e.g., "redact PII before proceeding")

**Key Properties**:
- **P-JUST**: 100% decisions have complete justifications

**Judgment Logic**:
- **ALLOW**: Risk score below threshold, no violations
- **RESTRICT**: Minor violations; conditions for proceeding (e.g., redaction)
- **BLOCK**: Significant violations; action prevented
- **TERMINATE**: Critical violations; agent execution halted

---

### 5. Audit & Integrity Layer
**Purpose**: Ensure tamper-evident logging and policy lineage  
**Responsibilities**:
- Append all decisions, policy changes, and system events to audit log
- Compute Merkle tree over audit snapshots; store root hash
- Verify audit log integrity (daily automated runs)
- Track policy lineage (version hash chain, approver signatures)
- Aggregate fairness metrics (statistical parity, disparate impact)
- Monitor SLAs (latency, throughput, escalation resolution times)

**Key Properties**:
- **P-AUD**: All events auditable with complete records
- **P-NONREP**: Cryptographically signed snapshots prevent repudiation
- **P-POL-LIN**: Policy version chain verifiable end-to-end
- **P-FAIR-SP**: Fairness metrics computed and monitored

**Data Structures**:
- Merkle tree (binary tree with leaf hashes, internal node = hash(left || right))
- Policy lineage chain (linked list of versions with hash pointers)

---

### 6. Human-in-the-Loop & Optimization
**Purpose**: Enable human oversight and continuous improvement  
**Responsibilities**:
- Escalate high-risk or low-confidence decisions to human review queue
- Collect human feedback (labels, corrections, appeal outcomes)
- Optimize thresholds and weights via multi-objective optimization
- A/B test configurations before promoting to production
- Promotion gates: Require metrics (accuracy, fairness, SLO compliance) to promote

**Key Properties**:
- **P-APPEAL**: Appeals reproducibly re-evaluate with original context

**Workflows**:
- Auto-escalation triggers (risk > threshold, confidence < threshold, BLOCK/TERMINATE)
- SLA tracking (median resolution time < 72h for appeals)
- Feedback loop: Human labels → retrain models → promote if passing gates

---

### 7. External Integrations & Outputs
**Purpose**: Export data, enable transparency, and integrate with infrastructure  
**Responsibilities**:
- **Observability**: Export metrics (risk scores, violations, latency) to Prometheus/Grafana via OpenTelemetry
- **Audit Portal**: Web UI to explore decisions, lineage, justifications
- **Appeals CLI**: Command-line tool for reproducible decision re-evaluation
- **Alerting**: Notify on threshold breaches, SLO violations, incidents
- **DSR API**: Data subject rights (access, deletion, rectification)

**Key Properties**:
- **P-JUST**: Portal displays complete decision justifications
- **P-APPEAL**: CLI reproduces exact decisions with diff artifacts

---

## Data Model

### Core Entities

#### Action
- **agent_id**: Unique identifier for the AI agent
- **action_id**: Unique identifier for this action
- **action_type**: Type of action (e.g., "response", "api_call", "data_access")
- **timestamp**: When the action was performed
- **context**: Dictionary of contextual fields (user input, environment, history, etc.)
- **cohort**: Logical grouping (e.g., "production", "staging", "beta")
- **tenant_id**: Multi-tenant identifier (for isolation)

#### Decision
- **action_id**: References the evaluated action
- **judgment**: One of {ALLOW, RESTRICT, BLOCK, TERMINATE}
- **risk_score**: Quantitative risk (0.0 to 1.0)
- **violations**: List of detected violations (type, severity, description)
- **justification**: Human-readable explanation with contributing factors
- **confidence**: Decision confidence (0.0 to 1.0)
- **restrictions**: Conditions if RESTRICT (e.g., "redact PII fields [email, phone]")
- **timestamp**: Decision time
- **policy_versions**: List of policy versions used in evaluation

#### Policy
- **policy_id**: Unique identifier
- **version_hash**: Cryptographic hash of policy content (SHA-256)
- **dependencies**: List of policy_ids this policy depends on
- **context_whitelist**: Allowed context fields for P-DATA-MIN
- **activation_status**: {active, quarantine, inactive}
- **approvers**: List of signatures for multi-sig activation
- **activation_timestamp**: When activated
- **deactivation_timestamp**: When deactivated (if applicable)

#### Audit Event
- **event_id**: Unique identifier
- **event_type**: Type (decision, policy_change, escalation, etc.)
- **timestamp**: Event time
- **actor**: Who/what caused the event (agent_id, admin_id, system)
- **payload**: Event-specific data (e.g., Decision object for decision events)
- **merkle_path**: Path from leaf to Merkle root for verification

#### Fairness Metric
- **metric_type**: Type (statistical_parity, disparate_impact_ratio, etc.)
- **protected_attribute**: Attribute being analyzed (race, gender, age, etc.)
- **reference_group**: Baseline group for comparison
- **protected_group**: Group being evaluated
- **value**: Metric value (e.g., SP difference = 0.05, ratio = 0.85)
- **threshold**: Acceptable threshold
- **compliant**: Boolean indicating if within threshold
- **timestamp**: When computed
- **sample_size**: Number of decisions analyzed

---

## Operational Modes

### Production Mode
- All features enabled: rule-based + ML blending (if promoted), anomaly detection, fairness monitoring
- Decisions enforced in real-time
- Audit logs continuously written
- Escalations to human reviewers

### Shadow Mode (ML Validation)
- ML predictions logged but not enforced
- Rule-based decisions remain authoritative
- Comparison metrics tracked (agreement rate, false positives/negatives)
- Used to validate ML models before promotion

### Quarantine Mode (Policy Testing)
- Policies loaded but not enforced
- Allows testing new policies without production impact
- Logged as "would-have" decisions
- Used during policy development and staged rollouts

### Dry-Run Mode (Pre-Deployment)
- No enforcement; all decisions logged as informational
- Used for impact assessment before activating governance

---

## Non-Functional Characteristics

### Performance
- **Latency**: p95 < 200ms, p99 < 500ms (per R-NF001)
- **Throughput**: 100-1000 RPS depending on configuration (per R-NF002)
- **Scalability**: Horizontal scaling via stateless design + regional deployments

### Reliability
- **Availability**: 99.9% uptime target (per R-NF003)
- **Durability**: Audit logs replicated, <1h RPO (per R-NF005)
- **Graceful Degradation**: Fall back to rule-based if ML fails (per R-NF004)

### Security
- **Authentication**: RBAC with SSO/SAML + MFA (per R-NF006)
- **Encryption**: TLS in transit, at-rest encryption (infrastructure)
- **Audit Integrity**: Merkle anchoring prevents tampering (per R-F005)

### Compliance
- **GDPR/CCPA**: PII redaction, data minimization, RTBF support (per G-003)
- **Fairness**: Statistical parity monitoring, protected attributes (per G-004)
- **Transparency**: Decision justifications, audit portal (per G-005)

---

## Deployment Topology

### Single-Node Deployment (Dev, Small Prod)
- All components in one process/container
- SQLite or local file storage
- Suitable for 100-200 RPS

### Multi-Node Deployment (Medium Prod)
- Stateless service instances behind load balancer
- Shared database (PostgreSQL, MySQL)
- Redis for caching and session state
- Suitable for 300-500 RPS

### Multi-Region Deployment (Large Prod)
- Regional instances with data residency compliance
- Regional databases with eventual consistency or sharding
- Global load balancer with geo-routing
- Suitable for 500-1000+ RPS

### Kubernetes / Cloud-Native
- Deployment manifests for K8s
- Horizontal pod autoscaling based on CPU/memory/custom metrics
- Persistent volumes for audit logs
- Service mesh for observability and mTLS

---

## Integration Patterns

### Synchronous (Real-Time)
- AI agent calls Nethical API for each action
- Nethical returns decision before agent proceeds
- Ensures enforcement at decision point
- Latency-sensitive; requires fast evaluation

### Asynchronous (Post-Facto)
- AI agent performs action, logs to queue
- Nethical processes actions from queue
- Decisions logged for audit; may trigger alerts or remediation
- Lower latency impact on agent; weaker enforcement

### Hybrid (Risk-Based)
- High-risk actions evaluated synchronously (BLOCK before proceeding)
- Low-risk actions evaluated asynchronously (audit-only)
- Balances enforcement strength and performance

---

## Extension Points

### Custom Detectors
- Pluggable violation detectors (safety, ethical, domain-specific)
- Implement interface: `detect(action, context) -> List[Violation]`

### Custom Risk Scorers
- Pluggable risk scoring algorithms
- Implement interface: `score(action, violations, history) -> float`

### Custom ML Models
- Bring your own model for risk prediction, anomaly detection
- Integration via standard API (e.g., REST, gRPC) or in-process

### Plugin Marketplace (F6)
- Community-contributed extensions (detectors, scorers, integrations)
- Trust scoring and review system
- Load via `governance.load_plugin(plugin_id)`

---

## Observability & Monitoring

### Metrics (OpenTelemetry + Prometheus)
- Actions processed per second (gauge, counter)
- Risk score distribution (histogram)
- Violations by type and severity (counter)
- Judgment distribution (counter: ALLOW, RESTRICT, BLOCK, TERMINATE)
- Latency percentiles (histogram: p50, p95, p99)
- Quota utilization (gauge)
- Fairness metrics (gauge)

### Traces (OpenTelemetry)
- End-to-end action evaluation trace
- Spans: ingestion, policy evaluation, ML inference, decision, audit log write

### Logs (Structured JSON)
- Decision logs (action_id, judgment, risk_score, violations, justification)
- Policy change logs (policy_id, version, activator, timestamp)
- Incident logs (alerts, escalations, errors)

### Dashboards (Grafana)
- Real-time request rates and latencies
- Violation heatmaps
- Risk score trends
- Fairness metric trends
- SLA compliance (latency, appeals resolution time)

---

## Security Model

### Trust Boundaries
- **External**: AI agents (untrusted; may be adversarial)
- **Internal**: Nethical governance platform (trusted execution environment)
- **Administrative**: Operators/admins (trusted personnel; separation of duties)
- **External Services**: SSO provider, timestamping service, ML model registry (trusted third parties)

### Threat Mitigation
- **Adversarial Inputs**: Prompt injection detection, input validation, anomaly detection
- **Insider Threats**: RBAC, multi-sig approvals, audit logging
- **Tampering**: Merkle anchoring, append-only logs, cryptographic signatures
- **DoS**: Quota enforcement, backpressure, rate limiting
- **Supply Chain**: SBOM, dependency pinning, reproducible builds

---

## State Management

### Stateless Components (Horizontally Scalable)
- Ingestion & preprocessing (context enrichment from read-only cache)
- Policy evaluation engine (policies pre-loaded; read-only during evaluation)
- Safety judge (stateless decision logic)

### Stateful Components (Requires Coordination)
- Audit log writer (append-only; coordinated via database transactions or distributed log)
- Fairness metrics aggregator (batch processing; eventual consistency acceptable)
- ML model state (trained models versioned and immutable once deployed)

### Caching Strategy
- Policy cache: Loaded policies cached in memory; TTL or event-driven invalidation
- Agent history cache: Recent actions cached (Redis) for quick lookup
- Context enrichment cache: Static metadata cached (e.g., tenant config, cohort info)

---

## Failure Modes & Recovery

### Component Failures
- **Database unavailable**: Buffer audit logs in memory; flush when reconnected; alert on buffer full
- **ML service down**: Fall back to rule-based only; log ML unavailable event
- **External service timeout** (SSO, timestamping): Use cached credentials or local timestamp; degrade gracefully

### Data Corruption
- **Audit log corruption**: Merkle verification detects; restore from backup; investigate incident
- **Policy corruption**: Version hash mismatch detected; reject corrupted policy; revert to last known good

### Configuration Errors
- **Invalid policy**: Cycle detection rejects; validation errors logged; admin notified
- **Misconfigured thresholds**: Monitor for anomalous decision rates; rollback if detected

---

## Success Criteria (Phase 2A)

Phase 2A (Core Informal Spec) is complete when:
1. ✅ All critical flows described (action ingestion → decision → audit)
2. ✅ System architecture diagram reviewed by stakeholders
3. ✅ Data model entities and relationships documented
4. ✅ Key properties (P-DET, P-TERM, P-DATA-MIN, etc.) mapped to components
5. ✅ Non-functional characteristics (performance, reliability, security) specified
6. ✅ Integration patterns and deployment topologies described
7. ✅ Failure modes and recovery procedures documented

---

## Related Documents
- state-model.md: Detailed state machine for policy and decision lifecycle
- transitions.md: State transition specifications
- api-contracts.md: API endpoint specifications
- policy_lineage.md: Policy lifecycle and lineage design (Phase 2B)
- fairness_metrics.md: Fairness criteria baseline (Phase 2C)
- requirements.md: Functional and non-functional requirements
- risk_register.md: Risks addressed by this architecture

---

**Status**: ✅ Phase 2A Deliverable - COMPLETE  
**Last Updated**: 2025-11-16  
**Owner**: Tech Lead / System Architect
