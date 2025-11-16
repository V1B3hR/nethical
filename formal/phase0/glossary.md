# Glossary

## Overview
This glossary provides unified terminology for the Nethical governance platform. Consistent vocabulary is essential for clear communication across technical, governance, and compliance domains.

---

## Core Concepts

### Agent
An autonomous AI system whose actions are monitored and evaluated by Nethical. Agents may be chatbots, decision systems, recommendation engines, or other AI-driven software components.

### Action
A discrete operation or decision performed by an agent. Actions are the primary unit of evaluation in Nethical.

### Policy
A rule or set of rules defining acceptable behavior, ethical constraints, or safety requirements. Policies are versioned, auditable, and subject to governance controls.

### Decision
The outcome of policy evaluation for a specific action. Decisions include risk assessment, judgment (ALLOW/RESTRICT/BLOCK/TERMINATE), and justification.

### Governance
The system of controls, processes, and oversight ensuring that AI operations are safe, ethical, transparent, and compliant with regulations.

---

## Judgment Categories

### ALLOW
Decision indicating the action is safe and compliant; no restrictions necessary.

### RESTRICT
Decision indicating the action requires constraints or modifications before proceeding (e.g., redact PII, apply rate limits).

### BLOCK
Decision indicating the action violates policies or safety constraints; action must be prevented.

### TERMINATE
Decision indicating severe violation requiring immediate termination of agent execution and escalation.

---

## Risk & Safety

### Risk Score
Quantitative measure (0.0 to 1.0) indicating potential harm or policy violation severity for an action. Higher scores represent greater risk.

### Violation
An action that breaches a policy constraint. Violations are categorized by type (ethical, safety, privacy, etc.) and severity (low, medium, high, critical).

### Anomaly
An action exhibiting unusual patterns compared to historical behavior, detected via statistical or ML-based methods.

### Drift
Gradual change in data distribution or decision patterns over time, potentially indicating model degradation or environmental changes.

---

## Governance Concepts

### Protected Attribute
A characteristic used in fairness analysis (e.g., race, gender, age). Decisions must not exhibit disparate impact across protected attribute groups.

### Statistical Parity
A fairness metric measuring whether decision rates are similar across groups defined by protected attributes.

### Counterfactual Fairness
A fairness criterion requiring that a decision would remain the same if a protected attribute were changed, all else equal.

### Disparate Impact
Unequal treatment or outcomes affecting a protected group disproportionately. Measured via statistical parity difference or ratio.

### Lineage
The traceable history of policy versions and their approvals, forming a verifiable chain of custody.

### Contestability
The ability for affected parties to challenge a decision and receive a transparent, reproducible re-evaluation.

---

## Audit & Integrity

### Audit Log
An immutable, timestamped record of all decisions, policy changes, and system events. Essential for accountability and forensic analysis.

### Merkle Tree / Merkle Root
A cryptographic data structure enabling tamper detection. Each audit snapshot is represented by a Merkle root hash; any modification invalidates the hash.

### Non-Repudiation
A property ensuring that recorded events cannot be denied or altered retroactively. Achieved via cryptographic signing and Merkle anchoring.

### Append-Only
A storage model where records can only be added, never modified or deleted. Ensures audit trail integrity.

### Provenance
The origin and history of an artifact (e.g., policy, model, decision), including authorship, approvals, and changes.

---

## Formal Methods

### Invariant
A property that must hold true at all times or in all reachable states of the system (e.g., "no cycles in policy graph").

### Property
A formal specification of desired system behavior (e.g., "all decisions are deterministic"). Properties are verified via proofs or model checking.

### Proof
A formal mathematical argument demonstrating that a property holds. Proofs may be mechanized in tools like TLA+, Lean, or Dafny.

### Model Checking
Automated verification technique that exhaustively explores system states to verify properties or find counterexamples.

### Admitted Lemma
A proof placeholder indicating that a property is assumed true but not yet formally verified. Critical lemmas should have zero admits.

---

## Policy Lifecycle

### Policy Version
A specific iteration of a policy, identified by a unique hash or version number. Each version is immutable once approved.

### Policy Activation
The process of enabling a policy version in production. Requires multi-signature approval for critical policies.

### Multi-Signature (Multi-Sig)
An approval mechanism requiring k distinct authorized signatures before a policy can be activated.

### Quarantine Mode
A safe state where a policy is loaded but not enforced, allowing testing without production impact.

### Policy Diff
A structured comparison between two policy versions, highlighting additions, deletions, and modifications.

---

## Privacy & Data

### Data Minimization
A principle requiring that only necessary data fields are accessed or stored for a given purpose. Enforced via whitelists and runtime checks.

### Differential Privacy
A mathematical privacy guarantee ensuring that individual records cannot be distinguished in aggregate statistics. Controlled by epsilon (ε) parameter.

### Redaction
The process of removing or masking sensitive information (e.g., PII) from data before storage or transmission.

### PII (Personally Identifiable Information)
Data that can identify an individual (e.g., name, email, SSN, IP address). Subject to privacy regulations like GDPR and CCPA.

### Right to Be Forgotten (RTBF)
A data subject right to request deletion of personal data. Requires special handling in audit logs and analytics.

### Data Residency
The requirement that data be stored and processed within specific geographic regions to comply with local regulations.

---

## Performance & Scalability

### RPS (Requests Per Second)
Throughput measure indicating the number of actions processed per second.

### Latency
Time elapsed from action submission to decision delivery. Key percentiles: p50 (median), p95, p99.

### SLO (Service Level Objective)
A measurable target for a service quality metric (e.g., "p95 latency < 200ms").

### SLA (Service Level Agreement)
A contractual commitment to meet specific SLOs. Violations may trigger penalties or escalations.

### Backpressure
A flow control mechanism that slows or rejects incoming requests when system capacity is exceeded.

### Quota
A limit on resource consumption (e.g., actions per second per agent or tenant). Enforced to prevent abuse and ensure fairness.

---

## Compliance & Regulation

### GDPR (General Data Protection Regulation)
EU regulation governing personal data privacy, including rights to access, rectification, and deletion.

### CCPA (California Consumer Privacy Act)
California law providing consumers with rights over their personal data.

### NIST AI RMF (AI Risk Management Framework)
Framework for managing risks associated with AI systems. Includes GOVERN, MAP, MEASURE, MANAGE functions.

### OWASP LLM Top 10
A list of top security risks for large language model applications (e.g., prompt injection, data poisoning).

### HIPAA (Health Insurance Portability and Accountability Act)
US regulation governing protected health information (PHI) privacy and security.

### FedRAMP (Federal Risk and Authorization Management Program)
US government program standardizing security assessment for cloud services.

---

## ML & Anomaly Detection

### ML Classifier
A machine learning model trained to predict outcomes (e.g., risk score, anomaly label) based on action features.

### Shadow Mode
An evaluation mode where ML predictions are logged but not enforced, allowing validation against rule-based systems.

### Blended Risk Score
A composite risk score combining rule-based and ML-based assessments, typically weighted by confidence and performance metrics.

### Anomaly Detection
The process of identifying actions that deviate significantly from normal behavior patterns.

### Drift Monitoring
Continuous tracking of data distribution changes over time, using metrics like PSI (Population Stability Index) or KL divergence.

---

## Adversarial & Security

### Prompt Injection
An attack where malicious input manipulates an LLM to bypass safety constraints or execute unintended actions.

### Jailbreak
An adversarial technique that circumvents an AI system's safety guardrails through carefully crafted prompts or inputs.

### Model Poisoning
An attack that corrupts a machine learning model's training data to induce incorrect behavior.

### Adversarial Example
Input specifically crafted to cause a model to make incorrect predictions or classifications.

### Red Team
A group performing simulated attacks to identify vulnerabilities and test security controls.

---

## Operational Concepts

### Cohort
A logical grouping of agents for monitoring, evaluation, or experimentation (e.g., "production", "staging", "beta").

### Escalation
The process of elevating a decision or event to human review, typically for high-risk or low-confidence cases.

### Human-in-the-Loop (HITL)
A workflow where humans review and provide feedback on system decisions, enabling continuous improvement.

### Feedback Loop
The process of collecting outcomes, evaluating system performance, and adjusting configurations or models.

### Promotion Gate
A quality threshold that must be met before a candidate model or configuration is promoted to production.

---

## Deployment & Operations

### Reproducible Build
A build process that produces bit-for-bit identical artifacts from the same source code, ensuring supply chain integrity.

### SBOM (Software Bill of Materials)
A comprehensive inventory of all software components, dependencies, and versions in a system.

### Signing / Artifact Signing
Cryptographic signing of release artifacts to verify authenticity and integrity.

### Observability
The ability to understand system internal state through logs, metrics, and traces.

### Probe / Runtime Probe
A runtime check that mirrors formal invariants, alerting on violations in production.

---

## Acronyms

- **API**: Application Programming Interface
- **CI/CD**: Continuous Integration / Continuous Deployment
- **CLI**: Command Line Interface
- **DAG**: Directed Acyclic Graph
- **DSR**: Data Subject Request
- **FGSM**: Fast Gradient Sign Method (adversarial attack)
- **IL**: Impact Level (security classification)
- **JSON**: JavaScript Object Notation
- **KPI**: Key Performance Indicator
- **ML**: Machine Learning
- **MVP**: Minimum Viable Product
- **NLP**: Natural Language Processing
- **PII**: Personally Identifiable Information
- **PSI**: Population Stability Index
- **RACI**: Responsible, Accountable, Consulted, Informed
- **RBAC**: Role-Based Access Control
- **SHA**: Secure Hash Algorithm
- **SLA**: Service Level Agreement
- **SLO**: Service Level Objective
- **SBOM**: Software Bill of Materials
- **TLA+**: Temporal Logic of Actions (formal specification language)
- **TTL**: Time To Live

---

## Property Identifiers

These properties are referenced throughout formal specifications and testing:

- **P-DET**: Determinism property (identical inputs → identical outputs)
- **P-TERM**: Termination property (all evaluations complete in bounded time)
- **P-AUD**: Auditability property (all events recorded in tamper-evident log)
- **P-NONREP**: Non-repudiation property (signed records cannot be denied)
- **P-FAIR-SP**: Statistical parity fairness property
- **P-FAIR-CF**: Counterfactual fairness property
- **P-POL-LIN**: Policy lineage integrity property
- **P-MULTI-SIG**: Multi-signature approval property
- **P-APPEAL**: Contestability property (reproducible re-evaluation)
- **P-DATA-MIN**: Data minimization property
- **P-TENANT-ISO**: Tenant isolation property (non-interference)
- **P-JUST**: Justification completeness property
- **P-AUTH**: Authorization boundary property

---

## Related Documents
- risk_register.md: Risks mapped to these concepts
- requirements.md: Functional requirements using this terminology
- state-model.md: System states and transitions
- core_model.tla: Formal specifications referencing properties

---

**Status**: ✅ Phase 0A Deliverable - COMPLETE  
**Last Updated**: 2025-11-16  
**Owner**: Tech Lead / Documentation Lead
