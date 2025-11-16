# System Assumptions

## Overview
This document explicitly states assumptions underlying the Nethical design, implementation, and operation. Violations of these assumptions may invalidate correctness guarantees and require re-evaluation of the system.

---

## Assumption Categories
- **Environmental**: External system and infrastructure assumptions
- **Operational**: Deployment and usage context
- **Technical**: Implementation and algorithm assumptions
- **Threat Model**: Security boundary and attacker capability assumptions
- **Regulatory**: Compliance and legal context

---

## Environmental Assumptions

### A-ENV-001: Trusted Execution Environment
- **Assumption**: The operating system and hardware where Nethical runs are not compromised by root-level malware.
- **Rationale**: Formal proofs and security controls cannot defend against OS-level compromise.
- **Mitigation**: Use hardened OS, security patching, host-based intrusion detection.
- **Impact if Violated**: Complete system compromise; audit logs may be tampered with.

---

### A-ENV-002: Reliable Storage Backend
- **Assumption**: The storage backend (filesystem, database, object store) provides durability guarantees and does not silently corrupt data.
- **Rationale**: Audit integrity depends on storage layer correctness.
- **Mitigation**: Use checksums, ECC storage, redundancy (RAID, replication).
- **Impact if Violated**: Audit logs may be lost or corrupted; P-AUD property violated.

---

### A-ENV-003: Network Reliability
- **Assumption**: Network communication between components (e.g., Nethical service, database, external services) is eventually reliable or failures are detectable.
- **Rationale**: Distributed system assumptions (timeouts, retries).
- **Mitigation**: Use timeout handling, exponential backoff, circuit breakers.
- **Impact if Violated**: Transient unavailability; potential data loss if retries fail.

---

### A-ENV-004: Time Synchronization
- **Assumption**: System clocks are reasonably synchronized (e.g., via NTP) with <1s skew.
- **Rationale**: Audit log ordering and timeout enforcement depend on monotonic time.
- **Mitigation**: Use NTP, monitor clock skew, prefer monotonic clocks for local ordering.
- **Impact if Violated**: Audit log timestamps may be incorrect; SLA measurements may be skewed.

---

### A-ENV-005: Cryptographic Primitives Secure
- **Assumption**: Cryptographic functions (SHA-256, AES, RSA, ECDSA, post-quantum algorithms) are computationally secure against current attacks.
- **Rationale**: Merkle anchoring, signatures, and encryption depend on crypto strength.
- **Mitigation**: Use NIST-approved algorithms, monitor for cryptanalytic breakthroughs.
- **Impact if Violated**: Audit integrity and non-repudiation may be compromised; privacy violations possible.

---

## Operational Assumptions

### A-OPS-001: Authorized Operators
- **Assumption**: System operators (admins, DevOps) with privileged access are trustworthy and follow security policies.
- **Rationale**: Insider threats are out of scope for technical controls; rely on organizational controls.
- **Mitigation**: Background checks, separation of duties, audit logging of admin actions, multi-party approval for critical operations.
- **Impact if Violated**: Insider may tamper with policies, audit logs, or configurations; governance controls bypassed.

---

### A-OPS-002: Responsible AI Agent Deployment
- **Assumption**: AI agents monitored by Nethical are deployed in good faith by responsible parties who intend to comply with policies.
- **Rationale**: Nethical is a safety/governance layer, not a foolproof prevention system against determined adversaries.
- **Mitigation**: Legal agreements, agent vetting, continuous monitoring, adversarial testing.
- **Impact if Violated**: Adversarial agents may attempt to evade detection; R-007 mitigations required.

---

### A-OPS-003: Adequate Resources Provisioned
- **Assumption**: Deployment environments provide sufficient compute, memory, storage, and network bandwidth to meet SLOs.
- **Rationale**: Performance requirements (R-NF001, R-NF002) depend on appropriate sizing.
- **Mitigation**: Capacity planning, load testing, auto-scaling, monitoring.
- **Impact if Violated**: Latency SLOs violated; potential cascading failures or service degradation.

---

### A-OPS-004: Human Oversight Availability
- **Assumption**: Human reviewers are available to respond to escalations within SLA timeframes (<72h for appeals, <24h for critical incidents).
- **Rationale**: Human-in-the-loop workflows depend on human capacity.
- **Mitigation**: Staffing plans, on-call rotations, escalation procedures.
- **Impact if Violated**: Appeals SLA (R-014) violated; governance quality degraded.

---

### A-OPS-005: Backup and Disaster Recovery Procedures
- **Assumption**: Regular backups are performed, tested, and restorable within Recovery Time Objective (RTO) and Recovery Point Objective (RPO) targets.
- **Rationale**: Data durability (R-NF005) depends on backup processes.
- **Mitigation**: Automated backups, DR drills, off-site replication.
- **Impact if Violated**: Data loss in disaster scenarios; audit trails may be unrecoverable.

---

## Technical Assumptions

### A-TECH-001: Single-Node Serialization (Phase 1-2)
- **Assumption**: Early phases assume single-node deployment with sequential policy evaluation (no concurrency).
- **Rationale**: Simplifies determinism proofs (P-DET) in Phase 3A.
- **Mitigation**: Document concurrency model in Phase 5 for multi-node deployments; extend proofs for distributed case.
- **Impact if Violated**: Determinism may be violated in distributed deployments without careful coordination.

---

### A-TECH-002: Policy Graph Acyclicity
- **Assumption**: Policy dependency graphs are acyclic (DAG property).
- **Rationale**: Termination guarantee (P-TERM) depends on acyclicity.
- **Mitigation**: Cycle detection at policy load time (R-F002); reject policies with cycles.
- **Impact if Violated**: Infinite loops; evaluation hangs; R-002 risk realized.

---

### A-TECH-003: Bounded Evaluation Depth
- **Assumption**: Policy evaluation has a finite maximum depth (e.g., 100 levels of dependency nesting).
- **Rationale**: Prevents stack overflow and ensures termination.
- **Mitigation**: Enforce depth limit in policy engine; configurable upper bound.
- **Impact if Violated**: Stack overflow crash; denial of service.

---

### A-TECH-004: ML Model Validity
- **Assumption**: Trained ML models generalize to production data distribution and do not exhibit catastrophic failure modes.
- **Rationale**: ML blending (R-F014) depends on model quality.
- **Mitigation**: Shadow mode (Phase 5), promotion gates, continuous monitoring, drift detection (R-F015).
- **Impact if Violated**: High false positive/negative rates; governance quality degraded; potential safety violations.

---

### A-TECH-005: Context Completeness
- **Assumption**: Action context provided to Nethical includes all necessary fields for policy evaluation.
- **Rationale**: Incomplete context may lead to incorrect decisions.
- **Mitigation**: Context schema validation, mandatory field checks, warnings for missing optional fields.
- **Impact if Violated**: False negatives (ALLOW when should BLOCK); safety violations.

---

### A-TECH-006: Immutable Policy Semantics
- **Assumption**: Once a policy version is activated, its semantics do not change (no in-place modification).
- **Rationale**: Lineage integrity (P-POL-LIN) and reproducibility (P-APPEAL) depend on immutability.
- **Mitigation**: Version control, hash-based identification, immutable storage.
- **Impact if Violated**: Contestability broken; audit trails unreliable.

---

### A-TECH-007: No Integer Overflow
- **Assumption**: Risk scores, counters, and numeric computations do not overflow.
- **Rationale**: Arithmetic correctness for decision logic.
- **Mitigation**: Use appropriate data types (float64, int64), range checks.
- **Impact if Violated**: Incorrect risk scores; potential security vulnerabilities.

---

### A-TECH-008: Dependencies Pinned
- **Assumption**: All dependencies (Python packages, libraries) are pinned to specific versions with hash verification.
- **Rationale**: Reproducible builds (R-NF007) and supply chain security (R-010).
- **Mitigation**: requirements-hashed.txt, SBOM generation, dependency scanning.
- **Impact if Violated**: Non-reproducible builds; supply chain attacks possible.

---

## Threat Model Assumptions

### A-THREAT-001: Adversary Capability Bounded
- **Assumption**: Adversaries have limited computational resources and cannot break cryptographic primitives (SHA-256, AES-256, RSA-2048, post-quantum algorithms).
- **Rationale**: Security proofs depend on computational hardness assumptions.
- **Mitigation**: Use NIST-approved crypto, key rotation, monitor crypto research.
- **Impact if Violated**: Crypto-based security controls (signatures, hashes) compromised.

---

### A-THREAT-002: No Insider with Root Access
- **Assumption**: Adversaries do not have root/admin access to production systems.
- **Rationale**: Root access bypasses all technical controls.
- **Mitigation**: Privileged access management, separation of duties, audit logging.
- **Impact if Violated**: Complete system compromise; assume breach scenario.

---

### A-THREAT-003: External Services Trustworthy
- **Assumption**: External services (timestamping, ML model registries, identity providers) are trustworthy and not compromised.
- **Rationale**: System depends on external trust anchors.
- **Mitigation**: Vendor security assessments, SLAs, redundancy, fallback mechanisms.
- **Impact if Violated**: Lineage integrity (timestamping), authentication (SSO), model poisoning risks.

---

### A-THREAT-004: Rate Limiting Effective
- **Assumption**: Quota enforcement and rate limiting mechanisms are sufficient to prevent denial-of-service attacks.
- **Rationale**: Availability (R-NF003) depends on resource controls.
- **Mitigation**: Quota system (existing), backpressure, DDoS protection (CDN, WAF).
- **Impact if Violated**: Service overload; availability SLOs violated.

---

### A-THREAT-005: No Side-Channel Attacks
- **Assumption**: Side-channel attacks (timing, power analysis, cache) are out of scope or mitigated at infrastructure level.
- **Rationale**: Nethical is not a cryptographic implementation; focus on application-layer security.
- **Mitigation**: Use constant-time crypto libraries where applicable, secure hardware (HSM) for keys.
- **Impact if Violated**: Potential key leakage; privacy violations.

---

### A-THREAT-006: Prompt Injection Detectable
- **Assumption**: Adversarial prompts (injection, jailbreak) have detectable patterns or anomalies.
- **Rationale**: R-F013 (adversarial detection) depends on heuristics and ML models.
- **Mitigation**: Adversarial testing (36 scenarios), continuous threat intelligence, red-team exercises.
- **Impact if Violated**: False negatives; adversarial inputs bypass detection.

---

## Regulatory Assumptions

### A-REG-001: Applicable Regulations Identified
- **Assumption**: All applicable regulations (GDPR, CCPA, EU AI Act, HIPAA, etc.) are correctly identified for the deployment context.
- **Rationale**: Compliance matrix (G-001) depends on accurate regulatory scoping.
- **Mitigation**: Legal counsel review, jurisdiction analysis, compliance audits.
- **Impact if Violated**: Non-compliance; regulatory penalties; reputational damage.

---

### A-REG-002: Protected Attributes Defined
- **Assumption**: Protected attributes for fairness analysis (race, gender, age, etc.) are correctly identified and relevant to the domain.
- **Rationale**: Fairness monitoring (R-F008) depends on attribute selection.
- **Mitigation**: Legal/ethics review, domain expert consultation, governance_drivers.md documentation.
- **Impact if Violated**: Incorrect fairness analysis; missed discrimination; regulatory violations.

---

### A-REG-003: Data Retention Policies Known
- **Assumption**: Data retention requirements (minimum and maximum) are known and documented for each data type.
- **Rationale**: GDPR/CCPA compliance requires retention limits; RTBF support.
- **Mitigation**: Legal review, data classification, retention policy documentation.
- **Impact if Violated**: Over-retention (privacy violations) or under-retention (compliance/audit failures).

---

### A-REG-004: Cross-Border Data Transfer Permitted
- **Assumption**: Cross-border data transfers (if any) are legally compliant with data residency and transfer mechanism requirements (e.g., EU Standard Contractual Clauses).
- **Rationale**: GDPR, CCPA, and other laws restrict international data transfers.
- **Mitigation**: Data residency policies (existing), regional deployments, legal agreements.
- **Impact if Violated**: Regulatory violations; data transfer injunctions.

---

### A-REG-005: Fairness Thresholds Legally Adequate
- **Assumption**: Fairness thresholds (e.g., statistical parity ≤0.10, disparate impact ratio ≥0.80) meet or exceed legal requirements.
- **Rationale**: Compliance with ECOA, FHA, EEOC guidelines.
- **Mitigation**: Legal review, case law analysis, industry standards.
- **Impact if Violated**: Fairness criteria too lenient; regulatory non-compliance; discrimination lawsuits.

---

## Data Assumptions

### A-DATA-001: Training Data Representativeness
- **Assumption**: ML training datasets are representative of production data distribution and include diverse populations.
- **Rationale**: Model generalization and fairness depend on training data quality.
- **Mitigation**: Data audits, diversity checks, synthetic data augmentation, drift monitoring.
- **Impact if Violated**: High false positives/negatives; fairness violations (R-004 risk).

---

### A-DATA-002: Ground Truth Accuracy
- **Assumption**: Human-labeled training data (ground truth) is accurate and unbiased.
- **Rationale**: Model quality depends on label quality.
- **Mitigation**: Inter-rater agreement checks, labeling guidelines, audits.
- **Impact if Violated**: Model bias; poor performance; fairness violations.

---

### A-DATA-003: Feature Engineering Validity
- **Assumption**: Engineered features (risk scores, derived metrics) capture relevant information without introducing bias or data leakage.
- **Rationale**: ML model fairness and accuracy depend on feature quality.
- **Mitigation**: Feature importance analysis, fairness audits, correlation checks.
- **Impact if Violated**: Biased models; proxy discrimination via correlated features.

---

### A-DATA-004: PII Detection Completeness
- **Assumption**: PII detection patterns (regex, ML models) cover ≥95% of real-world PII instances in relevant languages/formats.
- **Rationale**: Privacy compliance (R-F011) depends on detection accuracy.
- **Mitigation**: Comprehensive test datasets, continuous pattern updates, precision/recall monitoring.
- **Impact if Violated**: PII leakage; privacy violations; regulatory penalties.

---

## Formal Methods Assumptions

### A-FORMAL-001: Model Fidelity
- **Assumption**: Formal models (TLA+, Lean, Alloy) accurately represent the implemented system behavior.
- **Rationale**: Proofs are only as valid as the model's correspondence to reality.
- **Mitigation**: Model reviews, refinement mapping, runtime invariant checks (probes), manual inspection.
- **Impact if Violated**: False confidence in correctness; bugs in unmodeled behaviors.

---

### A-FORMAL-002: Property Completeness
- **Assumption**: Formalized properties (P-DET, P-TERM, P-AUD, etc.) comprehensively capture critical correctness requirements.
- **Rationale**: Unspecified properties cannot be proved.
- **Mitigation**: Requirements review, property brainstorming sessions, coverage tracking.
- **Impact if Violated**: Correctness gaps; unproven critical behaviors.

---

### A-FORMAL-003: Admitted Lemmas Justified
- **Assumption**: Lemmas with "admitted" (unproven) status are either non-critical or have planned proof schedules.
- **Rationale**: Admitted critical lemmas undermine assurance claims.
- **Mitigation**: Zero critical admits policy, proof debt tracking, burn-down sprints.
- **Impact if Violated**: Unproven critical properties; potential bugs.

---

### A-FORMAL-004: Tool Soundness
- **Assumption**: Formal verification tools (TLA+ TLC model checker, Lean proof assistant) are sound (no false proofs).
- **Rationale**: Trust in verification depends on tool correctness.
- **Mitigation**: Use mature, well-audited tools; cross-verify critical properties with multiple tools.
- **Impact if Violated**: False confidence from unsound proofs.

---

## Assumptions Review and Validation

### Review Cadence
- **Initial (Phase 1)**: Document all assumptions
- **Quarterly**: Review assumptions for validity; update as system evolves
- **Phase Transitions**: Reassess assumptions when entering new phases (e.g., concurrency in Phase 5)
- **Incident-Triggered**: Review assumptions if violated or incidents occur

### Assumption Violation Protocol
1. **Detection**: Monitoring, audits, or incidents identify assumption violation
2. **Impact Assessment**: Evaluate which guarantees are compromised
3. **Escalation**: Notify stakeholders per severity (immediate for CRITICAL)
4. **Mitigation**: Implement workarounds or system changes
5. **Documentation**: Update assumptions.md and related docs

---

## Assumptions Summary

| ID | Category | Severity if Violated | Mitigation Status |
|----|----------|---------------------|-------------------|
| A-ENV-001 | Environmental | CRITICAL | Operational controls |
| A-ENV-002 | Environmental | CRITICAL | Storage redundancy |
| A-ENV-003 | Environmental | MEDIUM | Network resilience |
| A-ENV-004 | Environmental | LOW | NTP monitoring |
| A-ENV-005 | Environmental | CRITICAL | Crypto standards |
| A-OPS-001 | Operational | CRITICAL | HR/organizational |
| A-OPS-002 | Operational | HIGH | Legal/monitoring |
| A-OPS-003 | Operational | HIGH | Capacity planning |
| A-OPS-004 | Operational | MEDIUM | Staffing plans |
| A-OPS-005 | Operational | HIGH | Backup procedures |
| A-TECH-001 | Technical | HIGH | Concurrency design (Phase 5) |
| A-TECH-002 | Technical | CRITICAL | Cycle detection |
| A-TECH-003 | Technical | MEDIUM | Depth limits |
| A-TECH-004 | Technical | HIGH | Shadow mode, gates |
| A-TECH-005 | Technical | HIGH | Schema validation |
| A-TECH-006 | Technical | CRITICAL | Immutable storage |
| A-TECH-007 | Technical | MEDIUM | Range checks |
| A-TECH-008 | Technical | HIGH | Dependency pinning |
| A-THREAT-001 | Threat Model | CRITICAL | Crypto standards |
| A-THREAT-002 | Threat Model | CRITICAL | Access controls |
| A-THREAT-003 | Threat Model | MEDIUM | Vendor assessments |
| A-THREAT-004 | Threat Model | MEDIUM | Quota enforcement |
| A-THREAT-005 | Threat Model | LOW | Infrastructure controls |
| A-THREAT-006 | Threat Model | MEDIUM | Adversarial testing |
| A-REG-001 | Regulatory | CRITICAL | Legal review |
| A-REG-002 | Regulatory | HIGH | Ethics review |
| A-REG-003 | Regulatory | HIGH | Legal review |
| A-REG-004 | Regulatory | CRITICAL | Regional deployments |
| A-REG-005 | Regulatory | HIGH | Legal standards |
| A-DATA-001 | Data | HIGH | Data audits |
| A-DATA-002 | Data | HIGH | Labeling QA |
| A-DATA-003 | Data | MEDIUM | Feature audits |
| A-DATA-004 | Data | HIGH | Pattern updates |
| A-FORMAL-001 | Formal Methods | HIGH | Model reviews |
| A-FORMAL-002 | Formal Methods | HIGH | Property reviews |
| A-FORMAL-003 | Formal Methods | CRITICAL | Zero admits policy |
| A-FORMAL-004 | Formal Methods | LOW | Tool selection |

---

## Related Documents
- requirements.md: Requirements that depend on these assumptions
- risk_register.md: Risks related to assumption violations
- compliance_matrix.md: Regulatory assumptions
- core_model.tla: Formal model assumptions

---

**Status**: ✅ Phase 1A Deliverable - COMPLETE  
**Last Updated**: 2025-11-16  
**Owner**: Tech Lead / Formal Methods Engineer
