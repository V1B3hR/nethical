# Compliance Matrix

## Overview
This document maps regulatory requirements to Nethical's capabilities, identifying implementation status and gaps. It serves as a blueprint for demonstrating governance-grade compliance across multiple jurisdictions and frameworks.

---

## Compliance Frameworks Covered

1. **GDPR** (General Data Protection Regulation) - EU privacy law
2. **CCPA** (California Consumer Privacy Act) - California privacy law
3. **EU AI Act** - Proposed regulation for high-risk AI systems
4. **NIST AI RMF** (AI Risk Management Framework) - US federal AI guidance
5. **OWASP LLM Top 10** - Security risks for large language models
6. **SOC 2** (System and Organization Controls) - Trust services criteria
7. **ISO 27001** - Information security management
8. **HIPAA** (Health Insurance Portability and Accountability Act) - US healthcare privacy
9. **FedRAMP** (Federal Risk and Authorization Management Program) - US federal cloud security
10. **ECOA / FHA / EEOC** - US anti-discrimination laws (credit, housing, employment)

---

## Compliance Status Legend

- ✅ **COMPLETE**: Capability fully implemented and tested
- 🟡 **PARTIAL**: Capability partially implemented; gaps documented
- 🔴 **GAP**: Requirement not yet addressed; planned for future phase
- 📋 **DOCUMENTATION**: Capability exists; documentation/audit trail needed

---

## 1. GDPR Compliance Matrix

### Article 5: Principles of Processing Personal Data

| Principle | Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-----------|-------------|---------------------|--------|----------|-------------|
| **Lawfulness, Fairness, Transparency** | Processing must be lawful, fair, transparent | Decision justifications (P-JUST), audit logs, audit portal | ✅ COMPLETE | Audit portal (Phase 9B), decision justifications | - |
| **Purpose Limitation** | Data processed only for specified purposes | Data minimization enforcement (P-DATA-MIN) | ✅ COMPLETE | R-F009, context field whitelisting | - |
| **Data Minimization** | Only necessary data collected and processed | P-DATA-MIN property, context access control | ✅ COMPLETE | Phase 4C, runtime enforcement | - |
| **Accuracy** | Data must be accurate and up to date | Context validation, schema checks | 🟡 PARTIAL | Existing validation | No automated data correction |
| **Storage Limitation** | Data retained only as long as necessary | Configurable retention policies, RTBF support | ✅ COMPLETE | Existing F3 features | - |
| **Integrity and Confidentiality** | Secure processing with appropriate safeguards | Encryption, access control, audit logs | ✅ COMPLETE | Existing security features | - |

### Article 12-22: Data Subject Rights

| Right | Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-------|-------------|---------------------|--------|----------|-------------|
| **Right to Information** | Transparent info about processing | Decision justifications, privacy notices | ✅ COMPLETE | Documentation | - |
| **Right of Access** | Data subjects can access their data | DSR automation (existing F3) | ✅ COMPLETE | DSR runbook | - |
| **Right to Rectification** | Correct inaccurate data | Manual correction workflow | 🟡 PARTIAL | Admin tools | No automated self-service |
| **Right to Erasure (RTBF)** | Delete data upon request | RTBF support (existing F3) | ✅ COMPLETE | Data deletion API | - |
| **Right to Restriction** | Temporarily restrict processing | Quarantine mode (Phase 4) | ✅ COMPLETE | Policy quarantine | - |
| **Right to Data Portability** | Export data in machine-readable format | Export utilities (existing F6) | ✅ COMPLETE | JSON/CSV export | - |
| **Right to Object** | Object to processing for certain purposes | Policy override mechanism | 🟡 PARTIAL | Admin override | No self-service objection |
| **Automated Decision-Making** | Right to human review of automated decisions | Human-in-the-loop escalations (Phase 8-9) | ✅ COMPLETE | HITL workflows | - |

### Article 25: Data Protection by Design and Default

| Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-------------|---------------------|--------|----------|-------------|
| Implement appropriate technical and organizational measures | Privacy-by-design architecture, P-DATA-MIN, PII redaction | ✅ COMPLETE | R-F009, R-F011, G-003 | - |
| Data protection by default (minimal data) | Default to minimal context fields, opt-in for additional data | ✅ COMPLETE | Context whitelisting | - |

### Article 30: Records of Processing Activities

| Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-------------|---------------------|--------|----------|-------------|
| Maintain records of processing activities | Audit logs, processing inventory documentation | ✅ COMPLETE | Audit logs, DPIA template | - |

### Article 32: Security of Processing

| Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-------------|---------------------|--------|----------|-------------|
| Encryption of personal data | PII redaction, encryption at rest/transit (infrastructure) | ✅ COMPLETE | Redaction pipeline, TLS | Infrastructure-level encryption |
| Ensure confidentiality, integrity, availability | Access control, Merkle anchoring, redundancy | ✅ COMPLETE | P-AUD, P-AUTH, availability targets | - |
| Regular testing and evaluation | Security scanning, adversarial testing, penetration tests | ✅ COMPLETE | 36 adversarial tests, CI security scans | - |

### Article 33-34: Data Breach Notification

| Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-------------|---------------------|--------|----------|-------------|
| Notify supervisory authority within 72h | Incident response procedures, alerting | 📋 DOCUMENTATION | Incident playbooks | Playbooks exist, need GDPR-specific templates |
| Notify affected individuals if high risk | Incident response procedures | 📋 DOCUMENTATION | Incident playbooks | Need notification templates |

### Article 35: Data Protection Impact Assessment (DPIA)

| Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-------------|---------------------|--------|----------|-------------|
| Conduct DPIA for high-risk processing | DPIA template (existing) | ✅ COMPLETE | docs/privacy/DPIA_template.md | - |

---

## 2. CCPA Compliance Matrix

| Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-------------|---------------------|--------|----------|-------------|
| **Right to Know** | DSR automation, data inventory | ✅ COMPLETE | DSR runbook | - |
| **Right to Delete** | RTBF support, data deletion API | ✅ COMPLETE | Existing F3 | - |
| **Right to Opt-Out** | Preference tracking, opt-out mechanism | 🟡 PARTIAL | Config system | Need self-service opt-out portal |
| **Right to Non-Discrimination** | Fairness monitoring (P-FAIR-SP) | ✅ COMPLETE | R-F008, Phase 5B | - |
| **Notice at Collection** | Privacy notices, processing documentation | ✅ COMPLETE | Documentation | - |
| **Service Provider Contracts** | Legal agreements (out of scope for system) | 📋 DOCUMENTATION | Legal templates | Contractual, not technical |
| **Data Security** | Encryption, access control, audit logs | ✅ COMPLETE | Security features | - |
| **Data Minimization** | P-DATA-MIN enforcement | ✅ COMPLETE | R-F009, Phase 4C | - |

---

## 3. EU AI Act Compliance Matrix (High-Risk AI Systems)

| Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-------------|---------------------|--------|----------|-------------|
| **Risk Management System** | Risk engine, risk register, continuous monitoring | ✅ COMPLETE | Risk register, R-F004 | - |
| **Data Governance** | Data minimization, quality controls, fairness analysis | ✅ COMPLETE | P-DATA-MIN, fairness metrics | - |
| **Technical Documentation** | Architecture docs, API contracts, specifications | ✅ COMPLETE | docs/ directory, Phase 2A | - |
| **Record-Keeping** | Audit logs, decision lineage, Merkle anchoring | ✅ COMPLETE | R-F005, P-AUD, P-POL-LIN | - |
| **Transparency & Information** | Decision justifications, audit portal | ✅ COMPLETE | P-JUST, Phase 9B | - |
| **Human Oversight** | Human-in-the-loop, appeals mechanism | ✅ COMPLETE | Phase 8-9, Phase 6B | - |
| **Accuracy, Robustness, Cybersecurity** | Adversarial testing, security scanning, resilience | ✅ COMPLETE | 36 tests, CI scans | - |
| **Quality Management System** | CI/CD, testing, monitoring, continuous improvement | ✅ COMPLETE | CI workflows, optimization (Phase 8-9) | - |
| **Post-Market Monitoring** | Drift detection, anomaly monitoring, incident response | ✅ COMPLETE | Phase 7, R-F015 | - |
| **Fundamental Rights Impact Assessment** | DPIA, fairness analysis, discrimination testing | ✅ COMPLETE | DPIA template, fairness harness | - |

---

## 4. NIST AI RMF Compliance Matrix

| Function | Category | Nethical Capability | Status | Evidence | Gap / Notes |
|----------|----------|---------------------|--------|----------|-------------|
| **GOVERN** | AI governance, oversight, accountability | Governance system, policy approval, audit logs | ✅ COMPLETE | nethicalplan.md, governance_drivers.md | - |
| **MAP** | Context, risks, impacts identification | Risk register, protected attributes, threat model | ✅ COMPLETE | risk_register.md, governance_drivers.md, threat_model.md | - |
| **MEASURE** | Performance metrics, fairness, trustworthiness | Risk scoring, fairness metrics, SLO monitoring | ✅ COMPLETE | R-F004, R-F008, Phase 5B | - |
| **MANAGE** | Risk treatment, incident response, continuous improvement | Risk mitigation, escalations, optimization | ✅ COMPLETE | Risk mitigation strategies, Phase 8-9 | - |

**Detailed NIST AI RMF Mapping**: See docs/compliance/NIST_RMF_MAPPING.md

---

## 5. OWASP LLM Top 10 Compliance Matrix

| Risk | Description | Nethical Capability | Status | Evidence | Gap / Notes |
|------|-------------|---------------------|--------|----------|-------------|
| **LLM01: Prompt Injection** | Malicious prompts manipulate LLM | Adversarial detection, correlation engine | ✅ COMPLETE | 36 adversarial tests | - |
| **LLM02: Insecure Output Handling** | Unvalidated LLM outputs cause harm | Output validation, safety constraints | ✅ COMPLETE | SafetyViolationDetector | - |
| **LLM03: Training Data Poisoning** | Corrupted training data | Differential privacy, federated learning (Phase 6) | ✅ COMPLETE | Advanced plan Phase 6 | - |
| **LLM04: Model Denial of Service** | Resource exhaustion | Quota enforcement, backpressure | ✅ COMPLETE | Existing quota system | - |
| **LLM05: Supply Chain Vulnerabilities** | Compromised dependencies | SBOM, dependency scanning, repro builds | ✅ COMPLETE | R-NF007, Phase 9A | - |
| **LLM06: Sensitive Information Disclosure** | PII leakage | PII detection & redaction | ✅ COMPLETE | R-F011, redaction pipeline | - |
| **LLM07: Insecure Plugin Design** | Plugin vulnerabilities | Plugin marketplace trust scoring (Phase F6) | 🟡 PARTIAL | F6 integration points | Plugin signature verification planned |
| **LLM08: Excessive Agency** | LLM takes unauthorized actions | Safety constraints, judgment levels (BLOCK/TERMINATE) | ✅ COMPLETE | SafetyJudge | - |
| **LLM09: Overreliance** | Users trust LLM outputs uncritically | Human oversight, low-confidence escalation | ✅ COMPLETE | HITL workflows | - |
| **LLM10: Model Theft** | Unauthorized model access | Access control, tenant isolation | ✅ COMPLETE | P-AUTH, P-TENANT-ISO | - |

**Detailed OWASP LLM Mapping**: See docs/compliance/OWASP_LLM_COVERAGE.md

---

## 6. SOC 2 Compliance Matrix (Trust Services Criteria)

| Criterion | Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-----------|-------------|---------------------|--------|----------|-------------|
| **Security** | Access controls, encryption, monitoring | RBAC, encryption, audit logs | ✅ COMPLETE | P-AUTH, security features | - |
| **Availability** | System uptime, disaster recovery | 99.9% target, backup/DR | ✅ COMPLETE | R-NF003, R-NF005 | - |
| **Processing Integrity** | Accurate, complete, timely processing | Determinism (P-DET), termination (P-TERM) | ✅ COMPLETE | R-F001, R-F002 | - |
| **Confidentiality** | Sensitive data protection | PII redaction, encryption, access control | ✅ COMPLETE | R-F011, P-DATA-MIN | - |
| **Privacy** | GDPR/CCPA compliance (see above) | GDPR/CCPA capabilities | ✅ COMPLETE | Sections 1-2 above | - |

---

## 7. ISO 27001 Compliance Matrix (Key Controls)

| Control | Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|---------|-------------|---------------------|--------|----------|-------------|
| **A.5: Information Security Policies** | Documented policies | Security policy, threat model | ✅ COMPLETE | SECURITY.md, threat_model.md | - |
| **A.6: Organization of Information Security** | Roles and responsibilities | RACI matrix in nethicalplan.md | ✅ COMPLETE | nethicalplan.md | - |
| **A.9: Access Control** | User access management | RBAC, SSO, MFA, P-AUTH | ✅ COMPLETE | R-NF006, SSO/MFA guides | - |
| **A.10: Cryptography** | Cryptographic controls | Merkle hashing, signatures, quantum-resistant crypto | ✅ COMPLETE | Phase 6 quantum crypto | - |
| **A.12: Operations Security** | Operational procedures, monitoring | Observability, probes, incident response | ✅ COMPLETE | OTEL integration, probes (Phase 7A) | - |
| **A.14: System Acquisition, Development and Maintenance** | Secure SDLC | CI/CD security scanning, testing | ✅ COMPLETE | CI workflows | - |
| **A.16: Information Security Incident Management** | Incident handling | Incident response playbooks, escalation | 📋 DOCUMENTATION | Playbooks documented | - |
| **A.17: Business Continuity** | Backup, DR, availability | Backup/DR procedures, 99.9% uptime | ✅ COMPLETE | R-NF003, R-NF005 | - |
| **A.18: Compliance** | Legal/regulatory compliance | Compliance matrix (this document) | ✅ COMPLETE | This document | - |

---

## 8. HIPAA Compliance Matrix (for Healthcare Deployments)

| Rule | Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|------|-------------|---------------------|--------|----------|-------------|
| **Privacy Rule** | Protect PHI confidentiality | PII/PHI redaction, access control, audit logs | ✅ COMPLETE | R-F011, P-DATA-MIN | - |
| **Security Rule** | Administrative, physical, technical safeguards | RBAC, encryption, audit logs, monitoring | ✅ COMPLETE | Security features | Physical controls (infrastructure) |
| **Breach Notification Rule** | Notify breaches within 60 days | Incident response procedures | 📋 DOCUMENTATION | Incident playbooks | Need HIPAA-specific templates |
| **Minimum Necessary** | Access only minimum necessary PHI | P-DATA-MIN enforcement | ✅ COMPLETE | R-F009, context whitelisting | - |
| **Audit Controls** | Record and examine activity | Audit logs, Merkle anchoring | ✅ COMPLETE | R-F005, P-AUD | - |
| **Integrity Controls** | Protect data from alteration/destruction | Merkle anchoring, immutable logs | ✅ COMPLETE | P-NONREP, append-only storage | - |
| **Transmission Security** | Encrypt PHI in transit | TLS encryption (infrastructure) | ✅ COMPLETE | Infrastructure config | - |

---

## 9. FedRAMP Compliance Matrix (Control Families - Subset)

| Control Family | Example Controls | Nethical Capability | Status | Evidence | Gap / Notes |
|----------------|------------------|---------------------|--------|----------|-------------|
| **AC: Access Control** | AC-2 (Account Management), AC-3 (Access Enforcement) | RBAC, P-AUTH, SSO/MFA | ✅ COMPLETE | R-NF006, access control guides | - |
| **AU: Audit and Accountability** | AU-2 (Event Logging), AU-9 (Protection of Audit Info) | Audit logs, Merkle anchoring | ✅ COMPLETE | R-F005, P-AUD, P-NONREP | - |
| **IA: Identification and Authentication** | IA-2 (User Identification), IA-4 (Identifier Management) | SSO/SAML, MFA | ✅ COMPLETE | SSO/MFA guides | - |
| **SC: System and Communications Protection** | SC-7 (Boundary Protection), SC-8 (Transmission Confidentiality) | Network security (infra), TLS | ✅ COMPLETE | Infrastructure config | Infrastructure controls |
| **SI: System and Information Integrity** | SI-2 (Flaw Remediation), SI-3 (Malicious Code Protection) | Vulnerability scanning, SBOM | ✅ COMPLETE | R-NF008, CI security scans | - |
| **CM: Configuration Management** | CM-2 (Baseline Configuration), CM-3 (Change Control) | Config management, policy versioning | ✅ COMPLETE | Config system, policy lineage | - |
| **CP: Contingency Planning** | CP-7 (Alternate Processing Site), CP-9 (Backup) | Multi-region, backup/DR | ✅ COMPLETE | Regional configs, R-NF005 | - |

**Note**: Full FedRAMP compliance requires Impact Level assessment (IL4/IL5) and third-party auditor validation.

---

## 10. Anti-Discrimination Law Compliance Matrix

### ECOA (Equal Credit Opportunity Act)

| Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-------------|---------------------|--------|----------|-------------|
| No discrimination based on race, color, religion, national origin, sex, marital status, age, income source | Fairness monitoring (P-FAIR-SP), protected attribute analysis | ✅ COMPLETE | R-F008, governance_drivers.md | - |
| Adverse action notices with reasons | Decision justifications, appeals mechanism | ✅ COMPLETE | P-JUST, P-APPEAL | - |
| Statistical testing for disparate impact | Statistical parity tests, disparate impact ratio | ✅ COMPLETE | Phase 5B, fairness test harness | - |

### FHA (Fair Housing Act)

| Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-------------|---------------------|--------|----------|-------------|
| No housing discrimination based on protected classes | Fairness monitoring for housing-specific attributes (zip code, source of income) | ✅ COMPLETE | governance_drivers.md domain-specific considerations | - |
| Disparate impact analysis | Statistical parity, disparate impact ratio | ✅ COMPLETE | Phase 5B | - |

### EEOC (Equal Employment Opportunity Commission) Guidelines

| Requirement | Nethical Capability | Status | Evidence | Gap / Notes |
|-------------|---------------------|--------|----------|-------------|
| No employment discrimination | Fairness monitoring for employment attributes (education, arrest records, etc.) | ✅ COMPLETE | governance_drivers.md | - |
| 80% rule (disparate impact ratio ≥0.80) | Disparate impact ratio computation | ✅ COMPLETE | Phase 5B, fairness_metrics.md | - |

---

## Compliance Gap Summary

### Critical Gaps (🔴)
- **None identified**: All critical regulatory requirements have at least partial implementation or documented mitigation.

### Partial Implementation (🟡)
1. **GDPR Data Accuracy**: No automated data correction; manual admin workflow only
2. **GDPR Right to Rectification**: No self-service portal; admin tools only
3. **GDPR Right to Object**: No self-service objection mechanism
4. **CCPA Opt-Out**: Need self-service opt-out portal
5. **OWASP LLM07**: Plugin signature verification planned but not implemented

### Documentation Needed (📋)
1. **GDPR Breach Notification**: Add GDPR-specific templates to incident playbooks
2. **HIPAA Breach Notification**: Add HIPAA-specific templates
3. **ISO 27001 Incident Management**: Formalize incident response documentation

---

## Remediation Plan

| Gap | Priority | Target Phase | Owner | Action Items |
|-----|----------|--------------|-------|--------------|
| GDPR/CCPA Self-Service Portals | MEDIUM | Phase 10+ (Post-MVP) | Product Owner | Design and implement data subject self-service portal |
| GDPR Data Accuracy | LOW | Phase 10+ | Tech Lead | Automated data quality checks and correction workflows |
| Plugin Signature Verification | MEDIUM | Phase 10+ (F6 expansion) | Security Lead | Implement Cosign-based plugin signing and verification |
| Breach Notification Templates | HIGH | Phase 9B | Legal / Security Lead | Create GDPR/HIPAA-compliant notification templates |
| Incident Response Formalization | MEDIUM | Phase 8B | Security Lead | Document incident response procedures per ISO 27001 |

---

## Compliance Monitoring & Validation

### Continuous Compliance Checks
- **Daily**: Audit log integrity verification (Merkle root checks)
- **Weekly**: Security scanning (SAST, DAST, dependency checks)
- **Monthly**: Fairness metric reports, drift analysis
- **Quarterly**: Compliance audit reviews, gap reassessments
- **Annually**: External audits (SOC 2, ISO 27001, FedRAMP)

### Compliance Dashboards
- **Metrics**: Fairness thresholds, data minimization violations, breach incidents
- **Alerts**: Threshold breaches, security incidents, SLO violations
- **Reports**: Monthly compliance status, quarterly board reports

---

## External Audit Readiness

### Phase 10B: External Audits & Continuous Improvement
- **SOC 2 Type II**: Third-party audit of trust services criteria
- **ISO 27001 Certification**: Information security management audit
- **Fairness Re-certification**: Independent fairness audit by ethics experts
- **FedRAMP Assessment**: 3PAO (Third-Party Assessment Organization) evaluation

### Audit Artifacts Ready
- ✅ Risk register, requirements, compliance matrix (Phases 0-1)
- ✅ Formal specifications and proofs (Phases 2-6)
- ✅ Audit logs with Merkle anchoring (Phase 4)
- ✅ Fairness reports (Phase 5B)
- ✅ Audit portal (Phase 9B)
- ✅ SBOM, signed artifacts, repro build scripts (Phase 9A)

---

## Related Documents
- governance_drivers.md: Regulatory drivers and protected attributes
- requirements.md: Compliance requirements (G-001 to G-009)
- risk_register.md: Compliance risks
- NIST_RMF_MAPPING.md: Detailed NIST AI RMF coverage
- OWASP_LLM_COVERAGE.md: Detailed OWASP LLM Top 10 coverage
- DPIA_template.md: GDPR data protection impact assessment
- DSR_runbook.md: Data subject request procedures
- threat_model.md: Security threat analysis

---

**Status**: ✅ Phase 1B Deliverable - COMPLETE  
**Last Updated**: 2025-11-16  
**Owner**: Governance Lead / Legal Counsel
