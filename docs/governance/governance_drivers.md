# Governance Drivers & Protected Attributes

## Overview
This document identifies the governance, compliance, and fairness domains that drive Nethical's design and operation. It establishes protected attributes for fairness analysis and maps regulatory requirements to system capabilities.

---

## Governance Domains

### 1. Safety Governance
**Purpose**: Prevent harmful or dangerous AI behaviors  
**Key Requirements**:
- Real-time action monitoring and risk assessment
- Safety constraint enforcement (unauthorized access, data modification, resource abuse)
- Immediate blocking and termination capabilities
- Comprehensive safety violation taxonomy

**Regulatory Drivers**:
- AI safety principles (IEEE, ISO/IEC)
- Product liability regulations
- Industry-specific safety standards

**Nethical Capabilities**:
- SafetyViolationDetector with multi-category checks
- SafetyJudge with graduated response (ALLOW/RESTRICT/BLOCK/TERMINATE)
- Real-time safety metrics and alerting

---

### 2. Ethical Compliance
**Purpose**: Ensure AI operations align with ethical principles and societal norms  
**Key Requirements**:
- Detection of harmful content, deception, discrimination
- Manipulation recognition (emotional, authority, social proof, scarcity)
- Ethical taxonomy tagging and reporting
- Transparent decision justifications

**Regulatory Drivers**:
- EU AI Act (high-risk AI systems)
- Industry codes of ethics
- Professional standards (medical, legal, financial)

**Nethical Capabilities**:
- EthicalViolationDetector with comprehensive pattern matching
- Ethical taxonomy system (Phase 4)
- ManipulationDetector with multi-strategy recognition
- Decision explainability and justification

---

### 3. Privacy & Data Protection
**Purpose**: Safeguard personal information and respect data subject rights  
**Key Requirements**:
- PII detection and redaction
- Data minimization enforcement
- Right to be forgotten (RTBF) support
- Differential privacy for aggregated analytics
- Data residency compliance

**Regulatory Drivers**:
- GDPR (EU General Data Protection Regulation)
- CCPA (California Consumer Privacy Act)
- HIPAA (Health Insurance Portability and Accountability Act)
- PIPEDA (Canada), LGPD (Brazil), etc.

**Nethical Capabilities**:
- 10+ PII type detection (email, SSN, credit cards, phone, IP, etc.)
- Redaction pipeline with configurable policies
- Data minimization with whitelist enforcement (P-DATA-MIN)
- Differential privacy with epsilon-delta guarantees
- Regional data residency policies

---

### 4. Fairness & Non-Discrimination
**Purpose**: Prevent biased decisions and ensure equitable treatment across population groups  
**Key Requirements**:
- Protected attribute identification
- Statistical parity monitoring
- Counterfactual fairness evaluation
- Disparate impact analysis
- Drift detection and recalibration

**Regulatory Drivers**:
- Equal Credit Opportunity Act (ECOA)
- Fair Housing Act (FHA)
- EU Anti-Discrimination Directives
- EEOC guidelines (employment)

**Nethical Capabilities**:
- Fairness metrics baseline (Phase 2C)
- Cohort-based fairness sampling
- Statistical parity tests (P-FAIR-SP)
- Counterfactual evaluation harness (P-FAIR-CF)
- Monthly fairness reporting and anti-drift recalibration

---

### 5. Transparency & Accountability
**Purpose**: Enable oversight, auditability, and contestability of AI decisions  
**Key Requirements**:
- Immutable audit trails
- Decision lineage and justification completeness
- Contestability mechanism (appeals)
- Policy version tracking and diffs
- Public transparency reporting

**Regulatory Drivers**:
- EU AI Act (transparency obligations)
- GDPR (right to explanation)
- Algorithmic accountability laws
- Public sector AI governance frameworks

**Nethical Capabilities**:
- Merkle-anchored audit logs (P-AUD, P-NONREP)
- Policy lineage with hash chains (P-POL-LIN)
- Appeals/contestability CLI (P-APPEAL)
- Comprehensive decision justifications (P-JUST)
- Audit portal for human-facing exploration

---

### 6. Security & Integrity
**Purpose**: Protect system from attacks, ensure data integrity and access control  
**Key Requirements**:
- Adversarial attack detection (prompt injection, jailbreak)
- Multi-signature policy approval (P-MULTI-SIG)
- Tenant isolation (P-TENANT-ISO)
- Supply chain security (reproducible builds, SBOM)
- Vulnerability management

**Regulatory Drivers**:
- NIST Cybersecurity Framework
- ISO 27001 (Information Security Management)
- SOC 2 compliance
- FedRAMP (federal cloud security)

**Nethical Capabilities**:
- Adversarial testing suite (36 test scenarios)
- Multi-sig policy activation workflow
- Formal tenant isolation proofs (non-interference)
- Deterministic reproducible builds with signing
- Continuous security scanning (SAST, DAST, dependency checks)

---

### 7. Operational Reliability
**Purpose**: Ensure system availability, performance, and resilience under load  
**Key Requirements**:
- SLO compliance (latency, throughput, availability)
- Quota enforcement and backpressure
- Graceful degradation under load
- Monitoring and alerting
- Incident response

**Regulatory Drivers**:
- Industry SLA standards
- Operational resilience regulations (financial services)
- Disaster recovery requirements

**Nethical Capabilities**:
- Performance optimization with risk-based gating
- Quota enforcement per agent/cohort/tenant
- Runtime probes mirroring formal invariants
- OpenTelemetry integration (metrics, traces, logs)
- SLO monitoring and violation alerting

---

## Protected Attributes

Protected attributes are characteristics that must not cause discriminatory decision outcomes. Nethical's fairness analysis monitors for disparate impact across these attributes.

### Primary Protected Attributes

| Attribute | Description | Regulatory Basis | Monitoring Priority |
|-----------|-------------|------------------|---------------------|
| **Race / Ethnicity** | Racial or ethnic group membership | Civil Rights Act, ECOA, FHA | CRITICAL |
| **Gender / Sex** | Gender identity or biological sex | Title VII, ECOA, FHA | CRITICAL |
| **Age** | Age group or birth year | ADEA, Age Discrimination Act | HIGH |
| **Disability Status** | Physical or mental disability | ADA, Section 504 | HIGH |
| **National Origin** | Country of origin or citizenship | Civil Rights Act, Immigration law | HIGH |
| **Religion** | Religious affiliation or belief | Title VII, Religious Freedom Acts | HIGH |
| **Veteran Status** | Military service history | VEVRAA, USERRA | MEDIUM |
| **Marital Status** | Marital or family status | Various state laws | MEDIUM |
| **Sexual Orientation** | Sexual identity or preference | Various state/local laws, EEOC guidance | MEDIUM |
| **Genetic Information** | Genetic test results or family history | GINA | MEDIUM |

### Domain-Specific Considerations

#### Financial Services
- Credit history (protected from circular discrimination)
- Zip code (proxy for race/socioeconomic status)
- Income level (requires fairness analysis)

#### Healthcare
- Pre-existing conditions
- Genetic predisposition
- Family medical history

#### Employment
- Prior arrest records (may not be sole basis)
- Education level (requires justification)
- Employment gaps

#### Housing
- Zip code / neighborhood
- Source of income (Section 8, public assistance)

---

## Fairness Metrics Baseline (Phase 2C Preview)

### Statistical Parity (SP)
**Definition**: Decision rates should be similar across protected groups.  
**Metric**: `|P(decision=ALLOW | group=A) - P(decision=ALLOW | group=B)|`  
**Threshold**: ≤ 0.10 (10 percentage points)  
**Monitoring**: Monthly batch analysis

### Disparate Impact Ratio
**Definition**: Ratio of favorable outcome rates between groups.  
**Metric**: `P(decision=ALLOW | protected) / P(decision=ALLOW | reference)`  
**Threshold**: ≥ 0.80 (80% rule from EEOC)  
**Monitoring**: Monthly batch analysis

### Counterfactual Fairness
**Definition**: Decision unchanged if protected attribute altered.  
**Metric**: Percentage of decisions stable under attribute perturbation  
**Threshold**: ≥ 0.95 (95% stability)  
**Monitoring**: Quarterly deep analysis

---

## Compliance Matrix Mapping

| Regulation | Domain | Key Requirements | Nethical Capabilities |
|------------|--------|------------------|----------------------|
| **GDPR** | Privacy | Data minimization, RTBF, transparency | P-DATA-MIN, redaction, appeals, audit logs |
| **CCPA** | Privacy | Consumer data rights, opt-out | DSR automation, data deletion, preference tracking |
| **EU AI Act** | AI Safety | High-risk AI transparency, human oversight | Audit portal, HITL, justifications |
| **NIST AI RMF** | AI Risk | GOVERN, MAP, MEASURE, MANAGE | Governance system, risk engine, monitoring, optimization |
| **OWASP LLM Top 10** | Security | 10 LLM risks mitigated | Adversarial detection, quota, PII redaction, audit |
| **ECOA** | Fairness | No credit discrimination | Statistical parity, protected attribute monitoring |
| **ADA** | Accessibility | Disability accommodation | No adverse impact from disability status |
| **SOC 2** | Security | Access control, audit, incident response | RBAC, immutable logs, escalation workflows |
| **HIPAA** | Healthcare | PHI protection, breach notification | PII/PHI redaction, audit trails, encryption |
| **FedRAMP** | Federal | Cloud security standards | Security controls, vulnerability scanning, SBOM |

---

## Governance Priorities by Phase

### Phase 0-1 (Current)
- Establish governance domains and protected attributes ✅
- Map regulatory requirements to system capabilities ✅
- Identify compliance gaps (to be addressed in future phases)

### Phase 2
- Formalize fairness metrics baseline (2C)
- Design policy lineage system (2B)
- Document API contracts with governance constraints (2A)

### Phase 3-4
- Implement Merkle anchoring for audit integrity (3B)
- Formalize multi-sig approval workflow (4B)
- Prove data minimization enforcement (4C)

### Phase 5-6
- Deploy fairness test harness with statistical parity checks (5B)
- Implement appeals/contestability mechanism (6B)
- Achieve ≥70% proof coverage (6A)

### Phase 7-8
- Operationalize governance metrics dashboard (7B)
- Validate negative properties via red-team testing (8B)

### Phase 9-10
- Launch audit portal for public transparency (9B)
- Conduct external fairness audit (10B)

---

## Ethical Principles

Nethical's governance system is guided by these core ethical principles:

1. **Beneficence**: AI should benefit humanity and minimize harm
2. **Non-Maleficence**: AI should not cause harm through action or inaction
3. **Autonomy**: Respect human agency and decision-making capacity
4. **Justice**: Fair and equitable treatment; no unjust discrimination
5. **Explicability**: Transparent, understandable, and contestable decisions
6. **Accountability**: Clear responsibility and oversight mechanisms

---

## Stakeholder Roles

| Stakeholder | Governance Responsibilities |
|-------------|---------------------------|
| **Governance Lead** | Define protected attributes, approve fairness metrics, oversee compliance |
| **Ethics Data Scientist** | Analyze fairness metrics, identify bias, recommend interventions |
| **Legal Counsel** | Interpret regulations, review compliance matrix, approve public transparency |
| **Security Lead** | Enforce access control, oversee audit integrity, manage vulnerability response |
| **Product Owner** | Balance governance requirements with operational needs, approve trade-offs |
| **Formal Methods Engineer** | Prove governance properties (isolation, data minimization, etc.) |

---

## Next Actions

1. **Phase 1B**: Expand compliance_matrix.md with detailed regulatory mapping
2. **Phase 2C**: Create fairness_metrics.md with statistical test specifications
3. **Phase 3B**: Design Merkle audit architecture in policy_lineage.md
4. **Phase 4B**: Specify multi-sig workflow in access_control_spec.md
5. **Phase 5B**: Implement fairness test harness

---

## References
- risk_register.md: Governance risks (R-004, R-005, R-006, R-012)
- nethicalplan.md: Phase-by-phase governance integration
- GDPR, CCPA, EU AI Act: Source regulatory texts
- NIST AI RMF: https://www.nist.gov/itl/ai-risk-management-framework
- OWASP LLM Top 10: https://owasp.org/www-project-top-10-for-large-language-model-applications/

---

**Status**: ✅ Phase 0B Deliverable - COMPLETE  
**Last Updated**: 2025-11-16  
**Owner**: Governance Lead / Ethics Data Scientist
