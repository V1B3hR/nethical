# Nethical Privacy Policy

**Version**: 1.0.0  
**Effective Date**: 2025-12-03  
**Last Updated**: 2025-12-03

---

## 1. Introduction

This Privacy Policy explains how Nethical ("we," "our," or "the software") handles data in the context of AI governance and safety enforcement. Nethical is designed to govern AI agents, not to collect or process personal user data.

---

## 2. What Nethical IS

Nethical is an **AI Governance Framework** designed for:

- **Policy Enforcement**: Enforcing organizational AI policies on AI agents
- **Safety Guardrails**: Implementing the 25 Fundamental Laws for AI safety
- **Risk Assessment**: Evaluating AI agent actions for potential risks
- **Compliance Validation**: Ensuring AI systems meet regulatory requirements (EU AI Act, GDPR, etc.)
- **Audit Logging**: Maintaining immutable logs of AI governance decisions
- **Ethical AI Operations**: Ensuring AI systems operate within defined ethical boundaries

### Target Use Cases

| Domain | Application |
|--------|-------------|
| Autonomous Vehicles | Safety decision governance for self-driving systems |
| Industrial Robots | Operational policy enforcement for manufacturing AI |
| Medical AI | Regulatory compliance for healthcare AI systems |
| Enterprise AI | Corporate AI governance and risk management |
| Cloud Services | AI safety controls for cloud-based AI platforms |

---

## 3. What Nethical is NOT

Nethical explicitly **does NOT**:

| Category | What We Do NOT Do |
|----------|-------------------|
| 🚫 Spyware | Nethical does not spy on users or collect personal information |
| 🚫 Keylogger | Nethical does not capture keystrokes or user inputs |
| 🚫 User Tracking | Nethical does not track user behavior, location, or activities |
| 🚫 Data Harvesting | Nethical does not collect or sell user data |
| 🚫 Surveillance | Nethical does not monitor human behavior |
| 🚫 Consumer Spying | Nethical is not installed on personal devices to monitor users |

### Key Principle

> **"Nethical governs AI agents, not humans."**

All data processing relates exclusively to AI agent operations, policy decisions, and governance events—never to human user surveillance.

---

## 4. Data Collected

### 4.1 What We DO Collect

Nethical collects and processes data strictly related to AI governance:

| Data Type | Description | Purpose |
|-----------|-------------|---------|
| **AI Governance Decisions** | Allow/Restrict/Block/Terminate decisions | Audit compliance, decision traceability |
| **Policy Evaluations** | Results of policy checks against AI actions | Policy enforcement, compliance verification |
| **Risk Scores** | Quantified risk assessments (0.0-1.0 scale) | Safety monitoring, threshold enforcement |
| **Action Metadata** | Action types, categories, timestamps | Decision context, audit trails |
| **Violation Records** | Detected policy violations | Compliance reporting, remediation |
| **Agent Identifiers** | Unique AI agent IDs (not human user IDs) | Agent lifecycle management |
| **Performance Metrics** | Latency, throughput, system health | Operational monitoring |

### 4.2 What We DO NOT Collect

| Data Type | Status |
|-----------|--------|
| Personal user data | ❌ Never collected |
| Keystrokes | ❌ Never collected |
| File contents | ❌ Never collected |
| Browsing history | ❌ Never collected |
| Location data | ❌ Never collected |
| Biometric data | ❌ Never collected |
| Conversation content | ❌ Never collected (only action metadata) |
| User credentials | ❌ Never collected |
| Private communications | ❌ Never collected |

---

## 5. Consent Model

### 5.1 Installation Consent

Nethical is **enterprise software** installed only by:

- System owners
- IT administrators
- DevOps teams
- AI platform operators

**Individual consumer consent is not required** because:
1. Nethical is not installed on personal devices
2. Nethical does not process personal data
3. Nethical governs AI systems, not human users

### 5.2 Organizational Consent

Organizations deploying Nethical consent to:
- Governance logging of AI agent decisions
- Policy enforcement on AI systems
- Compliance reporting for regulatory requirements

### 5.3 Data Subject Rights

When personal data is incidentally processed (e.g., an AI agent's action references a user):
- **Access**: Request copies of governance logs
- **Rectification**: Correct inaccurate records
- **Erasure**: Request deletion (subject to legal retention requirements)
- **Portability**: Export governance data in standard formats
- **Objection**: Object to specific processing activities

---

## 6. Regulatory Compliance

### 6.1 GDPR Compliance (EU General Data Protection Regulation)

| GDPR Requirement | Nethical Compliance |
|------------------|---------------------|
| **Article 5** - Data Minimization | Only governance data collected; no personal data |
| **Article 6** - Lawful Basis | Legitimate interest for AI safety; no consent needed for non-personal data |
| **Article 13/14** - Transparency | Full documentation of data practices |
| **Article 17** - Right to Erasure | Data retention policies with deletion mechanisms |
| **Article 22** - Automated Decisions | Explainability API for AI governance decisions |
| **Article 25** - Privacy by Design | Minimal data collection; security-first architecture |
| **Article 32** - Security | HSM, TPM, encryption, access controls |

### 6.2 CCPA Compliance (California Consumer Privacy Act)

| CCPA Requirement | Nethical Compliance |
|------------------|---------------------|
| **Right to Know** | Transparency documentation available |
| **Right to Delete** | Retention policies with deletion support |
| **Right to Opt-Out** | Not applicable (no personal data sale) |
| **Non-Discrimination** | Not applicable (no consumer service) |

**Note**: CCPA primarily applies to consumer personal information. Nethical's AI governance data typically does not constitute "personal information" under CCPA.

### 6.3 EU AI Act Compliance

Nethical is designed to help organizations comply with the EU AI Act:

| EU AI Act Requirement | Nethical Implementation |
|-----------------------|-------------------------|
| **Article 9** - Risk Management | Integrated risk scoring and assessment |
| **Article 10** - Data Governance | Governance logging and data quality controls |
| **Article 11** - Documentation | Comprehensive audit trails |
| **Article 12** - Record-Keeping | Immutable governance logs |
| **Article 13** - Transparency | Explainability API and decision tracing |
| **Article 14** - Human Oversight | Human-in-the-loop interface and controls |
| **Article 15** - Accuracy | Validation and testing frameworks |

---

## 7. Data Retention

| Data Category | Retention Period | Justification |
|---------------|------------------|---------------|
| Governance Decisions | 7 years | Regulatory audit requirements |
| Policy Evaluations | 1 year | Operational analysis |
| Performance Metrics | 90 days | System monitoring |
| Security Events | 7 years | Security compliance |
| Debug Logs | 30 days | Troubleshooting |

---

## 8. Data Security

### 8.1 Technical Measures

- **Encryption**: TLS 1.3 in transit, AES-256 at rest
- **Access Control**: RBAC, MFA, SSO/SAML
- **Hardware Security**: HSM for key management, TPM for device attestation
- **Audit Logging**: Immutable Merkle-tree anchored logs
- **Network Security**: Zero-trust architecture, network segmentation

### 8.2 Organizational Measures

- Security incident response procedures
- Regular penetration testing
- Employee security training
- Vendor security assessments

---

## 9. Third-Party Data Sharing

Nethical **does not sell** or share data with third parties for marketing purposes.

Data may be shared only:
- With cloud infrastructure providers (for hosting, under DPA)
- With auditors (for compliance verification)
- With law enforcement (if legally required)

---

## 10. International Data Transfers

When data is transferred internationally:
- Standard Contractual Clauses (SCCs) are used
- Data residency controls ensure EU data stays in EU regions
- Encryption protects data in transit

---

## 11. Contact Information

For privacy-related inquiries:

- **Email**: privacy@nethical.io
- **Documentation**: See `docs/legal/` for detailed compliance information
- **Data Protection Officer**: dpo@nethical.io

---

## 12. Policy Updates

This policy may be updated periodically. Changes will be documented in the version history below.

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-03 | Initial privacy policy |

---

## 13. Related Documents

- [Deployment Scope](docs/legal/DEPLOYMENT_SCOPE.md) - Appropriate vs. inappropriate use cases
- [Audit Response Guide](docs/legal/AUDIT_RESPONSE_GUIDE.md) - FAQ for responding to auditors
- [Data Flow Documentation](docs/legal/DATA_FLOW.md) - Visual data flow diagrams
- [Fundamental Laws](FUNDAMENTAL_LAWS.md) - 25 AI governance principles
- [Security Policy](SECURITY.md) - Security practices and vulnerability reporting

---

**"AI Governance for Safety, Not Surveillance"**
