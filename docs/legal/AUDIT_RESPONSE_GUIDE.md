# Nethical Audit Response Guide

**Version**: 1.0.0  
**Last Updated**: 2025-12-03

---

## Overview

This guide provides standardized responses to common auditor questions about Nethical. Use these responses when preparing for or responding to compliance audits, security assessments, or regulatory inquiries.

---

## Frequently Asked Questions

### 1. Data Collection & Privacy

#### Q: Does Nethical collect personal data?

**A: No, Nethical does not collect personal data.**

Nethical is an AI governance framework that monitors AI agent decisions, not human users. The data we collect includes:

| Collected | Not Collected |
|-----------|---------------|
| AI governance decisions (allow/block) | Personal user data |
| Policy evaluation results | Keystrokes |
| Risk scores (0.0-1.0) | Browsing history |
| Action metadata (type, timestamp) | File contents |
| Agent identifiers | User credentials |

**Key Principle**: "Nethical governs AI agents, not humans."

---

#### Q: Does Nethical track user behavior?

**A: No.**

Nethical tracks AI agent behavior, not human user behavior. We monitor:

- AI agent actions and decisions
- Policy compliance of AI systems
- Risk assessments of AI operations
- Governance events

We do NOT monitor:
- User browsing activity
- User keystrokes
- User location
- User communications

---

#### Q: What data does Nethical log?

**A: AI governance data only.**

| Data Logged | Purpose | Retention |
|-------------|---------|-----------|
| Action type | Decision context | 7 years |
| Governance decision | Audit compliance | 7 years |
| Risk score | Safety monitoring | 7 years |
| Timestamp | Audit trail | 7 years |
| Agent ID | Agent lifecycle | 7 years |
| Policy violations | Compliance reporting | 7 years |

**Not Logged**:
- User identity (unless explicitly required for audit)
- Personal data
- Conversation content
- File contents

---

### 2. Deployment & Scope

#### Q: Is Nethical installed on consumer devices?

**A: No.**

Nethical is enterprise software deployed only on:

- Enterprise AI platforms
- Cloud AI infrastructure
- Industrial AI systems
- Safety-critical systems (autonomous vehicles, medical AI)

Nethical is NOT deployed on:
- Personal computers
- Consumer smartphones
- Home devices
- Any device without administrator authorization

---

#### Q: Who authorizes Nethical installation?

**A: System owners and IT administrators only.**

Nethical requires explicit authorization from:

- IT/DevOps administrators
- System owners
- Platform operators
- Enterprise security teams

Individual user consent is not required because:
1. Nethical does not process personal data
2. Nethical governs AI systems, not users
3. Enterprise software falls under organizational authority

---

#### Q: Can Nethical be used for employee surveillance?

**A: No, this is explicitly prohibited.**

Employee surveillance violates:
- Nethical's design principles
- Fundamental Law 1 (Human Dignity)
- GDPR/CCPA requirements
- Our Deployment Scope policy

See: [Deployment Scope Guide](DEPLOYMENT_SCOPE.md)

---

### 3. Regulatory Compliance

#### Q: How does Nethical comply with GDPR?

**A: Nethical is designed for GDPR compliance.**

| GDPR Article | Compliance Approach |
|--------------|---------------------|
| **Art. 5** - Minimization | Only governance data; no personal data |
| **Art. 6** - Lawful Basis | Legitimate interest for AI safety |
| **Art. 17** - Erasure | Data retention policies with deletion |
| **Art. 22** - Automated Decisions | Explainability API available |
| **Art. 25** - Privacy by Design | Minimal collection architecture |
| **Art. 32** - Security | HSM, TPM, encryption, RBAC |

**Documentation**: See [Privacy Policy](../../PRIVACY.md) for full GDPR compliance details.

---

#### Q: How does Nethical comply with CCPA?

**A: CCPA is largely not applicable to Nethical.**

Nethical does not collect "personal information" as defined by CCPA. We do not:
- Collect consumer personal data
- Sell user data
- Profile consumers

However, if AI governance logs incidentally contain personal information:
- Right to Know: Documentation provided
- Right to Delete: Retention policies support deletion
- Right to Opt-Out: N/A (no data sale)

---

#### Q: How does Nethical support EU AI Act compliance?

**A: Nethical is a compliance tool for the EU AI Act.**

| EU AI Act Requirement | Nethical Support |
|-----------------------|------------------|
| **Art. 9** - Risk Management | Risk scoring framework |
| **Art. 10** - Data Governance | Governance logging |
| **Art. 11** - Documentation | Audit trail generation |
| **Art. 12** - Record-Keeping | Immutable logs (Merkle-anchored) |
| **Art. 13** - Transparency | Explainability API |
| **Art. 14** - Human Oversight | HITL interface |
| **Art. 15** - Accuracy | Validation framework |

---

### 4. User Rights

#### Q: Can users opt out of Nethical?

**A: The question does not apply to Nethical's use case.**

Since Nethical governs AI agents (not humans), there is no "user" to opt out. However:

**For AI Operators**:
- Organizations can choose not to deploy Nethical
- Specific AI agents can be excluded from governance
- Policies can be customized per use case

**For Data Subject Rights (GDPR Article 22)**:
If an AI governance decision affects a human (e.g., an AI denies a service):

1. The human can request an explanation via the Transparency API
2. The human can appeal the decision via the Human Oversight API
3. The decision can be reviewed and overridden by a human operator

**Note**: This applies to the AI's decision, not Nethical's governance of that AI.

---

#### Q: How do users request data access/deletion?

**A: Contact the deploying organization.**

Since Nethical is deployed by organizations, data requests should be directed to:

1. The organization operating the AI system
2. The organization's Data Protection Officer
3. The organization's privacy contact

Nethical provides APIs for organizations to:
- Export governance data
- Delete specific records (within retention policy limits)
- Generate compliance reports

---

### 5. Security

#### Q: How is data protected?

**A: Defense-in-depth security architecture.**

| Layer | Protection |
|-------|------------|
| **Transport** | TLS 1.3 encryption |
| **Storage** | AES-256 encryption at rest |
| **Access** | RBAC, MFA, SSO/SAML |
| **Keys** | HSM-backed key management |
| **Devices** | TPM attestation (edge) |
| **Audit** | Immutable Merkle-tree logs |
| **Network** | Zero-trust architecture |

---

#### Q: Is data encrypted?

**A: Yes, all data is encrypted.**

- **In Transit**: TLS 1.3
- **At Rest**: AES-256
- **Keys**: HSM-protected
- **Logs**: Cryptographically signed

---

#### Q: Who has access to governance data?

**A: Authorized personnel only.**

Access is controlled via:
- Role-Based Access Control (RBAC)
- Multi-Factor Authentication (MFA)
- SSO/SAML integration
- Audit logging of all access

Access roles:
| Role | Access Level |
|------|--------------|
| Admin | Full system access |
| Auditor | Read-only audit logs |
| Operator | Policy management |
| Viewer | Dashboard only |

---

### 6. Data Retention & Deletion

#### Q: How long is data retained?

**A: Per regulatory requirements.**

| Data Type | Retention | Justification |
|-----------|-----------|---------------|
| Governance decisions | 7 years | Audit compliance |
| Policy evaluations | 1 year | Operational analysis |
| Performance metrics | 90 days | System monitoring |
| Security events | 7 years | Security compliance |
| Debug logs | 30 days | Troubleshooting |

---

#### Q: Can data be deleted on request?

**A: Yes, subject to legal retention requirements.**

- Routine deletion per retention schedule
- On-request deletion (where legally permitted)
- Secure deletion (cryptographic erasure)
- Deletion audit trail maintained

---

### 7. Third-Party Sharing

#### Q: Is data shared with third parties?

**A: Not for commercial purposes.**

Data may be shared only with:

| Third Party | Purpose | Safeguards |
|-------------|---------|------------|
| Cloud providers | Infrastructure hosting | DPA, encryption |
| Auditors | Compliance verification | NDA, access controls |
| Law enforcement | Legal requirements | Legal process only |

Data is NOT shared for:
- Marketing
- Advertising
- Data brokerage
- Commercial sale

---

### 8. Certifications & Standards

#### Q: What certifications does Nethical have/support?

**A: Nethical supports multiple compliance frameworks.**

| Standard | Status | Scope |
|----------|--------|-------|
| ISO 27001 | Documentation complete | Information security |
| ISO 26262 | ASIL-D ready | Automotive safety |
| IEC 62443 | SL-3 ready | Industrial security |
| FDA Part 11 | Documentation complete | Medical device |
| EU AI Act | Compliance ready | AI regulation |
| SOC 2 | Framework aligned | Security controls |

---

## Providing Evidence to Auditors

### Available Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Privacy Policy | `/PRIVACY.md` | Data practices |
| Deployment Scope | `/docs/legal/DEPLOYMENT_SCOPE.md` | Appropriate use |
| Data Flow | `/docs/legal/DATA_FLOW.md` | Data architecture |
| Security Policy | `/SECURITY.md` | Security practices |
| Fundamental Laws | `/FUNDAMENTAL_LAWS.md` | Ethical framework |
| Architecture | `/ARCHITECTURE.md` | System design |
| Audit Trail | `/AUDIT.md` | Logging practices |

### System Capabilities for Audit

| Capability | API/Feature |
|------------|-------------|
| Audit log export | `/v2/audit/export` |
| Decision explanations | `/v2/explanations/{id}` |
| Policy documentation | `/v2/policies` |
| Compliance reports | `/v2/compliance/report` |
| Data subject requests | `/v2/dsr/{type}` |

---

## Escalation for Unusual Questions

If an auditor asks a question not covered here:

1. **Document** the question exactly
2. **Do not speculate** - say "I will get you accurate information"
3. **Escalate** to:
   - Legal team
   - Compliance officer
   - Data Protection Officer
4. **Follow up** with documented response

---

## Related Documents

- [Privacy Policy](../../PRIVACY.md) - Complete privacy practices
- [Deployment Scope](DEPLOYMENT_SCOPE.md) - Appropriate use guidance
- [Data Flow Documentation](DATA_FLOW.md) - Data flow diagrams
- [Security Policy](../../SECURITY.md) - Security architecture

---

**"Prepared for audit, committed to transparency."**
