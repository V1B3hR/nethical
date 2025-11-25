# UK Law Compliance Guide

## Overview

This document provides comprehensive guidance on Nethical's compliance with UK data protection and healthcare regulations:

- **UK GDPR** (retained EU law post-Brexit)
- **Data Protection Act 2018 (DPA 2018)**
- **NHS Data Security and Protection Toolkit (DSPT)**

## UK GDPR Compliance

The UK GDPR retains the core principles of EU GDPR with UK-specific provisions.

### Article 5: Principles of Processing

**Requirements:**
- Lawfulness, fairness, transparency
- Purpose limitation
- Data minimization
- Accuracy
- Storage limitation
- Integrity and confidentiality
- Accountability

**Nethical Implementation:**

| Principle | Module | Description |
|-----------|--------|-------------|
| Lawfulness | `nethical/security/data_compliance.py` | Processing purpose validation |
| Data Minimization | `nethical/core/data_minimization.py` | Automatic data reduction |
| Transparency | `nethical/explainability/transparency_report.py` | Processing transparency |
| Integrity | `nethical/security/encryption.py` | Data protection |

**Documentation:** `docs/privacy/DPIA_template.md`

### Article 6: Lawful Basis for Processing

**Nethical Implementation:**

The `DataSubjectRequestHandler` in `nethical/security/data_compliance.py` supports all processing purposes:

```python
class ProcessingPurpose(Enum):
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_INTEREST = "public_interest"
    LEGITIMATE_INTEREST = "legitimate_interest"
```

### Articles 12-22: Data Subject Rights

**Implementation Coverage:**

| Right | Article | Module | Status |
|-------|---------|--------|--------|
| Right to access | Art. 15 | `data_compliance.py` | ✅ Implemented |
| Right to rectification | Art. 16 | `data_compliance.py` | ✅ Implemented |
| Right to erasure | Art. 17 | `data_compliance.py` | ✅ Implemented |
| Right to restriction | Art. 18 | `data_compliance.py` | ✅ Implemented |
| Right to portability | Art. 20 | `data_compliance.py` | ✅ Implemented |
| Right to object | Art. 21 | `data_compliance.py` | ✅ Implemented |

**Example Usage:**

```python
from nethical.security.data_compliance import (
    DataSubjectRequestHandler,
    RequestType
)

handler = DataSubjectRequestHandler()

# Submit access request
request = handler.submit_request(
    request_type=RequestType.ACCESS,
    subject_id="user-123",
    verification_method="email"
)

# Process the request
result = handler.process_access_request(request.request_id)
```

**Documentation:** `docs/privacy/DSR_runbook.md`

### Article 25: Data Protection by Design

**Nethical Implementation:**

| Control | Module | Description |
|---------|--------|-------------|
| Differential Privacy | `nethical/core/differential_privacy.py` | Configurable epsilon |
| Redaction Pipeline | `nethical/core/redaction_pipeline.py` | PII redaction |
| Storage Partitioning | `nethical/storage/` | Regional data isolation |

**Configuration:**

```python
gov = IntegratedGovernance(
    privacy_mode="differential",
    epsilon=1.0,
    redaction_policy="aggressive",
    region_id="eu-west-1"
)
```

### Article 32: Security of Processing

**Nethical Implementation:**

| Control | Module | Evidence |
|---------|--------|----------|
| Encryption | `nethical/security/encryption.py` | TLS 1.3, AES-256 |
| Authentication | `nethical/security/auth.py` | MFA support |
| Access Control | `nethical/core/rbac.py` | Role-based access |

**Test Evidence:** `tests/test_security_hardening.py`

**Documentation:** `docs/SECURITY_HARDENING_GUIDE.md`

### Articles 33-34: Breach Notification

**Requirement:** Notify ICO within 72 hours of becoming aware of a personal data breach.

**Nethical Implementation:**

| Component | Module | Description |
|-----------|--------|-------------|
| SOC Integration | `nethical/security/soc_integration.py` | Incident alerting |
| Anomaly Detection | `nethical/security/anomaly_detection.py` | Breach detection |

**Documentation:** `docs/security/red_team_report_template.md`

### Article 35: Data Protection Impact Assessment

**Nethical provides DPIA template:** `docs/privacy/DPIA_template.md`

**DPIA includes:**
- System description
- Data types processed
- Privacy risks identified
- Mitigation measures
- Data subject rights implementation
- Assessment outcome checklist

## Data Protection Act 2018

### Section 64: Automated Decision-Making

**Requirement:** Rights related to decisions based solely on automated processing which produce legal effects.

**Nethical Implementation:**

| Component | Module | Description |
|-----------|--------|-------------|
| Decision Explainer | `nethical/explainability/decision_explainer.py` | Explain automated decisions |
| Human Review | `nethical/governance/human_review.py` | Override capability |
| Transparency | `nethical/explainability/transparency_report.py` | Processing information |

**Example:**

```python
from nethical.explainability.decision_explainer import DecisionExplainer

explainer = DecisionExplainer()
explanation = explainer.explain_decision(
    decision_id="dec-123",
    include_factors=True,
    include_alternatives=True
)
```

### Section 35: Law Enforcement Processing

For organizations processing data for law enforcement purposes:

| Requirement | Module | Implementation |
|-------------|--------|----------------|
| Data Classification | `policies/common/data_classification.yaml` | Category definitions |
| Policy Engine | `nethical/policy/engine.py` | Processing rules |
| Access Controls | `nethical/core/rbac.py` | Role-based restrictions |

## NHS Data Security and Protection Toolkit (DSPT)

The NHS DSPT has 10 data security standards. Nethical supports compliance with:

### Standard 1: Personal Confidential Data

**Requirement:** Staff ensure that personal confidential data is handled, stored and transmitted securely.

**Nethical Implementation:**

| Control | Module | Description |
|---------|--------|-------------|
| Redaction Pipeline | `nethical/core/redaction_pipeline.py` | PII detection and masking |
| Data Minimization | `nethical/core/data_minimization.py` | Reduce data exposure |
| Encryption | `nethical/security/encryption.py` | Data protection |

### Standard 3: Security Training

**Requirement:** All staff complete annual data security awareness training.

**Nethical Support:**
- Training Guide: `docs/TRAINING_GUIDE.md`
- Security awareness materials
- Training record templates

### Standard 7: Managing Access

**Requirement:** Only authorized staff can access data and systems.

**Nethical Implementation:**

| Control | Module | Evidence |
|---------|--------|----------|
| RBAC | `nethical/core/rbac.py` | Role-based access control |
| Authentication | `nethical/security/auth.py` | Identity verification |
| SSO/SAML | `nethical/security/sso.py` | Enterprise authentication |
| MFA | `nethical/security/mfa.py` | Multi-factor authentication |

**Documentation:** `docs/security/SSO_SAML_GUIDE.md`, `docs/security/MFA_GUIDE.md`

### Standard 8: Unsupported Systems

**Requirement:** No unsupported operating systems, software or internet browsers.

**Nethical Implementation:**
- Software Bill of Materials: `SBOM.json`
- Supply chain security: `docs/SUPPLY_CHAIN_SECURITY_GUIDE.md`
- Dependency management: `requirements-hashed.txt`

### Standard 10: Accountable Suppliers

**Requirement:** IT suppliers are held accountable via contracts.

**Nethical Support:**
- SBOM for dependency transparency
- Supply chain security guide
- Vendor assessment templates

## Regional Configuration

Configure Nethical for UK deployment:

```yaml
# policies/healthcare/regions.yaml
UK:
  compliance: [UK_GDPR, DPA_2018, NHS_DSPT]
  locales: [en_GB]
  data_residency_required: true
  export_controls_required: false
  default_geofencing_policy: OPEN
```

## Compliance Verification

```python
from nethical.security.regulatory_compliance import (
    UKLawCompliance,
    RegulatoryMappingGenerator
)

# Initialize UK compliance checker
uk_compliance = UKLawCompliance()

# Get all requirements
for req_id, req in uk_compliance.requirements.items():
    print(f"{req_id}: {req.title} - {req.implementation_status.value}")

# Generate compliance report
generator = RegulatoryMappingGenerator()
audit_report = generator.generate_audit_report(auditor_name="UK-ICO-Prep")
print(f"Compliance Score: {audit_report['summary']['compliance_score']}%")
```

## ICO Registration Checklist

For organizations registering with the Information Commissioner's Office:

- [ ] Complete DPIA using template
- [ ] Configure data residency for UK region
- [ ] Enable NHS DSPT controls if healthcare
- [ ] Document lawful basis for processing
- [ ] Implement data subject rights workflows
- [ ] Configure 72-hour breach notification
- [ ] Complete security training program
- [ ] Review SBOM and supply chain

---
**Version:** 1.0.0  
**Last Updated:** 2025-11-25  
**Regulation References:**
- UK GDPR (Retained EU Regulation 2016/679)
- Data Protection Act 2018
- NHS DSPT v5
