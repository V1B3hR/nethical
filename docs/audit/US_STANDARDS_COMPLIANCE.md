# US Standards Compliance Guide

## Overview

This document provides comprehensive guidance on Nethical's compliance with US AI governance and security standards:

- **NIST AI Risk Management Framework (AI RMF)**
- **HIPAA (Health Insurance Portability and Accountability Act)**
- **SOC 2 (System and Organization Controls 2)**
- **NIST 800-53 Security Controls**

## NIST AI Risk Management Framework (AI RMF 1.0)

The NIST AI RMF organizes AI risk management into four core functions:

### GOVERN Function

Cultivating and implementing an AI risk management culture.

| Subcategory | Requirement | Nethical Module | Evidence |
|-------------|-------------|-----------------|----------|
| GOVERN 1 | Policies and procedures | `nethical/core/governance.py` | `tests/test_governance_features.py` |
| GOVERN 2 | Accountability structures | `nethical/governance/human_review.py` | `tests/test_integrated_governance.py` |
| GOVERN 3 | Workforce diversity | `nethical/core/fairness_sampler.py` | `tests/test_regionalization.py` |
| GOVERN 4 | Organizational practices | `nethical/policy/engine.py` | Documentation |

**Implementation:**

```python
from nethical.core.governance import GovernanceCore

gov = GovernanceCore()
gov.set_policy_version("v2.1.0")
gov.enable_human_oversight(required_for=["high_risk_decisions"])
```

### MAP Function

Understanding the AI system context and characterizing risks.

| Subcategory | Requirement | Nethical Module | Evidence |
|-------------|-------------|-----------------|----------|
| MAP 1 | Context establishment | `nethical/core/fairness_sampler.py` | `tests/test_regionalization.py` |
| MAP 2 | Categorization | `nethical/core/ethical_taxonomy.py` | `tests/test_phase3.py` |
| MAP 3 | Capabilities/limitations | `nethical/explainability/decision_explainer.py` | `tests/test_advanced_explainability.py` |
| MAP 4 | Risks identification | `nethical/core/risk_engine.py` | `tests/test_governance_features.py` |

**Implementation:**

```python
from nethical.core.risk_engine import RiskEngine

risk_engine = RiskEngine()
risk_profile = risk_engine.assess_action({
    "action_type": "data_access",
    "user_role": "analyst",
    "data_sensitivity": "high"
})
```

### MEASURE Function

Analyzing, assessing, and measuring AI risks.

| Subcategory | Requirement | Nethical Module | Evidence |
|-------------|-------------|-----------------|----------|
| MEASURE 1 | Risk measurement | `nethical/core/ml_blended_risk.py` | `tests/test_performance_benchmarks.py` |
| MEASURE 2 | AI testing | `nethical/governance/ethics_benchmark.py` | `tests/adversarial/` |
| MEASURE 3 | Continuous monitoring | `nethical/core/sla_monitor.py` | `tests/test_observability.py` |
| MEASURE 4 | Feedback mechanisms | `nethical/core/human_feedback.py` | `tests/test_governance_features.py` |

**Implementation:**

```python
from nethical.governance.ethics_benchmark import EthicsBenchmark

benchmark = EthicsBenchmark()
results = benchmark.evaluate_system({
    "fairness_threshold": 0.8,
    "accuracy_threshold": 0.95,
    "robustness_threshold": 0.9
})
```

### MANAGE Function

Prioritizing, responding to, and managing AI risks.

| Subcategory | Requirement | Nethical Module | Evidence |
|-------------|-------------|-----------------|----------|
| MANAGE 1 | Risk response | `nethical/core/quarantine.py` | `tests/test_phase3.py` |
| MANAGE 2 | Risk treatment | `nethical/policy/engine.py` | `tests/test_phase3.py` |
| MANAGE 3 | Incident response | `nethical/security/soc_integration.py` | `tests/test_phase4_operational_security.py` |
| MANAGE 4 | Post-deployment monitoring | `nethical/monitors/` | `tests/test_observability.py` |

**Documentation:** `docs/compliance/NIST_RMF_MAPPING.md`

## HIPAA Compliance

For healthcare AI applications, Nethical supports HIPAA requirements.

### Administrative Safeguards (§164.308)

| Requirement | Section | Nethical Module | Description |
|-------------|---------|-----------------|-------------|
| Security Management | §164.308(a)(1) | `nethical/core/risk_engine.py` | Risk analysis |
| Assigned Security Responsibility | §164.308(a)(2) | `nethical/core/rbac.py` | Role assignment |
| Workforce Security | §164.308(a)(3) | `nethical/security/auth.py` | Access control |
| Information Access Management | §164.308(a)(4) | `nethical/core/rbac.py` | Access authorization |
| Security Awareness Training | §164.308(a)(5) | `docs/TRAINING_GUIDE.md` | Training materials |
| Security Incident Procedures | §164.308(a)(6) | `nethical/security/soc_integration.py` | Incident handling |
| Contingency Plan | §164.308(a)(7) | Documentation | Business continuity |
| Evaluation | §164.308(a)(8) | `nethical/governance/ethics_benchmark.py` | Periodic assessment |

### Technical Safeguards (§164.312)

| Requirement | Section | Nethical Module | Description |
|-------------|---------|-----------------|-------------|
| Access Control | §164.312(a)(1) | `nethical/core/rbac.py` | Unique user ID, auto logoff |
| Audit Controls | §164.312(b) | `nethical/security/audit_logging.py` | Activity logging |
| Integrity Controls | §164.312(c)(1) | `nethical/core/audit_merkle.py` | ePHI integrity |
| Authentication | §164.312(d) | `nethical/security/auth.py` | Person/entity auth |
| Transmission Security | §164.312(e)(1) | `nethical/security/encryption.py` | Encryption in transit |

### Implementation Example

```python
from nethical.security.compliance import HIPAAComplianceValidator

hipaa = HIPAAComplianceValidator()

# Validate PHI access
is_authorized = hipaa.validate_phi_access(
    user_id="provider-123",
    access_type="read"
)

# Check encryption compliance
status = hipaa.check_encryption_compliance()
print(f"Encryption Status: {status.value}")
```

### HIPAA Configuration

```yaml
# policies/healthcare/core.yaml
hipaa:
  phi_encryption: required
  audit_retention_years: 6
  access_controls: role_based
  transmission_security: tls_1_3
  minimum_necessary: enforced
```

## SOC 2 Compliance

SOC 2 Trust Service Criteria implementation.

### Common Criteria (CC) Controls

| Control | Category | Nethical Module | Evidence |
|---------|----------|-----------------|----------|
| CC6.1 | Logical Access | `nethical/core/rbac.py` | `tests/test_security_hardening.py` |
| CC6.2 | Prior Authorization | `nethical/security/auth.py` | `tests/test_security_hardening.py` |
| CC6.3 | Role-based Access | `nethical/core/rbac.py` | `tests/test_security_hardening.py` |
| CC6.6 | Security Monitoring | `nethical/security/anomaly_detection.py` | `tests/test_phase4_operational_security.py` |
| CC6.7 | Transmission Protection | `nethical/security/encryption.py` | `tests/test_security_hardening.py` |
| CC6.8 | Malicious Software | `nethical/security/threat_modeling.py` | `tests/test_phase5_threat_modeling.py` |
| CC7.1 | Change Management | `nethical/core/policy_diff.py` | `tests/test_train_governance.py` |
| CC7.2 | Change Monitoring | `nethical/monitors/` | `tests/test_observability.py` |
| CC7.3 | Change Testing | `nethical/governance/ethics_benchmark.py` | All test suites |
| CC7.4 | Incident Response | `nethical/security/soc_integration.py` | `tests/test_phase4_operational_security.py` |

### Availability Criteria (A)

| Control | Requirement | Nethical Module |
|---------|-------------|-----------------|
| A1.1 | Capacity planning | `nethical/core/load_balancer.py` |
| A1.2 | Backup recovery | `nethical/storage/` |
| A1.3 | Recovery testing | Test documentation |

### Confidentiality Criteria (C)

| Control | Requirement | Nethical Module |
|---------|-------------|-----------------|
| C1.1 | Confidential data identification | `nethical/core/redaction_pipeline.py` |
| C1.2 | Confidential data disposal | `nethical/core/data_minimization.py` |

### Processing Integrity (PI)

| Control | Requirement | Nethical Module |
|---------|-------------|-----------------|
| PI1.1 | Accurate processing | `nethical/governance/ethics_benchmark.py` |
| PI1.2 | Complete processing | `nethical/core/audit_merkle.py` |

### Privacy Criteria (P)

| Control | Requirement | Nethical Module |
|---------|-------------|-----------------|
| P1.1-P8.1 | Privacy principles | `nethical/security/data_compliance.py` |

## NIST 800-53 Security Controls

Nethical maps to relevant NIST 800-53 Rev 5 controls.

### Access Control (AC) Family

| Control | Title | Nethical Module |
|---------|-------|-----------------|
| AC-1 | Access Control Policy | `nethical/policy/engine.py` |
| AC-2 | Account Management | `nethical/security/auth.py` |
| AC-3 | Access Enforcement | `nethical/core/rbac.py` |
| AC-6 | Least Privilege | `nethical/core/rbac.py` |

### Audit and Accountability (AU) Family

| Control | Title | Nethical Module |
|---------|-------|-----------------|
| AU-2 | Event Logging | `nethical/security/audit_logging.py` |
| AU-3 | Content of Audit Records | `nethical/core/audit_merkle.py` |
| AU-6 | Audit Review and Reporting | `nethical/explainability/transparency_report.py` |
| AU-12 | Audit Record Generation | `nethical/security/audit_logging.py` |

### Identification and Authentication (IA) Family

| Control | Title | Nethical Module |
|---------|-------|-----------------|
| IA-2 | Identification and Authentication | `nethical/security/auth.py` |
| IA-5 | Authenticator Management | `nethical/security/mfa.py` |

### System and Communications Protection (SC) Family

| Control | Title | Nethical Module |
|---------|-------|-----------------|
| SC-8 | Transmission Confidentiality | `nethical/security/encryption.py` |
| SC-13 | Cryptographic Protection | `nethical/security/encryption.py` |
| SC-28 | Protection of Information at Rest | `nethical/security/encryption.py` |

### Incident Response (IR) Family

| Control | Title | Nethical Module |
|---------|-------|-----------------|
| IR-4 | Incident Handling | `nethical/security/soc_integration.py` |
| IR-5 | Incident Monitoring | `nethical/security/anomaly_detection.py` |
| IR-6 | Incident Reporting | `nethical/security/soc_integration.py` |

**Documentation:** `nethical/security/compliance.py` (NIST80053ControlMapper)

## FedRAMP Continuous Monitoring

For federal deployments, Nethical supports FedRAMP requirements.

```python
from nethical.security.compliance import FedRAMPMonitor

monitor = FedRAMPMonitor()

# Collect security metrics
metrics = monitor.collect_security_metrics()

# Generate POA&M
poam = monitor.generate_poam()

# Generate monthly report
report = monitor.generate_monthly_report()
```

## Regional Configuration

Configure Nethical for US deployment:

```yaml
# policies/healthcare/regions.yaml
US:
  compliance: [HIPAA, SOC2]
  locales: [en_US]
  data_residency_required: true
  export_controls_required: false
  default_geofencing_policy: OPEN
```

## Compliance Verification

```python
from nethical.security.regulatory_compliance import (
    USStandardsCompliance,
    RegulatoryMappingGenerator
)

# Initialize US compliance checker
us_compliance = USStandardsCompliance()

# Get all requirements
for req_id, req in us_compliance.requirements.items():
    print(f"{req_id}: {req.title}")

# Generate compliance report
generator = RegulatoryMappingGenerator()
mapping = generator.generate_mapping_table()

# Export for auditors
generator.generate_json_report("compliance_export.json")
```

## Audit Preparation Checklist

For SOC 2 Type II or HIPAA audits:

- [ ] Generate regulatory mapping table
- [ ] Export audit trail for review period
- [ ] Compile test evidence packages
- [ ] Document risk assessment results
- [ ] Prepare incident response documentation
- [ ] Review access control logs
- [ ] Validate encryption configuration
- [ ] Complete NIST RMF self-assessment

---
**Version:** 1.0.0  
**Last Updated:** 2025-11-25  
**Standard References:**
- NIST AI RMF 1.0 (January 2023)
- HIPAA Security Rule (45 CFR Part 164)
- SOC 2 Trust Service Criteria (2017)
- NIST SP 800-53 Rev 5
