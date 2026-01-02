# ISO/IEC 27001:2022 Annex A Controls Mapping

## Overview

This document provides a comprehensive mapping of Nethical's security controls, code modules, documentation, and tests to the ISO/IEC 27001:2022 Annex A controls. This mapping is designed to support certification audits and ongoing ISMS maintenance.

**Document Version:** 1.0  
**Last Updated:** 2025-11-26  
**Status:** Active  
**Owner:** Security Team

---

## Quick Reference

| Category | Controls | Implemented | Partial | Gap |
|----------|----------|-------------|---------|-----|
| A.5 Organizational controls | 37 | 32 | 4 | 1 |
| A.6 People controls | 8 | 6 | 2 | 0 |
| A.7 Physical controls | 14 | 5 | 2 | 7 |
| A.8 Technological controls | 34 | 31 | 2 | 1 |
| **Total** | **93** | **74** | **10** | **9** |

---

## A.5 Organizational Controls

### A.5.1 Policies for Information Security

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.1 | Policies for information security | ✅ Implemented | Information security policies documented and approved | `SECURITY.md`, `docs/SECURITY_HARDENING_GUIDE.md`, `docs/security/threat_model.md` |

**Code Modules:** `nethical/policy/engine.py`  
**Tests:** `tests/test_phase3.py`  
**ISMS Artifacts:** `docs/compliance/isms/information_security_policy.md`

### A.5.2 Information Security Roles and Responsibilities

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.2 | Information security roles and responsibilities | ✅ Implemented | RACI matrix defined, RBAC implemented | `nethicalplan.md`, `nethical/core/rbac.py` |

**Code Modules:** `nethical/core/rbac.py`, `nethical/security/auth.py`  
**Tests:** `tests/test_security_hardening.py`

### A.5.3 Segregation of Duties

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.3 | Segregation of duties | ✅ Implemented | RBAC with role separation, multi-tenant isolation | `nethical/core/rbac.py` |

**Code Modules:** `nethical/core/rbac.py`  
**Tests:** `tests/test_phase4_operational_security.py`

### A.5.4 Management Responsibilities

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.4 | Management responsibilities | 🟡 Partial | Management commitment documented, ongoing review processes | `nethicalplan.md`, `CONTRIBUTING.md` |

**ISMS Artifacts:** `docs/compliance/isms/management_commitment.md`

### A.5.5 Contact with Authorities

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.5 | Contact with authorities | ✅ Implemented | Incident response procedures include authority contact | `docs/compliance/INCIDENT_RESPONSE_POLICY.md` |

**Code Modules:** `nethical/security/soc_integration.py`

### A.5.6 Contact with Special Interest Groups

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.6 | Contact with special interest groups | ✅ Implemented | Security community engagement, vulnerability disclosure | `SECURITY.md`, `CONTRIBUTING.md` |

### A.5.7 Threat Intelligence

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.7 | Threat intelligence | ✅ Implemented | Threat modeling, adversarial testing, security scanning | `docs/security/threat_model.md`, `nethical/security/threat_modeling.py` |

**Code Modules:** `nethical/security/threat_modeling.py`, `nethical/security/penetration_testing.py`  
**Tests:** `tests/test_phase5_threat_modeling.py`, `tests/adversarial/`

### A.5.8 Information Security in Project Management

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.8 | Information security in project management | ✅ Implemented | Security built into SDLC, CI/CD security scans | `.github/workflows/`, `docs/SUPPLY_CHAIN_SECURITY_GUIDE.md` |

**Tests:** CI/CD security scanning workflows

### A.5.9 Inventory of Information and Other Associated Assets

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.9 | Inventory of information and other associated assets | ✅ Implemented | Asset inventory, SBOM generation | `SBOM.json`, `docs/compliance/isms/asset_register.md` |

**Code Modules:** SBOM generation in CI/CD  
**ISMS Artifacts:** `docs/compliance/isms/asset_register.md`

### A.5.10 Acceptable Use of Information and Other Associated Assets

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.10 | Acceptable use of information and other associated assets | ✅ Implemented | Data classification, usage policies | `policies/common/data_classification.yaml` |

**ISMS Artifacts:** `docs/compliance/isms/acceptable_use_policy.md`

### A.5.11 Return of Assets

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.11 | Return of assets | 🟡 Partial | Process documented, offboarding procedures | `docs/compliance/isms/asset_return_procedure.md` |

### A.5.12 Classification of Information

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.12 | Classification of information | ✅ Implemented | Data classification scheme implemented | `policies/common/data_classification.yaml`, `nethical/security/data_compliance.py` |

**Code Modules:** `nethical/security/data_compliance.py`

### A.5.13 Labelling of Information

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.13 | Labelling of information | ✅ Implemented | Automated data labeling, PII detection | `nethical/core/redaction_pipeline.py` |

**Code Modules:** `nethical/core/redaction_pipeline.py`  
**Tests:** `tests/test_privacy_features.py`

### A.5.14 Information Transfer

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.14 | Information transfer | ✅ Implemented | Encryption in transit, secure API design | `nethical/security/encryption.py`, `docs/api/API_USAGE.md` |

**Code Modules:** `nethical/security/encryption.py`

### A.5.15 Access Control

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.15 | Access control | ✅ Implemented | RBAC, authentication, authorization | `nethical/core/rbac.py`, `nethical/security/auth.py` |

**Code Modules:** `nethical/core/rbac.py`, `nethical/security/auth.py`, `nethical/security/sso.py`  
**Tests:** `tests/test_security_hardening.py`  
**Docs:** `docs/security/SSO_SAML_GUIDE.md`, `docs/security/MFA_GUIDE.md`

### A.5.16 Identity Management

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.16 | Identity management | ✅ Implemented | SSO/SAML integration, MFA support | `nethical/security/sso.py`, `nethical/security/mfa.py` |

**Code Modules:** `nethical/security/sso.py`, `nethical/security/mfa.py`, `nethical/security/authentication.py`  
**Docs:** `docs/security/SSO_SAML_GUIDE.md`, `docs/security/MFA_GUIDE.md`

### A.5.17 Authentication Information

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.17 | Authentication information | ✅ Implemented | Secure credential management, secret management | `nethical/security/secret_management.py` |

**Code Modules:** `nethical/security/secret_management.py`, `nethical/security/authentication.py`

### A.5.18 Access Rights

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.18 | Access rights | ✅ Implemented | RBAC with granular permissions | `nethical/core/rbac.py` |

**Code Modules:** `nethical/core/rbac.py`  
**Tests:** `tests/test_security_hardening.py`

### A.5.19 Information Security in Supplier Relationships

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.19 | Information security in supplier relationships | ✅ Implemented | Supply chain security, SBOM | `docs/SUPPLY_CHAIN_SECURITY_GUIDE.md`, `SBOM.json` |

**ISMS Artifacts:** `docs/compliance/isms/supplier_security_policy.md`

### A.5.20 Addressing Information Security within Supplier Agreements

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.20 | Addressing information security within supplier agreements | 🟡 Partial | Template agreements available | `docs/compliance/isms/supplier_agreement_template.md` |

### A.5.21 Managing Information Security in the ICT Supply Chain

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.21 | Managing information security in the ICT supply chain | ✅ Implemented | SBOM, dependency scanning, artifact signing | `SBOM.json`, `docs/SUPPLY_CHAIN_SECURITY_GUIDE.md` |

**Code Modules:** CI/CD dependency scanning

### A.5.22 Monitoring, Review and Change Management of Supplier Services

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.22 | Monitoring, review and change management of supplier services | ✅ Implemented | Dependency updates, vulnerability monitoring | CI/CD workflows, Dependabot |

### A.5.23 Information Security for Use of Cloud Services

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.23 | Information security for use of cloud services | ✅ Implemented | Cloud security controls, regional deployment | `docs/REGIONAL_DEPLOYMENT_GUIDE.md` |

### A.5.24 Information Security Incident Management Planning and Preparation

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.24 | Information security incident management planning and preparation | ✅ Implemented | Incident response policy, SOC integration | `docs/compliance/INCIDENT_RESPONSE_POLICY.md`, `nethical/security/soc_integration.py` |

**Code Modules:** `nethical/security/soc_integration.py`  
**Tests:** `tests/test_phase4_operational_security.py`

### A.5.25 Assessment and Decision on Information Security Events

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.25 | Assessment and decision on information security events | ✅ Implemented | Automated event assessment, anomaly detection | `nethical/security/anomaly_detection.py` |

**Code Modules:** `nethical/security/anomaly_detection.py`, `nethical/core/anomaly_detector.py`

### A.5.26 Response to Information Security Incidents

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.26 | Response to information security incidents | ✅ Implemented | Incident response procedures, escalation | `docs/compliance/INCIDENT_RESPONSE_POLICY.md` |

**Code Modules:** `nethical/core/quarantine.py`

### A.5.27 Learning from Information Security Incidents

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.27 | Learning from information security incidents | ✅ Implemented | Post-incident review, lessons learned | `docs/compliance/INCIDENT_RESPONSE_POLICY.md` |

### A.5.28 Collection of Evidence

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.28 | Collection of evidence | ✅ Implemented | Audit logging, Merkle anchoring, evidence collection | `nethical/security/audit_logging.py`, `nethical/core/audit_merkle.py` |

**Code Modules:** `nethical/security/audit_logging.py`, `nethical/core/audit_merkle.py`, `nethical/security/compliance.py`  
**Tests:** `tests/test_train_audit_logging.py`

### A.5.29 Information Security During Disruption

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.29 | Information security during disruption | ✅ Implemented | Business continuity, availability targets | `docs/PRODUCTION_READINESS_CHECKLIST.md` |

### A.5.30 ICT Readiness for Business Continuity

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.30 | ICT readiness for business continuity | ✅ Implemented | Multi-region support, 99.9% availability target | `docs/REGIONAL_DEPLOYMENT_GUIDE.md` |

### A.5.31 Legal, Statutory, Regulatory and Contractual Requirements

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.31 | Legal, statutory, regulatory and contractual requirements | ✅ Implemented | Compliance framework, regulatory mapping | `docs/compliance/REGULATORY_MAPPING_TABLE.md`, `formal/phase1/compliance_matrix.md` |

**Code Modules:** `nethical/security/compliance.py`, `nethical/security/regulatory_compliance.py`

### A.5.32 Intellectual Property Rights

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.32 | Intellectual property rights | ✅ Implemented | MIT License, SBOM, attribution | `LICENSE`, `SBOM.json` |

### A.5.33 Protection of Records

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.33 | Protection of records | ✅ Implemented | Audit log integrity, Merkle anchoring | `nethical/core/audit_merkle.py` |

**Code Modules:** `nethical/core/audit_merkle.py`

### A.5.34 Privacy and Protection of PII

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.34 | Privacy and protection of PII | ✅ Implemented | PII detection, redaction, differential privacy | `nethical/core/redaction_pipeline.py`, `nethical/core/differential_privacy.py` |

**Code Modules:** `nethical/core/redaction_pipeline.py`, `nethical/core/differential_privacy.py`, `nethical/core/data_minimization.py`  
**Tests:** `tests/test_privacy_features.py`  
**Docs:** `docs/F3_PRIVACY_GUIDE.md`, `docs/privacy/DPIA_template.md`

### A.5.35 Independent Review of Information Security

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.35 | Independent review of information security | ✅ Implemented | Audit portal, external audit readiness | `portal/audit_portal_spec.md` |

### A.5.36 Compliance with Policies, Rules and Standards for Information Security

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.36 | Compliance with policies, rules and standards for information security | ✅ Implemented | Compliance monitoring, policy enforcement | `nethical/policy/engine.py` |

**Code Modules:** `nethical/policy/engine.py`  
**Tests:** `tests/test_phase3.py`

### A.5.37 Documented Operating Procedures

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.5.37 | Documented operating procedures | ✅ Implemented | Operations documentation, runbooks | `docs/ops/`, `docs/operations/` |

---

## A.6 People Controls

### A.6.1 Screening

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.6.1 | Screening | 🔴 Gap | HR process - out of software scope | N/A - Organizational process |

**Note:** This is an organizational control outside the software scope.

### A.6.2 Terms and Conditions of Employment

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.6.2 | Terms and conditions of employment | 🔴 Gap | HR process - out of software scope | N/A - Organizational process |

**Note:** This is an organizational control outside the software scope.

### A.6.3 Information Security Awareness, Education and Training

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.6.3 | Information security awareness, education and training | ✅ Implemented | Training documentation, security guides | `docs/TRAINING_GUIDE.md`, `docs/SECURITY_HARDENING_GUIDE.md` |

### A.6.4 Disciplinary Process

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.6.4 | Disciplinary process | 🟡 Partial | HR process - template provided | `docs/compliance/isms/disciplinary_procedure.md` |

### A.6.5 Responsibilities After Termination or Change of Employment

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.6.5 | Responsibilities after termination or change of employment | 🟡 Partial | Offboarding procedures | `docs/compliance/isms/offboarding_checklist.md` |

### A.6.6 Confidentiality or Non-Disclosure Agreements

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.6.6 | Confidentiality or non-disclosure agreements | ✅ Implemented | NDA templates, confidentiality requirements | `docs/compliance/isms/nda_template.md` |

### A.6.7 Remote Working

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.6.7 | Remote working | ✅ Implemented | Zero trust security, VPN/secure access | `nethical/security/zero_trust.py` |

**Code Modules:** `nethical/security/zero_trust.py`

### A.6.8 Information Security Event Reporting

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.6.8 | Information security event reporting | ✅ Implemented | Incident reporting procedures | `docs/compliance/INCIDENT_RESPONSE_POLICY.md`, `SECURITY.md` |

---

## A.7 Physical Controls

### A.7.1 Physical Security Perimeters

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.1 | Physical security perimeters | 🔴 Gap | Infrastructure/cloud provider responsibility | N/A - Infrastructure |

### A.7.2 Physical Entry

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.2 | Physical entry | 🔴 Gap | Infrastructure/cloud provider responsibility | N/A - Infrastructure |

### A.7.3 Securing Offices, Rooms and Facilities

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.3 | Securing offices, rooms and facilities | 🔴 Gap | Infrastructure/cloud provider responsibility | N/A - Infrastructure |

### A.7.4 Physical Security Monitoring

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.4 | Physical security monitoring | 🔴 Gap | Infrastructure/cloud provider responsibility | N/A - Infrastructure |

### A.7.5 Protecting Against Physical and Environmental Threats

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.5 | Protecting against physical and environmental threats | 🔴 Gap | Infrastructure/cloud provider responsibility | N/A - Infrastructure |

### A.7.6 Working in Secure Areas

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.6 | Working in secure areas | 🟡 Partial | Secure development practices | `docs/SECURITY_HARDENING_GUIDE.md` |

### A.7.7 Clear Desk and Clear Screen

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.7 | Clear desk and clear screen | 🟡 Partial | Session timeout, screen lock policies | Application session management |

### A.7.8 Equipment Siting and Protection

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.8 | Equipment siting and protection | 🔴 Gap | Infrastructure responsibility | N/A - Infrastructure |

### A.7.9 Security of Assets Off-Premises

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.9 | Security of assets off-premises | ✅ Implemented | Encryption, secure data transfer | `nethical/security/encryption.py` |

### A.7.10 Storage Media

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.10 | Storage media | ✅ Implemented | Data encryption at rest, secure storage | `nethical/storage/` |

**Code Modules:** `nethical/storage/`

### A.7.11 Supporting Utilities

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.11 | Supporting utilities | 🔴 Gap | Infrastructure responsibility | N/A - Infrastructure |

### A.7.12 Cabling Security

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.12 | Cabling security | ✅ Implemented | Network encryption (TLS) | Infrastructure config |

### A.7.13 Equipment Maintenance

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.13 | Equipment maintenance | ✅ Implemented | Automated updates, dependency scanning | CI/CD, Dependabot |

### A.7.14 Secure Disposal or Re-Use of Equipment

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.7.14 | Secure disposal or re-use of equipment | ✅ Implemented | Data deletion, RTBF support | `nethical/core/data_minimization.py` |

---

## A.8 Technological Controls

### A.8.1 User Endpoint Devices

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.1 | User endpoint devices | ✅ Implemented | Zero trust architecture, device authentication | `nethical/security/zero_trust.py` |

### A.8.2 Privileged Access Rights

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.2 | Privileged access rights | ✅ Implemented | RBAC, privilege management | `nethical/core/rbac.py` |

**Code Modules:** `nethical/core/rbac.py`  
**Tests:** `tests/test_security_hardening.py`

### A.8.3 Information Access Restriction

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.3 | Information access restriction | ✅ Implemented | Data minimization, access control | `nethical/core/data_minimization.py`, `nethical/core/rbac.py` |

**Code Modules:** `nethical/core/data_minimization.py`  
**Tests:** `tests/test_privacy_features.py`

### A.8.4 Access to Source Code

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.4 | Access to source code | ✅ Implemented | Repository access control, branch protection | GitHub branch protection rules |

### A.8.5 Secure Authentication

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.5 | Secure authentication | ✅ Implemented | MFA, SSO/SAML, secure authentication | `nethical/security/mfa.py`, `nethical/security/sso.py` |

**Code Modules:** `nethical/security/mfa.py`, `nethical/security/sso.py`, `nethical/security/authentication.py`  
**Docs:** `docs/security/MFA_GUIDE.md`, `docs/security/SSO_SAML_GUIDE.md`

### A.8.6 Capacity Management

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.6 | Capacity management | ✅ Implemented | Quota enforcement, resource limits | `nethical/quotas.py` |

**Code Modules:** `nethical/quotas.py`  
**Tests:** `tests/test_phase4_core.py`

### A.8.7 Protection Against Malware

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.7 | Protection against malware | ✅ Implemented | Adversarial detection, input validation | `nethical/detectors/`, `nethical/security/input_validation.py` |

**Code Modules:** `nethical/detectors/`, `nethical/security/input_validation.py`  
**Tests:** `tests/adversarial/`

### A.8.8 Management of Technical Vulnerabilities

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.8 | Management of technical vulnerabilities | ✅ Implemented | Vulnerability scanning, penetration testing | `nethical/security/penetration_testing.py`, SBOM |

**Code Modules:** `nethical/security/penetration_testing.py`  
**Tests:** `tests/test_phase5_penetration_testing.py`  
**Docs:** `docs/SUPPLY_CHAIN_SECURITY_GUIDE.md`

### A.8.9 Configuration Management

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.9 | Configuration management | ✅ Implemented | Configuration management, policy versioning | `nethical/config/`, `nethical/core/policy_diff.py` |

**Code Modules:** `nethical/config/`, `nethical/core/policy_diff.py`  
**Tests:** `tests/test_train_governance.py`

### A.8.10 Information Deletion

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.10 | Information deletion | ✅ Implemented | Data deletion, RTBF, retention policies | `nethical/core/data_minimization.py` |

**Code Modules:** `nethical/core/data_minimization.py`  
**Tests:** `tests/test_privacy_features.py`

### A.8.11 Data Masking

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.11 | Data masking | ✅ Implemented | PII redaction, differential privacy | `nethical/core/redaction_pipeline.py`, `nethical/core/differential_privacy.py` |

**Code Modules:** `nethical/core/redaction_pipeline.py`, `nethical/core/differential_privacy.py`  
**Tests:** `tests/test_privacy_features.py`

### A.8.12 Data Leakage Prevention

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.12 | Data leakage prevention | ✅ Implemented | PII detection, output validation | `nethical/core/redaction_pipeline.py`, `nethical/detectors/` |

**Code Modules:** `nethical/core/redaction_pipeline.py`  
**Tests:** `tests/test_privacy_features.py`

### A.8.13 Information Backup

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.13 | Information backup | ✅ Implemented | Backup procedures, multi-region | `docs/PRODUCTION_READINESS_CHECKLIST.md` |

### A.8.14 Redundancy of Information Processing Facilities

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.14 | Redundancy of information processing facilities | ✅ Implemented | Multi-region deployment, load balancing | `nethical/core/load_balancer.py`, `docs/REGIONAL_DEPLOYMENT_GUIDE.md` |

**Code Modules:** `nethical/core/load_balancer.py`

### A.8.15 Logging

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.15 | Logging | ✅ Implemented | Comprehensive audit logging, Merkle anchoring | `nethical/security/audit_logging.py`, `nethical/core/audit_merkle.py` |

**Code Modules:** `nethical/security/audit_logging.py`, `nethical/core/audit_merkle.py`  
**Tests:** `tests/test_train_audit_logging.py`  
**Docs:** `docs/AUDIT_LOGGING_GUIDE.md`

### A.8.16 Monitoring Activities

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.16 | Monitoring activities | ✅ Implemented | Real-time monitoring, anomaly detection, observability | `nethical/monitors/`, `nethical/observability/` |

**Code Modules:** `nethical/monitors/`, `nethical/observability/`, `nethical/core/sla_monitor.py`  
**Tests:** `tests/test_observability.py`  
**Docs:** `docs/SEMANTIC_MONITORING_GUIDE.md`

### A.8.17 Clock Synchronization

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.17 | Clock synchronization | ✅ Implemented | UTC timestamps in audit logs | `nethical/security/audit_logging.py` |

### A.8.18 Use of Privileged Utility Programs

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.18 | Use of privileged utility programs | ✅ Implemented | Restricted access, audit logging | `nethical/core/rbac.py`, audit logging |

### A.8.19 Installation of Software on Operational Systems

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.19 | Installation of software on operational systems | ✅ Implemented | Controlled deployment, release management | `nethical/policy/release_management.py` |

**Code Modules:** `nethical/policy/release_management.py`  
**Docs:** `docs/versioning.md`

### A.8.20 Networks Security

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.20 | Networks security | ✅ Implemented | Network security controls, zero trust | `nethical/security/zero_trust.py` |

**Code Modules:** `nethical/security/zero_trust.py`

### A.8.21 Security of Network Services

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.21 | Security of network services | ✅ Implemented | Secure API design, rate limiting | `nethical/api/` |

**Code Modules:** `nethical/api/`

### A.8.22 Segregation of Networks

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.22 | Segregation of networks | ✅ Implemented | Multi-tenant isolation | Multi-tenant architecture |

### A.8.23 Web Filtering

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.23 | Web filtering | 🟡 Partial | Input validation, content filtering | `nethical/security/input_validation.py` |

### A.8.24 Use of Cryptography

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.24 | Use of cryptography | ✅ Implemented | Encryption, Merkle trees, quantum-resistant crypto | `nethical/security/encryption.py`, `nethical/security/quantum_crypto.py` |

**Code Modules:** `nethical/security/encryption.py`, `nethical/security/quantum_crypto.py`, `nethical/core/audit_merkle.py`  
**Docs:** `docs/security/QUANTUM_CRYPTO_GUIDE.md`

### A.8.25 Secure Development Life Cycle

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.25 | Secure development life cycle | ✅ Implemented | Secure SDLC, security scanning, code review | `.github/workflows/`, CI/CD pipelines |

### A.8.26 Application Security Requirements

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.26 | Application security requirements | ✅ Implemented | Security requirements documented | `docs/security/`, threat model |

### A.8.27 Secure System Architecture and Engineering Principles

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.27 | Secure system architecture and engineering principles | ✅ Implemented | Security architecture documented | `ARCHITECTURE.md` |

### A.8.28 Secure Coding

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.28 | Secure coding | ✅ Implemented | Code scanning, security testing | SAST, security tests |

**Tests:** `tests/adversarial/`, `tests/test_security_hardening.py`

### A.8.29 Security Testing in Development and Acceptance

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.29 | Security testing in development and acceptance | ✅ Implemented | Penetration testing, adversarial testing | `nethical/security/penetration_testing.py`, `tests/adversarial/` |

**Tests:** `tests/test_phase5_penetration_testing.py`, `tests/adversarial/`

### A.8.30 Outsourced Development

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.30 | Outsourced development | 🟡 Partial | Contribution guidelines, code review | `CONTRIBUTING.md` |

### A.8.31 Separation of Development, Test and Production Environments

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.31 | Separation of development, test and production environments | ✅ Implemented | Environment separation, config management | `nethical/config/` |

### A.8.32 Change Management

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.32 | Change management | ✅ Implemented | Policy diff, release management, version control | `nethical/core/policy_diff.py`, `nethical/policy/release_management.py` |

**Code Modules:** `nethical/core/policy_diff.py`, `nethical/policy/release_management.py`  
**Docs:** `docs/versioning.md`, `docs/compliance/isms/change_management_policy.md`

### A.8.33 Test Information

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.33 | Test information | ✅ Implemented | Test data management, anonymization | Test fixtures, differential privacy |

### A.8.34 Protection of Information Systems During Audit Testing

| Control ID | Control Name | Status | Implementation | Evidence |
|------------|--------------|--------|----------------|----------|
| A.8.34 | Protection of information systems during audit testing | ✅ Implemented | Audit controls, read-only access for auditors | Audit portal |

---

## Gap Analysis Summary

### Controls Not Applicable (Infrastructure/Organizational)

The following controls are primarily infrastructure or organizational controls that fall outside the scope of Nethical as a software platform. These would be addressed by the organization deploying Nethical or by cloud service providers:

| Control | Reason |
|---------|--------|
| A.6.1 Screening | HR organizational process |
| A.6.2 Terms and conditions of employment | HR organizational process |
| A.7.1-A.7.5, A.7.8, A.7.11 | Physical infrastructure controls |

### Remediation Plan

| Gap | Priority | Target | Action |
|-----|----------|--------|--------|
| A.5.4 Management responsibilities | Medium | ISMS docs | Create formal management review procedure |
| A.5.11 Return of assets | Low | ISMS docs | Create asset return procedure template |
| A.5.20 Supplier agreements | Medium | ISMS docs | Create supplier security agreement template |
| A.6.4 Disciplinary process | Low | ISMS docs | Create disciplinary procedure template |
| A.6.5 Termination responsibilities | Low | ISMS docs | Create offboarding checklist |
| A.8.23 Web filtering | Low | Code | Enhance input validation |
| A.8.30 Outsourced development | Low | Docs | Enhance contribution security guidelines |

---

## Evidence Collection Guide

### For Each Control

1. **Policy Documents**: Located in `docs/compliance/isms/`
2. **Code Implementation**: Reference specific modules in `nethical/`
3. **Test Evidence**: Reference test files in `tests/`
4. **Audit Logs**: Generated by `nethical/security/audit_logging.py`
5. **Configuration Evidence**: Found in `nethical/config/` and `config/`

### Audit Package Generation

Use the `EvidenceCollector` class in `nethical/security/compliance.py` to generate evidence packages:

```python
from nethical.security.compliance import EvidenceCollector

collector = EvidenceCollector()
package = collector.generate_evidence_package([
    "A.5.28",  # Collection of Evidence
    "A.8.15",  # Logging
    "A.8.24",  # Use of Cryptography
])
```

---

## Related Documents

- [NIST RMF Mapping](./NIST_RMF_MAPPING.md)
- [Regulatory Mapping Table](./REGULATORY_MAPPING_TABLE.md)
- [Incident Response Policy](./INCIDENT_RESPONSE_POLICY.md)
- [Compliance Matrix](../../formal/phase1/compliance_matrix.md)
- [ISMS Folder](./isms/)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-26 | Security Team | Initial ISO 27001:2022 Annex A mapping |

---

**Review Frequency:** Quarterly  
**Next Review:** 2026-02-26  
**Certification Target:** ISO/IEC 27001:2022
