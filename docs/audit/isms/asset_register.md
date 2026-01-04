# Asset Register

**Document ID:** ISMS-REG-001  
**Version:** 1.0  
**Classification:** Internal  
**ISO 27001 Control:** A.5.9

---

## 1. Purpose

This Asset Register provides an inventory of information and associated assets that are critical to the operation of Nethical. It supports risk assessment, access control, and incident response activities.

## 2. Asset Categories

### 2.1 Software Assets

| Asset ID | Asset Name | Description | Owner | Classification | Location |
|----------|------------|-------------|-------|----------------|----------|
| SW-001 | Nethical Core | Main governance engine | Development Team | Confidential | GitHub Repository |
| SW-002 | Nethical API | REST API interface | Development Team | Confidential | GitHub Repository |
| SW-003 | Nethical Security Module | Security controls and encryption | Security Team | Confidential | GitHub Repository |
| SW-004 | Nethical Detectors | Adversarial and safety detectors | Development Team | Confidential | GitHub Repository |
| SW-005 | Nethical Observability | Monitoring and metrics | Operations Team | Internal | GitHub Repository |

### 2.2 Data Assets

| Asset ID | Asset Name | Description | Owner | Classification | Retention |
|----------|------------|-------------|-------|----------------|-----------|
| DA-001 | Audit Logs | Security and governance audit trails | Security Team | Confidential | 7 years |
| DA-002 | Configuration Data | System and policy configuration | Operations Team | Confidential | Indefinite |
| DA-003 | Model Artifacts | ML model files and weights | ML Team | Confidential | As required |
| DA-004 | User Data | User profiles and access data | Product Team | PII/Confidential | Per policy |
| DA-005 | Training Data | ML training datasets | ML Team | Varies | As required |

### 2.3 Documentation Assets

| Asset ID | Asset Name | Description | Owner | Classification | Location |
|----------|------------|-------------|-------|----------------|----------|
| DOC-001 | Security Policies | ISMS policy documents | Security Team | Internal | docs/compliance/isms/ |
| DOC-002 | Technical Documentation | Architecture and API docs | Development Team | Public/Internal | docs/ |
| DOC-003 | Compliance Documentation | Regulatory mapping and evidence | Compliance Team | Internal | docs/compliance/ |
| DOC-004 | Operational Runbooks | Incident response and ops procedures | Operations Team | Internal | docs/ops/ |

### 2.4 Infrastructure Assets

| Asset ID | Asset Name | Description | Owner | Classification | Provider |
|----------|------------|-------------|-------|----------------|----------|
| INF-001 | Source Code Repository | GitHub repository | Development Team | Confidential | GitHub |
| INF-002 | CI/CD Pipeline | Build and deployment automation | DevOps Team | Internal | GitHub Actions |
| INF-003 | Container Registry | Docker image storage | DevOps Team | Internal | Varies |
| INF-004 | Secrets Management | API keys and credentials | Security Team | Highly Confidential | Varies |

### 2.5 Third-Party Dependencies

| Asset ID | Asset Name | Version | License | Security Status | SBOM Reference |
|----------|------------|---------|---------|-----------------|----------------|
| DEP-001 | numpy | >=1.24.0 | BSD | Monitored | SBOM.json |
| DEP-002 | scikit-learn | >=1.3.0 | BSD | Monitored | SBOM.json |
| DEP-003 | pydantic | >=2.6.1 | MIT | Monitored | SBOM.json |
| DEP-004 | fastapi | >=0.110.0 | MIT | Monitored | SBOM.json |
| DEP-005 | cryptography | * | Apache/BSD | Monitored | SBOM.json |

## 3. Classification Scheme

| Level | Description | Handling Requirements |
|-------|-------------|----------------------|
| **Public** | Information that can be shared publicly | No special handling |
| **Internal** | For internal use only | Access control required |
| **Confidential** | Business-sensitive information | Encryption, access logging |
| **Highly Confidential** | Critical secrets and PII | Strong encryption, MFA, audit |
| **PII** | Personally Identifiable Information | GDPR/CCPA compliance required |

## 4. Asset Lifecycle Management

### 4.1 Asset Identification

New assets shall be:
- Identified and documented during project initiation
- Classified according to the classification scheme
- Assigned an owner responsible for its security

### 4.2 Asset Review

Assets shall be reviewed:
- Annually as part of the ISMS review
- When significant changes occur
- When ownership changes

### 4.3 Asset Disposal

When assets are no longer required:
- Data shall be securely deleted per retention policies
- Access rights shall be revoked
- Asset shall be removed from this register

## 5. SBOM (Software Bill of Materials)

A complete list of software dependencies is maintained in:
- `SBOM.json` (root directory)
- Generated in SPDX and CycloneDX formats

Reference: `docs/SUPPLY_CHAIN_SECURITY_GUIDE.md`

## 6. Related Controls

| Control | Reference |
|---------|-----------|
| A.5.9 | Inventory of information and other associated assets |
| A.5.10 | Acceptable use of information and other associated assets |
| A.5.12 | Classification of information |
| A.5.13 | Labelling of information |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-26 | Security Team | Initial version |

**Last Review:** 2025-11-26  
**Next Review:** 2026-11-26  
**Owner:** Security Team
