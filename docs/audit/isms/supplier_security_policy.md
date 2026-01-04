# Supplier Security Policy

**Document ID:** ISMS-POL-004  
**Version:** 1.0  
**Classification:** Internal  
**ISO 27001 Controls:** A.5.19, A.5.20, A.5.21, A.5.22

---

## 1. Purpose

This Supplier Security Policy establishes the requirements for managing information security risks in supplier relationships, including third-party software dependencies, cloud services, and development tools.

## 2. Scope

This policy applies to:
- Software dependencies and libraries
- Cloud service providers
- Development tools and platforms
- Contracted development services
- Any third party with access to Nethical systems or data

## 3. Supplier Categories

### 3.1 Software Dependencies

Third-party libraries and frameworks used in Nethical:
- Tracked in `requirements.txt` and `pyproject.toml`
- Documented in SBOM (Software Bill of Materials)
- Subject to vulnerability scanning

### 3.2 Cloud Services

Cloud platforms and services used for deployment:
- Infrastructure as a Service (IaaS)
- Platform as a Service (PaaS)
- Software as a Service (SaaS)

### 3.3 Development Tools

Tools used in the development lifecycle:
- Source control (GitHub)
- CI/CD platforms (GitHub Actions)
- Security scanning tools

### 3.4 Contracted Development

External developers or contractors contributing to Nethical:
- Subject to contribution guidelines
- Must follow secure coding practices
- Code subject to security review

## 4. Supplier Selection

### 4.1 Security Assessment

Before selecting a supplier, assess:
- Security certifications (ISO 27001, SOC 2)
- Vulnerability disclosure practices
- Update and patch frequency
- License compatibility

### 4.2 Open Source Dependencies

For open source dependencies, verify:
- Active maintenance status
- Community security responsiveness
- Known vulnerability history
- License compatibility (prefer MIT, BSD, Apache)

### 4.3 Selection Criteria

| Criterion | Minimum Requirement |
|-----------|-------------------|
| Security certifications | Preferred: ISO 27001, SOC 2 |
| Vulnerability response | Defined disclosure process |
| Update frequency | Active maintenance |
| License | Open source compatible |

## 5. Supplier Agreements

### 5.1 Required Terms

Supplier agreements should include:
- Confidentiality obligations
- Data protection requirements
- Security incident notification
- Right to audit
- Termination provisions

### 5.2 Data Processing

Suppliers processing personal data must:
- Sign data processing agreements (GDPR Art. 28)
- Implement appropriate security measures
- Not sub-process without approval
- Support data subject rights

## 6. Supply Chain Security

### 6.1 SBOM (Software Bill of Materials)

Nethical maintains:
- Complete dependency inventory in `SBOM.json`
- Regular SBOM updates on releases
- SPDX and CycloneDX format support

**Reference:** `SBOM.json`, `docs/SUPPLY_CHAIN_SECURITY_GUIDE.md`

### 6.2 Dependency Scanning

All dependencies are scanned for:
- Known vulnerabilities (CVE)
- License compliance
- Malicious packages

**Tools:** Dependabot, Trivy, dependency-review-action

### 6.3 Artifact Signing

Where applicable:
- Sign released artifacts
- Verify signatures on dependencies
- Use package lock files

### 6.4 Reproducible Builds

Support for:
- Pinned dependency versions
- Build reproducibility
- Provenance attestation

## 7. Ongoing Monitoring

### 7.1 Vulnerability Monitoring

- Automated alerts for dependency vulnerabilities
- Regular security updates
- Patch SLA based on severity

| Severity | Patch SLA |
|----------|-----------|
| Critical | 24-48 hours |
| High | 7 days |
| Medium | 30 days |
| Low | Next release |

### 7.2 Supplier Review

Review suppliers:
- Annually for high-risk suppliers
- After security incidents
- When service changes significantly

### 7.3 Dependency Updates

- Regular dependency updates (monthly minimum)
- Security patches prioritized
- Update testing required

## 8. Incident Management

### 8.1 Supplier Incidents

When a supplier has a security incident:
1. Assess impact on Nethical
2. Implement mitigations
3. Document the incident
4. Review supplier status

### 8.2 Vulnerability Disclosure

When vulnerabilities are found in dependencies:
1. Report to supplier (responsible disclosure)
2. Implement workarounds if needed
3. Track fix timeline
4. Update when patch available

## 9. Termination

### 9.1 Exit Planning

For each critical supplier:
- Identify alternatives
- Document migration paths
- Maintain data portability

### 9.2 Dependency Replacement

When replacing a dependency:
- Security review new dependency
- Update SBOM
- Test thoroughly
- Document change

## 10. Related Documents

- [SBOM](../../../SBOM.json)
- [Supply Chain Security Guide](../../SUPPLY_CHAIN_SECURITY_GUIDE.md)
- [Asset Register](./asset_register.md)
- [Change Management Policy](./change_management_policy.md)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-26 | Security Team | Initial version |

**Approved By:** [Management Representative]  
**Approval Date:** [Date]  
**Next Review:** 2026-11-26
