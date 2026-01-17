# Cyber Resilience Act (CRA) Compliance Guide

## 1. Product Classification

**Product Type:** Software Component / Digital Service  
**Classification:** Nethical AI Safety & Governance Framework  
**Risk Level:** Important (Class II)  
**Justification:** Nethical is security-critical software used for AI system governance and safety validation. While not classified as "Critical" (e.g., industrial control systems), it performs important cybersecurity functions and is used in systems where security vulnerabilities could have significant impacts.

**Annex III Criteria Applied:**
- Security management and monitoring software
- Software used in critical infrastructure protection
- AI system governance and validation tools

---

## 2. Essential Cybersecurity Requirements (Annex I)

### Part I: Cybersecurity Properties

#### 2.1 Secure by Default Configuration

**Requirement:** Products must be delivered with secure configuration as default.

**Implementation:**
- Default deny policies for agent actions
- Mandatory authentication and authorization
- Encrypted communications (TLS 1.3+)
- Minimal permissions and privilege separation
- Secure defaults in `nethical/config/security_defaults.py`

**Evidence:**
- Configuration files enforce secure defaults
- Security hardening guide: `SECURITY.md`
- Default policy templates in `policies/` directory

#### 2.2 Protection Against Unauthorized Access

**Requirement:** Products must protect against unauthorized access to functions and data.

**Implementation:**
- Role-Based Access Control (RBAC) in `nethical/core/rbac.py`
- API authentication via JWT tokens
- Session management with secure token handling
- Audit logging of all access attempts

**Evidence:**
- RBAC implementation and tests
- Authentication flow documentation
- Access control tests in `tests/security/`

#### 2.3 Confidentiality, Integrity, Availability

**Requirement:** Ensure CIA triad for data and functions.

**Implementation:**
- **Confidentiality:** Encryption at rest and in transit, differential privacy
- **Integrity:** Merkle tree anchoring for audit trails, cryptographic signatures
- **Availability:** Redundancy, graceful degradation, circuit breakers

**Evidence:**
- Encryption implementation: `nethical/core/differential_privacy.py`
- Audit anchoring: `nethical/core/audit_merkle.py`
- High availability design: `docs/architecture/`

#### 2.4 Minimize Attack Surface

**Requirement:** Reduce potential attack vectors.

**Implementation:**
- Minimal dependencies (see `requirements.txt`)
- Dependency pinning with hash verification (`requirements-hashed.txt`)
- Optional features disabled by default
- Input validation and sanitization throughout
- Principle of least privilege

**Evidence:**
- Dependency management process in `requirements.txt` header
- Input validation in detectors and API endpoints
- Security testing in `tests/security/`

#### 2.5 Secure Update Mechanisms

**Requirement:** Updates must be secure and verifiable.

**Implementation:**
- Signed releases with GPG signatures
- Package integrity verification via PyPI hash checking
- Version pinning for reproducible builds
- Update notifications through security mailing list

**Evidence:**
- Release process documented in `CONTRIBUTING.md`
- Signed release tags in Git repository
- PyPI package with signature verification

---

### Part II: Vulnerability Handling

#### 2.6 Vulnerability Disclosure Policy

**Requirement:** Clear process for reporting and handling vulnerabilities.

**Implementation:**
- Security policy documented in `SECURITY.md`
- Private security reporting via GitHub Security Advisories
- Response SLA: Acknowledge within 48 hours, patch within 30 days (critical)
- CVE assignment for confirmed vulnerabilities

**Evidence:**
- `SECURITY.md` file with disclosure policy
- GitHub Security Advisory configuration
- Historical security advisory records

#### 2.7 Security Update Process

**Requirement:** Timely security updates and patches.

**Implementation:**
- Automated dependency scanning via Dependabot
- Security-first patch releases (semantic versioning)
- Backward-compatible security fixes where possible
- Security changelog in `CHANGELOG.md`

**Evidence:**
- Dependabot configuration in `.github/dependabot.yml`
- Release history showing security patches
- Update notification system

#### 2.8 Incident Response Procedures

**Requirement:** Documented incident response process.

**Implementation:**
- Incident response policy in `docs/compliance/INCIDENT_RESPONSE_POLICY.md`
- 24-hour notification for actively exploited vulnerabilities
- 72-hour notification for other security incidents
- Coordination with CSIRT and ENISA as required

**Evidence:**
- Incident response documentation
- Historical incident response records
- CSIRT contact procedures

#### 2.9 Coordinated Vulnerability Disclosure (CVD)

**Requirement:** Support coordinated disclosure with security researchers.

**Implementation:**
- 90-day disclosure timeline for non-critical issues
- Immediate disclosure for actively exploited vulnerabilities
- Security researcher acknowledgment program
- Bug bounty program (planned)

**Evidence:**
- CVD policy in `SECURITY.md`
- Security researcher hall of fame
- Responsible disclosure acknowledgments

---

## 3. CE Marking Requirements

### 3.1 Conformity Assessment Procedure

**Applicable Procedure:** Module A (Internal Production Control) for Class II products

**Process:**
1. Conduct internal security assessment
2. Prepare technical documentation (Section 4)
3. Execute conformity assessment against Annex I requirements
4. Draft EU Declaration of Conformity
5. Affix CE marking to product documentation

**Status:** In progress - Documentation being compiled

### 3.2 EU Declaration of Conformity

**Required Elements:**
- Product identification: Nethical AI Safety Framework
- Manufacturer: [Organization Name]
- Sole responsibility declaration
- Conformity with CRA essential requirements (Annex I)
- Applied harmonized standards
- Notified body details (if applicable)
- Place and date of issue
- Authorized signatory

**Template:** Available in `docs/compliance/conformity_assessment/`

### 3.3 Technical Documentation Requirements

**Documentation Package Includes:**
1. Product description and intended use
2. Design and manufacturing specifications
3. Risk assessment and management
4. Cybersecurity requirements conformity assessment
5. Software Bill of Materials (SBOM)
6. Test reports and security audits
7. User documentation
8. Support and maintenance plan

**Location:** `docs/compliance/conformity_assessment/technical_documentation/`

### 3.4 CE Marking Affixing

**Requirements:**
- Visible, legible, and indelible CE marking
- Followed by identification number (if notified body involved)
- Affixed to product packaging and documentation
- Digital CE marking for software products

**Implementation:** CE marking in README, package metadata, and online documentation

---

## 4. Security Documentation

### 4.1 SBOM (Software Bill of Materials)

**Format:** CycloneDX JSON  
**Location:** `SBOM.json` (repository root)  
**Update Frequency:** Every release  
**Generation Tool:** `cyclonedx-python` or manual compilation

**SBOM Contents:**
- All direct dependencies with versions
- Transitive dependencies
- License information
- Known vulnerabilities (via CVE references)
- Package hashes for verification

**Maintenance:**
```bash
# Generate SBOM
pip install cyclonedx-bom
cyclonedx-py -o SBOM.json
```

### 4.2 Secure Development Lifecycle

**Practices Implemented:**
1. **Threat Modeling:** Identify security threats during design
2. **Secure Coding:** Follow OWASP guidelines, input validation
3. **Code Review:** Mandatory peer review for all changes
4. **Security Testing:** Automated and manual security testing
5. **Dependency Management:** Regular updates, vulnerability scanning
6. **Incident Response:** Documented process for security incidents

**Documentation:** `docs/security/secure_development_lifecycle.md`

### 4.3 Security Testing Reports

**Testing Coverage:**
- Static Application Security Testing (SAST): CodeQL, Bandit
- Dependency Vulnerability Scanning: Dependabot, pip-audit
- Penetration Testing: Annual third-party assessment
- Fuzzing: Input validation fuzzing for critical components
- Compliance Testing: OWASP Top 10, CWE Top 25

**Reports Location:** `docs/compliance/conformity_assessment/security_testing/`

### 4.4 Vulnerability Management Process

**Process:**
1. **Discovery:** Automated scanning, security research, user reports
2. **Triage:** Assess severity using CVSS v3.1
3. **Remediation:** Develop and test patches
4. **Release:** Security patch release with advisory
5. **Disclosure:** Public disclosure after patch availability
6. **Monitoring:** Track exploitation and user updates

**Tool Integration:**
- GitHub Security Advisories
- CVE/NVD database
- CSIRT coordination
- Security mailing list

---

## 5. Support Duration

### 5.1 Security Support Period

**Policy:** Minimum 3 years from initial release date for each major version

**Current Support Matrix:**
| Version | Release Date | End of Support | Security Updates |
|---------|--------------|----------------|------------------|
| 2.x     | 2024-01-01   | 2027-01-01     | Active           |
| 1.x     | 2023-01-01   | 2026-01-01     | Security only    |

### 5.2 Update Policy and Schedule

**Regular Updates:**
- Security patches: As needed (within 30 days of disclosure)
- Bug fixes: Monthly minor releases
- Feature updates: Quarterly minor releases
- Major versions: Annually

**Update Channels:**
- PyPI package updates
- GitHub releases
- Security mailing list notifications
- RSS/Atom feed

### 5.3 End-of-Life Notification Process

**Timeline:**
- 12 months advance notice before end-of-life
- 6 months migration guide publication
- 3 months final reminder
- End-of-life date announcement

**Notification Channels:**
- Email to registered users
- Blog post and social media
- Banner in documentation
- Deprecation warnings in software

---

## 6. Incident Reporting

### 6.1 Reporting Timeline

**DSA/CRA Requirements:**
- **24 hours:** Initial notification of actively exploited vulnerabilities
- **72 hours:** Detailed incident report for security breaches
- **30 days:** Final incident report with remediation

### 6.2 Notification to CSIRT/ENISA

**Process:**
1. Detect security incident
2. Assess severity and impact
3. Notify appropriate CSIRT within 24 hours (critical) or 72 hours (other)
4. Provide initial assessment and affected systems
5. Submit detailed report with remediation plan
6. Final report after incident resolution

**Contact Points:**
- National CSIRT (vary by EU member state)
- ENISA (European Union Agency for Cybersecurity)
- CERT-EU for EU institutions

### 6.3 User Notification Requirements

**Trigger Events:**
- Active exploitation of vulnerability
- Data breach affecting user information
- Significant security degradation
- Mandatory security update required

**Notification Method:**
- Security advisory via GitHub and website
- Email to all registered users
- In-app notification (if applicable)
- Public disclosure after patch availability

**Content:**
- Nature of the incident
- Affected versions
- Remediation steps
- Timeline for resolution

---

## 7. Compliance Evidence

### 7.1 Implementation Artifacts

**Code Implementations:**
- UK OSA Compliance: `nethical/compliance/uk_osa.py`
- DSA Compliance: `nethical/compliance/dsa.py`
- CRA Compliance: `nethical/compliance/cra.py`
- Security Module: `nethical/security/`
- RBAC System: `nethical/core/rbac.py`
- Audit Logging: `nethical/core/audit_merkle.py`

**Test Coverage:**
- Security tests: `tests/security/`
- Compliance tests: `tests/test_*_compliance.py`
- Integration tests: `tests/core/`
- Penetration tests: `tests/resilience/`

**Documentation:**
- Security hardening: `SECURITY.md`
- Privacy policy: `PRIVACY.md`
- Compliance mapping: `docs/compliance/REGULATORY_MAPPING_TABLE.md`
- Audit reports: `docs/compliance/audit_report.json`

### 7.2 Conformity Assessment Records

**Assessment Date:** [To be completed during formal assessment]  
**Assessor:** [Internal/External auditor]  
**Assessment Report:** `docs/compliance/conformity_assessment/assessment_report.pdf`

**Checklist Coverage:**
- [x] Secure by default configuration
- [x] Access control implementation
- [x] Encryption and data protection
- [x] Vulnerability disclosure policy
- [x] Security update process
- [x] Incident response procedures
- [x] SBOM generation and maintenance
- [x] Technical documentation
- [ ] Third-party security audit (planned)
- [ ] CE marking affixing (pending formal assessment)

### 7.3 Continuous Compliance Monitoring

**Automated Checks:**
- Daily dependency vulnerability scans
- Weekly security test suite execution
- Monthly compliance audit
- Quarterly penetration testing

**Manual Reviews:**
- Annual third-party security audit
- Semi-annual compliance review
- Incident-driven assessments

**Metrics Tracked:**
- Mean time to patch (MTTP)
- Vulnerability density
- Security test coverage
- Incident response time

---

## 8. Contact Information

**Security Contact:** security@nethical.org (to be established)  
**Compliance Officer:** compliance@nethical.org (to be established)  
**Vulnerability Reports:** Use GitHub Security Advisories or security@nethical.org

**Regulatory Coordination:**
- Digital Services Coordinator: [Country-specific]
- ENISA: cert@enisa.europa.eu
- National CSIRT: [Country-specific]

---

## 9. Revision History

| Version | Date       | Changes                        | Author |
|---------|------------|--------------------------------|--------|
| 1.0     | 2024-01-14 | Initial CRA compliance guide   | Nethical Team |

---

## 10. References

- **Cyber Resilience Act:** [Regulation (EU) 2024/XXX](https://www.europarl.europa.eu/doceo/document/TA-9-2024-0130_EN.pdf)
- **CRA Annex I:** Essential Cybersecurity Requirements
- **CRA Annex III:** Product Classification Criteria
- **ENISA Guidelines:** [Cybersecurity for Products](https://www.enisa.europa.eu/)
- **ISO/IEC 27001:** Information Security Management
- **NIST Cybersecurity Framework:** [CSF 2.0](https://www.nist.gov/cyberframework)

---

**Document Status:** Living Document - Updated with each release  
**Next Review:** Quarterly or upon significant changes  
**Compliance Status:** In Progress - Conformity assessment ongoing
