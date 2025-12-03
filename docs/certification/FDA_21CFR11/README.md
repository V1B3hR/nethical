# FDA 21 CFR Part 11 Compliance

## Overview

This document describes Nethical's compliance with **FDA 21 CFR Part 11** - Electronic Records; Electronic Signatures. This regulation establishes requirements for electronic records and signatures in FDA-regulated industries including medical devices, pharmaceuticals, and biologics.

## Applicability

21 CFR Part 11 applies when:
- Electronic records are used to satisfy FDA predicate rules
- Electronic signatures are used in place of handwritten signatures
- The records are submitted to FDA or maintained for FDA inspection

**Nethical Context:** Medical AI governance decisions, audit trails, and policy approvals are electronic records subject to Part 11 when deployed in FDA-regulated healthcare AI applications.

## Scope of Compliance

### In Scope

| Record Type | Purpose | Part 11 Controls |
|-------------|---------|------------------|
| Governance Decisions | AI safety determinations | Full compliance |
| Audit Logs | Decision traceability | Full compliance |
| Policy Configurations | Governance rules | Full compliance |
| User Access Logs | System access tracking | Full compliance |
| Electronic Signatures | Policy approvals | Full compliance |

### Out of Scope

| Record Type | Reason |
|-------------|--------|
| Source code | Development artifact |
| Test data | Non-production |
| Marketing materials | Not FDA-regulated |

## Subpart B: Electronic Records

### § 11.10 Controls for Closed Systems

#### (a) Validation

**Requirement:** Systems shall be validated to ensure accuracy, reliability, consistent intended performance, and the ability to discern invalid or altered records.

**Implementation:**

| Control | Evidence | Location |
|---------|----------|----------|
| Installation Qualification (IQ) | Deployment verification | `deploy/validation/IQ/` |
| Operational Qualification (OQ) | Functional testing | `tests/` |
| Performance Qualification (PQ) | Production validation | `deploy/validation/PQ/` |
| Ongoing Validation | Continuous monitoring | Prometheus metrics |

**Validation Documentation:**
- System Requirements Specification
- Design Specification
- Test Protocols and Results
- Traceability Matrix
- Validation Summary Report

#### (b) Accurate and Complete Copies

**Requirement:** Generate accurate and complete copies of records in both human-readable and electronic form.

**Implementation:**

| Feature | Description |
|---------|-------------|
| Export formats | JSON, CSV, PDF |
| Completeness | All fields preserved |
| Verification | Checksum validation |
| Audit | Export events logged |

```python
# Export API
POST /v2/audit/export
Content-Type: application/json
{
    "format": "pdf",  # or "json", "csv"
    "date_range": {"start": "2025-01-01", "end": "2025-12-31"},
    "include_signatures": true
}
```

#### (c) Protection of Records

**Requirement:** Records protected throughout retention period.

**Implementation:**

| Protection | Mechanism |
|------------|-----------|
| Encryption at rest | AES-256-GCM |
| Encryption in transit | TLS 1.3 |
| Access control | RBAC + MFA |
| Backup | Encrypted, offsite |
| Retention | Configurable (default 7 years) |

#### (d) Limiting System Access

**Requirement:** Limit system access to authorized individuals.

**Implementation:**

| Control | Description |
|---------|-------------|
| Authentication | Username + password + MFA |
| Authorization | Role-based access control |
| Session management | Automatic timeout |
| Audit | All access logged |

**Access Levels:**

| Role | Permissions |
|------|-------------|
| Viewer | Read decisions, audit logs |
| Operator | Execute governance, view logs |
| Policy Author | Create/modify policies (approval required) |
| Approver | Approve policies, review changes |
| Administrator | User management, system config |
| Auditor | Read all, cannot modify |

#### (e) Secure, Computer-Generated Audit Trails

**Requirement:** Use secure, computer-generated, time-stamped audit trails to independently record date and time of operator entries and actions.

**Implementation:**

```python
# Audit Trail Entry Structure
class AuditEntry:
    event_id: UUID           # Unique identifier
    timestamp: datetime      # UTC, NTP-synchronized
    record_id: UUID          # Record affected
    action: str              # CREATE, READ, UPDATE, DELETE
    user_id: str             # Authenticated user
    previous_value: dict     # Before change
    new_value: dict          # After change
    reason: str              # Change reason
    signature: str           # Cryptographic signature
    merkle_hash: str         # Chain integrity
```

**Audit Trail Properties:**
- Immutable (append-only)
- Time-stamped with NTP-synchronized clock
- Cryptographically linked (Merkle tree)
- Independent of operator control
- Cannot be deleted or modified

#### (f) Operational System Checks

**Requirement:** Use operational system checks to enforce permitted sequencing of steps and events.

**Implementation:**

| Check | Enforcement |
|-------|-------------|
| Policy workflow | Draft → Review → Approved → Active |
| Approval sequence | Multiple approvers required |
| Record status | Cannot modify finalized records |
| Time sequencing | Chronological enforcement |

#### (g) Authority Checks

**Requirement:** Use authority checks to ensure only authorized individuals use the system, sign records, access the operation or system input/output device, alter a record, or perform the operation at hand.

**Implementation:**

| Operation | Required Authority |
|-----------|-------------------|
| Create policy | Policy Author role |
| Approve policy | Approver role |
| Modify configuration | Administrator role |
| Sign records | Signature authority |
| View audit trail | Any authenticated user |

#### (h) Device Checks

**Requirement:** Use device checks to determine the validity of source of data input or operational instruction.

**Implementation:**

| Check | Mechanism |
|-------|-----------|
| Input validation | Schema validation |
| Source verification | Client certificate |
| API authentication | API key + JWT |
| Edge device | TPM attestation |

#### (i) Training

**Requirement:** Persons who develop, maintain, or use electronic records/signature systems are trained.

**Implementation:**

| Training | Audience | Frequency |
|----------|----------|-----------|
| Part 11 Fundamentals | All users | Initial + annual |
| System Operation | Operators | Initial + change |
| Administration | Admins | Initial + change |
| GxP Awareness | All | Annual |

#### (j) Policies and Procedures

**Requirement:** Establish written policies holding individuals accountable.

**Documentation:**

| Policy | Location |
|--------|----------|
| Electronic Records Policy | `policies/electronic_records.md` |
| Electronic Signature Policy | `policies/electronic_signatures.md` |
| Audit Trail Policy | `policies/audit_trail.md` |
| Access Control Policy | `policies/access_control.md` |

#### (k) Documentation Controls

**Requirement:** Adequate controls over system documentation.

**Implementation:**

| Control | Description |
|---------|-------------|
| Version control | Git-based |
| Change history | Complete audit trail |
| Access control | Role-based |
| Retention | 7+ years |

### § 11.30 Controls for Open Systems

When records are transmitted via open systems (internet):

| Control | Implementation |
|---------|---------------|
| Encryption | TLS 1.3 required |
| Digital signatures | For data integrity |
| Additional security | VPN for sensitive data |

### § 11.50 Signature Manifestations

**Requirement:** Signed electronic records must contain:
- Printed name of signer
- Date and time of signature
- Meaning of signature (e.g., approval, authorship)

**Implementation:**

```python
class ElectronicSignature:
    signer_name: str         # Full name
    signer_id: str           # User identifier
    timestamp: datetime      # Date and time (UTC)
    meaning: str             # APPROVED, REVIEWED, AUTHORED
    record_id: UUID          # Record being signed
    signature_hash: str      # Cryptographic signature
    
# Example signed record
{
    "record_id": "a1b2c3d4-...",
    "content": {...},
    "signature": {
        "signer_name": "Dr. Jane Smith",
        "signer_id": "jsmith@hospital.org",
        "timestamp": "2025-12-03T14:30:00Z",
        "meaning": "APPROVED",
        "signature_hash": "SHA256:abc123..."
    }
}
```

### § 11.70 Signature/Record Linking

**Requirement:** Electronic signatures shall be linked to their respective records to prevent copying or transferring.

**Implementation:**

| Mechanism | Description |
|-----------|-------------|
| Cryptographic binding | Signature covers record hash |
| Record inclusion | Signature embedded in record |
| Tamper detection | Hash chain verification |
| Uniqueness | One-time signature tokens |

## Subpart C: Electronic Signatures

### § 11.100 General Requirements

#### (a) Uniqueness

**Requirement:** Each electronic signature is unique to one individual.

**Implementation:**
- User accounts are unique
- Signatures bound to authenticated user
- No shared accounts permitted

#### (b) Verification

**Requirement:** Signature verified before use.

**Implementation:**
- MFA required for signatures
- Password/biometric confirmation
- Session re-authentication for signatures

#### (c) Certification to FDA

**Requirement:** Prior to use, certify to FDA that electronic signatures are intended as legally binding.

**Template Certification:**
```
[Organization Name] hereby certifies that all electronic signatures executed by 
[its/our] employees, agents, or representatives, on [date], are the legally 
binding equivalent of traditional handwritten signatures.
```

### § 11.200 Electronic Signature Components

**Requirement:** Electronic signatures based upon biometrics or comprised of at least two distinct components (e.g., ID + password).

**Implementation:**

| Component 1 | Component 2 | Authentication |
|-------------|-------------|----------------|
| Username | Password | Two-factor |
| Username + Password | MFA token | Three-factor |
| Biometric | PIN | Two-factor |

### § 11.300 Controls for Identification Codes/Passwords

| Requirement | Implementation |
|-------------|---------------|
| Uniqueness | Unique user IDs |
| Periodic revision | Password expiry (90 days) |
| Loss management | Immediate disabling |
| Transaction safeguards | Signature-specific re-auth |
| Initial/periodic testing | Device/token testing |

## Compliance Matrix

### Part 11 Requirements Mapping

| Section | Requirement | Module | Status |
|---------|-------------|--------|--------|
| 11.10(a) | Validation | Deploy scripts | ✅ |
| 11.10(b) | Copies | Export API | ✅ |
| 11.10(c) | Protection | Encryption | ✅ |
| 11.10(d) | Access limits | RBAC | ✅ |
| 11.10(e) | Audit trails | Audit logging | ✅ |
| 11.10(f) | Sequencing | Workflows | ✅ |
| 11.10(g) | Authority | Authorization | ✅ |
| 11.10(h) | Device checks | Validation | ✅ |
| 11.10(i) | Training | Training program | ✅ |
| 11.10(j) | Policies | Policy documents | ✅ |
| 11.10(k) | Documentation | Version control | ✅ |
| 11.30 | Open systems | TLS, VPN | ✅ |
| 11.50 | Manifestations | Signature display | ✅ |
| 11.70 | Linking | Crypto binding | ✅ |
| 11.100 | Uniqueness | User management | ✅ |
| 11.200 | Components | MFA | ✅ |
| 11.300 | Controls | Password policy | ✅ |

## Electronic Signature Implementation

### Signature Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Record    │───►│   Review    │───►│  Signature  │───►│   Signed    │
│   Draft     │    │   Request   │    │   Capture   │    │   Record    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                  │
                          ▼                  ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   MFA       │    │  Signature  │
                   │   Challenge │    │  Reason     │
                   └─────────────┘    └─────────────┘
```

### API for Electronic Signatures

```python
# Request signature
POST /v2/signatures/request
{
    "record_id": "uuid",
    "record_type": "policy",
    "meaning": "APPROVED",
    "comment": "Approved after review"
}

# Response requires MFA
HTTP 202 Accepted
{
    "signature_request_id": "uuid",
    "mfa_required": true,
    "mfa_method": "totp"
}

# Complete with MFA
POST /v2/signatures/complete
{
    "signature_request_id": "uuid",
    "mfa_code": "123456"
}

# Response
HTTP 200 OK
{
    "signature_id": "uuid",
    "signer_name": "Dr. Jane Smith",
    "timestamp": "2025-12-03T14:30:00Z",
    "meaning": "APPROVED",
    "valid": true
}
```

## Validation Protocol

### Installation Qualification (IQ)

| Check | Verification |
|-------|-------------|
| Hardware meets specs | Configuration review |
| Software installed correctly | Installation verification |
| Security configurations | Settings review |
| Network connectivity | Connection tests |

### Operational Qualification (OQ)

| Test | Acceptance Criteria |
|------|---------------------|
| User authentication | Successful login/logout |
| Access control | Role enforcement |
| Audit logging | Complete trail |
| Electronic signatures | Compliant signatures |
| Data integrity | No corruption |
| Backup/restore | Successful recovery |

### Performance Qualification (PQ)

| Test | Acceptance Criteria |
|------|---------------------|
| Production workload | SLA met |
| Concurrent users | Performance maintained |
| Data retention | Records preserved |
| Signature verification | 100% valid |

## References

- FDA 21 CFR Part 11
- FDA Guidance for Industry: Part 11, Electronic Records; Electronic Signatures — Scope and Application
- GAMP 5: A Risk-Based Approach to Compliant GxP Computerized Systems

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-03  
**Classification:** GxP Compliance
