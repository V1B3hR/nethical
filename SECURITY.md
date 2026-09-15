# 📍 Documentation Relocated

> **⚠️ NOTICE: This file has been moved to the unified documentation structure.**

**New Location:** [`docs/laws_and_policies/SECURITY.md`](docs/laws_and_policies/SECURITY.md)

Please update your bookmarks and links.

**Quick Navigation:**
- [📖 Complete Documentation Index](docs/index.md)
- [⚖️ Laws & Policies](docs/laws_and_policies/)
- [📜 The 25 Fundamental Laws](docs/laws_and_policies/FUNDAMENTAL_LAWS.md)

---

# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Status | Patch Support |
| ------- | ------ | ------------- |
| **2.7.x** | **Active Production** | :white_check_mark: Full Support & SLA |
| 2.6.x   | Maintenance | :warning: Critical security fixes only |
| < 2.6   | End of Life | :x: Not supported (Upgrade required) |

## Security Advisories & Remediations

### 🛡️ GHSA-2026-cve-26007 / CVE-2026-26007 (Remediated in v2.7.0)
- **Severity:** Critical (CVSS 9.1)
- **Vulnerability:** Private key recovery when processing malicious public keys over binary elliptic curves (GF(2^m) curves) in `cryptography` < 46.0.5.
- **Affected Subsystems:** Merkle-DAG Ledger, Reversible Token Vault, and TEE Enclave Attestation.
- **Remediation in Nethical v2.7.0:**
  1. **Strict Dependency Pinning:** Pinned `cryptography>=50.0.0` in `pyproject.toml`, `requirements.txt`, and `nethical-edge/pyproject.toml`.
  2. **Curve Whitelisting & AST Audit:** Static and dynamic AST enforcement (`nethical/security/audit_crypto_curves.py`) prohibiting binary curves (`sect*`, `c2tnb*`, `c2onb*`). Only NIST prime curves (P-256, P-384, Ed25519) and NIST FIPS 204 ML-DSA-65 (Dilithium) are permitted.
  3. **Automated Key Rotation:** Integrated `ReversibleTokenVault.rotate_key()` and re-encryption pipeline.
  4. **Continuous Regression Testing:** Enforced in CI via `tests/security/test_crypto_audit.py`.

## Reporting a Vulnerability

We take the security of Nethical seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories**: Use GitHub's private vulnerability reporting feature at https://github.com/V1B3hR/nethical/security/advisories/new (preferred)
2. **Email**: Send an email to security@nethical.ai with the subject line "SECURITY: [Brief Description]"

### What to Include

Please include the following information in your report:

- Type of vulnerability (e.g., SQL injection, cross-site scripting, cryptographic flaw)
- Full paths of source file(s) related to the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Binding SLA Response Timeline (Technical Steering Committee Charter)

- **Acknowledgment**: Within **24 hours** for all reports.
- **Assessment & Triage**: Within **48 hours**.
- **Fix Development**:
  - **Critical (CVSS >= 9.0)**: **< 72 hours** to emergency patch release.
  - **High (CVSS 7.0 - 8.9)**: **< 7 days**.
  - **Medium (CVSS 4.0 - 6.9)**: **< 21 days**.
  - **Low (CVSS < 4.0)**: **< 45 days**.
- **Disclosure**: We coordinate responsible disclosure following patch availability. Full details are published via GitHub Security Advisories.

### Vulnerability Disclosure Policy

- We follow a coordinated disclosure model
- We will keep you informed of the progress towards a fix and full disclosure
- We may publicly disclose the vulnerability once a fix is available
- We will credit you in the security advisory (unless you prefer to remain anonymous)

## Security Features

Nethical includes several security features designed to protect AI agent systems:

### Core Security Capabilities

1. **Adversarial Detection**: Detects prompt injection, jailbreak attempts, and context confusion
2. **Resource Limits**: Quota enforcement and rate limiting to prevent resource exhaustion
3. **PII Detection**: Comprehensive detection of personally identifiable information
4. **Audit Logging**: Immutable audit trails with Merkle anchoring
5. **Multi-Tenant Isolation**: Storage partitioning and quota isolation by tenant/region

### Privacy & Compliance

1. **Differential Privacy**: Configurable privacy-preserving mechanisms
2. **Data Redaction**: PII redaction pipeline with reversible redaction
3. **Data Minimization**: Automated data retention and right-to-be-forgotten support
4. **Regional Compliance**: Support for GDPR, CCPA, and other data residency requirements

### Supply Chain Security

1. **SBOM Generation**: Software Bill of Materials (SBOM) in SPDX and CycloneDX formats
2. **Artifact Signing**: Cosign-based signing of releases and containers
3. **Provenance**: SLSA provenance attestations for builds
4. **Dependency Scanning**: Automated vulnerability scanning of dependencies

## Security Best Practices

When using Nethical in production:

1. **Enable All Security Features**: Enable quota enforcement, PII detection, and audit logging
2. **Configure Strong Limits**: Set appropriate rate limits and resource quotas
3. **Monitor Audit Logs**: Regularly review audit logs and Merkle anchors
4. **Update Regularly**: Keep Nethical and its dependencies up to date
5. **Use Encryption**: Enable encryption for sensitive data at rest and in transit
6. **Implement Access Controls**: Use proper authentication and authorization
7. **Regular Security Assessments**: Conduct periodic security reviews and penetration tests

## Security Advisories

Security advisories will be published at:
- GitHub Security Advisories: https://github.com/V1B3hR/nethical/security/advisories
- Release notes with security fixes will be clearly marked

## Security Testing

We conduct the following security testing:

- **SAST**: Bandit, Semgrep, CodeQL
- **Dependency Scanning**: Trivy, dependency-review-action
- **Secret Scanning**: TruffleHog
- **Adversarial Testing**: Comprehensive test suite for attack patterns
- **Penetration Testing**: Periodic external security assessments (planned)

## Contact

For security-related questions that are not vulnerability reports, you can:
- Open a discussion in GitHub Discussions
- Email security@nethical.ai

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities. Contributors who report valid security issues will be credited in our security advisories (with their permission).

---

This security policy is subject to change. Last updated: 2026-09-15
