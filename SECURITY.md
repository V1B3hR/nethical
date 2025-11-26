# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.2.x   | :white_check_mark: |
| 2.1.x   | :white_check_mark: |
| 2.0.x   | :warning: Critical fixes only |
| < 2.0   | :x:                |

## Reporting a Vulnerability

We take the security of Nethical seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories**: Use GitHub's private vulnerability reporting feature at https://github.com/V1B3hR/nethical/security/advisories/new (preferred)
2. **Email**: Send an email to security@nethical.ai with the subject line "SECURITY: [Brief Description]"

### What to Include

Please include the following information in your report:

- Type of vulnerability (e.g., SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the vulnerability
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### Response Timeline

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.
- **Assessment**: We will assess the vulnerability and determine its severity within 5 business days.
- **Fix Development**: Depending on severity, we will work on a fix with the following timelines:
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 60 days
- **Disclosure**: We will coordinate the disclosure timeline with you. We prefer to publicly disclose vulnerabilities after a fix is available.

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
- Email [INSERT_SECURITY_CONTACT]

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities. Contributors who report valid security issues will be credited in our security advisories (with their permission).

---

This security policy is subject to change. Last updated: 2025-10-15
