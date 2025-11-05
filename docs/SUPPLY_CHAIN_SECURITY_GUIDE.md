# Supply Chain Security Guide

## Overview

Nethical implements comprehensive supply chain security measures to protect against tampering, malicious dependencies, and supply chain attacks. This guide covers hash verification, SLSA compliance, and security scanning workflows.

## Table of Contents

1. [Hash Verification](#hash-verification)
2. [SLSA Compliance](#slsa-compliance)
3. [Dependency Management](#dependency-management)
4. [Security Scanning](#security-scanning)
5. [CI/CD Integration](#cicd-integration)
6. [Best Practices](#best-practices)

---

## Hash Verification

### Overview

Hash verification ensures that installed packages match their expected checksums, protecting against package tampering and man-in-the-middle attacks.

### Generating Hashed Requirements

Use the `generate_hashed_requirements.py` script to create requirements files with SHA256 hashes:

```bash
# Generate hashed requirements for requirements.txt
python scripts/generate_hashed_requirements.py --input requirements.txt

# Generate for all requirements files
python scripts/generate_hashed_requirements.py --all

# This creates:
# - requirements-hashed.txt
# - requirements-dev-hashed.txt
```

### Example Output

```
# Hashed requirements file for supply chain security
# Install with: pip install --require-hashes -r requirements-hashed.txt

pydantic==2.12.3 \
    --hash=sha256:abc123...  # Core data models and validation
numpy==1.26.4 \
    --hash=sha256:def456...  # Numerical operations
```

### Installing with Hash Verification

```bash
# Install with hash verification enabled
pip install --require-hashes -r requirements-hashed.txt

# This will:
# - Verify each package SHA256 hash before installation
# - Fail if any hash doesn't match
# - Protect against package tampering
```

### Updating Dependencies

When updating dependencies:

1. Update version in `requirements.txt`
2. Regenerate hashes: `python scripts/generate_hashed_requirements.py`
3. Test installation: `pip install --require-hashes -r requirements-hashed.txt`
4. Commit both files

```bash
# Example workflow
echo "pydantic==2.13.0  # Updated version" >> requirements.txt
python scripts/generate_hashed_requirements.py --input requirements.txt
pip install --require-hashes -r requirements-hashed.txt
git add requirements.txt requirements-hashed.txt
git commit -m "Update pydantic to 2.13.0 with hash verification"
```

## SLSA Compliance

### SLSA Framework

Supply-chain Levels for Software Artifacts (SLSA) is a framework for ensuring software supply chain integrity.

Nethical targets **SLSA Level 3** compliance:

| Level | Requirements | Status |
|-------|-------------|--------|
| **Level 1** | Version control, automated build | ✅ Complete |
| **Level 2** | Build service, build provenance | ✅ Complete |
| **Level 3** | Source/build platform hardening, non-falsifiable provenance | ✅ In Progress |
| **Level 4** | Two-party review, hermetic builds | 🔄 Future |

### SLSA Level 3 Requirements

#### 1. Version Control ✅

```bash
# All code in Git repository
git log --oneline

# Signed commits (recommended)
git config --global commit.gpgsign true
git commit -S -m "Signed commit"
```

#### 2. Build Provenance ✅

Generated automatically by GitHub Actions:

```yaml
# .github/workflows/sbom-sign.yml generates:
- provenance.json (SLSA v1.0 format)
- sbom.spdx.json (Software Bill of Materials)
- sbom.cyclonedx.json (Alternative SBOM format)
```

#### 3. Non-Falsifiable Provenance ✅

- Uses GitHub Actions with OIDC tokens
- Keyless signing with Sigstore/Cosign
- Artifacts signed automatically on release

```bash
# Verify artifact signature
cosign verify-blob \
  --signature artifact.sig \
  --certificate artifact.pem \
  artifact.whl
```

#### 4. Build Platform Hardening ✅

- GitHub Actions with isolated runners
- Dependency pinning with hash verification
- Security scanning in CI/CD
- Automated vulnerability checks

### Checking SLSA Compliance

Use the automated checker:

```bash
# CI/CD workflow runs automatically
# .github/workflows/hash-verification.yml

# Manual check:
git log -1 --show-signature  # Check signed commits
ls .github/workflows/*.yml   # Verify CI/CD workflows
grep -c '==' requirements.txt  # Check pinned dependencies
```

## Dependency Management

### Pinning Dependencies

All production dependencies must be pinned to exact versions:

```
# ✅ Good - Pinned version
pydantic==2.12.3

# ❌ Bad - Unpinned version
pydantic>=2.0.0
```

### Checking for Unpinned Dependencies

```bash
# Manual check
grep -E '^[a-zA-Z0-9_-]+>=' requirements.txt

# Automated check (CI/CD)
# Runs automatically in .github/workflows/hash-verification.yml
```

### Dependabot Configuration

Automated dependency updates via Dependabot:

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "V1B3hR"
    labels:
      - "security"
```

**Features:**
- Weekly dependency scans
- Automatic PRs for updates
- Security-focused updates
- Grouped minor/patch updates

### Supply Chain Dashboard

Monitor dependency health:

```bash
# Generate supply chain report
python scripts/supply_chain_dashboard.py --format markdown

# Output includes:
# - Dependency summary
# - SLSA compliance status
# - Outdated packages
# - Security recommendations
```

## Security Scanning

### Automated Scanning in CI/CD

Multiple security scanners run automatically:

```yaml
# .github/workflows/security.yml
- Bandit: Python code security scanner
- Semgrep: Static analysis security testing (SAST)
- CodeQL: Semantic code analysis
- Trivy: Vulnerability scanner
- TruffleHog: Secret detection
```

### Manual Security Scans

#### Scan Dependencies

```bash
# Check for known vulnerabilities
pip-audit

# Or use safety
safety check -r requirements.txt
```

#### Scan Code

```bash
# Python security issues
bandit -r nethical/

# SAST analysis
semgrep --config=auto nethical/

# Secret detection
trufflehog filesystem . --only-verified
```

#### Scan Docker Images

```bash
# Build image
docker build -t nethical:latest .

# Scan with Trivy
trivy image nethical:latest

# Scan with Grype
grype nethical:latest
```

## CI/CD Integration

### Hash Verification Workflow

Automatically runs on:
- Pull requests modifying requirements files
- Pushes to main/develop

```yaml
# .github/workflows/hash-verification.yml
jobs:
  generate-hashes:
    - Generate hashed requirements
    - Verify installation with hashes
    - Check for unpinned dependencies
    - Generate supply chain report
    - Comment on PR with results
  
  verify-slsa:
    - Check SLSA Level 3 requirements
    - Verify version control
    - Check CI/CD workflows
    - Validate dependency pinning
    - Verify SBOM generation
```

### SBOM Generation Workflow

Runs on releases and tags:

```yaml
# .github/workflows/sbom-sign.yml
jobs:
  generate-sbom:
    - Generate SBOM in SPDX format
    - Generate SBOM in CycloneDX format
    - Upload as artifacts
  
  sign-artifacts:
    - Build Python packages
    - Sign with Cosign (keyless)
    - Generate SHA256 checksums
    - Upload signed artifacts
  
  provenance:
    - Generate SLSA provenance
    - Attach to GitHub release
```

### PR Comments

On pull requests, get automatic feedback:

```
## 🔒 Supply Chain Security Report

### Dependency Summary
- Production dependencies: 15
- Development dependencies: 6
- Outdated packages: 2

### SLSA Compliance
**Current Level:** Level 3

✅ Version control: Git repository
✅ Signed commits: Present
✅ CI/CD: GitHub Actions workflows present
✅ Dependency management: 100% pinned
✅ SBOM generation: Workflow present
✅ Artifact signing: Cosign configured

### 📊 Hash Verification
✅ Hashed requirements generated

### 💡 Recommendations
- Update 2 outdated packages
- Review security advisories
```

## Best Practices

### 1. Always Pin Dependencies

```
# requirements.txt
pydantic==2.12.3  # Not >=2.0.0
numpy==1.26.4     # Not >=1.20.0
```

### 2. Use Hash Verification

```bash
# Always install with --require-hashes in production
pip install --require-hashes -r requirements-hashed.txt
```

### 3. Review Dependency Updates

Before merging Dependabot PRs:

1. Review changelog for breaking changes
2. Check security advisories
3. Run full test suite
4. Regenerate hashes
5. Update documentation if needed

### 4. Monitor Security Advisories

```bash
# GitHub Advisory Database
gh api /advisories

# Subscribe to security mailing lists
# - Python Security Response Team
# - GitHub Security Advisories
```

### 5. Scan Regularly

```bash
# Weekly security scans
pip-audit
bandit -r nethical/
trivy fs .
```

### 6. Sign Commits

```bash
# Configure GPG signing
git config --global commit.gpgsign true
git config --global user.signingkey YOUR_KEY_ID

# All commits will be signed
git commit -m "Update dependencies"
```

### 7. Use Minimal Images

```dockerfile
# Use minimal base images
FROM python:3.11-slim

# Install only required packages
RUN pip install --no-cache-dir --require-hashes -r requirements-hashed.txt
```

### 8. Implement Defense in Depth

- ✅ Hash verification
- ✅ Dependency pinning
- ✅ Security scanning
- ✅ SBOM generation
- ✅ Artifact signing
- ✅ Provenance tracking
- ✅ Automated updates
- ✅ Code review process

## Verification

### Verify Installation

```bash
# Check package integrity
pip check

# Verify all packages have expected versions
pip freeze | diff - requirements.txt

# Verify hashes
pip install --dry-run --require-hashes -r requirements-hashed.txt
```

### Verify SBOM

```bash
# View SBOM
cat sbom.spdx.json | jq '.packages[] | {name, version}'

# Validate SBOM
syft packages dir:. -o spdx-json | jq .
```

### Verify Signatures

```bash
# Download release artifacts
gh release download v0.1.0

# Verify wheel signature
cosign verify-blob \
  --signature nethical-0.1.0-py3-none-any.whl.sig \
  --certificate nethical-0.1.0-py3-none-any.whl.pem \
  nethical-0.1.0-py3-none-any.whl

# Verify checksum
sha256sum -c SHA256SUMS
```

## Incident Response

### If Dependency Compromised

1. **Immediate Actions**
   ```bash
   # Remove compromised package
   pip uninstall compromised-package
   
   # Update to safe version
   echo "safe-package==1.2.3" >> requirements.txt
   python scripts/generate_hashed_requirements.py
   pip install --require-hashes -r requirements-hashed.txt
   ```

2. **Investigation**
   - Check when package was added
   - Review all commits using the package
   - Scan for malicious code execution
   - Check logs for suspicious activity

3. **Communication**
   - Create security advisory
   - Notify users via GitHub/email
   - Update documentation
   - Publish incident report

4. **Prevention**
   - Review dependency sources
   - Implement additional scanning
   - Update security policies
   - Conduct security training

## Tools and Resources

### Scripts

- `scripts/generate_hashed_requirements.py` - Generate hashed requirements
- `scripts/supply_chain_dashboard.py` - Generate security dashboard

### Workflows

- `.github/workflows/hash-verification.yml` - Hash verification CI/CD
- `.github/workflows/sbom-sign.yml` - SBOM generation and signing
- `.github/workflows/security.yml` - Security scanning

### External Tools

- **Syft**: SBOM generation
- **Cosign**: Artifact signing
- **Trivy**: Vulnerability scanning
- **Bandit**: Python security scanner
- **pip-audit**: Dependency vulnerability checker

### References

- [SLSA Framework](https://slsa.dev/)
- [Sigstore Project](https://www.sigstore.dev/)
- [CycloneDX SBOM Standard](https://cyclonedx.org/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [NIST Secure Software Development Framework](https://csrc.nist.gov/projects/ssdf)

---

**Last Updated:** November 5, 2025  
**SLSA Level:** 3 (In Progress)  
**Compliance Status:** Active
