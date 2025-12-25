# Security Hardening Implementation Summary

This document provides an overview of the security hardening implementation for Nethical, addressing the requirements specified in the task.

## Overview

The implementation adds comprehensive security hardening across four main areas:
1. Server and Runtime Hardening
2. Access Control
3. Monitoring and Alerting
4. Dependency Management

## Quick Reference

### Documentation

| Document | Description | Path |
|----------|-------------|------|
| **Security Hardening Guide** | Complete hardening manual with practical examples | [docs/SECURITY_HARDENING.md](./SECURITY_HARDENING.md) |
| **Docker Security** | Docker deployment with AppArmor, rootless mode | [deploy/docker/README.md](../deploy/docker/README.md) |
| **SELinux Configuration** | SELinux policy for RHEL/CentOS systems | [deploy/selinux/README.md](../deploy/selinux/README.md) |
| **Kubernetes AppArmor** | AppArmor profile for Kubernetes | [deploy/kubernetes/apparmor-profile.yaml](../deploy/kubernetes/apparmor-profile.yaml) |
| **PR Template** | Security review checklist for pull requests | [.github/PULL_REQUEST_TEMPLATE.md](../.github/PULL_REQUEST_TEMPLATE.md) |

### Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| **update_environment.sh** | Automated environment updates | `./scripts/update_environment.sh [--security-only] [--dry-run]` |
| **check_dependency_confusion.py** | Detect dependency confusion attacks | `python scripts/check_dependency_confusion.py requirements.txt` |

### GitHub Actions Workflows

| Workflow | Description | Triggers |
|----------|-------------|----------|
| **security-monitoring.yml** | Monitor critical file changes, anomaly detection | Push, PR, Daily |
| **dependency-scan.yml** | Scan dependencies for vulnerabilities | Push (deps), PR (deps), Weekly |

## Implementation Details

### 1. Server and Runtime Hardening

#### Docker Security
- **Non-privileged User**: Containers run as UID 1000 by default
- **Read-only Filesystem**: Root filesystem is read-only with tmpfs for /tmp
- **Minimal Capabilities**: DROP ALL, add only NET_BIND_SERVICE
- **Docker Rootless Mode**: Complete setup guide for rootless deployments
- **AppArmor Profile**: Mandatory access control for Docker containers
- **Security Options**: no-new-privileges, AppArmor/SELinux labels

**Quick Start:**
```bash
docker run -d \
  --name nethical \
  --user 1000:1000 \
  --read-only \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --security-opt=no-new-privileges:true \
  --security-opt apparmor=nethical-docker \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  -p 8000:8000 \
  nethical:latest
```

#### AppArmor Profiles
- **Kubernetes**: `deploy/kubernetes/apparmor-profile.yaml`
- **Docker**: `deploy/docker/apparmor-profile`

Features:
- Restricted file system access
- Network protocol controls
- Deny dangerous system calls
- Process isolation
- Minimal capabilities

#### SELinux Policy
- **Policy Module**: `deploy/selinux/nethical.te`
- **Documentation**: `deploy/selinux/README.md`

Features:
- Type enforcement for Nethical domain
- File context definitions
- Network access controls
- Logging isolation
- Security denials for hardening

#### Environment Updates
- **Script**: `scripts/update_environment.sh`
- **Features**:
  - System package updates (apt/yum/dnf)
  - Python and pip updates
  - Dependency vulnerability scanning
  - Hash verification
  - Dry-run mode for testing

**Usage:**
```bash
# Dry run to see what would be updated
./scripts/update_environment.sh --dry-run

# Security updates only
./scripts/update_environment.sh --security-only

# Full update
./scripts/update_environment.sh
```

### 2. Access Control

#### Two-Factor Authentication (2FA)
- **GitHub**: Complete setup guide for organization and individual accounts
- **Hardware Keys**: Recommended for production access (YubiKey, Titan)
- **TOTP Apps**: Alternative authentication method documented

#### Least Privilege Principle
- **Container Level**: Minimal capabilities, non-root user
- **Kubernetes RBAC**: Example role definitions
- **File Permissions**: Restrictive permissions (600 for secrets, 640 for configs)

#### Secrets Management

**GitHub Secrets** (for CI/CD):
```yaml
env:
  API_KEY: ${{ secrets.API_KEY }}
  DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

**HashiCorp Vault** (for production):
```python
import hvac
client = hvac.Client(url='https://vault.example.com:8200')
secret = client.secrets.kv.v2.read_secret_version(path='nethical/production')
```

**Docker Secrets** (for Swarm):
```bash
echo "secret_value" | docker secret create nethical_api_key -
```

#### Best Practices
- Separate secrets for dev/staging/production
- Rotate secrets every 90 days
- Audit all secret access
- Never commit secrets to version control
- Use environment variables for configuration

### 3. Monitoring and Alerting

#### Security Monitoring Workflow
**File**: `.github/workflows/security-monitoring.yml`

**Features:**
- **Critical File Monitoring**: Alerts on changes to workflows, configs, security code
- **Large File Detection**: Flags files >1MB
- **Binary File Detection**: Alerts on executable modifications
- **Secret Pattern Detection**: Scans for exposed credentials
- **Mass Change Detection**: Flags >50 files changed
- **Anomaly Detection**: Unusual commit patterns, off-hours commits
- **Dependency Integrity**: Verifies hash-based verification

**Alerts:**
- GitHub Issues for critical findings
- PR comments for warnings
- Configurable severity levels

#### Runtime Monitoring
- **Structured Logging**: JSON-formatted security event logs
- **Prometheus Metrics**: Auth attempts, request duration, error rates
- **Health Checks**: `/health` and `/ready` endpoints
- **Alert Integration**: Slack, PagerDuty, email

### 4. Dependency Management

#### Dependency Scanning Workflow
**File**: `.github/workflows/dependency-scan.yml`

**Jobs:**
1. **pip-audit**: CVE vulnerability scanning
2. **safety**: Safety database checks
3. **bandit**: Security linting for Python code
4. **dependency-confusion-check**: Detect supply chain attacks
5. **dependency-review**: GitHub's dependency review action

**Scanning Tools:**
```bash
# pip-audit - official PyPA tool
pip-audit -r requirements.txt

# Safety - vulnerability database
safety check --json

# Bandit - security linting
bandit -r nethical/ -ll
```

#### Dependency Confusion Protection
**Script**: `scripts/check_dependency_confusion.py`

**Detects:**
- External package sources (non-PyPI)
- Typosquatting attempts
- Internal package conflicts
- Missing hash verification
- Suspicious package patterns

**Usage:**
```bash
python scripts/check_dependency_confusion.py requirements.txt
```

#### Hash Verification
- **File**: `requirements-hashed.txt`
- **Generation**: `pip-compile --generate-hashes`
- **Usage**: `pip install --require-hashes -r requirements-hashed.txt`

#### Best Practices
- Install only from official PyPI
- Use hash-based verification for production
- Pin exact versions for security-critical packages
- Use compatible release specifiers (~=) for frameworks
- Weekly automated vulnerability scans
- Dependency allowlists for high-security environments

## Security Review Process

### Pull Request Template
**File**: `.github/PULL_REQUEST_TEMPLATE.md`

**Checklist Includes:**
- Secrets and credentials verification
- Dependency vulnerability checks
- Input validation review
- Authentication/authorization verification
- Code security (bandit, semgrep)
- Infrastructure configuration review
- Monitoring and logging verification

### Review Commands
```bash
# Clone and checkout PR
gh pr checkout <PR_NUMBER>

# Run security scans
pip-audit -r requirements.txt
safety check
bandit -r nethical/ -ll

# Check for secrets
git log -p | grep -iE "password|secret|key|token|api_key"

# Run tests
pytest tests/ -v
```

## Deployment Checklist

### Pre-deployment
- [ ] Review Security Hardening Guide
- [ ] Choose deployment method (Docker, Kubernetes, bare metal)
- [ ] Configure secrets management (Vault, GitHub Secrets, Docker Secrets)
- [ ] Set up 2FA for all access
- [ ] Review and customize security profiles (AppArmor/SELinux)

### During Deployment
- [ ] Run environment update script
- [ ] Scan for dependency vulnerabilities
- [ ] Apply AppArmor/SELinux profiles
- [ ] Configure monitoring and alerting
- [ ] Set up log aggregation
- [ ] Configure backup and recovery

### Post-deployment
- [ ] Verify security configurations
- [ ] Test health checks and monitoring
- [ ] Run security scans (Trivy, Grype)
- [ ] Document deployment specifics
- [ ] Schedule regular security reviews
- [ ] Set up automated vulnerability scanning

## Maintenance

### Regular Tasks

**Weekly:**
- Review dependency scan results
- Check security monitoring alerts
- Review audit logs

**Monthly:**
- Update dependencies (with testing)
- Review and rotate credentials
- Security posture assessment
- Review access controls

**Quarterly:**
- Full security audit
- Penetration testing (if applicable)
- Update security documentation
- Team security training

### Update Process

1. **Check for updates:**
   ```bash
   ./scripts/update_environment.sh --dry-run
   ```

2. **Review changes:**
   - Check changelog for breaking changes
   - Review vulnerability reports
   - Test in staging environment

3. **Apply updates:**
   ```bash
   # Security updates only
   ./scripts/update_environment.sh --security-only
   
   # Or full update
   ./scripts/update_environment.sh
   ```

4. **Verify:**
   ```bash
   # Run tests
   pytest tests/
   
   # Scan for vulnerabilities
   pip-audit
   
   # Build and test
   docker build -t nethical:latest .
   docker run --rm nethical:latest pytest
   ```

5. **Deploy:**
   - Update production incrementally
   - Monitor for issues
   - Have rollback plan ready

## Troubleshooting

### Common Issues

**AppArmor denials:**
```bash
# Check denials
sudo dmesg | grep apparmor | grep DENIED

# Debug mode
sudo aa-complain /etc/apparmor.d/nethical-docker
```

**SELinux denials:**
```bash
# View denials
sudo ausearch -m avc -c nethical

# Generate policy from denials
sudo audit2allow -a -M nethical_local
```

**Dependency conflicts:**
```bash
# Check for conflicts
pip check

# Regenerate lock file
pip-compile --generate-hashes requirements.in
```

**Container permission issues:**
```bash
# Fix ownership
sudo chown -R 1000:1000 ./data

# Check container user
docker exec nethical id
```

## Support and Resources

### Documentation
- [Security Hardening Guide](./SECURITY_HARDENING.md)
- [Security Policy](../SECURITY.md)
- [Docker Security](../deploy/docker/README.md)
- [SELinux Configuration](../deploy/selinux/README.md)

### External Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)

### Getting Help
- GitHub Issues: Security-related issues
- Security Email: [SECURITY.md](../SECURITY.md) for vulnerability reporting
- Discussions: General security questions

---

**Last Updated:** 2025-12-25

For detailed implementation instructions, refer to the [Security Hardening Guide](./SECURITY_HARDENING.md).
