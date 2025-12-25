# Security Hardening Guide

This guide provides comprehensive security hardening recommendations for deploying and operating Nethical in production environments.

## Table of Contents

1. [Server and Runtime Hardening](#server-and-runtime-hardening)
2. [Access Control](#access-control)
3. [Monitoring and Alerting](#monitoring-and-alerting)
4. [Dependency Management](#dependency-management)

---

## Server and Runtime Hardening

### Environment Updates

#### Automated System Updates

Keep your system and dependencies up-to-date to protect against known vulnerabilities:

```bash
# Update system packages (Ubuntu/Debian)
sudo apt-get update && sudo apt-get upgrade -y

# Update Python and pip
python -m pip install --upgrade pip

# Update Nethical dependencies
pip install --upgrade -r requirements.txt
```

#### Automated Update Script

Use the provided update script for regular maintenance:

```bash
# Run the update script
./scripts/update_environment.sh

# Schedule automatic updates (add to crontab)
0 2 * * 0 /path/to/nethical/scripts/update_environment.sh >> /var/log/nethical-updates.log 2>&1
```

### Docker Security

#### Running as Non-Privileged User

The Nethical Docker image runs as a non-root user by default (UID 1000):

```dockerfile
# Already configured in Dockerfile
USER nethical
```

To run the container with additional security:

```bash
# Run with security options
docker run -d \
  --name nethical \
  --user 1000:1000 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --security-opt=no-new-privileges:true \
  -p 8000:8000 \
  nethical:latest
```

#### Docker Rootless Mode

Docker Rootless Mode allows running Docker daemon and containers as a non-root user, providing additional security isolation.

**Installation:**

```bash
# Install Docker Rootless
curl -fsSL https://get.docker.com/rootless | sh

# Configure environment
export PATH=/home/$USER/bin:$PATH
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock

# Add to ~/.bashrc for persistence
echo 'export PATH=/home/$USER/bin:$PATH' >> ~/.bashrc
echo 'export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock' >> ~/.bashrc
```

**Running Nethical in Rootless Mode:**

```bash
# Start Docker daemon (rootless)
systemctl --user start docker

# Run Nethical container
docker run -d \
  --name nethical-rootless \
  -p 8000:8000 \
  nethical:latest
```

**Benefits of Rootless Mode:**
- No root privileges required for Docker daemon
- Improved container isolation
- Reduced attack surface
- Protection against container escape vulnerabilities

**Limitations:**
- Some features may not work (overlay networks, specific storage drivers)
- Performance may be slightly reduced
- Port numbers below 1024 require additional configuration

For more details, see: https://docs.docker.com/engine/security/rootless/

### AppArmor Profile

AppArmor provides mandatory access control (MAC) security. A profile is already available in `deploy/kubernetes/apparmor-profile.yaml`.

**For Docker (non-Kubernetes):**

Create `/etc/apparmor.d/nethical-docker`:

```apparmor
#include <tunables/global>

profile nethical-docker flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  #include <abstractions/python>
  
  # Network access
  network inet stream,
  network inet6 stream,
  network inet tcp,
  network inet6 tcp,
  
  # Application files (read-only)
  /app/** r,
  /usr/lib/python3*/** r,
  /usr/local/lib/python3*/** r,
  
  # Data directory (read-write)
  /data/** rw,
  owner /tmp/** rw,
  
  # Required system files
  /etc/ssl/certs/** r,
  /etc/resolv.conf r,
  /etc/hosts r,
  
  # Deny dangerous operations
  deny capability sys_admin,
  deny capability sys_module,
  deny /proc/kcore r,
  deny /boot/** r,
  deny mount,
}
```

**Load the profile:**

```bash
# Install AppArmor utilities
sudo apt-get install -y apparmor-utils

# Load the profile
sudo apparmor_parser -r -W /etc/apparmor.d/nethical-docker

# Verify
sudo aa-status | grep nethical
```

**Run Docker with AppArmor:**

```bash
docker run -d \
  --security-opt apparmor=nethical-docker \
  --name nethical \
  nethical:latest
```

### SELinux Profile

For Red Hat/CentOS/Fedora systems using SELinux:

Create `/etc/selinux/local/nethical.te`:

```selinux
policy_module(nethical, 1.0.0)

# Type declarations
type nethical_t;
type nethical_exec_t;
type nethical_data_t;
type nethical_tmp_t;

# Application domain
application_domain(nethical_t, nethical_exec_t)

# File contexts
files_type(nethical_data_t)
files_tmp_file(nethical_tmp_t)

# Permissions
allow nethical_t self:process { fork signal sigchld };
allow nethical_t self:fifo_file rw_fifo_file_perms;
allow nethical_t self:tcp_socket create_stream_socket_perms;
allow nethical_t self:udp_socket create_socket_perms;

# Network permissions
corenet_tcp_bind_generic_node(nethical_t)
corenet_tcp_connect_http_port(nethical_t)
corenet_tcp_bind_all_unreserved_ports(nethical_t)

# File permissions
allow nethical_t nethical_data_t:dir manage_dir_perms;
allow nethical_t nethical_data_t:file manage_file_perms;
allow nethical_t nethical_tmp_t:dir manage_dir_perms;
allow nethical_t nethical_tmp_t:file manage_file_perms;

# Read access to configuration
files_read_etc_files(nethical_t)

# Logging
logging_send_syslog_msg(nethical_t)
```

**Compile and install:**

```bash
# Compile the policy
checkmodule -M -m -o nethical.mod nethical.te
semodule_package -o nethical.pp -m nethical.mod

# Install the policy module
sudo semodule -i nethical.pp

# Set file contexts
sudo semanage fcontext -a -t nethical_exec_t "/app/nethical.py"
sudo semanage fcontext -a -t nethical_data_t "/data(/.*)?"
sudo restorecon -R /app /data

# Verify
sudo semodule -l | grep nethical
```

### Security-Hardened Docker Compose

Example `docker-compose.yml` with security best practices:

```yaml
version: '3.8'

services:
  nethical:
    image: nethical:latest
    container_name: nethical
    user: "1000:1000"
    read_only: true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    security_opt:
      - no-new-privileges:true
      - apparmor=nethical-docker
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=100m
    volumes:
      - ./data:/data:rw
      - ./config:/app/config:ro
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Access Control

### Two-Factor Authentication (2FA)

**For GitHub (Repository Access):**

1. Enable 2FA for all GitHub accounts with repository access
2. Use hardware security keys (YubiKey, Titan) for enhanced security
3. Store recovery codes securely (encrypted password manager)

**Configuration:**
- GitHub Settings → Security → Two-factor authentication
- Choose: Authenticator app (TOTP) or Security keys (WebAuthn/U2F)
- **Strongly recommended**: Use security keys for production access

**For Team Members:**
```bash
# Require 2FA for organization
# GitHub Organization → Settings → Security → 
# "Require two-factor authentication for everyone in the organization"
```

### Least Privilege Principle

Apply minimal permissions at every level:

#### Container Permissions

```bash
# Run with minimal capabilities
docker run \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --read-only \
  nethical:latest
```

#### Kubernetes RBAC

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: nethical-minimal
  namespace: nethical
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get"]
  resourceNames: ["nethical-secrets"]
```

#### File System Permissions

```bash
# Set restrictive permissions
chmod 600 config/secrets.yaml  # Secrets
chmod 640 config/*.yaml         # Configs
chmod 755 scripts/*.sh          # Scripts (executable)
chmod 700 data/                 # Data directory

# Set ownership
chown -R nethical:nethical /app/data
```

### Secrets Management

**Never commit secrets to version control!**

#### Option 1: GitHub Secrets (Recommended for CI/CD)

```yaml
# .github/workflows/deploy.yml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy with secrets
        env:
          API_KEY: ${{ secrets.API_KEY }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          JWT_SECRET: ${{ secrets.JWT_SECRET }}
        run: |
          ./deploy.sh
```

**Setup GitHub Secrets:**
1. Repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add secrets: `API_KEY`, `DATABASE_URL`, `JWT_SECRET`, etc.

**Best practices:**
- Use environment-specific secrets (`PROD_API_KEY`, `DEV_API_KEY`)
- Rotate secrets regularly (every 90 days)
- Use GitHub Environments for approval workflows
- Audit secret access via GitHub audit log

#### Option 2: HashiCorp Vault (Recommended for Production)

**Installation:**

```bash
# Install Vault
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/

# Start Vault server (dev mode for testing)
vault server -dev

# In production, use proper storage backend
vault server -config=/etc/vault/config.hcl
```

**Configuration:**

```hcl
# /etc/vault/config.hcl
storage "file" {
  path = "/var/vault/data"
}

listener "tcp" {
  address     = "127.0.0.1:8200"
  tls_disable = 0
  tls_cert_file = "/etc/vault/tls/vault.crt"
  tls_key_file  = "/etc/vault/tls/vault.key"
}

api_addr = "https://vault.example.com:8200"
cluster_addr = "https://vault.example.com:8201"
ui = true
```

**Using Vault with Nethical:**

```python
# Example: Retrieve secrets from Vault
import hvac

# Initialize Vault client
client = hvac.Client(
    url='https://vault.example.com:8200',
    token=os.environ.get('VAULT_TOKEN')
)

# Read secret
secret = client.secrets.kv.v2.read_secret_version(
    path='nethical/production',
    mount_point='secret'
)

api_key = secret['data']['data']['api_key']
db_password = secret['data']['data']['db_password']
```

**Integration Script:**

```bash
#!/bin/bash
# scripts/load_secrets_from_vault.sh

export VAULT_ADDR='https://vault.example.com:8200'
export VAULT_TOKEN=$(cat ~/.vault-token)

# Read secrets from Vault
API_KEY=$(vault kv get -field=api_key secret/nethical/production)
DB_PASSWORD=$(vault kv get -field=db_password secret/nethical/production)

# Export for application
export NETHICAL_API_KEY="$API_KEY"
export NETHICAL_DB_PASSWORD="$DB_PASSWORD"

# Run Nethical
exec python nethical.py
```

#### Option 3: Docker Secrets (for Docker Swarm)

```bash
# Create secrets
echo "my_api_key" | docker secret create nethical_api_key -
echo "my_db_password" | docker secret create nethical_db_password -

# Use in docker-compose.yml
version: '3.8'
services:
  nethical:
    image: nethical:latest
    secrets:
      - nethical_api_key
      - nethical_db_password

secrets:
  nethical_api_key:
    external: true
  nethical_db_password:
    external: true
```

#### Secrets Management Best Practices

1. **Separation of Concerns**: Different secrets for dev/staging/production
2. **Rotation**: Rotate secrets every 90 days minimum
3. **Audit**: Log all secret access and modifications
4. **Encryption**: Always encrypt secrets at rest and in transit
5. **Principle of Least Privilege**: Grant access only to required services
6. **No Hardcoding**: Never hardcode secrets in source code
7. **Environment Variables**: Use environment variables for configuration
8. **Secret Scanning**: Enable GitHub secret scanning and push protection

---

## Monitoring and Alerting

### GitHub Actions Monitoring

#### Critical File Change Monitoring

A workflow monitors changes to critical files and alerts on suspicious modifications.

See `.github/workflows/security-monitoring.yml` for implementation.

**Monitored files:**
- Configuration files (`config/**`, `*.yaml`)
- Security-critical code (`nethical/security/**`)
- CI/CD workflows (`.github/workflows/**`)
- Dependency files (`requirements*.txt`, `pyproject.toml`)

#### Anomaly Detection

The monitoring workflow includes:
- Large file additions (>1MB)
- Binary file modifications
- Multiple file changes in single commit (>50 files)
- Changes to security-sensitive directories
- Suspicious patterns (credentials, API keys)

#### Alert Configuration

**Slack Integration:**

```yaml
# Add to workflow
- name: Send Slack alert
  if: steps.check.outputs.suspicious == 'true'
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK_URL }}
    payload: |
      {
        "text": "🚨 Security Alert: Suspicious changes detected",
        "blocks": [...]
      }
```

**Email Notifications:**

Configure in: Repository → Settings → Notifications → Actions

**PagerDuty Integration:**

```yaml
- name: Trigger PagerDuty alert
  if: steps.check.outputs.critical == 'true'
  run: |
    curl -X POST https://events.pagerduty.com/v2/enqueue \
      -H 'Content-Type: application/json' \
      -d '{
        "routing_key": "${{ secrets.PAGERDUTY_KEY }}",
        "event_action": "trigger",
        "payload": {
          "summary": "Critical security changes detected",
          "severity": "critical",
          "source": "GitHub Actions"
        }
      }'
```

### Runtime Monitoring

#### Application Logging

```python
# Configure structured logging
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log security events
logger.info(json.dumps({
    'event': 'authentication_attempt',
    'user': user_id,
    'ip': request.remote_addr,
    'success': True,
    'timestamp': datetime.utcnow().isoformat()
}))
```

#### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

# Define metrics
auth_attempts = Counter('nethical_auth_attempts_total', 
                       'Authentication attempts',
                       ['status'])
request_duration = Histogram('nethical_request_duration_seconds',
                            'Request duration')

# Record metrics
auth_attempts.labels(status='success').inc()
with request_duration.time():
    process_request()
```

#### Health Checks

```python
# nethical/api.py
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": __version__
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check for load balancers"""
    # Check dependencies
    db_healthy = await check_database()
    cache_healthy = await check_cache()
    
    if db_healthy and cache_healthy:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Not ready")
```

### Code Review Security Checklist

Add to `.github/PULL_REQUEST_TEMPLATE.md`:

```markdown
## Security Review Checklist

- [ ] No secrets or credentials committed
- [ ] Dependencies updated to latest secure versions
- [ ] Input validation implemented for user data
- [ ] Authentication/authorization checks present
- [ ] Security tests added for new features
- [ ] Documentation updated for security implications
- [ ] No new high/critical severity vulnerabilities (run: `pip-audit`)
- [ ] AppArmor/SELinux profiles updated if needed
- [ ] Changelog updated with security-related changes
```

---

## Dependency Management

### Dependency Scanning

#### Automated Scanning Workflow

See `.github/workflows/dependency-scan.yml` for automated scanning on every PR.

**Tools used:**
- **pip-audit**: Scans for known vulnerabilities
- **safety**: Checks against safety database
- **bandit**: Security linting for Python code

#### Manual Scanning

```bash
# Install scanning tools
pip install pip-audit safety bandit

# Scan for vulnerabilities
pip-audit --desc
safety check --json
bandit -r nethical/ -f json

# Fix vulnerabilities
pip-audit --fix
```

### Official Sources Only

**Always install from official sources:**

```bash
# ✅ Correct: Official PyPI
pip install nethical

# ✅ Correct: Official repository
pip install git+https://github.com/V1B3hR/nethical.git

# ❌ Wrong: Unofficial mirrors
pip install -i http://suspicious-mirror.com/simple nethical

# ❌ Wrong: Unverified sources
pip install nethical-modified.tar.gz
```

#### Verify Package Integrity

```bash
# Enable hash verification
pip install --require-hashes -r requirements-hashed.txt

# Verify signatures (when available)
pip install nethical --verify-signatures
```

### Dependency Confusion Protection

Dependency confusion attacks occur when an attacker publishes a malicious package with the same name as an internal package to a public repository.

#### Protection Measures

**1. Use requirements-hashed.txt**

```bash
# Generate hashed requirements
pip-compile --generate-hashes --output-file requirements-hashed.txt requirements.txt

# Install with hash verification (prevents substitution)
pip install --require-hashes -r requirements-hashed.txt
```

**2. Private Package Index**

```bash
# Configure pip to prefer private index
# ~/.config/pip/pip.conf or /etc/pip.conf
[global]
index-url = https://private-pypi.company.com/simple/
extra-index-url = https://pypi.org/simple/

# Or use environment variable
export PIP_INDEX_URL=https://private-pypi.company.com/simple/
export PIP_EXTRA_INDEX_URL=https://pypi.org/simple/
```

**3. Package Naming Convention**

Use unique prefixes for internal packages:

```
company-nethical-plugin
company-nethical-integration
```

**4. Namespace Packages**

Use namespace packages to avoid conflicts:

```python
# Internal package structure
company.nethical.plugin
company.nethical.integration
```

**5. Lock File with Integrity Checks**

```bash
# Generate lock file with pip-tools
pip-compile --generate-hashes requirements.in

# Or use Poetry
poetry lock

# Or use Pipenv
pipenv lock
```

**6. Dependency Allowlist**

Create `.github/workflows/dependency-allowlist.txt`:

```
# Allowed dependencies
bcrypt>=5.0.0
argon2-cffi>=25.1.0
fastapi>=0.100.0
# ... other approved packages
```

Validate in CI:

```yaml
- name: Check dependency allowlist
  run: |
    python scripts/check_dependencies.py requirements.txt .github/workflows/dependency-allowlist.txt
```

#### Detection Script

Create `scripts/check_dependency_confusion.py`:

```python
#!/usr/bin/env python3
"""Check for potential dependency confusion attacks"""
import subprocess
import json
import sys

def check_package_source(package_name):
    """Verify package comes from expected source"""
    result = subprocess.run(
        ['pip', 'show', package_name],
        capture_output=True,
        text=True
    )
    
    if 'Location' in result.stdout:
        location = [l for l in result.stdout.split('\n') 
                   if 'Location:' in l][0]
        
        # Check if from official PyPI or verified source
        if '/site-packages' in location:
            return True
    
    return False

def main():
    with open('requirements.txt') as f:
        packages = [line.split('==')[0].strip() 
                   for line in f if line.strip() 
                   and not line.startswith('#')]
    
    suspicious = []
    for package in packages:
        if not check_package_source(package):
            suspicious.append(package)
    
    if suspicious:
        print(f"⚠️  Suspicious packages detected: {suspicious}")
        sys.exit(1)
    else:
        print("✅ All packages verified")

if __name__ == '__main__':
    main()
```

### Dependency Pinning Best Practices

```txt
# requirements.txt - Pin exact versions

# Core dependencies - use exact versions
bcrypt==5.0.0
argon2-cffi==25.1.0

# Framework dependencies - use compatible releases
fastapi~=0.127.0  # Allows 0.127.x
pydantic~=2.12.5  # Allows 2.12.x

# Security-critical - exact versions only
cryptography==44.0.3
PyJWT==2.8.0

# Development dependencies - can be more flexible
pytest>=7.0.0,<8.0.0
black>=23.0.0,<24.0.0
```

### Regular Dependency Updates

```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade package-name

# Update all (with caution)
pip-compile --upgrade requirements.in

# Test thoroughly after updates
pytest tests/
```

### Supply Chain Security

**1. Enable Dependabot**

`.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "security-team"
    labels:
      - "dependencies"
      - "security"
```

**2. SBOM Generation**

```bash
# Generate Software Bill of Materials
pip install cyclonedx-bom
cyclonedx-py -r -i requirements.txt -o sbom.json

# Or use syft
syft packages dir:. -o json > sbom.json
```

**3. Vulnerability Scanning**

```bash
# Trivy scan
trivy fs --severity HIGH,CRITICAL .

# Grype scan
grype dir:. --fail-on high
```

---

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)

---

**Last Updated**: 2025-12-25

For security issues, please see [SECURITY.md](../SECURITY.md) for reporting procedures.
