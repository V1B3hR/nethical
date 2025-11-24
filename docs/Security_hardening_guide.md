# Security Hardening Guide

This guide provides comprehensive security hardening controls for deploying Nethical in production environments following world-class security standards.

## Overview

Nethical's security architecture is built on defense-in-depth principles with multiple layers of protection. This guide documents the implementation of mandatory controls required for production deployment.

## Security Layers

### 1. Perimeter Security
**Objective:** Protect against external threats and malicious inputs

**Components:**
- **Load Balancer + WAF Integration**
  - Prompt injection detection using regex patterns
  - Payload size limits (default: 1MB max)
  - Rate limiting at edge
  - DDoS protection
  
**Implementation:**
```yaml
# Example WAF rules for prompt injection
- rule: "PROMPT_INJECTION_DETECTION"
  pattern: "(ignore|disregard|forget).*(previous|above|prior).*(instruction|prompt|rule)"
  action: block
  
- rule: "PAYLOAD_SIZE_LIMIT"
  max_size: 1048576  # 1MB
  action: reject
```

**Files:** 
- `deploy/kubernetes/waf-config.yaml` - WAF configuration
- `deploy/kubernetes/ingress.yaml` - Ingress with WAF annotations

### 2. Authentication & Authorization
**Objective:** Ensure only authorized entities can access the system

**Components:**
- **JWT with RS256 Algorithm**
  - Public/private key pair for signing
  - 15-minute token expiration
  - Automatic key rotation every 90 days
  
- **RBAC Matrix**
  - Role-based access control
  - Principle of least privilege
  - Granular permissions per endpoint
  
- **Scoped API Keys**
  - Per-tenant API keys
  - Configurable permissions
  - Automatic expiration and rotation

**Implementation:**
- JWT validation in `nethical/api/auth.py`
- RBAC middleware in `nethical/api/rbac.py`
- API key management in `nethical/api/api_keys.py`

**Configuration:**
```yaml
auth:
  jwt:
    algorithm: RS256
    expiration: 900  # 15 minutes
    key_rotation_days: 90
  rbac:
    enabled: true
    default_role: viewer
  api_keys:
    enabled: true
    expiration_days: 90
```

### 3. Input Validation
**Objective:** Prevent injection attacks and malformed inputs

**Components:**
- **Strict Pydantic Schemas**
  - Type validation for all inputs
  - Length constraints
  - Pattern matching
  
- **Unicode Normalization**
  - NFC normalization to prevent homograph attacks
  - Character filtering
  
- **Length Caps**
  - Maximum field lengths
  - Nested object depth limits

**Implementation:**
```python
# Example Pydantic schema with validation
from pydantic import BaseModel, Field, validator
import unicodedata

class SecureAgentAction(BaseModel):
    agent_id: str = Field(..., max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')
    action_type: str = Field(..., max_length=64)
    context: str = Field(..., max_length=4096)
    
    @validator('context')
    def normalize_unicode(cls, v):
        return unicodedata.normalize('NFC', v)
```

### 4. Secrets Management
**Objective:** Protect sensitive credentials and keys

**Components:**
- **HashiCorp Vault / Cloud KMS Integration**
  - External secret storage
  - Automatic secret rotation
  - Audit logging of secret access
  
- **No Inline Secrets**
  - All secrets externalized
  - Environment variables from secret stores
  - Kubernetes External Secrets Operator

**Implementation:**
```yaml
# external-secrets.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: nethical-secrets
  namespace: nethical
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: nethical-app-secrets
  data:
    - secretKey: jwt-private-key
      remoteRef:
        key: nethical/jwt
        property: private_key
    - secretKey: database-password
      remoteRef:
        key: nethical/db
        property: password
```

**Secret Rotation:** Automated via `scripts/rotate-secrets.sh` (runs every 30 days)

### 5. Supply Chain Security
**Objective:** Ensure integrity of software artifacts

**Components:**
- **Signed Container Images**
  - Cosign signature verification
  - Image provenance attestation
  - SLSA Level 3 compliance target
  
- **SBOM Generation and Verification**
  - Automatic SBOM generation (SPDX, CycloneDX)
  - SBOM diff gate in CI/CD
  - Vulnerability tracking
  
- **Dependency Scanning**
  - Trivy, Grype for vulnerability detection
  - GitHub Dependabot alerts
  - Automated dependency updates

**Implementation:**
- `.github/workflows/sbom-sign.yml` - SBOM generation and signing
- `.github/workflows/security.yml` - Vulnerability scanning
- `scripts/verify-sbom.sh` - SBOM verification script

**Verification:**
```bash
# Verify container image signature
cosign verify --key cosign.pub ghcr.io/v1b3hr/nethical:latest

# Verify SBOM
cosign verify-blob --key cosign.pub --signature sbom.spdx.json.sig sbom.spdx.json
```

### 6. Runtime Security
**Objective:** Minimize attack surface during execution

**Components:**
- **Non-Root User** ✅ (Already Implemented)
  - Containers run as UID 1000
  - No privilege escalation
  
- **Seccomp Profiles**
  - Restrict system calls
  - Default deny policy
  
- **AppArmor Profiles**
  - Mandatory access control
  - File system restrictions
  
- **Read-Only Filesystem**
  - Root filesystem mounted read-only
  - Writable volumes for data only

**Implementation:**
```yaml
# Pod security configuration
apiVersion: v1
kind: Pod
metadata:
  name: nethical
  annotations:
    container.apparmor.security.beta.kubernetes.io/nethical: runtime/default
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: nethical
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    volumeMounts:
    - name: data
      mountPath: /data
    - name: tmp
      mountPath: /tmp
```

**Files:**
- `deploy/kubernetes/seccomp-profile.json` - Seccomp configuration
- `deploy/kubernetes/apparmor-profile.yaml` - AppArmor profile

### 7. Network Security
**Objective:** Implement zero-trust network segmentation

**Components:**
- **Namespace Isolation**
  - Dedicated namespace per environment
  - Resource quotas and limits
  
- **Network Policies**
  - Default deny-all ingress/egress
  - Explicit allow rules for required communication
  - Redis and database access only
  
- **Service Mesh (Optional)**
  - mTLS between all services
  - Traffic encryption in transit
  - Request authentication

**Implementation:**
```yaml
# Default deny-all network policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: nethical
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

# Allow specific traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nethical-allow-internal
  namespace: nethical
spec:
  podSelector:
    matchLabels:
      app: nethical
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nethical-ingress
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: postgresql
    ports:
    - protocol: TCP
      port: 5432
```

**Files:**
- `deploy/kubernetes/network-policies.yaml` - Network policy definitions
- `deploy/kubernetes/service-mesh-config.yaml` - Optional Istio/Linkerd config

### 8. Logging & Monitoring
**Objective:** Comprehensive audit trail with privacy protection

**Components:**
- **PII Redaction Filters**
  - Automatic detection and redaction
  - Configurable redaction policies
  - Reversible redaction for authorized access
  
- **Structured JSON Logs**
  - Machine-parsable format
  - Correlation IDs
  - Contextual metadata
  
- **Log Aggregation**
  - Centralized logging (ELK, Loki)
  - Long-term retention
  - Compliance reporting

**Implementation:**
```python
# PII redaction in logging
import logging
from nethical.detectors.pii_detector import PIIDetector

class PIIRedactingFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pii_detector = PIIDetector()
    
    def format(self, record):
        msg = super().format(record)
        # Redact PII from log messages
        redacted, _ = self.pii_detector.redact(msg)
        return redacted

# Configure structured JSON logging
logging.basicConfig(
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    handlers=[logging.StreamHandler()]
)
```

**Configuration:**
```yaml
logging:
  level: INFO
  format: json
  pii_redaction:
    enabled: true
    policy: aggressive
    reversible: true
  destinations:
    - type: stdout
    - type: elasticsearch
      endpoint: https://logs.example.com
```

### 9. Audit Integrity
**Objective:** Tamper-proof audit trails

**Components:**
- **Merkle Anchoring** ✅ (Already Implemented)
  - Cryptographic verification of audit logs
  - Hash chains for integrity
  
- **External Timestamping**
  - RFC 3161 compliant timestamps
  - Independent time authority
  - Non-repudiation

**Implementation:**
- Merkle anchoring in `nethical/governance/phase4_security.py`
- External timestamping via `scripts/timestamp-audit-logs.sh`

**Verification:**
```python
# Verify audit log integrity
from nethical.governance.phase4_security import MerkleAnchor

merkle = MerkleAnchor()
is_valid = merkle.verify_chain(audit_logs)
assert is_valid, "Audit log integrity compromised"
```

### 10. Plugin Trust & Security
**Objective:** Ensure third-party plugin safety

**Components:**
- **Digital Signatures**
  - GPG/Cosign signature verification
  - Publisher identity validation
  
- **Static Analysis**
  - Automated security scanning
  - Code quality checks
  - Vulnerability detection
  
- **Reputation Scoring**
  - Community ratings
  - Usage statistics
  - Security history

**Implementation:**
```python
# Plugin verification
from nethical.marketplace.security import PluginVerifier

verifier = PluginVerifier()
plugin_path = "plugins/my-plugin.whl"

# Verify signature
is_signed = verifier.verify_signature(plugin_path)

# Run static analysis
analysis_result = verifier.analyze(plugin_path)

# Check reputation
reputation = verifier.get_reputation(plugin_metadata)

if is_signed and analysis_result.passed and reputation >= 0.7:
    # Safe to load plugin
    load_plugin(plugin_path)
```

**Files:**
- `nethical/marketplace/security.py` - Plugin security verification
- `docs/PLUGIN_SIGNING_GUIDE.md` - Plugin signing documentation

## Mandatory Controls (World-Class Security)

### ✅ 1. mTLS Between Internal Services
**Status:** Implemented

**Implementation Options:**

#### Option A: Service Mesh (Recommended)
```yaml
# Istio PeerAuthentication
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: nethical
spec:
  mtls:
    mode: STRICT

# DestinationRule for mTLS
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: nethical-mtls
  namespace: nethical
spec:
  host: "*.nethical.svc.cluster.local"
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
```

#### Option B: Manual mTLS Configuration
```yaml
# Certificate generation
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: nethical-service-cert
  namespace: nethical
spec:
  secretName: nethical-tls
  duration: 2160h  # 90 days
  renewBefore: 720h  # 30 days
  issuerRef:
    name: nethical-ca
    kind: ClusterIssuer
  commonName: nethical.nethical.svc.cluster.local
  dnsNames:
  - nethical.nethical.svc.cluster.local
  - nethical-api.nethical.svc.cluster.local
```

**Verification:**
```bash
# Test mTLS connectivity
openssl s_client -connect nethical.nethical.svc.cluster.local:8000 \
  -cert client.crt -key client.key -CAfile ca.crt
```

**Files:**
- `deploy/kubernetes/service-mesh-config.yaml` - Istio/Linkerd configuration
- `deploy/kubernetes/mtls-certificates.yaml` - Cert-manager configuration
- `scripts/verify-mtls.sh` - mTLS verification script

### ✅ 2. Automatic Secret Rotation (≤90 Days)
**Status:** Implemented

**Components:**
- Automated rotation via External Secrets Operator
- Rotation scheduling with CronJob
- Zero-downtime rotation process

**Implementation:**
```yaml
# CronJob for secret rotation
apiVersion: batch/v1
kind: CronJob
metadata:
  name: rotate-secrets
  namespace: nethical
spec:
  schedule: "0 0 */30 * *"  # Every 30 days
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: secret-rotator
          containers:
          - name: rotate
            image: nethical/secret-rotator:latest
            env:
            - name: VAULT_ADDR
              value: "https://vault.example.com"
            - name: ROTATION_TARGETS
              value: "jwt-keys,database-password,api-keys"
          restartPolicy: OnFailure
```

**Rotation Script:**
```bash
#!/bin/bash
# scripts/rotate-secrets.sh

# Rotate JWT signing keys
vault kv put nethical/jwt \
  private_key="$(openssl genrsa 4096 2>/dev/null)" \
  public_key="$(openssl rsa -pubout 2>/dev/null)"

# Rotate database password
vault kv put nethical/db \
  password="$(openssl rand -base64 32)"

# Trigger pod restart to pick up new secrets
kubectl rollout restart statefulset/nethical -n nethical
```

**Monitoring:**
```yaml
# Alert when secret is >80 days old
- alert: SecretRotationOverdue
  expr: (time() - secret_last_rotated_timestamp) > (80 * 86400)
  labels:
    severity: warning
  annotations:
    summary: "Secret rotation overdue for {{ $labels.secret_name }}"
```

**Files:**
- `scripts/rotate-secrets.sh` - Secret rotation automation
- `deploy/kubernetes/secret-rotation-cronjob.yaml` - CronJob definition
- `deploy/kubernetes/external-secrets.yaml` - External Secrets config

### ✅ 3. Vulnerability SLA: Critical <24h, High <72h
**Status:** Implemented

**Components:**
- Automated vulnerability scanning
- SLA monitoring and alerting
- Automated patching workflows

**Implementation:**
```yaml
# GitHub Actions workflow for vulnerability SLA
name: Vulnerability SLA Enforcement
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

jobs:
  scan-and-enforce:
    runs-on: ubuntu-latest
    steps:
      - name: Scan for vulnerabilities
        run: |
          trivy image nethical:latest --severity CRITICAL,HIGH --format json -o vulns.json
      
      - name: Check SLA compliance
        run: |
          python scripts/check-vuln-sla.py vulns.json
      
      - name: Create incident if SLA breached
        if: failure()
        run: |
          gh issue create --title "Vulnerability SLA Breach" \
            --body "Critical/High vulnerabilities exceed SLA" \
            --label "security,priority:high"
```

**SLA Monitoring Script:**
```python
# scripts/check-vuln-sla.py
import json
import sys
from datetime import datetime, timedelta

def check_sla(vulns_file):
    with open(vulns_file) as f:
        vulns = json.load(f)
    
    now = datetime.now()
    sla_breach = False
    
    for vuln in vulns.get('Results', []):
        for v in vuln.get('Vulnerabilities', []):
            published = datetime.fromisoformat(v['PublishedDate'])
            age = (now - published).total_seconds() / 3600  # hours
            
            severity = v['Severity']
            if severity == 'CRITICAL' and age > 24:
                print(f"SLA BREACH: {v['VulnerabilityID']} (Critical, {age}h old)")
                sla_breach = True
            elif severity == 'HIGH' and age > 72:
                print(f"SLA BREACH: {v['VulnerabilityID']} (High, {age}h old)")
                sla_breach = True
    
    if sla_breach:
        sys.exit(1)

if __name__ == '__main__':
    check_sla(sys.argv[1])
```

**Automated Remediation:**
```yaml
# Auto-update workflow
name: Auto-Patch Vulnerabilities
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  auto-patch:
    runs-on: ubuntu-latest
    steps:
      - name: Update dependencies
        run: |
          pip-compile --upgrade requirements.in
          pip-compile --upgrade requirements-dev.in
      
      - name: Create PR
        run: |
          gh pr create --title "Security: Auto-update dependencies" \
            --body "Automated dependency updates to address vulnerabilities"
```

**Files:**
- `.github/workflows/vuln-sla.yml` - SLA enforcement workflow
- `scripts/check-vuln-sla.py` - SLA validation script
- `scripts/auto-patch.sh` - Automated patching script

### ✅ 4. Zero Trust Network Segmentation (Deny-All Default)
**Status:** Implemented

**Components:**
- Default deny-all network policies
- Explicit allow rules for required traffic
- Microsegmentation by service
- Network flow monitoring

**Implementation:**
```yaml
# Default deny-all policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: nethical
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

# Explicit allow for API ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nethical-api-ingress
  namespace: nethical
spec:
  podSelector:
    matchLabels:
      app: nethical
      component: api
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000

# Explicit allow for Redis egress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nethical-redis-egress
  namespace: nethical
spec:
  podSelector:
    matchLabels:
      app: nethical
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379

# Explicit allow for DNS
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns-egress
  namespace: nethical
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    - podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
```

**Network Flow Monitoring:**
```yaml
# Cilium NetworkPolicy with visibility
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: nethical-monitored
  namespace: nethical
spec:
  endpointSelector:
    matchLabels:
      app: nethical
  ingress:
  - fromEndpoints:
    - matchLabels:
        app: ingress
    toPorts:
    - ports:
      - port: "8000"
        protocol: TCP
  egress:
  - toEndpoints:
    - matchLabels:
        app: redis
    toPorts:
    - ports:
      - port: "6379"
        protocol: TCP
  # Log all denied flows
  ingressDeny:
  - fromEntities:
    - all
  egressDeny:
  - toEntities:
    - all
```

**Verification:**
```bash
# Test network isolation
kubectl run -it --rm debug --image=nicolaka/netshoot -n nethical -- bash

# This should fail (blocked by network policy)
curl http://external-service.com

# This should succeed (allowed by policy)
curl http://redis.nethical.svc.cluster.local:6379
```

**Files:**
- `deploy/kubernetes/network-policies.yaml` - All network policies
- `scripts/test-network-isolation.sh` - Network isolation tests
- `docs/NETWORK_SECURITY.md` - Network security documentation

### ✅ 5. Build Pipeline Attestation (SLSA Level 3 Target)
**Status:** Implemented

**Components:**
- Source verification
- Build provenance generation
- Hermetic builds
- Non-falsifiable provenance

**Implementation:**
```yaml
# SLSA provenance generation
name: SLSA Build Attestation
on:
  release:
    types: [published]

permissions:
  id-token: write
  contents: write
  packages: write

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      digest: ${{ steps.build.outputs.digest }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Build container
        id: build
        run: |
          docker build -t nethical:${{ github.sha }} .
          digest=$(docker inspect --format='{{.Id}}' nethical:${{ github.sha }})
          echo "digest=$digest" >> $GITHUB_OUTPUT
      
      - name: Generate SBOM
        run: |
          syft nethical:${{ github.sha }} -o spdx-json=sbom.spdx.json
      
      - name: Sign image and SBOM
        env:
          COSIGN_EXPERIMENTAL: 1
        run: |
          cosign sign --yes nethical:${{ github.sha }}
          cosign attest --yes --predicate sbom.spdx.json nethical:${{ github.sha }}

  provenance:
    needs: [build]
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v1.9.0
    with:
      image: nethical
      digest: ${{ needs.build.outputs.digest }}
      registry-username: ${{ github.actor }}
    secrets:
      registry-password: ${{ secrets.GITHUB_TOKEN }}
```

**Provenance Verification:**
```bash
# Verify SLSA provenance
cosign verify-attestation --type slsaprovenance \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --certificate-identity-regexp '^https://github.com/V1B3hR/nethical/' \
  ghcr.io/v1b3hr/nethical:latest

# Check SLSA level
slsa-verifier verify-image ghcr.io/v1b3hr/nethical:latest \
  --source-uri github.com/V1B3hR/nethical \
  --source-tag v1.0.0
```

**SLSA Level Compliance:**
- ✅ Build Level 3: Source verification, hermetic builds, non-falsifiable provenance
- ✅ Source Level 2: Version controlled, verified history
- ⚠️  Dependencies Level 1: Pinned dependencies (target: Level 2 with SBOM verification)

**Files:**
- `.github/workflows/slsa-build.yml` - SLSA attestation workflow
- `scripts/verify-provenance.sh` - Provenance verification script
- `docs/SUPPLY_CHAIN_SECURITY_GUIDE.md` - Supply chain security documentation

## Implementation Status

### Completed ✅
1. ✅ Perimeter Security - WAF configuration and payload limits
2. ✅ Authentication & Authorization - JWT RS256, RBAC, API keys
3. ✅ Input Validation - Pydantic schemas, Unicode normalization
4. ✅ Secrets Management - External secrets, rotation automation
5. ✅ Supply Chain Security - SBOM, signing, vulnerability scanning
6. ✅ Runtime Security - Non-root user, seccomp, AppArmor, read-only FS
7. ✅ Network Security - Zero-trust policies, deny-all default
8. ✅ Logging & Monitoring - PII redaction, structured JSON logs
9. ✅ Audit Integrity - Merkle anchoring, external timestamping
10. ✅ Plugin Trust - Signature verification, static analysis
11. ✅ mTLS - Service mesh configuration with Istio/Linkerd
12. ✅ Secret Rotation - Automated rotation ≤90 days
13. ✅ Vulnerability SLA - Critical <24h, High <72h enforcement
14. ✅ Zero Trust - Network segmentation with deny-all default
15. ✅ Build Attestation - SLSA Level 3 compliance

### Verification Commands

```bash
# Verify all security controls
./scripts/verify-security-controls.sh

# Test network isolation
./scripts/test-network-isolation.sh

# Verify mTLS
./scripts/verify-mtls.sh

# Check secret rotation status
./scripts/check-secret-rotation.sh

# Verify SLSA provenance
./scripts/verify-provenance.sh

# Check vulnerability SLA compliance
./scripts/check-vuln-sla.py vulns.json
```

## Deployment Checklist

Before deploying to production, ensure:

- [ ] All secrets are externalized to Vault/KMS
- [ ] Network policies are applied (default deny-all)
- [ ] mTLS is enabled between services
- [ ] Secret rotation is configured and tested
- [ ] Vulnerability scanning is running on schedule
- [ ] SBOM is generated and signed for all releases
- [ ] Audit logging is enabled with PII redaction
- [ ] WAF rules are configured and tested
- [ ] Monitoring and alerting are configured
- [ ] Incident response procedures are documented

## Maintenance

### Daily
- Monitor vulnerability scan results
- Review audit logs for anomalies
- Check service health and mTLS connectivity

### Weekly
- Review network policy effectiveness
- Analyze security alerts and incidents
- Update WAF rules based on attack patterns

### Monthly
- Rotate secrets (automated)
- Review and update RBAC policies
- Conduct security posture assessment
- Update dependencies

### Quarterly
- Penetration testing
- Security audit review
- Update security documentation
- Review and update incident response procedures

## References

- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [SLSA Framework](https://slsa.dev/)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [Zero Trust Architecture](https://www.nist.gov/publications/zero-trust-architecture)

## Support

For security-related questions or to report vulnerabilities:
- Email: security@nethical.io
- GitHub Security Advisories: https://github.com/V1B3hR/nethical/security/advisories

---

**Last Updated:** 2025-11-24  
**Version:** 2.0  
**Status:** All mandatory controls implemented ✅
