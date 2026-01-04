# Nethical Security Hardening Guide

This comprehensive guide defines mandatory security controls, implementation patterns, and verification procedures for deploying Nethical to production with world-class security standards.

---

## Table of Contents

1. [Threat Model Summary](#threat-model-summary)
2. [Mandatory Controls Matrix](#mandatory-controls-matrix)
3. [Build & Supply Chain Security](#build--supply-chain-security)
4. [Runtime Hardening](#runtime-hardening)
5. [Authentication & Authorization](#authentication--authorization)
6. [Input Validation & Payload Limits](#input-validation--payload-limits)
7. [Secrets Management](#secrets-management)
8. [Logging & PII Redaction](#logging--pii-redaction)
9. [WAF Rules & Edge Protection](#waf-rules--edge-protection)
10. [Dependency & Image Scanning](#dependency--image-scanning)
11. [Vulnerability Response](#vulnerability-response)
12. [Plugin Trust & Verification](#plugin-trust--verification)
13. [Continuous Security Verification](#continuous-security-verification)

---

## Threat Model Summary

### Primary Threat Vectors

1. **Prompt Injection Attacks**: Malicious prompts attempting to bypass safety controls
2. **Supply Chain Compromise**: Dependency vulnerabilities, malicious packages, compromised build artifacts
3. **Data Exfiltration**: Unauthorized access to PII, audit logs, or decision data
4. **Privilege Escalation**: Attempts to gain unauthorized access or elevated permissions
5. **Denial of Service**: Resource exhaustion, rate limit bypass, cache poisoning
6. **Plugin Ecosystem Attacks**: Malicious or vulnerable third-party plugins
7. **Insider Threats**: Unauthorized access by authenticated users with compromised credentials

### Security Architecture Principles

- **Defense in Depth**: Multiple overlapping security layers
- **Zero Trust**: Verify every request, deny by default
- **Least Privilege**: Minimal permissions for all components
- **Fail Secure**: Failures default to secure state
- **Audit Everything**: Comprehensive tamper-proof audit trail

---

## Mandatory Controls Matrix

| Control Domain | Control | Tool/Method | Verification Cadence | Owner |
|----------------|---------|-------------|---------------------|-------|
| **Supply Chain** | SBOM Generation | Syft/CycloneDX | Every build | DevOps |
| **Supply Chain** | Image Signing | Cosign | Every build | DevOps |
| **Supply Chain** | Provenance Attestation | SLSA Level 3 | Every release | DevOps |
| **Supply Chain** | Dependency Scanning | Trivy, Grype, Dependabot | Daily | Security |
| **Runtime** | Non-root User | Container UID 1000 | Every deployment | DevOps |
| **Runtime** | Read-only Filesystem | securityContext | Every deployment | DevOps |
| **Runtime** | Seccomp Profile | RuntimeDefault | Every deployment | DevOps |
| **Runtime** | AppArmor Profile | runtime/default | Every deployment | DevOps |
| **Network** | Zero Trust Segmentation | NetworkPolicy | Every deployment | NetSec |
| **Network** | mTLS Service Communication | Istio/Linkerd | Continuous | NetSec |
| **Network** | Deny-All Default | NetworkPolicy | Every deployment | NetSec |
| **Authentication** | JWT RS256 | API middleware | Continuous | AppSec |
| **Authentication** | API Key Validation | API middleware | Continuous | AppSec |
| **Authorization** | RBAC Enforcement | RBAC middleware | Continuous | AppSec |
| **Input Validation** | Pydantic Schemas | FastAPI validation | Continuous | AppSec |
| **Input Validation** | Unicode Normalization | Custom validators | Continuous | AppSec |
| **Input Validation** | Length Caps | Field validators | Continuous | AppSec |
| **Secrets** | External Secrets | Vault/KMS/ESO | Every deployment | DevOps |
| **Secrets** | Rotation (≤90 days) | Automated CronJob | Monthly | DevOps |
| **Logging** | PII Redaction | Custom formatter | Continuous | AppSec |
| **Logging** | Structured JSON | Python logging | Continuous | DevOps |
| **Edge** | WAF Rules | Cloud WAF/ModSecurity | Every deployment | NetSec |
| **Edge** | Payload Limits | Nginx/WAF | Every deployment | NetSec |
| **Edge** | Rate Limiting | API Gateway | Every deployment | DevOps |
| **Vulnerability** | Critical <24h | Automated workflow | Every 6 hours | Security |
| **Vulnerability** | High <72h | Automated workflow | Daily | Security |
| **Plugin** | Signature Verification | Cosign/GPG | Plugin load | AppSec |
| **Plugin** | Trust Score ≥80 | Reputation system | Plugin install | Marketplace |
| **Plugin** | Vulnerability Scan | Trivy | Plugin load | Security |
| **Audit** | Merkle Anchoring | Custom implementation | Every 5 minutes | Governance |
| **Audit** | External Timestamping | RFC 3161 TSA | Hourly | Governance |

---

## Build & Supply Chain Security

### SBOM Generation

**Objective**: Comprehensive bill of materials for all dependencies and artifacts.

**Implementation** (Reference Example):

**Note**: See actual implementation in `.github/workflows/sbom-sign.yml` for current SBOM generation process.

```yaml
# Example: .github/workflows/sbom.yml
name: SBOM Generation
on:
  push:
    branches: [main]
  release:
    types: [published]

jobs:
  generate-sbom:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Generate SBOM (SPDX)
        run: |
          syft dir:. -o spdx-json=sbom.spdx.json
      
      - name: Generate SBOM (CycloneDX)
        run: |
          syft dir:. -o cyclonedx-json=sbom.cyclonedx.json
      
      - name: Sign SBOM
        env:
          COSIGN_EXPERIMENTAL: 1
        run: |
          cosign sign-blob --yes sbom.spdx.json \
            --output-signature sbom.spdx.json.sig
          cosign sign-blob --yes sbom.cyclonedx.json \
            --output-signature sbom.cyclonedx.json.sig
      
      - name: Upload SBOM Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: sbom-${{ github.sha }}
          path: |
            sbom.*.json
            sbom.*.sig
```

**Verification**:

```bash
# Verify SBOM signature
cosign verify-blob --key cosign.pub \
  --signature sbom.spdx.json.sig \
  sbom.spdx.json

# Analyze SBOM for vulnerabilities
grype sbom:sbom.spdx.json --fail-on critical
```

**Tools**: Syft, CycloneDX Generator, Cosign

**Cadence**: Every build

**Artifacts**: `SBOM.json` published to release assets and registry

---

### Container Image Signing

**Objective**: Cryptographic verification of container image authenticity and integrity.

**Implementation** (Reference Example):

**Note**: This is a reference implementation. Adapt to your specific container registry and signing workflow.

```yaml
# Example: .github/workflows/container-sign.yml
name: Sign Container Images
on:
  push:
    tags: ['v*']

permissions:
  id-token: write
  packages: write
  contents: read

jobs:
  sign:
    runs-on: ubuntu-latest
    steps:
      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and Push
        id: build
        run: |
          docker build -t ghcr.io/v1b3hr/nethical:${{ github.ref_name }} .
          docker push ghcr.io/v1b3hr/nethical:${{ github.ref_name }}
          digest=$(docker inspect --format='{{index .RepoDigests 0}}' \
            ghcr.io/v1b3hr/nethical:${{ github.ref_name }})
          echo "digest=$digest" >> $GITHUB_OUTPUT
      
      - name: Sign Image with Keyless
        env:
          COSIGN_EXPERIMENTAL: 1
        run: |
          cosign sign --yes ${{ steps.build.outputs.digest }}
      
      - name: Verify Signature
        env:
          COSIGN_EXPERIMENTAL: 1
        run: |
          cosign verify \
            --certificate-identity-regexp '^https://github.com/V1B3hR/nethical/' \
            --certificate-oidc-issuer https://token.actions.githubusercontent.com \
            ${{ steps.build.outputs.digest }}
```

**Deployment-time Verification**:

```yaml
# deploy/kubernetes/image-policy.yaml
apiVersion: policy.sigstore.dev/v1beta1
kind: ClusterImagePolicy
metadata:
  name: nethical-image-policy
spec:
  images:
  - glob: "ghcr.io/v1b3hr/nethical:*"
  authorities:
  - keyless:
      url: https://fulcio.sigstore.dev
      identities:
      - issuerRegExp: ".*github.com/V1B3hR/nethical.*"
```

**Tools**: Cosign (Sigstore), Policy Controller

**SLSA Goal**: Level 3 (non-falsifiable provenance, hermetic builds)

---

### Supply Chain Levels for Software Artifacts (SLSA)

**Current Status**: Level 2 (Build provenance, version controlled source)

**Level 3 Requirements** (target):

- ✅ Source integrity: Git commit signing enforced
- ✅ Build service: GitHub Actions with OIDC authentication
- ✅ Provenance generation: SLSA provenance generated per build
- 🔄 Hermetic builds: In progress (isolated build environment)
- 🔄 Non-falsifiable provenance: Keyless signing with Sigstore

**Implementation** (Reference Example):

**Note**: SLSA provenance generation requires specific build pipeline configuration. This is a reference implementation.

```yaml
# Example: .github/workflows/slsa-build.yml
name: SLSA Build Provenance
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
      
      - name: Build Container
        id: build
        run: |
          docker build -t nethical:${{ github.sha }} .
          digest=$(docker inspect --format='{{.Id}}' nethical:${{ github.sha }})
          echo "digest=$digest" >> $GITHUB_OUTPUT
  
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

**Verification**:

```bash
# Verify SLSA provenance
slsa-verifier verify-image ghcr.io/v1b3hr/nethical:latest \
  --source-uri github.com/V1B3hR/nethical \
  --source-tag v1.0.0
```

---

## Runtime Hardening

### Non-Root User Execution

**Objective**: Prevent privilege escalation via container breakout.

**Implementation**:

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r nethical -g 1000 && \
    useradd -r -g nethical -u 1000 nethical && \
    mkdir -p /app /data /tmp && \
    chown -R nethical:nethical /app /data /tmp

USER nethical
WORKDIR /app

COPY --chown=nethical:nethical . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "nethical.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Kubernetes Configuration** (Reference Example):

**Note**: See `deploy/kubernetes/` directory for actual Kubernetes configurations. This is a reference example for deployment security context.

```yaml
# Example: deploy/kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nethical
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
      - name: nethical
        image: ghcr.io/v1b3hr/nethical:latest
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
              - ALL
```

**Verification**:

```bash
# Verify user in running container
kubectl exec -it nethical-xxx -- id
# Expected: uid=1000(nethical) gid=1000(nethical)
```

---

### Read-Only Root Filesystem

**Objective**: Prevent runtime filesystem modifications and malware persistence.

**Implementation**:

```yaml
# deploy/kubernetes/deployment.yaml
spec:
  containers:
  - name: nethical
    securityContext:
      readOnlyRootFilesystem: true
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: data
      mountPath: /data
    - name: cache
      mountPath: /app/.cache
  volumes:
  - name: tmp
    emptyDir: {}
  - name: data
    persistentVolumeClaim:
      claimName: nethical-data
  - name: cache
    emptyDir: {}
```

**Testing**:

```bash
# Attempt to write to root filesystem (should fail)
kubectl exec -it nethical-xxx -- touch /test.txt
# Expected: touch: cannot touch '/test.txt': Read-only file system
```

---

### Seccomp and AppArmor Profiles

**Objective**: Restrict system calls and mandatory access control.

**Seccomp Profile**:

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": [
        "accept", "accept4", "access", "arch_prctl", "bind", "brk",
        "clone", "close", "connect", "dup", "dup2", "epoll_create",
        "epoll_ctl", "epoll_wait", "exit", "exit_group", "fcntl",
        "fstat", "futex", "getcwd", "getdents64", "getpeername",
        "getpid", "getppid", "getsockname", "getsockopt", "listen",
        "lseek", "mmap", "mprotect", "munmap", "open", "openat",
        "pipe", "poll", "read", "readlink", "recvfrom", "rt_sigaction",
        "rt_sigprocmask", "sendto", "set_robust_list", "setsockopt",
        "socket", "stat", "write"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

**Kubernetes Configuration**:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nethical
  annotations:
    container.apparmor.security.beta.kubernetes.io/nethical: runtime/default
spec:
  securityContext:
    seccompProfile:
      type: Localhost
      localhostProfile: nethical-seccomp.json
```

**Files**: `deploy/kubernetes/seccomp-profile.json`, `deploy/kubernetes/apparmor-profile.yaml`

---

## Authentication & Authorization

### JWT Validation (RS256)

**Objective**: Cryptographically strong authentication with asymmetric keys.

**Implementation**:

```python
# nethical/api/auth.py
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from datetime import datetime, timedelta

security = HTTPBearer()

class JWTAuth:
    def __init__(self, public_key: str, algorithm: str = "RS256"):
        self.public_key = public_key
        self.algorithm = algorithm
    
    def verify_token(
        self, 
        credentials: HTTPAuthorizationCredentials = Security(security)
    ) -> dict:
        try:
            payload = jwt.decode(
                credentials.credentials,
                self.public_key,
                algorithms=[self.algorithm],
                options={"verify_exp": True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Key generation
def generate_rsa_keypair():
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=4096
    )
    
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_pem, public_pem
```

**Configuration**:

```yaml
# config/auth.yaml
jwt:
  algorithm: RS256
  expiration_seconds: 900  # 15 minutes
  issuer: "nethical.io"
  audience: "api.nethical.io"
  key_rotation_days: 90
```

**Key Rotation**:

Automated via External Secrets Operator and Kubernetes CronJob (every 90 days).

---

### RBAC (Role-Based Access Control)

**Objective**: Enforce least-privilege access based on roles.

**Role Definition**:

```python
# nethical/api/rbac.py
from enum import Enum
from typing import List, Set

class Role(str, Enum):
    ADMIN = "admin"
    REVIEWER = "reviewer"
    DEVELOPER = "developer"
    VIEWER = "viewer"

class Permission(str, Enum):
    # Decision permissions
    DECISION_EVALUATE = "decision:evaluate"
    DECISION_VIEW = "decision:view"
    
    # Policy permissions
    POLICY_CREATE = "policy:create"
    POLICY_UPDATE = "policy:update"
    POLICY_DELETE = "policy:delete"
    POLICY_VIEW = "policy:view"
    
    # Review permissions
    REVIEW_SUBMIT = "review:submit"
    REVIEW_VIEW = "review:view"
    
    # Audit permissions
    AUDIT_VIEW = "audit:view"
    AUDIT_EXPORT = "audit:export"

# Role to permissions mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.REVIEWER: {
        Permission.DECISION_VIEW,
        Permission.REVIEW_SUBMIT,
        Permission.REVIEW_VIEW,
        Permission.AUDIT_VIEW,
    },
    Role.DEVELOPER: {
        Permission.DECISION_EVALUATE,
        Permission.DECISION_VIEW,
        Permission.POLICY_VIEW,
    },
    Role.VIEWER: {
        Permission.DECISION_VIEW,
        Permission.POLICY_VIEW,
    },
}

def require_permission(permission: Permission):
    def decorator(func):
        async def wrapper(*args, user: dict, **kwargs):
            user_role = Role(user.get("role", "viewer"))
            if permission not in ROLE_PERMISSIONS[user_role]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator
```

**Usage**:

```python
@app.post("/api/v1/policy")
@require_permission(Permission.POLICY_CREATE)
async def create_policy(policy: PolicyCreate, user: dict = Depends(jwt_auth.verify_token)):
    # Policy creation logic
    pass
```

---

### mTLS for Service-to-Service Communication

**Objective**: Encrypted and authenticated communication between internal services.

**Istio Implementation** (Recommended):

```yaml
# deploy/kubernetes/istio-mtls.yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: nethical
spec:
  mtls:
    mode: STRICT

---
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

**Manual mTLS Configuration**:

```yaml
# deploy/kubernetes/mtls-certificates.yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: nethical-api-cert
  namespace: nethical
spec:
  secretName: nethical-api-tls
  duration: 2160h  # 90 days
  renewBefore: 720h  # 30 days before expiry
  issuerRef:
    name: nethical-ca
    kind: ClusterIssuer
  commonName: nethical-api.nethical.svc.cluster.local
  dnsNames:
  - nethical-api.nethical.svc.cluster.local
  - nethical-api
```

**Verification**:

```bash
# Test mTLS connection
openssl s_client -connect nethical-api.nethical.svc.cluster.local:8000 \
  -cert client.crt -key client.key -CAfile ca.crt
```

---

## Input Validation & Payload Limits

### Pydantic Schema Validation

**Objective**: Strict type and format validation for all API inputs.

**Implementation**:

```python
# nethical/api/models.py
from pydantic import BaseModel, Field, validator, constr
import unicodedata
import re

class AgentActionRequest(BaseModel):
    agent_id: constr(min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')
    action_type: constr(min_length=1, max_length=64)
    context: constr(max_length=4096)
    metadata: dict = Field(default_factory=dict, max_items=50)
    
    @validator('context')
    def normalize_unicode(cls, v):
        # NFC normalization to prevent homograph attacks
        return unicodedata.normalize('NFC', v)
    
    @validator('metadata')
    def validate_metadata(cls, v):
        # Prevent deeply nested objects
        max_depth = 3
        def check_depth(obj, depth=0):
            if depth > max_depth:
                raise ValueError(f"Metadata nesting exceeds {max_depth} levels")
            if isinstance(obj, dict):
                for value in obj.values():
                    check_depth(value, depth + 1)
        check_depth(v)
        return v

class PolicyCreateRequest(BaseModel):
    name: constr(min_length=1, max_length=255)
    description: constr(max_length=2000)
    rules: List[dict] = Field(..., min_items=1, max_items=100)
    priority: int = Field(ge=0, le=100)
    enabled: bool = True
    
    @validator('rules')
    def validate_rules(cls, v):
        for rule in v:
            if 'condition' not in rule or 'action' not in rule:
                raise ValueError("Each rule must have 'condition' and 'action'")
        return v
```

**Usage**:

```python
@app.post("/api/v1/agent/evaluate")
async def evaluate_action(request: AgentActionRequest):
    # Request is automatically validated by FastAPI + Pydantic
    result = await evaluator.evaluate(request)
    return result
```

---

### Payload Size Limits

**Objective**: Prevent resource exhaustion from oversized requests.

**Nginx Configuration**:

```nginx
# deploy/nginx/nginx.conf
http {
    client_max_body_size 1m;
    client_body_timeout 30s;
    client_header_timeout 30s;
    
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    server {
        listen 8080;
        
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            # Additional size limit for specific endpoints
            location /api/v1/policy {
                client_max_body_size 100k;
            }
            
            proxy_pass http://nethical-api:8000;
        }
    }
}
```

**FastAPI Middleware**:

```python
# nethical/api/middleware.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class PayloadSizeLimiter(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 1_048_576):  # 1MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Payload too large. Max size: {self.max_size} bytes"
            )
        return await call_next(request)

# Apply middleware
app.add_middleware(PayloadSizeLimiter, max_size=1_048_576)
```

---

## Secrets Management

### External Secrets Operator

**Objective**: Never store secrets in code, configuration, or environment variables directly.

**Implementation**:

```yaml
# deploy/kubernetes/external-secrets.yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: nethical
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "nethical-app"
          serviceAccountRef:
            name: nethical

---
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
    creationPolicy: Owner
  data:
    - secretKey: jwt-private-key
      remoteRef:
        key: nethical/jwt
        property: private_key
    - secretKey: jwt-public-key
      remoteRef:
        key: nethical/jwt
        property: public_key
    - secretKey: database-password
      remoteRef:
        key: nethical/db
        property: password
    - secretKey: redis-password
      remoteRef:
        key: nethical/redis
        property: password
```

**Deployment Reference**:

```yaml
# deploy/kubernetes/deployment.yaml
spec:
  containers:
  - name: nethical
    env:
    - name: JWT_PRIVATE_KEY
      valueFrom:
        secretKeyRef:
          name: nethical-app-secrets
          key: jwt-private-key
    - name: DATABASE_PASSWORD
      valueFrom:
        secretKeyRef:
          name: nethical-app-secrets
          key: database-password
```

---

### Automated Secret Rotation

**Objective**: Regular rotation to limit exposure window of compromised secrets.

**CronJob Configuration**:

```yaml
# deploy/kubernetes/secret-rotation.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: rotate-secrets
  namespace: nethical
spec:
  schedule: "0 0 1 * *"  # Monthly on 1st day
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 3
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: secret-rotator
          containers:
          - name: rotate
            image: ghcr.io/v1b3hr/nethical-secret-rotator:latest
            env:
            - name: VAULT_ADDR
              value: "https://vault.example.com"
            - name: ROTATION_TARGETS
              value: "jwt-keys,database-password,api-keys"
          restartPolicy: OnFailure
```

**Rotation Script**:

```bash
#!/bin/bash
# scripts/rotate-secrets.sh

set -euo pipefail

echo "Starting secret rotation..."

# Rotate JWT signing keys
echo "Rotating JWT keys..."
PRIVATE_KEY=$(openssl genrsa 4096 2>/dev/null)
PUBLIC_KEY=$(echo "$PRIVATE_KEY" | openssl rsa -pubout 2>/dev/null)

vault kv put nethical/jwt \
  private_key="$PRIVATE_KEY" \
  public_key="$PUBLIC_KEY" \
  rotated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Rotate database password
echo "Rotating database password..."
NEW_DB_PASSWORD=$(openssl rand -base64 32)

# Update password in Vault
vault kv put nethical/db \
  password="$NEW_DB_PASSWORD" \
  rotated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Update password in PostgreSQL
PGPASSWORD="$OLD_DB_PASSWORD" psql -h $DB_HOST -U nethical -c \
  "ALTER USER nethical WITH PASSWORD '$NEW_DB_PASSWORD';"

# Trigger rollout restart to pick up new secrets
kubectl rollout restart deployment/nethical -n nethical

echo "Secret rotation complete."
```

**Monitoring**:

```yaml
# Alert when secrets are approaching rotation deadline
- alert: SecretRotationOverdue
  expr: (time() - secret_last_rotated_timestamp_seconds) > (80 * 86400)
  labels:
    severity: warning
  annotations:
    summary: "Secret {{ $labels.secret_name }} rotation overdue ({{ $value | humanizeDuration }})"
    description: "Secret should be rotated every 90 days. Current age: {{ $value | humanizeDuration }}"
```

---

## Logging & PII Redaction

### PII Detection and Redaction

**Objective**: Protect user privacy in logs while maintaining debuggability.

**Implementation**:

```python
# nethical/logging/pii_redactor.py
import re
import logging
from typing import Dict, Pattern

class PIIRedactor:
    # Pattern definitions
    PATTERNS: Dict[str, Pattern] = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        'api_key': re.compile(r'\b[A-Za-z0-9]{32,}\b'),
    }
    
    def __init__(self, reversible: bool = False):
        self.reversible = reversible
        self.redaction_map: Dict[str, str] = {}
    
    def redact(self, text: str) -> tuple[str, Dict[str, str]]:
        """Redact PII from text. Returns (redacted_text, metadata)"""
        redacted = text
        detected_pii = []
        
        for pii_type, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                detected_pii.append(pii_type)
                for match in matches:
                    if self.reversible:
                        redaction_id = f"<REDACTED_{pii_type.upper()}_{len(self.redaction_map)}>"
                        self.redaction_map[redaction_id] = match
                        redacted = redacted.replace(match, redaction_id)
                    else:
                        redacted = redacted.replace(match, f"<REDACTED_{pii_type.upper()}>")
        
        metadata = {"detected_pii_types": detected_pii}
        return redacted, metadata

class PIIRedactingFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redactor = PIIRedactor(reversible=True)
    
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        redacted_msg, metadata = self.redactor.redact(msg)
        
        # Add PII detection metadata to log record
        if metadata['detected_pii_types']:
            redacted_msg += f" | pii_detected={','.join(metadata['detected_pii_types'])}"
        
        return redacted_msg
```

**Configuration**:

```python
# nethical/logging/config.py
import logging
import json
from pythonjsonlogger import jsonlogger

def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler with PII redaction
    console_handler = logging.StreamHandler()
    formatter = PIIRedactingFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # JSON handler for log aggregation
    json_handler = logging.StreamHandler()
    json_formatter = jsonlogger.JsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    json_handler.setFormatter(json_formatter)
    logger.addHandler(json_handler)
```

---

## WAF Rules & Edge Protection

### Prompt Injection Detection

**Objective**: Block malicious prompts attempting to bypass safety controls.

**ModSecurity Rules**:

```apache
# deploy/waf/prompt-injection.conf

# Rule 1: Ignore/disregard instructions
SecRule REQUEST_BODY "@rx (ignore|disregard|forget|bypass).*(previous|above|prior|earlier).*(instruction|prompt|rule|directive)" \
    "id:1001,\
    phase:2,\
    deny,\
    status:403,\
    log,\
    msg:'Prompt injection attempt detected',\
    tag:'OWASP_CRS/PROMPT_INJECTION',\
    severity:'CRITICAL'"

# Rule 2: Role manipulation
SecRule REQUEST_BODY "@rx (you are|act as|pretend to be|roleplay as).*(admin|root|system|developer)" \
    "id:1002,\
    phase:2,\
    deny,\
    status:403,\
    log,\
    msg:'Role manipulation attempt detected',\
    severity:'HIGH'"

# Rule 3: Instruction leakage
SecRule REQUEST_BODY "@rx (show|reveal|display|print).*(system prompt|instructions|rules)" \
    "id:1003,\
    phase:2,\
    deny,\
    status:403,\
    log,\
    msg:'Instruction leakage attempt detected',\
    severity:'HIGH'"

# Rule 4: Jailbreak keywords
SecRule REQUEST_BODY "@rx (DAN|jailbreak|unrestricted mode|developer mode|god mode)" \
    "id:1004,\
    phase:2,\
    deny,\
    status:403,\
    log,\
    msg:'Jailbreak attempt detected',\
    severity:'CRITICAL'"
```

**Cloud WAF Configuration** (AWS WAF):

```json
{
  "Name": "NethicalPromptInjectionRule",
  "Priority": 1,
  "Statement": {
    "OrStatement": {
      "Statements": [
        {
          "ByteMatchStatement": {
            "SearchString": "ignore previous",
            "FieldToMatch": {"Body": {}},
            "TextTransformations": [{"Priority": 0, "Type": "LOWERCASE"}],
            "PositionalConstraint": "CONTAINS"
          }
        },
        {
          "RegexMatchStatement": {
            "RegexString": "(forget|disregard).*(instruction|rule)",
            "FieldToMatch": {"Body": {}},
            "TextTransformations": [{"Priority": 0, "Type": "LOWERCASE"}]
          }
        }
      ]
    }
  },
  "Action": {"Block": {}},
  "VisibilityConfig": {
    "SampledRequestsEnabled": true,
    "CloudWatchMetricsEnabled": true,
    "MetricName": "PromptInjectionBlocked"
  }
}
```

---

### Rate Limiting & DDoS Protection

**Objective**: Prevent resource exhaustion and ensure fair usage.

**Nginx Rate Limiting**:

```nginx
# deploy/nginx/rate-limit.conf
http {
    # Define rate limit zones
    limit_req_zone $binary_remote_addr zone=global:10m rate=100r/s;
    limit_req_zone $http_x_api_key zone=api_key:10m rate=500r/s;
    limit_req_zone $request_uri zone=uri:10m rate=10r/s;
    
    # Connection limits
    limit_conn_zone $binary_remote_addr zone=addr:10m;
    
    server {
        # Global rate limit
        limit_req zone=global burst=200 nodelay;
        limit_conn addr 50;
        
        location /api/v1/agent/evaluate {
            # Higher limit for evaluation endpoint
            limit_req zone=api_key burst=1000 nodelay;
            limit_req_status 429;
            
            proxy_pass http://nethical-api:8000;
        }
        
        location /api/v1/policy {
            # Stricter limit for policy modifications
            limit_req zone=uri burst=5;
            limit_req_status 429;
            
            proxy_pass http://nethical-api:8000;
        }
    }
}
```

**Application-Level Rate Limiting**:

```python
# nethical/api/rate_limiter.py
from fastapi import HTTPException, Request
from redis import Redis
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """Token bucket algorithm"""
        now = datetime.now()
        window_start = (now - timedelta(seconds=window_seconds)).timestamp()
        
        # Remove old requests outside the window
        self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count requests in current window
        current_requests = self.redis.zcard(key)
        
        if current_requests >= max_requests:
            return False
        
        # Add current request
        self.redis.zadd(key, {str(now.timestamp()): now.timestamp()})
        self.redis.expire(key, window_seconds)
        
        return True

# Middleware
async def rate_limit_middleware(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    rate_limiter = request.app.state.rate_limiter
    key = f"rate_limit:{api_key}"
    
    if not rate_limiter.check_rate_limit(key, max_requests=1000, window_seconds=60):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"}
        )
    
    return await call_next(request)
```

---

## Dependency & Image Scanning

### Continuous Vulnerability Scanning

**Objective**: Detect and remediate vulnerabilities in dependencies and container images.

**GitHub Actions Workflow**:

**Note**: See actual implementation in `.github/workflows/security.yml` for current security scanning process.

```yaml
# Example: .github/workflows/security-scan.yml
name: Security Scanning
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  push:
    branches: [main]
  pull_request:

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Trivy dependency scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'
      
      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'
      
      - name: Fail on Critical vulnerabilities
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          exit-code: '1'
          severity: 'CRITICAL'
  
  image-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build image
        run: docker build -t nethical:test .
      
      - name: Run Trivy image scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'nethical:test'
          format: 'sarif'
          output: 'trivy-image-results.sarif'
      
      - name: Run Grype scan
        uses: anchore/scan-action@v3
        with:
          image: 'nethical:test'
          fail-build: true
          severity-cutoff: high
```

**SLA Enforcement**: See [Vulnerability Response](#vulnerability-response) section.

---

## Vulnerability Response

### Response SLA

**Objective**: Time-bound remediation based on vulnerability severity.

| Severity | Response Time | Remediation Time | Actions |
|----------|---------------|------------------|---------|
| **Critical** | <4 hours | <24 hours | Immediate patch, emergency release, incident declared |
| **High** | <24 hours | <72 hours | Priority patch, scheduled release, stakeholder notification |
| **Medium** | <72 hours | <30 days | Planned patch, regular release cycle |
| **Low** | <7 days | <90 days | Backlog item, routine maintenance |

### Automated SLA Monitoring

**Script** (Reference Implementation):

**Note**: This is a reference implementation. See `.github/workflows/vuln-sla.yml` for actual SLA enforcement workflow.

```python
# Example: scripts/check-vuln-sla.py
#!/usr/bin/env python3
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

SLA_HOURS = {
    'CRITICAL': 24,
    'HIGH': 72,
    'MEDIUM': 720,  # 30 days
    'LOW': 2160,    # 90 days
}

def check_sla(vuln_file: Path) -> bool:
    with open(vuln_file) as f:
        data = json.load(f)
    
    now = datetime.now()
    breaches = []
    
    for result in data.get('Results', []):
        for vuln in result.get('Vulnerabilities', []):
            vuln_id = vuln['VulnerabilityID']
            severity = vuln['Severity']
            
            # Parse published date
            published = datetime.fromisoformat(
                vuln['PublishedDate'].replace('Z', '+00:00')
            )
            
            age_hours = (now - published).total_seconds() / 3600
            sla_hours = SLA_HOURS.get(severity, float('inf'))
            
            if age_hours > sla_hours:
                breaches.append({
                    'id': vuln_id,
                    'severity': severity,
                    'age_hours': age_hours,
                    'sla_hours': sla_hours,
                    'package': vuln.get('PkgName'),
                })
    
    if breaches:
        print(f"❌ SLA BREACH: {len(breaches)} vulnerabilities exceed SLA")
        for breach in breaches:
            print(f"  {breach['id']} ({breach['severity']}): "
                  f"{breach['age_hours']:.1f}h old, SLA {breach['sla_hours']}h "
                  f"[{breach['package']}]")
        return False
    
    print("✅ All vulnerabilities within SLA")
    return True

if __name__ == '__main__':
    sys.exit(0 if check_sla(Path(sys.argv[1])) else 1)
```

**GitHub Actions Integration**:

**Note**: See actual implementation in `.github/workflows/vuln-sla.yml` for current SLA enforcement process.

```yaml
# Example: .github/workflows/vuln-sla-enforcement.yml
name: Vulnerability SLA Enforcement
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

jobs:
  check-sla:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Scan for vulnerabilities
        run: |
          trivy image ghcr.io/v1b3hr/nethical:latest \
            --format json --output vulns.json
      
      - name: Check SLA compliance
        run: python scripts/check-vuln-sla.py vulns.json
      
      - name: Create incident on breach
        if: failure()
        run: |
          gh issue create \
            --title "🚨 Vulnerability SLA Breach" \
            --body "Critical or High vulnerabilities exceed remediation SLA. See workflow run for details." \
            --label "security,priority:critical" \
            --assignee "@security-team"
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Plugin Trust & Verification

### Signature Verification

**Objective**: Ensure plugins are authentic and unmodified.

**Implementation**:

```python
# nethical/marketplace/plugin_verifier.py
import subprocess
from pathlib import Path
from typing import Optional

class PluginVerifier:
    def __init__(self, trusted_keys_dir: Path):
        self.trusted_keys_dir = trusted_keys_dir
    
    def verify_signature(self, plugin_path: Path) -> bool:
        """Verify plugin signature using Cosign"""
        sig_path = plugin_path.with_suffix(plugin_path.suffix + '.sig')
        
        if not sig_path.exists():
            print(f"❌ No signature found for {plugin_path}")
            return False
        
        try:
            # Verify with cosign
            result = subprocess.run([
                'cosign', 'verify-blob',
                '--key', str(self.trusted_keys_dir / 'cosign.pub'),
                '--signature', str(sig_path),
                str(plugin_path)
            ], capture_output=True, text=True, check=True)
            
            print(f"✅ Signature verified for {plugin_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Signature verification failed: {e.stderr}")
            return False
    
    def scan_vulnerabilities(self, plugin_path: Path) -> bool:
        """Scan plugin for vulnerabilities"""
        try:
            result = subprocess.run([
                'trivy', 'fs', '--severity', 'CRITICAL,HIGH',
                '--exit-code', '1', str(plugin_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ No critical vulnerabilities in {plugin_path}")
                return True
            else:
                print(f"❌ Vulnerabilities found in {plugin_path}")
                return False
        except subprocess.CalledProcessError:
            return False
    
    def check_reputation(self, plugin_metadata: dict) -> float:
        """Calculate plugin reputation score (0-1)"""
        score = 0.0
        
        # Author reputation (40%)
        author_score = plugin_metadata.get('author_reputation', 0) / 100
        score += author_score * 0.4
        
        # Download count (20%)
        downloads = plugin_metadata.get('downloads', 0)
        download_score = min(downloads / 10000, 1.0)
        score += download_score * 0.2
        
        # Rating (20%)
        rating = plugin_metadata.get('rating', 0) / 5.0
        score += rating * 0.2
        
        # Age/maturity (10%)
        days_old = plugin_metadata.get('days_since_creation', 0)
        age_score = min(days_old / 365, 1.0)
        score += age_score * 0.1
        
        # Security history (10%)
        vuln_count = plugin_metadata.get('vulnerability_count', 0)
        security_score = max(1.0 - (vuln_count * 0.2), 0)
        score += security_score * 0.1
        
        return score
```

### Trust Score Gating

**Objective**: Only allow trusted plugins in production environments.

**Configuration**:

```yaml
# config/plugin-trust.yaml
plugin_trust:
  minimum_score: 0.8  # 80/100
  require_signature: true
  require_vulnerability_scan: true
  
  # Trusted plugin publishers (bypass trust score)
  trusted_publishers:
    - "nethical-official"
    - "verified-partner-1"
  
  # Plugin permissions
  sandbox_enabled: true
  allowed_capabilities:
    - "network.http.client"
    - "filesystem.read"
  
  denied_capabilities:
    - "filesystem.write"
    - "process.execute"
    - "network.socket.raw"
```

---

## Continuous Security Verification

### Automated CI Gates

**Objective**: Prevent insecure code from reaching production.

**GitHub Actions Workflow**:

```yaml
# .github/workflows/security-gates.yml
name: Security Gates
on:
  pull_request:
  push:
    branches: [main]

jobs:
  security-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      # Gate 1: SAST
      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: p/security-audit p/owasp-top-ten
      
      # Gate 2: Secret scanning
      - name: Run TruffleHog
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD
      
      # Gate 3: Dependency check
      - name: Dependency audit
        run: |
          pip-audit --desc --fix
      
      # Gate 4: SBOM generation
      - name: Generate SBOM
        run: |
          syft dir:. -o spdx-json=sbom.json
      
      # Gate 5: Container scan
      - name: Build and scan image
        run: |
          docker build -t nethical:pr-${{ github.event.pull_request.number }} .
          trivy image --exit-code 1 --severity CRITICAL \
            nethical:pr-${{ github.event.pull_request.number }}
      
      # Gate 6: License compliance
      - name: Check licenses
        uses: fossas/fossa-action@main
        with:
          api-key: ${{ secrets.FOSSA_API_KEY }}
```

### Pre-commit Hooks

**Configuration**:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/trufflesecurity/trufflehog
    rev: v3.63.0
    hooks:
      - id: trufflehog
        name: TruffleHog
        entry: trufflehog filesystem --no-update
        language: system
  
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
  
  - repo: https://github.com/returntocorp/semgrep
    rev: v1.45.0
    hooks:
      - id: semgrep
        args: ['--config', 'p/security-audit', '--error']
```

---

## Document Relationships

This security hardening guide integrates with:

- **[Validation Plan](../VALIDATION_PLAN.md)**: Security test suites (SAST, DAST, dependency audit) reference validation metrics and cadence
- **[Production Readiness Checklist](./PRODUCTION_READINESS_CHECKLIST.md)**: Security checklist items detail-mapped to controls in this guide
- **[Ethics Validation Framework](./ETHICS_VALIDATION_FRAMEWORK.md)**: Audit integrity and plugin trust controls support ethics governance
- **[Benchmark Plan](./BENCHMARK_PLAN.md)**: Security overhead measured in performance benchmarks

---

**Version**: 1.0  
**Last Updated**: 2025-11-24  
**Status**: Comprehensive controls aligned with world-class security standards  
**Compliance**: NIST Cybersecurity Framework, OWASP Top 10, CIS Benchmarks, SLSA Level 3
