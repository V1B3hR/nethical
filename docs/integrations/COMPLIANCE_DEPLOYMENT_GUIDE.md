# Compliance Deployment Guide

This guide provides comprehensive documentation for deploying Nethical with sector-specific compliance configurations.

## Table of Contents

- [Compliance Framework Overview](#compliance-framework-overview)
- [Auditor Usage Notes](#auditor-usage-notes)
- [Client/Operator Usage Notes](#clientoperator-usage-notes)
- [Compliance Validation](#compliance-validation)
- [Audit Trail and Evidence](#audit-trail-and-evidence)

## Compliance Framework Overview

Nethical supports the following compliance frameworks through Kustomize overlays:

| Framework | Sector | Overlay Path | Key Controls |
|-----------|--------|--------------|--------------|
| HIPAA | Healthcare (US) | `overlays/compliance/hipaa` | PHI encryption, audit logging, access controls |
| NHS DSPT | Healthcare (UK) | `overlays/compliance/nhs-dspt` | UK GDPR, data residency, MFA |
| EU MDR | Medical Devices (EU) | `overlays/compliance/eu-mdr` | Algorithm transparency, vigilance reporting |
| FIPS 140-3 | Government/Defense | `overlays/compliance/fips` | Cryptographic module validation |
| NIST 800-53 | Government (US) | `overlays/compliance/nist` | AC, AU, SC, SI control families |
| FERPA | Education (US) | `overlays/compliance/ferpa` | Student data protection, consent |

## Auditor Usage Notes

### Pre-Audit Preparation

Before conducting a compliance audit, verify the deployment configuration:

```bash
# Check active compliance overlay
kubectl get pods -n nethical -o jsonpath='{.items[0].metadata.labels.compliance\.nethical\.io/framework}'

# Verify compliance annotations
kubectl get statefulset nethical -n nethical -o yaml | grep -A 10 "compliance.nethical.io"

# List all compliance-related resources
kubectl get all,configmaps,secrets,networkpolicies -n nethical \
  -l compliance.nethical.io/framework
```

### Evidence Collection

#### 1. Configuration Evidence

```bash
# Export deployment configuration
kubectl get statefulset nethical -n nethical -o yaml > evidence/statefulset-config.yaml

# Export ConfigMaps
kubectl get configmap -n nethical -o yaml > evidence/configmaps.yaml

# Export NetworkPolicies
kubectl get networkpolicy -n nethical -o yaml > evidence/networkpolicies.yaml

# Export RBAC configuration
kubectl get roles,rolebindings,clusterroles,clusterrolebindings \
  -n nethical -o yaml > evidence/rbac.yaml
```

#### 2. Security Evidence

```bash
# Verify pod security context
kubectl get pods -n nethical -o jsonpath='{.items[*].spec.securityContext}' | jq .

# Check container security context
kubectl get pods -n nethical -o jsonpath='{.items[*].spec.containers[*].securityContext}' | jq .

# Verify TLS configuration
kubectl exec -n nethical nethical-0 -- env | grep -i tls
kubectl exec -n nethical nethical-0 -- env | grep -i encrypt
```

#### 3. Audit Log Evidence

```bash
# Get audit log configuration
kubectl get configmap -n nethical nethical-*-audit-config -o yaml

# Export recent audit events (if using OpenTelemetry)
kubectl logs -n nethical -l app=otel-collector --tail=1000 > evidence/audit-logs.txt

# Verify Merkle anchoring
kubectl exec -n nethical nethical-0 -- python -c \
  "from nethical.core import MerkleAnchor; ma = MerkleAnchor(); print(ma.verify_chain())"
```

### Compliance Verification Matrix

| Control | HIPAA | NHS DSPT | FIPS | NIST | FERPA | EU MDR |
|---------|-------|----------|------|------|-------|--------|
| Encryption at Rest | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Encryption in Transit | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Audit Logging | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Access Control | ✓ | ✓ | - | ✓ | ✓ | - |
| MFA | ✓ | ✓ | - | ✓ | - | - |
| Session Timeout | ✓ | ✓ | - | ✓ | - | - |
| Data Residency | - | ✓ | - | - | - | ✓ |
| Algorithm Transparency | - | - | - | - | - | ✓ |
| FIPS Crypto | - | - | ✓ | - | - | - |

### Audit Retention Periods

| Framework | Retention Period | Configuration Key |
|-----------|------------------|-------------------|
| HIPAA | 6 years | `NETHICAL_AUDIT_RETENTION_DAYS=2190` |
| NHS DSPT | 7 years | `NETHICAL_AUDIT_RETENTION_DAYS=2555` |
| NIST 800-53 | 7 years | `NETHICAL_AUDIT_RETENTION_YEARS=7` |
| EU MDR | 10 years | `NETHICAL_AUDIT_RETENTION_YEARS=10` |
| FERPA | Permanent | `NETHICAL_AUDIT_RETENTION=permanent` |

## Client/Operator Usage Notes

### Deployment Checklist

#### Pre-Deployment

- [ ] Select appropriate compliance overlay
- [ ] Review and customize ConfigMap values
- [ ] Configure external secrets integration
- [ ] Set up monitoring and alerting
- [ ] Prepare backup strategy

#### Deployment

```bash
# 1. Create namespace with compliance labels
kubectl apply -f deploy/kubernetes/namespace.yaml

# 2. Apply compliance overlay
kubectl apply -k deploy/kubernetes/overlays/compliance/<framework>/

# 3. Verify deployment
kubectl get pods -n nethical -w

# 4. Run health check
kubectl exec -n nethical nethical-0 -- python -c "import nethical; print('healthy')"
```

#### Post-Deployment

- [ ] Verify all pods are running
- [ ] Test API endpoints
- [ ] Confirm audit logging
- [ ] Validate encryption configuration
- [ ] Run compliance validation tests

### Day-2 Operations

#### Scaling

```bash
# Scale up
kubectl scale statefulset nethical -n nethical --replicas=5

# Verify HPA
kubectl get hpa -n nethical
```

#### Updates

```bash
# Rolling update
helm upgrade nethical deploy/helm/nethical \
  -f values-production.yaml \
  --set image.tag=<new-version>

# Monitor rollout
kubectl rollout status statefulset nethical -n nethical
```

#### Secret Rotation

```bash
# Trigger manual rotation
kubectl create job --from=cronjob/rotate-secrets rotation-manual -n nethical

# Monitor rotation
kubectl logs -n nethical -l job-name=rotation-manual
```

### Troubleshooting

#### Pod Not Starting

```bash
# Check events
kubectl get events -n nethical --sort-by='.lastTimestamp'

# Check pod description
kubectl describe pod nethical-0 -n nethical

# Check SCC (OpenShift)
oc get pod nethical-0 -n nethical -o yaml | grep scc
```

#### Compliance Validation Failures

```bash
# Check compliance configuration
kubectl get configmap nethical-*-config -n nethical -o yaml

# Verify environment variables
kubectl exec -n nethical nethical-0 -- env | grep NETHICAL

# Check compliance labels
kubectl get pods -n nethical -o jsonpath='{.items[*].metadata.labels}' | jq .
```

## Compliance Validation

### Automated Validation Script

```bash
#!/bin/bash
# compliance-validation.sh

FRAMEWORK=$1
NAMESPACE=${2:-nethical}

echo "Validating $FRAMEWORK compliance in namespace $NAMESPACE"

# Check framework label
ACTUAL_FRAMEWORK=$(kubectl get pods -n $NAMESPACE \
  -o jsonpath='{.items[0].metadata.labels.compliance\.nethical\.io/framework}')

if [ "$ACTUAL_FRAMEWORK" != "$FRAMEWORK" ]; then
  echo "ERROR: Expected framework $FRAMEWORK but found $ACTUAL_FRAMEWORK"
  exit 1
fi

# Check encryption configuration
ENCRYPTION=$(kubectl exec -n $NAMESPACE nethical-0 -- \
  env | grep NETHICAL_ENCRYPTION_AT_REST | cut -d= -f2)

if [ "$ENCRYPTION" != "true" ]; then
  echo "ERROR: Encryption at rest not enabled"
  exit 1
fi

# Check audit logging
AUDIT=$(kubectl exec -n $NAMESPACE nethical-0 -- \
  env | grep NETHICAL_AUDIT_ENABLED | cut -d= -f2)

if [ "$AUDIT" != "true" ]; then
  echo "ERROR: Audit logging not enabled"
  exit 1
fi

echo "Compliance validation passed for $FRAMEWORK"
```

### Framework-Specific Validations

> **Note**: The commands below use `nethical-0` as the pod name which works for StatefulSet deployments.
> For other deployment types, use `kubectl get pods -n nethical -l app.kubernetes.io/name=nethical -o name | head -1`
> to get the first pod name, or use `kubectl exec -n nethical deploy/nethical -- ...` for Deployments.

#### Helper: Get First Pod Name

```bash
# Set POD variable for reuse
POD=$(kubectl get pods -n nethical -l app.kubernetes.io/name=nethical -o name | head -1)
# Example: pod/nethical-0
```

#### HIPAA Validation

```bash
# Verify PHI encryption
kubectl exec -n nethical $POD -- env | grep -E "PHI|ENCRYPTION"

# Check session timeout (max 15 minutes)
kubectl exec -n nethical $POD -- env | grep SESSION_TIMEOUT

# Verify audit retention (min 6 years)
kubectl exec -n nethical $POD -- env | grep AUDIT_RETENTION
```

#### FIPS Validation

```bash
# Verify FIPS mode
kubectl exec -n nethical $POD -- env | grep FIPS

# Check crypto algorithms
kubectl exec -n nethical $POD -- python -c \
  "import ssl; print('FIPS:', ssl.FIPS_mode())"

# Verify TLS configuration
kubectl exec -n nethical $POD -- env | grep TLS_MIN_VERSION
```

#### NIST 800-53 Validation

```bash
# Verify impact level
kubectl exec -n nethical $POD -- env | grep IMPACT_LEVEL

# Check MFA configuration
kubectl exec -n nethical $POD -- env | grep MFA

# Verify lockout settings
kubectl exec -n nethical $POD -- env | grep LOCKOUT
```

## Audit Trail and Evidence

### Audit Event Types

| Event Type | Description | Retention |
|------------|-------------|-----------|
| `authentication` | Login/logout events | Framework-specific |
| `authorization` | Access granted/denied | Framework-specific |
| `data_access` | Read/write operations | Framework-specific |
| `configuration` | Setting changes | 1 year minimum |
| `security` | Security incidents | 7 years minimum |

### Generating Audit Reports

```bash
# Generate compliance report
kubectl exec -n nethical nethical-0 -- python -c "
from nethical.security.audit_logging import AuditLogger
logger = AuditLogger()
report = logger.generate_compliance_report(
    start_date='2024-01-01',
    end_date='2024-12-31',
    framework='hipaa'
)
print(report)
"

# Export to file
kubectl exec -n nethical nethical-0 -- python -c "
from nethical.security.audit_logging import AuditLogger
logger = AuditLogger()
logger.export_report('/data/audit-report-2024.json', format='json')
" && kubectl cp nethical/nethical-0:/data/audit-report-2024.json ./audit-report-2024.json
```

### Evidence Preservation

For compliance audits, preserve the following evidence:

1. **Configuration Evidence**
   - Kubernetes manifests
   - Helm values files
   - ConfigMaps and Secrets (redacted)
   - Network policies

2. **Security Evidence**
   - Pod security contexts
   - RBAC configurations
   - TLS certificates (metadata only)
   - Encryption settings

3. **Operational Evidence**
   - Audit logs
   - Access logs
   - Change history
   - Incident reports

4. **Validation Evidence**
   - Compliance scan results
   - Vulnerability scan results
   - Penetration test reports
   - Security assessment reports

## Support

For compliance-related questions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical/docs/compliance
