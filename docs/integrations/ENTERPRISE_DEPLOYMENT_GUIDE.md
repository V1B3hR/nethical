# Enterprise Kubernetes Deployment Guide

This guide covers enterprise-grade deployment of Nethical in managed Kubernetes environments (GKE, EKS, AKS), with compliance configurations and operational best practices.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Cloud Provider Setup](#cloud-provider-setup)
- [Compliance Overlays](#compliance-overlays)
- [Deployment Strategies](#deployment-strategies)
- [RBAC Configuration](#rbac-configuration)
- [Secrets Management](#secrets-management)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Backup and Disaster Recovery](#backup-and-disaster-recovery)
- [Operational Runbooks](#operational-runbooks)

## Prerequisites

### Required Components

| Component | Version | Purpose |
|-----------|---------|---------|
| Kubernetes | 1.25+ | Container orchestration |
| Helm | 3.10+ | Package management |
| kubectl | 1.25+ | Cluster interaction |
| Kustomize | 4.0+ | Configuration management |

### Recommended Add-ons

| Add-on | Purpose |
|--------|---------|
| cert-manager | TLS certificate management |
| external-secrets-operator | External secrets integration |
| prometheus-operator | Monitoring stack |
| ingress-nginx | Ingress controller |
| velero | Backup and restore |

## Cloud Provider Setup

### Google Kubernetes Engine (GKE)

```bash
# Create GKE cluster with Workload Identity
gcloud container clusters create nethical-prod \
  --region=us-central1 \
  --num-nodes=3 \
  --machine-type=n2-standard-4 \
  --enable-workload-identity \
  --workload-pool=PROJECT_ID.svc.id.goog \
  --enable-network-policy \
  --enable-shielded-nodes \
  --enable-dataplane-v2

# Deploy with GKE overlay
kubectl apply -k deploy/kubernetes/overlays/cloud-providers/gke/
```

### Amazon Elastic Kubernetes Service (EKS)

```bash
# Create EKS cluster with IRSA
eksctl create cluster \
  --name nethical-prod \
  --region us-east-1 \
  --nodegroup-name standard \
  --node-type m5.xlarge \
  --nodes 3 \
  --with-oidc \
  --managed

# Deploy with EKS overlay
kubectl apply -k deploy/kubernetes/overlays/cloud-providers/eks/
```

### Azure Kubernetes Service (AKS)

```bash
# Create AKS cluster with Workload Identity
az aks create \
  --resource-group nethical-rg \
  --name nethical-prod \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-workload-identity \
  --enable-oidc-issuer \
  --network-policy azure

# Deploy with AKS overlay
kubectl apply -k deploy/kubernetes/overlays/cloud-providers/aks/
```

## Compliance Overlays

Nethical provides Kustomize overlays for various compliance frameworks:

### HIPAA (Healthcare - US)

```bash
# Deploy with HIPAA compliance
kubectl apply -k deploy/kubernetes/overlays/compliance/hipaa/

# Verify HIPAA labels
kubectl get pods -n nethical -l compliance.nethical.io/framework=hipaa
```

**Key HIPAA Controls:**
- PHI encryption at rest (AES-256-GCM)
- 6-year audit log retention
- 15-minute session timeout
- Emergency access procedures
- Merkle-anchored audit trail

### NHS DSPT (Healthcare - UK)

```bash
# Deploy with NHS DSPT compliance
kubectl apply -k deploy/kubernetes/overlays/compliance/nhs-dspt/
```

**Key NHS DSPT Controls:**
- UK GDPR compliance
- 7-year audit retention
- Multi-factor authentication
- Data residency enforcement
- Quarterly access reviews

### FIPS 140-3 (Government/Defense)

```bash
# Deploy with FIPS compliance
kubectl apply -k deploy/kubernetes/overlays/compliance/fips/
```

**Key FIPS Controls:**
- FIPS 140-3 validated cryptographic modules
- Approved algorithms only (AES, SHA-2, ECDSA)
- Hardware RNG for key generation
- Daily cryptographic self-tests

### NIST 800-53 (Government - US)

```bash
# Deploy with NIST compliance
kubectl apply -k deploy/kubernetes/overlays/compliance/nist/
```

**Key NIST Controls:**
- AC (Access Control): MFA, session timeout, lockout
- AU (Audit): 7-year retention, tamper-proof logs
- SC (System Protection): TLS 1.3, encryption at rest
- SI (System Integrity): Vulnerability scanning

### FERPA (Education - US)

```bash
# Deploy with FERPA compliance
kubectl apply -k deploy/kubernetes/overlays/compliance/ferpa/
```

**Key FERPA Controls:**
- Student data protection
- Consent-based access
- Disclosure logging
- Directory information controls

### EU MDR (Medical Devices - EU)

```bash
# Deploy with EU MDR compliance
kubectl apply -k deploy/kubernetes/overlays/compliance/eu-mdr/
```

**Key EU MDR Controls:**
- Algorithm transparency
- 10-year audit retention
- Vigilance reporting
- Post-market surveillance

### Combining Overlays

Create a custom overlay for multiple compliance requirements:

```yaml
# overlays/combined-hipaa-gke/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../compliance/hipaa
  - ../cloud-providers/gke
```

```bash
kubectl apply -k overlays/combined-hipaa-gke/
```

## Deployment Strategies

### Rolling Update (Default)

Standard rolling update with zero downtime:

```bash
helm upgrade nethical deploy/helm/nethical \
  -f values-production.yaml \
  --set image.tag=0.2.0
```

### Canary Deployment

Gradual rollout with traffic splitting:

```bash
# Deploy canary with 10% traffic
helm install nethical-canary deploy/helm/nethical \
  -f values-canary.yaml \
  --set canary.weight=10

# Monitor canary
kubectl get pods -n nethical -l deployment-type=canary

# Increase canary traffic
helm upgrade nethical-canary deploy/helm/nethical \
  -f values-canary.yaml \
  --set canary.weight=50

# Promote canary to stable
helm upgrade nethical deploy/helm/nethical \
  -f values-production.yaml \
  --set image.tag=0.2.0

# Remove canary
helm uninstall nethical-canary
```

### Blue-Green Deployment

Zero-downtime deployment with instant rollback:

```bash
# Deploy both blue and green
helm install nethical-bg deploy/helm/nethical \
  -f values-blue-green.yaml

# Switch traffic to green
helm upgrade nethical-bg deploy/helm/nethical \
  -f values-blue-green.yaml \
  --set activeSlot=green \
  --set service.selector.deployment-slot=green

# Rollback to blue if needed
helm upgrade nethical-bg deploy/helm/nethical \
  -f values-blue-green.yaml \
  --set activeSlot=blue \
  --set service.selector.deployment-slot=blue
```

## RBAC Configuration

Nethical provides pre-defined roles for enterprise access control:

| Role | Description | Permissions |
|------|-------------|-------------|
| `nethical-admin` | Full access | Create, update, delete all resources |
| `nethical-operator` | Operations | View, restart, scale, logs |
| `nethical-auditor` | Compliance audit | Read-only access to all resources |
| `nethical-developer` | Development | View, logs, port-forward |
| `nethical-viewer` | Minimal access | View pods and services only |

### Apply RBAC Configuration

```bash
# Apply RBAC roles and bindings
kubectl apply -f deploy/kubernetes/rbac.yaml

# Verify roles
kubectl get roles,rolebindings -n nethical

# Test role permissions
kubectl auth can-i get pods -n nethical --as-group=nethical-operators
```

### Integrate with Identity Provider

```yaml
# Example: Azure AD integration
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: nethical-auditor-binding
  namespace: nethical
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: nethical-auditor
subjects:
  - kind: Group
    name: "AZURE_AD_GROUP_ID"  # Azure AD group object ID
    apiGroup: rbac.authorization.k8s.io
```

## Secrets Management

### External Secrets Operator

Deploy with external secrets integration:

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets

# Apply secret configuration
kubectl apply -f deploy/kubernetes/external-secrets.yaml
```

### Secret Rotation

Enable automatic secret rotation:

```bash
# Apply secret rotation CronJob
kubectl apply -f deploy/kubernetes/secret-rotation-cronjob.yaml

# Check rotation status
kubectl get cronjobs -n nethical
kubectl logs -n nethical -l job=secret-rotation
```

## Monitoring and Alerting

### Prometheus Metrics

Nethical exposes metrics on port 8888:

| Metric | Description |
|--------|-------------|
| `nethical_requests_total` | Total API requests |
| `nethical_violations_total` | Detected violations |
| `nethical_latency_seconds` | Request latency histogram |
| `nethical_quota_usage_ratio` | Quota utilization |
| `nethical_pii_detections_total` | PII detection events |

### ServiceMonitor Configuration

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: nethical
  namespace: nethical
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: nethical
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
```

### Alert Rules

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: nethical-alerts
  namespace: nethical
spec:
  groups:
    - name: nethical
      rules:
        - alert: NethicalHighErrorRate
          expr: rate(nethical_errors_total[5m]) > 0.05
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "High error rate detected"
            
        - alert: NethicalViolationSpike
          expr: rate(nethical_violations_total[5m]) > 10
          for: 2m
          labels:
            severity: warning
          annotations:
            summary: "Unusual violation rate detected"
```

## Backup and Disaster Recovery

### Velero Backup

```bash
# Install Velero
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.8.0 \
  --bucket nethical-backups \
  --backup-location-config region=us-east-1

# Create backup schedule
velero schedule create nethical-daily \
  --schedule="0 2 * * *" \
  --include-namespaces nethical \
  --ttl 720h
```

### Disaster Recovery Procedure

1. **Assess Impact**
   ```bash
   kubectl get all -n nethical
   kubectl describe pods -n nethical
   ```

2. **Restore from Backup**
   ```bash
   velero restore create --from-backup nethical-daily-TIMESTAMP
   velero restore describe nethical-daily-TIMESTAMP
   ```

3. **Verify Recovery**
   ```bash
   kubectl get pods -n nethical
   curl -k https://nethical.example.com/health
   ```

## Operational Runbooks

### Scale Up Procedure

```bash
# Scale StatefulSet
kubectl scale statefulset nethical -n nethical --replicas=10

# Or adjust HPA
kubectl patch hpa nethical -n nethical \
  -p '{"spec":{"maxReplicas":50}}'
```

### Emergency Rollback

```bash
# Helm rollback
helm rollback nethical 1

# Or kubectl rollback
kubectl rollout undo statefulset nethical -n nethical
```

### Log Investigation

```bash
# Recent logs
kubectl logs -n nethical -l app.kubernetes.io/name=nethical --tail=100

# Specific time range
kubectl logs -n nethical nethical-0 --since=1h

# All replicas
kubectl logs -n nethical -l app.kubernetes.io/name=nethical --all-containers
```

### Health Check

```bash
# Check endpoints
kubectl get endpoints nethical -n nethical

# Check pod health
kubectl exec -n nethical nethical-0 -- python -c "import nethical; print('healthy')"

# Check metrics
kubectl port-forward -n nethical svc/nethical 8888:8888
curl http://localhost:8888/metrics
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical/docs
