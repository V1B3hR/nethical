# Nethical Deployment Guide

This guide covers deploying Nethical in production using Kubernetes and Helm.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Quick Start](#quick-start)
- [Environment-Specific Deployments](#environment-specific-deployments)
- [Production Best Practices](#production-best-practices)
- [Monitoring and Observability](#monitoring-and-observability)
- [Backup and Disaster Recovery](#backup-and-disaster-recovery)
- [Troubleshooting](#troubleshooting)

## Related Documentation

For enterprise and regulated sector deployments, see:

- **[Enterprise Kubernetes Deployment Guide](../docs/kubernetes/ENTERPRISE_DEPLOYMENT_GUIDE.md)**: Cloud provider configurations (GKE/EKS/AKS), RBAC, secrets management
- **[Compliance Deployment Guide](../docs/kubernetes/COMPLIANCE_DEPLOYMENT_GUIDE.md)**: HIPAA, NHS DSPT, FIPS, NIST 800-53, FERPA, EU MDR overlays
- **[OpenShift Migration Guide](../docs/kubernetes/OPENSHIFT_MIGRATION_GUIDE.md)**: Red Hat OpenShift compatibility and migration path
- **[Kubernetes Overlays](kubernetes/overlays/README.md)**: Kustomize overlays for compliance and cloud providers

## Prerequisites

### Required

- Kubernetes cluster (v1.19+)
- Helm 3.0+
- kubectl configured to access your cluster
- Persistent volume provisioner

### Recommended for Production

- Ingress controller (nginx, traefik, etc.)
- cert-manager for TLS certificates
- Metrics server for HPA
- Prometheus and Grafana for monitoring
- External secrets operator (for secret management)
- Velero for backups

## Deployment Options

Nethical can be deployed in two ways:

1. **Kubernetes Manifests**: Direct kubectl apply of YAML files
2. **Helm Chart**: Templated deployment with values customization (recommended)

## Quick Start

### Using Helm (Recommended)

```bash
# Add namespace
kubectl create namespace nethical

# Install with default values
helm install nethical deploy/helm/nethical --namespace nethical

# Check deployment
kubectl get all -n nethical

# Access the service (port-forward)
kubectl port-forward -n nethical svc/nethical 8000:8000
```

### Using kubectl

```bash
# Apply all manifests
kubectl apply -f deploy/kubernetes/

# Check status
kubectl get all -n nethical
```

## Environment-Specific Deployments

### Development Environment

Optimized for local testing with minimal resources:

```bash
helm install nethical deploy/helm/nethical \
  --namespace nethical \
  --create-namespace \
  -f deploy/helm/nethical/values-dev.yaml
```

**Features:**
- Single replica
- 256Mi-1Gi memory
- No autoscaling
- 5Gi storage
- Port-forward access only

### Staging Environment

Production-like setup for testing:

```bash
helm install nethical deploy/helm/nethical \
  --namespace nethical \
  --create-namespace \
  -f deploy/helm/nethical/values-staging.yaml
```

**Features:**
- 2 replicas (scales to 5)
- 512Mi-2Gi memory per pod
- Basic autoscaling (CPU/Memory)
- 20Gi storage
- Ingress with staging TLS

### Production Environment

High-availability production deployment:

```bash
helm install nethical deploy/helm/nethical \
  --namespace nethical \
  --create-namespace \
  -f deploy/helm/nethical/values-production.yaml
```

**Features:**
- 5 replicas (scales to 30)
- 1-4Gi memory per pod
- Aggressive autoscaling
- 100Gi SSD storage
- Production ingress with TLS
- Pod anti-affinity across zones
- Strict pod disruption budget

## Production Best Practices

### 1. Security

#### Use External Secret Management

Instead of Kubernetes secrets, use:

```bash
# AWS Secrets Manager (with External Secrets Operator)
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: nethical-secrets
spec:
  secretStoreRef:
    name: aws-secrets-manager
  target:
    name: nethical-app-secrets
  data:
  - secretKey: api-key
    remoteRef:
      key: prod/nethical/api-key
```

#### Pod Security Standards

Apply pod security standards:

```bash
kubectl label namespace nethical \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

#### Network Policies

Restrict network traffic:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nethical-network-policy
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: nethical
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 53  # DNS
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### 2. High Availability

#### Multi-Zone Deployment

Distribute pods across availability zones:

```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchExpressions:
        - key: app.kubernetes.io/name
          operator: In
          values:
          - nethical
      topologyKey: topology.kubernetes.io/zone
```

#### Pod Disruption Budget

Ensure minimum availability during updates:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: nethical
spec:
  minAvailable: 3  # For 5-replica setup
  selector:
    matchLabels:
      app.kubernetes.io/name: nethical
```

### 3. Resource Management

#### Set Resource Requests and Limits

Always define resources:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

#### Enable Autoscaling

Use HPA for automatic scaling:

```bash
# HPA is enabled in values-production.yaml
# Monitor with:
kubectl get hpa -n nethical -w
```

### 4. Storage

#### Use Fast Storage Class

For production, use SSD-backed storage:

```yaml
persistence:
  enabled: true
  size: 100Gi
  storageClass: "fast-ssd"  # or "gp3", "pd-ssd"
```

#### Enable Volume Snapshots

Configure volume snapshot policies:

```yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshotClass
metadata:
  name: nethical-snapshots
driver: kubernetes.io/aws-ebs  # or your CSI driver
deletionPolicy: Retain
parameters:
  type: gp3
```

## Monitoring and Observability

### Prometheus Integration

Nethical exposes Prometheus metrics on port 8888:

```yaml
# ServiceMonitor for Prometheus Operator
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
```

### Grafana Dashboards

Import Nethical dashboards:

1. Access Grafana (default: http://localhost:3000)
2. Import dashboard from `dashboards/nethical-overview.json`
3. Configure data source to Prometheus

### Key Metrics to Monitor

- **Actions per second**: Request rate
- **P95/P99 latency**: Response time percentiles
- **Error rate**: Failed requests
- **Quota utilization**: Rate limit usage
- **Violations detected**: Safety/ethical violations
- **PII detections**: Privacy protection events

### Alerting Rules

Example Prometheus alerts:

```yaml
groups:
- name: nethical
  interval: 30s
  rules:
  - alert: NethicalHighErrorRate
    expr: rate(nethical_errors_total[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      
  - alert: NethicalHighLatency
    expr: histogram_quantile(0.95, nethical_request_duration_seconds) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "P95 latency above 500ms"
```

## Backup and Disaster Recovery

### Velero Setup

Install Velero for backup:

```bash
# Install Velero (example for AWS)
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.8.0 \
  --bucket nethical-backups \
  --backup-location-config region=us-east-1 \
  --snapshot-location-config region=us-east-1
```

### Backup Schedule

Create automated backups:

```yaml
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: nethical-daily-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  template:
    includedNamespaces:
    - nethical
    ttl: 720h  # 30 days retention
```

### Manual Backup

```bash
# Create backup
velero backup create nethical-manual-backup --include-namespaces nethical

# Check backup status
velero backup describe nethical-manual-backup

# Restore from backup
velero restore create --from-backup nethical-manual-backup
```

## Troubleshooting

### Common Issues

#### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n nethical

# Describe pod for events
kubectl describe pod <pod-name> -n nethical

# Check logs
kubectl logs -n nethical <pod-name>
```

#### PVC Not Binding

```bash
# Check PVC status
kubectl get pvc -n nethical

# Check available storage classes
kubectl get storageclass

# Describe PVC for details
kubectl describe pvc <pvc-name> -n nethical
```

#### High Memory Usage

```bash
# Check resource usage
kubectl top pods -n nethical

# Scale up resources
helm upgrade nethical deploy/helm/nethical \
  --set resources.limits.memory=8Gi \
  --reuse-values
```

#### Ingress Not Working

```bash
# Check ingress status
kubectl get ingress -n nethical

# Describe ingress
kubectl describe ingress nethical -n nethical

# Check ingress controller logs
kubectl logs -n ingress-nginx -l app.kubernetes.io/name=ingress-nginx
```

### Debug Mode

Enable debug logging:

```bash
helm upgrade nethical deploy/helm/nethical \
  --set extraEnv[0].name=LOG_LEVEL \
  --set extraEnv[0].value=DEBUG \
  --reuse-values
```

### Performance Issues

Check HPA metrics:

```bash
# View HPA status
kubectl get hpa -n nethical

# Describe HPA
kubectl describe hpa nethical -n nethical

# Check metrics server
kubectl top pods -n nethical
kubectl top nodes
```

## Upgrade Procedures

### Helm Upgrade

```bash
# Update to new version
helm upgrade nethical deploy/helm/nethical \
  -f values-production.yaml \
  --set image.tag=0.2.0

# Check rollout status
kubectl rollout status statefulset/nethical -n nethical
```

### Rollback

```bash
# View history
helm history nethical

# Rollback to previous version
helm rollback nethical

# Rollback to specific revision
helm rollback nethical 2
```

## Multi-Region Deployment

For global deployment, deploy to multiple regions:

```bash
# Region 1: US East
helm install nethical-us-east deploy/helm/nethical \
  --namespace nethical \
  -f values-production.yaml \
  --set nethical.regionId=us-east-1

# Region 2: EU West
helm install nethical-eu-west deploy/helm/nethical \
  --namespace nethical \
  -f values-production.yaml \
  --set nethical.regionId=eu-west-1

# Region 3: Asia Pacific
helm install nethical-ap-south deploy/helm/nethical \
  --namespace nethical \
  -f values-production.yaml \
  --set nethical.regionId=ap-south-1
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical
- Deployment Guide: deploy/DEPLOYMENT_GUIDE.md
- Kubernetes README: deploy/kubernetes/README.md
- Helm README: deploy/helm/nethical/README.md
