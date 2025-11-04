# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying Nethical in a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (v1.19+)
- kubectl configured to access your cluster
- Persistent volume provisioner (for StatefulSet storage)
- Optional: Ingress controller (nginx, traefik, etc.)
- Optional: Metrics server (for HPA)

## Quick Start

### Deploy with kubectl

```bash
# Apply all manifests
kubectl apply -f deploy/kubernetes/

# Check deployment status
kubectl get all -n nethical

# View logs
kubectl logs -n nethical -l app.kubernetes.io/name=nethical

# Port forward to access the service
kubectl port-forward -n nethical svc/nethical 8000:8000
```

### Deploy Specific Components

```bash
# 1. Create namespace
kubectl apply -f namespace.yaml

# 2. Create ConfigMaps
kubectl apply -f configmap.yaml

# 3. Create Secrets (edit first!)
kubectl apply -f secret.yaml

# 4. Create PVC (if not using volumeClaimTemplates)
kubectl apply -f pvc.yaml

# 5. Deploy StatefulSet
kubectl apply -f statefulset.yaml

# 6. Create Services
kubectl apply -f service.yaml

# 7. Optional: Create Ingress
kubectl apply -f ingress.yaml

# 8. Optional: Enable autoscaling
kubectl apply -f hpa.yaml

# 9. Optional: Add PodDisruptionBudget
kubectl apply -f pdb.yaml
```

## Configuration

### ConfigMap

Edit `configmap.yaml` to customize Nethical configuration:

- `NETHICAL_REGION_ID`: AWS region or datacenter identifier
- `NETHICAL_REQUESTS_PER_SECOND`: Rate limiting threshold
- `NETHICAL_PRIVACY_MODE`: Privacy mode (differential, standard, strict)
- `NETHICAL_ENABLE_QUOTA`: Enable quota enforcement

### Secrets

Edit `secret.yaml` to add your secrets:

```yaml
stringData:
  API_KEY: "your-api-key"
  DB_PASSWORD: "your-db-password"
```

For production, use external secret management:
- HashiCorp Vault
- AWS Secrets Manager
- Azure Key Vault
- Google Secret Manager
- Sealed Secrets

### Storage

The StatefulSet uses `volumeClaimTemplates` for persistent storage:

```yaml
resources:
  requests:
    storage: 10Gi
storageClassName: standard  # Change to your storage class
```

### Resources

Adjust resource limits in `statefulset.yaml`:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

## High Availability

### Multi-Replica Setup

For HA deployment, scale the StatefulSet:

```bash
kubectl scale statefulset nethical --replicas=3 -n nethical
```

Or edit the manifest:

```yaml
spec:
  replicas: 3
```

### Horizontal Pod Autoscaler

Enable HPA for automatic scaling:

```bash
kubectl apply -f hpa.yaml

# Check HPA status
kubectl get hpa -n nethical
```

The HPA scales based on:
- CPU utilization (70% target)
- Memory utilization (80% target)

### Pod Disruption Budget

PDB ensures minimum availability during disruptions:

```yaml
spec:
  minAvailable: 1  # At least 1 pod must be available
```

## Ingress

### NGINX Ingress Controller

```yaml
annotations:
  cert-manager.io/cluster-issuer: letsencrypt-prod
  nginx.ingress.kubernetes.io/rewrite-target: /
```

### TLS/HTTPS

Add TLS configuration:

```yaml
tls:
- hosts:
  - nethical.example.com
  secretName: nethical-tls
```

Generate certificate with cert-manager:

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

## Monitoring

### Prometheus

The StatefulSet includes Prometheus annotations:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8888"
  prometheus.io/path: "/metrics"
```

### Health Checks

Three types of probes are configured:

- **Liveness**: Restarts unhealthy pods
- **Readiness**: Controls traffic routing
- **Startup**: Handles slow-starting applications

## Upgrade & Rollback

### Rolling Update

```bash
# Update image
kubectl set image statefulset/nethical nethical=nethical:v0.2.0 -n nethical

# Check rollout status
kubectl rollout status statefulset/nethical -n nethical
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo statefulset/nethical -n nethical
```

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -n nethical
kubectl describe pod <pod-name> -n nethical
```

### View Logs

```bash
kubectl logs -n nethical <pod-name>
kubectl logs -n nethical <pod-name> --previous  # Previous container
```

### Debug Container

```bash
kubectl exec -it -n nethical <pod-name> -- /bin/bash
```

### Events

```bash
kubectl get events -n nethical --sort-by='.lastTimestamp'
```

### Resource Usage

```bash
kubectl top pods -n nethical
kubectl top nodes
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f deploy/kubernetes/

# Or delete namespace
kubectl delete namespace nethical
```

## Best Practices

1. **Security**:
   - Use non-root user (UID 1000)
   - Enable Pod Security Standards
   - Use NetworkPolicies
   - Rotate secrets regularly

2. **Storage**:
   - Use appropriate storage class
   - Configure backup policies
   - Monitor disk usage

3. **Scaling**:
   - Set appropriate resource requests/limits
   - Use HPA for automatic scaling
   - Configure PDB for availability

4. **Monitoring**:
   - Enable Prometheus scraping
   - Configure alerting rules
   - Set up dashboards

5. **Updates**:
   - Use rolling updates
   - Test in staging first
   - Have rollback plan ready

## References

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [StatefulSet Best Practices](https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/)
- [Nethical Repository](https://github.com/V1B3hR/nethical)
