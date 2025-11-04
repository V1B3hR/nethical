# Nethical Helm Chart

A Helm chart for deploying Nethical - Safety governance system for AI agents.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- PV provisioner support in the underlying infrastructure (if persistence is enabled)

## Installing the Chart

### From Local Directory

```bash
# Install with default values
helm install nethical deploy/helm/nethical

# Install with custom values
helm install nethical deploy/helm/nethical -f my-values.yaml

# Install in specific namespace
helm install nethical deploy/helm/nethical --namespace nethical --create-namespace
```

### With Custom Values

```bash
# Install with custom configuration
helm install nethical deploy/helm/nethical \
  --set replicaCount=3 \
  --set nethical.regionId=eu-west-1 \
  --set autoscaling.enabled=true
```

## Upgrading the Chart

```bash
# Upgrade to new values
helm upgrade nethical deploy/helm/nethical -f my-values.yaml

# Force recreation of pods
helm upgrade nethical deploy/helm/nethical --force
```

## Uninstalling the Chart

```bash
# Uninstall release
helm uninstall nethical

# Uninstall and delete namespace
helm uninstall nethical --namespace nethical
kubectl delete namespace nethical
```

## Configuration

The following table lists the configurable parameters and their default values.

### Basic Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Image repository | `nethical` |
| `image.tag` | Image tag | `"latest"` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `nameOverride` | Override chart name | `""` |
| `fullnameOverride` | Override full name | `""` |

### Service Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `service.type` | Service type | `ClusterIP` |
| `service.port` | Service port | `8000` |
| `service.metricsPort` | Metrics port | `8888` |

### Ingress Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `false` |
| `ingress.className` | Ingress class name | `"nginx"` |
| `ingress.hosts[0].host` | Hostname | `nethical.example.com` |
| `ingress.tls` | TLS configuration | `[]` |

### Resource Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `resources.limits.cpu` | CPU limit | `1000m` |
| `resources.limits.memory` | Memory limit | `2Gi` |
| `resources.requests.cpu` | CPU request | `250m` |
| `resources.requests.memory` | Memory request | `512Mi` |

### Autoscaling Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `autoscaling.enabled` | Enable HPA | `false` |
| `autoscaling.minReplicas` | Minimum replicas | `1` |
| `autoscaling.maxReplicas` | Maximum replicas | `10` |
| `autoscaling.targetCPUUtilizationPercentage` | Target CPU % | `70` |
| `autoscaling.targetMemoryUtilizationPercentage` | Target Memory % | `80` |

### Persistence Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `persistence.enabled` | Enable persistence | `true` |
| `persistence.accessMode` | PVC access mode | `ReadWriteOnce` |
| `persistence.size` | PVC size | `10Gi` |
| `persistence.storageClass` | Storage class | `nil` |

### Nethical Application Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nethical.storageDir` | Data storage directory | `/data/nethical` |
| `nethical.regionId` | Region identifier | `us-east-1` |
| `nethical.logicalDomain` | Logical domain | `default` |
| `nethical.enableMerkle` | Enable Merkle anchoring | `true` |
| `nethical.enableQuarantine` | Enable quarantine mode | `true` |
| `nethical.enableQuota` | Enable quota enforcement | `true` |
| `nethical.enableOtel` | Enable OpenTelemetry | `true` |
| `nethical.requestsPerSecond` | Rate limit (RPS) | `10.0` |
| `nethical.maxPayloadBytes` | Max payload size | `1000000` |
| `nethical.privacyMode` | Privacy mode | `differential` |
| `nethical.epsilon` | Differential privacy epsilon | `1.0` |
| `nethical.redactionPolicy` | PII redaction policy | `aggressive` |

### Probes Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `probes.liveness.enabled` | Enable liveness probe | `true` |
| `probes.readiness.enabled` | Enable readiness probe | `true` |
| `probes.startup.enabled` | Enable startup probe | `true` |

## Examples

### Minimal Production Setup

```yaml
# production-values.yaml
replicaCount: 3

image:
  tag: "0.1.0"
  pullPolicy: Always

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

persistence:
  enabled: true
  size: 50Gi
  storageClass: "fast-ssd"

nethical:
  regionId: "us-east-1"
  requestsPerSecond: 100.0
  privacyMode: "differential"

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: nethical.prod.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: nethical-tls
      hosts:
        - nethical.prod.example.com
```

Install with production values:

```bash
helm install nethical deploy/helm/nethical -f production-values.yaml
```

### Development Setup

```yaml
# dev-values.yaml
replicaCount: 1

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 100m
    memory: 256Mi

autoscaling:
  enabled: false

persistence:
  enabled: true
  size: 5Gi

nethical:
  requestsPerSecond: 10.0
  privacyMode: "standard"
```

### High Availability Setup

```yaml
# ha-values.yaml
replicaCount: 5

autoscaling:
  enabled: true
  minReplicas: 5
  maxReplicas: 30

podDisruptionBudget:
  enabled: true
  minAvailable: 3

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - nethical
        topologyKey: kubernetes.io/hostname
```

## Testing

### Dry Run

```bash
# Test chart rendering
helm install nethical deploy/helm/nethical --dry-run --debug

# Test with custom values
helm install nethical deploy/helm/nethical -f my-values.yaml --dry-run --debug
```

### Template Rendering

```bash
# Render templates
helm template nethical deploy/helm/nethical

# Render with values
helm template nethical deploy/helm/nethical -f my-values.yaml
```

### Lint Chart

```bash
# Lint the chart
helm lint deploy/helm/nethical

# Lint with custom values
helm lint deploy/helm/nethical -f my-values.yaml
```

## Troubleshooting

### Check Release Status

```bash
helm status nethical
helm list
```

### View Release Values

```bash
helm get values nethical
helm get all nethical
```

### Debug Installation

```bash
# Get rendered manifests
helm get manifest nethical

# Check events
kubectl get events --sort-by='.lastTimestamp'

# Check pod status
kubectl get pods -l app.kubernetes.io/name=nethical
kubectl describe pod <pod-name>
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

## Best Practices

1. **Version Control**: Keep your values files in version control
2. **Secrets Management**: Use external secrets manager for production
3. **Resource Limits**: Always set resource limits in production
4. **High Availability**: Use multiple replicas with PDB
5. **Monitoring**: Enable Prometheus metrics and Grafana dashboards
6. **Backups**: Configure backup solutions for persistent data
7. **Security**: Use NetworkPolicies and Pod Security Standards

## Advanced Configuration

### Using External Secrets

For production, use external secret management:

```yaml
externalSecrets:
  enabled: true
  # Configuration depends on your secret manager
```

### Custom Policies

Override policy files:

```yaml
nethical:
  policies:
    correlationRules: |
      version: "1.0"
      rules:
        - name: "multi-step-attack"
          conditions:
            - type: "sequential"
              window: 300
    ethicsTaxonomy: |
      {
        "version": "1.0",
        "taxonomies": [
          {"name": "privacy", "severity": "high"}
        ]
      }
```

### Redis Integration

Enable Redis for caching:

```yaml
redis:
  enabled: true
  host: "redis-master.redis.svc.cluster.local"
  port: 6379
```

### Extra Environment Variables

```yaml
extraEnv:
  - name: CUSTOM_VAR
    value: "custom_value"
  - name: SECRET_VAR
    valueFrom:
      secretKeyRef:
        name: my-secret
        key: secret-key
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical
