# Nethical Edge Helm Chart

Ultra-low latency AI Safety Governance for edge deployments including autonomous vehicles, industrial robots, and safety-critical systems.

## Features

- **Ultra-Low Latency**: <10ms p99 decision latency
- **Offline-First**: Continues operation when disconnected
- **Minimal Footprint**: Optimized for edge hardware
- **Predictive Caching**: Pre-computes common decisions
- **Cloud Sync**: Periodic synchronization with central governance

## Installation

```bash
# Add the Nethical Helm repository
helm repo add nethical https://charts.nethical.io
helm repo update

# Install on edge nodes
helm install nethical-edge nethical/nethical-edge \
  --namespace nethical-edge \
  --create-namespace \
  --set nethical.edge.cloudSync.endpoint=https://api.nethical.example.com
```

## Configuration

### Edge Node Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 1 core | 2+ cores |
| Memory | 128MB | 256MB |
| Storage | 1GB | 5GB |
| Architecture | ARM64, AMD64 | ARM64, AMD64 |

### Key Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nethical.edgeMode` | Enable edge optimizations | `true` |
| `nethical.edge.latency.p99TargetMs` | P99 latency target | `25` |
| `nethical.edge.offline.enabled` | Enable offline mode | `true` |
| `nethical.edge.cloudSync.enabled` | Enable cloud sync | `true` |

## Deployment Targets

### Autonomous Vehicles

```yaml
# values-av.yaml
resources:
  requests:
    cpu: 200m
    memory: 256Mi
  limits:
    cpu: 1000m
    memory: 512Mi

nethical:
  edge:
    latency:
      p99TargetMs: 10
    prediction:
      enabled: true
      cacheWarmupOnStartup: true
```

### Industrial Robots

```yaml
# values-robot.yaml
resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 256Mi

nethical:
  edge:
    latency:
      p99TargetMs: 5
    offline:
      enabled: true
      maxOfflineDecisions: 50000
```

### Medical Devices

```yaml
# values-medical.yaml
resources:
  requests:
    cpu: 250m
    memory: 256Mi

nethical:
  edge:
    latency:
      p99TargetMs: 25
  enableMerkle: true  # Enable for audit trail
```

## Offline Operation

The edge chart includes robust offline fallback:

1. **Last Known Good Policies**: Uses cached policies when disconnected
2. **Conservative Risk Thresholds**: Applies stricter safety margins
3. **Decision Queueing**: Logs decisions for later sync
4. **Automatic Reconnection**: Syncs when connectivity restored

## Monitoring

Metrics exposed at `:8888/metrics`:

- `nethical_edge_decision_latency_ms` - Decision latency histogram
- `nethical_edge_cache_hit_ratio` - Cache hit rate
- `nethical_edge_offline_decisions_total` - Offline decision count
- `nethical_edge_sync_lag_seconds` - Time since last sync

## License

Apache 2.0 - See [LICENSE](../../../LICENSE)
