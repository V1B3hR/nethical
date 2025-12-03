# Global Deployment Guide

This guide covers deploying Nethical across multiple global regions for worldwide coverage, including infrastructure setup, compliance considerations, and satellite edge deployments.

## Overview

Nethical's global infrastructure provides:

- **17+ Regions**: Americas, Europe, Asia-Pacific, and China
- **Multi-Cloud**: AWS, GCP, Azure, and Alibaba Cloud support
- **Compliance**: GDPR, CCPA, LGPD, PIPEDA, CSL/PIPL, and more
- **Satellite Edge**: Global connectivity for remote/maritime/aviation

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Global Load Balancer                            │
│                    (Cloudflare / AWS Global Accelerator)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   Americas    │           │    Europe     │           │   Asia-Pac    │
│               │           │               │           │               │
│ ┌───────────┐ │           │ ┌───────────┐ │           │ ┌───────────┐ │
│ │  us-east  │ │           │ │  eu-west  │ │           │ │ ap-south  │ │
│ │  us-west  │ │           │ │ eu-central│ │           │ │ ap-northeast│
│ │  ca-centr │ │           │ │  eu-north │ │           │ │ ap-southeast│
│ │  sa-east  │ │           │ │  eu-south │ │           │ └───────────┘ │
│ └───────────┘ │           │ └───────────┘ │           │               │
│               │           │               │           │ ┌───────────┐ │
│  Redis Cluster│           │  Redis Cluster│           │ │  China    │ │
│  PostgreSQL   │           │  PostgreSQL   │           │ │ (Isolated)│ │
│               │           │  (Primary)    │           │ └───────────┘ │
└───────────────┘           └───────────────┘           └───────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │       Satellite Edge          │
                    │   (Starlink/OneWeb/Iridium)   │
                    │                               │
                    │  ┌─────────┐  ┌─────────┐    │
                    │  │Maritime │  │Aviation │    │
                    │  └─────────┘  └─────────┘    │
                    │                               │
                    │  ┌─────────┐  ┌─────────┐    │
                    │  │ Remote  │  │ Mobile  │    │
                    │  └─────────┘  └─────────┘    │
                    └───────────────────────────────┘
```

## Regional Deployments

### Americas

| Region | Location | Use Case | Notes |
|--------|----------|----------|-------|
| `us-east-1` | Virginia | Primary Americas | Low latency to East Coast |
| `us-west-2` | Oregon | West Coast | Gaming, media |
| `us-central1` | Iowa | Central US | GCP-based |
| `ca-central-1` | Montreal | Canada | PIPEDA compliant |
| `sa-east-1` | São Paulo | South America | LGPD compliant |

Deploy to Americas:

```bash
# US East (Primary)
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/us-east-1/

# US West
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/us-west-2/

# Canada
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/ca-central-1/

# Brazil
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/sa-east-1/
```

### Europe (GDPR Compliant)

| Region | Location | Use Case | Compliance |
|--------|----------|----------|------------|
| `eu-west-1` | Ireland | Primary EU | GDPR |
| `eu-west-2` | London | UK | UK-GDPR |
| `eu-central-1` | Frankfurt | Central EU | GDPR |
| `eu-north-1` | Stockholm | Nordic | GDPR |
| `eu-south-1` | Milan | Southern EU | GDPR |
| `europe-east1` | Warsaw | Eastern EU | GDPR |

Deploy to Europe:

```bash
# Primary EU
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/eu-west-1/

# All EU regions with GDPR compliance
for region in eu-central-1 eu-north-1 eu-south-1 europe-east1 eu-west-2; do
    kubectl apply -k deploy/kubernetes/multi-cluster/overlays/$region/
done
```

### Asia-Pacific

| Region | Location | Use Case | Compliance |
|--------|----------|----------|------------|
| `ap-south-1` | Mumbai | India, Middle East | - |
| `ap-southeast-2` | Sydney | Australia, NZ | Australian Privacy Act |
| `ap-northeast-1` | Tokyo | Japan | APPI |
| `ap-northeast-2` | Seoul | South Korea | PIPA |

Deploy to APAC:

```bash
# Mumbai
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/ap-south-1/

# Sydney
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/ap-southeast-2/

# Tokyo
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/ap-northeast-1/

# Seoul
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/ap-northeast-2/
```

### China (Isolated)

| Region | Location | Compliance | Notes |
|--------|----------|------------|-------|
| `cn-north-1` | Beijing | CSL, PIPL | AWS China |
| `cn-east-2` | Shanghai | CSL, PIPL | Alibaba Cloud |

**Important**: China regions are isolated with no cross-border data transfers.

```bash
# Beijing
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/cn-north-1/

# Shanghai
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/cn-east-2/
```

### Satellite Edge

For remote, maritime, and aviation deployments:

```bash
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/satellite/
```

## Infrastructure Setup

### Prerequisites

1. **Kubernetes Clusters**: One per region (EKS, GKE, or AKS)
2. **Service Mesh**: Istio 1.18+ for cross-region traffic
3. **DNS**: Global load balancing (Route53, Cloud DNS)
4. **Secrets**: HashiCorp Vault or cloud secret managers

### Terraform Deployment

Use the provided Terraform modules:

```bash
cd deploy/terraform/multi-region

# Initialize
terraform init

# Plan
terraform plan -var="regions=[\"us-east-1\",\"eu-west-1\",\"ap-south-1\"]"

# Apply
terraform apply
```

Modules included:
- VPC with multi-AZ subnets
- EKS clusters with autoscaling
- RDS PostgreSQL with cross-region replication
- Redis Global Datastore
- VPC peering for connectivity

### Redis Cluster Setup

Deploy regional Redis clusters:

```bash
# Americas
kubectl apply -f deploy/redis/americas-cluster.yaml

# Europe (GDPR compliant)
kubectl apply -f deploy/redis/europe-cluster.yaml

# APAC
kubectl apply -f deploy/redis/apac-cluster.yaml

# Satellite edge
kubectl apply -f deploy/redis/satellite-edge.yaml
```

## Compliance Configuration

### GDPR (Europe)

All EU region overlays include:

```yaml
commonLabels:
  compliance/gdpr: "true"

commonAnnotations:
  nethical.io/data-residency: "eu"
  nethical.io/compliance-framework: "gdpr"
```

NetworkPolicy enforces EU-only data transfers:

```yaml
ingress:
  - from:
      - podSelector:
          matchLabels:
            compliance/gdpr: "true"
```

### China CSL/PIPL

China regions are completely isolated:

```yaml
configMapGenerator:
  - name: nethical-config
    literals:
      - NETHICAL_DATA_LOCALIZATION=strict
      - NETHICAL_CROSS_BORDER_TRANSFER=disabled
```

### Regional Compliance Summary

| Region | Framework | Data Residency | Cross-Border |
|--------|-----------|----------------|--------------|
| EU | GDPR | EU only | Requires SCCs |
| UK | UK-GDPR | UK/EU | Adequacy decision |
| Canada | PIPEDA | Canada | With consent |
| Brazil | LGPD | Brazil | With consent |
| Australia | Privacy Act | Australia | Permitted |
| Japan | APPI | Japan | With consent |
| Korea | PIPA | Korea | With consent |
| China | CSL/PIPL | China only | Prohibited* |

*Cross-border transfers require security assessment and approval.

## Cache Hierarchy

The global cache architecture provides:

| Level | Location | Latency | TTL |
|-------|----------|---------|-----|
| L1 | In-process | <1ms | 30s |
| L2 | Regional Redis | <5ms | 5min |
| L3 | Global Redis | <50ms | 15min |
| Satellite | Edge cache | Variable | 30min |

### Configuration

```python
from nethical.cache import CacheHierarchy, HierarchyConfig

config = HierarchyConfig(
    enable_l1=True,
    enable_l2=True,
    enable_l3=True,
    enable_satellite=True,  # Enable for satellite edge
    
    l1_ttl_seconds=30,
    l2_ttl_seconds=300,
    l3_ttl_seconds=900,
    satellite_ttl_seconds=1800,  # Longer for satellite
    
    compression_enabled=True,
    bandwidth_aware_sync=True,
)

cache = CacheHierarchy(config)
```

## Monitoring

### Prometheus Metrics

All regions expose metrics:

```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9090"
```

Key metrics:
- `nethical_cache_hit_rate` - Cache hit rate per level
- `nethical_latency_seconds` - Request latency histogram
- `nethical_satellite_connection_status` - Satellite link health
- `nethical_failover_events_total` - Failover count

### Grafana Dashboards

Import dashboards from `dashboards/`:
- `global-overview.json` - Multi-region overview
- `regional-detail.json` - Per-region details
- `satellite-connectivity.json` - Satellite metrics
- `cache-hierarchy.json` - Cache performance

### Alerting

Example alert rules:

```yaml
groups:
  - name: nethical-global
    rules:
      - alert: RegionDown
        expr: up{job="nethical"} == 0
        for: 5m
        labels:
          severity: critical
          
      - alert: HighCrossRegionLatency
        expr: nethical_cross_region_latency_seconds > 0.5
        for: 10m
        labels:
          severity: warning
          
      - alert: SatelliteFailover
        expr: increase(nethical_failover_events_total[5m]) > 3
        labels:
          severity: warning
```

## Failover & Disaster Recovery

### Regional Failover

Automatic failover via Route53 health checks:

```yaml
# Route53 health check
Type: HTTPS
ResourcePath: /health
FailureThreshold: 3
RequestInterval: 30
```

### Cross-Region Replication

PostgreSQL streaming replication:

```
Primary (eu-west-1) → Read Replica (us-east-1)
                   → Read Replica (ap-south-1)
```

Redis Global Datastore:

```
Primary (us-east-1) ↔ Secondary (eu-west-1)
                    ↔ Secondary (ap-south-1)
```

### Satellite Failover

Automatic failover between terrestrial and satellite:

```python
from nethical.connectivity.satellite import FailoverManager

# Configure failover
failover = FailoverManager(config, satellite_provider)
await failover.start_monitoring()

# Auto-switches when terrestrial fails
# Auto-returns when terrestrial recovers
```

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| L1 cache hit | <1ms | All regions |
| L2 regional cache | <5ms | Same region |
| L3 global cache | <50ms | Cross-region |
| Satellite failover | <10s | Connection switch |
| Regional failover | <60s | DNS propagation |
| API availability | 99.9% | Per region |
| Global availability | 99.99% | Multi-region |

## Troubleshooting

### Region Not Responding

```bash
# Check pods
kubectl get pods -n nethical --context=$REGION_CONTEXT

# Check events
kubectl get events -n nethical --context=$REGION_CONTEXT --sort-by='.lastTimestamp'

# Check Istio
istioctl analyze -n nethical --context=$REGION_CONTEXT
```

### Cross-Region Latency

```bash
# Test latency
kubectl exec -n nethical deploy/nethical-api --context=$REGION1 -- \
    curl -w "@/tmp/curl-format.txt" -o /dev/null -s \
    https://nethical-api.$REGION2.nethical.io/health
```

### Cache Sync Issues

```python
# Check cache status
metrics = cache.get_metrics()
print(f"L1 hit rate: {metrics['l1_hit_rate']:.2%}")
print(f"Satellite pending: {metrics['satellite_metrics']['pending_sync']}")

# Force sync
await cache.sync_satellite()
```

### Satellite Connection

```python
# Check satellite health
signal = await provider.get_signal_info()
print(f"Online: {signal['is_online']}")
print(f"Latency: {signal['latency_ms']}ms")

# Check failover status
status = failover.get_status()
print(f"Active: {status['active_connection']}")
```

## Best Practices

1. **Regional Affinity**: Route users to nearest region
2. **Cache Warming**: Pre-populate caches during deployment
3. **Gradual Rollout**: Deploy to one region, verify, then expand
4. **Compliance First**: Verify data residency before deployment
5. **Monitoring**: Set up alerts before going live
6. **Backup Strategy**: Regular backups with cross-region copies
7. **Disaster Recovery**: Test failover procedures quarterly
8. **Satellite Planning**: Pre-position cache data for offline operation

## References

- [Kubernetes Multi-Cluster Deployment](https://kubernetes.io/docs/concepts/cluster-administration/federation/)
- [Istio Multi-Cluster](https://istio.io/latest/docs/setup/install/multicluster/)
- [AWS Global Accelerator](https://aws.amazon.com/global-accelerator/)
- [GDPR Compliance](https://gdpr.eu/)
- [Satellite Integration Guide](SATELLITE_INTEGRATION_GUIDE.md)
