# Nethical Deployment Guide

This guide covers deploying Nethical in production environments, including Docker, Kubernetes, and bare-metal deployments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Configuration](#configuration)
- [Security Considerations](#security-considerations)
- [Monitoring](#monitoring)
- [Scaling](#scaling)

## Prerequisites

- Docker 20.10+ (for containerized deployments)
- Kubernetes 1.24+ (for K8s deployments)
- Redis 6+ (recommended for production)
- Python 3.10+ (for bare-metal)

## Docker Deployment

### Using Docker Compose

The easiest way to deploy Nethical is with Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'

services:
  nethical-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NETHICAL_STORAGE_DIR=/data
      - NETHICAL_REGION_ID=us-east-1
      - REDIS_URL=redis://redis:6379
    volumes:
      - nethical-data:/data
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

volumes:
  nethical-data:
  redis-data:
```

Start the services:

```bash
docker-compose up -d
```

### Building the Docker Image

```bash
docker build -t nethical:latest .
```

### Running Standalone

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  -e NETHICAL_STORAGE_DIR=/data \
  nethical:latest
```

## Kubernetes Deployment

### Basic Deployment

```yaml
# nethical-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nethical-api
  labels:
    app: nethical
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nethical
  template:
    metadata:
      labels:
        app: nethical
    spec:
      containers:
      - name: nethical
        image: nethical:latest
        ports:
        - containerPort: 8000
        env:
        - name: NETHICAL_STORAGE_DIR
          value: /data
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: nethical-secrets
              key: redis-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        volumeMounts:
        - name: data
          mountPath: /data
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: nethical-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: nethical-api
spec:
  selector:
    app: nethical
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### Horizontal Pod Autoscaler

```yaml
# nethical-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nethical-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nethical-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NETHICAL_STORAGE_DIR` | Data storage directory | `./nethical_data` |
| `NETHICAL_REGION_ID` | Geographic region | `None` |
| `NETHICAL_DATA_RESIDENCY_POLICY` | Compliance policy | `None` |
| `REDIS_URL` | Redis connection URL | `None` |
| `NETHICAL_LOG_LEVEL` | Logging level | `INFO` |
| `NETHICAL_ENABLE_METRICS` | Enable Prometheus metrics | `true` |

### Configuration File

Create `config/production.yaml`:

```yaml
governance:
  region_id: "eu-west-1"
  data_residency_policy: "EU_GDPR"
  enable_shadow_mode: true
  enable_ml_blending: true
  enable_anomaly_detection: true

security:
  enable_quota_enforcement: true
  requests_per_second: 100
  max_payload_bytes: 10000000

monitoring:
  enable_metrics: true
  metrics_port: 9090
```

## Security Considerations

### TLS Configuration

Always use TLS in production:

```bash
uvicorn nethical.api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --ssl-keyfile /certs/key.pem \
  --ssl-certfile /certs/cert.pem
```

### API Authentication

Enable API key authentication:

```python
from nethical.api.auth import configure_auth

configure_auth(
    mode="strict",  # or "permissive" for development
    api_keys=["your-secure-api-key"],
)
```

### Network Security

- Use internal networks for Redis connections
- Implement network policies in Kubernetes
- Use a reverse proxy (nginx, traefik) for TLS termination

## Monitoring

### Prometheus Metrics

Nethical exposes Prometheus metrics at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'nethical'
    static_configs:
      - targets: ['nethical-api:9090']
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `nethical_requests_total` | Total API requests |
| `nethical_request_duration_seconds` | Request latency |
| `nethical_violations_detected_total` | Violations detected |
| `nethical_decisions_total` | Governance decisions |

### Health Checks

```bash
# Liveness probe
curl http://localhost:8000/health

# Readiness probe
curl http://localhost:8000/status
```

## Scaling

### Horizontal Scaling

Nethical is stateless and can be horizontally scaled:

1. Add more replicas in Kubernetes
2. Use load balancer to distribute traffic
3. Ensure Redis is shared across instances

### Performance Tuning

```bash
# Increase workers for better throughput
uvicorn nethical.api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

### Caching

Enable Redis caching for better performance:

```python
governance = IntegratedGovernance(
    storage_dir="./data",
    redis_client=redis_client,  # Redis connection
)
```

## Backup and Recovery

### Data Backup

```bash
# Backup storage directory
tar -czvf nethical-backup-$(date +%Y%m%d).tar.gz /data/nethical_data

# Backup Redis
redis-cli BGSAVE
```

### Disaster Recovery

1. Maintain multiple replicas across availability zones
2. Regular backups of storage directory and Redis
3. Test recovery procedures periodically

## Troubleshooting

### Common Issues

1. **Connection refused to Redis**
   - Check Redis URL configuration
   - Verify network connectivity

2. **High memory usage**
   - Increase memory limits
   - Enable garbage collection tuning

3. **Slow response times**
   - Add more workers
   - Check Redis performance
   - Enable caching

### Logs

```bash
# Docker logs
docker logs nethical-api

# Kubernetes logs
kubectl logs -f deployment/nethical-api
```

## Next Steps

- [Security Hardening Guide](SECURITY_HARDENING_GUIDE.md)
- [API Reference](API_USAGE.md)
- [Monitoring Guide](GOVERNANCE_OBSERVABILITY.md)
