# Regional Configuration Files

This directory contains production-ready configuration files for multi-region deployments of Nethical to meet the short-term (6-month) scalability targets.

## Scalability Targets

These configurations support:
- ✅ **100+ sustained RPS** across 3 regions
- ✅ **500+ peak RPS** with burst capacity
- ✅ **1,000+ concurrent agents** distributed across regions
- ✅ **10M actions with full audit trails** using efficient storage
- ✅ **3-5 regional deployments** with independent operation

## Available Configurations

### us-east-1.env
**Region**: US East (Virginia)  
**Target Capacity**: 200 RPS sustained, 350 concurrent agents  
**Features**: 
- Standard privacy mode
- Full ML stack (shadow + blended + anomaly detection)
- Merkle anchoring for audit trails
- Performance optimization enabled

**Use case**: Primary production region for North American traffic

```bash
# Deploy with Docker
docker run -d \
  --name nethical-us-east-1 \
  --env-file config/us-east-1.env \
  -p 8001:8000 \
  nethical:latest

# Or with docker-compose
docker-compose --env-file config/us-east-1.env up
```

### eu-west-1.env
**Region**: EU West (Ireland)  
**Target Capacity**: 200 RPS sustained, 350 concurrent agents  
**Features**: 
- Differential privacy mode (GDPR compliant)
- Aggressive PII redaction
- Full ML stack
- Right-to-be-forgotten (RTBF) support
- Data residency enforcement

**Use case**: European production region with GDPR compliance

```bash
# Deploy with Docker
docker run -d \
  --name nethical-eu-west-1 \
  --env-file config/eu-west-1.env \
  -p 8002:8000 \
  nethical:latest
```

### ap-south-1.env
**Region**: Asia Pacific (Mumbai)  
**Target Capacity**: 150 RPS sustained, 300 concurrent agents  
**Features**: 
- Standard privacy mode
- Conservative ML configuration (shadow + anomaly only)
- Optimized for lower latency
- Performance optimization enabled

**Use case**: Asia Pacific production region with latency optimization

```bash
# Deploy with Docker
docker run -d \
  --name nethical-ap-south-1 \
  --env-file config/ap-south-1.env \
  -p 8003:8000 \
  nethical:latest
```

## Architecture Overview

```
                    Load Balancer
                    (Round-robin)
                          |
        +-----------------+------------------+
        |                 |                  |
   us-east-1        eu-west-1          ap-south-1
   200 RPS          200 RPS            150 RPS
   350 agents       350 agents         300 agents
        |                 |                  |
        +--------Redis Cluster---------------+
                (Quota Coordination)
```

**Total Capacity**: 550 RPS sustained, 1,000+ concurrent agents

## Hardware Requirements

Each regional instance requires:
- **CPU**: 8 vCPU (physical cores preferred)
- **Memory**: 32 GB RAM
- **Storage**: 200 GB SSD (NVMe preferred)
- **Network**: 10 Gbps link

## Shared Infrastructure

- **Redis Cluster**: 3 nodes × 16 GB RAM (high availability)
- **Observability Stack**: Prometheus + Grafana + OTEL Collector
- **Load Balancer**: Application load balancer with health checks

## Customization

### Adjusting RPS Targets

To modify the RPS target for a region:

```bash
# In region config file
NETHICAL_REQUESTS_PER_SECOND=300.0  # Increase from 200 to 300
```

### Enabling/Disabling Features

Toggle features to balance performance vs. functionality:

```bash
# Disable ML blending for lower latency
NETHICAL_ENABLE_ML_BLENDING=false

# Enable quarantine for suspicious agents
NETHICAL_ENABLE_QUARANTINE=true

# Adjust privacy mode
NETHICAL_PRIVACY_MODE=differential  # or 'standard' or 'none'
```

### Regional Variations

Different regions can use different configurations:

| Feature | US East | EU West | AP South |
|---------|---------|---------|----------|
| Privacy Mode | Standard | Differential | Standard |
| ML Blending | ✅ Yes | ✅ Yes | ❌ No |
| Redaction | Standard | Aggressive | Standard |
| Special | - | GDPR | Low Latency |

## Deployment Steps

### 1. Infrastructure Provisioning

```bash
# Provision infrastructure (Terraform, CloudFormation, etc.)
# - 3 EC2 instances (c6i.2xlarge or equivalent)
# - ElastiCache Redis cluster (3 nodes)
# - Application Load Balancer
# - Prometheus/Grafana (t3.xlarge)
```

### 2. Deploy Regional Instances

```bash
# Clone repository
git clone https://github.com/V1B3hR/nethical.git
cd nethical

# Build Docker image
docker build -t nethical:latest .

# Deploy each region
docker run -d --name nethical-us-east-1 --env-file config/us-east-1.env -p 8001:8000 nethical:latest
docker run -d --name nethical-eu-west-1 --env-file config/eu-west-1.env -p 8002:8000 nethical:latest
docker run -d --name nethical-ap-south-1 --env-file config/ap-south-1.env -p 8003:8000 nethical:latest
```

### 3. Configure Load Balancer

```yaml
# HAProxy example
backend nethical_backend
  balance roundrobin
  option httpchk GET /health
  server us-east-1 10.0.1.10:8000 check
  server eu-west-1 10.0.2.10:8000 check
  server ap-south-1 10.0.3.10:8000 check
```

### 4. Setup Monitoring

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'nethical'
    static_configs:
      - targets:
        - 'nethical-us-east-1:8000'
        - 'nethical-eu-west-1:8000'
        - 'nethical-ap-south-1:8000'
```

## Validation Testing

Use the load generator to validate each region:

```bash
# Test US East region
python examples/perf/generate_load.py \
  --agents 350 \
  --rps 200 \
  --duration 300 \
  --endpoint http://us-east-1:8000/process_action

# Test EU West region  
python examples/perf/generate_load.py \
  --agents 350 \
  --rps 200 \
  --duration 300 \
  --endpoint http://eu-west-1:8000/process_action

# Test AP South region
python examples/perf/generate_load.py \
  --agents 300 \
  --rps 150 \
  --duration 300 \
  --endpoint http://ap-south-1:8000/process_action
```

## Monitoring & Alerts

### Key Metrics

Monitor these metrics per region:
- Actions per second (target: within 5% of RPS limit)
- Latency percentiles (p50 < 50ms, p95 < 200ms, p99 < 500ms)
- Error rate (< 0.1%)
- CPU usage (< 70% sustained)
- Memory usage (< 85%)
- Storage growth (< 10 GB/day per region)

### Alert Thresholds

```yaml
# Critical alerts
- p95 latency > 500ms for 5 minutes
- Error rate > 1% for 5 minutes
- CPU > 90% for 5 minutes

# Warning alerts
- p95 latency > 200ms for 5 minutes
- CPU > 80% for 10 minutes
- Storage > 180 GB
```

## Cost Estimation

Monthly cost per deployment (AWS pricing):

| Component | Cost |
|-----------|------|
| 3× c6i.2xlarge (8 vCPU, 16 GB) | $734 |
| 3× 200 GB gp3 SSD | $48 |
| Redis cluster (3× r6g.xlarge) | $544 |
| Load balancer | $40 |
| Observability (t3.xlarge) | $120 |
| Data transfer (~5 TB) | $450 |
| **Total** | **~$1,936/mo** |

**Optimization**: Use reserved instances for 40% savings (~$1,160/mo)

## Troubleshooting

### High Latency

If latency exceeds SLOs:
1. Check CPU/memory usage
2. Reduce enabled features (disable ML blending)
3. Increase hardware resources
4. Enable caching (Redis)

### Low Throughput

If RPS is below target:
1. Check quota settings (`NETHICAL_REQUESTS_PER_SECOND`)
2. Verify load balancer configuration
3. Monitor network bandwidth
4. Scale horizontally (add more regions)

### Storage Growth

If storage grows too fast:
1. Enable compression (`NETHICAL_ENABLE_COMPRESSION=true`)
2. Reduce retention period
3. Archive cold data to S3
4. Implement storage tiering

## Additional Resources

- [Scalability Targets Guide](../docs/ops/SCALABILITY_TARGETS.md) - Comprehensive deployment guide
- [Performance Sizing Guide](../docs/ops/PERFORMANCE_SIZING.md) - Capacity planning
- [SLOs](../docs/ops/SLOs.md) - Service level objectives
- [Load Testing](../examples/perf/README.md) - Performance validation
- [Roadmap](../roadmap.md) - Future scaling targets

## Support

For questions or issues:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical/tree/main/docs

---

**Last Updated**: 2025-10-16  
**Applies to**: Nethical v0.1.0+
