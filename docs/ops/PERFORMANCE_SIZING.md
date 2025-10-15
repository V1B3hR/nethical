# Performance Sizing and Deployment Guide

## Overview

This guide helps operators size Nethical deployments for target agent counts and request rates, with a focus on tight-budget configurations that still provide headroom for very good performance.

For detailed service level objectives, see [SLOs.md](./SLOs.md).

## Performance SLOs

Nethical targets the following performance characteristics under normal operation:

- **Throughput**: 100–1000 actions/sec per instance
- **Latency**: 
  - p50 < 50ms
  - p95 < 200ms  
  - p99 < 500ms
- **Availability**: 99.9% uptime
- **False Positive Rate**: < 5% for BLOCK decisions

These SLOs apply to baseline configurations with core features enabled. Performance varies based on:
- Enabled features (see Feature Cost Tiers below)
- Hardware resources (CPU, memory, network)
- Workload characteristics (payload size, violation rates)
- Storage backend (local disk, Redis, S3)

## Agent → RPS Translation

Multi-agent LLM systems typically run 2–8 specialized agents per workflow (4–10 on the high end). Each agent generates actions at varying rates depending on the use case.

### Translation Examples

| Agents | Actions/Agent/Sec | Aggregate RPS | Use Case |
|--------|------------------|---------------|----------|
| 100 | 0.2 | 20 RPS | Low-activity monitoring |
| 100 | 0.5 | 50 RPS | Moderate interactive systems |
| 100 | 1.0 | 100 RPS | High-activity workflows |
| 500 | 0.2 | 100 RPS | Large fleet, low rate |
| 200 | 0.5 | 100 RPS | Medium fleet, moderate rate |
| 1000 | 0.5 | 500 RPS | Large-scale production |
| 2000 | 0.5 | 1000 RPS | High-scale production |

**Rule of thumb**: Start with 0.2–0.5 actions/agent/sec for typical LLM workflows. Measure actual rates during pilot testing.

## Feature Cost Tiers

Nethical features have varying performance impacts. Group them into tiers for capacity planning:

### Core (Low Overhead: ~5-15ms added latency)

These features provide essential safety with minimal performance impact:

- ✅ **Ethical/Safety Checks**: Rule-based violation detection (intent, ethical, safety, manipulation)
- ✅ **Quotas & Backpressure**: Rate limiting, per-agent/cohort quotas
- ✅ **Standard Redaction**: Basic PII detection and masking (email, phone, SSN)
- ✅ **Audit Logging**: Event recording (without Merkle anchoring)
- ✅ **Basic Risk Scoring**: Rule-based risk calculation

**Baseline capacity**: 500-1000 RPS on modern hardware (4-8 CPU cores, 8-16GB RAM)

### Medium (Moderate Overhead: ~15-40ms added latency)

Features that add noticeable but manageable cost:

- ⚠️ **Merkle Anchoring**: Batched immutable audit trail (batch every 100 events or 5 seconds)
- ⚠️ **SLA Monitoring**: Real-time SLA tracking and alerting
- ⚠️ **Ethical Taxonomy Tagging**: Violation categorization and tagging
- ⚠️ **Policy Diff Auditing**: Configuration change tracking
- ⚠️ **Correlation Detection**: Multi-step attack pattern matching (sliding window)
- ⚠️ **Quarantine Mode**: Temporary action isolation

**Expected impact**: ~20-30% throughput reduction vs. Core features  
**Recommended capacity**: 300-700 RPS

### Heavy (High Overhead: ~50-150ms added latency)

Compute-intensive features that provide advanced protection:

- 🔴 **ML Shadow Mode**: Parallel ML classifier for all actions
- 🔴 **ML Blended Enforcement**: ML + rule hybrid decisions (gray zone)
- 🔴 **Anomaly Detection**: Sequence, behavioral, and distribution drift monitoring
- 🔴 **Deep PII Policies**: Aggressive redaction with context analysis
- 🔴 **Differential Privacy**: Privacy-preserving aggregations (epsilon < 1.0)
- 🔴 **Human-in-the-Loop**: Escalation queue management with SLA tracking

**Expected impact**: ~50-70% throughput reduction vs. Core features  
**Recommended capacity**: 100-300 RPS (or run heavy features in shadow mode initially)

## Tight-Budget Profile

This profile balances cost and performance, providing strong governance while leaving headroom for growth.

### Starting Configuration

**Hardware**: 
- 2-4 CPU cores (physical or vCPU)
- 8-16 GB RAM
- 50 GB SSD storage
- 1 Gbps network

**Target Capacity**: 100 RPS (200 agents @ 0.5 actions/sec)

**Enabled Features**:
```yaml
Core Tier (Always On):
  - Ethical/safety rule checks ✅
  - Quotas & backpressure ✅
  - Standard redaction ✅
  - Basic audit logging ✅
  
Medium Tier (Selective):
  - Merkle anchoring (batched) ✅
  - SLA monitoring ✅
  - Ethical taxonomy ⚠️ (enable if needed)
  
Heavy Tier (Shadow Mode):
  - ML shadow ⚠️ (log scores, don't enforce)
  - Anomaly detection ⚠️ (alerting only)
  - ML blended ❌ (disable initially)
  - Deep PII ❌ (use standard tier)
```

**Environment Configuration** (see `examples/perf/tight_budget_config.env`):
```bash
NETHICAL_ENABLE_QUOTA=true
NETHICAL_REQUESTS_PER_SECOND=100.0
NETHICAL_ENABLE_OTEL=true
NETHICAL_PRIVACY_MODE=standard  # or "differential" if needed
NETHICAL_ENABLE_MERKLE=true
NETHICAL_ENABLE_QUARANTINE=false  # enable when needed
```

### Growth Path

As load increases or budget allows, scale in stages:

**Stage 1: Vertical Scaling** (100 → 300 RPS)
- Add 2-4 more CPU cores
- Increase RAM to 16-32 GB
- Enable quarantine and taxonomy tagging
- Consider Redis for caching

**Stage 2: Feature Graduation** (300 → 500 RPS)
- Promote ML shadow → ML blended (gray zone only)
- Enable anomaly detection enforcement
- Add aggressive PII redaction if required
- Increase Merkle batch size

**Stage 3: Horizontal Scaling** (500+ RPS)
- Deploy 2+ instances behind load balancer
- Shard by `region_id` or `logical_domain`
- Shared Redis cluster for state
- Centralized observability (Prometheus/Grafana)

## Tuning Checklist

Optimize performance within a given hardware profile:

### 1. Caching & Memoization
- ✅ **Policy Cache**: Cache loaded policies (ttl: 5 minutes)
- ✅ **Pass Decisions**: Cache low-risk decisions (ttl: 60 seconds, invalidate on policy change)
- ✅ **PII Patterns**: Compile regex patterns once at startup

### 2. Sampling & Throttling
- ⚠️ **Sample Expensive Detectors**: Run ML classifiers on 10-50% of traffic initially
- ⚠️ **Anomaly Detection Window**: Reduce sequence length (default: 3 → 5 actions)
- ⚠️ **Drift Calculation**: Run PSI/KL checks every N events (default: 100)

### 3. Batching & Async Processing
- ✅ **Merkle Batching**: Anchor every 100 events or 5 seconds (configurable)
- ✅ **Background Tasks**: Offload SLA checks, drift reporting to async workers
- ✅ **Bulk PII Detection**: Process multiple payloads together when possible

### 4. Threshold Tuning
- 🔧 **Gray Zone Width**: Widen gray zone (0.3-0.7) to reduce ML blending overhead
- 🔧 **Risk Decay**: Increase decay rate to forget old violations faster
- 🔧 **Low-Confidence Threshold**: Raise threshold (0.5 → 0.7) to reduce escalations

### 5. Storage Optimization
- 💾 **Retention Policies**: Limit audit log retention (default: 90 days)
- 💾 **Compression**: Enable compression for archived logs
- 💾 **Indexing**: Create indexes on `agent_id`, `timestamp`, `violation_type`

## Load Testing Recipe

Use `examples/perf/generate_load.py` to validate sizing and measure actual performance.

### Basic Test
```bash
# Test tight-budget profile at 100 RPS for 60 seconds
python examples/perf/generate_load.py \
  --agents 200 \
  --rps 100 \
  --duration 60 \
  --cohort production \
  --storage-dir ./nethical_perf
```

### Feature Comparison Test
```bash
# Core features only
python examples/perf/generate_load.py \
  --agents 200 --rps 100 --duration 60 \
  --no-shadow --no-ml-blend --no-anomaly

# With ML shadow
python examples/perf/generate_load.py \
  --agents 200 --rps 100 --duration 60 \
  --shadow --no-ml-blend --no-anomaly

# Full stack (heavy)
python examples/perf/generate_load.py \
  --agents 200 --rps 100 --duration 60 \
  --shadow --ml-blend --anomaly --merkle
```

### Stress Test
```bash
# Find breaking point
python examples/perf/generate_load.py \
  --agents 1000 --rps 500 --duration 120 \
  --cohort stress-test
```

### Analyzing Results

The load generator outputs:
- **Console**: Real-time summary (count, RPS, latency percentiles, errors)
- **CSV**: `perf_results.csv` with per-action details

Key metrics to watch:
- **Achieved RPS**: Should match target within 5-10%
- **p95 latency**: Should stay < 200ms (SLO target)
- **p99 latency**: Should stay < 500ms (SLO target)
- **Error rate**: Should be < 1% (excluding intentional quota rejections)

If SLOs are missed:
1. Reduce enabled features (move heavy → shadow mode)
2. Increase hardware resources (vertical scale)
3. Tune thresholds (sampling, batching, caching)
4. Consider horizontal scaling (multiple instances)

## Scale-Out Architecture

For deployments exceeding 500-1000 RPS, use horizontal scaling:

### Sharding Strategy

**Option 1: Region-based sharding**
```
Instance 1: region_id=us-east-1
Instance 2: region_id=eu-west-1
Instance 3: region_id=ap-south-1
```

**Option 2: Domain-based sharding**
```
Instance 1: logical_domain=customer-service
Instance 2: logical_domain=content-moderation
Instance 3: logical_domain=analytics
```

**Option 3: Hash-based sharding**
```
hash(agent_id) % N → instance N
```

### Shared State

Use Redis or equivalent for:
- Quota state (per-agent/cohort counters)
- Distributed rate limiting (token bucket)
- ML model cache (shared classifiers)
- Escalation queue (centralized human review)

### Load Balancing

- **Layer 4 (TCP)**: Simple round-robin or least-connections
- **Layer 7 (HTTP)**: Route by `region_id` or `logical_domain` header
- **Client-side**: Sticky sessions for stateful workflows

### Observability

- **Centralized Metrics**: Prometheus federation or remote write
- **Distributed Tracing**: OpenTelemetry with Jaeger/Tempo backend
- **Log Aggregation**: Loki, Elasticsearch, or CloudWatch Logs

## Monitoring & Alerting

Track these metrics in production (see [SLOs.md](./SLOs.md)):

**Golden Signals**:
- Request rate (actions/sec)
- Latency (p50, p95, p99)
- Error rate (%)
- Saturation (CPU, memory, quota utilization)

**Governance-Specific**:
- Violations per minute (by type)
- Risk score distribution
- Quota rejections (rate-limited agents)
- Escalation queue depth
- PII detections

**Alerts** (recommended thresholds):
- 🚨 **Critical**: p95 latency > 500ms, error rate > 1%, availability < 99.5%
- ⚠️ **Warning**: p95 latency > 200ms, CPU > 80%, memory > 85%
- ℹ️ **Info**: Quota throttling > 10%, unusual traffic patterns

## Example Deployments

### Small (Pilot / Development)
- **Scale**: 50 RPS, 100 agents
- **Hardware**: 2 vCPU, 4 GB RAM
- **Features**: Core + selective medium tier
- **Cost**: ~$20-50/month (cloud VM)

### Medium (Production Single-Region)
- **Scale**: 200 RPS, 400-500 agents
- **Hardware**: 4 vCPU, 16 GB RAM, Redis cache
- **Features**: Core + medium + ML shadow
- **Cost**: ~$100-200/month

### Large (Multi-Region Production)
- **Scale**: 1000 RPS, 2000+ agents
- **Hardware**: 3x instances (8 vCPU, 32 GB RAM each), Redis cluster
- **Features**: Full stack, regional sharding
- **Cost**: ~$500-1000/month

## Additional Resources

- [Service Level Objectives (SLOs)](./SLOs.md)
- [Load Generator README](../../examples/perf/generate_load.py)
- [Sample Configuration](../../examples/perf/tight_budget_config.env)
- [Docker Deployment Guide](../../README.md#-deployment)
- [OpenTelemetry Integration](../../README.md#-observability)

---

**Last Updated**: 2025-10-15  
**Applies to**: Nethical v0.1.0+
