# Performance Testing Examples

This directory contains tools and configurations for performance testing and capacity planning for Nethical deployments.

## Contents

- **`generate_load.py`** - Load generator for throughput and latency testing
- **`tight_budget_config.env`** - Sample configuration for budget-conscious deployments

## Load Generator

The load generator (`generate_load.py`) simulates multiple AI agents producing actions at a target aggregate rate to validate SLOs and measure capacity.

### Quick Start

```bash
# Basic test: 100 agents at 50 RPS for 60 seconds
python generate_load.py --agents 100 --rps 50 --duration 60

# Test with specific feature set
python generate_load.py \
  --agents 200 \
  --rps 100 \
  --duration 60 \
  --shadow \
  --merkle \
  --privacy-mode standard
```

### Common Usage Patterns

**1. Baseline Performance Test (Core Features Only)**
```bash
python generate_load.py \
  --agents 200 --rps 100 --duration 60 \
  --no-shadow --no-ml-blend --no-anomaly
```

**2. Full-Stack Test (All Features)**
```bash
python generate_load.py \
  --agents 200 --rps 100 --duration 60 \
  --shadow --ml-blend --anomaly --merkle
```

**3. Stress Test (Find Breaking Point)**
```bash
python generate_load.py \
  --agents 1000 --rps 500 --duration 120 \
  --cohort stress-test
```

**4. Regional Sharding Test**
```bash
# Test multiple regions
python generate_load.py --agents 100 --rps 50 --region-id us-east-1 &
python generate_load.py --agents 100 --rps 50 --region-id eu-west-1 &
wait
```

### Output

The load generator produces:

1. **Console Output**: Real-time summary with:
   - Total actions and achieved RPS
   - Latency percentiles (p50, p95, p99)
   - Error rates and SLO compliance

2. **CSV File**: Per-action details including:
   - `agent_id`, `action_id`, `timestamp`
   - `latency_ms`, `status`, `error`
   - `violation_detected`

### SLO Validation

The generator automatically checks against defined SLOs:
- ✓ **p95 latency < 200ms**
- ✓ **p99 latency < 500ms**
- ✓ **Achieved RPS within 10% of target**
- ✓ **Error rate < 1%**

Exit code is 0 (success) if all SLOs pass, 1 (failure) otherwise.

### CLI Options

Run `python generate_load.py --help` for full documentation.

Key options:
- `--agents N` - Number of agents to simulate (default: 100)
- `--rps N` - Target aggregate requests/sec (default: 50)
- `--duration N` - Test duration in seconds (default: 60)
- `--storage-dir PATH` - Data storage location (default: ./nethical_perf)
- `--output FILE` - CSV output file (default: perf_results.csv)

Feature flags:
- `--shadow` / `--no-shadow` - ML shadow mode
- `--ml-blend` / `--no-ml-blend` - ML blended enforcement
- `--anomaly` / `--no-anomaly` - Anomaly detection
- `--merkle` / `--no-merkle` - Merkle anchoring
- `--quota` / `--no-quota` - Quota enforcement
- `--privacy-mode {standard,differential,none}` - Privacy mode
- `--redaction-policy {minimal,standard,aggressive}` - PII redaction

## Sample Configuration

The `tight_budget_config.env` file provides recommended starting values for deployments targeting ~100 RPS with headroom for growth.

### Usage

**With docker-compose:**
```bash
docker-compose --env-file examples/perf/tight_budget_config.env up
```

**Direct export:**
```bash
export $(cat examples/perf/tight_budget_config.env | grep -v '^#' | xargs)
python your_app.py
```

### Key Settings

- **Quota**: `NETHICAL_ENABLE_QUOTA=true`, `NETHICAL_REQUESTS_PER_SECOND=100.0`
- **Observability**: `NETHICAL_ENABLE_OTEL=true`
- **Privacy**: `NETHICAL_PRIVACY_MODE=standard`
- **Audit**: `NETHICAL_ENABLE_MERKLE=true`
- **ML**: Heavy features in shadow mode initially

### Growth Path

The config includes commented guidance for scaling:

1. **Stage 1**: Vertical scale (100 → 300 RPS)
   - Add CPU/RAM, enable quarantine, add Redis

2. **Stage 2**: Feature graduation (300 → 500 RPS)
   - Promote ML shadow → blended
   - Enable anomaly enforcement

3. **Stage 3**: Horizontal scale (500+ RPS)
   - Multiple instances, regional sharding
   - Shared Redis, centralized observability

## Analyzing Results

### Interpreting Latency Metrics

- **p50**: Half of requests complete faster (typical experience)
- **p95**: 95% of requests complete faster (SLO target: < 200ms)
- **p99**: 99% of requests complete faster (SLO target: < 500ms)

If SLOs are missed:
1. Disable heavy features (ML blended, anomaly)
2. Run expensive detectors in shadow mode
3. Increase hardware resources
4. Tune sampling/batching/caching parameters

### CSV Analysis

The output CSV can be analyzed with pandas, Excel, or other tools:

```python
import pandas as pd

df = pd.read_csv('perf_results.csv')

# Latency by agent
df.groupby('agent_id')['latency_ms'].describe()

# Violations over time
df[df['violation_detected'] == True].groupby('timestamp').size()

# Error analysis
df[df['status'] == 'error'].groupby('error').size()
```

## Integration with CI/CD

Add performance regression tests to your CI pipeline:

```yaml
# .github/workflows/perf-test.yml
- name: Performance baseline test
  run: |
    python examples/perf/generate_load.py \
      --agents 50 --rps 25 --duration 30 \
      --no-shadow --no-ml-blend --no-anomaly
  timeout-minutes: 5
```

## Further Reading

- [Performance Sizing Guide](../../docs/ops/PERFORMANCE_SIZING.md) - Detailed capacity planning
- [SLOs](../../docs/ops/SLOs.md) - Service level objectives
- [README](../../README.md#-performance--sizing) - Overview and quick start
