# Advanced Validation Testing Guide

This document describes the advanced validation testing suite for production-grade confidence in Nethical.

## Overview

The advanced validation testing module provides comprehensive testing capabilities for validating system behavior under various conditions:

- **High Iteration Tests**: 10,000 to 100,000 iterations to expose rare edge cases and tail latencies
- **Worker Concurrency**: 50-70 workers for realistic loads, 100+ workers for stress testing
- **Variable Scaling**: Ramp-up and ramp-down logic to test elasticity and resilience
- **Soak Tests**: Sustained runs over hours/days to catch memory leaks or performance drift
- **Chaos/Failure Injection**: Fault injection to validate system resilience

## Quick Start

### Running Standard Tests

```bash
# Run standard validation tests
pytest tests/validation/test_advanced_validation.py -v -s

# Or use the convenience script
./scripts/run_advanced_validation.sh
```

### Running Extended Tests

```bash
# Run extended tests with 100k iterations
pytest tests/validation/test_advanced_validation.py -v -s --run-extended

# Or use the script
./scripts/run_advanced_validation.sh --extended
```

### Running Soak Tests

```bash
# Run 2-hour soak test
pytest tests/validation/test_advanced_validation.py -v -s --run-soak

# Or use the script
./scripts/run_advanced_validation.sh --soak
```

## Test Categories

### 1. High Iteration Tests

Tests that run many iterations to expose rare edge cases:

| Test | Iterations | Workers | Purpose |
|------|------------|---------|---------|
| `test_high_iteration_10k` | 10,000 | 50 | Standard edge case detection |
| `test_high_iteration_100k` | 100,000 | 70 | Extended edge case detection |

**Success Criteria:**
- Success rate ≥ 95%
- P95 latency < 200ms
- P99 latency < 500ms

### 2. Worker Concurrency Tests

Tests with varying numbers of concurrent workers:

| Test | Workers | Purpose |
|------|---------|---------|
| `test_worker_concurrency_realistic` | 60 | Simulate realistic multi-user loads |
| `test_stress_100_workers` | 100 | Peak traffic simulation |
| `test_stress_150_workers` | 150 | Extreme stress testing |

**Success Criteria:**
- Realistic load: ≥ 95% success rate
- Stress test: ≥ 80% success rate
- Extreme stress: ≥ 70% success rate

### 3. Variable Scaling Tests

Tests with ramp-up and ramp-down of workers:

| Test | Min Workers | Max Workers | Purpose |
|------|-------------|-------------|---------|
| `test_worker_scaling_ramp` | 5 | 70 | Elasticity testing |

**Success Criteria:**
- Success rate ≥ 90% during scaling
- No errors during ramp transitions

### 4. Soak Tests

Long-running tests for stability validation:

| Test | Duration | Workers | Purpose |
|------|----------|---------|---------|
| `test_soak_short` | 5 min | 20 | Quick CI/CD validation |
| `test_soak_2hour` | 2 hours | 30 | Production validation |

**Success Criteria:**
- Memory growth < 5%
- Performance degradation < 20%
- Success rate ≥ 95%

### 5. Chaos/Failure Injection Tests

Tests with injected failures for resilience:

| Test | Failure Rate | Purpose |
|------|--------------|---------|
| `test_chaos_failure_injection` | 1% | Validate graceful degradation |

**Failure Types Injected:**
- Timeout
- Slow response
- Random exception
- Memory pressure
- CPU spike

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NETHICAL_ADVANCED_ITERATIONS` | 10000 | Override iteration count |
| `NETHICAL_MAX_WORKERS` | 150 | Override max worker count |
| `NETHICAL_SOAK_DURATION` | 300 | Override soak duration (seconds) |

### Test Configuration

The `ValidationConfig` dataclass provides all configuration options:

```python
@dataclass
class ValidationConfig:
    # Iteration settings
    min_iterations: int = 1000
    standard_iterations: int = 10000
    extended_iterations: int = 100000
    
    # Worker settings
    min_workers: int = 5
    standard_workers: int = 50
    high_workers: int = 70
    stress_workers: int = 100
    max_workers: int = 150
    
    # Soak test settings
    short_soak_duration_seconds: float = 300.0  # 5 minutes
    long_soak_duration_seconds: float = 7200.0  # 2 hours
    
    # SLO thresholds
    success_rate_threshold: float = 0.95
    p95_latency_threshold_ms: float = 200.0
    p99_latency_threshold_ms: float = 500.0
    
    # Failure injection
    failure_injection_rate: float = 0.01  # 1%
```

## Output and Reports

### Report Location

All test reports are saved to:
```
tests/validation/results/advanced/
```

### Report Types

Each test generates:

1. **JSON Report** (`{test_name}_{timestamp}.json`)
   - Machine-readable format
   - Complete metrics and statistics
   - Suitable for CI/CD integration

2. **Markdown Report** (`{test_name}_{timestamp}.md`)
   - Human-readable format
   - Summary tables
   - Pass/fail indicators

### Report Contents

Reports include:

- **Summary**: Duration, total requests, success rate, throughput
- **Latency Statistics**: Mean, median, P50, P90, P95, P99, min, max
- **Memory Analysis**: Initial/final memory, growth percentage, leak detection
- **Performance Analysis**: Early vs late latency, degradation detection
- **Test Result**: Pass/fail with specific issues identified

## CI/CD Integration

### GitHub Actions Example

```yaml
jobs:
  validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -e ".[test]"
      
      - name: Run quick validation
        run: ./scripts/run_advanced_validation.sh --quick
      
      - name: Upload reports
        uses: actions/upload-artifact@v4
        with:
          name: validation-reports
          path: tests/validation/results/advanced/
```

### Nightly Extended Tests

```yaml
on:
  schedule:
    - cron: '0 0 * * *'  # Midnight daily

jobs:
  extended-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run extended validation
        run: ./scripts/run_advanced_validation.sh --extended
        timeout-minutes: 60
```

## Interpreting Results

### Success Rate Thresholds

| Test Type | Expected Success Rate |
|-----------|----------------------|
| Standard | ≥ 95% |
| Stress (100 workers) | ≥ 80% |
| Extreme (150 workers) | ≥ 70% |
| With failures injected | ≥ 85% |

### Memory Leak Detection

A memory leak is flagged when:
- Memory growth > 5% between first and last 10% of samples
- Consistent upward trend in memory usage

### Performance Degradation

Performance degradation is flagged when:
- Mean latency increases > 20% over time
- P95 latency exceeds thresholds late in test

## Troubleshooting

### Common Issues

1. **Low Success Rate**
   - Check system resources (CPU, memory)
   - Review error logs for patterns
   - Reduce worker count

2. **High Latency**
   - Profile the governance evaluation code
   - Check for lock contention
   - Review database/storage performance

3. **Memory Leak Detected**
   - Run with profiler (e.g., memory_profiler)
   - Check for unclosed resources
   - Review caching behavior

4. **Test Timeout**
   - Reduce iteration count
   - Reduce worker count
   - Increase timeout settings

### Debug Mode

Run with verbose logging:
```bash
PYTHONUNBUFFERED=1 pytest tests/validation/test_advanced_validation.py -v -s --log-cli-level=DEBUG
```

## Extending the Tests

### Adding New Test Scenarios

```python
@pytest.mark.asyncio
async def test_custom_scenario(test_runner, output_dir):
    """Custom test scenario"""
    results = test_runner.run_high_iteration_test(
        iterations=5000,
        workers=40,
        inject_failures=False,
    )
    
    files = results.save_report(output_dir)
    
    # Custom assertions
    assert results.get_success_rate() >= 0.95
```

### Adding New Failure Types

```python
class FailureType(Enum):
    TIMEOUT = "timeout"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_SPIKE = "cpu_spike"
    RANDOM_EXCEPTION = "random_exception"
    SLOW_RESPONSE = "slow_response"
    # Add new types here
    NETWORK_PARTITION = "network_partition"
```

## Performance Baselines

| Metric | Target | Acceptable | Critical |
|--------|--------|------------|----------|
| P95 Latency | < 100ms | < 200ms | > 500ms |
| P99 Latency | < 200ms | < 500ms | > 1000ms |
| Success Rate | > 99% | > 95% | < 90% |
| Memory Growth | < 2% | < 5% | > 10% |
| Throughput | > 100 RPS | > 50 RPS | < 20 RPS |

## Related Documentation

- [Performance Testing Guide](../docs/ops/PERFORMANCE_TESTING.md)
- [Scalability Targets](../docs/ops/SCALABILITY_TARGETS.md)
- [Chaos Testing Guide](../tests/resilience/README.md)
