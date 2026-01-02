# Performance Regression Detection Guide

## Overview

Nethical implements automated performance regression detection in CI/CD to catch performance issues early. This guide covers benchmarking, regression detection, and performance monitoring.

## Table of Contents

1. [Automated Benchmarking](#automated-benchmarking)
2. [Regression Detection](#regression-detection)
3. [Memory Profiling](#memory-profiling)
4. [Performance History](#performance-history)
5. [Local Performance Testing](#local-performance-testing)
6. [Optimization Guidelines](#optimization-guidelines)

---

## Automated Benchmarking

### Workflow Overview

The performance regression workflow runs automatically on:
- **Pull Requests**: Compare PR performance against base branch
- **Pushes to main**: Track performance history
- **Manual trigger**: On-demand performance testing

```yaml
# .github/workflows/performance-regression.yml
trigger:
  - pull_request (branches: main, develop)
  - push (branches: main)
  - workflow_dispatch
```

### Benchmark Configuration

Default benchmark parameters:

```bash
agents: 100          # Number of concurrent agents
rps: 50             # Requests per second
duration: 30        # Test duration in seconds
cohort: benchmark   # Test cohort identifier
```

### Metrics Tracked

- **P50 Latency**: Median response time
- **P95 Latency**: 95th percentile response time
- **P99 Latency**: 99th percentile response time
- **Throughput**: Requests per second achieved
- **Error Rate**: Percentage of failed requests

## Regression Detection

### Detection Thresholds

Performance regression is flagged when:

| Metric | Threshold | Action |
|--------|-----------|--------|
| P50 Latency | >10% increase | ⚠️ Warning |
| P95 Latency | >15% increase | ⚠️ Warning |
| P99 Latency | >20% increase | ℹ️ Info |
| Error Rate | >1% increase | ❌ Fail |

### Example PR Comment

When a pull request is created, the workflow automatically comments:

```markdown
## ✅ Performance Benchmark Results

### No Performance Regression

| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| P50 Latency | 45.2 ms | 46.1 ms | +2.0% |
| P95 Latency | 98.5 ms | 101.2 ms | +2.7% |

✅ Performance is within acceptable limits.

---
*Benchmark Configuration: 100 agents, 50 RPS, 30 seconds*
*Baseline: `main`*
```

### Regression Detected

```markdown
## ⚠️ Performance Benchmark Results

### **Performance Regression Detected**

| Metric | Baseline | Current | Change |
|--------|----------|---------|--------|
| P50 Latency | 45.2 ms | 52.8 ms | +16.8% |
| P95 Latency | 98.5 ms | 125.3 ms | +27.2% |

⚠️ **Warning**: P50 latency has increased by more than 10%.
Please review the changes for potential performance impacts.

---
*Benchmark Configuration: 100 agents, 50 RPS, 30 seconds*
*Baseline: `main`*
```

## Memory Profiling

### Automated Memory Testing

The workflow includes memory profiling:

```python
# Tracks memory usage during test
- Start memory: 150 MB
- End memory: 180 MB
- Increase: 30 MB (for 100 actions)
```

### Memory Leak Detection

Flags potential memory leaks:

```
✅ Memory usage is acceptable: 30.5 MB
⚠️  WARNING: High memory usage detected: 125.8 MB
```

### Memory Profiling Output

Detailed memory profile uploaded as artifact:

```
Line #    Mem usage    Increment  Occurrences   Line Contents
==============================================================
    10     45.2 MiB     45.2 MiB           1   @profile
    11                                         def test_governance_memory():
    12     45.5 MiB      0.3 MiB           1       gov = IntegratedGovernance()
    13     46.8 MiB      1.3 MiB         100       for i in range(100):
    14     46.8 MiB      0.0 MiB         100           result = gov.process_action(...)
```

## Performance History

### Tracking Over Time

On every push to `main`, performance metrics are tracked:

```bash
benchmark-history/
├── benchmark_20251105_090000.csv
├── benchmark_20251105_120000.csv
└── benchmark_20251105_150000.csv
```

### Visualizing Trends

Use the collected data to create performance dashboards:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark history
df = pd.read_csv('benchmark-history/benchmark_*.csv')

# Plot P50 latency over time
plt.plot(df['timestamp'], df['p50_latency'])
plt.xlabel('Time')
plt.ylabel('P50 Latency (ms)')
plt.title('Performance Trend')
plt.show()
```

## Local Performance Testing

### Running Benchmarks Locally

```bash
# Basic benchmark
python examples/perf/generate_load.py \
  --agents 100 \
  --rps 50 \
  --duration 30

# High load test
python examples/perf/generate_load.py \
  --agents 500 \
  --rps 200 \
  --duration 60

# With specific features
python examples/perf/generate_load.py \
  --agents 200 \
  --rps 100 \
  --duration 30 \
  --shadow \
  --ml-blend
```

### Profiling Locally

#### CPU Profiling

```bash
# Using cProfile
python -m cProfile -o profile.stats examples/perf/generate_load.py --agents 100 --rps 50 --duration 30

# View results
python -m pstats profile.stats
>>> sort cumtime
>>> stats 20
```

#### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler examples/perf/generate_load.py --agents 100 --rps 50 --duration 30

# Or use memray
pip install memray
memray run examples/perf/generate_load.py --agents 100 --rps 50 --duration 30
memray flamegraph memray-output.bin
```

#### Line Profiling

```bash
# Install line profiler
pip install line-profiler

# Add @profile decorator to functions
# Run profiler
kernprof -l -v examples/perf/generate_load.py --agents 100 --rps 50 --duration 30
```

### Comparing Branches

```bash
# Run baseline benchmark on main
git checkout main
python examples/perf/generate_load.py --agents 100 --rps 50 --duration 30 --output baseline.csv

# Run benchmark on feature branch
git checkout feature-branch
python examples/perf/generate_load.py --agents 100 --rps 50 --duration 30 --output feature.csv

# Compare results
python scripts/compare_benchmarks.py baseline.csv feature.csv
```

## Optimization Guidelines

### 1. Identify Bottlenecks

```python
# Use profiling to find slow code
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = gov.process_action(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

### 2. Optimize Hot Paths

Focus on code executed most frequently:

```python
# Before: Slow string concatenation
description = ""
for item in items:
    description += str(item)

# After: Fast string joining
description = "".join(str(item) for item in items)
```

### 3. Cache Expensive Operations

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(value):
    # Expensive operation
    return result
```

### 4. Use Async Operations

```python
# Before: Sequential operations
for action in actions:
    result = process_action(action)

# After: Concurrent operations
results = await asyncio.gather(*[
    process_action(action) for action in actions
])
```

### 5. Optimize Data Structures

```python
# Before: O(n) lookup
if item in list_of_items:
    ...

# After: O(1) lookup
if item in set_of_items:
    ...
```

### 6. Reduce Memory Allocations

```python
# Before: Creates new list each time
def get_violations():
    violations = []
    # ... add violations ...
    return violations

# After: Reuse list
class Detector:
    def __init__(self):
        self._violations = []
    
    def get_violations(self):
        self._violations.clear()
        # ... add violations ...
        return self._violations
```

### 7. Profile in Production

Enable performance monitoring:

```python
from nethical.performanceprofiling import PerformanceProfiler

profiler = PerformanceProfiler()
profiler.start_monitoring()

# Your application code

report = profiler.generate_report()
```

## Performance Targets

### Short-Term (6 Months)

- ✅ **100 sustained RPS**: Achieved
- ✅ **P95 < 200ms**: Achieved
- ✅ **P99 < 500ms**: Achieved

### Medium-Term (12 Months)

- ✅ **1,000 sustained RPS**: Target
- ✅ **P95 < 150ms**: Target
- ✅ **P99 < 400ms**: Target

### Long-Term (24 Months)

- 🎯 **10,000 sustained RPS**: Goal
- 🎯 **P95 < 100ms**: Goal
- 🎯 **P99 < 300ms**: Goal

## CI/CD Integration Details

### Workflow Jobs

#### 1. performance-benchmark

- Runs on every PR and main push
- Compares current branch to baseline
- Comments on PR with results
- Uploads artifacts

#### 2. memory-profiling

- Runs memory profiler
- Detects memory leaks
- Uploads profile reports

#### 3. benchmark-history

- Runs on main pushes only
- Tracks performance over time
- Stores historical data

### Artifacts Generated

- `performance-results/` - Benchmark CSV files
- `memory-profile.txt` - Memory profiling output
- `benchmark-history/` - Historical performance data

### Accessing Artifacts

```bash
# Download from GitHub Actions
gh run download <run-id>

# Or via web UI
# Actions > Workflow Run > Artifacts
```

## Troubleshooting

### High Latency

**Symptoms:**
- P50/P95 latency increased
- Throughput decreased

**Investigation:**
```bash
# Profile the code
python -m cProfile -o profile.stats app.py

# Check for blocking operations
# Look for I/O waits, database queries, external API calls
```

**Solutions:**
- Add caching for repeated operations
- Use async I/O for external calls
- Optimize database queries
- Add connection pooling

### Memory Leaks

**Symptoms:**
- Memory usage grows over time
- Application crashes with OOM

**Investigation:**
```bash
# Use memory profiler
python -m memory_profiler app.py

# Or use memray for detailed analysis
memray run app.py
memray flamegraph output.bin
```

**Solutions:**
- Clear caches periodically
- Close file handles/connections
- Use weak references for caches
- Implement object pooling

### Regression After Changes

**Symptoms:**
- CI/CD workflow flags regression
- Performance degraded in specific areas

**Investigation:**
```bash
# Compare code changes
git diff main..feature-branch

# Profile specific changes
python -m cProfile -o before.stats app.py  # Before changes
python -m cProfile -o after.stats app.py   # After changes
```

**Solutions:**
- Revert problematic changes
- Optimize new code
- Add performance tests for affected areas
- Document performance requirements

## Best Practices

### 1. Run Benchmarks Regularly

```bash
# Daily benchmark on main branch
# Automated in CI/CD

# Weekly full performance suite
# Manual trigger or scheduled workflow
```

### 2. Set Performance Budgets

```python
# Define maximum acceptable latencies
PERFORMANCE_BUDGET = {
    'p50': 50,   # 50ms
    'p95': 200,  # 200ms
    'p99': 500   # 500ms
}

# Fail tests if budget exceeded
assert p50_latency < PERFORMANCE_BUDGET['p50']
```

### 3. Monitor in Production

```python
# Use OpenTelemetry for production monitoring
from opentelemetry import trace, metrics

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

latency_histogram = meter.create_histogram(
    name="action_latency",
    description="Action processing latency",
    unit="ms"
)

with tracer.start_as_current_span("process_action"):
    start = time.time()
    result = process_action(action)
    latency = (time.time() - start) * 1000
    latency_histogram.record(latency)
```

### 4. Document Performance Requirements

```python
def process_action(action):
    """
    Process an action through governance.
    
    Performance Requirements:
    - Latency: < 50ms (P50), < 200ms (P95)
    - Throughput: 100 RPS per instance
    - Memory: < 100 MB for 1000 actions
    """
    pass
```

### 5. Review Performance on Every PR

- Check automated benchmark results
- Look for warning signs
- Profile locally if needed
- Optimize before merging

## Resources

### Tools

- `examples/perf/generate_load.py` - Load generator
- `nethical/performanceprofiling.py` - Performance profiler
- `scripts/compare_benchmarks.py` - Benchmark comparison (to be created)

### Documentation

- `docs/PERFORMANCE_OPTIMIZATION_GUIDE.md` - Optimization strategies
- `docs/PERFORMANCE_PROFILING_GUIDE.md` - Profiling instructions
- `docs/ops/PERFORMANCE_SIZING.md` - Capacity planning

### External Resources

- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [Profiling Python](https://docs.python.org/3/library/profile.html)
- [Memory Profiler](https://pypi.org/project/memory-profiler/)
- [OpenTelemetry](https://opentelemetry.io/)

---

**Last Updated:** November 5, 2025  
**Workflow Version:** 1.0.0  
**Performance Target:** SLSA Level 3 compliant
