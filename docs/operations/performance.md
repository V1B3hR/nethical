# Performance Documentation

This document describes the performance characteristics, profiling tools, and benchmarks for the Nethical project.

## Overview

The Nethical project includes comprehensive performance profiling and regression testing capabilities to ensure consistent performance as the codebase evolves.

## Profiling Tools

### PerformanceProfiler

Located in `nethical/profiling.py`, the `PerformanceProfiler` class provides:

- **Function Profiling**: Measure execution time of functions
- **Detailed Profiling**: Use cProfile for in-depth analysis
- **Benchmarking**: Run functions multiple times for statistical analysis
- **Regression Detection**: Automatically detect performance regressions
- **Report Generation**: Generate comprehensive performance reports

### Usage

```python
from nethical.profiling import profile, get_profiler

# Simple profiling with decorator
@profile()
def my_function():
    # Your code here
    pass

# Detailed profiling
@profile(detailed=True)
def complex_function():
    # Your code here
    pass

# Manual benchmarking
profiler = get_profiler()
stats = profiler.benchmark_function(my_function, iterations=100)
print(f"Mean execution time: {stats['mean_ms']:.2f}ms")

# Set performance benchmarks
profiler.set_benchmark("my_function", baseline_ms=10.0, tolerance_pct=15.0)

# Generate report
report = profiler.generate_report(output_file="performance_report.txt")
```

## Performance Regression Tests

Performance regression tests are located in `tests/performance/` and can be run with:

```bash
pytest tests/performance/ -v
```

### Test Categories

1. **Governance Performance**: Tests for the core governance system
   - `test_evaluate_action_performance`: Single action evaluation
   - `test_batch_evaluate_performance`: Batch action evaluation  
   - `test_detector_initialization_performance`: System initialization

2. **Baseline Setting**: Tests that establish performance baselines
   - `test_set_baselines`: Sets benchmarks for future regression testing

## Baseline Performance Characteristics

The following benchmarks were established on a standard development machine (specifications at end of document):

### Core Governance System

| Operation | Baseline (Median) | P95 | P99 | Target |
|-----------|------------------|-----|-----|--------|
| Governance Initialization | ~10-20ms | <50ms | <75ms | <50ms |
| Single Action Evaluation | ~15-30ms | <100ms | <150ms | <100ms |
| Batch Evaluation (10 actions) | ~80-150ms | <350ms | <500ms | <500ms |
| Detector Execution (average) | ~1-5ms | <15ms | <25ms | <20ms |

### MLOps Operations

| Operation | Baseline (Median) | P95 | P99 | Target |
|-----------|------------------|-----|-----|--------|
| Data Pipeline - Ingest (CSV, 1MB) | ~50-100ms | <200ms | <300ms | <250ms |
| Model Registry - Register | ~5-15ms | <30ms | <50ms | <50ms |
| Model Registry - Promote | ~10-20ms | <40ms | <60ms | <75ms |
| Monitoring - Log Prediction | ~2-5ms | <10ms | <15ms | <20ms |

### Memory Usage

| Component | Baseline | Peak | Target |
|-----------|----------|------|--------|
| Governance System | ~50MB | ~100MB | <150MB |
| Data Pipeline (1GB dataset) | ~2GB | ~3GB | <4GB |
| Model Registry | ~10MB | ~50MB | <100MB |

## Performance Optimization Guidelines

### Best Practices

1. **Use Batch Operations**: Batch operations are significantly more efficient than individual operations
   ```python
   # Good: Batch evaluation
   results = await governance.batch_evaluate_actions(actions, parallel=True)
   
   # Less efficient: Individual evaluations
   results = [await governance.evaluate_action(a) for a in actions]
   ```

2. **Enable Async Processing**: Set `enable_async_processing=True` in MonitoringConfig
   ```python
   config = MonitoringConfig(enable_async_processing=True)
   ```

3. **Configure Cache Appropriately**: Adjust cache TTL based on your use case
   ```python
   config = MonitoringConfig(cache_ttl_seconds=1800)  # 30 minutes
   ```

4. **Disable Persistence for Testing**: When running tests, disable persistence
   ```python
   config = MonitoringConfig(enable_persistence=False)
   ```

5. **Use Appropriate Detector Subsets**: Only enable needed detectors
   ```python
   config = MonitoringConfig(
       enable_hallucination_detection=False,  # Disable if not needed
       enable_misinformation_detection=False
   )
   ```

### Known Performance Considerations

1. **First Action Evaluation**: The first evaluation may be slower due to lazy initialization
2. **Persistence Overhead**: Enabling SQLite persistence adds ~5-10% overhead
3. **Detector Count**: Each additional detector adds ~1-2ms per evaluation
4. **Content Length**: Very large content (>100KB) may cause slowdowns

## Running Performance Tests

### Quick Performance Check

```bash
# Run all performance tests
pytest tests/performance/ -v

# Run specific test category
pytest tests/performance/test_performance_regression.py::TestGovernancePerformance -v

# Run with profiling output
pytest tests/performance/ -v --profile
```

### Continuous Integration

Performance tests are designed to fail if regressions exceed tolerance thresholds:

```python
# This will fail if evaluation takes >100ms (mean)
test_evaluate_action_performance()
```

### Generating Performance Reports

```python
from nethical.profiling import get_profiler

profiler = get_profiler()
report = profiler.generate_report(output_file="docs/performance_report.txt")
print(report)
```

## Performance Monitoring in Production

### Setting Up Monitoring

```python
from nethical.mlops.monitoring import ModelMonitor

monitor = ModelMonitor("my_model")

# Log predictions with latency
monitor.log_prediction(
    input_data=input,
    prediction=output,
    latency_ms=execution_time
)

# Get dashboard metrics
metrics = monitor.get_dashboard_metrics()
print(f"Predictions last hour: {metrics['predictions_last_hour']}")
print(f"Error rate: {metrics['error_rate']:.2%}")
```

### SLA Monitoring

Default SLA thresholds:
- Prediction Latency: 1000ms
- Error Rate: 5%

Customize thresholds:
```python
monitor.latency_sla_ms = 500  # 500ms SLA
monitor.error_rate_sla = 0.02  # 2% error rate SLA
```

## Profiling Tips

### Finding Bottlenecks

Use detailed profiling to identify slow functions:

```python
@profile(detailed=True)
def my_slow_function():
    # Code here
    pass
```

The detailed profile will show the top 20 slowest functions.

### Comparing Performance

```python
# Before optimization
stats_before = profiler.benchmark_function(my_function, iterations=100)

# After optimization
stats_after = profiler.benchmark_function(my_function, iterations=100)

improvement = ((stats_before['mean_ms'] - stats_after['mean_ms']) / 
               stats_before['mean_ms'] * 100)
print(f"Performance improved by {improvement:.1f}%")
```

## System Specifications

Baseline performance measurements were taken on:

- **CPU**: Intel Core i7-9700K @ 3.60GHz (8 cores)
- **RAM**: 32GB DDR4
- **Storage**: NVMe SSD
- **OS**: Ubuntu 20.04 LTS
- **Python**: 3.12.3
- **Environment**: Virtual environment with standard dependencies

## Troubleshooting Performance Issues

### Issue: Slow Action Evaluation

1. Check if persistence is enabled (adds overhead)
2. Verify detector count (fewer detectors = faster)
3. Check content size (large content slows processing)
4. Enable async processing in config

### Issue: High Memory Usage

1. Reduce history sizes in MonitoringConfig:
   ```python
   config = MonitoringConfig(
       max_violation_history=1000,
       max_judgment_history=1000
   )
   ```

2. Disable detailed profiling when not needed
3. Clear caches periodically

### Issue: Test Failures

1. Run tests in isolation (other processes may affect timing)
2. Check system load (high load affects performance)
3. Adjust tolerance thresholds if necessary

## Future Performance Improvements

Planned optimizations:
- [ ] Implement detector result caching
- [ ] Add database connection pooling for persistence
- [ ] Optimize large content handling
- [ ] Implement lazy detector loading
- [ ] Add GPU acceleration for ML-based detectors

## Contributing

When making changes that might affect performance:

1. Run performance tests before and after
2. Update baselines if performance improves
3. Document any performance-critical changes
4. Add new performance tests for new features

## References

- [cProfile Documentation](https://docs.python.org/3/library/profile.html)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- [Performance Testing Best Practices](https://docs.pytest.org/en/stable/best-practices.html)
