# Performance Profiling Guide

This guide covers the performance profiling and benchmarking capabilities in Nethical.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Features](#features)
4. [API Reference](#api-reference)
5. [Benchmarking](#benchmarking)
6. [Regression Testing](#regression-testing)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

## Overview

Nethical includes a comprehensive performance profiling system that helps you:

- **Profile Functions**: Measure execution time and resource usage
- **Benchmark Performance**: Run functions multiple times to get statistical metrics
- **Detect Regressions**: Automatically detect when performance degrades
- **Generate Reports**: Create detailed performance reports

The profiling system is designed for:
- Development and debugging
- Performance optimization
- Continuous integration testing
- Production monitoring

## Quick Start

### Basic Profiling

```python
from nethical.performanceprofiling import profile

@profile()
def my_function():
    # Your code here
    result = sum(range(1000000))
    return result

# Function is automatically profiled each time it runs
result = my_function()
```

### Using a Custom Profiler

```python
from nethical.performanceprofiling import PerformanceProfiler

# Create profiler instance
profiler = PerformanceProfiler(results_dir="profiling_results")

@profiler.profile()
def my_function():
    return sum(range(1000000))

# Run function
my_function()

# Get results
results = profiler.get_results('my_function')
print(f"Mean execution time: {results['mean_ms']:.2f}ms")

# Generate report
profiler.generate_report("performance_report.txt")
```

## Features

### 1. Function Profiling

Profile individual functions with minimal overhead:

```python
@profiler.profile()
def basic_function():
    return 42

@profiler.profile(name="custom_name")
def renamed_function():
    return 42

@profiler.profile(detailed=True)
def detailed_function():
    # Detailed profiling includes call stack information
    return sum(range(10000))
```

### 2. Benchmarking

Run functions multiple times to get statistical metrics:

```python
def function_to_benchmark():
    return sorted([random.random() for _ in range(1000)])

stats = profiler.benchmark_function(
    function_to_benchmark,
    iterations=100
)

print(f"Mean: {stats['mean_ms']:.2f}ms")
print(f"Median: {stats['median_ms']:.2f}ms")
print(f"P95: {stats['p95_ms']:.2f}ms")
print(f"P99: {stats['p99_ms']:.2f}ms")
```

### 3. Performance Baselines

Set performance baselines for regression testing:

```python
# Run initial benchmark
stats = profiler.benchmark_function(my_function, iterations=100)

# Set baseline with 20% tolerance
profiler.set_benchmark(
    name="my_function",
    baseline_ms=stats['median_ms'],
    tolerance_pct=20.0
)

# Future runs will automatically check for regressions
@profiler.profile(name="my_function")
def my_function():
    # If this takes >20% longer than baseline, a warning is printed
    return sum(range(1000000))
```

### 4. Regression Detection

Automatic detection of performance regressions:

```python
# Set a benchmark
profiler.set_benchmark("critical_function", baseline_ms=10.0, tolerance_pct=10.0)

@profiler.profile(name="critical_function")
def critical_function():
    time.sleep(0.015)  # 15ms - exceeds 10ms baseline by >10%

critical_function()
# Output: [WARNING] Performance regression detected for critical_function: 
#         50.0% slower than baseline (15.00ms vs 10.00ms)
```

### 5. Performance Reports

Generate comprehensive reports:

```python
report = profiler.generate_report(output_file="report.txt")
print(report)
```

Example output:
```
================================================================================
PERFORMANCE PROFILING REPORT
================================================================================
Generated: 2024-12-10T10:30:00

BENCHMARKS
--------------------------------------------------------------------------------
  evaluate_action:
    Baseline: 25.50ms
    Tolerance: ±20%
    Recent Avg: 26.30ms
    Status: OK (+3.1%)

  process_batch:
    Baseline: 150.00ms
    Tolerance: ±15%
    Recent Avg: 165.00ms
    Status: REGRESSION (+10.0%)

PROFILING RESULTS
--------------------------------------------------------------------------------
  evaluate_action:
    Samples: 100
    Mean: 26.30ms
    Median: 26.00ms
    Min: 24.50ms
    Max: 28.90ms

  process_batch:
    Samples: 50
    Mean: 165.00ms
    Median: 164.50ms
    Min: 158.00ms
    Max: 172.00ms
```

## API Reference

### PerformanceProfiler

Main profiler class for performance tracking.

#### Constructor

```python
PerformanceProfiler(results_dir: Union[str, Path] = "profiling_results")
```

**Parameters:**
- `results_dir`: Directory to store profiling results and benchmarks

#### Methods

##### `profile(func=None, *, name=None, detailed=False)`

Decorator to profile a function.

**Parameters:**
- `func`: Function to profile (or None if used as @profile())
- `name`: Custom name for the function (optional)
- `detailed`: Enable detailed profiling with call stack (default: False)

**Returns:** Decorated function

**Example:**
```python
@profiler.profile()
def my_func():
    pass

@profiler.profile(name="custom", detailed=True)
def detailed_func():
    pass
```

##### `benchmark_function(func, iterations=100, **kwargs)`

Benchmark a function by running it multiple times.

**Parameters:**
- `func`: Function to benchmark
- `iterations`: Number of times to run the function
- `**kwargs`: Arguments to pass to the function

**Returns:** Dictionary with statistics (mean_ms, median_ms, std_ms, min_ms, max_ms, p95_ms, p99_ms)

##### `set_benchmark(name, baseline_ms, tolerance_pct=10.0)`

Set a performance benchmark for regression testing.

**Parameters:**
- `name`: Name of the benchmark
- `baseline_ms`: Baseline execution time in milliseconds
- `tolerance_pct`: Acceptable performance degradation percentage (default: 10%)

##### `get_results(function_name=None)`

Get profiling results.

**Parameters:**
- `function_name`: Name of function to get results for (optional, returns all if None)

**Returns:** Dictionary with statistics

##### `generate_report(output_file=None)`

Generate a performance report.

**Parameters:**
- `output_file`: Optional file path to save report (Path or string)

**Returns:** Report as string

### ProfileResult

Dataclass representing a single profiling result.

**Attributes:**
- `function_name`: Name of the profiled function
- `execution_time_ms`: Execution time in milliseconds
- `call_count`: Number of calls
- `timestamp`: Timestamp of the profile
- `stats`: Optional detailed statistics
- `metadata`: Additional metadata

### Benchmark

Dataclass representing a performance benchmark.

**Attributes:**
- `name`: Benchmark name
- `baseline_ms`: Baseline execution time
- `tolerance_pct`: Tolerance percentage
- `samples`: List of recent samples

**Methods:**
- `check_regression(current_ms)`: Check if current time represents a regression
- `update_baseline(new_baseline_ms)`: Update the baseline
- `add_sample(sample_ms)`: Add a performance sample

### Global Functions

#### `profile(func=None, *, name=None, detailed=False)`

Convenience decorator using the global profiler instance.

```python
from nethical.performanceprofiling import profile

@profile()
def my_function():
    pass
```

#### `get_profiler()`

Get the global profiler instance.

```python
from nethical.performanceprofiling import get_profiler

profiler = get_profiler()
results = profiler.get_results()
```

## Benchmarking

### Running Benchmarks

```python
from nethical.performanceprofiling import PerformanceProfiler

profiler = PerformanceProfiler()

def my_function(n):
    return sum(range(n))

# Benchmark with different inputs
for n in [1000, 10000, 100000]:
    stats = profiler.benchmark_function(
        my_function,
        iterations=100,
        n=n
    )
    print(f"n={n}: mean={stats['mean_ms']:.2f}ms, "
          f"p95={stats['p95_ms']:.2f}ms")
```

### Setting Baselines

```python
# Initial benchmark to establish baseline
stats = profiler.benchmark_function(critical_function, iterations=200)

# Set conservative baseline (use median to avoid outliers)
profiler.set_benchmark(
    name="critical_function",
    baseline_ms=stats['median_ms'],
    tolerance_pct=15.0  # Allow 15% degradation
)
```

## Regression Testing

### Integration with pytest

```python
# tests/test_performance.py
import pytest
from nethical.performanceprofiling import PerformanceProfiler

profiler = PerformanceProfiler(results_dir="tests/performance/results")

@pytest.fixture
def system_under_test():
    # Setup code
    return MySystem()

class TestPerformance:
    def test_function_performance(self, system_under_test):
        """Test that function meets performance requirements"""
        stats = profiler.benchmark_function(
            system_under_test.critical_operation,
            iterations=50
        )
        
        # Assert performance requirements
        assert stats['mean_ms'] < 100, \
            f"Function too slow: {stats['mean_ms']:.2f}ms"
        assert stats['p95_ms'] < 150, \
            f"P95 too slow: {stats['p95_ms']:.2f}ms"
```

### Continuous Integration

```python
# ci_benchmark.py
from nethical.performanceprofiling import PerformanceProfiler

profiler = PerformanceProfiler()

# Run benchmarks
for func_name, func in critical_functions.items():
    stats = profiler.benchmark_function(func, iterations=100)
    profiler.set_benchmark(func_name, stats['median_ms'], tolerance_pct=10.0)

# Generate report
report = profiler.generate_report("ci_performance_report.txt")

# Check for regressions
if "REGRESSION" in report:
    print("❌ Performance regressions detected!")
    sys.exit(1)
else:
    print("✅ All performance benchmarks passed")
```

## Best Practices

### 1. Profile Early and Often

Profile during development to catch performance issues early:

```python
@profile()
def new_feature():
    # Your code here
    pass
```

### 2. Use Appropriate Iterations

- Development: 10-50 iterations for quick feedback
- CI: 100-200 iterations for stability
- Production baselines: 500-1000 iterations

### 3. Set Realistic Tolerances

- Fast functions (<10ms): 20-30% tolerance
- Medium functions (10-100ms): 15-20% tolerance
- Slow functions (>100ms): 10-15% tolerance

### 4. Profile with Representative Data

```python
# Bad: Profiling with empty data
stats = profiler.benchmark_function(lambda: process_data([]))

# Good: Profiling with realistic data
test_data = load_representative_dataset()
stats = profiler.benchmark_function(
    lambda: process_data(test_data),
    iterations=100
)
```

### 5. Track Performance Over Time

```python
# Save benchmark results with metadata
stats = profiler.benchmark_function(my_function, iterations=100)
profiler.set_benchmark(
    name=f"my_function_v{version}",
    baseline_ms=stats['median_ms'],
    tolerance_pct=15.0
)
```

## Examples

### Example 1: Basic Profiling

```python
from nethical.performanceprofiling import PerformanceProfiler

profiler = PerformanceProfiler()

@profiler.profile()
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Profile the function
result = fibonacci(10)

# Get results
results = profiler.get_results('fibonacci')
print(f"Fibonacci executed in {results['mean_ms']:.2f}ms")
```

### Example 2: Comparing Implementations

```python
profiler = PerformanceProfiler()

@profiler.profile(name="implementation_a")
def implementation_a(data):
    return sorted(data)

@profiler.profile(name="implementation_b")
def implementation_b(data):
    data_copy = data.copy()
    data_copy.sort()
    return data_copy

# Test both implementations
test_data = [random.random() for _ in range(10000)]

for _ in range(100):
    implementation_a(test_data.copy())
    implementation_b(test_data.copy())

# Compare results
results_a = profiler.get_results('implementation_a')
results_b = profiler.get_results('implementation_b')

print(f"Implementation A: {results_a['mean_ms']:.2f}ms")
print(f"Implementation B: {results_b['mean_ms']:.2f}ms")

if results_a['mean_ms'] < results_b['mean_ms']:
    print("Implementation A is faster!")
else:
    print("Implementation B is faster!")
```

### Example 3: Monitoring Production Performance

```python
from nethical.performanceprofiling import profile

@profile(detailed=True)
async def process_request(request):
    # Process the request
    result = await handle_request(request)
    return result

# In production, periodically check performance
profiler = get_profiler()

async def performance_monitor():
    while True:
        await asyncio.sleep(3600)  # Check every hour
        
        results = profiler.get_results('process_request')
        if results:
            if results['mean_ms'] > 100:  # Alert threshold
                send_alert(f"High latency detected: {results['mean_ms']:.2f}ms")
```

### Example 4: Regression Testing in CI

```python
# test_performance_regression.py
import pytest
from nethical.performanceprofiling import PerformanceProfiler

profiler = PerformanceProfiler(results_dir="tests/performance")

class TestPerformanceRegression:
    def test_no_regression(self):
        """Ensure performance hasn't regressed"""
        from myapp import critical_function
        
        stats = profiler.benchmark_function(
            critical_function,
            iterations=100
        )
        
        # Check against baseline
        profiler.set_benchmark(
            "critical_function",
            baseline_ms=50.0,  # Established baseline
            tolerance_pct=15.0
        )
        
        # Assert performance
        assert stats['mean_ms'] < 57.5, \
            f"Performance regression: {stats['mean_ms']:.2f}ms > 57.5ms"
```

## Performance Characteristics

### Profiler Overhead

The profiling system is designed to have minimal overhead:

- **Simple profiling**: <1% overhead
- **Detailed profiling**: <5% overhead
- **Benchmark mode**: No overhead (measures actual execution)

### Storage

- Benchmarks are stored in JSON format (~1KB per benchmark)
- Profiling results are kept in memory (can be persisted if needed)
- Reports are generated on-demand

### Thread Safety

The profiler is thread-safe for:
- Multiple threads profiling different functions
- Concurrent benchmark operations

Not recommended for:
- Profiling the same function from multiple threads simultaneously

## Troubleshooting

### Issue: Inconsistent Results

**Problem:** Benchmark results vary significantly between runs.

**Solutions:**
1. Increase iterations (try 200-500)
2. Disable other processes
3. Use median instead of mean
4. Add warm-up iterations

### Issue: High Overhead

**Problem:** Profiling slows down the application significantly.

**Solutions:**
1. Use simple profiling instead of detailed
2. Profile selectively (not every function)
3. Disable profiling in production
4. Use sampling instead of continuous profiling

### Issue: Missing Results

**Problem:** No profiling results available.

**Solutions:**
1. Ensure functions are actually called
2. Check results directory permissions
3. Verify profiler instance is correct
4. Check for exceptions in profiled functions

## Related Documentation

- [Testing Guide](TESTING_GUIDE.md)
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md)
- [CI/CD Integration](CI_CD_INTEGRATION.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical/docs
