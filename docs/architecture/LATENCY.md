# Nethical Latency Engineering Guide

## Overview

Latency is critical for real-time AI applications, especially in robotics and safety-critical systems.

**Problem Statement:**
> 500ms latency spikes are DANGEROUS for robotics — robot can crash into wall!

This guide covers Nethical's latency engineering system designed to monitor, enforce, and optimize inference latency.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  LatencyMonitor  │ │ LatencyBudget   │ │ InferenceCache  │
│  (Tracking)      │ │ (Thresholds)    │ │ (Caching)       │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Latency Alert System                         │
│              (Callbacks, Logging, Failsafe Triggers)            │
└─────────────────────────────────────────────────────────────────┘
```

## Latency Budgets

### Default Budgets

Nethical provides pre-configured latency budgets for common use cases:

| Budget | Target | Warning | Critical | Max |
|--------|--------|---------|----------|-----|
| Robotics | 10ms | 50ms | 100ms | 200ms |
| Real-time | 20ms | 100ms | 200ms | 500ms |
| Interactive | 100ms | 300ms | 1000ms | 5000ms |
| Batch | 1000ms | 5000ms | 30000ms | 60000ms |

### Custom Budget

```python
from nethical.core.latency import LatencyBudget

# Custom budget for autonomous vehicles
av_budget = LatencyBudget(
    name="autonomous_vehicle",
    target_ms=5.0,      # Target: 5ms for perception
    warning_ms=20.0,    # Warning at 20ms
    critical_ms=50.0,   # Critical at 50ms
    max_ms=100.0,       # Hard limit: 100ms
)
```

### Budget Classification

```python
level = budget.classify(latency_ms=75.0)
# Returns: LatencyLevel.CRITICAL
```

## Latency Monitoring

### Basic Monitoring

```python
from nethical.core.latency import LatencyMonitor, ROBOTICS_BUDGET

monitor = LatencyMonitor(budget=ROBOTICS_BUDGET)

# Record measurements
monitor.record(latency_ms=8.5, operation="inference")
monitor.record(latency_ms=12.3, operation="inference")
monitor.record(latency_ms=150.0, operation="inference")  # Triggers alert!
```

### Alert Callbacks

```python
def on_latency_alert(alert):
    if alert.level == LatencyLevel.CRITICAL:
        trigger_failsafe_mode()
    elif alert.level == LatencyLevel.VIOLATION:
        emergency_stop()

monitor = LatencyMonitor(
    budget=ROBOTICS_BUDGET,
    alert_callback=on_latency_alert,
)
```

### Statistics

```python
# Get latency statistics
stats = monitor.get_stats()
print(f"p50: {stats.p50_ms}ms")
print(f"p99: {stats.p99_ms}ms")
print(f"Max: {stats.max_ms}ms")

# Time-windowed stats
recent_stats = monitor.get_stats(window_seconds=60.0)
```

### Health Status

```python
health = monitor.get_health_status()
print(f"Health: {health['health']}")  # 'healthy', 'warning', 'degraded', 'critical'
print(f"p99 within target: {health['p99_within_target']}")
```

## Inference Caching

For repeated patterns, caching can eliminate inference latency entirely:

```python
from nethical.core.latency import InferenceCache

cache = InferenceCache(
    max_size=1000,       # Cache up to 1000 entries
    ttl_seconds=300.0,   # 5 minute TTL
)

# Check cache before inference
cached_result = cache.get(inputs)
if cached_result is not None:
    return cached_result

# Perform inference
result = model(inputs)

# Store in cache
cache.put(inputs, result)
```

### Cache Statistics

```python
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Cache size: {stats['size']}/{stats['max_size']}")
```

## Decorators

### Automatic Latency Tracking

```python
from nethical.core.latency import latency_tracked

@latency_tracked(monitor, operation="model_inference")
def run_inference(inputs):
    return model(inputs)
```

### Latency Budget Enforcement

```python
from nethical.core.latency import with_latency_budget, ROBOTICS_BUDGET

def on_violation(latency_ms):
    print(f"Budget violated! Latency: {latency_ms}ms")

@with_latency_budget(ROBOTICS_BUDGET, on_violation=on_violation)
def critical_operation():
    # This operation will trigger callback if too slow
    return perform_inference()
```

## Optimization Strategies

### 1. Model Compilation

Pre-compile models to reduce cold-start latency:

```python
from nethical.core.accelerators import get_best_accelerator

accelerator = get_best_accelerator()
compiled_model = accelerator.compile_model(model, example_inputs)
```

### 2. Mixed Precision

Use FP16/BF16 for faster inference:

```python
config = AcceleratorConfig(
    mixed_precision=True,  # Enable FP16
)
```

### 3. Batch Accumulation

For throughput-critical workloads, accumulate requests into batches:

```python
# Instead of:
for input in inputs:
    model(input)  # High per-sample overhead

# Use:
accelerator.batch_execute(model, inputs, batch_size=32)
```

### 4. Inference Caching

Cache results for repeated patterns:

```python
cache = InferenceCache(max_size=1000)
result = cache.get(inputs) or model(inputs)
cache.put(inputs, result)
```

### 5. Async Execution

For non-blocking inference:

```python
config = AcceleratorConfig(async_execution=True)
accelerator = manager.create_accelerator(config)
```

## Fundamental Laws Alignment

The latency system aligns with Nethical's Fundamental Laws:

- **Law 21 (Primacy of Human Safety)**: Latency budgets ensure AI responds fast enough to protect humans
- **Law 23 (Fail-Safe Design)**: Critical latency violations trigger failsafe modes
- **Law 15 (Audit Compliance)**: All latency metrics are logged for audit

## Example: Robotics Safety System

```python
from nethical.core.latency import (
    LatencyMonitor,
    LatencyBudget,
    LatencyLevel,
)
from nethical.core.accelerators import get_best_accelerator

# Configure for robotics safety
budget = LatencyBudget(
    name="robot_perception",
    target_ms=10.0,
    warning_ms=30.0,
    critical_ms=50.0,
    max_ms=100.0,
)

# Alert handler with safety actions
def safety_handler(alert):
    if alert.level == LatencyLevel.VIOLATION:
        robot.emergency_stop()
        log.critical(f"Emergency stop: {alert.message}")
    elif alert.level == LatencyLevel.CRITICAL:
        robot.reduce_speed(factor=0.5)
        log.warning(f"Speed reduced: {alert.message}")

# Initialize monitoring
monitor = LatencyMonitor(
    budget=budget,
    alert_callback=safety_handler,
)

# Get accelerator
accelerator = get_best_accelerator()

# Main perception loop
while robot.is_running():
    start = time.perf_counter()
    
    # Get sensor data
    camera_frame = robot.get_camera_frame()
    
    # Run perception model
    inputs = preprocess(camera_frame)
    detections = accelerator.execute(perception_model, inputs)
    accelerator.synchronize()
    
    # Record latency
    latency_ms = (time.perf_counter() - start) * 1000
    monitor.record(latency_ms, operation="perception")
    
    # Use detections
    robot.update_world_model(detections)
```

## Metrics and Dashboards

### Available Metrics

| Metric | Description |
|--------|-------------|
| `latency_p50_ms` | Median latency |
| `latency_p99_ms` | 99th percentile latency |
| `latency_max_ms` | Maximum observed latency |
| `warning_count` | Number of warning alerts |
| `critical_count` | Number of critical alerts |
| `violation_count` | Number of budget violations |

### Prometheus Export

```python
def export_prometheus_metrics(monitor):
    stats = monitor.get_stats()
    health = monitor.get_health_status()
    
    return {
        "nethical_latency_p50_ms": stats.p50_ms,
        "nethical_latency_p99_ms": stats.p99_ms,
        "nethical_latency_violations_total": health["counters"]["violations"],
    }
```

## Troubleshooting

### High p99 Latency

1. Check for GC pauses in Python
2. Verify accelerator is properly initialized
3. Consider reducing batch size
4. Enable model compilation

### Intermittent Spikes

1. Check for CPU thermal throttling
2. Monitor GPU memory pressure
3. Look for competing processes
4. Consider dedicated hardware

### Cache Not Helping

1. Verify input patterns are repeating
2. Check TTL settings
3. Increase cache size
4. Profile cache hit rate
