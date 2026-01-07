# Performance Optimization

This document describes the performance optimization techniques used in Nethical's ultra-low latency threat detection system.

## Overview

The optimization framework achieves production-grade performance through:
- **Model Optimization:** Quantization, pruning, knowledge distillation
- **Request Optimization:** LRU caching, dynamic batching, request coalescing
- **Architectural Optimization:** Async I/O, parallel execution, minimal allocations

## Performance Targets

```python
TARGETS = {
    "shadow_ai": 20,          # ms
    "deepfake": 30,           # ms
    "polymorphic": 50,        # ms
    "prompt_injection": 15,   # ms
    "ai_vs_ai": 25,          # ms
    "unified_average": 50,    # ms
}

LOAD_TARGETS = {
    "throughput": 5000,       # requests/second
    "avg_latency": 50,        # ms
    "p95_latency": 100,       # ms
    "p99_latency": 200,       # ms
}
```

## Model Optimization

### 1. Quantization

Convert model weights from FP32 to INT8 or INT4 for faster inference.

```python
from nethical.optimization import ModelOptimizer

# Quantize to INT8
optimized_model = ModelOptimizer.optimize_for_latency(
    model=original_model,
    target_latency_ms=50,
    quantization_bits=8,
)
```

**Benefits:**
- 4x smaller model size (INT8 vs FP32)
- 2-4x faster inference
- Minimal accuracy loss (<1% typically)

### 2. Model Pruning

Remove 30% of least important weights to reduce computation.

```python
# Prune 30% of weights
optimized_model = ModelOptimizer.optimize_for_latency(
    model=original_model,
    pruning_ratio=0.3,
)
```

**Benefits:**
- 30% fewer operations
- 20-30% faster inference
- Negligible accuracy impact

### 3. ONNX Runtime Conversion

Convert models to ONNX format for optimized inference.

```python
# Convert to ONNX
optimized_model = ModelOptimizer.optimize_for_latency(
    model=original_model,
    use_onnx=True,
)
```

**Benefits:**
- Cross-platform optimization
- Hardware-specific acceleration
- Reduced inference time by 2-3x

### 4. Knowledge Distillation

Train smaller student models from larger teacher models.

```python
# Distill knowledge from teacher to student
student_model = ModelOptimizer.knowledge_distillation(
    teacher_model=large_model,
    student_model=small_model,
    training_data=data,
    temperature=3.0,
    alpha=0.5,
)
```

**Benefits:**
- 5-10x smaller models
- 90-95% of teacher accuracy
- Significant latency reduction

## Request Optimization

### 1. LRU Cache

Cache recent query results for instant retrieval.

```python
from nethical.optimization import RequestOptimizer

# Create optimizer with 10k cache
optimizer = RequestOptimizer(cache_maxsize=10000)

# Process with caching
results = await optimizer.process(
    requests=[req1, req2, req3],
    processor=detector.detect,
    enable_cache=True,
)

# Check metrics
metrics = optimizer.get_metrics()
# {
#     "cache_hit_rate": 0.45,
#     "cache_size": 8234,
#     ...
# }
```

**Benefits:**
- Near-zero latency for cached queries
- Reduces backend load
- Typical hit rates: 30-50%

**Configuration:**
- **Cache Size:** 10,000 entries (configurable)
- **Eviction:** LRU (Least Recently Used)
- **Key:** MD5 hash of request

### 2. Dynamic Batching

Automatically batch requests for throughput optimization.

```python
from nethical.optimization import DynamicBatcher, DynamicBatcherConfig

# Configure batcher
config = DynamicBatcherConfig(
    max_batch_size=32,
    timeout_ms=10.0,
)

batcher = DynamicBatcher(config)

# Requests are automatically batched
result = await batcher.add_request(request, processor)
```

**How it works:**
1. Requests accumulate up to `max_batch_size`
2. Batch processes after `timeout_ms` or when full
3. Results distributed to individual futures

**Benefits:**
- 2-4x higher throughput
- Efficient GPU/hardware utilization
- Automatic tuning

**Configuration:**
- **Max Batch Size:** 32 (optimal for most hardware)
- **Timeout:** 10ms (balance latency vs throughput)

### 3. Request Coalescing

Deduplicate concurrent identical requests.

```python
from nethical.optimization import RequestCoalescer

coalescer = RequestCoalescer()

# Multiple identical requests share result
result = await coalescer.coalesce(request, processor)
```

**Benefits:**
- Eliminates redundant computation
- Reduces peak load
- Saves 10-20% processing under high load

## Architectural Optimizations

### 1. Async/Await

All I/O operations are non-blocking.

```python
# Good: Non-blocking
async def detect_violations(self, context):
    tasks = [
        self._detect_api(),
        self._detect_gpu(),
        self._detect_files(),
    ]
    results = await asyncio.gather(*tasks)

# Bad: Blocking
def detect_violations(self, context):
    result1 = self._detect_api()
    result2 = self._detect_gpu()
    result3 = self._detect_files()
```

**Benefits:**
- Concurrent execution
- Efficient CPU utilization
- Scales to 10,000+ concurrent requests

### 2. Parallel Detection

Multiple detection methods run concurrently.

```python
# Shadow AI detector runs 4 methods in parallel
detection_tasks = [
    self._detect_api_calls(traffic),
    self._detect_gpu_usage(system),
    self._detect_model_files(files),
    self._detect_suspicious_ports(traffic),
]
results = await asyncio.gather(*detection_tasks)
```

**Benefits:**
- 2-4x faster than sequential
- Exploits multi-core CPUs
- Reduces tail latency

### 3. Lightweight Pattern Matching

Use compiled regex for fast screening.

```python
# Pre-compiled patterns (done once)
PATTERNS = {
    "openai": re.compile(r"api\.openai\.com/v1/..."),
    ...
}

# Fast matching
for provider, pattern in PATTERNS.items():
    if pattern.search(url):
        # Match found
```

**Benefits:**
- Regex ~1000x faster than ML models
- Tier-1 screening in 2-5ms
- Fallback to ML only when needed

### 4. Memory Efficiency

Minimize allocations in hot paths.

```python
# Good: Reuse structures
self._query_history = deque(maxlen=1000)  # Fixed size

# Bad: Unbounded growth
self._query_history = []  # Grows indefinitely
```

**Benefits:**
- Predictable memory usage
- No GC pauses
- Consistent latency

## Latency Breakdown

Typical latency for each component:

### Shadow AI Detector (<20ms target)
```
API Detection:      2-5ms   (regex patterns)
GPU Detection:      3-6ms   (system metrics)
File Detection:     4-7ms   (filesystem scan)
Port Scan:          2-4ms   (network check)
---
Total:              11-22ms (parallel execution)
```

### Prompt Injection Guard (<15ms target)
```
Tier 1 (Regex):     2-5ms   (fast patterns)
Tier 2 (ML):        10-18ms (if triggered)
---
Total:              2-18ms  (conditional)
```

### Deepfake Detector (<30ms target)
```
Frequency Analysis: 5-8ms   (FFT)
Metadata Check:     1-2ms   (header parsing)
Face Landmarks:     8-12ms  (lightweight model)
CNN Inference:      10-15ms (quantized)
---
Total:              24-37ms (parallel execution)
```

## Profiling

### Profile Individual Detector

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run detector
await detector.detect_violations(context)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

### Profile with py-spy

```bash
# Install py-spy
pip install py-spy

# Profile running process
py-spy record -o profile.svg --pid <PID>

# Profile script
py-spy record -o profile.svg -- python benchmark.py
```

### Generate Flamegraph

```bash
# With py-spy
py-spy record -o flamegraph.svg -f speedscope -- python benchmark.py

# View in browser
open flamegraph.svg
```

## Optimization Workflow

1. **Benchmark baseline:**
   ```bash
   python tests/benchmark_individual_detectors.py
   ```

2. **Identify bottlenecks:**
   ```bash
   py-spy record -o profile.svg -- python benchmark.py
   ```

3. **Optimize hot paths:**
   - Replace Python loops with numpy
   - Use compiled regex
   - Enable caching

4. **Re-benchmark:**
   ```bash
   python tests/benchmark_individual_detectors.py
   ```

5. **Load test:**
   ```bash
   python tests/benchmark_latency.py
   ```

## Production Recommendations

### Hardware

**CPU:**
- 8+ cores for concurrent detection
- AVX2 support for ONNX Runtime
- High single-thread performance

**Memory:**
- 8GB minimum
- 16GB recommended for caching

**GPU (Optional):**
- CUDA-capable for ML inference
- 8GB+ VRAM for batch processing

### Configuration

```python
from nethical.detectors.realtime import RealtimeThreatDetectorConfig
from nethical.optimization import RequestOptimizer, DynamicBatcherConfig

# Optimized config
detector_config = RealtimeThreatDetectorConfig(
    parallel_detection=True,
    max_latency_ms=50,
)

batcher_config = DynamicBatcherConfig(
    max_batch_size=32,
    timeout_ms=10,
)

optimizer = RequestOptimizer(
    cache_maxsize=10000,
    batcher_config=batcher_config,
)
```

### Monitoring

Track these metrics:

```python
# Get metrics
metrics = detector.get_metrics()

# Alert on:
if metrics["p95_latency_ms"] > 100:
    alert("P95 latency exceeded")

if metrics["avg_latency_ms"] > 50:
    alert("Average latency exceeded")

# Log periodically
logger.info(f"Detector metrics: {metrics}")
```

### Scaling

**Horizontal Scaling:**
- Deploy multiple detector instances
- Load balance with nginx/envoy
- Use Redis for shared cache

**Vertical Scaling:**
- Increase CPU cores
- Add GPU for ML inference
- Expand cache size

**Edge Deployment:**
- Deploy at edge for lowest latency
- Use CDN for static models
- Sync state with central

## Troubleshooting

### High Latency

1. **Check hot paths:**
   ```bash
   py-spy top --pid <PID>
   ```

2. **Verify async usage:**
   - All I/O must be async
   - Use `asyncio.gather()` for parallel

3. **Check cache hit rate:**
   ```python
   metrics = optimizer.get_metrics()
   if metrics["cache_hit_rate"] < 0.3:
       # Increase cache size
   ```

### Low Throughput

1. **Enable batching:**
   ```python
   optimizer = RequestOptimizer(
       batcher_config=DynamicBatcherConfig(
           max_batch_size=32,
       )
   )
   ```

2. **Increase parallelism:**
   ```python
   detector = RealtimeThreatDetector(
       config=RealtimeThreatDetectorConfig(
           parallel_detection=True,
       )
   )
   ```

3. **Profile bottlenecks:**
   ```bash
   py-spy record -o profile.svg -- python benchmark.py
   ```

## Performance Testing

### Unit Tests

```bash
# Test individual detector performance
pytest tests/detectors/realtime/test_shadow_ai_detector.py::test_performance_target
```

### Benchmarks

```bash
# Individual detector benchmarks
python tests/benchmark_individual_detectors.py

# Load testing
python tests/benchmark_latency.py
```

### Continuous Monitoring

```python
# Add to CI/CD
def test_performance_regression():
    detector = RealtimeThreatDetector()

    # Benchmark
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        await detector.evaluate_threat(data, "all")
        latencies.append((time.perf_counter() - start) * 1000)

    p95 = sorted(latencies)[95]
    assert p95 < 100, f"P95 latency {p95}ms exceeds target"
```

## Best Practices

1. **Always use async/await**
2. **Enable caching for production**
3. **Monitor P95/P99 latencies**
4. **Profile before optimizing**
5. **Test under load**
6. **Use ONNX Runtime for ML models**
7. **Set resource limits**
8. **Implement circuit breakers**

## References

- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/)
- [Python Async Best Practices](https://docs.python.org/3/library/asyncio.html)
- [LRU Cache Implementation](https://docs.python.org/3/library/functools.html#functools.lru_cache)
- [Profiling with py-spy](https://github.com/benfred/py-spy)
