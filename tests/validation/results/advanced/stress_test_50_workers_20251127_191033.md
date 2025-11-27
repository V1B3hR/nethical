# Advanced Validation Test Report: stress_test_50_workers

**Generated**: 2025-11-27T19:10:33.684866

## Summary

- **Duration**: 22.23 seconds
- **Total Requests**: 2,600
- **Successful**: 2,594
- **Failed**: 6
- **Success Rate**: 99.77%
- **Throughput**: 116.95 req/sec

## Latency Statistics

| Metric | Value (ms) |
|--------|------------|
| mean | 418.16 |
| median | 398.53 |
| min | 198.34 |
| max | 5105.90 |
| p50 | 398.55 |
| p90 | 479.98 |
| p95 | 492.51 |
| p99 | 542.41 |
| stddev | 129.51 |

## Memory Analysis

- Initial Memory: 107.51 MB
- Final Memory: 149.18 MB
- Growth: 41.68 MB (+38.77%)
- Leak Detected: ❌ YES

## Performance Degradation Analysis

- Early Mean Latency: 419.18 ms
- Late Mean Latency: 414.70 ms
- Degradation: -1.07%
- Degradation Detected: ✅ NO

## Test Result: ❌ FAILED

### Issues Detected

- P95 latency 492.51ms > 200ms
- Memory leak detected
