# Benchmark Scripts

This directory contains performance benchmark scripts for the Nethical governance system.

## Purpose

Performance benchmarking scripts to validate system performance against defined SLOs and identify regressions. See [BENCHMARK_PLAN.md](../../docs/BENCHMARK_PLAN.md) for comprehensive benchmarking methodology and scenarios.

## Directory Structure

```
scripts/benchmark/
├── README.md (this file)
├── k6/                    # k6 load testing scripts
├── locust/                # Locust scenario-based testing scripts  
└── examples/              # Example implementations
```

## Current Status

**Status**: 🔧 In Development

The benchmark scripts referenced in `docs/BENCHMARK_PLAN.md` are reference implementations. This directory is a placeholder for actual benchmark script development.

## Implementation Guide

### Quick Start with Existing Tests

The repository includes working performance tests in `tests/validation/test_performance_validation.py`. These tests validate:
- Decision evaluation latency
- Cache performance
- Concurrent request handling

Run performance tests:
```bash
pytest tests/validation/test_performance_validation.py -v
```

### Tools

Recommended tools for performance testing:
- **k6**: High-performance load testing (install: `brew install k6` or see https://k6.io)
- **Locust**: Python-based load testing (install: `pip install locust`)
- **pytest**: For validation test suite (already installed)

### Integration with CI/CD

Current automated performance testing:
- `.github/workflows/performance.yml` - Regular performance validation
- `.github/workflows/performance-regression.yml` - Regression detection

### Metrics & Thresholds

Performance targets (from `validation_config.yaml`):
- p50 latency: <100ms
- p95 latency: <200ms
- p99 latency: <500ms
- Error rate: <0.5%

## Related Documentation

- [Benchmark Plan](../../docs/BENCHMARK_PLAN.md) - Comprehensive benchmark methodology
- [Validation Plan](../../VALIDATION_PLAN.md) - Overall validation strategy
- [Performance Tests](../../tests/validation/test_performance_validation.py) - Current test implementation
