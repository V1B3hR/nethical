# Ethics Validation Scripts

This directory contains scripts for ethics detection validation, threshold tuning, and drift monitoring.

## Purpose

Scripts for validating, calibrating, and monitoring ethics detection quality across violation categories. See [ETHICS_VALIDATION_FRAMEWORK.md](../../docs/ETHICS_VALIDATION_FRAMEWORK.md) for comprehensive methodology.

## Current Status

**Status**: 🔧 In Development

The ethics scripts referenced in `docs/ETHICS_VALIDATION_FRAMEWORK.md` are reference implementations. This directory is a placeholder for actual script development.

## Quick Start with Existing Tests

The repository includes working ethics validation tests in `tests/validation/test_ethics_benchmark.py`:

Run ethics tests:
```bash
pytest tests/validation/test_ethics_benchmark.py -v
```

## Metrics & Thresholds

Ethics detection targets (from `validation_config.yaml`):
- Overall F1: ≥0.90
- Precision: ≥0.92
- Recall: ≥0.88
- Harmful content recall: ≥0.95
- Privacy precision: ≥0.96

## Related Documentation

- [Ethics Validation Framework](../../docs/ETHICS_VALIDATION_FRAMEWORK.md) - Comprehensive validation methodology
- [Validation Plan](../../VALIDATION_PLAN.md) - Overall validation strategy  
- [Ethics Tests](../../tests/validation/test_ethics_benchmark.py) - Current test implementation
