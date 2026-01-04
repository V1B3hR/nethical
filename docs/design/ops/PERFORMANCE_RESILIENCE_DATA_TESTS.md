# Performance, Resilience, and Data Storage Testing

This document provides an overview of the comprehensive test suite for requirements 5 (Performance), 6 (Resilience), and 7 (Data & Storage).

## Quick Reference

### Test Commands

```bash
# Run all quick tests (CI/CD friendly, ~5-10 minutes)
pytest tests/performance/test_load_sustained.py::test_sustained_load_short -v -s
pytest tests/performance/test_load_burst.py -v -s
pytest tests/performance/test_load_soak.py::test_soak_short -v -s
pytest tests/resilience/test_chaos.py -v -s
pytest tests/resilience/test_backup_restore.py -v -s
pytest tests/storage/test_compression.py -v -s
pytest tests/storage/test_projections.py -v -s

# Run extended tests (requires --run-extended flag)
pytest tests/performance/test_load_sustained.py --run-extended -v -s

# Run soak test (requires --run-soak flag, 2 hours)
pytest tests/performance/test_load_soak.py --run-soak -v -s
```

## Requirement 5: Performance

### 5.1 Sustained Load Test

**File**: `tests/performance/test_load_sustained.py`

**Purpose**: Validates system handles sustained load over extended periods with consistent performance.

**Test Configurations**:
- **Short test** (60s): For CI/CD pipelines
- **Extended test** (10min): For more thorough validation

**Pass Criteria**:
- Success rate ≥ 95%
- P95 response time < 200ms (short) / P99 < 500ms (extended)
- Minimum throughput maintained

**Artifacts**:
- Raw metrics JSON with all request data
- Summary report JSON with statistics
- Human-readable Markdown report

**Example Output**:
```
=== Starting Sustained Load Test (60s) ===
Test completed:
  Total requests: 600
  Success rate: 98.50%
  Throughput: 10.02 req/sec
  Mean response time: 45.32ms
  P95 response time: 125.67ms

Artifacts saved:
  raw_file: tests/performance/results/load_tests/sustained_load_raw_20250124_082130.json
  report_file: tests/performance/results/load_tests/sustained_load_report_20250124_082130.json
  md_file: tests/performance/results/load_tests/sustained_load_report_20250124_082130.md
```

### 5.2 Burst Load Test

**File**: `tests/performance/test_load_burst.py`

**Purpose**: Tests system handles sudden traffic spikes (5× baseline load).

**Test Configuration**:
- Baseline: 100 requests
- Burst: 500 requests (5× baseline)

**Pass Criteria**:
- Success rate ≥ 90% during burst
- P95 response time degradation ≤ 300%

**Artifacts**:
- Raw request data for both phases
- Comparison report showing degradation
- Pass/fail determination

**Example Output**:
```
=== Starting Burst Load Test (5× baseline) ===
Phase 1: Running baseline load (100 requests)...
Baseline completed:
  Success rate: 99.00%
  Mean response time: 42.15ms
  P95 response time: 89.34ms

Phase 2: Running burst load (500 requests - 5× baseline)...
Burst completed:
  Success rate: 94.20%
  Mean response time: 67.82ms
  P95 response time: 198.45ms

Performance degradation under burst:
  Mean response time: +60.88%
  P95 response time: +122.15%
  Success rate change: -4.80%

=== Validating Pass Criteria ===
Success rate ≥ 90%: ✅ PASS (94.20%)
P95 degradation ≤ 300%: ✅ PASS (122.15%)

Overall: ✅ PASSED
```

### 5.3 Soak Test

**File**: `tests/performance/test_load_soak.py`

**Purpose**: Detects memory leaks and performance degradation over extended runtime.

**Test Configurations**:
- **Short test** (5min): For CI/CD
- **Full soak** (2h): For production validation

**Pass Criteria**:
- Memory growth < 5%
- Performance degradation < 20%
- Success rate ≥ 95%

**Artifacts**:
- Memory samples over time
- Performance degradation analysis
- Memory leak detection report

**Example Output**:
```
=== Starting Short Soak Test (5 minutes) ===
Test completed:
  Duration: 0.08 hours
  Total requests: 1500
  Success rate: 97.80%
  Memory growth: +2.34%
  Performance degradation: +8.12%

✅ All criteria passed
```

## Requirement 6: Resilience

### 6.1 Chaos Testing

**File**: `tests/resilience/test_chaos.py`

**Purpose**: Tests system resilience under failure conditions.

**Test Scenarios**:
1. **Pod Kill Recovery**: Kill pod, measure recovery time
2. **Region Failover**: Simulate regional failure, measure failover time
3. **Quorum Recovery**: Multiple pod failures, validate quorum reestablishment

**Pass Criteria**:
- Pod recovery < 30 seconds
- Region failover < 15 seconds
- Quorum recovery < 45 seconds
- Quorum maintained (≥2/3 pods)

**Modes**:
- **Simulation mode**: For CI/CD without Kubernetes
- **Real mode**: For production testing with actual kubectl commands

**Example Output**:
```
=== Testing Pod Kill Recovery ===
Killing pod nethical-0 in namespace nethical...
Waiting for pod to recover...

Pod recovery:
  Ready: True
  Recovery time: 4.23s
  Target: 30s

Recovery time ≤ 30s: ✅ PASS
```

### 6.3 Backup & Restore

**File**: `tests/resilience/test_backup_restore.py`

**Purpose**: Validates automated backup and restore procedures.

**Test Workflow**:
1. Create backup of data directory
2. Simulate system failure
3. Restore from backup
4. Validate restored data integrity
5. Verify RTO compliance

**Pass Criteria**:
- Backup completes successfully
- Restore completes successfully
- All validation checks pass
- Total RTO < 15 minutes

**Example Output**:
```
=== Starting Backup Restore Dry-Run Test ===

Step 1: Creating backup...
  Status: ✅ Success
  Duration: 1.89s

Step 3: Restoring from backup...
  Status: ✅ Success
  Duration: 2.76s

Step 4: Validating restored data...
  Validation checks: 3
  Passed: 3
  Failed: 0
    ✅ restore_dir_exists: Restore directory exists
    ✅ metadata_exists: Metadata file exists
    ✅ metadata_valid: Metadata valid: 2025-01-24T08:21:30.123456

Step 5: Checking RTO...
  Backup time: 1.89s
  Restore time: 2.76s
  Total RTO: 4.65s (0.08 min)
  Target RTO: 900s (15 min)
  RTO Met: ✅ YES

=== Test Results ===
Overall: ✅ PASSED
```

## Requirement 7: Data & Storage

### 7.1 & 7.2 Compression Testing

**File**: `tests/storage/test_compression.py`

**Purpose**: Validates compression ratios meet >5:1 aggregate target.

**Test Coverage**:
- Individual tier compression ratios
- Aggregate compression with optimized distribution
- Different data types (JSON, metrics, binary)

**Pass Criteria**:
- Hot tier: ~1:1 (no compression)
- Warm tier: ~2:1 (LZ4)
- Cool tier: ~5:1 (ZSTD)
- Cold tier: ~10:1 (ZSTD max)
- Archive tier: ~15:1 (LZMA)
- **Aggregate: >5:1**

**Example Output**:
```
=== Testing Aggregate Compression Ratio ===

Tier Distribution:
  hot: 5% @ 1.00:1
  warm: 10% @ 2.01:1
  cool: 20% @ 5.12:1
  cold: 50% @ 10.08:1
  archive: 15% @ 14.89:1

Aggregate Compression Ratio: 5.03:1
Target: >5:1

Aggregate ratio ≥ 5.0:1: ✅ PASS
```

### 7.3 Storage Projections

**File**: `tests/storage/test_projections.py`

**Purpose**: Validates 12-month storage projections and budget compliance.

**Test Coverage**:
- Monthly storage growth projections
- Cost calculations per tier
- Budget threshold validation
- Scaling analysis (1×, 10×, 100×)

**Pass Criteria**:
- Month 12 storage within expected range (28-32 GB)
- Month 12 cost < $50 budget threshold
- Linear growth for raw data
- Sub-linear growth for compressed data

**Example Output**:
```
=== Testing Monthly Storage Projections ===

Projected Storage:
Month    Raw GB       Compressed GB    Ratio   
--------------------------------------------------
1            8.0           8.0      1.00:1
3           24.0          13.9      1.73:1
6           48.0          18.6      2.58:1
12          96.0          28.9      3.32:1

✅ Monthly projections validated

=== Testing Budget Threshold ===

Budget Threshold: $50.00/month

Month 12 Total: 28.92 GB @ $1.79/month
Budget Remaining: $48.21
Budget Utilization: 3.6%

✅ Budget threshold validated ($1.79 < $50.00)
```

## Artifacts and Reports

All tests generate detailed artifacts in the following locations:

### Performance
- **Location**: `tests/performance/results/load_tests/`
- **Files**:
  - `sustained_load_raw_*.json` - Raw request data
  - `sustained_load_report_*.json` - Summary statistics
  - `sustained_load_report_*.md` - Human-readable report
  - Similar files for burst and soak tests

### Resilience
- **Location**: `tests/resilience/results/`
- **Files**:
  - `pod_kill_raw_*.json` - Chaos test events
  - `pod_kill_report_*.md` - Chaos test report
  - `backup_restore_raw_*.json` - Backup/restore metrics
  - `backup_restore_report_*.md` - Backup/restore report

### Storage
- **Location**: `tests/storage/results/`
- **Files**:
  - `compression_report_*.md` - Compression analysis
  - `storage_projection_*.json` - 12-month projections
  - `storage_projection_*.md` - Projection visualization
  - `cost_projection_*.json` - Cost analysis

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Performance and Resilience Tests

on:
  push:
    branches: [main, develop]
  pull_request:
  schedule:
    - cron: '0 2 * * *'  # Nightly

jobs:
  quick-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -e .[test]
      - name: Run quick performance tests
        run: |
          pytest tests/performance/test_load_sustained.py::test_sustained_load_short -v
          pytest tests/performance/test_load_burst.py -v
          pytest tests/performance/test_load_soak.py::test_soak_short -v
      - name: Run resilience tests
        run: |
          pytest tests/resilience/test_chaos.py -v
          pytest tests/resilience/test_backup_restore.py -v
      - name: Run storage tests
        run: |
          pytest tests/storage/ -v
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: test-reports
          path: tests/**/results/

  extended-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -e .[test]
      - name: Run extended tests
        run: |
          pytest tests/performance/test_load_sustained.py --run-extended -v
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: extended-test-reports
          path: tests/**/results/
```

## Best Practices

### Running Tests Locally

1. **Install dependencies**:
   ```bash
   pip install -e .[test]
   ```

2. **Run quick validation**:
   ```bash
   pytest tests/performance tests/resilience tests/storage -v
   ```

3. **Run specific test**:
   ```bash
   pytest tests/performance/test_load_burst.py -v -s
   ```

4. **Run extended tests** (only when needed):
   ```bash
   pytest tests/performance/test_load_sustained.py --run-extended -v -s
   ```

### Interpreting Results

- **Success rate < 95%**: Investigate errors, may indicate system issues
- **High response times**: Check system resources, may need optimization
- **Memory growth >5%**: Investigate memory leaks
- **RTO exceeded**: Review backup/restore procedures
- **Compression ratio <5:1**: Adjust tier distribution or compression settings

### Troubleshooting

**Test timeouts**:
- Check system resources (CPU, memory)
- Verify test configuration (duration, RPS)
- Review test logs for errors

**Artifacts not generated**:
- Check permissions on results directories
- Verify output_dir fixture
- Ensure tests complete successfully

**Budget threshold failures**:
- Review storage projections
- Adjust tier distribution
- Consider more aggressive compression

## References

- [Performance Documentation](../performance.md)
- [Data Storage Retention](DATA_STORAGE_RETENTION.md)
- [SLO Definitions](slo_definitions.md)
- [Backup & DR Guide](backup_dr.md)
