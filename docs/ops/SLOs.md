# Service Level Objectives (SLOs)

## Overview
This document defines target SLOs for Nethical AI Governance system.

## Availability SLO
- **Target**: 99.9% uptime (8.76 hours downtime/year)
- **Measurement**: HTTP health check every 30s
- **Monitoring**: Grafana dashboard panel

## Latency SLOs

### process_action() Method
- **p50**: < 50ms
- **p95**: < 200ms
- **p99**: < 500ms
- **Measurement**: OpenTelemetry traces

### Quota Check
- **p95**: < 10ms
- **Measurement**: `QuotaEnforcer.check_quota()` timing

### PII Detection
- **p95**: < 50ms per 1KB of text
- **Measurement**: `PIIDetector.detect_all()` timing

## Throughput SLOs
- **Minimum**: 100 actions/second per instance
- **Target**: 500 actions/second per instance
- **Peak**: 1000 actions/second with burst capacity

## Accuracy SLOs

### False Positive Rate
- **Target**: < 5% for BLOCK decisions
- **Measurement**: Manual review of blocked actions
- **Alert**: > 10% triggers investigation

### False Negative Rate
- **Target**: < 2% for adversarial attacks
- **Measurement**: Adversarial test suite results
- **Alert**: Any test failure triggers review

### PII Detection Accuracy
- **Precision**: > 95%
- **Recall**: > 90%
- **Measurement**: Labeled test dataset evaluation

## Resource Utilization SLOs
- **CPU**: < 70% average, < 90% peak
- **Memory**: < 80% average, < 95% peak
- **Storage Growth**: < 10GB/day at 1M actions/day
- **Disk IOPS**: < 80% capacity

## Error Rate SLOs
- **5xx Errors**: < 0.1%
- **4xx Errors**: < 1%
- **Quota Rejections**: < 5% of total requests

## Data Integrity SLOs
- **Merkle Verification**: 100% pass rate
- **Audit Log Completeness**: 99.99%
- **Data Loss**: 0% (zero tolerance)

## Recovery Time Objectives (RTO/RPO)
- **RTO**: < 1 hour (time to restore service)
- **RPO**: < 5 minutes (acceptable data loss)

## Alerting Thresholds

### Critical (PagerDuty)
- Availability < 99.5% over 5min
- p95 latency > 1000ms
- Error rate > 1%
- Merkle verification failure
- Storage > 95% full

### Warning (Slack)
- Availability < 99.9% over 15min
- p95 latency > 500ms
- CPU > 80%
- Memory > 85%
- False positive rate > 7%

### Info (Logging)
- Quota throttling > 10%
- Unusual traffic patterns
- Policy changes

## Monitoring Dashboard Panels
1. Request rate (actions/sec)
2. Latency percentiles (p50, p95, p99)
3. Error rates by type
4. Quota utilization by agent/cohort
5. Risk score distribution
6. Violation counts by type/severity
7. Resource utilization (CPU/mem/disk)
8. Merkle verification status
9. PII detections over time
10. Escalation queue depth

---
Review Frequency: Monthly
Last Updated: 2025-10-15
