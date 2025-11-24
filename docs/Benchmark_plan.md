# Benchmark & Performance Plan

## Tools
- k6 for HTTP load (scripts/benchmark/k6/*.js)
- Locust for scenario-based concurrent agent simulation
- Custom Python harness for policy impact evaluation

## Scenarios
1. Baseline Decision Flow (evaluate action)
2. PII Heavy Payload
3. Mixed Traffic (50% evaluate, 30% check_pii, 20% status)
4. Escalation Queue Surge
5. Plugin Marketplace Query Storm
6. Global Multi-Region (simulated latency injection)

## Outputs
- Prometheus scrape → Grafana panels (export JSON)
- Raw metrics: CSV (timestamp, rps, p50, p95, p99, error_rate)
- Comparative trend graphs stored under docs/perf/reports/

## Acceptance
- All scenarios meet latency gates
- No sustained memory growth >5% over 2h soak
- CPU utilization stable (60–70% target)

## Continuous
Nightly smaller “pulse” run; full suite weekly.
