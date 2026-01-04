# Nethical Benchmark Plan

This comprehensive plan defines performance testing scenarios, tooling, metrics, acceptance criteria, and operational procedures to ensure Nethical meets production-grade performance and resilience standards.

---

## Table of Contents

1. [Tooling & Infrastructure](#tooling--infrastructure)
2. [Benchmark Scenarios](#benchmark-scenarios)
3. [Metrics Captured](#metrics-captured)
4. [Acceptance Criteria](#acceptance-criteria)
5. [Test Artifacts](#test-artifacts)
6. [Execution Cadence](#execution-cadence)
7. [Soak Test Parameters](#soak-test-parameters)
8. [Burst Test Method](#burst-test-method)
9. [Post-Test Analysis](#post-test-analysis)
10. [Regression Detection & Policy](#regression-detection--policy)

---

## Tooling & Infrastructure

### Primary Tools

#### 1. k6 (Load Testing)

**Purpose**: HTTP load generation for API endpoint testing.

**Why k6**: 
- High performance (Golang-based)
- Scriptable in JavaScript
- Built-in metrics (p95, p99 latency)
- Integration with Prometheus/Grafana

**Installation**:
```bash
# macOS
brew install k6

# Linux
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
  --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | \
  sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

**Basic Usage**:
```bash
k6 run --vus 100 --duration 5m scripts/benchmark/k6/baseline_test.js
```

---

#### 2. Locust (Scenario-Based Testing)

**Purpose**: Complex user behavior simulation with concurrent agents.

**Why Locust**:
- Python-based (easy integration with Nethical code)
- Distributed load generation
- Web UI for real-time monitoring
- Custom user scenarios

**Installation**:
```bash
pip install locust
```

**Basic Usage**:
```bash
locust -f scripts/benchmark/locust/agent_scenarios.py \
  --host https://nethical.example.com \
  --users 500 --spawn-rate 50 --run-time 10m
```

---

#### 3. Custom Python Harness

**Purpose**: Policy-specific evaluation and complex workflow testing.

**Note**: See `tests/validation/test_performance_validation.py` for actual performance testing implementation.

**Features**:
- Policy impact measurement
- Multi-step decision flows
- Custom metrics (cache hit, policy evaluation time)
- Integration with Nethical internals

**Usage** (Example):
```bash
# Reference implementation - adapt to your testing needs
python scripts/benchmark/policy_impact.py \
  --policy-file policies/production_v1.yaml \
  --scenario escalation_surge \
  --duration 600
```

---

### Supporting Infrastructure

#### Monitoring Stack

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./dashboards:/etc/grafana/provisioning/dashboards
  
  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"

volumes:
  prometheus-data:
  grafana-data:
```

#### Load Generator Cluster

```yaml
# deploy/kubernetes/load-generator.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: k6-load-generator
  namespace: testing
spec:
  replicas: 5
  selector:
    matchLabels:
      app: k6-load
  template:
    metadata:
      labels:
        app: k6-load
    spec:
      containers:
      - name: k6
        image: grafana/k6:latest
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
```

---

## Benchmark Scenarios

### Scenario 1: Baseline Decision Flow

**Objective**: Establish performance baseline under normal operating conditions.

**Note**: The following scenarios are reference implementations. Actual benchmark scripts should be created in `scripts/benchmark/` directory. See existing workflows in `.github/workflows/performance.yml` and `performance-regression.yml` for current performance testing approach.

**Profile**:
- **RPS**: 1,000 requests/second
- **Duration**: 30 minutes
- **Mix**: 100% `/api/v1/agent/evaluate` endpoint
- **Payload**: 1KB average (typical agent action context)
- **Cache State**: Warm (90% hit ratio target)

**k6 Script** (Reference Example):

```javascript
// scripts/benchmark/k6/baseline_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '2m', target: 500 },   // Ramp-up
    { duration: '20m', target: 1000 }, // Steady state
    { duration: '5m', target: 1000 },  // Sustained
    { duration: '3m', target: 0 },     // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<200', 'p(99)<500'],
    errors: ['rate<0.005'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export default function () {
  const payload = JSON.stringify({
    agent_id: `agent_${__VU}_${__ITER}`,
    action_type: 'generate_text',
    context: 'User wants to know about AI safety best practices.',
    metadata: {
      user_id: `user_${Math.floor(Math.random() * 10000)}`,
      session_id: `session_${__VU}`,
      timestamp: new Date().toISOString(),
    },
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': __ENV.API_KEY,
    },
  };

  const res = http.post(`${BASE_URL}/api/v1/agent/evaluate`, payload, params);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
    'has decision': (r) => JSON.parse(r.body).decision !== undefined,
  }) || errorRate.add(1);

  sleep(0.01); // 10ms think time
}
```

**Expected Results**:
- p50 latency: <100ms
- p95 latency: <200ms
- p99 latency: <500ms
- Error rate: <0.5%
- Cache hit ratio: >90%

---

### Scenario 2: PII Heavy Payload

**Objective**: Test performance with PII detection workload.

**Profile**:
- **RPS**: 500 requests/second
- **Duration**: 15 minutes
- **Mix**: 100% `/api/v1/check_pii` endpoint
- **Payload**: 10KB average (large text with potential PII)
- **Detection Load**: High (multiple PII types per request)

**Locust Script** (Reference Example):

```python
# Example: scripts/benchmark/locust/pii_heavy.py
from locust import HttpUser, task, between
import json
import random

class PIIHeavyUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    pii_samples = [
        "John Doe, SSN: 123-45-6789, email: john@example.com, phone: 555-123-4567",
        "Patient Jane Smith (DOB: 01/15/1980) has medical record #MR-789456",
        "Credit card 4532-1234-5678-9010 expiring 12/25, CVV: 123",
        "My address is 123 Main St, Apt 4B, New York, NY 10001",
    ]
    
    @task
    def check_pii(self):
        # Generate large text with embedded PII
        text_parts = [random.choice(self.pii_samples) for _ in range(20)]
        large_text = " ".join(text_parts) + " " + ("Lorem ipsum dolor sit amet. " * 100)
        
        payload = {
            "text": large_text,
            "redaction_mode": "aggressive",
        }
        
        with self.client.post(
            "/api/v1/check_pii",
            json=payload,
            headers={"X-API-Key": "test-key"},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'detected_pii' in data:
                    response.success()
                else:
                    response.failure("Missing detected_pii field")
            else:
                response.failure(f"Status {response.status_code}")
```

**Expected Results**:
- p50 latency: <300ms
- p95 latency: <600ms
- p99 latency: <1000ms
- PII detection accuracy: >95%
- Redaction correctness: 100%

---

### Scenario 3: Mixed Traffic

**Objective**: Realistic production traffic simulation with multiple endpoint types.

**Profile**:
- **RPS**: 800 requests/second total
- **Duration**: 45 minutes
- **Mix**:
  - 50% evaluate action
  - 30% check PII
  - 15% policy query
  - 5% audit log query
- **Payload**: Variable (0.5KB - 10KB)

**k6 Script** (Reference Example):

```javascript
// Example: scripts/benchmark/k6/mixed_traffic.js
import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';

export const options = {
  scenarios: {
    evaluate: {
      executor: 'constant-arrival-rate',
      rate: 400,
      timeUnit: '1s',
      duration: '45m',
      preAllocatedVUs: 200,
      exec: 'evaluateAction',
    },
    check_pii: {
      executor: 'constant-arrival-rate',
      rate: 240,
      timeUnit: '1s',
      duration: '45m',
      preAllocatedVUs: 120,
      exec: 'checkPII',
    },
    policy: {
      executor: 'constant-arrival-rate',
      rate: 120,
      timeUnit: '1s',
      duration: '45m',
      preAllocatedVUs: 60,
      exec: 'queryPolicy',
    },
    audit: {
      executor: 'constant-arrival-rate',
      rate: 40,
      timeUnit: '1s',
      duration: '45m',
      preAllocatedVUs: 20,
      exec: 'queryAudit',
    },
  },
  thresholds: {
    'http_req_duration{scenario:evaluate}': ['p(95)<200'],
    'http_req_duration{scenario:check_pii}': ['p(95)<600'],
    'http_req_duration{scenario:policy}': ['p(95)<100'],
    'http_req_duration{scenario:audit}': ['p(95)<500'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

export function evaluateAction() {
  const res = http.post(
    `${BASE_URL}/api/v1/agent/evaluate`,
    JSON.stringify({ agent_id: 'test', action_type: 'generate', context: 'test' }),
    { headers: { 'Content-Type': 'application/json', 'X-API-Key': __ENV.API_KEY } }
  );
  check(res, { 'evaluate 200': (r) => r.status === 200 });
}

export function checkPII() {
  const res = http.post(
    `${BASE_URL}/api/v1/check_pii`,
    JSON.stringify({ text: 'John Doe, SSN: 123-45-6789' }),
    { headers: { 'Content-Type': 'application/json', 'X-API-Key': __ENV.API_KEY } }
  );
  check(res, { 'pii 200': (r) => r.status === 200 });
}

export function queryPolicy() {
  const res = http.get(
    `${BASE_URL}/api/v1/policy/active`,
    { headers: { 'X-API-Key': __ENV.API_KEY } }
  );
  check(res, { 'policy 200': (r) => r.status === 200 });
}

export function queryAudit() {
  const res = http.get(
    `${BASE_URL}/api/v1/audit/recent?limit=100`,
    { headers: { 'X-API-Key': __ENV.API_KEY } }
  );
  check(res, { 'audit 200': (r) => r.status === 200 });
}
```

---

### Scenario 4: Escalation Queue Surge

**Objective**: Test system behavior under human review queue backlog.

**Profile**:
- **RPS**: 300 requests/second
- **Duration**: 20 minutes
- **Mix**: 80% low-confidence decisions (trigger escalation)
- **Queue Depth**: 0 → 5,000+ items
- **Reviewer Actions**: 10 concurrent reviewers processing queue

**Custom Harness** (Reference Example):

```python
# Example: scripts/benchmark/escalation_surge.py
import asyncio
import aiohttp
import time
from datetime import datetime

async def generate_escalation(session, agent_id, iteration):
    """Generate borderline decision to trigger escalation"""
    payload = {
        "agent_id": agent_id,
        "action_type": "content_generation",
        "context": "Ambiguous content that might be harmful but unclear context...",
        "metadata": {"test": "escalation_surge", "iteration": iteration}
    }
    
    async with session.post(
        "http://localhost:8000/api/v1/agent/evaluate",
        json=payload,
        headers={"X-API-Key": "test-key"}
    ) as resp:
        return await resp.json()

async def process_review_queue(session, reviewer_id):
    """Simulate human reviewer processing queue"""
    while True:
        try:
            # Fetch next item from queue
            async with session.get(
                "http://localhost:8000/api/v1/review/queue/next",
                headers={"X-API-Key": "reviewer-key"}
            ) as resp:
                if resp.status == 200:
                    item = await resp.json()
                    
                    # Simulate review time (20-60 seconds)
                    await asyncio.sleep(30)
                    
                    # Submit decision
                    decision = {
                        "decision_id": item['id'],
                        "reviewer_decision": "approve",
                        "confidence": 0.9,
                        "notes": "Reviewed by automated test"
                    }
                    await session.post(
                        "http://localhost:8000/api/v1/review/submit",
                        json=decision,
                        headers={"X-API-Key": "reviewer-key"}
                    )
                elif resp.status == 404:
                    # Queue empty, wait
                    await asyncio.sleep(5)
        except Exception as e:
            print(f"Reviewer {reviewer_id} error: {e}")
            await asyncio.sleep(10)

async def run_surge_test(duration_seconds=1200, rps=300):
    """Run escalation surge test"""
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        iteration = 0
        
        # Start reviewers
        reviewer_tasks = [
            asyncio.create_task(process_review_queue(session, i))
            for i in range(10)
        ]
        
        # Generate load
        while time.time() - start_time < duration_seconds:
            batch_tasks = [
                generate_escalation(session, f"agent_{i}", iteration)
                for i in range(rps // 10)  # Batch size
            ]
            await asyncio.gather(*batch_tasks)
            await asyncio.sleep(0.1)  # 10 batches/sec = RPS
            iteration += 1
        
        # Cancel reviewer tasks
        for task in reviewer_tasks:
            task.cancel()

if __name__ == '__main__':
    asyncio.run(run_surge_test(duration_seconds=1200, rps=300))
```

**Expected Results**:
- Queue depth peaks: <10,000 items
- Processing latency increase: <50% at peak
- No queue processor crashes
- Reviewer throughput: 150-180 items/hour per reviewer

---

### Scenario 5: Plugin Marketplace Queries

**Objective**: Test plugin search and metadata retrieval performance.

**Profile**:
- **RPS**: 200 requests/second
- **Duration**: 10 minutes
- **Mix**:
  - 60% plugin search
  - 25% plugin details
  - 15% download/install simulation
- **Catalog Size**: 10,000+ plugins

**k6 Script** (Reference Example):

```javascript
// Example: scripts/benchmark/k6/marketplace.js
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 100,
  duration: '10m',
  thresholds: {
    http_req_duration: ['p(95)<150'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const search_terms = ['security', 'pii', 'content', 'ethics', 'compliance'];

export default function () {
  const rand = Math.random();
  
  if (rand < 0.6) {
    // Plugin search
    const term = search_terms[Math.floor(Math.random() * search_terms.length)];
    const res = http.get(
      `${BASE_URL}/api/v1/marketplace/search?q=${term}&limit=20`,
      { headers: { 'X-API-Key': __ENV.API_KEY } }
    );
    check(res, { 'search 200': (r) => r.status === 200 });
    
  } else if (rand < 0.85) {
    // Plugin details
    const plugin_id = `plugin_${Math.floor(Math.random() * 10000)}`;
    const res = http.get(
      `${BASE_URL}/api/v1/marketplace/plugin/${plugin_id}`,
      { headers: { 'X-API-Key': __ENV.API_KEY } }
    );
    check(res, { 'details 200 or 404': (r) => [200, 404].includes(r.status) });
    
  } else {
    // Download stats
    const plugin_id = `plugin_${Math.floor(Math.random() * 1000)}`;  // Popular plugins
    const res = http.post(
      `${BASE_URL}/api/v1/marketplace/download/${plugin_id}`,
      null,
      { headers: { 'X-API-Key': __ENV.API_KEY } }
    );
    check(res, { 'download tracked': (r) => [200, 404].includes(r.status) });
  }
}
```

---

### Scenario 6: Multi-Region Simulation

**Objective**: Test performance with simulated network latency (global deployment).

**Profile**:
- **RPS**: 500 requests/second aggregate
- **Duration**: 30 minutes
- **Regions**: US-East, US-West, EU-Central, APAC
- **Latency Injection**: 10-200ms per region
- **Mix**: Baseline decision flow

**Implementation**:

```bash
# scripts/benchmark/multi_region.sh
#!/bin/bash

# US-East (baseline, no latency)
k6 run --vus 50 --duration 30m \
  -e BASE_URL=http://us-east.nethical.example.com \
  -e REGION=us-east \
  scripts/benchmark/k6/baseline_test.js &

# US-West (+30ms)
k6 run --vus 50 --duration 30m \
  -e BASE_URL=http://us-west.nethical.example.com \
  -e REGION=us-west \
  --http-debug="full" \
  scripts/benchmark/k6/baseline_test.js &

# EU-Central (+80ms)
k6 run --vus 50 --duration 30m \
  -e BASE_URL=http://eu.nethical.example.com \
  -e REGION=eu-central \
  scripts/benchmark/k6/baseline_test.js &

# APAC (+150ms)
k6 run --vus 50 --duration 30m \
  -e BASE_URL=http://apac.nethical.example.com \
  -e REGION=apac \
  scripts/benchmark/k6/baseline_test.js &

wait
```

**Expected Results**:
- US-East: p95 <200ms
- US-West: p95 <250ms
- EU-Central: p95 <300ms
- APAC: p95 <400ms
- Error rate consistent across regions: <0.5%

---

## Metrics Captured

### Application Metrics

| Metric | Source | Type | Description |
|--------|--------|------|-------------|
| **RPS** (Requests/sec) | k6, Locust | Gauge | Throughput rate |
| **Latency p50** | k6, Prometheus | Histogram | Median response time |
| **Latency p95** | k6, Prometheus | Histogram | 95th percentile latency |
| **Latency p99** | k6, Prometheus | Histogram | 99th percentile latency |
| **Error Rate** | k6, Application logs | Counter | HTTP 5xx / total requests |
| **Cache Hit Ratio** | Redis metrics | Gauge | Cache hits / total cache requests |
| **DB QPS** | PostgreSQL stats | Gauge | Database queries per second |
| **Queue Depth** | Application metrics | Gauge | Escalation queue size |
| **Decision Accuracy** | Ethics evaluator | Gauge | % decisions matching ground truth |

### Infrastructure Metrics

| Metric | Source | Type | Description |
|--------|--------|------|-------------|
| **CPU Utilization** | Prometheus node-exporter | Gauge | % CPU usage per pod |
| **Memory Usage** | Prometheus node-exporter | Gauge | Memory consumption (MB) |
| **Memory Growth Rate** | Calculated | Gauge | MB/hour (detect leaks) |
| **Network I/O** | Prometheus node-exporter | Counter | Bytes in/out per second |
| **Disk I/O** | Prometheus node-exporter | Counter | Read/write ops per second |
| **Pod Restart Count** | Kubernetes | Counter | Number of pod restarts |
| **Connection Pool** | Application metrics | Gauge | Active DB/Redis connections |

### Business Metrics

| Metric | Source | Type | Description |
|--------|--------|------|-------------|
| **Decisions/Hour** | Application logs | Counter | Total decisions made |
| **Violations Detected** | Application logs | Counter | Total violations by category |
| **Review SLA** | Review queue | Histogram | Time from escalation to review |
| **Plugin Load Time** | Marketplace | Histogram | Time to load/verify plugin |
| **Policy Evaluation Time** | Governance engine | Histogram | Time to evaluate policy rules |

---

## Acceptance Criteria

### Performance Gates

All benchmarks must meet these criteria to pass:

| Scenario | Metric | Threshold | Reference |
|----------|--------|-----------|-----------|
| **Baseline** | p95 latency | <200ms | Validation_plan.md |
| **Baseline** | p99 latency | <500ms | Validation_plan.md |
| **Baseline** | Error rate | <0.5% | Validation_plan.md |
| **Baseline** | Cache hit ratio | >90% | Validation_plan.md |
| **PII Heavy** | p95 latency | <600ms | Internal target |
| **PII Heavy** | PII detection accuracy | >95% | Ethics validation |
| **Mixed Traffic** | Overall p95 | <300ms | Weighted average |
| **Escalation Surge** | Queue depth | <10,000 | Capacity target |
| **Marketplace** | Search p95 | <150ms | User experience |
| **Multi-Region** | p95 (US) | <250ms | Regional SLA |

### Resource Utilization

| Resource | Limit | Rationale |
|----------|-------|-----------|
| **CPU** | 60-70% sustained | Headroom for bursts |
| **Memory** | <80% of limit | Prevent OOM kills |
| **Memory Growth** | <5% over 2h | No memory leaks |
| **Connection Pool** | <80% capacity | Connection availability |
| **Disk I/O Wait** | <10% | Storage not bottleneck |

### Stability

| Criterion | Requirement |
|-----------|-------------|
| **Zero Pod Restarts** | No crashes during test |
| **No Error Spikes** | Error rate variance <0.1% |
| **Consistent Throughput** | RPS variance <10% |
| **Graceful Degradation** | Response time increases <20% at 2× load |

---

## Test Artifacts

### Output Files

```
artifacts/benchmarks/YYYY-MM-DD_HH-MM/
├── results_summary.json          # Aggregated results
├── k6_output_baseline.json       # k6 JSON output
├── k6_output_mixed.json
├── locust_stats.csv              # Locust statistics
├── locust_failures.csv           # Failure log
├── prometheus_metrics.tar.gz     # Raw Prometheus data
├── grafana_dashboard.json        # Dashboard export
├── logs/
│   ├── application.log           # Application logs during test
│   ├── nginx_access.log          # Access logs
│   └── nginx_error.log           # Error logs
├── screenshots/
│   ├── grafana_overview.png      # Dashboard screenshots
│   ├── grafana_latency.png
│   └── grafana_errors.png
└── analysis/
    ├── latency_distribution.png  # Visualizations
    ├── throughput_timeline.png
    └── resource_usage.png
```

### Summary JSON Format

```json
{
  "test_metadata": {
    "timestamp": "2024-11-24T10:00:00Z",
    "git_commit": "abc123",
    "version": "v2.4.0",
    "duration_seconds": 1800,
    "scenarios": ["baseline", "mixed"]
  },
  "results": {
    "baseline": {
      "rps_avg": 1005,
      "rps_max": 1120,
      "latency": {
        "p50": 95,
        "p95": 185,
        "p99": 420,
        "max": 1250
      },
      "error_rate": 0.0023,
      "http_status": {
        "200": 1805000,
        "429": 150,
        "500": 4200
      },
      "cache_hit_ratio": 0.912
    },
    "mixed": { /* ... */ }
  },
  "resources": {
    "cpu_avg_percent": 68,
    "cpu_max_percent": 82,
    "memory_avg_mb": 3200,
    "memory_max_mb": 4100,
    "memory_growth_mb_per_hour": 12
  },
  "acceptance": {
    "passed": true,
    "failures": []
  }
}
```

### Grafana Dashboard Export

Export dashboard as JSON and commit to repository:

```bash
# Export Grafana dashboard
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  http://localhost:3000/api/dashboards/uid/nethical-benchmark \
  | jq '.dashboard' > artifacts/benchmarks/grafana_dashboard.json
```

### CSV Format (Locust)

```csv
Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%
POST,/api/v1/agent/evaluate,100000,250,98,105,45,2100,512,166.67,0.42,98,110,125,135,160,185,220,280,850,1200,2100
GET,/api/v1/policy/active,50000,10,45,48,20,450,256,83.33,0.02,45,50,55,60,70,80,95,120,280,400,450
```

---

## Execution Cadence

### Nightly Pulse Run

**Duration**: 15 minutes  
**Scenarios**: Baseline only  
**Purpose**: Quick sanity check for regressions  
**Schedule**: Daily at 2:00 AM UTC

**Note**: See actual implementation in `.github/workflows/performance.yml` for current automated performance testing.

```yaml
# Example: .github/workflows/nightly-pulse.yml
name: Nightly Performance Pulse
on:
  schedule:
    - cron: '0 2 * * *'

jobs:
  pulse-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy test environment
        run: docker-compose -f docker-compose.test.yml up -d
      
      - name: Run k6 pulse test
        run: |
          k6 run --vus 100 --duration 15m \
            --out json=pulse_results.json \
            scripts/benchmark/k6/baseline_test.js
      
      - name: Check thresholds
        run: |
          python scripts/benchmark/check_results.py \
            --results pulse_results.json \
            --thresholds config/benchmark_thresholds.yaml \
            --fail-on-breach
      
      - name: Create issue on failure
        if: failure()
        run: |
          gh issue create \
            --title "Performance regression detected in nightly pulse" \
            --body "See workflow run for details" \
            --label "performance,regression"
```

### Weekly Full Suite

**Duration**: 2-3 hours  
**Scenarios**: All 6 scenarios  
**Purpose**: Comprehensive performance validation  
**Schedule**: Saturday at 1:00 AM UTC

```bash
# scripts/benchmark/weekly_full_suite.sh
#!/bin/bash

set -euo pipefail

TIMESTAMP=$(date +%Y-%m-%d_%H-%M)
ARTIFACT_DIR="artifacts/benchmarks/${TIMESTAMP}"
mkdir -p "$ARTIFACT_DIR"

echo "Starting weekly full benchmark suite at $TIMESTAMP"

# 1. Baseline
echo "Running baseline test..."
k6 run --vus 1000 --duration 30m \
  --out json="${ARTIFACT_DIR}/k6_baseline.json" \
  scripts/benchmark/k6/baseline_test.js

# 2. PII Heavy
echo "Running PII heavy test..."
locust -f scripts/benchmark/locust/pii_heavy.py \
  --host http://localhost:8000 \
  --users 500 --spawn-rate 50 --run-time 15m \
  --csv "${ARTIFACT_DIR}/locust_pii"

# 3. Mixed Traffic
echo "Running mixed traffic test..."
k6 run --duration 45m \
  --out json="${ARTIFACT_DIR}/k6_mixed.json" \
  scripts/benchmark/k6/mixed_traffic.js

# 4. Escalation Surge
echo "Running escalation surge test..."
python scripts/benchmark/escalation_surge.py \
  --duration 1200 --rps 300 \
  --output "${ARTIFACT_DIR}/escalation_results.json"

# 5. Marketplace
echo "Running marketplace test..."
k6 run --vus 100 --duration 10m \
  --out json="${ARTIFACT_DIR}/k6_marketplace.json" \
  scripts/benchmark/k6/marketplace.js

# 6. Multi-Region
echo "Running multi-region test..."
bash scripts/benchmark/multi_region.sh

# Generate summary report
python scripts/benchmark/generate_report.py \
  --artifact-dir "$ARTIFACT_DIR" \
  --output "${ARTIFACT_DIR}/results_summary.json"

# Export Grafana dashboard
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
  http://localhost:3000/api/dashboards/uid/nethical-benchmark \
  | jq '.dashboard' > "${ARTIFACT_DIR}/grafana_dashboard.json"

echo "Benchmark suite complete. Results in $ARTIFACT_DIR"

# Commit results to repository
git add "$ARTIFACT_DIR"
git commit -m "Weekly benchmark results ${TIMESTAMP}"
git push
```

### On-Demand (Pre-Release)

**Trigger**: Manual or on release tag  
**Duration**: 4-6 hours (includes soak test)  
**Purpose**: Release validation

---

## Soak Test Parameters

### Configuration

**Objective**: Detect memory leaks, resource exhaustion, and long-term stability issues.

**Parameters**:
- **Duration**: 2 hours minimum, 24 hours comprehensive
- **Load**: 70% of baseline capacity (700 RPS)
- **Scenario**: Mixed traffic (realistic load)
- **Monitoring**: Every 1 minute

**k6 Soak Test**:

```javascript
// scripts/benchmark/k6/soak_test.js
export const options = {
  stages: [
    { duration: '10m', target: 700 },  // Ramp-up
    { duration: '2h', target: 700 },   // Sustained (2h minimum)
    { duration: '10m', target: 0 },    // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<250'],
    http_req_failed: ['rate<0.01'],
  },
};

// Same mixed traffic logic as Scenario 3
```

**Memory Leak Detection**:

```python
# scripts/benchmark/analyze_soak.py
import pandas as pd
import numpy as np
from scipy import stats

def detect_memory_leak(memory_samples, threshold_pct=5):
    """
    Detect memory leak via linear regression on memory usage over time.
    
    Args:
        memory_samples: List of (timestamp, memory_mb) tuples
        threshold_pct: Maximum acceptable growth percentage
    
    Returns:
        leak_detected: bool
        growth_rate_mb_per_hour: float
    """
    df = pd.DataFrame(memory_samples, columns=['timestamp', 'memory_mb'])
    df['hours'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds() / 3600
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['hours'], df['memory_mb']
    )
    
    # Growth rate in MB/hour
    growth_rate = slope
    
    # Calculate percentage growth over test duration
    initial_memory = df['memory_mb'].iloc[0]
    final_memory = initial_memory + (growth_rate * df['hours'].iloc[-1])
    growth_pct = ((final_memory - initial_memory) / initial_memory) * 100
    
    leak_detected = growth_pct > threshold_pct and p_value < 0.05
    
    return leak_detected, growth_rate, growth_pct

# Run analysis
memory_data = load_memory_metrics('prometheus_metrics.csv')
leak, rate, pct = detect_memory_leak(memory_data, threshold_pct=5)

if leak:
    print(f"⚠️ Memory leak detected: {rate:.2f} MB/hour ({pct:.1f}% growth)")
else:
    print(f"✅ No memory leak: {rate:.2f} MB/hour ({pct:.1f}% growth)")
```

---

## Burst Test Method

### Configuration

**Objective**: Validate system behavior under sudden traffic spikes (5× baseline).

**Parameters**:
- **Baseline**: 1,000 RPS
- **Burst**: 5,000 RPS
- **Burst Duration**: 5 minutes
- **Recovery**: Return to baseline

**k6 Burst Test**:

```javascript
// scripts/benchmark/k6/burst_test.js
export const options = {
  stages: [
    { duration: '5m', target: 1000 },   // Baseline
    { duration: '30s', target: 5000 },  // Rapid ramp-up
    { duration: '5m', target: 5000 },   // Burst sustained
    { duration: '30s', target: 1000 },  // Rapid ramp-down
    { duration: '5m', target: 1000 },   // Recovery
  ],
  thresholds: {
    'http_req_duration{stage:burst}': ['p(95)<400'],  // Relaxed during burst
    'http_req_duration{stage:recovery}': ['p(95)<200'], // Must recover
    http_req_failed: ['rate<0.02'],  // Allow 2% errors during burst
  },
};

export default function () {
  const stage = __ENV.STAGE || 'baseline';
  // Same baseline test logic
}
```

**Acceptance**:
- Error rate during burst: <2%
- p95 latency during burst: <400ms
- Recovery time to baseline: <2 minutes
- No pod crashes

---

## Post-Test Analysis

### Automated Analysis Script

```python
# scripts/benchmark/analyze_results.py
import json
import sys
from pathlib import Path

def analyze_benchmark(results_file, thresholds_file):
    """
    Analyze benchmark results and detect regressions.
    
    Returns:
        report: dict with analysis
        passed: bool
    """
    with open(results_file) as f:
        results = json.load(f)
    
    with open(thresholds_file) as f:
        thresholds = yaml.safe_load(f)
    
    report = {
        'passed': True,
        'violations': [],
        'warnings': [],
        'improvements': [],
    }
    
    # Check latency thresholds
    for scenario, metrics in results['results'].items():
        threshold = thresholds['scenarios'][scenario]
        
        if metrics['latency']['p95'] > threshold['p95_ms']:
            report['violations'].append({
                'scenario': scenario,
                'metric': 'p95_latency',
                'value': metrics['latency']['p95'],
                'threshold': threshold['p95_ms'],
            })
            report['passed'] = False
        
        if metrics['error_rate'] > threshold['max_error_rate']:
            report['violations'].append({
                'scenario': scenario,
                'metric': 'error_rate',
                'value': metrics['error_rate'],
                'threshold': threshold['max_error_rate'],
            })
            report['passed'] = False
    
    # Check resource utilization
    if results['resources']['cpu_max_percent'] > 90:
        report['warnings'].append({
            'metric': 'cpu_utilization',
            'value': results['resources']['cpu_max_percent'],
            'message': 'High CPU usage may indicate capacity issue',
        })
    
    if results['resources']['memory_growth_mb_per_hour'] > 50:
        report['violations'].append({
            'metric': 'memory_leak',
            'value': results['resources']['memory_growth_mb_per_hour'],
            'message': 'Potential memory leak detected',
        })
        report['passed'] = False
    
    return report, report['passed']

if __name__ == '__main__':
    report, passed = analyze_benchmark(
        sys.argv[1],  # results file
        sys.argv[2],  # thresholds file
    )
    
    print(json.dumps(report, indent=2))
    sys.exit(0 if passed else 1)
```

### Regression Detection

**Baseline Comparison**:

```python
# scripts/benchmark/detect_regression.py
def detect_regression(current_results, baseline_results, threshold_pct=10):
    """
    Compare current results against baseline.
    
    Args:
        current_results: Current test results
        baseline_results: Historical baseline results
        threshold_pct: Percentage degradation to trigger regression
    
    Returns:
        regressions: List of detected regressions
    """
    regressions = []
    
    for scenario in current_results['results']:
        current = current_results['results'][scenario]
        baseline = baseline_results['results'].get(scenario, {})
        
        if not baseline:
            continue
        
        # Check p95 latency regression
        p95_current = current['latency']['p95']
        p95_baseline = baseline['latency']['p95']
        p95_change_pct = ((p95_current - p95_baseline) / p95_baseline) * 100
        
        if p95_change_pct > threshold_pct:
            regressions.append({
                'scenario': scenario,
                'metric': 'p95_latency',
                'baseline': p95_baseline,
                'current': p95_current,
                'change_pct': p95_change_pct,
            })
        
        # Check throughput regression
        rps_current = current.get('rps_avg', 0)
        rps_baseline = baseline.get('rps_avg', 0)
        rps_change_pct = ((rps_current - rps_baseline) / rps_baseline) * 100
        
        if rps_change_pct < -threshold_pct:  # Negative = worse
            regressions.append({
                'scenario': scenario,
                'metric': 'throughput',
                'baseline': rps_baseline,
                'current': rps_current,
                'change_pct': rps_change_pct,
            })
    
    return regressions
```

---

## Regression Detection & Policy

### Policy

**Definition**: A regression is detected when:
1. **Latency**: p95 increases by >10% compared to baseline, OR
2. **Throughput**: RPS decreases by >10% compared to baseline, OR
3. **Error Rate**: Error rate increases by >0.5% absolute, OR
4. **Resource**: Memory growth >5% over 2h soak test

### Response

| Severity | Condition | Action |
|----------|-----------|--------|
| **Critical** | p95 >2× baseline OR error rate >5% | Block deployment, immediate investigation |
| **High** | p95 >50% baseline OR throughput <50% | Require approval to proceed |
| **Medium** | p95 >10% baseline OR throughput <10% | Create issue, notify team |
| **Low** | Warning but within thresholds | Log for trend analysis |

### Automated Enforcement

```yaml
# .github/workflows/performance-gate.yml
name: Performance Gate
on:
  pull_request:
  push:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run benchmark
        run: |
          k6 run --vus 100 --duration 5m \
            --out json=results.json \
            scripts/benchmark/k6/baseline_test.js
      
      - name: Detect regression
        run: |
          python scripts/benchmark/detect_regression.py \
            --current results.json \
            --baseline artifacts/benchmarks/baseline.json \
            --threshold 10 \
            --fail-on-regression
      
      - name: Comment on PR
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '⚠️ Performance regression detected. See workflow run for details.'
            })
```

---

## Document Relationships

This benchmark plan integrates with:

- **[Validation Plan](../VALIDATION_PLAN.md)**: References performance thresholds (p95 <200ms, p99 <500ms, error rate <0.5%)
- **[Production Readiness Checklist](./PRODUCTION_READINESS_CHECKLIST.md)**: Performance & resilience section validated by benchmark results
- **[Security Hardening Guide](./SECURITY_HARDENING_GUIDE.md)**: Security overhead measured in performance benchmarks
- **[Ethics Validation Framework](./ETHICS_VALIDATION_FRAMEWORK.md)**: Ethics evaluation latency benchmarked

---

**Version**: 1.0  
**Last Updated**: 2025-11-24  
**Status**: Comprehensive benchmark plan aligned with validation plan metrics  
**Next Review**: Quarterly (update scenarios based on production patterns)
