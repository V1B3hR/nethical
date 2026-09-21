# Monitoring and Alerting Guide

Complete guide for Nethical's production-grade monitoring, observability, and alerting system.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Prometheus Metrics](#prometheus-metrics)
- [Grafana Dashboards](#grafana-dashboards)
- [Alerting System](#alerting-system)
- [Profiling](#profiling)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)

## Overview

Nethical includes a comprehensive monitoring and alerting system with:

- **Prometheus Metrics**: 15+ metrics tracking requests, latency, threats, errors, cache performance
- **Grafana Dashboards**: Pre-built dashboards for overview, detectors, performance, and alerts
- **Multi-Channel Alerting**: Slack, Email, PagerDuty, Discord, and custom webhooks
- **Flamegraph Profiling**: Production-safe profiling with py-spy (<1% overhead)
- **Rate Limiting**: Prevents alert storms
- **Health Checks**: Automatic detector health monitoring

## Quick Start

### 1. Install Dependencies

```bash
pip install prometheus-client>=0.19.0 aiohttp>=3.9.0

# Optional: For production profiling
pip install py-spy>=0.3.14
```

### 2. Start Metrics Server

```python
from nethical.monitoring import start_metrics_server_async

# Start metrics server on port 9091
server = await start_metrics_server_async(port=9091)

# Metrics available at http://localhost:9091/metrics
```

### 3. Track Metrics

```python
from nethical.monitoring import get_prometheus_metrics

metrics = get_prometheus_metrics()

# Track a request
metrics.track_request(
    detector_type="prompt_injection",
    latency=0.042,  # seconds
    status="success"
)

# Track a threat detection
metrics.track_threat(
    detector_type="prompt_injection",
    threat_level="HIGH",
    category="malicious_prompt",
    confidence=0.95
)
```

### 4. Start Full Monitoring Stack

```bash
# Using Docker Compose
docker-compose -f docker-compose.monitoring.yml up -d

# Access dashboards
# Grafana: http://localhost:3000 (admin/changeme)
# Prometheus: http://localhost:9090
# Metrics: http://localhost:9091/metrics
```

## Prometheus Metrics

### Available Metrics

#### Request Metrics

```
nethical_requests_total{detector_type, status}
  Total number of threat detection requests
  Labels: detector_type, status (success/failure)

nethical_request_latency_seconds{detector_type}
  Request latency histogram
  Buckets: 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0
```

#### Threat Detection Metrics

```
nethical_threats_detected_total{detector_type, threat_level, threat_category}
  Total threats detected
  Labels: detector_type, threat_level, threat_category

nethical_confidence_score{detector_type}
  Confidence scores for threat detections
  Histogram with buckets: 0.0-1.0 (0.1 increments)

nethical_false_positives_total{detector_type}
  Total false positives reported
```

#### Performance Metrics

```
nethical_model_inference_seconds{model_name}
  Model inference time histogram
  Buckets: 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5

nethical_cache_hits_total{cache_type}
  Cache hits counter

nethical_cache_misses_total{cache_type}
  Cache misses counter
```

#### Error Metrics

```
nethical_errors_total{detector_type, error_type}
  Total errors by detector and type

nethical_detector_health{detector_type}
  Detector health status (0=unhealthy, 1=healthy)
```

#### System Metrics

```
nethical_active_detectors{detector_type}
  Number of active detector instances

nethical_queue_size{queue_name}
  Current queue size

nethical_throughput_requests_per_second{detector_type}
  Throughput summary
```

### Querying Metrics

#### Request Rate

```promql
# Requests per second by detector
rate(nethical_requests_total[5m])

# Success rate
rate(nethical_requests_total{status="success"}[5m]) / rate(nethical_requests_total[5m])
```

#### Latency Percentiles

```promql
# P50 latency
histogram_quantile(0.50, rate(nethical_request_latency_seconds_bucket[5m]))

# P95 latency
histogram_quantile(0.95, rate(nethical_request_latency_seconds_bucket[5m]))

# P99 latency
histogram_quantile(0.99, rate(nethical_request_latency_seconds_bucket[5m]))
```

#### Threat Detection Rate

```promql
# Threats per second
rate(nethical_threats_detected_total[5m])

# Threat rate (threats / total requests)
rate(nethical_threats_detected_total[5m]) / rate(nethical_requests_total[5m])
```

#### Cache Hit Rate

```promql
# Cache hit rate percentage
rate(nethical_cache_hits_total[5m]) / (rate(nethical_cache_hits_total[5m]) + rate(nethical_cache_misses_total[5m])) * 100
```

## Grafana Dashboards

### Available Dashboards

1. **Nethical Overview** (`nethical-overview.json`)
   - Requests per second by detector
   - Latency percentiles (P50, P95, P99)
   - Threats detected by type
   - Cache hit rate
   - Active detectors count
   - Error rate
   - Model inference time
   - Request latency heatmap
   - Confidence score distribution

2. **Detector Details** (Future)
   - Per-detector deep dive
   - Individual detector health
   - Detector-specific error rates

3. **Performance Deep Dive** (Future)
   - Detailed latency analysis
   - Resource utilization
   - Queue depths

4. **Active Alerts** (Future)
   - Real-time alert status
   - Alert history
   - Alert acknowledgments

### Accessing Dashboards

1. Open Grafana at `http://localhost:3000`
2. Login with credentials (default: admin/changeme)
3. Navigate to Dashboards → Nethical
4. Select desired dashboard

### Custom Queries

Use Prometheus as datasource and add custom panels with PromQL queries:

```promql
# Example: Top 5 detectors by request volume
topk(5, rate(nethical_requests_total[5m]))

# Example: Average latency by detector
avg(rate(nethical_request_latency_seconds_sum[5m]) / rate(nethical_request_latency_seconds_count[5m])) by (detector_type)
```

## Alerting System

### Multi-Channel Support

Nethical supports alerting through:

- **Slack**: Webhook-based alerts with rich formatting
- **Email**: SMTP-based email alerts with HTML formatting
- **PagerDuty**: Critical incident management
- **Discord**: Webhook-based Discord notifications
- **Custom Webhooks**: JSON POST to any endpoint

### Configuration

```python
from nethical.alerting import AlertManager, AlertSeverity, AlertChannel

# Configure alert manager
config = {
    'enabled': True,
    'max_alerts_per_minute': 10,
    'slack_webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
    'discord_webhook_url': 'https://discord.com/api/webhooks/YOUR/WEBHOOK',
    'pagerduty_api_key': 'your-pagerduty-key',
    'smtp': {
        'host': 'smtp.gmail.com',
        'port': 587,
        'use_tls': True,
        'username': 'your-email@gmail.com',
        'password': 'your-app-password',
        'from': 'alerts@nethical.ai',
        'to': 'security-team@company.com'
    }
}

alert_manager = AlertManager(config)
```

### Sending Alerts

```python
# Send a critical alert to multiple channels
await alert_manager.send_alert(
    title="Critical Threat Detected",
    message="High-severity threat detected in production",
    severity=AlertSeverity.CRITICAL,
    channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY, AlertChannel.EMAIL],
    metadata={
        'detector': 'prompt_injection',
        'confidence': '0.95',
        'threat_level': 'CRITICAL'
    }
)
```

### Predefined Alert Rules

```python
from nethical.alerting import AlertRules
from nethical.observability.metrics import get_metrics_collector

# Get current metrics
metrics_collector = get_metrics_collector()
metrics = metrics_collector.get_all_metrics()

# Evaluate all rules
await AlertRules.evaluate_all_rules(metrics, alert_manager)

# Or check individual rules
await AlertRules.check_high_latency(metrics, alert_manager, threshold_ms=200)
await AlertRules.check_high_threat_rate(metrics, alert_manager, threshold=0.5)
await AlertRules.check_error_rate(metrics, alert_manager, threshold=0.05)
```

### Rate Limiting

Alerts are automatically rate-limited to prevent storms:

- **INFO/WARNING**: Max 1 per minute per alert key
- **CRITICAL**: Always sent (not rate limited)
- Configurable via `max_alerts_per_minute`

## Profiling

### Production Profiling with py-spy

```python
from nethical.profiling import FlamegraphProfiler

profiler = FlamegraphProfiler(output_dir="profiling_results")

# Profile a running process
flamegraph_path = profiler.profile_pid(
    pid=12345,
    duration_seconds=60,
    rate_hz=100
)

# Profile a script
flamegraph_path = profiler.profile_script(
    script_path="your_script.py",
    duration_seconds=60,
    rate_hz=100
)
```

### Development Profiling with cProfile

```python
from nethical.profiling import FlamegraphProfiler

profiler = FlamegraphProfiler()

def your_function():
    # Your code here
    pass

# Profile the function
result, report_path = profiler.profile_sync(your_function)
print(f"Report saved to: {report_path}")
```

### Flamegraph Analysis

Flamegraphs show:
- **Width**: Time spent in function (wider = more time)
- **Height**: Call stack depth
- **Color**: Random (for differentiation)

Look for:
- Wide plateaus (hotspots)
- Unexpected deep stacks
- Surprisingly wide functions

## Configuration

### Environment Variables

```bash
# Metrics
export ENABLE_METRICS=true
export METRICS_PORT=9091

# Alerting
export ENABLE_ALERTING=true
export SLACK_WEBHOOK_URL=https://hooks.slack.com/...
export DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
export PAGERDUTY_API_KEY=your-key
export SMTP_USERNAME=your-email@gmail.com
export SMTP_PASSWORD=your-password

# Grafana
export GRAFANA_PASSWORD=your-secure-password
```

### Configuration File

See `config/monitoring.yaml` for full configuration options.

## Docker Deployment

### Using Docker Compose

```bash
# Start all monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f

# Stop services
docker-compose -f docker-compose.monitoring.yml down

# Stop and remove volumes (data loss!)
docker-compose -f docker-compose.monitoring.yml down -v
```

### Service URLs

- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Nethical Metrics: http://localhost:9091/metrics
- Nethical API: http://localhost:8000

### Data Persistence

Monitoring data is persisted in Docker volumes:
- `prometheus_data`: Prometheus time-series data
- `grafana_data`: Grafana dashboards and settings

## Troubleshooting

### Metrics Not Appearing

1. Check metrics server is running:
   ```bash
   curl http://localhost:9091/health
   ```

2. Verify Prometheus is scraping:
   ```bash
   curl http://localhost:9090/targets
   ```

3. Check for errors in logs:
   ```bash
   docker-compose -f docker-compose.monitoring.yml logs nethical
   ```

### Grafana Not Showing Data

1. Verify Prometheus datasource:
   - Grafana → Configuration → Data Sources
   - Test connection to Prometheus

2. Check Prometheus has data:
   ```bash
   curl http://localhost:9090/api/v1/query?query=nethical_requests_total
   ```

3. Verify time range in Grafana matches data

### Alerts Not Sending

1. Check AlertManager configuration:
   ```python
   alert_manager = AlertManager(config)
   print(alert_manager.enabled)  # Should be True
   ```

2. Test webhook URLs manually:
   ```bash
   curl -X POST https://hooks.slack.com/... -d '{"text":"test"}'
   ```

3. Check alert history:
   ```python
   history = alert_manager.get_alert_history(limit=10)
   for alert in history:
       print(alert)
   ```

### High Memory Usage

1. Reduce Prometheus retention:
   ```yaml
   # prometheus.yml
   storage.tsdb.retention.time: 15d  # Reduce from 30d
   ```

2. Reduce scrape frequency:
   ```yaml
   scrape_interval: 30s  # Increase from 15s
   ```

3. Limit metric cardinality (avoid high-cardinality labels)

### Performance Impact

Monitoring overhead is minimal:
- Prometheus metrics: <0.1ms per operation
- py-spy profiling: <1% CPU overhead
- cProfile: 5-10% CPU overhead (development only)

## Best Practices

1. **Set Appropriate Alert Thresholds**: Start conservative, adjust based on normal traffic
2. **Use Rate Limiting**: Prevent alert fatigue
3. **Monitor the Monitors**: Set up alerts for monitoring system health
4. **Regular Review**: Review dashboards and alerts weekly
5. **Profile in Production**: Use py-spy for production profiling
6. **Archive Old Data**: Clean up old profiling results regularly
7. **Document Custom Metrics**: If adding custom metrics, document in team wiki
8. **Test Alerting**: Regularly test alert channels
9. **Secure Credentials**: Use environment variables, never commit secrets
10. **Backup Grafana**: Export dashboard JSON regularly

## Support

For issues or questions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical/docs
