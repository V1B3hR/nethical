# Nethical Monitoring & Alerting System

Production-grade observability and alerting for threat detection.

## Quick Start

```bash
# Install dependencies
pip install prometheus-client aiohttp

# Optional: For production profiling
pip install py-spy

# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access dashboards
# Grafana: http://localhost:3000 (admin/changeme)
# Prometheus: http://localhost:9090
# Metrics: http://localhost:9091/metrics
```

## Features

### 📊 Prometheus Metrics
- 15+ metric types tracking all aspects of threat detection
- Request latency histograms (P50, P95, P99)
- Threat detection rates by type and severity
- Model inference times
- Cache performance metrics
- Error tracking

### 🔔 Multi-Channel Alerting
- Slack, Email, PagerDuty, Discord, Custom webhooks
- Rate limiting to prevent alert storms
- Predefined alert rules for common scenarios
- Alert history and management

### 🔥 Profiling
- Production-safe flamegraph profiling with py-spy (<1% overhead)
- cProfile fallback for development
- Function, PID, and script profiling

### 📈 Grafana Dashboards
- Pre-built overview dashboard with 9 panels
- Auto-provisioned datasources
- Ready-to-use visualizations

## Architecture

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│   Nethical   │────▶│  Metrics       │────▶│  Prometheus  │
│   Detectors  │     │  Server:9091   │     │    :9090     │
└──────────────┘     └────────────────┘     └──────┬───────┘
                                                     │
                     ┌────────────────┐              │
                     │  Alert         │              │
                     │  Manager       │              │
                     └────────────────┘              │
                            │                        │
                            ▼                        ▼
                     ┌────────────────┐     ┌──────────────┐
                     │ Slack/Email/   │     │   Grafana    │
                     │ PagerDuty      │     │    :3000     │
                     └────────────────┘     └──────────────┘
```

## Usage

### Python Integration

```python
from nethical.monitoring import get_prometheus_metrics, start_metrics_server_async
from nethical.alerting import AlertManager, AlertSeverity, AlertChannel

# Start metrics server
server = await start_metrics_server_async(port=9091)

# Track metrics
metrics = get_prometheus_metrics()
metrics.track_request("prompt_injection", 0.042, "success")
metrics.track_threat("prompt_injection", "HIGH", "malicious", 0.95)

# Setup alerting
alert_manager = AlertManager({
    'enabled': True,
    'slack_webhook_url': 'YOUR_WEBHOOK_URL'
})

await alert_manager.send_alert(
    title="Critical Threat",
    message="High-severity threat detected",
    severity=AlertSeverity.CRITICAL,
    channels=[AlertChannel.SLACK]
)
```

### Docker Deployment

```bash
# Start full stack
docker-compose -f docker-compose.monitoring.yml up -d

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f

# Stop
docker-compose -f docker-compose.monitoring.yml down
```

## Configuration

See `config/monitoring.yaml` for full configuration options.

Key settings:
- `prometheus.enabled`: Enable Prometheus metrics
- `metrics_server.port`: Metrics HTTP server port (default: 9091)
- `alerting.enabled`: Enable alerting system
- `alerting.slack_webhook_url`: Slack webhook URL
- `profiling.sampling_rate_hz`: Profiling sample rate (default: 100)

## Metrics

### Request Metrics
- `nethical_requests_total` - Total requests by detector and status
- `nethical_request_latency_seconds` - Request latency histogram

### Threat Metrics
- `nethical_threats_detected_total` - Threats by type/level/category
- `nethical_confidence_score` - Detection confidence scores

### Performance Metrics
- `nethical_model_inference_seconds` - Model inference times
- `nethical_cache_hits_total` / `nethical_cache_misses_total` - Cache performance

### System Metrics
- `nethical_active_detectors` - Number of active detectors
- `nethical_errors_total` - Error counts by type
- `nethical_detector_health` - Detector health status (0/1)

## PromQL Examples

```promql
# Request rate per second
rate(nethical_requests_total[5m])

# P95 latency
histogram_quantile(0.95, rate(nethical_request_latency_seconds_bucket[5m]))

# Threat detection rate
rate(nethical_threats_detected_total[5m]) / rate(nethical_requests_total[5m])

# Cache hit rate
rate(nethical_cache_hits_total[5m]) / 
  (rate(nethical_cache_hits_total[5m]) + rate(nethical_cache_misses_total[5m]))
```

## Documentation

- [Complete Guide](../docs/monitoring-and-alerting.md) - Full documentation with troubleshooting
- [Example](../examples/monitoring/complete_monitoring_demo.py) - Complete working example
- [Tests](../tests/monitoring/) - Integration tests

## Performance Impact

- Prometheus metrics: **<0.1ms per operation**
- py-spy profiling: **<1% CPU overhead**
- Total monitoring overhead: **<0.5ms per request**

## Development

```bash
# Run tests
pytest tests/monitoring/ -v

# Run example
python examples/monitoring/complete_monitoring_demo.py

# Compile Python files
python -m py_compile nethical/monitoring/*.py nethical/alerting/*.py
```

## Troubleshooting

### Metrics not appearing
1. Check metrics server: `curl http://localhost:9091/health`
2. Verify Prometheus scraping: `curl http://localhost:9090/targets`
3. Check logs: `docker-compose logs nethical`

### Grafana not showing data
1. Test Prometheus datasource in Grafana
2. Verify Prometheus has data: `curl http://localhost:9090/api/v1/query?query=nethical_requests_total`
3. Check time range in Grafana

### Alerts not sending
1. Verify webhook URLs are configured
2. Test webhooks manually with curl
3. Check alert history in code

## Support

For issues or questions:
- [GitHub Issues](https://github.com/V1B3hR/nethical/issues)
- [Documentation](../docs/monitoring-and-alerting.md)
