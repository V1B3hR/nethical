# External Integrations Guide

This guide covers the external integrations available in Nethical for logging systems, ML platforms, and webhooks/APIs.

## Table of Contents

1. [Overview](#overview)
2. [Logging System Connectors](#logging-system-connectors)
3. [ML Platform Integrations](#ml-platform-integrations)
4. [Webhook and API Integrations](#webhook-and-api-integrations)
5. [Best Practices](#best-practices)
6. [Examples](#examples)

## Overview

Nethical provides comprehensive integration capabilities to connect with external systems:

- **Logging Connectors**: Send logs to syslog, AWS CloudWatch, JSON files
- **ML Platform Integrations**: Track experiments in MLflow, W&B, SageMaker
- **Webhook/API Integrations**: Send notifications to Slack, Discord, custom endpoints

All integrations are designed to be:
- Easy to configure
- Fault-tolerant with retry logic
- Minimal overhead
- Production-ready

## Logging System Connectors

### Available Connectors

#### 1. Syslog Connector

Send logs to local or remote syslog daemon (UDP/TCP).

```python
from nethical.integrations.logging_connectors import (
    SyslogConnector,
    LogLevel,
    LogEntry
)
from datetime import datetime

# Create syslog connector
connector = SyslogConnector(
    host='localhost',
    port=514,
    facility=16,  # LOG_LOCAL0
    protocol='UDP'
)

# Send log entry
entry = LogEntry(
    timestamp=datetime.now(),
    level=LogLevel.INFO,
    message="Application started",
    source="nethical",
    metadata={"version": "1.0.0"}
)

connector.send(entry)
connector.close()
```

#### 2. CloudWatch Connector

Send logs to AWS CloudWatch Logs (stub implementation).

```python
from nethical.integrations.logging_connectors import CloudWatchConnector

# Create CloudWatch connector
connector = CloudWatchConnector(
    log_group="nethical-logs",
    log_stream="production",
    region="us-east-1",
    batch_size=100
)

# Send logs (batched automatically)
for i in range(150):
    entry = LogEntry(
        timestamp=datetime.now(),
        level=LogLevel.INFO,
        message=f"Processing item {i}",
        source="worker"
    )
    connector.send(entry)

connector.close()
```

**Note**: This is a stub implementation. For production use, install boto3 and implement the commented code sections.

#### 3. JSON File Connector

Write structured logs to JSON file (one entry per line).

```python
from nethical.integrations.logging_connectors import JSONFileConnector

# Create file connector
connector = JSONFileConnector(
    filepath="logs/application.jsonl",
    buffer_size=10  # Flush after 10 entries
)

# Send logs
entry = LogEntry(
    timestamp=datetime.now(),
    level=LogLevel.ERROR,
    message="Database connection failed",
    source="database",
    metadata={"retries": 3, "error_code": "ECONNREFUSED"}
)

connector.send(entry)
connector.close()
```

### Log Aggregator

Send logs to multiple destinations simultaneously:

```python
from nethical.integrations.logging_connectors import LogAggregator

# Create aggregator
aggregator = LogAggregator()

# Add multiple connectors
aggregator.add_connector(SyslogConnector())
aggregator.add_connector(JSONFileConnector("logs/app.jsonl"))
aggregator.add_connector(CloudWatchConnector("my-group", "my-stream"))

# Log to all connectors at once
aggregator.log(
    level=LogLevel.WARNING,
    message="High memory usage detected",
    source="monitor",
    memory_mb=1500,
    threshold_mb=1000
)

# Cleanup
aggregator.close_all()
```

## ML Platform Integrations

### Available Platforms

#### 1. MLflow Integration

Track experiments in MLflow (stub implementation).

```python
from nethical.integrations.ml_platforms import MLflowIntegration

# Create integration
mlflow = MLflowIntegration(tracking_uri="http://localhost:5000")

# Start experiment run
run_id = mlflow.start_run(
    experiment_name="model_training",
    run_name="xgboost_v1"
)

# Log parameters
mlflow.log_parameters(run_id, {
    "model": "xgboost",
    "max_depth": 10,
    "learning_rate": 0.1,
    "n_estimators": 100
})

# Log metrics
mlflow.log_metrics(run_id, {
    "train_accuracy": 0.98,
    "val_accuracy": 0.95,
    "test_accuracy": 0.94
})

# Log artifacts
mlflow.log_artifact(run_id, "models/xgboost_model.pkl")
mlflow.log_artifact(run_id, "plots/confusion_matrix.png")

# End run
mlflow.end_run(run_id, status="completed")
```

**Note**: This is a stub implementation. For production use, install mlflow and implement the commented code sections.

#### 2. Weights & Biases (W&B) Integration

Track experiments in W&B (stub implementation).

```python
from nethical.integrations.ml_platforms import WandBIntegration

# Create integration
wandb = WandBIntegration(
    project="nethical-experiments",
    entity="my-team"
)

# Start run
run_id = wandb.start_run(
    experiment_name="neural_network_training",
    run_name="resnet50_v2"
)

# Log config
wandb.log_parameters(run_id, {
    "architecture": "resnet50",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 64
})

# Log metrics (can be called multiple times for same metric)
for epoch in range(10):
    wandb.log_metrics(run_id, {
        "epoch": epoch,
        "loss": 0.5 - epoch * 0.04,
        "accuracy": 0.7 + epoch * 0.02
    }, step=epoch)

# End run
wandb.end_run(run_id, "completed")
```

#### 3. SageMaker Integration

Track training jobs in AWS SageMaker (stub implementation).

```python
from nethical.integrations.ml_platforms import SageMakerIntegration

# Create integration
sagemaker = SageMakerIntegration(
    region="us-west-2",
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Start training job
job_id = sagemaker.start_run("fraud_detection_model")

# Log hyperparameters
sagemaker.log_parameters(job_id, {
    "max_depth": 5,
    "eta": 0.2,
    "gamma": 4,
    "subsample": 0.8
})

# Log metrics (sent to CloudWatch)
sagemaker.log_metrics(job_id, {
    "train:rmse": 0.05,
    "validation:rmse": 0.08
})

# Upload model to S3
sagemaker.log_artifact(job_id, "s3://my-bucket/models/fraud-model.tar.gz")

# Complete job
sagemaker.end_run(job_id, "completed")
```

### Multi-Platform Manager

Track experiments across multiple platforms simultaneously:

```python
from nethical.integrations.ml_platforms import MLPlatformManager

# Create manager
manager = MLPlatformManager()

# Add platforms
manager.add_platform("mlflow", MLflowIntegration())
manager.add_platform("wandb", WandBIntegration("my-project"))
manager.add_platform("sagemaker", SageMakerIntegration())

# Start runs on all platforms
run_ids = manager.start_run_all(
    experiment_name="ensemble_model",
    run_name="run_001"
)

# Log to all platforms
manager.log_metrics_all(run_ids, {
    "accuracy": 0.96,
    "f1_score": 0.94
})

# End all runs
manager.end_run_all(run_ids, "completed")
```

## Webhook and API Integrations

### Webhook Dispatchers

#### 1. Generic HTTP Webhook

Send POST requests with JSON payloads:

```python
from nethical.integrations.webhook import (
    HTTPWebhookDispatcher,
    WebhookPayload
)

# Create dispatcher
dispatcher = HTTPWebhookDispatcher(
    url="https://api.example.com/webhook",
    headers={"X-API-Key": "your-api-key"},
    timeout=30,
    max_retries=3,
    retry_delay=5
)

# Send webhook
payload = WebhookPayload(
    event_type="model_deployed",
    data={
        "model_id": "model_123",
        "version": "1.0.0",
        "accuracy": 0.95
    },
    metadata={"environment": "production"}
)

delivery = dispatcher.dispatch(payload)

print(f"Delivery status: {delivery.status}")
print(f"Response code: {delivery.response_code}")
```

#### 2. Slack Integration

Send messages and alerts to Slack:

```python
from nethical.integrations.webhook import SlackWebhookDispatcher

# Create Slack dispatcher
slack = SlackWebhookDispatcher(
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
)

# Send simple message
slack.send_message(
    text="Model training completed successfully!",
    username="Nethical Bot",
    icon_emoji=":robot_face:"
)

# Send formatted alert
slack.send_alert(
    title="Performance Degradation Detected",
    message="Model accuracy dropped below threshold",
    severity="warning",
    fields={
        "Model": "fraud_detector_v2",
        "Current Accuracy": "0.88",
        "Threshold": "0.90",
        "Environment": "production"
    }
)
```

#### 3. Discord Integration

Send messages to Discord:

```python
from nethical.integrations.webhook import DiscordWebhookDispatcher

# Create Discord dispatcher
discord = DiscordWebhookDispatcher(
    webhook_url="https://discord.com/api/webhooks/YOUR/WEBHOOK/URL"
)

# Send message
discord.send_message(
    content="🚀 New model deployed to production!",
    username="Nethical Bot",
    embeds=[{
        "title": "Deployment Details",
        "description": "Model v2.0 successfully deployed",
        "color": 0x00ff00,
        "fields": [
            {"name": "Version", "value": "2.0", "inline": True},
            {"name": "Accuracy", "value": "96%", "inline": True}
        ]
    }]
)
```

### Webhook Manager

Manage multiple webhooks and broadcast to all:

```python
from nethical.integrations.webhook import WebhookManager

# Create manager
manager = WebhookManager()

# Add webhooks
manager.add_webhook("slack", SlackWebhookDispatcher(slack_url))
manager.add_webhook("discord", DiscordWebhookDispatcher(discord_url))
manager.add_webhook("custom", HTTPWebhookDispatcher(custom_url))

# Broadcast to all webhooks
manager.broadcast(
    event_type="system_alert",
    data={
        "alert_type": "high_error_rate",
        "error_rate": 0.15,
        "threshold": 0.05
    },
    priority="high"
)

# Send to specific webhook
manager.send_to(
    webhook_name="slack",
    event_type="notification",
    data={"message": "Daily report ready"}
)
```

### Generic API Integration

Create custom API integrations:

```python
from nethical.integrations.webhook import GenericAPIIntegration

# Create API integration
api = GenericAPIIntegration(
    base_url="https://api.example.com",
    api_key="your-api-key",
    auth_header="Authorization"
)

# Authenticate
api.authenticate()

# Send data
result = api.send_data({
    "metric": "model_accuracy",
    "value": 0.95,
    "timestamp": "2024-12-10T10:00:00Z"
})

# Receive data
data = api.receive_data()
```

## Best Practices

### 1. Error Handling

Always handle potential failures:

```python
from nethical.integrations.logging_connectors import LogAggregator

aggregator = LogAggregator()

try:
    aggregator.add_connector(SyslogConnector())
    aggregator.log(LogLevel.INFO, "Application started", "app")
except Exception as e:
    print(f"Logging failed: {e}")
    # Fallback to local logging
finally:
    aggregator.close_all()
```

### 2. Use Aggregators/Managers

Centralize management of multiple integrations:

```python
# Good: Centralized management
manager = MLPlatformManager()
manager.add_platform("mlflow", MLflowIntegration())
manager.add_platform("wandb", WandBIntegration("project"))

run_ids = manager.start_run_all("experiment")
# ...
manager.end_run_all(run_ids)

# Bad: Managing manually
mlflow = MLflowIntegration()
wandb = WandBIntegration("project")
mlflow_run = mlflow.start_run("experiment")
wandb_run = wandb.start_run("experiment")
# ... more manual management
```

### 3. Graceful Degradation

Continue operation even if integrations fail:

```python
webhook_manager = WebhookManager()

try:
    webhook_manager.add_webhook("slack", SlackWebhookDispatcher(slack_url))
except Exception as e:
    print(f"Warning: Slack integration failed: {e}")
    # Continue without Slack

# Application continues regardless
```

### 4. Configuration Management

Use configuration files for integration settings:

```python
import json

# Load configuration
with open('config.json') as f:
    config = json.load(f)

# Setup integrations from config
if config.get('logging', {}).get('enabled'):
    aggregator = LogAggregator()
    
    if 'syslog' in config['logging']:
        aggregator.add_connector(SyslogConnector(**config['logging']['syslog']))
    
    if 'cloudwatch' in config['logging']:
        aggregator.add_connector(CloudWatchConnector(**config['logging']['cloudwatch']))
```

### 5. Testing

Test integrations in isolation:

```python
import pytest
from unittest.mock import patch

@patch('urllib.request.urlopen')
def test_webhook_dispatch(mock_urlopen):
    """Test webhook dispatching"""
    mock_response = MagicMock()
    mock_response.getcode.return_value = 200
    mock_urlopen.return_value.__enter__.return_value = mock_response
    
    dispatcher = HTTPWebhookDispatcher("https://test.com/webhook")
    payload = WebhookPayload("test_event", {"key": "value"})
    
    delivery = dispatcher.dispatch(payload)
    assert delivery.status == WebhookStatus.SUCCESS
```

## Examples

### Example 1: Complete Logging Setup

```python
from nethical.integrations.logging_connectors import (
    LogAggregator,
    SyslogConnector,
    JSONFileConnector,
    CloudWatchConnector,
    LogLevel
)

# Setup logging infrastructure
def setup_logging():
    aggregator = LogAggregator()
    
    # Local syslog for immediate visibility
    aggregator.add_connector(SyslogConnector(
        host='localhost',
        port=514,
        protocol='UDP'
    ))
    
    # JSON file for archival
    aggregator.add_connector(JSONFileConnector(
        filepath='logs/application.jsonl',
        buffer_size=10
    ))
    
    # CloudWatch for centralized monitoring
    aggregator.add_connector(CloudWatchConnector(
        log_group='production-logs',
        log_stream='nethical-app',
        region='us-east-1',
        batch_size=100
    ))
    
    return aggregator

# Use in application
aggregator = setup_logging()

aggregator.log(LogLevel.INFO, "Application started", "main", version="1.0.0")
aggregator.log(LogLevel.WARNING, "High memory usage", "monitor", memory_mb=1500)
aggregator.log(LogLevel.ERROR, "Database connection failed", "db", retries=3)

aggregator.close_all()
```

### Example 2: ML Experiment Tracking

```python
from nethical.integrations.ml_platforms import MLPlatformManager

def train_model_with_tracking():
    # Setup tracking
    manager = MLPlatformManager()
    manager.add_platform("mlflow", MLflowIntegration("http://localhost:5000"))
    manager.add_platform("wandb", WandBIntegration("ml-experiments", "team"))
    
    # Start experiment
    run_ids = manager.start_run_all("sentiment_analysis", "bert_finetune")
    
    # Log hyperparameters
    params = {
        "model": "bert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 32,
        "epochs": 3
    }
    
    for platform, run_id in run_ids.items():
        manager.platforms[platform].log_parameters(run_id, params)
    
    # Training loop
    for epoch in range(3):
        # ... training code ...
        
        metrics = {
            "epoch": epoch,
            "train_loss": 0.5 - epoch * 0.1,
            "val_accuracy": 0.8 + epoch * 0.05
        }
        
        manager.log_metrics_all(run_ids, metrics, step=epoch)
    
    # Save model
    for platform, run_id in run_ids.items():
        manager.platforms[platform].log_artifact(run_id, "models/bert_finetuned.pth")
    
    # Complete experiment
    manager.end_run_all(run_ids, "completed")

train_model_with_tracking()
```

### Example 3: Alert Notification System

```python
from nethical.integrations.webhook import WebhookManager

def setup_alerts():
    manager = WebhookManager()
    
    # Slack for team notifications
    manager.add_webhook("slack", SlackWebhookDispatcher(
        "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    ))
    
    # Discord for community
    manager.add_webhook("discord", DiscordWebhookDispatcher(
        "https://discord.com/api/webhooks/YOUR/WEBHOOK/URL"
    ))
    
    # Custom webhook for internal systems
    manager.add_webhook("internal", HTTPWebhookDispatcher(
        "https://internal.company.com/alerts"
    ))
    
    return manager

def send_alert(severity, title, message, **details):
    manager = setup_alerts()
    
    if severity == "critical":
        # Broadcast critical alerts everywhere
        manager.broadcast(
            event_type="critical_alert",
            data={
                "title": title,
                "message": message,
                "details": details
            },
            severity=severity
        )
    elif severity == "warning":
        # Send warnings only to Slack
        manager.send_to(
            "slack",
            event_type="warning_alert",
            data={"title": title, "message": message}
        )
    else:
        # Info messages to Discord only
        manager.send_to(
            "discord",
            event_type="info",
            data={"title": title, "message": message}
        )

# Usage
send_alert(
    severity="critical",
    title="Model Performance Degraded",
    message="Production model accuracy dropped to 85%",
    model="fraud_detector_v2",
    threshold="90%",
    current="85%"
)
```

### Example 4: Integrated Monitoring System

```python
from nethical.integrations.logging_connectors import LogAggregator
from nethical.integrations.webhook import WebhookManager

class MonitoringSystem:
    def __init__(self):
        # Setup logging
        self.log_aggregator = LogAggregator()
        self.log_aggregator.add_connector(
            JSONFileConnector("logs/monitoring.jsonl")
        )
        
        # Setup alerts
        self.alert_manager = WebhookManager()
        self.alert_manager.add_webhook(
            "slack",
            SlackWebhookDispatcher(SLACK_WEBHOOK_URL)
        )
    
    def log_metric(self, metric_name, value, **metadata):
        """Log a metric"""
        self.log_aggregator.log(
            LogLevel.INFO,
            f"Metric: {metric_name} = {value}",
            "metrics",
            metric=metric_name,
            value=value,
            **metadata
        )
    
    def check_threshold(self, metric_name, value, threshold, condition="<"):
        """Check metric against threshold and alert if needed"""
        alert_needed = False
        
        if condition == "<" and value < threshold:
            alert_needed = True
        elif condition == ">" and value > threshold:
            alert_needed = True
        
        if alert_needed:
            self.alert_manager.send_to(
                "slack",
                event_type="threshold_breach",
                data={
                    "metric": metric_name,
                    "value": value,
                    "threshold": threshold,
                    "condition": condition
                }
            )
            
            self.log_aggregator.log(
                LogLevel.WARNING,
                f"Threshold breach: {metric_name}",
                "alerts",
                metric=metric_name,
                value=value,
                threshold=threshold
            )
    
    def cleanup(self):
        """Cleanup resources"""
        self.log_aggregator.close_all()

# Usage
monitor = MonitoringSystem()

# Monitor model performance
monitor.log_metric("model_accuracy", 0.95, model="v2.0")
monitor.check_threshold("model_accuracy", 0.85, 0.90, condition="<")

monitor.cleanup()
```

## Troubleshooting

### Issue: Syslog Connection Failed

**Problem:** Cannot connect to syslog server.

**Solutions:**
1. Check if syslog daemon is running: `systemctl status rsyslog`
2. Verify firewall allows port 514
3. Try TCP instead of UDP
4. Check `/etc/rsyslog.conf` for remote logging settings

### Issue: CloudWatch Logs Not Appearing

**Problem:** Logs not showing in CloudWatch console.

**Solutions:**
1. Install boto3: `pip install boto3`
2. Implement the actual CloudWatch code (currently stub)
3. Verify AWS credentials are configured
4. Check IAM permissions for CloudWatch Logs
5. Verify log group and stream exist

### Issue: Webhook Delivery Failed

**Problem:** Webhooks returning errors or timing out.

**Solutions:**
1. Verify webhook URL is correct
2. Check network connectivity
3. Increase timeout value
4. Enable retries
5. Check webhook endpoint logs

## Related Documentation

- [Performance Profiling Guide](PERFORMANCE_PROFILING_GUIDE.md)
- [Testing Guide](TESTING_GUIDE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/V1B3hR/nethical/issues
- Documentation: https://github.com/V1B3hR/nethical/docs
