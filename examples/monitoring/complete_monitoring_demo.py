"""Example: Complete monitoring and alerting setup for Nethical.

This example demonstrates:
1. Setting up Prometheus metrics
2. Starting the metrics server
3. Tracking various metrics
4. Configuring multi-channel alerting
5. Testing alert rules
"""

import asyncio
import time
import random
from nethical.monitoring import get_prometheus_metrics, start_metrics_server_async
from nethical.alerting import AlertManager, AlertSeverity, AlertChannel, AlertRules
from nethical.observability.metrics import get_metrics_collector


async def simulate_detector_activity(metrics, duration_seconds=30):
    """Simulate detector activity with metrics tracking."""
    print(f"\n📊 Simulating detector activity for {duration_seconds} seconds...")
    
    detector_types = ["prompt_injection", "deepfake", "shadow_ai", "polymorphic"]
    threat_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    start_time = time.time()
    request_count = 0
    
    while time.time() - start_time < duration_seconds:
        detector = random.choice(detector_types)
        
        # Simulate request processing
        latency = random.uniform(0.01, 0.5)
        await asyncio.sleep(latency)
        
        # Track request
        status = "success" if random.random() > 0.05 else "failure"
        metrics.track_request(detector, latency, status)
        request_count += 1
        
        # Randomly detect threats
        if random.random() > 0.7:  # 30% threat rate
            threat_level = random.choice(threat_levels)
            confidence = random.uniform(0.6, 0.99)
            metrics.track_threat(
                detector_type=detector,
                threat_level=threat_level,
                category="simulated_threat",
                confidence=confidence
            )
        
        # Simulate cache operations
        cache_hit = random.random() > 0.3
        metrics.track_cache("detection_cache", cache_hit)
        
        # Simulate model inference
        if random.random() > 0.5:
            inference_time = random.uniform(0.001, 0.1)
            metrics.track_model_inference(f"{detector}_model", inference_time)
        
        # Random errors
        if status == "failure":
            error_type = random.choice(["timeout", "validation_error", "model_error"])
            metrics.track_error(detector, error_type)
    
    print(f"✅ Simulated {request_count} requests")


async def setup_alerting():
    """Setup alerting with configuration."""
    print("\n🔔 Setting up alerting...")
    
    # Configure alert manager
    # NOTE: Set these environment variables for real alerts
    config = {
        'enabled': True,
        'max_alerts_per_minute': 10,
        # Uncomment and set these for real alerting:
        # 'slack_webhook_url': 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
        # 'discord_webhook_url': 'https://discord.com/api/webhooks/YOUR/WEBHOOK',
        # 'pagerduty_api_key': 'your-pagerduty-key',
        # 'smtp': {
        #     'host': 'smtp.gmail.com',
        #     'port': 587,
        #     'use_tls': True,
        #     'username': 'your-email@gmail.com',
        #     'password': 'your-app-password',
        #     'from': 'alerts@nethical.ai',
        #     'to': 'security-team@company.com'
        # }
    }
    
    alert_manager = AlertManager(config)
    
    # Send a test alert (won't actually send without webhook URLs)
    await alert_manager.send_alert(
        title="Monitoring System Started",
        message="Nethical monitoring and alerting system is now active",
        severity=AlertSeverity.INFO,
        channels=[AlertChannel.SLACK],
        metadata={
            'environment': 'demo',
            'version': '1.0.0'
        }
    )
    
    print("✅ Alerting configured")
    return alert_manager


async def check_alert_rules(alert_manager):
    """Check alert rules against current metrics."""
    print("\n🔍 Evaluating alert rules...")
    
    # Get current metrics
    metrics_collector = get_metrics_collector(enable_prometheus=False)
    
    # Simulate some metrics for testing
    metrics_collector.record_action("prompt_injection", "DENY", latency_seconds=0.25)
    metrics_collector.record_violation("prompt_injection", "HIGH", "injection_detector")
    
    metrics = metrics_collector.get_all_metrics()
    
    # Evaluate all alert rules
    await AlertRules.evaluate_all_rules(
        metrics,
        alert_manager,
        config={
            'high_latency_threshold_ms': 200,
            'high_threat_rate_threshold': 0.3,
            'high_error_rate_threshold': 0.05
        }
    )
    
    print("✅ Alert rules evaluated")


async def main():
    """Main monitoring demo."""
    print("=" * 60)
    print("Nethical Monitoring and Alerting Demo")
    print("=" * 60)
    
    # Step 1: Get Prometheus metrics instance
    print("\n1️⃣  Initializing Prometheus metrics...")
    metrics = get_prometheus_metrics()
    print("✅ Prometheus metrics initialized")
    
    # Step 2: Start metrics server
    print("\n2️⃣  Starting metrics server on port 9091...")
    server = await start_metrics_server_async(metrics=metrics, port=9091)
    print("✅ Metrics server started at http://localhost:9091/metrics")
    print("   Health check: http://localhost:9091/health")
    
    # Step 3: Set active detectors
    print("\n3️⃣  Registering active detectors...")
    for detector in ["prompt_injection", "deepfake", "shadow_ai", "polymorphic"]:
        metrics.set_active_detectors(detector, 1)
    print("✅ Detectors registered")
    
    # Step 4: Setup alerting
    alert_manager = await setup_alerting()
    
    # Step 5: Simulate activity
    await simulate_detector_activity(metrics, duration_seconds=30)
    
    # Step 6: Check alert rules
    await check_alert_rules(alert_manager)
    
    # Step 7: Display metrics
    print("\n📈 Current Metrics:")
    print(f"   Metrics endpoint: http://localhost:9091/metrics")
    print(f"   Grafana: http://localhost:3000 (if running)")
    print(f"   Prometheus: http://localhost:9090 (if running)")
    
    # Step 8: Keep server running
    print("\n⏳ Metrics server running. Press Ctrl+C to stop...")
    print("   Open http://localhost:9091/metrics in your browser to see metrics")
    
    try:
        # Keep running for 5 minutes
        await asyncio.sleep(300)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping metrics server...")
    
    # Cleanup
    await server.stop_async()
    print("✅ Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
