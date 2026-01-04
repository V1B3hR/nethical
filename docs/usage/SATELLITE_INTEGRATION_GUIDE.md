# Satellite Integration Guide

This guide covers the integration of Nethical with satellite connectivity systems for global edge deployments including maritime, aviation, and remote terrestrial locations.

## Overview

Nethical's satellite connectivity module provides:

- **Multiple Provider Support**: Starlink, OneWeb, Iridium, and future-ready Kuiper integration
- **GPS/GNSS Tracking**: Real-time positioning with geofencing capabilities
- **Automatic Failover**: Seamless switching between terrestrial and satellite links
- **Latency Optimization**: Adaptive handling of variable satellite delays
- **Offline Operation**: Local cache with sync-on-reconnect

## Supported Satellite Systems

### SpaceX Starlink (LEO)

**Best for**: High-bandwidth applications, consumer/enterprise connectivity

| Specification | Value |
|--------------|-------|
| Latency | 20-40ms typical, 100ms+ spikes |
| Bandwidth | 50-200 Mbps download |
| Coverage | Global (excluding polar >57°) |
| Orbit | LEO (~550km) |

```python
from nethical.connectivity.satellite import StarlinkProvider, ConnectionConfig

config = ConnectionConfig(
    endpoint="192.168.100.1",  # Dishy local address
    timeout_seconds=30.0,
    provider_options={
        "dish_address": "192.168.100.1",
        "grpc_port": 9200,
        "enable_ipv6": True,
    }
)

provider = StarlinkProvider(config)
await provider.connect()

# Get dish status
status = await provider.get_signal_info()
print(f"Latency: {status['latency_ms']}ms, Obstruction: {status['obstruction_percent']}%")
```

### OneWeb (LEO)

**Best for**: Enterprise/government, maritime, aviation

| Specification | Value |
|--------------|-------|
| Latency | ~32ms |
| Bandwidth | 50+ Mbps |
| Coverage | Global including polar |
| Orbit | LEO (~1,200km) |

```python
from nethical.connectivity.satellite import OneWebProvider

provider = OneWebProvider(config)
await provider.connect()
```

### Iridium (LEO)

**Best for**: Global coverage, safety-critical, low-bandwidth IoT

| Specification | Value |
|--------------|-------|
| Latency | 100-300ms |
| Bandwidth | 2.4 kbps (SBD), 704 kbps (Certus) |
| Coverage | True global (including poles) |
| Reliability | Very high |

```python
from nethical.connectivity.satellite import IridiumProvider

config = ConnectionConfig(
    provider_options={
        "serial_port": "/dev/ttyUSB0",
        "service_type": "sbd",  # or "certus"
    }
)

provider = IridiumProvider(config)
await provider.connect()

# Send SBD message (max 340 bytes)
message_id = await provider.send_sbd_message(b"Safety-critical data")
```

### Amazon Project Kuiper (Future)

**Status**: Under development - stub implementation available for future readiness

```python
from nethical.connectivity.satellite import KuiperProvider

provider = KuiperProvider(config)
availability = await provider.check_service_availability()
# Returns status and expected launch timeline
```

## GPS/GNSS Tracking

### Basic Positioning

```python
from nethical.connectivity.satellite import GPSTracker, GNSSConstellation

tracker = GPSTracker(
    constellations=[
        GNSSConstellation.GPS,
        GNSSConstellation.GLONASS,
        GNSSConstellation.GALILEO,
        GNSSConstellation.BEIDOU,
    ],
    update_interval_seconds=1.0,
)

await tracker.start_tracking()
position = await tracker.get_position()

print(f"Location: {position.latitude}, {position.longitude}")
print(f"Altitude: {position.altitude_m}m")
print(f"Accuracy: {position.horizontal_accuracy_m}m")
print(f"Satellites: {position.satellites_used} used, {position.satellites_visible} visible")
```

### Geofencing

```python
from nethical.connectivity.satellite import GeofenceType

# Create circular geofence
geofence = tracker.create_circular_geofence(
    fence_id="restricted_area_1",
    name="Restricted Zone",
    center_lat=37.7749,
    center_lon=-122.4194,
    radius_m=1000,  # 1km radius
)

# Register callbacks
def on_enter(fence, position):
    print(f"Entered {fence.name}")
    
def on_exit(fence, position):
    print(f"Exited {fence.name}")

tracker.register_callback("on_geofence_enter", on_enter)
tracker.register_callback("on_geofence_exit", on_exit)
```

### Location-Aware Routing

```python
# Calculate distance and bearing to destination
current = await tracker.get_position()
destination = Position(latitude=40.7128, longitude=-74.0060)  # NYC

distance = current.distance_to(destination)
bearing = current.bearing_to(destination)

print(f"Distance: {distance/1000:.1f}km, Bearing: {bearing:.1f}°")
```

## Automatic Failover

### Configuration

```python
from nethical.connectivity.satellite import (
    FailoverManager,
    FailoverConfig,
    StarlinkProvider,
)

config = FailoverConfig(
    # Thresholds
    latency_threshold_ms=500.0,
    packet_loss_threshold_percent=5.0,
    bandwidth_threshold_kbps=100.0,
    
    # Health checks
    health_check_interval_seconds=30.0,
    consecutive_failures_for_failover=3,
    
    # Failback behavior
    auto_failback=True,
    failback_delay_seconds=60.0,
    min_time_on_backup_seconds=30.0,
)

satellite_provider = StarlinkProvider()
failover = FailoverManager(config, satellite_provider)

# Start monitoring
await failover.start_monitoring()
```

### Failover Events

```python
def on_failover(event):
    print(f"Failover: {event.from_connection.value} -> {event.to_connection.value}")
    print(f"Reason: {event.reason.value}")
    print(f"Duration: {event.duration_ms}ms")

failover.register_callback("on_failover", on_failover)
```

### Manual Control

```python
# Force failover to satellite
await failover.force_failover_to_satellite()

# Force failback to terrestrial
await failover.force_failback_to_terrestrial()

# Check status
status = failover.get_status()
print(f"Active: {status['active_connection']}")
print(f"Satellite healthy: {status['satellite_healthy']}")
```

## Latency Optimization

### Adaptive Timeouts

```python
from nethical.connectivity.satellite import LatencyOptimizer, RequestPriority

optimizer = LatencyOptimizer()

# Record latency measurements
optimizer.record_measurement(latency_ms=35.0, success=True)
optimizer.record_measurement(latency_ms=42.0, success=True)
optimizer.record_measurement(latency_ms=150.0, success=True)  # Spike

# Get adaptive timeout
timeout = optimizer.get_adaptive_timeout(
    priority=RequestPriority.HIGH,
    payload_size_bytes=1024,
)
print(f"Recommended timeout: {timeout}ms")

# Check if request should be deferred
if optimizer.should_defer_request(RequestPriority.LOW):
    print("Consider deferring low-priority request")
```

### Request Batching

```python
# Queue requests for batch processing
await optimizer.queue_request(
    request_id="req-001",
    data=b"payload data",
    priority=RequestPriority.NORMAL,
)

# Register batch callback
def on_batch_ready(batch):
    print(f"Batch ready with {len(batch)} requests")
    for request in batch:
        # Process batched requests
        pass

optimizer.register_callback("on_batch_ready", on_batch_ready)
```

### Profile-Based Optimization

```python
# Get current latency profile
profile = optimizer.current_profile
print(f"Current profile: {profile.value}")  # e.g., "good", "degraded"

# Get optimization recommendations
recommendations = optimizer.get_optimization_recommendations()
for rec in recommendations['recommendations']:
    print(f"- {rec}")

# Get detailed statistics
stats = optimizer.get_statistics()
print(f"Average latency: {stats['mean_ms']:.1f}ms")
print(f"P95 latency: {stats['p95_ms']:.1f}ms")
```

## Connection Metrics

```python
from nethical.connectivity.satellite import SatelliteMetrics

metrics = SatelliteMetrics(sample_window_seconds=300)

# Record samples
metrics.record_sample(
    latency_ms=30.0,
    jitter_ms=5.0,
    signal_dbm=-65.0,
    packet_success=True,
)

# Get current metrics
current = metrics.get_current_metrics()
print(f"Latency: {current.latency_avg_ms}ms avg, {current.latency_p95_ms}ms p95")
print(f"Packet loss: {current.packet_loss_percent}%")
print(f"Signal quality: {current.signal_quality.value}")

# Get health assessment
health = metrics.get_health_assessment()
print(f"Status: {health['status']}")
print(f"Quality score: {health['quality_score']:.2f}")
for rec in health['recommendations']:
    print(f"  - {rec}")
```

## Kubernetes Deployment

### Satellite Overlay

Deploy Nethical with satellite-optimized settings:

```bash
kubectl apply -k deploy/kubernetes/multi-cluster/overlays/satellite/
```

This includes:
- Extended Istio timeouts (60s default)
- Aggressive retry policies (5 attempts, 15s per try)
- Connection pool optimizations for variable latency
- Offline-capable local caching

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NETHICAL_SATELLITE_MODE` | Enable satellite optimizations | `false` |
| `NETHICAL_LATENCY_TOLERANCE_MS` | Maximum acceptable latency | `500` |
| `NETHICAL_CACHE_TTL_MULTIPLIER` | TTL multiplier for satellite | `3` |
| `NETHICAL_OFFLINE_CACHE_ENABLED` | Enable offline caching | `true` |
| `NETHICAL_COMPRESSION_ENABLED` | Enable payload compression | `true` |

### Redis Edge Cache

For satellite edge nodes, deploy the lightweight Redis:

```bash
kubectl apply -f deploy/redis/satellite-edge.yaml
```

Features:
- 2GB memory footprint (vs 16GB for regional)
- Always-sync persistence for offline durability
- Extended timeouts for satellite latency
- Local-path storage for edge devices

## Best Practices

### 1. Design for Variable Latency

```python
# Always use adaptive timeouts
timeout = optimizer.get_adaptive_timeout(priority)

# Handle timeout gracefully
try:
    result = await asyncio.wait_for(operation, timeout=timeout/1000)
except asyncio.TimeoutError:
    # Fall back to cached data or queue for later
    result = satellite_cache.get(key)
```

### 2. Implement Request Prioritization

```python
# Prioritize safety-critical requests
await provider.send(data, priority=RequestPriority.URGENT)

# Defer non-critical operations during degraded conditions
if optimizer.current_profile in (LatencyProfile.POOR, LatencyProfile.CRITICAL):
    if request.priority == RequestPriority.LOW:
        queue_for_later(request)
```

### 3. Use Local Caching

```python
from nethical.cache import SatelliteCache, SatelliteCacheConfig

config = SatelliteCacheConfig(
    default_ttl_seconds=1800,  # 30 minutes
    compression_enabled=True,
    persistence_enabled=True,
)

cache = SatelliteCache(config)

# Write with automatic compression and persistence
cache.set("key", value)

# Read locally (works offline)
value = cache.get("key")

# Sync when back online
cache.is_online = True
await cache.sync_pending()
```

### 4. Handle Obstructions (Starlink)

```python
if isinstance(provider, StarlinkProvider):
    status = await provider.get_signal_info()
    
    if status['is_obstructed']:
        logger.warning(f"Obstruction: {status['obstruction_percent']:.1f}%")
        # Implement fallback logic
```

### 5. Monitor Connection Quality

```python
# Regular health checks
health = metrics.get_health_assessment()

if health['status'] == 'critical':
    # Trigger failover or alert
    await failover.force_failover_to_satellite()
```

## Troubleshooting

### Connection Issues

```python
# Check provider status
signal = await provider.get_signal_info()
print(f"Online: {signal.get('is_online', False)}")
print(f"Signal strength: {signal.get('signal_strength_dbm', 'N/A')} dBm")

# Check failover status
status = failover.get_status()
print(f"Terrestrial healthy: {status['terrestrial_healthy']}")
print(f"Satellite healthy: {status['satellite_healthy']}")
```

### High Latency

```python
# Check current conditions
stats = optimizer.get_statistics()
print(f"Current latency: {stats['mean_ms']:.1f}ms")
print(f"Trend: {'increasing' if optimizer.latency_trend > 0 else 'stable/decreasing'}")

# Get recommendations
recs = optimizer.get_optimization_recommendations()
for rec in recs['recommendations']:
    print(f"- {rec}")
```

### Cache Sync Issues

```python
# Check sync status
sync_status = cache.get_sync_status()
print(f"Pending: {sync_status['pending']}")
print(f"Conflicts: {sync_status['conflict']}")
print(f"Offline queue: {sync_status['offline_queue']}")

# Force sync
synced = await cache.sync_pending()
print(f"Synced {synced} entries")
```

## API Reference

See the module docstrings for complete API documentation:

- `nethical.connectivity.satellite.base` - Base classes and interfaces
- `nethical.connectivity.satellite.starlink` - Starlink provider
- `nethical.connectivity.satellite.oneweb` - OneWeb provider
- `nethical.connectivity.satellite.iridium` - Iridium provider
- `nethical.connectivity.satellite.kuiper` - Kuiper provider (stub)
- `nethical.connectivity.satellite.gps_tracker` - GPS/GNSS tracking
- `nethical.connectivity.satellite.failover` - Automatic failover
- `nethical.connectivity.satellite.latency_optimizer` - Latency optimization
- `nethical.connectivity.satellite.metrics` - Connection metrics
