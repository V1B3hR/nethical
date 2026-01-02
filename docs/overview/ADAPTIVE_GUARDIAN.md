# Adaptive Guardian - Intelligent Throttling Security System

## Overview

The Adaptive Guardian is an intelligent, self-adjusting security monitoring system that automatically adapts its intensity based on the current threat landscape. Like a professional speedway racer who accelerates on straights and brakes into corners, it balances performance and security dynamically.

## Concept: "The Intelligent Speedway Racer"

```
SPRINT 🏎️  → CRUISE 🚗 → ALERT ⚠️ → DEFENSE 🛡️ → LOCKDOWN 🔒
   │            │           │           │            │
 0.02ms      0.05ms      0.2ms       1.0ms        10ms
   │            │           │           │            │
 "Full gas"  "Stable"   "Caution"  "Defensive"  "Red flag"
```

## Guardian Modes

| Mode | Overhead | Pulse Interval | Tripwire Sensitivity | Features |
|------|----------|----------------|---------------------|----------|
| **SPRINT** 🏎️ | ~0.02ms | 60s | CRITICAL only | Minimal checks, atomic counters only |
| **CRUISE** 🚗 | ~0.05ms | 30s | HIGH+ | Normal operation, balanced |
| **ALERT** ⚠️ | ~0.2ms | 10s | MEDIUM+ | Cross-module correlation active |
| **DEFENSE** 🛡️ | ~1.0ms | 5s | LOW+ | Deep inspection, ML detection |
| **LOCKDOWN** 🔒 | ~10ms | 1s | ALL | Full security, confirmation required |

## Automatic Mode Adaptation

The Guardian continuously analyzes the "track conditions" (threat landscape) and automatically switches modes based on:

- **Recent alerts** count and severity
- **Number of modules** with anomalies
- **Error rate** trends
- **Response time** trends
- **Cross-module correlation** score
- **External signals** (known attacks, maintenance windows)

Threat score ranges:
- 0.0-0.1 → SPRINT (clear track)
- 0.1-0.3 → CRUISE (normal)
- 0.3-0.6 → ALERT (suspicious activity)
- 0.6-0.8 → DEFENSE (active threats)
- 0.8-1.0 → LOCKDOWN (under attack)

## Layered Security Architecture

### Layer 1: Instant Tripwires (all modes)
Always-active checks with varying sensitivity:
- **Hard limits**: Response time > 5s = always alert
- **Critical decisions**: Detection of BLOCK/QUARANTINE with errors
- **Response time spikes**: 10x baseline increase
- **Error rate**: Sliding window monitoring

### Layer 2: Atomic Metrics (~0.01ms)
Fast counters using Welford's algorithm:
- Response time statistics
- Error counts
- Decision tracking
- Running mean/variance

### Layer 3: Pulse Analysis (background)
Interval varies by mode (1s to 60s):
- Statistical drift detection
- Cross-module correlation (ALERT+ modes)
- Trend analysis
- Threat score computation

### Layer 4: Watchdog (separate process)
Independent monitoring:
- Monitors Guardian itself
- Alerts if Guardian becomes unresponsive
- Heartbeat every 5 seconds
- "Who watches the watchmen?"

## Quick Start

### Basic Usage

```python
from nethical.security import (
    get_guardian,
    record_metric,
    trigger_lockdown,
    clear_lockdown,
    get_mode,
    get_status,
)

# Get the global Guardian instance
guardian = get_guardian()
guardian.start()

# Record a metric
record_metric(
    module="SafetyJudge",
    response_time_ms=45.2,
    decision="ALLOW",
    error=False,
)

# Check current mode
mode = get_mode()  # Returns GuardianMode enum
print(f"Current mode: {mode}")

# Get full status
status = get_status()
print(f"Threat score: {status['threat_analysis']['overall_score']:.3f}")

# Emergency lockdown
trigger_lockdown(reason="external_threat_intel")

# Clear lockdown
clear_lockdown()
```

### Using the @monitored Decorator

```python
from nethical.security import monitored

class SafetyJudge:
    @monitored("SafetyJudge")
    async def evaluate(self, action):
        # Your evaluation logic
        result = await self._evaluate_action(action)
        return result
```

The decorator automatically:
- Times the function execution
- Records response time
- Tracks errors
- Extracts decision from result

### Manual Integration

```python
import time
from nethical.security import record_metric

async def my_detector(action):
    start = time.perf_counter()
    error = False
    decision = "ALLOW"
    
    try:
        # Detection logic
        violations = await detect(action)
        decision = "BLOCK" if violations else "ALLOW"
        return violations
    except Exception as e:
        error = True
        raise
    finally:
        response_time_ms = (time.perf_counter() - start) * 1000
        record_metric("MyDetector", response_time_ms, decision, error)
```

## API Reference

### Core Functions

#### `get_guardian() -> AdaptiveGuardian`
Get the global Guardian singleton instance.

#### `record_metric(module, response_time_ms, decision="ALLOW", error=False) -> MetricRecord`
Record a metric for monitoring. Returns a record with any triggered alerts.

**Parameters:**
- `module` (str): Module name (e.g., "SafetyJudge")
- `response_time_ms` (float): Response time in milliseconds
- `decision` (str): Decision made (ALLOW, BLOCK, QUARANTINE, etc.)
- `error` (bool): Whether an error occurred

#### `trigger_lockdown(reason="manual")`
Trigger manual lockdown mode.

**Parameters:**
- `reason` (str): Reason for lockdown (for audit trail)

#### `clear_lockdown()`
Clear manual lockdown and return to automatic mode selection.

#### `get_mode() -> GuardianMode`
Get the current guardian mode.

**Returns:** One of SPRINT, CRUISE, ALERT, DEFENSE, LOCKDOWN

#### `get_status() -> Dict`
Get comprehensive Guardian status including:
- Current mode and configuration
- Threat analysis
- Performance metrics
- Statistics
- Watchdog status

### Classes

#### `AdaptiveGuardian`
Main guardian class. Usually accessed via singleton `get_guardian()`.

**Methods:**
- `start()`: Start background pulse analysis
- `stop()`: Stop the Guardian
- `record_metric(...)`: Record a metric
- `trigger_lockdown(reason)`: Manual lockdown
- `clear_lockdown()`: Clear manual lockdown
- `get_mode()`: Get current mode
- `get_status()`: Get full status
- `get_statistics()`: Get statistics

#### `GuardianMode` (Enum)
Guardian operational modes:
- `SPRINT`: Minimal overhead, clear track
- `CRUISE`: Normal operation, balanced
- `ALERT`: Suspicious activity detected
- `DEFENSE`: Active threats, deep inspection
- `LOCKDOWN`: Under attack, maximum security

## Integration Examples

### With Existing Detectors

```python
from nethical.detectors import BaseDetector
from nethical.security import monitored

class MyDetector(BaseDetector):
    @monitored("MyDetector")
    async def detect_violations(self, action):
        # Detection logic
        violations = []
        # ... your code ...
        return violations
```

### With SafetyJudge

```python
from nethical.judges import SafetyJudge
from nethical.security import monitored, get_mode, GuardianMode

class MonitoredSafetyJudge(SafetyJudge):
    @monitored("SafetyJudge")
    async def evaluate_action(self, action, violations):
        # Adapt behavior based on Guardian mode
        mode = get_mode()
        
        if mode == GuardianMode.SPRINT:
            # Fast path
            return await self._fast_evaluation(action, violations)
        elif mode == GuardianMode.LOCKDOWN:
            # Full evaluation
            return await self._full_evaluation(action, violations)
        else:
            # Normal
            return await super().evaluate_action(action, violations)
```

### With External Threat Intelligence

```python
from nethical.security import trigger_lockdown, clear_lockdown

class ThreatIntelService:
    async def process_threat_feed(self, threat_data):
        threat_level = threat_data.get("level", 0)
        
        if threat_level >= 0.9:
            trigger_lockdown(f"external_threat_level_{threat_level}")
        elif threat_level < 0.3:
            clear_lockdown()
```

## Statistics and Monitoring

The Guardian tracks comprehensive statistics:

```python
status = get_status()

# Current mode
print(f"Mode: {status['current_mode']} {status['mode_emoji']}")

# Threat analysis
threat = status['threat_analysis']
print(f"Threat Score: {threat['overall_score']:.3f}")
print(f"Alert Count: {threat['alert_count']}")
print(f"Error Rate: {threat['error_rate']:.1%}")

# Performance
perf = status['performance']
print(f"Avg Overhead: {perf['avg_overhead_ms']:.3f}ms")
print(f"Max Overhead: {perf['max_overhead_ms']:.3f}ms")

# Statistics
stats = status['statistics']
print(f"Total Metrics: {stats['total_metrics_recorded']}")
print(f"Total Alerts: {stats['total_alerts_triggered']}")
print(f"Manual Lockdowns: {stats['manual_lockdowns']}")
print(f"Automatic Lockdowns: {stats['automatic_lockdowns']}")

# Mode durations
for mode, duration in stats['mode_durations'].items():
    print(f"Time in {mode}: {duration:.1f}s")
```

## Performance

Target overhead by mode:
- **SPRINT**: <0.02ms per request
- **CRUISE**: <0.05ms per request
- **ALERT**: <0.2ms per request
- **DEFENSE**: <1.0ms per request
- **LOCKDOWN**: <10ms per request

Mode switching: <1ms

Watchdog heartbeat: Every 5 seconds

Memory: <10MB for all tracking data

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Adaptive Guardian                         │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Watchdog (separate thread)                        │
│  ├─ Monitors Guardian itself                                │
│  ├─ Heartbeat every 5s                                      │
│  └─ Alerts if Guardian unresponsive                         │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Pulse Analysis (background thread)                │
│  ├─ Track Analyzer                                          │
│  ├─ Threat Score Computation                                │
│  ├─ Mode Recommendation                                     │
│  └─ Interval: 1s-60s (mode-dependent)                       │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Atomic Metrics (~0.01ms)                          │
│  ├─ Welford's running statistics                            │
│  ├─ Response time tracking                                  │
│  └─ Error counting                                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Instant Tripwires (always active)                 │
│  ├─ Hard limits (5s response time)                          │
│  ├─ Response time spikes (10x)                              │
│  ├─ Error rate monitoring                                   │
│  └─ Critical decision detection                             │
└─────────────────────────────────────────────────────────────┘
```

## Design Philosophy

Like a professional speedway racer:

- **Sprint on straights**: When the track is clear, go full speed (minimal overhead)
- **Brake into corners**: When danger approaches, slow down and increase vigilance
- **Full stop for crashes**: When under attack, prioritize security over speed
- **Always wear safety gear**: Watchdog always running, critical tripwires always active

The system should be:
- ⚡ **Fast** when it can be
- 🛡️ **Secure** when it needs to be
- 🎯 **Decisive** in its responses
- 🔄 **Adaptive** to changing conditions

## Testing

Run comprehensive tests:

```bash
pytest tests/security/test_adaptive_guardian.py -v
```

Run specific test classes:

```bash
# Test guardian modes
pytest tests/security/test_adaptive_guardian.py::TestGuardianModes -v

# Test tripwires
pytest tests/security/test_adaptive_guardian.py::TestTripwires -v

# Test automatic transitions
pytest tests/security/test_adaptive_guardian.py::TestAutomaticModeTransitions -v

# Performance benchmarks
pytest tests/security/test_adaptive_guardian.py::TestPerformanceBenchmarks -v
```

## Example Application

See `examples/adaptive_guardian_integration.py` for a complete integration example showing:

1. Using the @monitored decorator
2. Manual metric recording
3. Integration with governance systems
4. External threat intelligence integration
5. Status dashboard
6. Best practices

Run the example:

```bash
python examples/adaptive_guardian_integration.py
```

## Best Practices

1. **Start Guardian early** in application lifecycle
2. **Use @monitored decorator** for simple monitoring
3. **Use record_metric()** for fine-grained control
4. **Monitor Guardian status** in dashboards
5. **Set up alerts** for watchdog failures
6. **Integrate with external threat intelligence**
7. **Test manual lockdown** procedures
8. **Monitor performance overhead** in production
9. **Track mode transitions** for capacity planning
10. **Use Guardian statistics** for security audits

## Troubleshooting

### Guardian not adapting modes

Check that the Guardian pulse thread is running:

```python
status = get_status()
print(status['watchdog']['running'])
```

Ensure you're not in manual lockdown:

```python
status = get_status()
if status['manual_lockdown']:
    clear_lockdown()
```

### High overhead

Check current mode:

```python
mode = get_mode()
print(f"Current mode: {mode}")
```

If in LOCKDOWN, investigate threat score:

```python
status = get_status()
threat = status['threat_analysis']
print(f"Threat score: {threat['overall_score']:.3f}")
print(f"Contributing factors: {threat['contributing_factors']}")
```

### Watchdog alerts

If watchdog reports Guardian unresponsive, check:

1. CPU availability
2. Thread starvation
3. Blocking operations in monitored code

## License

Part of the Nethical project. See main project license.

## Contributing

Contributions welcome! Areas for enhancement:

- Additional tripwire types
- ML-based threat prediction
- Integration with more detectors
- Performance optimization
- Dashboard UI

See CONTRIBUTING.md in the main project.
