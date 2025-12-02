# Nethical Edge - Standalone Edge Deployment Package

A lightweight, standalone package for deploying Nethical governance on edge devices.
Designed for autonomous vehicles, industrial robots, medical devices, and other safety-critical systems.

## Features

- **Ultra-Low Latency**: <10ms p99 decision latency
- **Offline-First**: Full functionality without network connectivity
- **Minimal Footprint**: <256MB memory, optimized for embedded systems
- **Cross-Platform**: ARM64, x86_64, RISC-V support
- **Safety-Critical**: Fail-safe defaults, deterministic behavior

## Supported Platforms

| Platform | Architecture | Status | Memory |
|----------|-------------|--------|--------|
| NVIDIA Jetson | ARM64 | ✅ Supported | 256MB |
| Raspberry Pi 4/5 | ARM64 | ✅ Supported | 128MB |
| Intel NUC | x86_64 | ✅ Supported | 256MB |
| BeagleBone | ARM32 | ⚠️ Experimental | 64MB |
| RISC-V boards | RISC-V | 📋 Planned | TBD |

## Quick Start

### 1. Install

```bash
# Using pip (recommended)
pip install nethical-edge

# Using the install script
curl -sSL https://raw.githubusercontent.com/V1B3hR/nethical/main/nethical-edge/scripts/install.sh | bash

# From source
cd nethical-edge
pip install -e .
```

### 2. Configure

```yaml
# config/edge.yaml
edge:
  device_id: "my-robot-001"
  mode: "autonomous"
  
governance:
  latency_target_ms: 10
  offline_mode: "conservative"
  
sync:
  enabled: true
  cloud_endpoint: "https://api.nethical.io"
  interval_sec: 30
```

### 3. Run

```python
from nethical_edge import EdgeGovernor

# Initialize
governor = EdgeGovernor(
    device_id="my-robot-001",
    config_path="config/edge.yaml"
)

# Evaluate actions
result = governor.evaluate(
    action="move_arm_to_position",
    action_type="physical_action",
    context={"position": [100, 200, 50]}
)

if result.decision == "ALLOW":
    robot.execute_action()
elif result.decision == "RESTRICT":
    robot.execute_with_limits(result.restrictions)
else:
    robot.safe_stop()
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Edge Device                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Nethical Edge                        │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │   │
│  │  │  Policy    │  │  Decision  │  │    Offline     │  │   │
│  │  │  Cache     │  │  Engine    │  │    Fallback    │  │   │
│  │  │  (L1)      │  │  (JIT)     │  │    (CRDT)      │  │   │
│  │  └────────────┘  └────────────┘  └────────────────┘  │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐  │   │
│  │  │  Pattern   │  │   Risk     │  │     Sync       │  │   │
│  │  │  Profiler  │  │  Scorer    │  │    Manager     │  │   │
│  │  └────────────┘  └────────────┘  └────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                   │
│                           ▼                                   │
│                   Robot/AV Control                            │
└───────────────────────────┬─────────────────────────────────┘
                            │ (When online)
                            ▼
                    ┌───────────────┐
                    │  Cloud Sync   │
                    │  (CRDT-based) │
                    └───────────────┘
```

## Memory Optimization

Nethical Edge is optimized for resource-constrained environments:

```python
# Minimal mode for <64MB devices
governor = EdgeGovernor(
    device_id="sensor-001",
    mode="minimal",
    cache_size_mb=16,
    disable_jit=True  # Use pure Python
)

# Standard mode for 128-256MB devices
governor = EdgeGovernor(
    device_id="robot-001",
    mode="standard",
    cache_size_mb=64
)

# Full mode for >256MB devices
governor = EdgeGovernor(
    device_id="vehicle-001",
    mode="full",
    cache_size_mb=256,
    predictive_enabled=True
)
```

## Latency Targets

| Mode | p50 | p95 | p99 | Memory |
|------|-----|-----|-----|--------|
| Minimal | 2ms | 5ms | 10ms | 32MB |
| Standard | 1ms | 3ms | 8ms | 64MB |
| Full | 0.5ms | 2ms | 5ms | 128MB+ |

## Offline Mode

Nethical Edge operates in three offline modes:

1. **Conservative** (default): Only allow pre-approved actions
2. **Permissive**: Allow most actions, log for later review
3. **Emergency**: Minimal restrictions, safety-critical only

```python
# Configure offline behavior
governor.set_offline_mode("conservative")

# Check connectivity status
if not governor.is_connected():
    print("Operating in offline mode")
    print(f"Last sync: {governor.last_sync_time}")
```

## Cross-Compilation

### ARM64 (Raspberry Pi, Jetson)

```bash
# Install cross-compilation toolchain
./scripts/setup-cross-compile.sh arm64

# Build for ARM64
./scripts/build.sh --target arm64 --output dist/

# Deploy to device
scp dist/nethical-edge-arm64.tar.gz pi@raspberry:/opt/nethical/
```

### NVIDIA Jetson

```bash
# Use NVIDIA's optimized build
./scripts/build-jetson.sh --cuda

# Includes CUDA-accelerated risk scoring
```

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Configuration Reference](docs/CONFIGURATION.md)
- [API Reference](docs/API.md)
- [Jetson Deployment](docs/NVIDIA_JETSON.md)
- [Raspberry Pi Guide](docs/RASPBERRY_PI.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## License

Apache 2.0 - See [LICENSE](../LICENSE)
