# NVIDIA Jetson Deployment Guide

This guide covers deploying Nethical Edge on NVIDIA Jetson devices
(Nano, TX2, Xavier, Orin) for autonomous vehicle and robotics applications.

## Prerequisites

- NVIDIA Jetson device with JetPack 5.0+ installed
- Python 3.8+
- At least 2GB available storage
- Network connectivity (for initial setup)

## Supported Jetson Devices

| Device | Memory | Status | Notes |
|--------|--------|--------|-------|
| Jetson Orin Nano | 4-8GB | ✅ Full Support | Recommended |
| Jetson Orin NX | 8-16GB | ✅ Full Support | Best performance |
| Jetson AGX Orin | 32-64GB | ✅ Full Support | Enterprise grade |
| Jetson Xavier NX | 8GB | ✅ Full Support | - |
| Jetson TX2 | 8GB | ✅ Full Support | - |
| Jetson Nano | 4GB | ⚠️ Limited | Use minimal mode |

## Installation

### Quick Install

```bash
# Install with CUDA support
curl -sSL https://raw.githubusercontent.com/V1B3hR/nethical/main/nethical-edge/scripts/install.sh | \
    CUDA_ENABLED=true bash
```

### Manual Install

1. **Update system packages**

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

2. **Install Python dependencies**

```bash
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-numpy \
    libffi-dev
```

3. **Install CUDA-optimized NumPy**

```bash
# For Jetson, use NVIDIA's optimized packages
pip3 install --upgrade pip
pip3 install numpy --extra-index-url https://developer.download.nvidia.com/compute/redist
```

4. **Install Nethical Edge**

```bash
pip3 install nethical-edge[cuda]
```

## Configuration

Create `/etc/nethical/edge.yaml`:

```yaml
edge:
  device_id: "jetson-av-001"
  mode: "full"
  platform: "jetson"
  
governance:
  latency_target_ms: 5
  offline_mode: "conservative"
  fundamental_laws_enabled: true
  
# CUDA acceleration
cuda:
  enabled: true
  device: 0
  memory_fraction: 0.3
  
# Risk scoring optimizations
risk_scoring:
  use_cuda: true
  batch_size: 32
  
# Pattern recognition
pattern_profiler:
  enabled: true
  use_tensorrt: true
  
cache:
  size_mb: 512
  use_gpu_memory: false  # Keep false for stability
  
sync:
  enabled: true
  cloud_endpoint: "https://api.nethical.io"
  interval_sec: 30
  priority: "high"
```

## CUDA Optimization

Nethical Edge can use CUDA for:

1. **Risk Score Calculation** - Matrix operations for risk scoring
2. **Pattern Matching** - Fast policy matching with GPU
3. **Feature Extraction** - Parallel feature extraction

### Enable CUDA Acceleration

```python
from nethical_edge import EdgeGovernor

governor = EdgeGovernor(
    device_id="jetson-av-001",
    cuda_enabled=True,
    cuda_device=0,
)

# Check CUDA status
print(f"CUDA enabled: {governor.cuda_enabled}")
print(f"CUDA device: {governor.cuda_device_name}")
```

### TensorRT Integration (Optional)

For even faster inference:

```bash
# Install TensorRT Python bindings
pip3 install tensorrt

# Enable in config
```

```yaml
cuda:
  use_tensorrt: true
  tensorrt_precision: "fp16"  # or "int8" for faster inference
```

## Performance Tuning

### Power Mode

Set the Jetson to maximum performance:

```bash
# For Orin
sudo nvpmodel -m 0
sudo jetson_clocks

# Verify
nvpmodel -q
```

### Memory Management

```python
# Limit GPU memory usage
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from nethical_edge import EdgeGovernor

governor = EdgeGovernor(
    device_id="jetson-av-001",
    cuda_memory_fraction=0.3,  # Use only 30% of GPU memory
)
```

### Latency Benchmarks

Run benchmarks on your device:

```bash
python3 -m nethical_edge.benchmarks --device jetson
```

Expected results on Jetson Orin:

| Operation | p50 | p95 | p99 |
|-----------|-----|-----|-----|
| Decision (cached) | 0.1ms | 0.3ms | 0.5ms |
| Decision (cold) | 0.5ms | 1.5ms | 3.0ms |
| Risk scoring (CUDA) | 0.2ms | 0.5ms | 1.0ms |
| Pattern match | 0.1ms | 0.2ms | 0.5ms |

## ROS2 Integration

For robotics applications with ROS2:

```python
import rclpy
from rclpy.node import Node
from nethical_edge import EdgeGovernor

class NethicalNode(Node):
    def __init__(self):
        super().__init__('nethical_governance')
        
        self.governor = EdgeGovernor(
            device_id=f"ros2-{self.get_namespace()}",
            mode="full",
            cuda_enabled=True,
        )
        
        # Subscribe to action requests
        self.action_sub = self.create_subscription(
            ActionRequest,
            '/robot/action_request',
            self.evaluate_action,
            10
        )
        
        # Publish governance decisions
        self.decision_pub = self.create_publisher(
            GovernanceDecision,
            '/robot/governance_decision',
            10
        )
    
    def evaluate_action(self, msg):
        result = self.governor.evaluate(
            action=msg.action,
            action_type=msg.action_type,
            context={
                'position': msg.position,
                'velocity': msg.velocity,
                'timestamp': msg.header.stamp.sec,
            }
        )
        
        decision = GovernanceDecision()
        decision.decision = result.decision.value
        decision.risk_score = result.risk_score
        decision.latency_ms = result.latency_ms
        
        self.decision_pub.publish(decision)

def main():
    rclpy.init()
    node = NethicalNode()
    rclpy.spin(node)
    rclpy.shutdown()
```

## Systemd Service

Create a systemd service for automatic startup:

```bash
sudo tee /etc/systemd/system/nethical-edge.service << EOF
[Unit]
Description=Nethical Edge Governance Service
After=network.target

[Service]
Type=simple
User=nvidia
ExecStart=/usr/bin/python3 -m nethical_edge --config /etc/nethical/edge.yaml
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nethical-edge
sudo systemctl start nethical-edge
```

## Troubleshooting

### CUDA Not Detected

```bash
# Check CUDA installation
nvcc --version
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall CUDA toolkit if needed
sudo apt-get install nvidia-cuda-toolkit
```

### High Latency

1. Check power mode: `nvpmodel -q`
2. Run `jetson_clocks` for maximum performance
3. Reduce cache size if memory constrained
4. Disable pattern profiler for minimal mode

### Memory Issues

```bash
# Check memory usage
free -h
tegrastats

# Reduce memory usage in config:
```

```yaml
cache:
  size_mb: 128  # Reduce from 512
cuda:
  memory_fraction: 0.2  # Reduce GPU memory
```

## Security Considerations

For safety-critical deployments:

1. **Secure Boot**: Enable Jetson secure boot
2. **Encryption**: Enable storage encryption
3. **Network**: Use VPN for cloud sync
4. **Updates**: Enable automatic security updates

```yaml
security:
  secure_boot: true
  encryption: true
  tls_verify: true
  min_tls_version: "1.3"
```
