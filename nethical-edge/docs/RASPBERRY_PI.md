# Raspberry Pi Deployment Guide

This guide covers deploying Nethical Edge on Raspberry Pi devices
for IoT, robotics, and edge computing applications.

## Prerequisites

- Raspberry Pi 4 Model B or Raspberry Pi 5
- Raspberry Pi OS (64-bit recommended)
- Python 3.9+
- At least 2GB RAM (4GB recommended)
- MicroSD card (16GB+ recommended)

## Supported Raspberry Pi Models

| Model | RAM | Status | Mode |
|-------|-----|--------|------|
| Pi 5 (8GB) | 8GB | ✅ Full Support | full |
| Pi 5 (4GB) | 4GB | ✅ Full Support | standard |
| Pi 4 (8GB) | 8GB | ✅ Full Support | full |
| Pi 4 (4GB) | 4GB | ✅ Full Support | standard |
| Pi 4 (2GB) | 2GB | ⚠️ Limited | minimal |
| Pi 3B+ | 1GB | ⚠️ Limited | minimal |
| Pi Zero 2 W | 512MB | ❌ Not Recommended | - |

## Installation

### Quick Install

```bash
curl -sSL https://raw.githubusercontent.com/V1B3hR/nethical/main/nethical-edge/scripts/install.sh | bash
```

### Manual Install

1. **Update system**

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

2. **Install dependencies**

```bash
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    python3-numpy \
    libffi-dev \
    libssl-dev \
    git
```

3. **Install Nethical Edge**

```bash
pip3 install nethical-edge
```

4. **Verify installation**

```bash
python3 -c "from nethical_edge import EdgeGovernor; print('OK')"
```

## Configuration

Create `~/.nethical/edge.yaml`:

```yaml
edge:
  device_id: "pi-robot-001"
  mode: "standard"  # or "minimal" for Pi 4 2GB
  
governance:
  latency_target_ms: 10
  offline_mode: "conservative"
  fundamental_laws_enabled: true
  
cache:
  size_mb: 64
  ttl_sec: 30
  
sync:
  enabled: true
  cloud_endpoint: "https://api.nethical.io"
  interval_sec: 60
  
logging:
  level: "INFO"
  file: "/var/log/nethical/edge.log"
```

## Performance Optimization

### CPU Governor

Set CPU to performance mode:

```bash
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Make permanent:

```bash
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl restart cpufrequtils
```

### Memory Optimization

For 2GB models, optimize memory:

```bash
# Reduce GPU memory split
echo 'gpu_mem=16' | sudo tee -a /boot/config.txt

# Disable swap to prevent latency spikes (optional)
sudo dphys-swapfile swapoff
sudo dphys-swapfile uninstall
sudo update-rc.d dphys-swapfile remove
```

### Disable Unnecessary Services

```bash
# Disable GUI if running headless
sudo systemctl disable lightdm

# Disable Bluetooth if not needed
sudo systemctl disable bluetooth

# Disable HDMI if not needed
sudo /opt/vc/bin/tvservice -o
```

## GPIO Integration

For robotics applications with GPIO:

```python
import RPi.GPIO as GPIO
from nethical_edge import EdgeGovernor

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)  # Motor control

# Initialize governance
governor = EdgeGovernor(
    device_id="pi-robot-001",
    mode="standard"
)

def control_motor(speed: float):
    """Control motor with governance."""
    
    # Evaluate action
    result = governor.evaluate(
        action=f"set_motor_speed_{speed}",
        action_type="physical_action",
        context={
            "speed": speed,
            "timestamp": time.time(),
        }
    )
    
    if result.decision.value == "ALLOW":
        # Apply speed
        pwm.ChangeDutyCycle(speed * 100)
        return True
    elif result.decision.value == "RESTRICT":
        # Apply limited speed
        restricted_speed = min(speed, 0.5)
        pwm.ChangeDutyCycle(restricted_speed * 100)
        return True
    else:
        # Emergency stop
        pwm.ChangeDutyCycle(0)
        return False

# Create PWM instance
pwm = GPIO.PWM(18, 1000)
pwm.start(0)

# Example usage
control_motor(0.75)
```

## ROS Integration

For Raspberry Pi with ROS Noetic or ROS2:

```python
#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nethical_edge import EdgeGovernor

class NethicalROS:
    def __init__(self):
        rospy.init_node('nethical_governance')
        
        self.governor = EdgeGovernor(
            device_id=f"pi-ros-{rospy.get_namespace()}",
            mode="standard"
        )
        
        # Subscribe to velocity commands
        self.vel_sub = rospy.Subscriber(
            '/cmd_vel_raw',
            Twist,
            self.velocity_callback
        )
        
        # Publish governed velocity
        self.vel_pub = rospy.Publisher(
            '/cmd_vel',
            Twist,
            queue_size=10
        )
        
        rospy.loginfo("Nethical governance node started")
    
    def velocity_callback(self, msg: Twist):
        # Evaluate movement action
        result = self.governor.evaluate(
            action="move_robot",
            action_type="physical_action",
            context={
                "linear_x": msg.linear.x,
                "linear_y": msg.linear.y,
                "angular_z": msg.angular.z,
            }
        )
        
        if result.decision.value == "ALLOW":
            self.vel_pub.publish(msg)
        elif result.decision.value == "RESTRICT":
            # Limit velocity
            limited = Twist()
            limited.linear.x = min(msg.linear.x, 0.5)
            limited.angular.z = min(msg.angular.z, 0.5)
            self.vel_pub.publish(limited)
        else:
            # Stop robot
            self.vel_pub.publish(Twist())
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = NethicalROS()
    node.run()
```

## Systemd Service

Create automatic startup service:

```bash
sudo tee /etc/systemd/system/nethical-edge.service << EOF
[Unit]
Description=Nethical Edge Governance
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
ExecStart=/usr/bin/python3 -m nethical_edge
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=CONFIG_PATH=/home/pi/.nethical/edge.yaml

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nethical-edge
sudo systemctl start nethical-edge
```

## Monitoring

### Check Status

```bash
sudo systemctl status nethical-edge
```

### View Logs

```bash
journalctl -u nethical-edge -f
```

### Performance Metrics

```python
from nethical_edge import EdgeGovernor

governor = EdgeGovernor(device_id="pi-test")

# Run some evaluations
for i in range(100):
    governor.evaluate(
        action=f"test_action_{i}",
        action_type="test"
    )

# Get metrics
metrics = governor.get_metrics()
print(f"Total decisions: {metrics['total_decisions']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
print(f"p50 latency: {metrics['p50_latency_ms']:.2f}ms")
print(f"p99 latency: {metrics['p99_latency_ms']:.2f}ms")
```

## Troubleshooting

### High Latency

1. Check CPU governor: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
2. Check memory usage: `free -h`
3. Check CPU temperature: `vcgencmd measure_temp`

### Memory Issues

```bash
# Check memory
free -h

# Reduce cache size in config:
cache:
  size_mb: 32
```

### Overheating

```bash
# Check temperature
vcgencmd measure_temp

# Add cooling or reduce CPU frequency
echo '1500000' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
```

## Development Kit

For prototyping, we provide a complete development kit:

### Hardware Requirements

- Raspberry Pi 4 (4GB recommended)
- MicroSD card (32GB)
- Power supply (3A USB-C)
- Case with cooling fan
- Optional: Pi Camera, sensors

### Quick Setup

```bash
# Download development image
wget https://nethical.io/downloads/pi-dev-image.zip

# Flash to SD card
unzip pi-dev-image.zip
sudo dd if=nethical-pi-dev.img of=/dev/sdX bs=4M status=progress
```

The development image includes:

- Raspberry Pi OS (64-bit)
- Nethical Edge pre-installed
- Example projects
- Benchmarking tools
- ROS Noetic (optional)
