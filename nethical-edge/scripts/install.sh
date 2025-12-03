#!/bin/bash
# Nethical Edge - Installation Script
#
# Usage: curl -sSL https://raw.githubusercontent.com/V1B3hR/nethical/main/nethical-edge/scripts/install.sh | bash
#
# This script installs Nethical Edge on edge devices.

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Detect architecture
detect_arch() {
    local arch=$(uname -m)
    case $arch in
        x86_64)
            echo "x86_64"
            ;;
        aarch64|arm64)
            echo "arm64"
            ;;
        armv7l|armhf)
            echo "arm32"
            ;;
        riscv64)
            echo "riscv64"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Detect platform
detect_platform() {
    if [ -f /proc/device-tree/model ]; then
        local model=$(cat /proc/device-tree/model 2>/dev/null || echo "unknown")
        case "$model" in
            *"Raspberry Pi"*)
                echo "raspberry_pi"
                ;;
            *"NVIDIA Jetson"*)
                echo "jetson"
                ;;
            *"BeagleBone"*)
                echo "beaglebone"
                ;;
            *)
                echo "generic"
                ;;
        esac
    else
        echo "generic"
    fi
}

# Check system requirements
check_requirements() {
    log_step "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        local py_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ $(echo "$py_version >= 3.9" | bc -l) -eq 0 ]]; then
            log_error "Python 3.9+ required, found $py_version"
            exit 1
        fi
        log_info "Python $py_version found"
    else
        log_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
    
    # Check available memory
    local mem_mb=$(free -m | awk '/Mem:/ {print $2}')
    if [ "$mem_mb" -lt 256 ]; then
        log_warn "Low memory detected (${mem_mb}MB). Using minimal mode."
        INSTALL_MODE="minimal"
    elif [ "$mem_mb" -lt 512 ]; then
        log_info "Standard memory (${mem_mb}MB). Using standard mode."
        INSTALL_MODE="standard"
    else
        log_info "Sufficient memory (${mem_mb}MB). Using full mode."
        INSTALL_MODE="full"
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_warn "pip3 not found. Installing..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y python3-pip
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3-pip
        fi
    fi
}

# Install dependencies
install_dependencies() {
    log_step "Installing dependencies..."
    
    # System dependencies
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            python3-dev \
            python3-numpy \
            libffi-dev \
            libssl-dev
    fi
    
    # Python dependencies
    pip3 install --user --upgrade pip wheel
}

# Download and install Nethical Edge
install_nethical_edge() {
    log_step "Installing Nethical Edge..."
    
    local arch=$(detect_arch)
    local platform=$(detect_platform)
    
    log_info "Detected architecture: $arch"
    log_info "Detected platform: $platform"
    
    # Install from PyPI (preferred)
    if pip3 install --user nethical-edge 2>/dev/null; then
        log_info "Installed from PyPI"
        return 0
    fi
    
    # Fallback: Install from GitHub
    log_info "Installing from GitHub..."
    pip3 install --user git+https://github.com/V1B3hR/nethical.git#subdirectory=nethical-edge
}

# Configure for specific platform
configure_platform() {
    log_step "Configuring for platform..."
    
    local platform=$(detect_platform)
    
    case $platform in
        jetson)
            log_info "Configuring for NVIDIA Jetson..."
            # Enable CUDA if available
            if [ -d /usr/local/cuda ]; then
                echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
                echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
            fi
            ;;
        raspberry_pi)
            log_info "Configuring for Raspberry Pi..."
            # Optimize for Pi
            ;;
        *)
            log_info "Using generic configuration"
            ;;
    esac
}

# Create default configuration
create_config() {
    log_step "Creating default configuration..."
    
    local config_dir="$HOME/.nethical"
    mkdir -p "$config_dir"
    
    cat > "$config_dir/edge.yaml" << EOF
# Nethical Edge Configuration
edge:
  device_id: "$(hostname)"
  mode: "$INSTALL_MODE"
  
governance:
  latency_target_ms: 10
  offline_mode: "conservative"
  fundamental_laws_enabled: true
  
sync:
  enabled: true
  cloud_endpoint: "https://api.nethical.io"
  interval_sec: 30
  
cache:
  size_mb: 64
  ttl_sec: 30
  
logging:
  level: "INFO"
  file: "$config_dir/edge.log"
EOF
    
    log_info "Configuration created: $config_dir/edge.yaml"
}

# Create systemd service (optional)
create_service() {
    if [ "$(id -u)" -eq 0 ]; then
        log_step "Creating systemd service..."
        
        cat > /etc/systemd/system/nethical-edge.service << EOF
[Unit]
Description=Nethical Edge Governance Service
After=network.target

[Service]
Type=simple
User=$SUDO_USER
ExecStart=/usr/bin/python3 -m nethical_edge
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=CONFIG_PATH=$HOME/.nethical/edge.yaml

[Install]
WantedBy=multi-user.target
EOF
        
        systemctl daemon-reload
        log_info "Service created: nethical-edge.service"
        log_info "Enable with: sudo systemctl enable nethical-edge"
        log_info "Start with: sudo systemctl start nethical-edge"
    fi
}

# Verify installation
verify_installation() {
    log_step "Verifying installation..."
    
    python3 -c "
from nethical_edge import create_governor
governor = create_governor('test-device', mode='minimal')
print('Nethical Edge installation verified!')
print(f'Governor created for device: test-device')
" || {
        log_error "Installation verification failed"
        exit 1
    }
    
    log_info "Installation successful!"
}

# Main installation flow
main() {
    echo ""
    echo "=========================================="
    echo "     Nethical Edge Installation Script   "
    echo "=========================================="
    echo ""
    
    check_requirements
    install_dependencies
    install_nethical_edge
    configure_platform
    create_config
    
    if [ "${CREATE_SERVICE:-false}" = "true" ]; then
        create_service
    fi
    
    verify_installation
    
    echo ""
    echo "=========================================="
    echo "     Installation Complete!              "
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Edit configuration: ~/.nethical/edge.yaml"
    echo "  2. Run: python3 -m nethical_edge"
    echo ""
    echo "Documentation: https://github.com/V1B3hR/nethical/tree/main/nethical-edge"
    echo ""
}

# Run main
main "$@"
