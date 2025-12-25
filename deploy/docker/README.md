# Docker Security Configuration for Nethical

This directory contains security configurations for deploying Nethical with Docker.

## Files

- `apparmor-profile` - AppArmor profile for Docker containers
- `README.md` - This file

## Quick Start - Secure Docker Deployment

### Basic Secure Deployment

```bash
# Run with security hardening
docker run -d \
  --name nethical \
  --user 1000:1000 \
  --read-only \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --security-opt=no-new-privileges:true \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  -v ./data:/data:rw \
  -v ./config:/app/config:ro \
  -p 8000:8000 \
  nethical:latest
```

### With AppArmor Profile

1. **Install AppArmor utilities** (if not already installed):

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y apparmor-utils

# Fedora/RHEL
sudo dnf install -y apparmor-utils
```

2. **Load the AppArmor profile**:

```bash
# Copy profile to system location
sudo cp apparmor-profile /etc/apparmor.d/nethical-docker

# Load the profile
sudo apparmor_parser -r -W /etc/apparmor.d/nethical-docker

# Verify profile is loaded
sudo aa-status | grep nethical-docker
```

3. **Run container with AppArmor**:

```bash
docker run -d \
  --name nethical \
  --user 1000:1000 \
  --read-only \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --security-opt apparmor=nethical-docker \
  --security-opt=no-new-privileges:true \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  -v ./data:/data:rw \
  -v ./config:/app/config:ro \
  -p 8000:8000 \
  nethical:latest
```

## Docker Rootless Mode

Docker Rootless Mode provides additional security by running the Docker daemon and containers as a non-root user.

### Installation

```bash
# Install Docker Rootless
curl -fsSL https://get.docker.com/rootless | sh

# Configure environment variables
export PATH=/home/$USER/bin:$PATH
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock

# Add to shell profile for persistence
echo 'export PATH=/home/$USER/bin:$PATH' >> ~/.bashrc
echo 'export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock' >> ~/.bashrc
source ~/.bashrc
```

### Usage

```bash
# Start Docker daemon (rootless)
systemctl --user enable docker
systemctl --user start docker

# Run Nethical container (rootless)
docker run -d \
  --name nethical-rootless \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  -v ./data:/data:rw \
  -p 8000:8000 \
  nethical:latest
```

### Benefits

- ✅ No root privileges required for Docker daemon
- ✅ Improved container isolation
- ✅ Reduced attack surface
- ✅ Protection against container escape vulnerabilities

### Limitations

- ⚠️ Some storage drivers may not work (use fuse-overlayfs or vfs)
- ⚠️ Overlay networks not supported
- ⚠️ Port numbers below 1024 require additional configuration
- ⚠️ Slightly reduced performance compared to rootful mode

## Docker Compose Example

Create `docker-compose.yml` with security best practices:

```yaml
version: '3.8'

services:
  nethical:
    image: nethical:latest
    container_name: nethical
    user: "1000:1000"
    read_only: true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    security_opt:
      - no-new-privileges:true
      - apparmor=nethical-docker
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=100m
    volumes:
      - ./data:/data:rw
      - ./config:/app/config:ro
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - NETHICAL_SEMANTIC=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    networks:
      - nethical-net

networks:
  nethical-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

Run with:

```bash
docker-compose up -d
```

## Security Best Practices

### 1. User and Permissions

```bash
# Always run as non-root user
--user 1000:1000

# Or create custom user in Dockerfile
RUN useradd -m -u 1000 nethical
USER nethical
```

### 2. Filesystem Protection

```bash
# Read-only root filesystem
--read-only

# Writable tmpfs for temporary files
--tmpfs /tmp:rw,noexec,nosuid,size=100m

# Mount volumes with specific permissions
-v ./data:/data:rw        # Read-write data
-v ./config:/app/config:ro  # Read-only config
```

### 3. Capabilities

```bash
# Drop all capabilities
--cap-drop=ALL

# Add only required capabilities
--cap-add=NET_BIND_SERVICE  # For binding to ports
```

### 4. Security Options

```bash
# Prevent privilege escalation
--security-opt=no-new-privileges:true

# Use AppArmor profile
--security-opt apparmor=nethical-docker

# Use SELinux label (for RHEL/CentOS)
--security-opt label=type:container_t
```

### 5. Network Isolation

```bash
# Use custom network
docker network create nethical-net
--network nethical-net

# Or use host network (less secure, only if needed)
--network host
```

### 6. Resource Limits

```bash
# Limit CPU usage
--cpus="1.0"

# Limit memory
--memory="1g"
--memory-swap="1g"

# Limit PIDs (prevent fork bombs)
--pids-limit=100
```

### 7. Logging and Monitoring

```bash
# Configure logging
--log-driver json-file
--log-opt max-size=10m
--log-opt max-file=3

# Monitor container
docker stats nethical
docker logs -f nethical
```

## Verification

### Check Container Security

```bash
# Inspect security settings
docker inspect nethical --format '{{.State.Status}}'
docker inspect nethical --format '{{.HostConfig.SecurityOpt}}'
docker inspect nethical --format '{{.HostConfig.CapDrop}}'
docker inspect nethical --format '{{.HostConfig.ReadonlyRootfs}}'

# Check running as non-root
docker exec nethical id

# Verify AppArmor profile
docker exec nethical cat /proc/self/attr/current
```

### Run Security Scan

```bash
# Scan image for vulnerabilities
docker scan nethical:latest

# Or use Trivy
trivy image nethical:latest

# Or use Grype
grype nethical:latest
```

## Troubleshooting

### AppArmor Issues

```bash
# Check if AppArmor is enabled
sudo aa-status

# Check for denials
sudo dmesg | grep apparmor | grep DENIED

# Set profile to complain mode (for debugging)
sudo aa-complain /etc/apparmor.d/nethical-docker

# Re-enable enforce mode
sudo aa-enforce /etc/apparmor.d/nethical-docker
```

### Permission Issues

```bash
# Check volume permissions
ls -la ./data ./config

# Fix ownership (if needed)
sudo chown -R 1000:1000 ./data
sudo chmod -R 755 ./data

# For config (read-only)
sudo chown -R 1000:1000 ./config
sudo chmod -R 644 ./config
```

### Port Binding Issues

```bash
# Check if port is already in use
sudo netstat -tulpn | grep 8000

# Use different port
-p 8080:8000

# For rootless mode with privileged ports
echo "net.ipv4.ip_unprivileged_port_start=80" | sudo tee /etc/sysctl.d/99-rootless.conf
sudo sysctl --system
```

## Additional Resources

- [Docker Security](https://docs.docker.com/engine/security/)
- [AppArmor](https://gitlab.com/apparmor/apparmor/-/wikis/home)
- [Docker Rootless Mode](https://docs.docker.com/engine/security/rootless/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [Main Security Guide](../../docs/SECURITY_HARDENING.md)

---

**Last Updated:** 2025-12-25
