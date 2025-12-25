# SELinux Configuration for Nethical

This directory contains SELinux policy modules for deploying Nethical on RHEL/CentOS/Fedora systems with mandatory access control.

## Overview

SELinux (Security-Enhanced Linux) provides mandatory access control (MAC) security mechanism that enforces access control policies defined by system administrators. This policy module restricts Nethical's access to system resources, providing defense-in-depth security.

## Files

- `nethical.te` - SELinux policy module source file
- `README.md` - This file

## Prerequisites

- Red Hat Enterprise Linux, CentOS, Fedora, or compatible distribution
- SELinux enabled and in enforcing mode
- Root or sudo access
- SELinux development tools

## Installation

### 1. Install SELinux Tools

```bash
# RHEL/CentOS 7-8
sudo yum install -y policycoreutils-python selinux-policy-devel

# RHEL/CentOS 9+ / Fedora
sudo dnf install -y policycoreutils-python-utils selinux-policy-devel
```

### 2. Compile the Policy Module

```bash
cd deploy/selinux

# Compile the policy
checkmodule -M -m -o nethical.mod nethical.te

# Package the module
semodule_package -o nethical.pp -m nethical.mod
```

### 3. Install the Policy Module

```bash
# Install the module
sudo semodule -i nethical.pp

# Verify installation
sudo semodule -l | grep nethical
```

### 4. Set File Contexts

Define SELinux contexts for Nethical files:

```bash
# Application executable
sudo semanage fcontext -a -t nethical_exec_t "/app/nethical.py"
sudo semanage fcontext -a -t nethical_exec_t "/usr/local/bin/nethical"

# Data directory
sudo semanage fcontext -a -t nethical_data_t "/data(/.*)?"

# Configuration directory
sudo semanage fcontext -a -t nethical_config_t "/app/config(/.*)?"

# Log directory
sudo semanage fcontext -a -t nethical_log_t "/var/log/nethical(/.*)?"

# Apply contexts
sudo restorecon -Rv /app /data /var/log/nethical
```

### 5. Verify Configuration

```bash
# Check SELinux status
sudo sestatus

# Check file contexts
ls -Z /app/nethical.py
ls -Z /data
ls -Z /app/config

# Check policy is loaded
sudo seinfo -t | grep nethical
```

## Docker/Container Deployment

For containerized deployments, you can use SELinux with Docker:

### Option 1: Use the Container SELinux Policy

```bash
# Run with SELinux enabled
docker run -d \
  --name nethical \
  --security-opt label=type:container_t \
  -v /data:/data:Z \
  nethical:latest
```

The `:Z` flag on the volume mount automatically sets the correct SELinux context for the container.

### Option 2: Custom SELinux Policy for Containers

```bash
# Create custom policy for containers
cat > nethical_container.te << 'EOF'
policy_module(nethical_container, 1.0.0)

require {
    type container_t;
    type container_file_t;
    class process { execmem };
}

# Allow Python to use executable memory (for JIT)
allow container_t self:process execmem;
EOF

# Compile and install
checkmodule -M -m -o nethical_container.mod nethical_container.te
semodule_package -o nethical_container.pp -m nethical_container.mod
sudo semodule -i nethical_container.pp
```

## Kubernetes with SELinux

For Kubernetes deployments with SELinux:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nethical
spec:
  securityContext:
    seLinuxOptions:
      type: nethical_t
      level: "s0:c123,c456"
  containers:
  - name: nethical
    image: nethical:latest
    securityContext:
      seLinuxOptions:
        type: nethical_t
        level: "s0:c123,c456"
    volumeMounts:
    - name: data
      mountPath: /data
      seLinuxOptions:
        type: nethical_data_t
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: nethical-data
```

## Troubleshooting

### Check SELinux Denials

```bash
# View recent SELinux denials
sudo ausearch -m avc -ts recent

# View denials for Nethical
sudo ausearch -m avc -c nethical

# Generate policy from denials (if needed)
sudo audit2allow -a -M nethical_local
sudo semodule -i nethical_local.pp
```

### Common Issues

**Issue:** Permission denied errors

```bash
# Check if SELinux is causing the denial
sudo tail /var/log/audit/audit.log | grep denied

# Temporarily set to permissive mode for testing
sudo setenforce 0

# Test application
./nethical.py

# Re-enable enforcing mode
sudo setenforce 1
```

**Issue:** Port binding failures

```bash
# Allow binding to custom port (e.g., 8000)
sudo semanage port -a -t http_port_t -p tcp 8000

# Or for any unreserved port
sudo semanage port -m -t unreserved_port_t -p tcp 8000
```

**Issue:** Database connection failures

```bash
# Allow network connections to database
sudo setsebool -P nis_enabled 1
sudo setsebool -P httpd_can_network_connect_db 1
```

### Generate Custom Policy from Audit Logs

If you encounter denials:

```bash
# Collect denials
sudo ausearch -m avc -ts recent > denials.txt

# Generate policy module
sudo audit2allow -M nethical_custom < denials.txt

# Review the generated policy
cat nethical_custom.te

# Install if appropriate
sudo semodule -i nethical_custom.pp
```

## Policy Maintenance

### Updating the Policy

```bash
# Modify nethical.te with your changes

# Increment version in policy_module() declaration
# policy_module(nethical, 1.0.1)

# Recompile and install
checkmodule -M -m -o nethical.mod nethical.te
semodule_package -o nethical.pp -m nethical.mod
sudo semodule -u nethical.pp
```

### Removing the Policy

```bash
# Remove the policy module
sudo semodule -r nethical

# Remove file contexts
sudo semanage fcontext -d "/app/nethical.py"
sudo semanage fcontext -d "/data(/.*)?"
sudo semanage fcontext -d "/app/config(/.*)?"
sudo semanage fcontext -d "/var/log/nethical(/.*)?"

# Restore default contexts
sudo restorecon -Rv /app /data /var/log/nethical
```

## Security Considerations

The Nethical SELinux policy implements several security controls:

1. **Process Isolation**: Nethical runs in its own security domain (`nethical_t`)
2. **File Access Control**: Restricted access to specific directories only
3. **Network Restrictions**: Only allowed network operations (HTTP/S)
4. **Capability Restrictions**: Minimal Linux capabilities
5. **No Privilege Escalation**: Prevents privilege escalation attempts
6. **Read-Only Configurations**: Configuration files are read-only
7. **Logging Isolation**: Separate logging context

## Testing

Test the policy in permissive mode first:

```bash
# Set domain to permissive
sudo semanage permissive -a nethical_t

# Run Nethical and test functionality
./run_tests.sh

# Check for denials
sudo ausearch -m avc -c nethical

# If all works correctly, set to enforcing
sudo semanage permissive -d nethical_t
```

## Additional Resources

- [SELinux Project](https://github.com/SELinuxProject)
- [Red Hat SELinux Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/using_selinux/)
- [CentOS SELinux HowTo](https://wiki.centos.org/HowTos/SELinux)
- [Fedora SELinux User Guide](https://docs.fedoraproject.org/en-US/quick-docs/getting-started-with-selinux/)

## Support

For SELinux-related issues:
1. Check the troubleshooting section above
2. Review audit logs: `sudo ausearch -m avc`
3. Consult the main [SECURITY.md](../../SECURITY.md) for reporting security issues
4. Open an issue on GitHub with SELinux audit logs attached

---

**Last Updated:** 2025-12-25
