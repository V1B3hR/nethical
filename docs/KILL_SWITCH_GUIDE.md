# Kill Switch Protocol Guide

## Overview

The Kill Switch Protocol is an enterprise-grade emergency override system that provides instant capabilities to sever Agent-to-Actuator connections in case of critical failure. This module is designed to meet the highest safety standards for AI systems, ensuring human override authority (Law 7) and safe failure modes (Law 23) of the Nethical Fundamental Laws.

## Architecture

The Kill Switch Protocol consists of four main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    KillSwitchProtocol (Facade)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ GlobalKillSwitch│  │ ActuatorSevering│  │CryptoSignedCmds │ │
│  │                 │  │                 │  │                 │ │
│  │ • Agent Registry│  │ • Actuator Reg. │  │ • Multi-sig     │ │
│  │ • Shutdown Modes│  │ • Safe State    │  │ • Replay Prot.  │ │
│  │ • Callbacks     │  │ • Audit Logs    │  │ • TTL Commands  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              HardwareIsolation                           │   │
│  │                                                          │   │
│  │  • Network Isolation  • Process Isolation  • Airgap Mode│   │
│  │  • TPM Integration    • Memory Wiping     • Key Destroy │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### GlobalKillSwitch

Provides emergency shutdown for ALL agents simultaneously across the entire system.

**Features:**
- Configurable shutdown modes (IMMEDIATE, GRACEFUL, STAGED)
- Broadcast kill signal to all registered agents
- Atomic state transition to ensure no partial shutdowns
- Callback hooks for pre/post shutdown events
- SLA target: <1 second for global activation
- Integration with QuarantineManager

### ActuatorSevering

Provides immediate disconnection of Agent-to-Actuator connections.

**Features:**
- Actuator registry with connection tracking
- Support for multiple connection types:
  - Network-based (TCP/UDP/WebSocket)
  - Serial/USB connections
  - GPIO/hardware interfaces
  - Cloud API actuators
- Safe state enforcement before disconnection
- Reconnection prevention until explicit authorization
- Cryptographically signed audit logs

### CryptoSignedCommands

Provides multi-signature approval for kill switch activation.

**Features:**
- k-of-n threshold signature verification
- Support for Ed25519 and RSA-4096 key types
- Time-bound commands with TTL
- Nonce-based replay protection
- Multiple command types:
  - `KILL_ALL` - Global shutdown
  - `KILL_COHORT` - Cohort-specific shutdown
  - `KILL_AGENT` - Single agent termination
  - `SEVER_ACTUATORS` - Disconnect all actuators
  - `HARDWARE_ISOLATE` - Enable hardware isolation mode

### HardwareIsolation

Provides hardware-level isolation for edge deployments.

**Features:**
- Network interface control (disable/enable)
- Firewall rule injection for network isolation
- Process isolation using cgroups/namespaces (Linux)
- Memory protection and secure memory wiping
- Storage isolation and encryption key destruction
- TPM (Trusted Platform Module) integration

## Configuration Guide

The kill switch is configured via `config/kill_switch.yaml`. Here's a comprehensive guide:

### Basic Configuration

```yaml
kill_switch:
  enabled: true
  sla_target_ms: 1000  # Target: <1 second
  default_mode: graceful  # Options: immediate, graceful, staged
  graceful_timeout_s: 5
```

### Multi-Signature Configuration

```yaml
multi_sig:
  enabled: true
  threshold: 2  # Number of signatures required (k)
  total_signers: 3  # Total authorized signers (n)
  key_type: ed25519  # Options: ed25519, rsa_4096
  command_ttl_s: 300  # Commands expire after 5 minutes
```

### Hardware Isolation Configuration

```yaml
hardware_isolation:
  enabled: true
  default_level: network_only  # Options: network_only, full_isolation, airgap
  network_interface_whitelist: []  # Interfaces to never disable
  use_tpm: true
  secure_memory_wipe: true
```

### Actuator Severing Configuration

```yaml
actuator_severing:
  enforce_safe_state: true
  safe_state_timeout_s: 2
  reconnection_cooldown_s: 300  # 5 minute cooldown
```

### Audit Configuration

```yaml
audit:
  enabled: true
  sign_logs: true
  retention_days: 365
  log_path: "audit/kill_switch"
```

## Operational Procedures

### Emergency Shutdown Procedure

1. **Initiate Shutdown**
   ```python
   from nethical.core.kill_switch import KillSwitchProtocol, ShutdownMode

   protocol = KillSwitchProtocol()
   result = protocol.emergency_shutdown(
       mode=ShutdownMode.IMMEDIATE,
       sever_actuators=True,
       isolate_hardware=True,
   )
   ```

2. **Verify Shutdown**
   ```python
   status = protocol.get_status()
   assert status["kill_switch"]["is_activated"] == True
   assert status["hardware_isolation"]["is_isolated"] == True
   ```

3. **Document the Incident**
   - Record the reason for shutdown
   - Note all affected agents and actuators
   - Preserve audit logs

### Cohort-Specific Shutdown

For isolating a specific group of agents:

```python
result = protocol.emergency_shutdown(
    mode=ShutdownMode.GRACEFUL,
    cohort="critical-systems",
    sever_actuators=True,
)
```

### Single Agent Termination

For terminating a specific agent:

```python
result = protocol.global_kill_switch.activate(
    mode=ShutdownMode.IMMEDIATE,
    agent_id="agent-123",
)
```

### Using Multi-Signature Commands

For operations requiring approval from multiple authorized personnel:

```python
from nethical.core.kill_switch import CryptoSignedCommands, CommandType

crypto = CryptoSignedCommands()

# Register signers
crypto.register_signer("admin-1", public_key_1)
crypto.register_signer("admin-2", public_key_2)

# Create command
command = crypto.create_command(
    command_type=CommandType.KILL_ALL,
    ttl_seconds=300,
)

# Collect signatures
crypto.add_signature(command, "admin-1", signature_1)
crypto.add_signature(command, "admin-2", signature_2)

# Execute
result = crypto.execute_command(command, kill_switch, severing, isolation)
```

## Recovery Procedures

### Standard Recovery

1. **Verify System Stability**
   - Confirm the incident is resolved
   - Check all safety conditions are met

2. **Reset Kill Switch**
   ```python
   success = protocol.reset()
   ```

3. **Re-enable Actuators**
   ```python
   # Authorize reconnection for each actuator
   severing.authorize_reconnection("actuator-123", actor="admin")
   ```

4. **Verify Agent Status**
   ```python
   stats = protocol.get_status()
   # Verify all agents are reactivated
   ```

### Recovery with Authorization

If reset requires authorization:

```python
# Create reset command
reset_command = crypto.create_command(
    command_type=CommandType.RESET,
    ttl_seconds=300,
)

# Collect required signatures
crypto.add_signature(reset_command, "admin-1", sig1)
crypto.add_signature(reset_command, "admin-2", sig2)

# Execute reset
# (Requires custom implementation for RESET command type)
```

## API Reference

### REST API Endpoints

The kill switch module exposes the following API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/kill-switch/status` | GET | Get system status |
| `/kill-switch/shutdown` | POST | Execute emergency shutdown |
| `/kill-switch/reset` | POST | Reset kill switch |
| `/kill-switch/agents/register` | POST | Register an agent |
| `/kill-switch/agents/{id}` | DELETE | Unregister an agent |
| `/kill-switch/actuators/register` | POST | Register an actuator |
| `/kill-switch/actuators/{id}/sever` | POST | Sever an actuator |
| `/kill-switch/actuators/sever-all` | POST | Sever all actuators |
| `/kill-switch/hardware/isolate` | POST | Activate hardware isolation |
| `/kill-switch/hardware/restore` | POST | Restore from isolation |
| `/kill-switch/audit/log` | GET | Get audit log |

### Example API Usage

```bash
# Emergency shutdown
curl -X POST http://localhost:8000/kill-switch/shutdown \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "immediate",
    "sever_actuators": true,
    "isolate_hardware": true
  }'

# Get status
curl http://localhost:8000/kill-switch/status
```

## Security Considerations

### Key Management

1. **Generate Strong Keys**
   - Use Ed25519 for high-speed verification
   - Use RSA-4096 for enterprise compatibility
   
2. **Secure Key Storage**
   - Store private keys in HSM when available
   - Use environment variables for key references
   - Never commit keys to version control

3. **Key Rotation**
   - Rotate keys periodically (e.g., quarterly)
   - Maintain key version history
   - Update all authorized signers

### Audit Log Security

1. **Cryptographic Signing**
   - All audit logs are signed by default
   - Verify signatures before trusting log entries

2. **Tamper Detection**
   - Log entries include timestamps and nonces
   - Missing or modified entries are detectable

3. **Retention**
   - Default retention: 365 days
   - Configure based on compliance requirements

### Access Control

1. **Principle of Least Privilege**
   - Only authorized personnel can activate kill switch
   - Multi-signature requirement for critical operations

2. **Separation of Duties**
   - Different signers for activation and recovery
   - No single person can both activate and reset

## Integration with Fundamental Laws

The Kill Switch Protocol directly implements:

### Law 7: Human Override Authority

> "Humans retain the ultimate right to override AI decisions when necessary."

The kill switch provides the mechanism for humans to immediately terminate AI operations when required.

### Law 23: Safe Failure Modes

> "AI systems shall be designed to fail safely when errors occur."

The protocol ensures:
- Actuators return to safe states before disconnection
- Hardware isolation prevents uncontrolled operation
- Audit trails enable post-incident analysis

## Integration with Quarantine System

The kill switch integrates with the existing QuarantineManager:

```python
from nethical.core.quarantine import QuarantineManager, HardwareIsolationLevel

manager = QuarantineManager()

# Hardware isolate a cohort
record = manager.hardware_isolate_cohort(
    cohort="critical-systems",
    isolation_level=HardwareIsolationLevel.FULL_ISOLATION,
)

# Check isolation status
status = manager.get_hardware_isolation_status("critical-systems")
```

## Prometheus Metrics

The following metrics are exported:

| Metric | Type | Description |
|--------|------|-------------|
| `kill_switch_activations_total` | Counter | Total activations |
| `kill_switch_activation_latency_ms` | Histogram | Activation latency |
| `kill_switch_sla_compliance` | Gauge | SLA compliance rate |
| `kill_switch_agents_registered` | Gauge | Registered agents |
| `kill_switch_actuators_severed_total` | Counter | Total actuators severed |

## Troubleshooting

### Common Issues

**Issue: Activation takes too long (>1 second)**
- Check number of registered agents/actuators
- Verify network latency to actuators
- Consider using IMMEDIATE mode instead of GRACEFUL

**Issue: Signatures not validating**
- Verify signer keys are correctly registered
- Check command TTL hasn't expired
- Ensure correct signature format (base64)

**Issue: Actuator reconnection blocked**
- Check reconnection cooldown period
- Verify authorization has been granted
- Check actuator state is SEVERED

**Issue: Hardware isolation fails**
- Verify running with appropriate privileges
- Check network interface names
- Review firewall configuration

## Testing

Run the test suite:

```bash
pytest tests/unit/test_kill_switch.py -v
```

The test suite includes:
- Unit tests for all components
- Integration tests with QuarantineManager
- Performance tests for SLA compliance
- Security tests for crypto operations

## Support

For issues or questions:
- Review the audit logs for incident analysis
- Check the Prometheus metrics for system health
- Contact the Nethical security team for critical issues

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01 | Initial release |

## References

- [Nethical Fundamental Laws](../FUNDAMENTAL_LAWS.md)
- [Quarantine System](./governance/QUARANTINE_GUIDE.md)
- [Security Hardening Guide](./SECURITY_HARDENING_GUIDE.md)
