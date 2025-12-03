# TPM Attestation Failure Response Runbook

> **Version:** 1.0  
> **Last Updated:** 2025-12-03  
> **Severity:** P2 - High  
> **Estimated Response Time:** 30-120 minutes

---

## Overview

This runbook provides procedures for responding to Trusted Platform Module (TPM) attestation failures on edge devices, indicating potential device tampering or compromise.

### Fundamental Laws Alignment

- **Law 2 (Right to Integrity)**: TPM verifies device integrity
- **Law 22 (Digital Security)**: TPM provides hardware-backed security
- **Law 23 (Fail-Safe Design)**: Safe mode on attestation failure

---

## Detection Triggers

| Trigger | Source | Severity |
|---------|--------|----------|
| PCR value mismatch | Remote attestation | P2 |
| Quote verification failure | Attestation service | P2 |
| Secure boot failure | Boot measurement | P1 |
| TPM unavailable | Device health check | P3 |
| Multiple attestation failures | Monitoring | P2 |

---

## Immediate Actions (0-30 minutes)

### Step 1: Identify Affected Device

```bash
# Check attestation status
python -m nethical.cli edge attest-status --device-id <device-id>

# List all failing devices
python -m nethical.cli edge list-failing-attestations

# View device details
python -m nethical.cli edge device-info --device-id <device-id>
```

### Step 2: Quarantine Device

```yaml
quarantine_steps:
  - [ ] Remove device from active policy sync
  - [ ] Block device API access
  - [ ] Enable safe mode on device
  - [ ] Notify device owner/operator
```

```bash
# Quarantine the device
python -m nethical.cli edge quarantine \
  --device-id <device-id> \
  --reason "attestation failure"

# Block device credentials
python -m nethical.cli edge revoke-credentials --device-id <device-id>
```

### Step 3: Assess Impact

```yaml
impact_assessment:
  questions:
    - Was the device processing sensitive data?
    - When did attestation last succeed?
    - Are other devices in same location affected?
    - What policies were synced to device?
    
  data_at_risk:
    - Cached policies
    - Runtime decisions
    - Local audit logs
    - Sealed secrets
```

---

## Investigation (30-120 minutes)

### Step 4: Collect Device Evidence

```yaml
evidence_collection:
  remote_collection:
    - Last known attestation quote
    - PCR history
    - Boot measurements
    - Network logs
    
  physical_collection:
    - Device physical inspection
    - BIOS/UEFI logs
    - Console output
    - Storage integrity
```

```bash
# Export device attestation history
python -m nethical.cli edge export-attestation \
  --device-id <device-id> \
  --output /tmp/attestation_evidence/

# Get detailed PCR comparison
python -m nethical.cli edge compare-pcrs \
  --device-id <device-id> \
  --baseline <baseline-id>
```

### Step 5: Root Cause Analysis

| PCR Index | Purpose | Common Failure Causes |
|-----------|---------|----------------------|
| PCR 0 | BIOS/firmware | Firmware update, tampering |
| PCR 1 | BIOS config | BIOS settings change |
| PCR 2 | Option ROMs | Hardware addition |
| PCR 4 | Boot manager | Bootloader update |
| PCR 7 | Secure boot | Certificate change |
| PCR 10+ | Application | Software update |

```yaml
analysis_checklist:
  legitimate_causes:
    - [ ] Authorized firmware update?
    - [ ] Hardware maintenance?
    - [ ] Operating system patch?
    - [ ] Configuration change?
    
  suspicious_indicators:
    - [ ] Unexpected PCR changes?
    - [ ] Physical access detected?
    - [ ] Unknown software installed?
    - [ ] Network anomalies?
```

---

## Resolution Paths

### Path A: Legitimate Change (Re-baseline)

If failure is due to authorized changes:

```bash
# Update device baseline
python -m nethical.cli edge update-baseline \
  --device-id <device-id> \
  --approve-current-state \
  --reason "Authorized firmware update" \
  --approved-by "admin@example.com"

# Remove quarantine
python -m nethical.cli edge unquarantine --device-id <device-id>

# Verify attestation
python -m nethical.cli edge attest --device-id <device-id>
```

### Path B: Device Compromise (Remediation)

If tampering is suspected:

```yaml
remediation_steps:
  phase_1_containment:
    - Keep device quarantined
    - Revoke all device secrets
    - Notify incident team
    
  phase_2_recovery:
    - Re-image device from known-good image
    - Re-provision TPM
    - Re-establish attestation baseline
    - Restore configuration from backup
    
  phase_3_validation:
    - Verify new attestation passes
    - Test all functionality
    - Re-enable device
```

```bash
# Re-provision device (after physical recovery)
python -m nethical.cli edge provision \
  --device-id <device-id> \
  --force-new-attestation-key

# Verify attestation
python -m nethical.cli edge attest --device-id <device-id>

# Restore to production
python -m nethical.cli edge unquarantine --device-id <device-id>
```

### Path C: TPM Hardware Failure

If TPM hardware has failed:

```yaml
hardware_failure_steps:
  - Document hardware failure
  - Replace TPM module (if possible)
  - Replace entire device (if necessary)
  - Re-provision new device
  - Update device inventory
```

---

## Safe Mode Operations

While a device is in safe mode:

```yaml
safe_mode_restrictions:
  allowed:
    - Read-only operations
    - Conservative policy defaults
    - Local decision logging
    
  blocked:
    - Policy updates
    - Configuration changes
    - Sensitive data access
    - External API calls
```

---

## Post-Incident

### Documentation

- Complete incident report
- Update device inventory
- Document root cause
- Update baseline if needed

### Improvements

- Review attestation monitoring
- Evaluate baseline update process
- Consider attestation frequency
- Review physical security

---

## Appendix: Device Types & Considerations

### Autonomous Vehicles

```yaml
av_considerations:
  safety_priority: Maximum
  response_time: Immediate
  safe_mode: Conservative driving only
  special_handling: Contact vehicle operations team
```

### Industrial Robots

```yaml
robot_considerations:
  safety_priority: High
  response_time: Within 1 hour
  safe_mode: Pause operations
  special_handling: Coordinate with plant safety
```

### Medical Devices

```yaml
medical_considerations:
  safety_priority: Maximum
  response_time: Immediate
  safe_mode: Clinical fallback
  special_handling: FDA reporting may be required
```

---

## Key Contacts

| Role | Contact |
|------|---------|
| Edge Security Team | edge-security@example.com |
| Device Operations | device-ops@example.com |
| Physical Security | physical-security@example.com |
