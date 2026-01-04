# HSM Compromise Response Runbook

> **Version:** 1.0  
> **Last Updated:** 2025-12-03  
> **Severity:** P1 - Critical  
> **Estimated Response Time:** 15-60 minutes

---

## Overview

This runbook provides procedures for responding to Hardware Security Module (HSM) compromise or suspected compromise events.

### Fundamental Laws Alignment

- **Law 2 (Right to Integrity)**: HSM protects system integrity
- **Law 22 (Digital Security)**: HSM provides cryptographic protection
- **Law 23 (Fail-Safe Design)**: Safe degradation during HSM issues

---

## Detection Triggers

| Trigger | Source | Action |
|---------|--------|--------|
| HSM tamper alert | HSM monitoring | Immediate investigation |
| Unauthorized HSM login | Audit logs | Immediate lockdown |
| HSM availability failure | Health checks | Failover + investigation |
| Key ceremony anomaly | Audit logs | Investigation |
| Signature verification failure | Application | Investigate HSM state |

---

## Immediate Actions (0-15 minutes)

### Step 1: Initial Assessment

```bash
# Check HSM status
python -m nethical.cli hsm status

# Review recent HSM operations
python -m nethical.cli hsm audit --last 1h

# Check for active sessions
python -m nethical.cli hsm sessions list
```

### Step 2: Isolate HSM

```yaml
isolation_steps:
  - [ ] Terminate all active HSM sessions
  - [ ] Disable HSM network access (if possible)
  - [ ] Enable HSM fallback mode (software crypto)
  - [ ] Alert Incident Commander
```

```bash
# Enable software fallback
python -m nethical.cli hsm failover --mode software

# Terminate HSM sessions
python -m nethical.cli hsm sessions terminate-all
```

---

## Investigation (15-60 minutes)

### Step 3: Evidence Collection

```yaml
evidence_collection:
  hsm_logs:
    - Export HSM audit logs
    - Preserve tamper indicators
    - Document physical access logs
    
  system_logs:
    - Application HSM operation logs
    - Network traffic to/from HSM
    - Authentication logs
    
  configuration:
    - HSM configuration backup
    - Key ceremony records
    - Access control lists
```

```bash
# Export HSM audit logs
python -m nethical.cli hsm export-logs --output /tmp/hsm_incident_logs/

# Capture system state
kubectl logs -l app=nethical-api > /tmp/nethical_api_logs.txt
```

### Step 4: Impact Assessment

```yaml
impact_assessment:
  questions:
    - Were any keys extracted?
    - Were any unauthorized operations performed?
    - Is the HSM physically compromised?
    - Are signatures from this period trustworthy?
    
  affected_systems:
    - Policy signing
    - Audit log Merkle root signing
    - JWT signing
    - Encryption keys
```

---

## Containment & Eradication

### Step 5: Key Rotation (If Compromise Confirmed)

**⚠️ Key rotation requires formal key ceremony**

```yaml
key_rotation_ceremony:
  preparation:
    - Notify all stakeholders
    - Schedule key ceremony
    - Prepare new key material
    - Document custodians and witnesses
    
  execution:
    - Generate new HSM keys
    - Sign new keys with hardware attestation
    - Update dependent systems
    - Revoke old keys
    
  verification:
    - Verify new key operations
    - Confirm old keys cannot sign
    - Update all applications
```

```bash
# Initiate key ceremony (requires manual steps)
python -m nethical.cli hsm ceremony start \
  --type rotation \
  --reason "security incident" \
  --custodians "alice,bob,charlie"

# After ceremony completion
python -m nethical.cli hsm ceremony complete \
  --ceremony-id <id> \
  --video-recording <recording-id>
```

### Step 6: System Recovery

```yaml
recovery_steps:
  - [ ] Restore HSM from known-good state (if needed)
  - [ ] Apply security patches
  - [ ] Re-establish HSM connectivity
  - [ ] Verify all cryptographic operations
  - [ ] Disable software fallback
```

```bash
# Verify HSM operations
python -m nethical.cli hsm verify-operations

# Disable software fallback
python -m nethical.cli hsm failover --mode hardware
```

---

## Post-Incident

### Step 7: Documentation

- Complete incident report
- Document key ceremony records
- Update key inventory
- Review and update access controls

### Step 8: Improvements

- Review HSM monitoring
- Evaluate physical security
- Update tamper detection
- Consider additional HSM redundancy

---

## Key Contacts

| Role | Contact |
|------|---------|
| HSM Administrator | hsm-admin@example.com |
| Security Team Lead | security@example.com |
| HSM Vendor Support | [Vendor contact] |

---

## Appendix: HSM Commands Reference

```bash
# Status and health
python -m nethical.cli hsm status
python -m nethical.cli hsm health-check

# Session management
python -m nethical.cli hsm sessions list
python -m nethical.cli hsm sessions terminate --session-id <id>

# Audit and logging
python -m nethical.cli hsm audit --last 24h
python -m nethical.cli hsm export-logs --output <path>

# Failover
python -m nethical.cli hsm failover --mode [hardware|software]

# Key management (requires ceremony)
python -m nethical.cli hsm ceremony start --type [generation|rotation|destruction]
python -m nethical.cli hsm ceremony complete --ceremony-id <id>
```
