# Security Incident Response Runbook

> **Version:** 1.0  
> **Last Updated:** 2025-12-03  
> **Status:** Active  
> **Compliance:** NIST 800-61, ISO 27035, GDPR Article 33

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Incident Classification](#incident-classification)
3. [Response Team](#response-team)
4. [Detection & Identification](#detection--identification)
5. [Containment Procedures](#containment-procedures)
6. [Eradication & Recovery](#eradication--recovery)
7. [Post-Incident Activities](#post-incident-activities)
8. [Specific Runbooks](#specific-runbooks)
9. [Contact Information](#contact-information)

---

## Overview

This runbook provides structured procedures for responding to security incidents in the Nethical platform. All team members should be familiar with these procedures.

### Fundamental Laws Alignment

This incident response process aligns with:
- **Law 2 (Right to Integrity)**: Protecting system integrity during incidents
- **Law 15 (Audit Compliance)**: Maintaining audit trails throughout response
- **Law 22 (Digital Security)**: Protecting digital assets during incidents
- **Law 23 (Fail-Safe Design)**: Ensuring safe degradation during incidents

### Response Objectives

1. **Minimize Impact**: Contain the incident to prevent further damage
2. **Preserve Evidence**: Maintain forensic integrity
3. **Restore Operations**: Return to normal operations safely
4. **Prevent Recurrence**: Implement improvements

---

## Incident Classification

### Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| **P1 - Critical** | System compromise, data breach, active attack | 15 minutes | Immediate |
| **P2 - High** | Significant vulnerability, partial compromise | 1 hour | Within 2 hours |
| **P3 - Medium** | Security anomaly, policy violation | 4 hours | Within 24 hours |
| **P4 - Low** | Minor security event, informational | 24 hours | Weekly review |

### Incident Categories

| Category | Examples | Typical Severity |
|----------|----------|------------------|
| **Data Breach** | Unauthorized data access, exfiltration | P1-P2 |
| **System Compromise** | Malware, unauthorized access | P1-P2 |
| **DDoS Attack** | Service disruption, resource exhaustion | P2-P3 |
| **Insider Threat** | Malicious employee activity | P1-P2 |
| **Policy Violation** | Unauthorized configuration change | P3-P4 |
| **Vulnerability** | Newly discovered security flaw | P2-P3 |

---

## Response Team

### Incident Response Team Structure

```
┌─────────────────────────────────────┐
│      Incident Commander (IC)        │
│   Overall coordination & decisions  │
└────────────────┬────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│Technical│ │ Comms   │ │ Legal & │
│  Lead   │ │  Lead   │ │Compliance│
└─────────┘ └─────────┘ └─────────┘
```

### Roles & Responsibilities

| Role | Responsibilities |
|------|------------------|
| **Incident Commander** | Overall coordination, escalation decisions, executive communication |
| **Technical Lead** | Technical investigation, containment, remediation |
| **Communications Lead** | Internal/external communications, stakeholder updates |
| **Legal/Compliance** | Regulatory notifications, legal guidance |

---

## Detection & Identification

### Step 1: Initial Alert Triage

**Timeframe**: 0-15 minutes

```yaml
triage_checklist:
  - [ ] Verify alert is not false positive
  - [ ] Identify affected systems/services
  - [ ] Determine initial severity classification
  - [ ] Create incident ticket
  - [ ] Notify on-call responder
```

### Step 2: Initial Assessment

**Timeframe**: 15-60 minutes

```yaml
assessment_checklist:
  - [ ] Confirm incident scope
  - [ ] Identify attack vector (if applicable)
  - [ ] Determine data/systems at risk
  - [ ] Document timeline of events
  - [ ] Collect initial evidence
```

### Detection Sources

| Source | Description | Location |
|--------|-------------|----------|
| Security Dashboard | Grafana security metrics | `/dashboards/security.json` |
| SIEM Alerts | Splunk/Elastic alerts | External SIEM |
| HSM Logs | Hardware security events | HSM console |
| TPM Attestation | Edge device integrity | Edge monitoring |
| Application Logs | Runtime security events | CloudWatch/ELK |

---

## Containment Procedures

### Immediate Containment (P1/P2)

```bash
# 1. Isolate affected systems
kubectl cordon <affected-node>
kubectl drain <affected-node> --ignore-daemonsets

# 2. Block suspicious IPs
# Add to WAF blocklist
aws wafv2 update-ip-set --name nethical-blocklist \
  --addresses <suspicious-ip>/32

# 3. Revoke compromised credentials
python -m nethical.cli security revoke-token --token-id <id>

# 4. Enable emergency mode
python -m nethical.cli admin set-mode --mode restricted
```

### Network Isolation

```yaml
containment_network:
  actions:
    - Block suspicious source IPs at WAF
    - Enable stricter rate limiting
    - Isolate affected pods/containers
    - Disable external API access if needed
    
  verification:
    - Confirm blocked connections in logs
    - Verify legitimate traffic still flows
    - Monitor for lateral movement
```

### Credential Containment

```yaml
containment_credentials:
  actions:
    - Rotate all API keys
    - Invalidate active sessions
    - Reset service account credentials
    - Update secrets in Vault
    
  hsm_actions:
    - Rotate HSM keys if compromised
    - Document key ceremony for new keys
    - Update all dependent services
```

---

## Eradication & Recovery

### Eradication Steps

```yaml
eradication:
  malware:
    - Identify all infected systems
    - Remove malicious files/processes
    - Patch exploited vulnerabilities
    - Verify removal with AV scan
    
  unauthorized_access:
    - Remove unauthorized accounts
    - Revoke all active sessions
    - Reset affected credentials
    - Review and revert configuration changes
    
  data_breach:
    - Identify exfiltration method
    - Close exfiltration path
    - Assess data scope
    - Prepare notification materials
```

### Recovery Procedures

```yaml
recovery:
  phase_1_validation:
    - Verify eradication complete
    - Confirm no persistence mechanisms
    - Test system integrity
    
  phase_2_restoration:
    - Restore from known-good backups if needed
    - Apply security patches
    - Reconfigure security controls
    
  phase_3_monitoring:
    - Enable enhanced monitoring
    - Watch for re-infection
    - Validate normal operations
```

---

## Post-Incident Activities

### Post-Incident Review

**Timeframe**: Within 5 business days

```yaml
post_incident_review:
  meeting:
    attendees:
      - Incident Response Team
      - Affected system owners
      - Security leadership
      
    agenda:
      - Timeline reconstruction
      - What worked well
      - What needs improvement
      - Action items
      
  documentation:
    - Complete incident report
    - Update runbooks if needed
    - Document lessons learned
    - Create improvement tickets
```

### Notification Requirements

| Incident Type | Notification Requirement | Timeframe |
|--------------|-------------------------|-----------|
| GDPR Data Breach | DPA notification | 72 hours |
| PCI Data Breach | PCI Council, acquirer | 72 hours |
| HIPAA Breach | HHS, affected individuals | 60 days |
| SOC 2 | Auditor notification | As agreed |

---

## Specific Runbooks

### [HSM Compromise](./hsm_compromise.md)

Procedures for responding to HSM security events.

### [TPM Attestation Failure](./tpm_attestation_failure.md)

Procedures for handling edge device integrity failures.

### [Data Breach](./data_breach.md)

Procedures for responding to unauthorized data access.

### [DDoS Attack](./ddos_attack.md)

Procedures for mitigating denial of service attacks.

### [Insider Threat](./insider_threat.md)

Procedures for handling malicious insider activity.

---

## Contact Information

### Escalation Path

| Priority | Contact | Method |
|----------|---------|--------|
| P1 | Security Team Lead | Phone + Slack |
| P2 | On-call Security | PagerDuty |
| P3 | Security Team | Slack #security |
| P4 | Security Team | Email |

### External Contacts

| Contact | Purpose | When to Contact |
|---------|---------|-----------------|
| Legal Counsel | Legal guidance | Data breach, subpoena |
| Cyber Insurance | Claim filing | Significant incident |
| Law Enforcement | Criminal activity | Evidence of crime |
| Regulators | Compliance notification | Breach notification required |

---

## Appendix A: Quick Reference Commands

```bash
# View current security status
python -m nethical.cli security status

# List active sessions
python -m nethical.cli security list-sessions

# Emergency lockdown
python -m nethical.cli admin lockdown --reason "security incident"

# View audit logs
python -m nethical.cli audit search --last 24h --severity critical

# HSM status
python -m nethical.cli hsm status

# TPM attestation check
python -m nethical.cli edge attest --device-id <id>
```

---

## Appendix B: Templates

### Incident Report Template

```markdown
# Incident Report: [INCIDENT-ID]

## Summary
- **Date/Time Detected**: 
- **Date/Time Contained**: 
- **Date/Time Resolved**: 
- **Severity**: 
- **Category**: 

## Description
[Brief description of the incident]

## Timeline
| Time | Event |
|------|-------|
| | |

## Impact
- Systems affected:
- Data affected:
- Users affected:

## Root Cause
[Analysis of root cause]

## Actions Taken
1. 
2. 
3. 

## Recommendations
1. 
2. 

## Lessons Learned
- 
```

---

**Document Owner:** Security Team  
**Review Cycle:** Quarterly  
**Next Review:** Q1 2026
