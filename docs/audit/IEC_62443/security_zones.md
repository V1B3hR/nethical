# IEC 62443 Security Zones and Conduits

## Document Information

| Field | Value |
|-------|-------|
| Document ID | ZC-001 |
| Version | 1.0 |
| Date | 2025-12-03 |
| Author | Nethical Security Team |
| Status | Active |

## 1. Overview

This document defines the security zones and conduits for Nethical AI Governance deployments in industrial environments per IEC 62443-3-2.

## 2. Zone Definitions

### Zone 1: Enterprise Network

| Attribute | Value |
|-----------|-------|
| Zone ID | Z-ENT-001 |
| Security Level Target | SL-2 |
| Assets | Cloud management console, analytics, policy authoring |
| Connectivity | Internet, corporate network |
| Trust Level | Managed |

**Assets:**
- Nethical Cloud Services
- Policy Management Console
- Analytics Dashboard
- User Directory Integration

### Zone 2: Demilitarized Zone (DMZ)

| Attribute | Value |
|-----------|-------|
| Zone ID | Z-DMZ-001 |
| Security Level Target | SL-3 |
| Assets | API gateway, authentication proxy, logging collectors |
| Connectivity | Enterprise ↔ Industrial |
| Trust Level | Inspected |

**Assets:**
- API Gateway
- Web Application Firewall
- Authentication Proxy
- Log Aggregation

### Zone 3: Industrial Control Zone

| Attribute | Value |
|-----------|-------|
| Zone ID | Z-ICS-001 |
| Security Level Target | SL-3 |
| Assets | Edge governance engine, local policy cache, audit storage |
| Connectivity | DMZ, Safety Systems |
| Trust Level | High assurance |

**Assets:**
- Nethical Edge Governor
- TPM/HSM Security Module
- Local Policy Cache
- Audit Log Storage
- Offline Fallback System

### Zone 4: Safety Instrumented System (SIS)

| Attribute | Value |
|-----------|-------|
| Zone ID | Z-SIS-001 |
| Security Level Target | SL-4 |
| Assets | Robot control, emergency stop, safety PLCs |
| Connectivity | Industrial Control Zone only |
| Trust Level | Highest assurance |

**Note:** Zone 4 is managed by OEM/integrator; Nethical provides governance interface only.

## 3. Conduit Definitions

### Conduit C1: Enterprise to DMZ

| Attribute | Value |
|-----------|-------|
| Conduit ID | C-ENT-DMZ-001 |
| Source Zone | Enterprise |
| Destination Zone | DMZ |
| Allowed Traffic | HTTPS (443), SSH (22) |
| Security Controls | TLS 1.3, WAF, Rate Limiting |

**Data Flows:**
| Direction | Type | Protocol | Purpose |
|-----------|------|----------|---------|
| → | Management | HTTPS | Console access |
| ← | Telemetry | HTTPS | Metrics, logs |
| → | Configuration | HTTPS | Policy updates |

### Conduit C2: DMZ to Industrial Control

| Attribute | Value |
|-----------|-------|
| Conduit ID | C-DMZ-ICS-001 |
| Source Zone | DMZ |
| Destination Zone | Industrial Control |
| Allowed Traffic | gRPC (50051), NATS (4222) |
| Security Controls | mTLS, Certificate pinning |

**Data Flows:**
| Direction | Type | Protocol | Purpose |
|-----------|------|----------|---------|
| → | Sync | gRPC | Policy sync |
| ← | Audit | NATS | Event streaming |
| ↔ | Health | gRPC | Heartbeat |

### Conduit C3: Industrial Control to Safety System

| Attribute | Value |
|-----------|-------|
| Conduit ID | C-ICS-SIS-001 |
| Source Zone | Industrial Control |
| Destination Zone | Safety System |
| Allowed Traffic | OPC-UA (4840), Safety protocol |
| Security Controls | Hardware isolation, allowlist |

**Data Flows:**
| Direction | Type | Protocol | Purpose |
|-----------|------|----------|---------|
| → | Governance | Custom | Decision output |
| ← | Action | Custom | Action requests |

## 4. Zone Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   INTERNET                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                              [External Firewall]
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ZONE 1: ENTERPRISE (SL-2)                            │
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │
│   │   Cloud     │   │   Policy    │   │  Analytics  │   │    User     │   │
│   │  Services   │   │   Console   │   │  Dashboard  │   │  Directory  │   │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                              ══════════════════
                              │  CONDUIT C1   │
                              │ TLS 1.3, WAF  │
                              ══════════════════
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ZONE 2: DMZ (SL-3)                                 │
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │
│   │     API     │   │     WAF     │   │    Auth     │   │     Log     │   │
│   │   Gateway   │   │             │   │    Proxy    │   │  Collector  │   │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                              ══════════════════
                              │  CONDUIT C2   │
                              │ mTLS, CertPin │
                              ══════════════════
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                   ZONE 3: INDUSTRIAL CONTROL (SL-3)                         │
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │
│   │    Edge     │   │   TPM/HSM   │   │   Policy    │   │   Audit     │   │
│   │  Governor   │   │   Module    │   │   Cache     │   │   Storage   │   │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   │
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐                                        │
│   │   Offline   │   │    Safe     │                                        │
│   │   Fallback  │   │   Defaults  │                                        │
│   └─────────────┘   └─────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                              ══════════════════
                              │  CONDUIT C3   │
                              │ HW Isolation  │
                              ══════════════════
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ZONE 4: SAFETY SYSTEM (SL-4)                           │
│                           [OEM Responsibility]                              │
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │
│   │  Robot AI   │   │  Safety PLC │   │  Actuators  │   │   Sensors   │   │
│   │   Engine    │   │  (E-Stop)   │   │             │   │             │   │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 5. Security Controls by Zone

### Zone 1: Enterprise

| Control | Implementation |
|---------|---------------|
| Authentication | SSO + MFA |
| Authorization | RBAC |
| Encryption | TLS 1.3 in transit, AES-256 at rest |
| Monitoring | SIEM integration |
| Access Control | Network ACLs |

### Zone 2: DMZ

| Control | Implementation |
|---------|---------------|
| Firewall | Stateful inspection |
| WAF | OWASP rule set |
| Rate Limiting | Per-client limits |
| Intrusion Detection | Network IDS |
| Certificate Validation | Strict chain verification |

### Zone 3: Industrial Control

| Control | Implementation |
|---------|---------------|
| Device Authentication | TPM attestation |
| Integrity Verification | Secure boot |
| Key Storage | HSM/TPM |
| Offline Capability | Local policy cache |
| Audit Logging | Tamper-evident logs |
| Safe Defaults | Fail-secure operation |

### Zone 4: Safety System

| Control | OEM Requirement |
|---------|----------------|
| Physical Isolation | Air gap for E-Stop |
| Hardwired Safety | Independent of software |
| SIL Certification | Per IEC 61508 |
| Redundancy | Dual-channel |

## 6. Conduit Security Controls

### C1: Enterprise → DMZ

| Control | Configuration |
|---------|--------------|
| Protocol | HTTPS only |
| TLS Version | 1.3 minimum |
| Cipher Suites | AEAD only |
| Certificate | Public CA with OCSP |
| Rate Limit | 1000 req/min per client |

### C2: DMZ → Industrial Control

| Control | Configuration |
|---------|--------------|
| Protocol | gRPC over TLS |
| Authentication | Mutual TLS |
| Certificate | Private CA with CRL |
| Pinning | Certificate hash pinning |
| Heartbeat | 30 second interval |
| Timeout | 5 second connection |

### C3: Industrial Control → Safety System

| Control | Configuration |
|---------|--------------|
| Protocol | OPC-UA or custom |
| Physical | Dedicated interface |
| Allowlist | Fixed endpoints only |
| Latency | < 5ms |
| Fallback | Hardware override |

## 7. Risk Assessment

### Zone Risk Levels

| Zone | Threats | Impact | Likelihood | Risk |
|------|---------|--------|------------|------|
| Enterprise | External attack | Medium | Medium | Moderate |
| DMZ | Targeted attack | High | Medium | High |
| Industrial Control | Insider, APT | Critical | Low | High |
| Safety System | Physical access | Critical | Very Low | Moderate |

### Residual Risks

| ID | Risk | Mitigation | Residual |
|----|------|------------|----------|
| RR-1 | Zero-day exploitation | Defense in depth | Low |
| RR-2 | Insider threat | Audit, least privilege | Low |
| RR-3 | Supply chain attack | SBOM, verification | Medium |
| RR-4 | Physical tampering | TPM attestation | Low |

## 8. Maintenance

### Review Schedule

| Activity | Frequency |
|----------|-----------|
| Zone boundary review | Annual |
| Conduit validation | Quarterly |
| Firewall rule audit | Monthly |
| Access control review | Quarterly |
| Risk reassessment | Annual |

### Change Management

All changes to zone/conduit configuration require:
1. Security impact assessment
2. Testing in non-production
3. Change approval board
4. Implementation during maintenance window
5. Post-change verification

## 9. References

- IEC 62443-3-2:2020 Security Risk Assessment
- IEC 62443-3-3:2013 System Security Requirements
- NIST SP 800-82 Guide to ICS Security

---

**Document Version:** 1.0  
**Last Updated:** 2025-12-03
