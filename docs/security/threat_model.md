# Threat Model - Nethical AI Governance System

## Overview
This document provides a comprehensive threat model using the STRIDE methodology for threats, vulnerabilities, and mitigation strategies.

## STRIDE Analysis Summary

### S - Spoofing
- **Threats**: Agent identity spoofing, component impersonation
- **Mitigations**: Agent authentication, Merkle audit logs, risk profiling
- **Code**: `IntegratedGovernance.process_action()`, `RiskEngine`, `MerkleAnchor`

### T - Tampering
- **Threats**: Audit log tampering, policy manipulation, risk score modification
- **Mitigations**: Merkle anchoring (immutable logs), policy diff auditing, quarantine
- **Code**: `nethical/core/audit_merkle.py`, `nethical/core/policy_diff.py`

### R - Repudiation
- **Threats**: Action/decision denial
- **Mitigations**: Comprehensive audit logging, Merkle proofs, SLA monitoring
- **Code**: All process methods log with timestamps and Merkle anchors

### I - Information Disclosure
- **Threats**: PII leakage, cross-tenant data exposure
- **Mitigations**: PII detection/redaction, differential privacy, storage partitioning
- **Code**: `nethical/utils/pii.py`, `nethical/core/redaction_pipeline.py`, `nethical/core/differential_privacy.py`

### D - Denial of Service
- **Threats**: Volume attacks, resource exhaustion
- **Mitigations**: Quota enforcement, rate limiting, backpressure
- **Code**: `nethical/quotas.py`, `QuotaEnforcer.check_quota()`

### E - Elevation of Privilege
- **Threats**: Privilege escalation, policy bypass
- **Mitigations**: Multi-tier risk assessment, quarantine, ML shadow mode, human escalation
- **Code**: `RiskEngine`, `QuarantineManager`, `MLShadowClassifier`, `EscalationQueue`

## Key Attack Scenarios

1. **Prompt Injection**: Detected by `AdversarialDetector`, tested in `tests/adversarial/test_context_confusion.py`
2. **PII Exfiltration**: Detected by `PIIDetector`, tested in `tests/adversarial/test_privacy_harvesting.py`
3. **Resource Exhaustion**: Blocked by `QuotaEnforcer`, tested in `tests/adversarial/test_resource_exhaustion.py`

## Security Controls Matrix

| Control | Implementation | Status |
|---------|---------------|--------|
| Authentication | JWT tokens + API keys (`nethical/security/auth.py`) | ✅ Complete |
| Multi-Factor Authentication | TOTP + backup codes (`nethical/security/mfa.py`) | ✅ Complete |
| SSO/SAML | SAML 2.0, OAuth, OIDC (`nethical/security/sso.py`) | ✅ Complete |
| Authorization | RBAC + Risk-based decisions (`nethical/core/rbac.py`) | ✅ Complete |
| Access Control | Role hierarchy (admin, operator, auditor, viewer) | ✅ Complete |
| Audit Logging | Merkle-anchored + RBAC audit trail | ✅ Complete |
| Data Protection | PII detection/redaction | ✅ Complete |
| Rate Limiting | Quota enforcement | ✅ Complete |
| Supply Chain | Dependabot + SBOM + signing + SLSA L3 | ✅ Complete |
| Threat Model | Automated validation (CI/CD) | ✅ Complete |

See full details in implementation.

## Recent Security Enhancements (Phase 1)

### 1.1 Role-Based Access Control (RBAC)
**Implementation**: `nethical/core/rbac.py`

- **Role Hierarchy**: 
  - Admin: Full system control
  - Operator: Execute actions and manage quarantine
  - Auditor: Read-only access to logs and violations
  - Viewer: Basic read access to metrics and policies
  
- **Features**:
  - Decorator-based access control (`@require_role`, `@require_permission`)
  - Fine-grained permissions (16+ permission types)
  - Custom permission grants
  - Comprehensive audit trail
  - Hierarchical role inheritance

### 1.2 JWT Authentication System
**Implementation**: `nethical/security/auth.py`

- **Token Types**:
  - Access tokens (short-lived, 1 hour default)
  - Refresh tokens (long-lived, 7 days default)
  
- **Features**:
  - HS256-signed JWT tokens
  - Token revocation support
  - API key management for service-to-service auth
  - Token refresh mechanism
  - Expiration and validation
  
- **API Keys**:
  - SHA-256 hashed storage
  - Optional expiration
  - Per-user key management
  - Enable/disable functionality
  - Last-used tracking

### 1.3 Multi-Factor Authentication (MFA)
**Implementation**: `nethical/security/mfa.py`

- **Methods**:
  - TOTP (Time-based One-Time Password)
  - Backup recovery codes
  - SMS verification (framework)
  
- **Features**:
  - QR code generation for easy enrollment
  - Mandatory MFA for admin operations
  - Backup code management
  - User-friendly setup flow
  - 21 comprehensive tests

### 1.4 SSO/SAML Integration
**Implementation**: `nethical/security/sso.py`

- **Supported Protocols**:
  - SAML 2.0 Service Provider
  - OAuth 2.0
  - OpenID Connect (OIDC)
  
- **Features**:
  - Multiple IdP configuration support
  - Flexible attribute mapping
  - User auto-provisioning
  - Group/role synchronization
  - 21 comprehensive tests

### 1.5 Supply Chain Security
**Implementation**: `.github/dependabot.yml`, `scripts/supply_chain_dashboard.py`

- Automated dependency updates (weekly)
- Dependency version pinning with hash verification
- SLSA Level 3 compliance tracking
- SBOM generation
- Security vulnerability monitoring
- GitHub Actions version management
- Docker image updates

### 1.6 Threat Model Automation
**Implementation**: `.github/workflows/threat-model.yml`

- Automated STRIDE validation on PRs
- Security controls mapping to code
- Coverage metrics calculation
- PR comments with security status
- Weekly scheduled validation
